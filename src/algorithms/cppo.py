"""
Centralized PPO (single-agent) wrapper for RLlib.

A single PPO policy controls **all** warehouses simultaneously, receiving the
full global observation and outputting a joint action vector.  This removes
every MARL challenge (non-stationarity, partial observability, credit
assignment) and serves as an upper-bound baseline on what RL can achieve.
"""

import os
import random
import uuid
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.schema import EnvironmentConfig

import numpy as np
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.connectors.env_to_module import MeanStdFilter
from ray.tune.registry import register_env

from src.algorithms.base import BaseAlgorithmWrapper
from src.algorithms.models.rlmodules.base import ActorCriticRLModule
from src.environment.envs.multi_env import InventoryEnvironment
from src.environment.envs.single_env import CentralizedEnvWrapper


class CPPOWrapper(BaseAlgorithmWrapper):
    """
    Initializes the CPPOWrapper for centralized (single-agent) PPO.

    A single policy receives the full global observation and outputs a joint
    action vector covering all warehouses.  Internally, the environment is
    wrapped in :class:`CentralizedEnvWrapper` which splits the action and
    aggregates rewards.
    """

    def __init__(
        self,
        env: InventoryEnvironment,
        cppo_config,
        train_seed: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ):
        """
        Initializes the CPPOWrapper for centralized (single-agent) PPO.

        Args:
            env (InventoryEnvironment): Template ``InventoryEnvironment`` (used to query spaces).
            cppo_config (AlgorithmConfig): Algorithm config (same schema as IPPO/MAPPO).
            train_seed (Optional[int]): Seed for training environments and RLlib framework. Defaults to None.
            eval_seed (Optional[int]): Seed for evaluation environments. Defaults to None.
        """

        # Store environment and config
        self.env = env
        self.env_config = env.env_config
        self.env_name = f"{CentralizedEnvWrapper.metadata['name']}_{uuid.uuid4().hex[:8]}"
        self.cppo_config = cppo_config
        self.train_seed = train_seed
        self.eval_seed = eval_seed

        # Create factory function that creates new environment instances
        env_factory = self.create_env_factory(self.env_config)
        
        # Register the factory
        register_env(self.env_name, env_factory)
        
        # Extract config values
        shared_params = self.cppo_config.shared
        cppo_params = self.cppo_config.algorithm_specific
        networks_params = cppo_params.networks.model_dump()
        self.max_seq_len = self.extract_max_seq_len(networks_params)
        self.num_env_runners = shared_params.num_env_runners
        self.num_envs_per_env_runner = shared_params.num_envs_per_env_runner

        # Store obs normalization mode and 
        # set the parameter sharing flag on the env tofalse for centralized PPO
        self.obs_normalization = cppo_params.obs_normalization
        self.env.obs_normalization = self.obs_normalization
        self.env.include_warehouse_id = False

        # Build a temporary CentralizedEnvWrapper to get observation and action spaces
        template_wrapper = CentralizedEnvWrapper(self.env_config)
        self.obs_space = template_wrapper.observation_space
        self.action_space = template_wrapper.action_space
        self.obs_dim = self.obs_space.shape[0]

        # Create the model config
        # Note: The centralized agent sees the full global obs as one flat vector.
        # Map it entirely to the "local" slot so ActorCriticRLModule treats
        # the whole observation as input (no local/global split).
        model_config = {
            "networks": networks_params,
            "observation_space": self.obs_space,
            "action_space": self.action_space,
            "actor_obs_type": "local",
            "critic_obs_type": "local",
            "local_obs_dim": self.obs_dim,
            "global_obs_dim": 0,
            "logstd_init": cppo_params.logstd_init,
            "logstd_floor": cppo_params.logstd_floor,
        }
        if self.max_seq_len is not None:
            model_config["max_seq_len"] = self.max_seq_len

        # Create RLModule spec using RLModuleSpec
        rl_module_spec = RLModuleSpec(
            module_class=ActorCriticRLModule,
            observation_space=self.obs_space,
            action_space=self.action_space,
            model_config=model_config,
        )

        # Create the policy mapping function
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: DEFAULT_MODULE_ID

        # Read precomputed observation statistics from the template environment
        self.obs_stats = self.env.obs_stats

        # Build train env_config for RLlib
        train_env_config = {
            "seed": self.train_seed,
            "data_mode": "train",
            "obs_normalization": self.obs_normalization,
            "obs_stats": self.obs_stats,
        }

        # Build eval env_config for RLlib
        eval_env_config = {
            "seed": self.eval_seed,
            "data_mode": "val",
            "obs_normalization": self.obs_normalization,
            "obs_stats": self.obs_stats,
            "num_eval_episodes": shared_params.num_eval_episodes,
        }

        # Build training config for RLlib
        training_kwargs = dict(
            use_kl_loss=cppo_params.use_kl_loss,
            grad_clip=cppo_params.grad_clip,
            lr=shared_params.learning_rate,
            train_batch_size_per_learner=shared_params.batch_size,
            num_epochs=shared_params.num_epochs,
            minibatch_size=shared_params.batch_size // shared_params.num_minibatches,
            shuffle_batch_per_epoch=True,
            vf_loss_coeff=cppo_params.vf_loss_coeff,
            vf_clip_param=cppo_params.vf_clip_param,
            entropy_coeff=cppo_params.entropy_coeff,
            clip_param=cppo_params.clip_param,
            use_gae=cppo_params.use_gae,
            lambda_=cppo_params.lam,
            gamma=cppo_params.gamma,
        )

        # Build environment runners config for RLlib
        env_runners_kwargs = {
            "num_env_runners": self.num_env_runners,
            "num_envs_per_env_runner": self.num_envs_per_env_runner,
        }
        if self.obs_normalization == "meanstd":
            env_runners_kwargs["env_to_module_connector"] = (
                lambda env, spaces, device: MeanStdFilter()
            )

        # Create PPO config for single-agent setup
        ppo_config = (
            PPOConfig()
            .debugging(seed=self.train_seed)
            .reporting(
                metrics_num_episodes_for_smoothing=shared_params.num_eval_episodes
            )
            .environment(
                env=self.env_name,
                clip_actions=True,
                normalize_actions=False,
                env_config=train_env_config,
            )
            .rl_module(rl_module_spec=rl_module_spec)
            .training(**training_kwargs)
            .env_runners(**env_runners_kwargs)
            .evaluation(
                evaluation_interval=shared_params.eval_interval,
                evaluation_duration=shared_params.num_eval_episodes,
                evaluation_parallel_to_training=shared_params.evaluation_parallel_to_training,
                evaluation_config={
                    "env": self.env_name,
                    "clip_actions": True,
                    "normalize_actions": False,
                    "use_worker_filter_stats": False,
                    "env_config": eval_env_config,
                    "explore": False,
                    "num_envs_per_env_runner": 1,
                },
            )
        )

        # Seed global RNGs right before build so that weight initialisation
        # (and any other framework randomness) is fully deterministic
        if self.train_seed is not None:
            random.seed(self.train_seed)
            np.random.seed(self.train_seed)
            torch.manual_seed(self.train_seed)
            torch.use_deterministic_algorithms(True)

        # Build trainer and set training parameters
        self.trainer = ppo_config.build_algo(
            logger_creator=self.noop_logger_creator,
        )
        self.num_iterations = shared_params.num_iterations
        self.checkpoint_freq = shared_params.checkpoint_freq

        # Load warm-start weights from a previous training run if curriculum learning is enabled
        if cppo_params.warmstart_weights_path is not None:
            from src.utils.weight_transfer import load_module_weights
            load_module_weights(
                self.trainer, DEFAULT_MODULE_ID, cppo_params.warmstart_weights_path
            )
            weights = self.trainer.learner_group.get_weights()
            self.trainer.env_runner.set_weights(weights)

    # ------------------------------------------------------------------
    # BaseAlgorithmWrapper Overrides
    # ------------------------------------------------------------------

    @staticmethod
    def create_env_factory(
        env_config: 'EnvironmentConfig',
    ) -> Callable[[Dict[str, Any]], CentralizedEnvWrapper]:
        """
        Creates an environment factory function for RLlib.

        Overrides :meth:`BaseAlgorithmWrapper.create_env_factory` which
        produces ``ParallelPettingZooEnv`` (multi-agent) wrappers instead.

        Args:
            env_config (EnvironmentConfig): Environment configuration to use for creating instances.
            
        Returns:
            env_factory (Callable): Factory function that RLlib calls to create environment instances.
        """
        from src.utils.seed_manager import SeedManager

        # Per-worker counter used in place of vector_index
        _env_counter = [0]

        def env_factory(env_meta: Dict[str, Any] = None) -> CentralizedEnvWrapper:
            """
            Factory function that RLlib calls to create environment instances.
            RLlib calls this multiple times to create environment instances for different workers.
            
            Args:
                env_meta (Dict[str, Any]): Dict from RLlib containing environment metadata 
                    (e.g., {"seed": <train_seed or eval_seed>, "data_mode": "train"})
            """

            # Extract base seed from RLlib's config (train_seed or eval_seed)
            seed = None
            if env_meta:
                seed = env_meta.get("seed")

            # Derive a unique per-environment seed so that parallel envs
            # across workers produce diverse (but deterministic) episodes.
            if seed is not None:
                data_mode = (
                    env_meta.get("data_mode", "train") if env_meta else "train"
                )
                if data_mode == "train":
                    worker_index = os.getpid()
                    env_index = _env_counter[0]
                    _env_counter[0] += 1
                    seed = SeedManager.derive_env_seed(
                        seed, worker_index, env_index
                    )

            # Create a new CentralizedEnvWrapper instance
            env = CentralizedEnvWrapper(
                env_config, seed=seed, env_meta=env_meta
            )

            return env

        return env_factory

    def rollout(
        self,
        env: InventoryEnvironment,
        num_episodes: int = 1,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Runs manual rollout episodes collecting detailed per-step data for visualization.

        The caller (``EvaluationRunner``) passes a bare
        ``InventoryEnvironment`` (multi-agent PettingZoo env), **not** a 
        ``CentralizedEnvWrapper``. ``rollout()`` then steps the raw multi-agent
        env directly so that per-warehouse rewards, infos, and observations
        remain accessible for the visualization pipeline. The global-obs
        extraction and action-splitting logic mirrors what ``CentralizedEnvWrapper`` 
        does internally.

        Args:
            env: ``InventoryEnvironment`` configured for evaluation.
            num_episodes: Number of episodes to collect.

        Returns:
            List of episode-data dicts (one per episode).
        """

        # Enable detailed step info collection
        env.collect_step_info = True

        # Get the RLModule
        module = self.trainer.get_module(module_id=DEFAULT_MODULE_ID)

        # Determine if policies are recurrent (GRU) by checking initial state
        initial_state = module.get_initial_state()
        is_recurrent = len(initial_state) > 0

        # Get action distribution class for deterministic action sampling
        dist_cls = module.get_inference_action_dist_cls()

        # Store env parameters and compute the local observation dimension 
        n_skus = env.n_skus
        n_warehouses = env.n_warehouses
        local_obs_dim = env._compute_local_obs_dim()

        # Determine the normalization mode
        obs_norm_mode = getattr(self, "obs_normalization", "off")

        # For the "meanstd" mode, extract observation filters to replicate the
        # normalization applied by the MeanStdFilter env-to-module connector
        obs_filter = None
        if obs_norm_mode == "meanstd":
            filters = self._get_obs_filters()
            if filters is not None:
                obs_filter = filters.get(None)
        
        # Run manual rollout
        all_episodes: List[Dict[str, np.ndarray]] = []
        for _ in range(num_episodes):
            # Initialize episode data and reset environment
            episode_data = defaultdict(list)
            obs_dict, _ = env.reset()

            # Extract the global observation from the multi-agent observation dictionary
            global_obs = obs_dict[env.agents[0]][local_obs_dim:]

            # Reset hidden states for recurrent policies
            if is_recurrent:
                states = {
                    k: v.unsqueeze(0) for k, v in initial_state.items()
                }

            # Run manual rollout loop
            done = False
            while not done:
                with torch.no_grad():
                    # Store raw observation before normalization
                    obs_raw = np.array(global_obs, dtype=np.float32)

                    # Normalize observations using training filter stats
                    obs = obs_raw.copy()
                    if obs_filter is not None:
                        obs = obs_filter(obs, update=False)
                    obs_norm = np.array(obs, dtype=np.float32)

                    # Convert observations to tensors with batch dimension
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    batch = {Columns.OBS: obs_tensor}

                    # Add hidden states for recurrent policies
                    if is_recurrent:
                        batch[Columns.STATE_IN] = states

                    # Forward through RLModule
                    output = module._forward_inference(batch)

                    # Sample deterministic action from distribution
                    action_logits = output[Columns.ACTION_DIST_INPUTS]
                    
                    # GRU models return (B, seq_len=1, dim) — squeeze seq_len
                    if action_logits.dim() == 3:
                        action_logits = action_logits.squeeze(1)
                    dist = dist_cls.from_logits(action_logits)
                    det_dist = dist.to_deterministic()
                    action = det_dist.sample().squeeze(0).cpu().numpy()
                    action = np.clip(action, -1.0, 1.0)

                    # Extract mu and std from action distribution inputs
                    # ACTION_DIST_INPUTS = [means, log_std] (log_std from nn.Parameter)
                    logits_np = action_logits.squeeze(0).cpu().numpy()
                    action_dim = action.shape[-1]
                    mu = logits_np[:action_dim]
                    sigma = np.exp(logits_np[action_dim:])

                    # Update hidden states for recurrent policies
                    if is_recurrent:
                        states = output.get(Columns.STATE_OUT, {})

                # Reshape flat joint arrays to (n_warehouses, n_skus) for visualization
                actions_array = action.reshape(n_warehouses, n_skus)
                mu_array = mu.reshape(n_warehouses, n_skus)
                sigma_array = sigma.reshape(n_warehouses, n_skus)
                obs_raw_per_wh = obs_raw.reshape(n_warehouses, -1)
                obs_norm_per_wh = obs_norm.reshape(n_warehouses, -1)

                episode_data["actions_raw"].append(actions_array)
                episode_data["actor_mu"].append(mu_array)
                episode_data["actor_sigma"].append(sigma_array)
                episode_data["obs_raw"].append(obs_raw_per_wh)
                episode_data["obs_normalized"].append(obs_norm_per_wh)

                # Split actions back into per-warehouse dict and step inner env
                per_wh_actions = {
                    agent_id: action[i * n_skus : (i + 1) * n_skus]
                    for i, agent_id in enumerate(env.agents)
                }
                obs_dict, rewards, terms, truncs, infos = env.step(per_wh_actions)

                # Extract the global observation from the multi-agent observation dictionary
                global_obs = obs_dict[env.agents[0]][local_obs_dim:]

                # Extract step info
                step_info = infos[env.agents[0]]
                for key, value in step_info.items():
                    episode_data[key].append(
                        value.copy() if isinstance(value, np.ndarray) else value
                    )

                # Record per-warehouse rewards
                rewards_array = np.array(
                    [rewards[a] for a in env.agents], dtype=np.float32
                )
                episode_data["rewards"].append(rewards_array)

                done = all(truncs.values()) or all(terms.values())

            # Convert all lists to numpy arrays and add env metadata for visualization
            episode_data = {k: np.array(v) for k, v in episode_data.items()}
            episode_data["n_skus"] = env.n_skus
            episode_data["max_expected_lead_time"] = env.max_expected_lead_time
            episode_data["feature_config"] = env.feature_config.model_dump()
            episode_data["include_warehouse_id"] = env.include_warehouse_id
            episode_data["rolling_window"] = env.rolling_window
            all_episodes.append(episode_data)

        # Disable step info collection after rollout
        env.collect_step_info = False
        
        return all_episodes

