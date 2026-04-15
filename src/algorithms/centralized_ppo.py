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
from typing import Any, Callable, Dict, List, Optional

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


class CentralizedPPOWrapper(BaseAlgorithmWrapper):
    """
    Wrapper for centralized (single-agent) PPO.

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
        Args:
            env: Template ``InventoryEnvironment`` (used to query spaces).
            cppo_config: Algorithm config (same schema as IPPO/MAPPO).
            train_seed: Seed for training environments and RLlib framework.
            eval_seed: Seed for evaluation environments.
        """

        self.env = env
        self.env_config = env.env_config
        self.env_name = f"single_env_{uuid.uuid4().hex[:8]}"
        self.cppo_config = cppo_config
        self.train_seed = train_seed
        self.eval_seed = eval_seed

        env_factory = self._create_env_factory(self.env_config)
        register_env(self.env_name, env_factory)

        shared_params = cppo_config.shared
        cppo_params = cppo_config.algorithm_specific
        networks_params = cppo_params.networks.model_dump()
        self.max_seq_len = self.extract_max_seq_len(networks_params)
        self.num_env_runners = shared_params.num_env_runners
        self.num_envs_per_env_runner = shared_params.num_envs_per_env_runner

        self.obs_normalization = cppo_params.obs_normalization
        self.env.obs_normalization = self.obs_normalization
        self.env.include_warehouse_id = False

        # Build a temporary CentralizedEnvWrapper to read spaces
        template_wrapper = CentralizedEnvWrapper(self.env_config)
        self.obs_space = template_wrapper.observation_space
        self.action_space = template_wrapper.action_space
        obs_dim = self.obs_space.shape[0]

        # The centralized agent sees the full global obs as one flat vector.
        # Map it entirely to the "local" slot so ActorCriticRLModule treats
        # the whole observation as input (no local/global split).
        model_config = {
            "networks": networks_params,
            "observation_space": self.obs_space,
            "action_space": self.action_space,
            "actor_obs_type": "local",
            "critic_obs_type": "local",
            "local_obs_dim": obs_dim,
            "global_obs_dim": 0,
            "logstd_init": cppo_params.logstd_init,
            "logstd_floor": cppo_params.logstd_floor,
        }
        if self.max_seq_len is not None:
            model_config["max_seq_len"] = self.max_seq_len

        rl_module_spec = RLModuleSpec(
            module_class=ActorCriticRLModule,
            observation_space=self.obs_space,
            action_space=self.action_space,
            model_config=model_config,
        )

        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: DEFAULT_MODULE_ID

        self.obs_stats = self.env.obs_stats

        # ---- env configs for RLlib workers --------------------------------
        train_env_config = {
            "seed": self.train_seed,
            "data_mode": "train",
            "obs_normalization": self.obs_normalization,
            "obs_stats": self.obs_stats,
        }
        eval_env_config = {
            "seed": self.eval_seed,
            "data_mode": "val",
            "obs_normalization": self.obs_normalization,
            "obs_stats": self.obs_stats,
            "num_eval_episodes": shared_params.num_eval_episodes,
        }

        # ---- training kwargs ----------------------------------------------
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

        # ---- env-runner kwargs --------------------------------------------
        env_runners_kwargs = {
            "num_env_runners": self.num_env_runners,
            "num_envs_per_env_runner": self.num_envs_per_env_runner,
        }
        if self.obs_normalization == "meanstd":
            env_runners_kwargs["env_to_module_connector"] = (
                lambda env, spaces, device: MeanStdFilter()
            )

        # ---- build PPO config (single-agent, no .multi_agent()) -----------
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

        if self.train_seed is not None:
            random.seed(self.train_seed)
            np.random.seed(self.train_seed)
            torch.manual_seed(self.train_seed)
            torch.use_deterministic_algorithms(True)

        self.trainer = ppo_config.build_algo(
            logger_creator=self.noop_logger_creator,
        )
        self.num_iterations = shared_params.num_iterations
        self.checkpoint_freq = shared_params.checkpoint_freq

        if cppo_params.warmstart_weights_path is not None:
            from src.utils.weight_transfer import load_module_weights

            load_module_weights(
                self.trainer, DEFAULT_MODULE_ID, cppo_params.warmstart_weights_path
            )
            weights = self.trainer.learner_group.get_weights()
            self.trainer.env_runner.set_weights(weights)

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    @staticmethod
    def _create_env_factory(
        env_config,
    ) -> Callable[[Dict[str, Any]], CentralizedEnvWrapper]:
        """Creates an env factory that produces :class:`CentralizedEnvWrapper`
        instances for RLlib workers."""
        from src.utils.seed_manager import SeedManager

        _env_counter = [0]

        def env_factory(env_meta: Dict[str, Any] = None) -> CentralizedEnvWrapper:
            seed = None
            if env_meta:
                seed = env_meta.get("seed")

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

            return CentralizedEnvWrapper(
                env_config, seed=seed, env_meta=env_meta
            )

        return env_factory

    # ------------------------------------------------------------------
    # Rollout  (overrides BaseAlgorithmWrapper.rollout)
    # ------------------------------------------------------------------

    def rollout(
        self,
        env: InventoryEnvironment,
        num_episodes: int = 1,
    ) -> List[Dict[str, np.ndarray]]:
        """Runs manual rollout episodes for visualization.

        The caller (``EvaluationRunner``) passes a bare
        ``InventoryEnvironment``.  This method queries the single centralized
        policy, splits the joint action into per-warehouse sub-actions, and
        records data in the same per-warehouse format that
        :func:`generate_visualizations` expects.

        Args:
            env: ``InventoryEnvironment`` configured for evaluation.
            num_episodes: Number of episodes to collect.

        Returns:
            List of episode-data dicts (one per episode).
        """

        env.collect_step_info = True

        module = self.trainer.get_module(module_id=DEFAULT_MODULE_ID)
        initial_state = module.get_initial_state()
        is_recurrent = len(initial_state) > 0
        dist_cls = module.get_inference_action_dist_cls()

        local_obs_dim = env._compute_local_obs_dim()
        n_skus = env.n_skus
        n_warehouses = env.n_warehouses

        obs_norm_mode = getattr(self, "obs_normalization", "off")
        obs_filter = None
        if obs_norm_mode == "meanstd":
            obs_filter = self._get_single_agent_obs_filter()

        all_episodes: List[Dict[str, np.ndarray]] = []

        for _ in range(num_episodes):
            episode_data = defaultdict(list)
            obs_dict, _ = env.reset()

            # Extract global obs from the multi-agent obs dict
            global_obs = obs_dict[env.agents[0]][local_obs_dim:]

            if is_recurrent:
                states = {
                    k: v.unsqueeze(0) for k, v in initial_state.items()
                }

            done = False
            while not done:
                with torch.no_grad():
                    obs_raw = np.array(global_obs, dtype=np.float32)

                    obs_for_net = obs_raw.copy()
                    if obs_filter is not None:
                        obs_for_net = obs_filter(obs_for_net, update=False)
                    obs_norm = np.array(obs_for_net, dtype=np.float32)

                    obs_tensor = torch.tensor(obs_for_net, dtype=torch.float32).unsqueeze(0)
                    batch = {Columns.OBS: obs_tensor}

                    if is_recurrent:
                        batch[Columns.STATE_IN] = states

                    output = module._forward_inference(batch)

                    action_logits = output[Columns.ACTION_DIST_INPUTS]
                    if action_logits.dim() == 3:
                        action_logits = action_logits.squeeze(1)

                    dist = dist_cls.from_logits(action_logits)
                    det_dist = dist.to_deterministic()
                    action = det_dist.sample().squeeze(0).cpu().numpy()
                    action = np.clip(action, -1.0, 1.0)

                    logits_np = action_logits.squeeze(0).cpu().numpy()
                    action_dim = action.shape[-1]
                    mu = logits_np[:action_dim]
                    sigma = np.exp(logits_np[action_dim:])

                    if is_recurrent:
                        states = output.get(Columns.STATE_OUT, {})

                # Reshape flat joint arrays to (n_warehouses, n_skus) for viz
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

                # Split action into per-warehouse dict and step inner env
                per_wh_actions = {
                    agent_id: action[i * n_skus : (i + 1) * n_skus]
                    for i, agent_id in enumerate(env.agents)
                }
                obs_dict, rewards, terms, truncs, infos = env.step(per_wh_actions)

                global_obs = obs_dict[env.agents[0]][local_obs_dim:]

                step_info = infos[env.agents[0]]
                for key, value in step_info.items():
                    episode_data[key].append(
                        value.copy() if isinstance(value, np.ndarray) else value
                    )

                rewards_array = np.array(
                    [rewards[a] for a in env.agents], dtype=np.float32
                )
                episode_data["rewards"].append(rewards_array)

                done = all(truncs.values()) or all(terms.values())

            episode_data = {k: np.array(v) for k, v in episode_data.items()}
            episode_data["n_skus"] = env.n_skus
            episode_data["max_expected_lead_time"] = env.max_expected_lead_time
            episode_data["feature_config"] = env.feature_config.model_dump()
            episode_data["include_warehouse_id"] = env.include_warehouse_id
            episode_data["rolling_window"] = env.rolling_window
            all_episodes.append(episode_data)

        env.collect_step_info = False
        return all_episodes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_single_agent_obs_filter(self):
        """Extracts the single-agent obs filter from the MeanStdFilter
        connector (if present)."""
        from ray.rllib.connectors.env_to_module.mean_std_filter import (
            MeanStdFilter as MeanStdFilterConnector,
        )

        try:
            pipeline = self.trainer.env_runner._env_to_module
            for connector in pipeline:
                if isinstance(connector, MeanStdFilterConnector):
                    if connector._filters is None:
                        return None
                    # Single-agent: _filters is keyed by DEFAULT_MODULE_ID
                    return connector._filters.get(DEFAULT_MODULE_ID)
        except Exception:
            pass
        return None
