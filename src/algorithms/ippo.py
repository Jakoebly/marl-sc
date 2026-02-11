from typing import Dict, Any, Optional
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.algorithms.base import BaseAlgorithmWrapper
from src.algorithms.models.rlmodules.base import ActorCriticRLModule
from src.environment.environment import InventoryEnvironment


class IPPOWrapper(BaseAlgorithmWrapper):
    """
    Wrapper for IPPO (Independent PPO) algorithm.
    
    Each agent learns independently with its own actor and critic networks.
    Supports parameter sharing option.
    """
    
    def __init__(self, env, ippo_config, train_seed: Optional[int] = None, eval_seed: Optional[int] = None):
        """
        Initializes the IPPO wrapper.
        
        Args:
            env (InventoryEnvironment): InventoryEnvironment instance (PettingZoo ParallelEnv)
            ippo_config (AlgorithmConfig): AlgorithmConfig instance
            train_seed (Optional[int]): Seed for training environments and RLlib framework seeding
                (`.debugging(seed=...)` and training `env_config["seed"]`). Defaults to None.
            eval_seed (Optional[int]): Seed for evaluation environments 
                (`evaluation_config["env_config"]["seed"]`). Defaults to None.
        """

        # Store environment and config
        self.env = env
        self.env_config = self.env.env_config
        self.env_name = self.env.metadata["name"]
        self.ippo_config = ippo_config
        self.train_seed = train_seed
        self.eval_seed = eval_seed

        # Create factory function that creates new environment instances
        env_factory = self.create_env_factory(self.env_config)
        
        # Register the factory
        register_env(self.env_name, env_factory)
        
        # Extract config values
        shared_params = self.ippo_config.shared
        ippo_params = self.ippo_config.algorithm_specific
        networks_params = ippo_params.networks.model_dump()
        parameter_sharing = ippo_params.parameter_sharing
        max_seq_len = self.extract_max_seq_len(networks_params)
        num_env_runners = shared_params.num_env_runners
        num_envs_per_env_runner = shared_params.num_envs_per_env_runner
        
        # Get observation and action spaces
        obs_space = env.observation_space(env.agents[0])
        action_space = env.action_space(env.agents[0])
        
        # Create model config
        model_config = {
            "networks": networks_params,
            "observation_space": obs_space,
            "action_space": action_space,
            "use_centralized_critic": False

        }
        if max_seq_len is not None:
            model_config["max_seq_len"] = max_seq_len

        # Create RLModule spec using RLModuleSpec
        rl_module_spec = RLModuleSpec(
            module_class=ActorCriticRLModule,
            observation_space=obs_space,
            action_space=action_space,
            model_config=model_config
        )
        
        # Determine multi-agent setup based on parameter sharing
        if parameter_sharing:
            # Single policy shared across all agents
            policies = {"shared_policy"}
            policy_mapping_fn = lambda agent_id, *args, **kwargs: "shared_policy"
            module_specs = {f"shared_policy": rl_module_spec}
        else:
            # Separate policy per agent (each gets same spec but separate instance)
            policies = {f"policy_{agent_id}" for agent_id in env.agents}
            policy_mapping_fn = lambda agent_id, *args, **kwargs: f"policy_{agent_id}"
            module_specs = {f"policy_{agent_id}": rl_module_spec for agent_id in env.agents}
        
        # Store policy mapping function for future use (e.g., rollout)
        self.policy_mapping_fn = policy_mapping_fn
        
        # Create PPO config with multi-agent setup included in chain
        ppo_config = (
            PPOConfig()
            .debugging(seed=self.train_seed)
            .environment(
                env=self.env_name, 
                clip_actions=True,
                env_config={
                    "seed": self.train_seed,
                    "data_mode": "train" 
                }
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .rl_module(
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs=module_specs
                )
            )
            .training(
                lr=shared_params.learning_rate,
                train_batch_size_per_learner=shared_params.batch_size,
                num_epochs=shared_params.num_epochs,
                minibatch_size=shared_params.batch_size // shared_params.num_minibatches, 
                shuffle_batch_per_epoch=True,
                vf_loss_coeff=ippo_params.vf_loss_coeff,
                entropy_coeff=ippo_params.entropy_coeff,
                clip_param=ippo_params.clip_param,
                use_gae=ippo_params.use_gae,
                lambda_=ippo_params.lam,
                gamma=ippo_params.gamma
            )
            .env_runners(
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner
            )
            .evaluation(
                evaluation_interval=shared_params.eval_interval,
                evaluation_duration=shared_params.num_eval_episodes,
                evaluation_parallel_to_training=shared_params.evaluation_parallel_to_training,
                evaluation_config={
                    "env": self.env_name, 
                    "clip_actions": True,
                    "env_config": {
                        "seed": self.eval_seed,
                        "data_mode": "val" 
                    }
                }
            )
        )
        
        # Build trainer and set training parameters
        self.trainer = ppo_config.build_algo()
        self.num_iterations = shared_params.num_iterations
        self.checkpoint_freq = shared_params.checkpoint_freq


