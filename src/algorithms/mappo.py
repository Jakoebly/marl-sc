from typing import Dict, Any, Optional
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from src.algorithms.base import BaseAlgorithmWrapper
from src.algorithms.models.rlmodules.base import ActorCriticRLModule
from src.environment.environment import InventoryEnvironment


class MAPPOWrapper(BaseAlgorithmWrapper):
    """
    Wrapper for MAPPO (Multi-Agent PPO) algorithm.
    
    MAPPO follows centralized training with decentralized execution (CDTE) where 
    agents select actions from local observations while training uses a centralized
    critic with access to global or joint information. Supports a parameter sharing option.
    """
    
    def __init__(self, env, mappo_config, root_seed: Optional[int] = None):
        """
        Initializes the MAPPO wrapper.
        
        Args:
            env (InventoryEnvironment): InventoryEnvironment instance (PettingZoo ParallelEnv)
            mappo_config (AlgorithmConfig): AlgorithmConfig instance
            root_seed (Optional[int]): Root seed for RLlib framework seeding and environment instances.
                Used for both `.debugging(seed=root_seed)` and `env_config["seed"]`. Defaults to None.
        """

        # Store environment and config
        self.env = env
        self.env_config = self.env.env_config
        self.env_name = self.env.metadata["name"]
        self.mappo_config = mappo_config
        self.root_seed = root_seed

        # Create factory function that creates new environment instances
        env_factory = self.create_env_factory(self.env_config)
        
        # Register the factory
        register_env(self.env_name, env_factory)
        
        # Extract config values
        shared_params = self.mappo_config.shared
        mappo_params = self.mappo_config.algorithm_specific
        networks_params = mappo_params.networks.model_dump()
        parameter_sharing = mappo_params.parameter_sharing
        max_seq_len = self.extract_max_seq_len(networks_params)
        num_env_runners = shared_params.num_env_runners
        num_envs_per_env_runner = shared_params.num_envs_per_env_runner
        
        # Get observation and action spaces
        action_space = env.action_space(env.agents[0])
        obs_space = env.observation_space(env.agents[0])
        
        # Create model config with CTDE support
        model_config = {
            "networks": networks_params,
            "observation_space": obs_space,
            "action_space": action_space,
            "use_centralized_critic": True
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
        
        # Create PPO config with multi-agent setup included in chain
        ppo_config = (
            PPOConfig()
            .debugging(seed=self.root_seed)
            .environment(
                env=self.env_name, 
                clip_actions=True,
                env_config={
                    "seed": self.root_seed,
                    "data_mode": "train"  # Training uses train data
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
                vf_loss_coeff=mappo_params.vf_loss_coeff,
                entropy_coeff=mappo_params.entropy_coeff,
                clip_param=mappo_params.clip_param,
                use_gae=mappo_params.use_gae,
                lambda_=mappo_params.lam
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
                        "seed": self.root_seed,
                        "data_mode": "val" 
                    }
                }
            )
        )
        
        # Build trainer and set training parameters
        self.trainer = ppo_config.build_algo()
        self.num_iterations = shared_params.num_iterations
        self.checkpoint_freq = shared_params.checkpoint_freq
    
    def train(self) -> Dict[str, Any]:
        """Run one training iteration.
        
        Returns:
            metrics (Dict[str, Any]): Training metrics dictionary.
        """
        result = self.trainer.train()
        return result
    
    def get_policy(self):
        """
        Get trained policy.
        
        Returns:
            Trained policy object
        """
        # Return the first policy (or shared policy if parameter sharing)
        policy_id = list(self.trainer.config.multi_agent.policies.keys())[0]
        return self.trainer.get_policy(policy_id)
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        self.trainer.save(checkpoint_dir=path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        self.trainer.restore(path)

