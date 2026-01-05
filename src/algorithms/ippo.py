from typing import Any, Dict, TYPE_CHECKING

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env import PettingZooEnv

from src.algorithms.base import BaseAlgorithmWrapper

if TYPE_CHECKING:
    from src.environment.environment import InventoryEnvironment
    from src.config.schema import IPPOConfig


class IPPOWrapper(BaseAlgorithmWrapper):
    """Independent PPO wrapper using RLlib PPO with independent policies."""
    
    def __init__(self, env: 'InventoryEnvironment', config: 'IPPOConfig'):
        """Initialize IPPO wrapper.
        
        Args:
            env: InventoryEnvironment instance (PettingZoo ParallelEnv)
            config: IPPO configuration
        """

        # Store environment and configuration
        self.env = env
        self.config = config
        
        # Convert PettingZoo ParallelEnv to RLlib format
        rllib_env = PettingZooEnv(env)
        
        # Get observation and action spaces (same for all agents)
        obs_space = env.observation_space()
        act_space = env.action_space()
        
        # Get algorithm specific parameters
        use_gae = config.algorithm_specific.get("use_gae", True)
        lambda_gae = config.algorithm_specific.get("lambda", 0.95)
        parameter_sharing = config.algorithm_specific.get("parameter_sharing", False)
        
        # Configure policies based on parameter_sharing flag (shared or independent)
        if parameter_sharing:
            policies = {
                "shared_policy": (None, obs_space, act_space, {})
            }
            policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: "shared_policy"
        else:
            policies = {
                agent_id: (None, obs_space, act_space, {})
                for agent_id in env.agents
            }
            policy_mapping_fn = lambda agent_id, episode, worker, **kwargs: agent_id
        
        # Build RLlib config
        rllib_config = (
            PPOConfig()
            .environment(env=rllib_env)
            .training(
                lr=config.shared.learning_rate,
                train_batch_size=config.shared.batch_size,
                sgd_minibatch_size=config.shared.batch_size // 4, 
                num_sgd_iter=10,
                vf_loss_coeff=config.algorithm_specific.get("vf_loss_coeff", 0.5),
                entropy_coeff=config.algorithm_specific.get("entropy_coeff", 0.01),
                clip_param=config.algorithm_specific.get("clip_param", 0.2),
                use_gae=use_gae,
                lambda_=lambda_gae,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .rollouts(num_rollout_workers=0)  
            .resources(num_gpus=0)
        )
        
        # Create RLlib trainer
        self.trainer = PPO(config=rllib_config)
    
    def train(self) -> Dict[str, Any]:
        """Run one training iteration.
        
        Returns:
            metrics (Dict[str, Any]): Training metrics dictionary.
        """
        result = self.trainer.train()
        return result
    
    def get_policy(self):
        """Get trained policy.
        
        Returns:
            policy: Trained policy object.
        """
        if self.config.algorithm_specific.get("parameter_sharing", False):
            # Shared policy
            return self.trainer.get_policy("shared_policy")
        else:
            # Independent policies - return first agent's policy
            return self.trainer.get_policy(self.env.agents[0])
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path (str): Path to save checkpoint.
        """
        self.trainer.save(checkpoint_dir=path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path (str): Path to load checkpoint from.
        """
        self.trainer.restore(checkpoint_path=path)

