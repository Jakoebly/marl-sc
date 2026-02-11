from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Callable

import numpy as np
import torch

if TYPE_CHECKING:
    from src.environment.environment import InventoryEnvironment
    from src.config.schema import AlgorithmConfig, EnvironmentConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class BaseAlgorithmWrapper(ABC):
    """Base class for RLlib algorithm wrappers."""

    @abstractmethod
    def __init__(self, env: 'InventoryEnvironment', config: 'AlgorithmConfig'):
        """Initializes algorithm wrapper.
        
        Args:
            env (InventoryEnvironment): InventoryEnvironment instance (PettingZoo ParallelEnv)
            config (AlgorithmConfig): Algorithm configuration
        """
        pass

    def train(self) -> Dict[str, Any]:
        """Runs training loop.
        
        Returns:
            metrics (Dict[str, Any]): Training metrics dictionary.
        """
        result = self.trainer.train()
        return result

    def evaluate(self, eval_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Runs evaluation using RLlib's Algorithm.evaluate().
        
        Temporarily overrides evaluation_duration if eval_episodes is provided,
        then restores the original value after evaluation completes.
        
        Args:
            eval_episodes (Optional[int]): Number of evaluation episodes. 
                If None, uses num_eval_episodes from config.
        
        Returns:
            metrics (Dict[str, Any]): Evaluation metrics dictionary.
        """
        
        # Store original evaluation duration
        original_duration = self.trainer.config.evaluation_duration

        # Temporarily override evaluation duration if eval_episodes is provided
        # Config is frozen after build_algo(), so we unfreeze it temporarily
        if eval_episodes is not None:
            self.trainer.config._is_frozen = False
            self.trainer.config.evaluation_duration = eval_episodes
            self.trainer.config._is_frozen = True

        # Evaluate the algorithm
        try:
            result = self.trainer.evaluate()

        # Restore original evaluation duration
        finally:
            self.trainer.config._is_frozen = False
            self.trainer.config.evaluation_duration = original_duration
            self.trainer.config._is_frozen = True

        return result

    def rollout(self, env: 'InventoryEnvironment', num_episodes: int = 1) -> List[Dict[str, np.ndarray]]:
        """
        Runs manual rollout episodes collecting detailed per-step data for visualization.
        
        Steps through the environment using the trained policy, recording observations,
        actions, costs, demand, fulfillment, and other intermediate values at each timestep.
        Handles both recurrent (GRU) and non-recurrent policies.
        
        Args:
            env (InventoryEnvironment): Environment instance configured for evaluation 
                (with eval_seed and data_mode="val"). Must have collect_step_info=True.
            num_episodes (int): Number of episodes to roll out. Defaults to 1.
            
        Returns:
            all_episodes (List[Dict[str, np.ndarray]]): List of episode data dicts. Each dict
                maps metric names to numpy arrays with shape (T, ...) where T is episode length.
        """

        from ray.rllib.core.columns import Columns

        # Enable detailed step info collection
        env.collect_step_info = True
        all_episodes = []

        # Get RLModule
        sample_policy_id = self.policy_mapping_fn(env.agents[0])
        sample_module = self.trainer.get_module(module_id=sample_policy_id)

        # Determine if policies are recurrent (GRU) by checking initial state
        initial_state = sample_module.get_initial_state()
        is_recurrent = len(initial_state) > 0

        # Get action distribution class for deterministic action sampling
        dist_cls = sample_module.get_inference_action_dist_cls()

        # Run manual rollout
        for ep in range(num_episodes):
            # Initialize episode data and reset environment
            episode_data = defaultdict(list)
            obs, info = env.reset()

            # Reset hidden states for recurrent policies
            if is_recurrent:
                states = {}
                for agent_id in env.agents:
                    policy_id = self.policy_mapping_fn(agent_id)
                    module = self.trainer.get_module(module_id=policy_id)
                    init_state = module.get_initial_state()
                    # Add batch dimension: (num_layers*dirs, hidden) -> (1, num_layers*dirs, hidden)
                    states[agent_id] = {k: v.unsqueeze(0) for k, v in init_state.items()} 

            # Run manual rollout loop
            done = False
            while not done:
                # Query trained policy for actions (deterministic / no exploration)
                actions = {}
                for agent_id in env.agents:
                    # Get policy ID and module
                    policy_id = self.policy_mapping_fn(agent_id)
                    module = self.trainer.get_module(module_id=policy_id)

                    with torch.no_grad():
                        # Convert observations to tensors with batch dimension
                        obs_tensor = {
                            k: torch.tensor(np.array(v), dtype=torch.float32).unsqueeze(0)
                            for k, v in obs[agent_id].items()
                        }
                        batch = {Columns.OBS: obs_tensor}

                        # Add hidden states for recurrent policies
                        if is_recurrent:
                            batch[Columns.STATE_IN] = states[agent_id]

                        # Forward through RLModule
                        output = module._forward_inference(batch)

                        # Sample deterministic action from distribution
                        action_logits = output[Columns.ACTION_DIST_INPUTS]
                        # GRU models return (B, seq_len=1, dim) — squeeze seq_len
                        if action_logits.dim() == 3:
                            action_logits = action_logits.squeeze(1)
                        dist = dist_cls.from_logits(action_logits)
                        det_dist = dist.to_deterministic()
                        action_tensor = det_dist.sample()
                        action = action_tensor.squeeze(0).cpu().numpy()

                        # Update hidden states for recurrent policies
                        if is_recurrent:
                            states[agent_id] = output.get(Columns.STATE_OUT, {})

                    actions[agent_id] = action

                # Record raw actions as (n_warehouses, n_skus) array
                actions_array = np.array([actions[a] for a in env.agents])
                episode_data["actions_raw"].append(actions_array)

                # Step environment and record step info 
                # Infos contain detailed step data when collect_step_info=True
                obs, rewards, terms, truncs, infos = env.step(actions)

                # Extract step info (shared across all agents)
                step_info = infos[env.agents[0]]
                for key, value in step_info.items():
                    episode_data[key].append(value.copy() if isinstance(value, np.ndarray) else value)

                # Record per-warehouse rewards
                rewards_array = np.array([rewards[a] for a in env.agents])
                episode_data["rewards"].append(rewards_array)

                # Check if episode is done
                done = all(truncs.values()) or all(terms.values())

            # Convert all lists to numpy arrays
            episode_data = {k: np.array(v) for k, v in episode_data.items()}
            all_episodes.append(episode_data)

        # Disable flag after rollout
        env.collect_step_info = False

        return all_episodes

    def save_checkpoint(self, path: str):
        """Saves model checkpoint.
        
        Args:
            path (str): Path to save checkpoint.
        """
        self.trainer.save(checkpoint_dir=path)
    
    def load_checkpoint(self, path: str):
        """Loads model checkpoint.
        
        Args:
            path (str): Path to load checkpoint from.
        """
        self.trainer.restore(path)

    def get_policy(self):
        """Gets trained policy.
        
        Returns:
            policy (Any): Trained policy object.
        """
        # Return the first policy (or shared policy if parameter sharing)
        policy_id = list(self.trainer.config.multi_agent.policies.keys())[0]
        return self.trainer.get_policy(policy_id)

    @staticmethod
    def create_env_factory(env_config: 'EnvironmentConfig') -> Callable[[Dict[str, Any]], 'ParallelPettingZooEnv']:
        """
        Creates an environment factory function for RLlib.
        
        Args:
            env_config (EnvironmentConfig): Environment configuration to use for creating instances.
            
        Returns:
            env_factory (Callable): Factory function that RLlib calls to create environment instances.
        """
        from src.environment.environment import InventoryEnvironment
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from typing import Dict, Any
        
        def env_factory(env_meta: Dict[str, Any] = None):
            """
            Factory function that RLlib calls to create environment instances.
            RLlib calls this multiple times to create environment instances for different workers.
            
            Args:
                env_meta (Dict[str, Any]): Dict from RLlib containing environment metadata 
                    (e.g., {"seed": <train_seed or eval_seed>, "data_mode": "train"})
            """
            
            # Extract seed from RLlib's config (train_seed or eval_seed depending on runner type)
            seed = None
            if env_meta:
                seed = env_meta.get("seed")
            
            # Create a new InventoryEnvironment instance
            env = InventoryEnvironment(
                env_config=env_config, 
                seed=seed,  
                env_meta=env_meta
            )
            
            # Wrap for RLlib compatibility
            return ParallelPettingZooEnv(env)
        
        return env_factory
    
    @staticmethod
    def has_gru_from_config(networks_config: Dict[str, Any]) -> bool:
        """
        Searches recursively for GRU architecture in network configs and subconfigs 
        based on the 'type' field.
        
        Args:
            networks_config (Dict[str, Any]): Network configurations (may be nested).
            
        Returns:
            bool: True if any network in the configuration is a GRU, False otherwise.
        """

        # Recursively search for type 'gru' in the network configurations   
        if isinstance(networks_config, dict):
            if networks_config.get("type") == "gru":
                return True
            for value in networks_config.values():
                if isinstance(value, dict):
                    if BaseAlgorithmWrapper.has_gru_from_config(value):
                        return True
        return False

    @staticmethod
    def extract_max_seq_len(networks_config: Dict[str, Any]) -> Optional[int]:
        """
        Recursively search for max_seq_len in network configs.
        
        Args:
            networks_config (Dict[str, Any]): Network configuration dictionary (may be nested).
            
        Returns:
            Optional[int]: max_seq_len value if found. None if no max_seq_len is found.
        """

        # Recursively search for max_seq_len in the network configurations
        if isinstance(networks_config, dict):
            if "max_seq_len" in networks_config:
                return networks_config["max_seq_len"]
            for value in networks_config.values():
                if isinstance(value, dict):
                    result = BaseAlgorithmWrapper.extract_max_seq_len(value)
                    if result is not None:
                        return result
        return None


    
