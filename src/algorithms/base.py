from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.environment.environment import InventoryEnvironment
    from src.config.schema import AlgorithmConfig, EnvironmentConfig
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


class BaseAlgorithmWrapper(ABC):
    """Base class for RLlib algorithm wrappers."""
    
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
                    (e.g., {"seed": root_seed, "data_mode": "train"})
            """
            # Extract seed from RLlib's config (contains root_seed)
            root_seed = None
            if env_meta:
                root_seed = env_meta.get("seed")
            
            # Create a new InventoryEnvironment instance
            env = InventoryEnvironment(
                env_config=env_config, 
                seed=root_seed,  
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

    @abstractmethod
    def __init__(self, env: 'InventoryEnvironment', config: 'AlgorithmConfig'):
        """Initializes algorithm wrapper.
        
        Args:
            env (InventoryEnvironment): InventoryEnvironment instance (PettingZoo ParallelEnv)
            config (AlgorithmConfig): Algorithm configuration
        """
        pass
    
    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Runs training loop.
        
        Returns:
            metrics (Dict[str, Any]): Training metrics dictionary.
        """
        pass
    
    @abstractmethod
    def get_policy(self):
        """Gets trained policy.
        
        Returns:
            policy (Any): Trained policy object.
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Saves model checkpoint.
        
        Args:
            path (str): Path to save checkpoint.
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """Loads model checkpoint.
        
        Args:
            path (str): Path to load checkpoint from.
        """
        pass

