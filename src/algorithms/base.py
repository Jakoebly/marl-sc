from abc import ABC, abstractmethod
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.environment import InventoryEnvironment
    from src.config.schema import AlgorithmConfig


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

