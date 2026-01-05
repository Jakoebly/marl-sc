from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from src.algorithms.base import BaseAlgorithmWrapper
    from src.environment.environment import InventoryEnvironment
    from src.config.schema import AlgorithmConfig


# Registry mapping algorithm names to wrapper classes
ALGORITHM_REGISTRY: Dict[str, Type['BaseAlgorithmWrapper']] = {}


def register_algorithm(name: str, algorithm_class: Type['BaseAlgorithmWrapper']):
    """Registers an algorithm wrapper.
    
    Args:
        name (str): Algorithm name (e.g., "ippo", "mappo", "happo")
        algorithm_class (Type[BaseAlgorithmWrapper]): Class implementing the BaseAlgorithmWrapper interface.
    """
    ALGORITHM_REGISTRY[name] = algorithm_class

def get_algorithm(name: str, env: 'InventoryEnvironment', config: 'AlgorithmConfig') -> 'BaseAlgorithmWrapper':
    """Builds an algorithm wrapper instance by retrieving the algorithm 
    name from the registry and instantiating it with the given environment
    and configuration.
    
    Args:
        name (str): Algorithm name
        env (InventoryEnvironment): InventoryEnvironment instance
        config (AlgorithmConfig): Algorithm configuration
        
    Returns:
        algorithm_wrapper (BaseAlgorithmWrapper): Algorithm wrapper instance
    """

    # Check if the algorithm name is registered
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available algorithms: {list(ALGORITHM_REGISTRY.keys())}")
    
    # Get the algorithm class from the registry and instantiate it
    algorithm_class = ALGORITHM_REGISTRY[name]
    return algorithm_class(env, config)


# Register algorithms
from src.algorithms.ippo import IPPOWrapper
from src.algorithms.mappo import MAPPOWrapper
from src.algorithms.happo import HAPPOWrapper

register_algorithm("ippo", IPPOWrapper)
register_algorithm("mappo", MAPPOWrapper)
register_algorithm("happo", HAPPOWrapper)

