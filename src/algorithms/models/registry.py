from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .architectures import NetworkArchitecture

# Registry mapping architecture names to architecture classes
ARCHITECTURE_REGISTRY: Dict[str, Type['NetworkArchitecture']] = {}


def register_architecture(name: str, architecture_class: Type['NetworkArchitecture']):
    """Registers a network architecture.
    
    Args:
        name (str): Architecture name (e.g., "mlp", "gru", "cnn")
        architecture_class (Type[NetworkArchitecture]): Architecture class inheriting from NetworkArchitecture
    """

    ARCHITECTURE_REGISTRY[name] = architecture_class

def get_architecture(name: str) -> Type['NetworkArchitecture']:
    """Gets architecture class from registry.
    
    Args:
        name (str): Architecture name
        
    Returns:
        architecture_class (Type[NetworkArchitecture]): Architecture class
    """

    # Check if architecture name is registered
    if name not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture: {name}. "
            f"Available: {list(ARCHITECTURE_REGISTRY.keys())}"
        )

    # Get the architecture class from the registry
    architecture_class = ARCHITECTURE_REGISTRY[name]
    return architecture_class


# Register all architectures
from .architectures import MLPArchitecture, GRUArchitecture, CNNArchitecture

register_architecture("mlp", MLPArchitecture)
register_architecture("gru", GRUArchitecture)
register_architecture("cnn", CNNArchitecture)

