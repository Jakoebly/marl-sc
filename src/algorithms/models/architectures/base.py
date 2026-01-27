from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class NetworkArchitecture(ABC):
    """Implements a base class for network architecture building blocks.
    All architecture implementations (MLP, GRU, CNN, etc.) should inherit
    from this class and implement the build() method.
    """
    
    @abstractmethod
    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        name: str = "",
    ) -> nn.Module:
        """Builds a network module.

        Args:
            input_dim (int): Input dimension for the network
            output_dim (int): Output dimension for the network
            config (Dict[str, Any]): Architecture-specific configuration dictionary
            name (str): Optional name prefix for layers (for debugging)
            
        Returns:
            nn.Module: PyTorch module implementing the architecture
        """
        pass

