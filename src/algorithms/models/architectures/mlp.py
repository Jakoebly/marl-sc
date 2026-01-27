from typing import Dict, Any
import torch.nn as nn

from .base import NetworkArchitecture


class MLPArchitecture(NetworkArchitecture):
    """Implements an MLP (Multi-Layer Perceptron) architecture by creating a
    feedforward neural network with configurable hidden layers and activation 
    functions.
    """
    
    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        name: str = "",
    ) -> nn.Module:
        """Builds an MLP network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Configuration dictionary with keys:
                - hidden_sizes: List[int] - Hidden layer sizes (default: [256, 256])
                - activation (str): Activation function name (default: "relu")
                - output_activation (Optional[str]): Output activation (default: None)
            name (str): Optional name prefix

        Returns:
            nn.Sequential: PyTorch module implementing the MLP architecture
        """

        # Get configuration parameters
        hidden_sizes = config.get("hidden_sizes", [128, 128])
        activation = config.get("activation", "relu")
        output_activation = config.get("output_activation", None)
        
        # Initialize layers list and current dimension
        layers = []
        current_dim = input_dim
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(self._get_activation(activation))
            current_dim = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Optional output activation
        if output_activation:
            layers.append(self._get_activation(output_activation))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Gets activation function module by name.
        
        Args:
            activation (str): Activation name (relu, tanh, elu, gelu, sigmoid)
            
        Returns:
            nn.Module: Activation function module
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation.lower(), nn.ReLU())

