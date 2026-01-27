from typing import Dict, Any
import torch.nn as nn

from .base import NetworkArchitecture


class CNNArchitecture(NetworkArchitecture):
    """Implements a CNN (Convolutional Neural Network) architecture by creating a
    CNN-based network with configurable layers and activation functions.
    
    Note: This is a simplified implementation that may need adaptation based on observation format.
    """
    
    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        name: str = "",
    ) -> nn.Module:
        """Builds a CNN network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Configuration dictionary with keys:
                - channels (List[int]): Number of channels per layer (default: [32, 64, 128])
                - kernel_sizes (List[int]): Kernel sizes per layer (default: [3, 3, 3])
                - activation (str): Activation function (default: "relu")
            name (str): Optional name prefix
            
        Returns:
            nn.Sequential module
        """

        # Get configuration parameters
        channels = config.get("channels", [32, 64, 128])
        kernel_sizes = config.get("kernel_sizes", [3, 3, 3])
        activation = config.get("activation", "relu")
        
        # Initialize layers list and input channels
        layers = []
        in_channels = 1  # Adapt based on your observation format
        
        # Build convolutional layers
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size))
            layers.append(self._get_activation(activation))
            in_channels = out_channels
        
        # Flatten and output projection
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(channels[-1], output_dim))
        
        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Gets activation function module by name.
        
        Args:
            activation (str): Activation name (relu, tanh, elu)
            
        Returns:
            nn.Module: Activation function module
        """

        # Get activation function module by name
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }
        
        return activations.get(activation.lower(), nn.ReLU())

