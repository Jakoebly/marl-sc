from typing import Dict, Any, Optional
import torch.nn as nn
from gymnasium.spaces import Space, Box

from .base import NetworkArchitecture
from .mu_sigma_head import MuSigmaHead


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
        action_space: Optional[Space] = None,
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
            action_space (Optional[Space]): Action space (e.g. Box, Discrete).
            name (str): Optional name prefix

        Returns:
            nn.Module: PyTorch module implementing the MLP architecture
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
        
        # If action space is a Box, build mu and sigma branches
        if isinstance(action_space, Box):
            # Set mu and sigma hidden sizes
            MU_HIDDEN_SIZES = [256]
            SIGMA_HIDDEN_SIZES = [30]

            # Build mu branch
            current_mu_dim = current_dim
            mu_layers = []
            for hidden_size in MU_HIDDEN_SIZES:
                mu_layers.append(nn.Linear(current_mu_dim, hidden_size))
                mu_layers.append(self._get_activation(activation))
                current_mu_dim = hidden_size
            mu_layers.append(nn.Linear(current_mu_dim, output_dim))
            mu_layers.append(self._get_activation("tanh"))

            # Build sigma branch
            sigma_layers = []
            current_sigma_dim = current_dim
            for hidden_size in SIGMA_HIDDEN_SIZES:
                sigma_layers.append(nn.Linear(current_sigma_dim, hidden_size))
                sigma_layers.append(self._get_activation(activation))
                current_sigma_dim = hidden_size
            sigma_layers.append(nn.Linear(current_sigma_dim, output_dim))
            sigma_layers.append(self._get_activation("relu"))

            # Build mu sigma head
            mu_sigma_head = MuSigmaHead(
                shared_layers=nn.Sequential(*layers),
                mu_branch=nn.Sequential(*mu_layers),
                sigma_branch=nn.Sequential(*sigma_layers),
                action_low=action_space.low,
                action_high=action_space.high,
            )

            return mu_sigma_head

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

