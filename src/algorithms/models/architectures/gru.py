from typing import Dict, Any, Optional
import torch.nn as nn
from gymnasium.spaces import Space, Box

from .base import NetworkArchitecture
from .mu_sigma_head import MuSigmaHead


class GRUArchitecture(NetworkArchitecture):
    """Implements a GRU (Gated Recurrent Unit) architecture by creating a
    GRU-based network with configurable layers, optional bidirectional 
    processing, and configurable activation functions.
    """
    
    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        action_space: Optional[Space] = None,
        name: str = "",
    ) -> nn.Module:
        """Builds a GRU network.
        
        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Configuration dictionary with keys:
                - hidden_size (int): GRU hidden size (default: 128)
                - num_layers (int): Number of GRU layers (default: 2)
                - bidirectional (bool): Whether to use bidirectional GRU (default: False)
                - dropout (float): Dropout rate (default: 0.0)
                - activation (Optional[str]): Activation function after GRU output, before projection (default: None)
                - output_activation (Optional[str]): Activation function after output projection (default: None)
            action_space (Optional[Space]): Action space (e.g. Box, Discrete).
            name (str): Optional name prefix
            
        Returns:
            nn.ModuleDict: PyTorch module implementing the GRU architecture
        """

        # Get configuration parameters
        hidden_size = config.get("hidden_size", 128)
        num_layers = config.get("num_layers", 2)
        bidirectional = config.get("bidirectional", False)
        dropout = config.get("dropout", 0.0)
        activation = config.get("activation", None)
        output_activation = config.get("output_activation", None)
        
        # Build GRU layer
        gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        
        # GRU output dimension depends on bidirectional flag
        gru_output_dim = hidden_size * (2 if bidirectional else 1)

        # Optional activation after GRU output (before projection)
        if activation:
            layers.append(self._get_activation(activation))
        
        # If action space is a Box, build mu and sigma branches
        if isinstance(action_space, Box):
            # Set mu and sigma hidden sizes
            MU_HIDDEN_SIZES = [256]
            SIGMA_HIDDEN_SIZES = [30]

            # Build mu branch
            current_mu_dim = gru_output_dim
            mu_layers = []
            for hidden_size in MU_HIDDEN_SIZES:
                mu_layers.append(nn.Linear(current_mu_dim, hidden_size))
                mu_layers.append(self._get_activation(activation))
                current_mu_dim = hidden_size
            mu_layers.append(nn.Linear(current_mu_dim, output_dim))
            mu_layers.append(self._get_activation("tanh"))

            # Build sigma branch
            current_sigma_dim = gru_output_dim
            sigma_layers = []
            for hidden_size in SIGMA_HIDDEN_SIZES:
                sigma_layers.append(nn.Linear(current_sigma_dim, hidden_size))
                sigma_layers.append(self._get_activation(activation))
                current_sigma_dim = hidden_size
            sigma_layers.append(nn.Linear(current_sigma_dim, output_dim))
            sigma_layers.append(self._get_activation("relu"))

            # Build mu sigma head
            output_proj = MuSigmaHead(
                shared_layers=nn.Sequential(*layers),
                mu_branch=nn.Sequential(*mu_layers),
                sigma_branch=nn.Sequential(*sigma_layers),
                action_low=action_space.low,
                action_high=action_space.high,
            )
        
        # If action space is not a Box, build a simple output projection
        else:            
            # Output projection layer
            layers = []
            layers.append(nn.Linear(gru_output_dim, output_dim))
            
            # Optional output activation
            if output_activation:
                layers.append(self._get_activation(output_activation))
            
            # Create projection module (may include activations)
            if len(layers) == 1:
                # Only linear layer, no activations
                output_proj = layers[0]
            else:
                # Multiple layers (activation + linear, or linear + activation, or both)
                output_proj = nn.Sequential(*layers)
        
        return nn.ModuleDict({
            "gru": gru,
            "output_proj": output_proj,
        })
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Gets activation function module by name.
        
        Args:
            activation (str): Activation name (relu, tanh, elu, gelu, sigmoid)
            
        Returns:
            nn.Module: Activation function module
        """
        
        # Get activation function module by name
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "sigmoid": nn.Sigmoid(),
        }
        return activations.get(activation.lower(), nn.ReLU())

