from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class MuSigmaHead(nn.Module):
    """Dual-head output module that produces ``[mu, log_std]`` from backbone
    features.  Designed to sit on top of any backbone network (MLP, GRU, ...)
    built separately.

    The mu head is clamped to the action space bounds, and the log_std head
    is clamped to ``[LOG_STD_MIN, LOG_STD_MAX]``.  The concatenated output
    ``[mu, log_std]`` matches what RLlib's ``TorchDiagGaussian.from_logits``
    expects for sampling.

    Optional output activations can be applied independently to each head
    (before clamping).
    """

    LOG_STD_MIN = -4.6
    LOG_STD_MAX = 4.6

    def __init__(
        self,
        backbone_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        output_activation_mu: Optional[str] = None,
        output_activation_sigma: Optional[str] = None,
    ):
        super().__init__()

        # Initialize mu and sigma heads
        self.mu_head = nn.Linear(backbone_dim, action_dim)
        self.sigma_head = nn.Linear(backbone_dim, action_dim)
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

        # Get mu and sigma activations
        self.mu_activation = (
            self._get_activation(output_activation_mu) 
            if output_activation_mu else None
        )
        self.sigma_activation = (
            self._get_activation(output_activation_sigma) 
            if output_activation_sigma else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing ``[mu, log_std]`` concatenated along the last dim.

        Args:
            features (torch.Tensor): Backbone output.
                Shape: ``(B, backbone_dim)`` or ``(B, seq_len, backbone_dim)``.

        Returns:
            torch.Tensor: ``[mu, log_std]`` concatenated.
                Shape: ``(B, 2 * action_dim)`` or ``(B, seq_len, 2 * action_dim)``.
        """

        # Forward through mu head
        mu = self.mu_head(features)

        # Apply activation if specified
        if self.mu_activation is not None:
            mu = self.mu_activation(mu)

        # Forward through sigma head
        log_std = self.sigma_head(features)

        # Apply activation if specified
        if self.sigma_activation is not None:
            log_std = self.sigma_activation(log_std)

        # Clip log_std to the range [-5.0, 2.0]
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return torch.cat([mu, log_std], dim=-1)

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