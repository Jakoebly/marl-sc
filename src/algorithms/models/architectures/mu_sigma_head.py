import torch
import torch.nn as nn
import numpy as np


class MuSigmaHead(nn.Module):
    """Implements a dual-headed output module for continuous action spaces.
    Splits a shared representation into parallel mu (mean) and sigma 
    (standard deviation) branches, clips each to valid ranges, and 
    concatenates them along the last dimension.
    """

    SIGMA_MIN = 1e-2
    SIGMA_MAX = 1e2

    def __init__(
        self,
        shared_layers: nn.Module,
        mu_branch: nn.Module,
        sigma_branch: nn.Module,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ):
        super().__init__()
        self.shared_layers = shared_layers
        self.mu_branch = mu_branch
        self.sigma_branch = sigma_branch
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the shared layers, mu branch, and sigma branch.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """

        # Forward through shared layers
        shared_out = self.shared_layers(x)

        # Forward through mu branch
        mu = self.mu_branch(shared_out)
        # Clip mu to the action space
        mu = torch.clamp(mu, self.action_low, self.action_high)

        sigma = self.sigma_branch(shared_out)
        sigma = torch.clamp(sigma, self.SIGMA_MIN, self.SIGMA_MAX)

        return torch.cat([mu, sigma], dim=-1)
