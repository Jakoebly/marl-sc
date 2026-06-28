import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Space

from .base import NetworkArchitecture


def _module_activation(name: str) -> nn.Module:
    """Gets activation function module by name.

    Args:
        name (str): Activation name (relu, tanh, elu, gelu, sigmoid)

    Returns:
        nn.Module: Activation function module
    """

    # Get activation function module by name
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    
    return activations.get(name.lower(), nn.ReLU)()


def _tf_activation(name: str):
    """Gets activation function by name for Transformer feed-forward blocks.

    ``nn.TransformerEncoderLayer`` accepts a unary callable for ``activation``.

    Args:
        name (str): Activation name (relu, tanh, elu, gelu, sigmoid)

    Returns:
        callable: Activation function
    """

    # Get activation function by name
    activations = {
        "relu": F.relu,
        "gelu": F.gelu,
        "elu": F.elu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    return activations.get(name.lower(), F.relu)


class PerSkuAttentionNetwork(nn.Module):
    """Implements a Per-SKU self-attention network by creating a Transformer
    encoder over SKU tokens (per-block / per-warehouse) with configurable
    encoder layers, attention heads, and MLP head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        obs_layout: Dict[str, Any],
    ):
        super().__init__()

        # Parse observation layout (one warehouse block)
        block_size = int(obs_layout["block_size"])
        n_skus = int(obs_layout["n_skus"])
        per_sku_indices: List[List[int]] = obs_layout["per_sku_indices"]
        context_indices: List[int] = obs_layout["context_indices"]

        # Validate block size and ensure the flat observation splits evenly into blocks
        if block_size <= 0:
            raise ValueError(f"PerSkuAttentionNetwork: invalid block_size {block_size}")
        if input_dim % block_size != 0:
            raise ValueError(
                f"PerSkuAttentionNetwork: input_dim ({input_dim}) is not a multiple of "
                f"block_size ({block_size}); cannot tokenize per-SKU."
            )
        n_blocks = input_dim // block_size

        # Validate that every SKU exposes the same number of per-SKU features
        per_sku_feat = len(per_sku_indices[0]) if n_skus > 0 else 0
        if any(len(idx) != per_sku_feat for idx in per_sku_indices):
            raise ValueError(
                "PerSkuAttentionNetwork: inconsistent per-SKU feature counts across SKUs"
            )
        if per_sku_feat == 0:
            raise ValueError(
                "PerSkuAttentionNetwork: no per-SKU features found in observation layout"
            )

        # Store layout dimensions used during the forward pass
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.n_skus = n_skus
        self.per_sku_feat = per_sku_feat
        self.context_size = len(context_indices)

        # Build flattened gather index for per-SKU columns within one block
        sku_flat = [c for row in per_sku_indices for c in row]
        self.register_buffer(
            "sku_flat_idx", torch.tensor(sku_flat, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "context_idx",
            torch.tensor(context_indices, dtype=torch.long),
            persistent=False,
        )

        # Get configuration parameters
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        n_layers = int(config.get("n_layers", 1))
        ff_hidden = int(config["ff_hidden"])
        dropout = float(config.get("dropout", 0.0))
        activation_name = config.get("activation", "relu")
        head_hidden_sizes = config.get("head_hidden_sizes", []) or []
        output_activation = config.get("output_activation", None)

        self.d_model = d_model

        # Build shared per-SKU token projection
        self.token_proj = nn.Linear(per_sku_feat, d_model)

        # Build Transformer encoder over SKU tokens (per-block / per-warehouse)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation=_tf_activation(activation_name),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Build MLP head over attended tokens and context features
        head_input_dim = n_blocks * n_skus * d_model + n_blocks * self.context_size
        layers: List[nn.Module] = []
        current = head_input_dim
        for hidden in head_hidden_sizes:
            layers.append(nn.Linear(current, hidden))
            layers.append(_module_activation(activation_name))
            current = hidden

        # Append output layer
        layers.append(nn.Linear(current, output_dim))

        # Append optional output activation
        if output_activation:
            layers.append(_module_activation(output_activation))
        self.head = nn.Sequential(*layers)

        # Initialize attention-weight capture for evaluation rollouts (--visualize)
        self._capture_attn = False
        self._last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass through the Per-SKU attention network.

        Args:
            x (torch.Tensor): Flat observation tensor with shape
                ``(..., n_blocks * block_size)`` (typically ``(B, input_dim)``).

        Returns:
            torch.Tensor: Output tensor with shape ``(..., output_dim)``.
        """
        leading = x.shape[:-1]
        total = x.shape[-1]
        expected = self.n_blocks * self.block_size
        if total != expected:
            raise ValueError(
                f"PerSkuAttentionNetwork.forward: expected last dim {expected} "
                f"(n_blocks={self.n_blocks} * block_size={self.block_size}), got {total}"
            )

        # Collapse leading dimensions into a single batch dimension
        batch = math.prod(leading) if len(leading) > 0 else 1
        x = x.reshape(batch, self.n_blocks, self.block_size)

        # Gather per-SKU tokens
        sku = x[..., self.sku_flat_idx].reshape(
            batch, self.n_blocks, self.n_skus, self.per_sku_feat
        )

        # Project tokens and run per-block self-attention
        tokens = self.token_proj(sku)
        tokens = tokens.reshape(batch * self.n_blocks, self.n_skus, self.d_model)

        # Encode SKU tokens (optionally capture attention weights for visualization)
        if self._capture_attn:
            attended, attn = self._encode_with_attn(tokens)
            self._last_attn = attn.reshape(
                batch, self.n_blocks, self.n_skus, self.n_skus
            ).detach()
        else:
            attended = self.encoder(tokens)
        attended = attended.reshape(batch, self.n_blocks * self.n_skus * self.d_model)

        # Concatenate per-block context features, if any
        if self.context_size > 0:
            ctx = x[..., self.context_idx].reshape(
                batch, self.n_blocks * self.context_size
            )
            features = torch.cat([attended, ctx], dim=-1)
        else:
            features = attended

        out = self.head(features)

        # Restore leading dimensions
        return out.reshape(*leading, out.shape[-1])

    def _encode_with_attn(self, tokens):
        """Runs the Transformer encoder while capturing self-attention weights.

        Used only when ``self._capture_attn`` is enabled (evaluation /
        visualization). Assumes ``norm_first=False`` (the PyTorch default used
        here) so captured weights match the encoder's internal computation.

        Args:
            tokens (torch.Tensor): SKU token embeddings with shape
                ``(N, n_skus, d_model)`` where ``N = batch * n_blocks``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Encoded tokens with shape
                ``(N, n_skus, d_model)`` and head-averaged attention weights
                with shape ``(N, n_skus, n_skus)``, averaged over encoder layers.
        """

        # Initialize input tokens and attention weight matrix list
        h = tokens
        mats = []

        # Run encoder layers and collect self-attention weights
        for layer in self.encoder.layers:
            _, w = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=True
            )
            mats.append(w)
            h = layer(h)

        # Apply final layer norm, if present
        if self.encoder.norm is not None:
            h = self.encoder.norm(h)

        # Average attention weights across encoder layers
        attn = torch.stack(mats, dim=0).mean(dim=0)
        return h, attn


class AttentionArchitecture(NetworkArchitecture):
    """Implements a Per-SKU self-attention architecture by creating a
    Transformer-based network with configurable encoder layers, attention heads,
    and MLP head.
    """

    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        action_space: Optional[Space] = None,
        name: str = "",
        obs_layout: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        """Builds a Per-SKU attention network.

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            config (Dict[str, Any]): Configuration dictionary with keys:
                - d_model (int): Transformer model dimension
                - n_heads (int): Number of attention heads
                - n_layers (int): Number of Transformer encoder layers (default: 1)
                - ff_hidden (int): Feed-forward hidden dimension
                - dropout (float): Dropout rate (default: 0.0)
                - activation (str): Activation function name (default: "relu")
                - head_hidden_sizes (List[int]): Hidden layer sizes for the MLP head (default: [])
                - output_activation (Optional[str]): Output activation (default: None)
            action_space (Optional[Space]): Action space (e.g. Box, Discrete).
            name (str): Optional name prefix
            obs_layout (Optional[Dict[str, Any]]): Per-SKU observation layout
                (required; injected by the RLModule from model_config).

        Returns:
            nn.Module: PyTorch module implementing the Per-SKU attention architecture
        """
        if obs_layout is None:
            raise ValueError(
                "AttentionArchitecture requires obs_layout (per-SKU observation layout). "
                "It is injected by the RLModule from model_config['obs_layout']; ensure the "
                "algorithm wrapper sets it."
            )
        return PerSkuAttentionNetwork(input_dim, output_dim, config, obs_layout)
