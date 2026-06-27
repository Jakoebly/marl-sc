import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Space

from .base import NetworkArchitecture


def _module_activation(name: str) -> nn.Module:
    """Returns an activation module by name (used for the MLP head)."""
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    return activations.get(name.lower(), nn.ReLU)()


def _tf_activation(name: str):
    """Returns a functional activation for the Transformer feed-forward block.

    ``nn.TransformerEncoderLayer`` accepts a unary callable for ``activation``.
    """
    activations = {
        "relu": F.relu,
        "gelu": F.gelu,
        "elu": F.elu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    return activations.get(name.lower(), F.relu)


class PerSkuAttentionNetwork(nn.Module):
    """Per-SKU self-attention backbone.

    The flat observation is interpreted as ``n_blocks`` repeated warehouse
    blocks (1 block for a local observation, ``1 + n_warehouses`` for a MAPPO
    global critic, ``n_warehouses`` for the CPPO joint observation). Within
    each block the per-SKU columns are gathered into ``n_skus`` tokens of width
    ``per_sku_feat`` (using the index map in ``obs_layout``, which handles both
    contiguous and strided — e.g. pipeline — per-SKU blocks). Tokens are
    projected to ``d_model`` and processed by a Transformer encoder whose
    attention runs across the SKU tokens *within each block* (per-warehouse
    attention with shared weights). The attended tokens are flattened,
    concatenated with the non-per-SKU context features (warehouse-id one-hot and
    aggregates), and passed through an MLP head.

    The number of blocks is derived from ``input_dim // block_size`` at
    construction time, so actor (local) and critic (global) instances size
    their heads correctly.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        obs_layout: Dict[str, Any],
    ):
        super().__init__()

        # --- Observation layout (one warehouse block) ---
        block_size = int(obs_layout["block_size"])
        n_skus = int(obs_layout["n_skus"])
        per_sku_indices: List[List[int]] = obs_layout["per_sku_indices"]
        context_indices: List[int] = obs_layout["context_indices"]

        if block_size <= 0:
            raise ValueError(f"PerSkuAttentionNetwork: invalid block_size {block_size}")
        if input_dim % block_size != 0:
            raise ValueError(
                f"PerSkuAttentionNetwork: input_dim ({input_dim}) is not a multiple of "
                f"block_size ({block_size}); cannot tokenize per-SKU."
            )
        n_blocks = input_dim // block_size

        per_sku_feat = len(per_sku_indices[0]) if n_skus > 0 else 0
        if any(len(idx) != per_sku_feat for idx in per_sku_indices):
            raise ValueError(
                "PerSkuAttentionNetwork: inconsistent per-SKU feature counts across SKUs"
            )
        if per_sku_feat == 0:
            raise ValueError(
                "PerSkuAttentionNetwork: no per-SKU features found in observation layout"
            )

        self.block_size = block_size
        self.n_blocks = n_blocks
        self.n_skus = n_skus
        self.per_sku_feat = per_sku_feat
        self.context_size = len(context_indices)

        # Flattened gather index for per-SKU columns within one block:
        # (n_skus * per_sku_feat,). Non-persistent: rebuilt from config on load.
        sku_flat = [c for row in per_sku_indices for c in row]
        self.register_buffer(
            "sku_flat_idx", torch.tensor(sku_flat, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "context_idx",
            torch.tensor(context_indices, dtype=torch.long),
            persistent=False,
        )

        # --- Architecture hyperparameters ---
        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        n_layers = int(config.get("n_layers", 1))
        ff_hidden = int(config["ff_hidden"])
        dropout = float(config.get("dropout", 0.0))
        activation_name = config.get("activation", "relu")
        head_hidden_sizes = config.get("head_hidden_sizes", []) or []
        output_activation = config.get("output_activation", None)

        self.d_model = d_model

        # Shared per-SKU token projection.
        self.token_proj = nn.Linear(per_sku_feat, d_model)

        # Transformer encoder over the SKU tokens (per-block / per-warehouse).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden,
            dropout=dropout,
            activation=_tf_activation(activation_name),
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # MLP head over [flattened attended tokens | flattened context].
        head_input_dim = n_blocks * n_skus * d_model + n_blocks * self.context_size
        layers: List[nn.Module] = []
        current = head_input_dim
        for hidden in head_hidden_sizes:
            layers.append(nn.Linear(current, hidden))
            layers.append(_module_activation(activation_name))
            current = hidden
        layers.append(nn.Linear(current, output_dim))
        if output_activation:
            layers.append(_module_activation(output_activation))
        self.head = nn.Sequential(*layers)

        # Attention-weight capture, used only by evaluation rollouts when
        # --visualize is set. Disabled during training so the forward path and
        # numerics are unchanged. Toggled on by BaseAlgorithmWrapper.rollout.
        self._capture_attn = False
        self._last_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Flat observation. Shape: ``(..., n_blocks * block_size)``
                (typically ``(B, input_dim)``).

        Returns:
            torch.Tensor: ``(..., output_dim)``.
        """
        leading = x.shape[:-1]
        total = x.shape[-1]
        expected = self.n_blocks * self.block_size
        if total != expected:
            raise ValueError(
                f"PerSkuAttentionNetwork.forward: expected last dim {expected} "
                f"(n_blocks={self.n_blocks} * block_size={self.block_size}), got {total}"
            )

        # Collapse any leading dims (B, or B x T) into a single batch dim.
        batch = math.prod(leading) if len(leading) > 0 else 1
        x = x.reshape(batch, self.n_blocks, self.block_size)

        # Gather per-SKU tokens: (batch, n_blocks, n_skus, per_sku_feat).
        sku = x[..., self.sku_flat_idx].reshape(
            batch, self.n_blocks, self.n_skus, self.per_sku_feat
        )

        # Project tokens, then run per-block attention by folding blocks into
        # the batch dim so SKUs attend only within their own warehouse block.
        tokens = self.token_proj(sku)  # (batch, n_blocks, n_skus, d_model)
        tokens = tokens.reshape(batch * self.n_blocks, self.n_skus, self.d_model)
        # Encode the SKU tokens. In capture mode (eval/visualization) also record
        # the per-SKU self-attention weights; otherwise use the plain fast path.
        if self._capture_attn:
            attended, attn = self._encode_with_attn(tokens)
            self._last_attn = attn.reshape(
                batch, self.n_blocks, self.n_skus, self.n_skus
            ).detach()
        else:
            attended = self.encoder(tokens)  # (batch * n_blocks, n_skus, d_model)
        attended = attended.reshape(batch, self.n_blocks * self.n_skus * self.d_model)

        # Concatenate the (per-block) context features, if any.
        if self.context_size > 0:
            ctx = x[..., self.context_idx].reshape(
                batch, self.n_blocks * self.context_size
            )
            features = torch.cat([attended, ctx], dim=-1)
        else:
            features = attended

        out = self.head(features)  # (batch, output_dim)

        # Restore leading dims.
        return out.reshape(*leading, out.shape[-1])

    def _encode_with_attn(self, tokens):
        """Run the Transformer encoder while capturing self-attention weights.

        Used only for evaluation/visualization (``self._capture_attn``). Returns
        ``(encoded, attn)`` where ``attn`` is the head-averaged attention matrix
        averaged over encoder layers, shape ``(N, n_skus, n_skus)`` with
        ``N = batch * n_blocks``. Assumes ``norm_first=False`` (the PyTorch
        default used here) so each layer's ``self_attn`` sees its raw input and
        the captured weights match the encoder's internal computation; the
        returned ``encoded`` equals ``self.encoder(tokens)`` for this config.
        """
        h = tokens
        mats = []
        for layer in self.encoder.layers:
            _, w = layer.self_attn(
                h, h, h, need_weights=True, average_attn_weights=True
            )
            mats.append(w)
            h = layer(h)
        if self.encoder.norm is not None:
            h = self.encoder.norm(h)
        attn = torch.stack(mats, dim=0).mean(dim=0)
        return h, attn


class AttentionArchitecture(NetworkArchitecture):
    """Builds a :class:`PerSkuAttentionNetwork` from an architecture config."""

    def build(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        action_space: Optional[Space] = None,
        name: str = "",
        obs_layout: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        if obs_layout is None:
            raise ValueError(
                "AttentionArchitecture requires obs_layout (per-SKU observation layout). "
                "It is injected by the RLModule from model_config['obs_layout']; ensure the "
                "algorithm wrapper sets it."
            )
        return PerSkuAttentionNetwork(input_dim, output_dim, config, obs_layout)
