"""DINO-WM Transformer world model for predicting future patch latents.

This module implements a simple Transformer-based world model that predicts
action-conditioned future DINOv2 spatial patch features.

Architecture:
- Input: current patch latents [B, T, P, D] and actions [B, T, A]
- Output: predicted future patch latents [B, H, P, D]

The model uses:
- Patch embedding projection
- Action embedding projection
- Transformer encoder for temporal modeling
- Patch prediction head for future latent prediction
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import math


class DINOwMTransformer(nn.Module):
    """Transformer world model for DINO-WM.

    Predicts future patch latents from current patch latents and actions.

    Shape contract:
    - Input patch_latents: [B, T, P, D] (context patch latents)
    - Input actions: [B, T, A] (action history)
    - Output: [B, H, P, D] (predicted future patch latents)

    Args:
        patch_dim: Number of spatial patches (P).
        feature_dim: Patch feature dimension (D).
        action_dim: Action dimension (A).
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer layers.
        future_horizon: Number of future timesteps to predict (H).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        patch_dim: int = 256,
        feature_dim: int = 384,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        future_horizon: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_dim = patch_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.future_horizon = future_horizon

        # Patch embedding projection: D -> hidden_dim
        self.patch_proj = nn.Linear(feature_dim, hidden_dim)

        # Action embedding projection: A -> hidden_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Temporal position encoding
        self.temporal_pos_encoding = nn.Parameter(
            torch.randn(1, 64, hidden_dim) * 0.02  # Max 64 timesteps
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Future prediction head: hidden_dim -> feature_dim
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        patch_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            patch_latents: [B, T, P, D] context patch latents
            actions: [B, T, A] action history

        Returns:
            predicted: [B, H, P, D] predicted future patch latents
        """
        B, T, P, D = patch_latents.shape
        _, _, A = actions.shape
        H = self.future_horizon

        # Project patches to hidden dim: [B, T, P, D] -> [B, T, P, hidden]
        patch_emb = self.patch_proj(patch_latents)

        # Project actions: [B, T, A] -> [B, T, hidden]
        action_emb = self.action_proj(actions)

        # Add temporal position encoding to actions
        action_emb = action_emb + self.temporal_pos_encoding[:, :T, :]

        # Expand action_emb to match patch dimensions: [B, T, hidden] -> [B, T, 1, hidden]
        action_expanded = action_emb.unsqueeze(2).expand(-1, -1, P, -1)

        # Combine patch and action embeddings
        combined = patch_emb + action_expanded

        # Reshape for Transformer: [B, T*P, hidden]
        combined_flat = combined.reshape(B, T * P, self.hidden_dim)

        # Apply Transformer
        transformer_out = self.transformer(combined_flat)

        # Reshape back: [B, T*P, hidden] -> [B, T, P, hidden]
        transformer_out = transformer_out.reshape(B, T, P, self.hidden_dim)

        # Take last timestep for prediction
        last_hidden = transformer_out[:, -1, :, :]  # [B, P, hidden]

        # Predict future patch latents for each horizon step
        predictions = []
        for h in range(H):
            # Predict next patch latent
            pred = self.pred_head(last_hidden)  # [B, P, D]
            predictions.append(pred)

            # Update last_hidden with prediction for autoregressive prediction
            last_hidden = last_hidden + self.patch_proj(pred)

        # Stack predictions: [B, H, P, D]
        predicted = torch.stack(predictions, dim=1)

        return predicted

    def predict_one_step(
        self,
        patch_latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next patch latent (H=1).

        Args:
            patch_latents: [B, T, P, D] context patch latents
            actions: [B, T, A] action history

        Returns:
            predicted: [B, 1, P, D] predicted next patch latent
        """
        # Save original future_horizon
        orig_h = self.future_horizon
        self.future_horizon = 1

        predicted = self.forward(patch_latents, actions)

        # Restore original future_horizon
        self.future_horizon = orig_h

        return predicted


def build_dinowm_model(config: dict[str, Any]) -> DINOwMTransformer:
    """Build DINO-WM model from config.

    Args:
        config: Model configuration dict with keys:
            - patch_dim: Number of spatial patches
            - feature_dim: Patch feature dimension
            - action_dim: Action dimension
            - hidden_dim: Transformer hidden dimension
            - num_heads: Number of attention heads
            - num_layers: Number of Transformer layers
            - future_horizon: Number of future timesteps
            - dropout: Dropout rate

    Returns:
        DINOwMTransformer model instance.
    """
    return DINOwMTransformer(
        patch_dim=config.get("patch_dim", 256),
        feature_dim=config.get("feature_dim", 384),
        action_dim=config.get("action_dim", 7),
        hidden_dim=config.get("hidden_dim", 256),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        future_horizon=config.get("future_horizon", 2),
        dropout=config.get("dropout", 0.1),
    )


__all__ = ["DINOwMTransformer", "build_dinowm_model"]
