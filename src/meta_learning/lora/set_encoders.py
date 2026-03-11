from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor


class BaseSetEncoder(ABC, nn.Module):
    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_dim: int,
        z_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        label_embedding_dim: int = 32,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.hidden_dim = hidden_dim
        self.label_embedding_dim = label_embedding_dim
        self.class_emb = nn.Embedding(num_classes + 1, self.label_embedding_dim)

        # --- The Phi (φ) Network ---
        # Processes individual support elements.
        # Note: It now outputs hidden_dim instead of z_dim.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- The Rho (ρ) Network ---
        # Processes the globally aggregated set representation into the final z_dim.
        self.post_aggregation_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    @abstractmethod
    def aggregate(self, fused_support: Tensor) -> Tensor:
        # fused_support: (batch_size, k_shots, hidden_dim) -> (batch_size, hidden_dim)
        raise NotImplementedError

    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        batch_size, k_shots, C, T = support_x.shape
        x_flat = support_x.view(batch_size * k_shots, C, T)
        # (B * K, C, T)
        y_flat = support_y.view(batch_size * k_shots)
        # (B * K)

        # 1. Feature & Label Extraction
        h_x = self.feature_encoder.encode(x_flat)  # type: ignore[attr-defined]
        # (B * K, F)
        h_y = self.class_emb(y_flat)
        # (B * K, E)
        h_combined = torch.cat([h_x, h_y], dim=1)
        # (B * K, F + E)

        # 2. Element-wise processing (φ)
        fused_support = self.fusion_mlp(h_combined)
        # (B * K, H)
        fused_support = fused_support.view(batch_size, k_shots, -1)
        # (B, K, H)

        # 3. Permutation-invariant aggregation
        aggregated = self.aggregate(fused_support)
        # (B, H)

        # 4. Global set processing (ρ) to yield final context vector
        z = self.post_aggregation_mlp(aggregated)
        # (B, Z)

        return z


class MeanSetEncoder(BaseSetEncoder):
    def aggregate(self, fused_support: Tensor) -> Tensor:
        return fused_support.mean(dim=1)


class SelfAttentionMeanSetEncoder(BaseSetEncoder):
    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_dim: int,
        z_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__(
            feature_encoder=feature_encoder,
            feature_dim=feature_dim,
            z_dim=z_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
        self.self_attention = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            batch_first=True,
        )

    def aggregate(self, fused_support: Tensor) -> Tensor:
        attended = self.self_attention(fused_support)
        return attended.mean(dim=1)


class QueryAttentionSetEncoder(BaseSetEncoder):
    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_dim: int,
        z_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__(
            feature_encoder=feature_encoder,
            feature_dim=feature_dim,
            z_dim=z_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.xavier_normal_(self.query)

    def aggregate(self, fused_support: Tensor) -> Tensor:
        batch_size = fused_support.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        attended, _ = self.attention(query, fused_support, fused_support)
        return attended.squeeze(1)
