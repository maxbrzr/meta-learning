from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseClassAwareSetEncoder(ABC, nn.Module):
    """
    Class-aware set encoder that returns per-class embeddings.
    Subclasses define how support shots become class embeddings.
    The base class only converts class embeddings to a single task vector for LoRA.
    """

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
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.label_embedding_dim = label_embedding_dim
        self.class_emb = nn.Embedding(num_classes + 1, self.label_embedding_dim)

        # Element-wise processing phi
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Task projection rho_t from flattened class embeddings.
        self.task_post_aggregation_mlp = nn.Sequential(
            nn.Linear(num_classes * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    @abstractmethod
    def build_class_embeddings(
        self, fused_support: Tensor, support_y: Tensor
    ) -> Tensor:
        # fused_support: (B, K, H), support_y: (B, K) -> (B, N, H)
        raise NotImplementedError

    def _group_mean(self, fused_support: Tensor, support_y: Tensor) -> Tensor:
        one_hot = F.one_hot(support_y.long(), num_classes=self.num_classes).to(
            fused_support.dtype
        )
        # (B, K, N)
        class_counts = one_hot.sum(dim=1).clamp(min=1.0)
        # (B, N)
        class_sums = torch.einsum("bkn,bkh->bnh", one_hot, fused_support)
        # (B, N, H)
        class_means = class_sums / class_counts.unsqueeze(-1)
        # (B, N, H)
        return class_means

    def to_task_embedding(self, class_embeddings: Tensor) -> Tensor:
        batch_size = class_embeddings.shape[0]
        flattened = class_embeddings.reshape(
            batch_size, self.num_classes * self.hidden_dim
        )
        # (B, N * H)
        return self.task_post_aggregation_mlp(flattened)

    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        batch_size, k_shots, C, T = support_x.shape
        x_flat = support_x.view(batch_size * k_shots, C, T)
        # (B * K, C, T)
        y_flat = support_y.view(batch_size * k_shots)
        # (B * K)

        h_x = self.feature_encoder.encode(x_flat)  # type: ignore[attr-defined]
        # (B * K, F)
        h_y = self.class_emb(y_flat)
        # (B * K, E)
        h_combined = torch.cat([h_x, h_y], dim=1)
        # (B * K, F + E)

        fused_support = self.fusion_mlp(h_combined).view(batch_size, k_shots, -1)
        # (B, K, H)
        class_embeddings = self.build_class_embeddings(fused_support, support_y)
        # (B, N, H)
        return class_embeddings


class MeanClassAwareSetEncoder(BaseClassAwareSetEncoder):
    def build_class_embeddings(
        self, fused_support: Tensor, support_y: Tensor
    ) -> Tensor:
        return self._group_mean(fused_support, support_y)


class SelfAttentionMeanClassAwareSetEncoder(BaseClassAwareSetEncoder):
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

    def build_class_embeddings(
        self, fused_support: Tensor, support_y: Tensor
    ) -> Tensor:
        attended_support = self.self_attention(fused_support)
        return self._group_mean(attended_support, support_y)


class QueryAttentionClassAwareSetEncoder(BaseClassAwareSetEncoder):
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
        self.class_queries = nn.Parameter(torch.empty(1, num_classes, hidden_dim))
        nn.init.xavier_normal_(self.class_queries)

    def build_class_embeddings(
        self, fused_support: Tensor, support_y: Tensor
    ) -> Tensor:
        del support_y
        batch_size = fused_support.shape[0]
        queries = self.class_queries.expand(batch_size, -1, -1)
        class_embeddings, _ = self.attention(queries, fused_support, fused_support)
        # (B, N, H)
        return class_embeddings
