import torch
import torch.nn as nn


class SetEncoder(nn.Module):
    """
    Permutation-invariant encoder that maps a support set of (samples, labels)
    to a single style vector 'z'.

    Uses 'Late Fusion': Encodes x and y separately, then concatenates.
    """

    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        z_dim: int,
        feature_dim: int,
        hidden_dim: int = 64,
        class_embed_dim: int = 32,
    ):
        super().__init__()

        # Dependency Injection: The black box encoder for time-series samples
        # Expected input: (Batch, Channels, Time) -> Output: (Batch, feature_dim)
        self.feature_encoder = encoder

        # Label Embedding (includes +1 for null/unconditioned class)
        # We match the embedding size to feature_dim or hidden_dim usually
        self.class_emb = nn.Embedding(num_classes + 1, class_embed_dim)

        # The Fusion MLP (Post-Concatenation)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + class_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        # null_condition: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            support_x: (Batch, K_Shots, Time, Channels)
            support_y: (Batch, K_Shots) - Integer labels
            null_condition: If True, forces the null embedding (simulating no context)

        Returns:
            z: (Batch, z_dim)
        """
        batch_size, k_shots, T, C = support_x.shape

        # 1. Flatten Batch and Shots to process in parallel
        # shape: (Batch * K, T, C)
        x_flat = support_x.view(batch_size * k_shots, T, C)
        y_flat = support_y.view(batch_size * k_shots)

        # 2. Extract Signal Features (Black Box)
        # shape: (Batch * K, feature_dim)
        h_x = self.feature_encoder(x_flat)

        # # 3. Extract Label Embeddings
        # if null_condition:
        #     # Create a tensor of the "Null" class index (usually largest index)
        #     null_idx = self.class_emb.num_embeddings - 1
        #     y_flat = torch.full_like(y_flat, null_idx)

        # shape: (Batch * K, label_emb_dim)
        h_y = self.class_emb(y_flat)

        # 4. Late Fusion (Concatenate)
        # shape: (Batch * K, feature_dim + label_emb_dim)
        h_combined = torch.cat([h_x, h_y], dim=1)

        # 5. Project to Latent Space
        # shape: (Batch * K, z_dim)
        r_i = self.fusion_mlp(h_combined)

        # 6. Aggregate (Set Operation)
        # Reshape back to (Batch, K, z_dim)
        r_i = r_i.view(batch_size, k_shots, -1)

        # Mean Pooling over the 'shots' dimension
        z = torch.mean(r_i, dim=1)  # (Batch, z_dim)

        return z
