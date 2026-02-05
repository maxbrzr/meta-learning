import torch
import torch.nn as nn


class AdaptiveGroupNorm1d(nn.Module):
    """
    Adaptive Group Normalization for 1D time-series data.

    It replaces standard static affine parameters with dynamic ones predicted
    from a condition vector 'z' (Subject Embedding).

    Formula: y = (1 + gamma(z)) * norm(x) + beta(z)
    """

    def __init__(self, num_features: int, z_dim: int, num_groups: int = 8):
        super().__init__()

        self.num_features = num_features
        self.group_norm = nn.GroupNorm(num_groups, num_features, affine=False)

        # Projection layer to map z -> (gamma, beta) for each feature channel
        # Output dim is 2 * num_features (one gamma and one beta per channel)
        self.projection = nn.Linear(z_dim, 2 * num_features)

        # Initialization trick:
        # Initialize weights/bias to 0 so the layer starts as a standard GroupNorm
        # (gamma=0 -> multiplier=1, beta=0 -> shift=0)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (Batch, Time, Channels)
            z: Conditioning vector of shape (Batch, z_dim)
        """
        # 1. Standard Group Norm (without affine)
        out = self.group_norm(x)

        # 2. Predict dynamic parameters
        # style shape: (Batch, 2 * Channels)
        style = self.projection(z)

        # Reshape to (Batch, 2, Channels, 1) to broadcast over Time dimension
        style = style.view(-1, 2, self.num_features, 1)

        gamma = style[:, 0, :, :]  # Scale
        beta = style[:, 1, :, :]  # Shift

        # 3. Apply Modulation
        # We use (1 + gamma) to ensure gradient flow at initialization
        out = (1 + gamma) * out + beta

        return out
