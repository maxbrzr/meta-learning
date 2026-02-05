from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- 1. Adaptive Group Norm (The Modulation Layer) ---
class AdaptiveGroupNorm1d(nn.Module):
    """
    Adaptive Group Normalization.
    Replaces static affine parameters with dynamic ones predicted from 'z'.
    """

    def __init__(self, num_features: int, z_dim: int, num_groups: int = 4):
        super().__init__()
        self.num_features = num_features
        # Affine=False because we provide gamma/beta manually
        self.group_norm = nn.GroupNorm(num_groups, num_features, affine=False)

        # Projection: z -> (gamma, beta)
        self.projection = nn.Linear(z_dim, 2 * num_features)

        # Init zero to start as standard identity identity mapping
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Channels, Time)
        z: (Batch, z_dim)
        """
        out = self.group_norm(x)

        # Predict dynamic parameters
        style = self.projection(z)  # (Batch, 2*C)
        style = style.view(-1, 2, self.num_features, 1)  # (Batch, 2, C, 1)

        gamma = style[:, 0, :, :]
        beta = style[:, 1, :, :]

        return (1 + gamma) * out + beta


# --- 2. Adaptive Convolutional Block ---
class AdaptiveConvBlock(nn.Module):
    """
    Wraps Conv1d -> ReLU -> AdaptiveGroupNorm into a single module
    that accepts 'z'.
    """

    def __init__(self, in_c: int, out_c: int, z_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()
        self.norm = AdaptiveGroupNorm1d(out_c, z_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.relu(x)
        x = self.norm(x, z)
        return x


# --- 3. Set Encoder Components ---


class LightweightFeatureExtractor(nn.Module):
    """
    A small, separate CNN just for the Set Encoder.
    It doesn't need to be as deep as TinyHAR.
    """

    def __init__(self, in_channels: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(16, feature_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SetEncoder(nn.Module):
    """
    Maps support set (Samples + Labels) -> Style Vector z.
    Uses Late Fusion.
    """

    def __init__(
        self,
        feature_encoder: nn.Module,
        feature_dim: int,
        z_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.label_embedding_dim = 32

        # +1 for the null/unconditioned token
        self.class_emb = nn.Embedding(num_classes + 1, self.label_embedding_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(
        self,
        support_x: Tensor,
        support_y: Tensor,
        #  null_condition: bool = False
    ) -> Tensor:
        # support_x: (Batch, K, C, T)
        # support_y: (Batch, K)
        batch_size, k_shots, C, T = support_x.shape

        x_flat = support_x.view(batch_size * k_shots, C, T)
        y_flat = support_y.view(batch_size * k_shots)

        # 1. Encode Signal
        h_x = self.feature_encoder(x_flat)

        # # 2. Encode Label
        # if null_condition:
        #     null_idx = self.class_emb.num_embeddings - 1
        #     y_flat = torch.full_like(y_flat, null_idx)

        h_y = self.class_emb(y_flat)

        # 3. Fuse
        h_combined = torch.cat([h_x, h_y], dim=1)

        # 4. Project & Aggregate
        r_i = self.fusion_mlp(h_combined)
        r_i = r_i.view(batch_size, k_shots, -1)
        z = torch.mean(r_i, dim=1)

        return z


# --- 4. Meta-TinyHAR (The Main Model) ---


class MetaTinyHAR(nn.Module):
    def __init__(
        self,
        input_channels: int,
        window_size: int,
        num_classes: int,
        num_filters: int = 28,
        cross_channel_interaction_heads: int = 4,
        dropout: float = 0.2,
        z_dim: int = 64,
    ):
        super(MetaTinyHAR, self).__init__()

        self.input_channels = input_channels
        self.window_size = window_size
        self.num_filters = num_filters
        self.z_dim = z_dim
        self.num_classes = num_classes

        # --- A. The Set Encoder System ---
        # Helper encoder for the support set
        support_feat_extractor = LightweightFeatureExtractor(
            input_channels, feature_dim=32
        )

        self.set_encoder = SetEncoder(
            feature_encoder=support_feat_extractor,
            feature_dim=32,
            z_dim=z_dim,
            num_classes=num_classes,
        )

        # --- B. Adaptive Convolutional Subnet ---
        # Replaced nn.Sequential with ModuleList to handle 'z' passing manually
        self.conv_layers = nn.ModuleList(
            [
                AdaptiveConvBlock(1, num_filters, z_dim),  # Layer 1
                AdaptiveConvBlock(num_filters, num_filters, z_dim),  # Layer 2
                AdaptiveConvBlock(num_filters, num_filters, z_dim),  # Layer 3
                AdaptiveConvBlock(num_filters, num_filters, z_dim),  # Layer 4
            ]
        )

        # Calculate temporal dim reduction
        self._flattened_temporal_dim = self._get_conv_output_dim(window_size)

        # --- C. Transformer Encoder (Standard) ---
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_filters,
                nhead=cross_channel_interaction_heads,
                dim_feedforward=num_filters * 2,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # --- D. Cross-Channel Fusion (Standard) ---
        self.fusion_dim = 2 * num_filters
        self.fusion_layer = nn.Linear(input_channels * num_filters, self.fusion_dim)

        # --- E. LSTM & Attention (Standard) ---
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attention_layer = nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Tanh())
        self.gamma = nn.Parameter(torch.zeros(1))

        # Prediction
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def _get_conv_output_dim(self, input_size: int) -> int:
        size = input_size
        for _ in range(4):
            size = (size + 4 - 5) // 2 + 1
        return size

    def forward(
        self,
        x: Tensor,
        support_x: Optional[Tensor] = None,
        support_y: Optional[Tensor] = None,
        precomputed_z: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: Query input (Batch, Time, Channels)
            support_x: Calibration samples (Batch, K, Channels, Time)
            support_y: Calibration labels (Batch, K)
            precomputed_z: If you already have z, skip the set encoder.
        """

        # 1. Compute Subject Embedding (z)
        if precomputed_z is not None:
            z = precomputed_z
        elif support_x is not None and support_y is not None:
            z = self.set_encoder(support_x, support_y)
        else:
            raise ValueError(
                "Either support_x/support_y or precomputed_z must be provided."
            )
            # # Fallback: Run Set Encoder in "Null" mode (Unconditioned)
            # # Create dummy support input to trigger null path
            # B = x.shape[0]
            # dummy_x = torch.zeros(
            #     B, 1, self.input_channels, self.window_size, device=x.device
            # )
            # dummy_y = torch.zeros(B, 1, dtype=torch.long, device=x.device)
            # z = self.set_encoder(dummy_x, dummy_y)

        # Standardize input shape: (B, T, C)
        if x.shape[1] == self.input_channels and x.shape[2] == self.window_size:
            x = x.permute(0, 2, 1)

        B, T, C = x.shape

        # --- 2. Adaptive Convolutional Subnet ---
        # Reshape: Treat sensors as independent samples (B*C, 1, T)
        x_reshaped = x.permute(0, 2, 1).reshape(B * C, 1, T)

        # EXPAND Z: The conv net sees B*C items, but we only have B z-vectors.
        # We must repeat z for each channel of the subject.
        # z: (B, z_dim) -> (B, C, z_dim) -> (B*C, z_dim)
        z_expanded = z.unsqueeze(1).repeat(1, C, 1).view(B * C, -1)

        curr_feat = x_reshaped
        for conv_block in self.conv_layers:
            # Pass z into every block
            curr_feat = conv_block(curr_feat, z_expanded)

        # Output: (B*C, F, T*)
        features_conv = curr_feat
        _, F_dim, T_star = features_conv.shape

        # --- 3. Transformer (Standard TinyHAR Logic) ---
        # (B*C, F, T*) -> (B, T*, C, F)
        features_trans_in = features_conv.reshape(B, C, F_dim, T_star).permute(
            0, 3, 1, 2
        )
        features_trans_in_flat = features_trans_in.reshape(B * T_star, C, F_dim)

        features_trans_out = self.transformer_encoder(features_trans_in_flat)

        # --- 4. Fusion & LSTM (Standard TinyHAR Logic) ---
        features_fused_in = features_trans_out.reshape(B * T_star, C * F_dim)
        features_fused = self.fusion_layer(features_fused_in)
        features_seq = features_fused.reshape(B, T_star, self.fusion_dim)

        lstm_out, _ = self.lstm(features_seq)

        # --- 5. Attention & Prediction ---
        att_weights = F.softmax(self.attention_layer(lstm_out), dim=1)
        global_context = torch.sum(lstm_out * att_weights, dim=1)
        last_step_feature = lstm_out[:, -1, :]

        final_feature = last_step_feature + (self.gamma * global_context)

        logits = self.classifier(final_feature)
        return logits


# --- Example Usage ---
if __name__ == "__main__":
    BATCH_SIZE = 4
    K_SHOTS = 5
    CHANNELS = 6
    WINDOW_SIZE = 64
    CLASSES = 8

    # Instantiate
    model = MetaTinyHAR(
        input_channels=CHANNELS, window_size=WINDOW_SIZE, num_classes=CLASSES, z_dim=32
    )

    # 1. Query Data (New samples to classify)
    query_x = torch.randn(BATCH_SIZE, WINDOW_SIZE, CHANNELS)

    # 2. Support Set (Calibration Data: 5 samples per subject)
    # Shape: (Batch, K, Channels, Time)
    support_x = torch.randn(BATCH_SIZE, K_SHOTS, CHANNELS, WINDOW_SIZE)
    support_y = torch.randint(0, CLASSES, (BATCH_SIZE, K_SHOTS))

    # Forward Pass
    logits = model(query_x, support_x, support_y)

    print("Meta-TinyHAR output shape:", logits.shape)  # Expected: (4, 8)

    # Test Null Condition (No support set provided)
    logits_null = model(query_x)
    print("Null-Condition output shape:", logits_null.shape)
