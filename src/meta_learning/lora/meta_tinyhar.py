from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from meta_learning.models.tiny_har import TinyHAR

# --- 1. Set Encoder Components (Untouched) ---


class LightweightFeatureExtractor(nn.Module):
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

    def encode(self, x: Tensor) -> Tensor:
        return self.forward(x)


class SetEncoder(nn.Module):
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
        self.class_emb = nn.Embedding(num_classes + 1, self.label_embedding_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(feature_dim + self.label_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.post_pool = nn.Sequential(
            nn.Linear(2 * z_dim, z_dim),
            nn.ReLU(),
            nn.Linear(z_dim, z_dim),
        )

    def forward(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        batch_size, k_shots, C, T = support_x.shape
        x_flat = support_x.view(batch_size * k_shots, C, T)
        y_flat = support_y.view(batch_size * k_shots)

        h_x = self.feature_encoder.encode(x_flat)  # type: ignore[attr-defined]
        h_y = self.class_emb(y_flat)
        h_combined = torch.cat([h_x, h_y], dim=1)

        r_i = self.fusion_mlp(h_combined)
        r_i = r_i.view(batch_size, k_shots, -1)
        z_mean = torch.mean(r_i, dim=1)
        z_std = torch.std(r_i, dim=1, unbiased=False)
        z = self.post_pool(torch.cat([z_mean, z_std], dim=1))

        return z


# --- 2. T2L Hypernetwork for LoRA Generation ---


class T2LHypernetwork(nn.Module):
    """
    Generates LoRA matrices (A and B) for Conv1D layers based on the Subject Embedding (z).
    Follows the 'L' Architecture and Bias-HyperInit from the T2L paper.
    """

    def __init__(self, z_dim: int, layer_configs: list, r: int = 8):
        super().__init__()
        self.r = r
        self.num_layers = len(layer_configs)

        # Learnable embedding for each target layer
        self.layer_emb = nn.Embedding(self.num_layers, 32)

        # Shared MLP backbone
        self.mlp = nn.Sequential(
            nn.Linear(z_dim + 32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # ModuleLists to hold the specific output heads for each layer
        self.heads_A = nn.ModuleList()
        self.heads_B = nn.ModuleList()

        for config in layer_configs:
            in_c, out_c, k = config["in_c"], config["out_c"], config["k"]

            # Head A: Output size = rank * (in_channels * kernel_size)
            head_a = nn.Linear(128, r * in_c * k)
            # Head B: Output size = out_channels * rank
            head_b = nn.Linear(128, out_c * r)

            # --- Bias-HyperInit ---
            # Head A: zero weights, uniform bias bounds based on input dimension
            nn.init.zeros_(head_a.weight)
            bound = 1.0 / (in_c * k)
            nn.init.uniform_(head_a.bias, -bound, bound)

            # Head B: zero weights and zero bias (ensures delta W is 0 at start)
            nn.init.zeros_(head_b.weight)
            nn.init.zeros_(head_b.bias)

            self.heads_A.append(head_a)
            self.heads_B.append(head_b)

    def forward(self, z: Tensor, layer_idx: int):
        # z shape: (Batch, z_dim)
        Batch = z.shape[0]

        # Get layer embedding and concatenate with z
        l_emb = self.layer_emb(torch.tensor(layer_idx, device=z.device))
        l_emb = l_emb.unsqueeze(0).expand(Batch, -1)  # (Batch, 32)

        h = torch.cat([z, l_emb], dim=1)  # (Batch, z_dim + 32)
        h = self.mlp(h)  # (Batch, 128)

        # Generate flattened matrices
        A_flat = self.heads_A[layer_idx](h)
        B_flat = self.heads_B[layer_idx](h)

        return A_flat, B_flat


# --- 3. Dynamic LoRA Conv Block ---


class DynamicLoRAConv1d(nn.Module):
    """
    Standard Conv1d + Dynamically generated LoRA adapter (W = W0 + B*A).
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: int = 2,
    ):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = kernel_size
        self.stride = stride
        self.padding = padding

        # Base frozen/trainable weights
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        num_groups = min(4, out_c)
        while out_c % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_c)

    def forward(
        self,
        x: Tensor,
        A_flat: Tensor,
        B_flat: Tensor,
        num_sensors: int,
        r: int,
        lora_scale: float,
    ):
        """
        x: (Batch * num_sensors, in_c, T)
        A_flat: (Batch, r * in_c * k)
        B_flat: (Batch, out_c * r)
        """
        Batch = A_flat.shape[0]

        # 1. Compute Base Convolution
        out_base = self.conv(x)

        # 2. Reshape generated vectors into LoRA matrices A and B
        # A: (Batch, r, in_c * k)
        # B_mat: (Batch, out_c, r)
        A = A_flat.view(Batch, r, self.in_c * self.k)
        B_mat = B_flat.view(Batch, self.out_c, r)

        # 3. Compute Delta W = B_mat @ A -> (Batch, out_c, in_c * k)
        delta_W = torch.bmm(B_mat, A)
        delta_W = delta_W.view(Batch, self.out_c, self.in_c, self.k)
        delta_W = delta_W * lora_scale

        # 4. Batched Dynamic Convolution Trick using `groups`
        # Reshape input to a single massive batch item: (1, Batch * num_sensors * in_c, T)
        x_flat = x.view(1, Batch * num_sensors * self.in_c, x.shape[-1])

        # Repeat delta_W for every sensor channel: (Batch, num_sensors, out_c, in_c, k)
        delta_W_rep = delta_W.unsqueeze(1).expand(
            Batch, num_sensors, self.out_c, self.in_c, self.k
        )

        # Flatten weight for grouped conv: (Batch * num_sensors * out_c, in_c, k)
        weight_flat = delta_W_rep.reshape(
            Batch * num_sensors * self.out_c, self.in_c, self.k
        )

        # Apply convolution with groups = Batch * num_sensors
        lora_out = F.conv1d(
            x_flat,
            weight=weight_flat,
            stride=self.stride,
            padding=self.padding,
            groups=Batch * num_sensors,
        )

        # Reshape output back to: (Batch * num_sensors, out_c, T_out)
        lora_out = lora_out.view(Batch * num_sensors, self.out_c, lora_out.shape[-1])

        # 5. Add adapter to base, then activate and normalize
        out = out_base + lora_out
        out = self.relu(out)
        out = self.norm(out)

        return out


# --- 4. The MetaTinyHAR Model (LoRA Edition) ---


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
        lora_rank: int = 8,
        lora_alpha: float = 8.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.window_size = window_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # --- A. Set Encoder System ---
        support_feat_extractor = TinyHAR(
            input_channels=input_channels,
            window_size=window_size,
            num_classes=num_classes,
            num_filters=num_filters,
        )

        # # LightweightFeatureExtractor(
        #     input_channels, feature_dim=32
        # )

        self.set_encoder = SetEncoder(
            feature_encoder=support_feat_extractor,
            feature_dim=support_feat_extractor.fusion_dim,
            z_dim=z_dim,
            num_classes=num_classes,
        )

        # --- B. Hypernetwork for LoRA Generation ---
        # Define shapes of the Conv1d layers to adapt
        self.layer_configs = [
            {"in_c": 1, "out_c": num_filters, "k": 5},  # Layer 0
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 1
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 2
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 3
        ]

        self.hypernetwork = T2LHypernetwork(z_dim, self.layer_configs, r=lora_rank)

        # --- C. Dynamic Convolutional Subnet ---
        self.conv_layers = nn.ModuleList(
            [DynamicLoRAConv1d(cfg["in_c"], cfg["out_c"]) for cfg in self.layer_configs]
        )

        # --- D. Standard TinyHAR Backend ---
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

        self.fusion_dim = 2 * num_filters
        self.fusion_layer = nn.Linear(input_channels * num_filters, self.fusion_dim)

        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attention_layer = nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Tanh())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(
        self,
        x: Tensor,
        support_x: Optional[Tensor] = None,
        support_y: Optional[Tensor] = None,
        precomputed_z: Optional[Tensor] = None,
    ) -> Tensor:

        # 1. Extract Subject Embedding
        if precomputed_z is not None:
            z = precomputed_z
        elif support_x is not None and support_y is not None:
            z = self.set_encoder(support_x, support_y)
        else:
            raise ValueError("Either support_x/y or precomputed_z is required.")

        # Ensure shape is (B, T, C) and swap to (B, C, T) internally
        if x.shape[1] == self.input_channels and x.shape[2] == self.window_size:
            x = x.permute(0, 2, 1)

        B, T, C = x.shape

        # 2. Reshape for Independent Sensor Processing
        # (B, T, C) -> (B, C, T) -> (B * C, 1, T)
        curr_feat = x.permute(0, 2, 1).reshape(B * C, 1, T)

        # 3. Dynamic Forward Pass through Conv Layers
        for i, conv_block in enumerate(self.conv_layers):
            # Generate A and B vectors specifically for this layer
            A_flat, B_flat = self.hypernetwork(z, layer_idx=i)

            # Apply base conv + generated LoRA adapter
            curr_feat = conv_block(
                curr_feat,
                A_flat,
                B_flat,
                num_sensors=C,
                r=self.lora_rank,
                lora_scale=self.lora_alpha / self.lora_rank,
            )

        # 4. Standard TinyHAR Transformer & LSTM Backend
        features_conv = curr_feat
        _, F_dim, T_star = features_conv.shape

        features_trans_in = features_conv.reshape(B, C, F_dim, T_star).permute(
            0, 3, 1, 2
        )
        features_trans_in_flat = features_trans_in.reshape(B * T_star, C, F_dim)
        features_trans_out = self.transformer_encoder(features_trans_in_flat)

        features_fused_in = features_trans_out.reshape(B * T_star, C * F_dim)
        features_fused = self.fusion_layer(features_fused_in)
        features_seq = features_fused.reshape(B, T_star, self.fusion_dim)

        lstm_out, _ = self.lstm(features_seq)

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

    model = MetaTinyHAR(
        input_channels=CHANNELS,
        window_size=WINDOW_SIZE,
        num_classes=CLASSES,
        z_dim=32,
        lora_rank=8,  # Using rank 8 as in the T2L paper
    )

    query_x = torch.randn(BATCH_SIZE, WINDOW_SIZE, CHANNELS)
    support_x = torch.randn(BATCH_SIZE, K_SHOTS, CHANNELS, WINDOW_SIZE)
    support_y = torch.randint(0, CLASSES, (BATCH_SIZE, K_SHOTS))

    logits = model(query_x, support_x, support_y)
    print("Meta-TinyHAR (LoRA version) output shape:", logits.shape)
