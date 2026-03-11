from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from meta_learning.lora.hypernet import ClassAwareT2LHypernetwork, T2LHypernetwork
from meta_learning.lora.set_encoders import (
    BaseSetEncoder,
    MeanSetEncoder,
    QueryAttentionSetEncoder,
    SelfAttentionMeanSetEncoder,
)
from meta_learning.lora.set_encoders_class_aware import (
    BaseClassAwareSetEncoder,
    MeanClassAwareSetEncoder,
    QueryAttentionClassAwareSetEncoder,
    SelfAttentionMeanClassAwareSetEncoder,
)
from meta_learning.models.tiny_har import TinyHAR


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
        # (B, out_c, r)

        # 3. Compute Delta W = B_mat @ A -> (Batch, out_c, in_c * k)
        delta_W = torch.bmm(B_mat, A)
        delta_W = delta_W.view(Batch, self.out_c, self.in_c, self.k)
        delta_W = delta_W * lora_scale
        # (B, out_c, in_c, k)

        # 4. Batched Dynamic Convolution Trick using `groups`
        # Reshape input to a single massive batch item: (1, Batch * num_sensors * in_c, T)
        x_flat = x.view(1, Batch * num_sensors * self.in_c, x.shape[-1])
        # (1, B * S * in_c, T)

        # Repeat delta_W for every sensor channel: (Batch, num_sensors, out_c, in_c, k)
        delta_W_rep = delta_W.unsqueeze(1).expand(
            Batch, num_sensors, self.out_c, self.in_c, self.k
        )
        # (B, S, out_c, in_c, k)

        # Flatten weight for grouped conv: (Batch * num_sensors * out_c, in_c, k)
        weight_flat = delta_W_rep.reshape(
            Batch * num_sensors * self.out_c, self.in_c, self.k
        )
        # (B * S * out_c, in_c, k)

        # Apply convolution with groups = Batch * num_sensors
        lora_out = F.conv1d(
            x_flat,
            weight=weight_flat,
            stride=self.stride,
            padding=self.padding,
            groups=Batch * num_sensors,
        )
        lora_out = lora_out.view(Batch * num_sensors, self.out_c, lora_out.shape[-1])
        # (B * S, out_c, T_out)

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
        set_encoder_variant: str = "mean",
        set_encoder_num_heads: int = 4,
        class_aware: bool = False,
        hypernetwork_variant: str = "task",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.window_size = window_size
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.class_aware = class_aware
        self.hypernetwork_variant = hypernetwork_variant
        self.last_class_embeddings: Optional[Tensor] = None

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

        if class_aware:
            if set_encoder_variant == "mean":
                self.set_encoder: BaseSetEncoder | BaseClassAwareSetEncoder = (
                    MeanClassAwareSetEncoder(
                        feature_encoder=support_feat_extractor,
                        feature_dim=support_feat_extractor.fusion_dim,
                        z_dim=z_dim,
                        num_classes=num_classes,
                    )
                )
            elif set_encoder_variant == "self_attention_mean":
                self.set_encoder = SelfAttentionMeanClassAwareSetEncoder(
                    feature_encoder=support_feat_extractor,
                    feature_dim=support_feat_extractor.fusion_dim,
                    z_dim=z_dim,
                    num_classes=num_classes,
                    num_heads=set_encoder_num_heads,
                )
            elif set_encoder_variant == "query_attention":
                self.set_encoder = QueryAttentionClassAwareSetEncoder(
                    feature_encoder=support_feat_extractor,
                    feature_dim=support_feat_extractor.fusion_dim,
                    z_dim=z_dim,
                    num_classes=num_classes,
                    num_heads=set_encoder_num_heads,
                )
            else:
                raise ValueError(
                    "set_encoder_variant must be one of: "
                    "'mean', 'self_attention_mean', 'query_attention'. "
                    f"Got: {set_encoder_variant}"
                )
        else:
            if set_encoder_variant == "mean":
                self.set_encoder = MeanSetEncoder(
                    feature_encoder=support_feat_extractor,
                    feature_dim=support_feat_extractor.fusion_dim,
                    z_dim=z_dim,
                    num_classes=num_classes,
                )
            elif set_encoder_variant == "self_attention_mean":
                self.set_encoder = SelfAttentionMeanSetEncoder(
                    feature_encoder=support_feat_extractor,
                    feature_dim=support_feat_extractor.fusion_dim,
                    z_dim=z_dim,
                    num_classes=num_classes,
                    num_heads=set_encoder_num_heads,
                )
            elif set_encoder_variant == "query_attention":
                self.set_encoder = QueryAttentionSetEncoder(
                    feature_encoder=support_feat_extractor,
                    feature_dim=support_feat_extractor.fusion_dim,
                    z_dim=z_dim,
                    num_classes=num_classes,
                    num_heads=set_encoder_num_heads,
                )
            else:
                raise ValueError(
                    "set_encoder_variant must be one of: "
                    "'mean', 'self_attention_mean', 'query_attention'. "
                    f"Got: {set_encoder_variant}"
                )

        # --- B. Hypernetwork for LoRA Generation ---
        # Define shapes of the Conv1d layers to adapt
        self.layer_configs = [
            {"in_c": 1, "out_c": num_filters, "k": 5},  # Layer 0
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 1
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 2
            {"in_c": num_filters, "out_c": num_filters, "k": 5},  # Layer 3
        ]

        if hypernetwork_variant == "task":
            self.hypernetwork = T2LHypernetwork(z_dim, self.layer_configs, r=lora_rank)
        elif hypernetwork_variant == "class_aware":
            if not class_aware:
                raise ValueError(
                    "hypernetwork_variant='class_aware' requires class_aware=True."
                )
            assert isinstance(self.set_encoder, BaseClassAwareSetEncoder)
            self.hypernetwork = ClassAwareT2LHypernetwork(
                class_embedding_dim=self.set_encoder.hidden_dim,
                layer_configs=self.layer_configs,
                r=lora_rank,
            )
        else:
            raise ValueError(
                "hypernetwork_variant must be one of: 'task', 'class_aware'. "
                f"Got: {hypernetwork_variant}"
            )

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

    def load_pretrained_tinyhar(
        self, tinyhar_state_dict: dict[str, Tensor]
    ) -> dict[str, int]:
        """
        Load compatible weights from a vanilla TinyHAR checkpoint.
        Returns counts of loaded and skipped tensors.
        """
        current_state = self.state_dict()
        loaded = 0
        skipped = 0

        # 1) TinyHAR support encoder inside set encoder.
        for key, value in tinyhar_state_dict.items():
            target_key = f"set_encoder.feature_encoder.{key}"
            if target_key in current_state and current_state[target_key].shape == value.shape:
                current_state[target_key].copy_(value)
                loaded += 1
            else:
                skipped += 1

        # 2) Shared backend blocks (identical naming between TinyHAR and MetaTinyHAR).
        backend_prefixes = (
            "transformer_encoder.",
            "fusion_layer.",
            "lstm.",
            "attention_layer.",
            "classifier.",
        )
        for key, value in tinyhar_state_dict.items():
            if key == "gamma" or key.startswith(backend_prefixes):
                if key in current_state and current_state[key].shape == value.shape:
                    current_state[key].copy_(value)
                    loaded += 1
                else:
                    skipped += 1

        # 3) Conv kernels: TinyHAR conv_subnet -> MetaTinyHAR conv_layers.*.conv
        for layer_idx in range(len(self.conv_layers)):
            src_w = f"conv_subnet.{layer_idx}.0.weight"
            src_b = f"conv_subnet.{layer_idx}.0.bias"
            dst_w = f"conv_layers.{layer_idx}.conv.weight"
            dst_b = f"conv_layers.{layer_idx}.conv.bias"

            if src_w in tinyhar_state_dict and dst_w in current_state:
                if current_state[dst_w].shape == tinyhar_state_dict[src_w].shape:
                    current_state[dst_w].copy_(tinyhar_state_dict[src_w])
                    loaded += 1
                else:
                    skipped += 1
            if src_b in tinyhar_state_dict and dst_b in current_state:
                if current_state[dst_b].shape == tinyhar_state_dict[src_b].shape:
                    current_state[dst_b].copy_(tinyhar_state_dict[src_b])
                    loaded += 1
                else:
                    skipped += 1

        return {"loaded": loaded, "skipped": skipped}

    def freeze_for_meta_learning(self) -> None:
        """
        Freeze all parameters except meta-learner components:
        set encoder (including support feature encoder) and hypernetwork.
        """
        for param in self.parameters():
            param.requires_grad = False

        for param in self.set_encoder.parameters():
            param.requires_grad = True

        for param in self.hypernetwork.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: Tensor,
        support_x: Optional[Tensor] = None,
        support_y: Optional[Tensor] = None,
        precomputed_z: Optional[Tensor] = None,
        precomputed_class_embeddings: Optional[Tensor] = None,
    ) -> Tensor:

        # 1. Extract Subject Embedding
        if precomputed_class_embeddings is not None:
            class_embeddings = precomputed_class_embeddings
            # (B, N, H)
            self.last_class_embeddings = class_embeddings
            if self.class_aware:
                assert isinstance(self.set_encoder, BaseClassAwareSetEncoder)
                z = self.set_encoder.to_task_embedding(class_embeddings)
                # (B, Z)
            elif precomputed_z is not None:
                z = precomputed_z
            else:
                raise ValueError(
                    "precomputed_class_embeddings with class_aware=False "
                    "also requires precomputed_z."
                )
        elif precomputed_z is not None:
            z = precomputed_z
            class_embeddings = None
            self.last_class_embeddings = None
        elif support_x is not None and support_y is not None:
            if self.class_aware:
                class_embeddings = self.set_encoder(support_x, support_y)
                # (B, N, H)
                assert isinstance(self.set_encoder, BaseClassAwareSetEncoder)
                self.last_class_embeddings = class_embeddings
                z = self.set_encoder.to_task_embedding(class_embeddings)
                # (B, Z)
            else:
                z = self.set_encoder(support_x, support_y)
                # (B, Z)
                class_embeddings = None
                self.last_class_embeddings = None
        else:
            raise ValueError("Either support_x/y or precomputed_z is required.")

        # Ensure shape is (B, T, C) and swap to (B, C, T) internally
        if x.shape[1] == self.input_channels and x.shape[2] == self.window_size:
            x = x.permute(0, 2, 1)
            # (B, T, C)

        B, T, C = x.shape

        # 2. Reshape for Independent Sensor Processing
        # (B, T, C) -> (B, C, T) -> (B * C, 1, T)
        curr_feat = x.permute(0, 2, 1).reshape(B * C, 1, T)
        # (B * C, 1, T)

        # 3. Dynamic Forward Pass through Conv Layers
        for i, conv_block in enumerate(self.conv_layers):
            # Generate A and B vectors specifically for this layer
            if self.hypernetwork_variant == "class_aware":
                if class_embeddings is None:
                    raise ValueError(
                        "Class-aware hypernetwork requires class embeddings. "
                        "Provide support_x/support_y or precomputed_class_embeddings."
                    )
                A_flat, B_flat = self.hypernetwork(class_embeddings, layer_idx=i)
            else:
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
        # (B * T*, C, F)

        features_fused_in = features_trans_out.reshape(B * T_star, C * F_dim)
        # (B * T*, C * F)
        features_fused = self.fusion_layer(features_fused_in)
        # (B * T*, D_fuse)
        features_seq = features_fused.reshape(B, T_star, self.fusion_dim)
        # (B, T*, D_fuse)

        lstm_out, _ = self.lstm(features_seq)

        att_weights = F.softmax(self.attention_layer(lstm_out), dim=1)
        global_context = torch.sum(lstm_out * att_weights, dim=1)
        last_step_feature = lstm_out[:, -1, :]
        # (B, D_fuse)
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
