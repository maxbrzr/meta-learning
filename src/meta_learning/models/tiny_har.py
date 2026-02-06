import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TinyHAR(nn.Module):
    """
    TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition.

    Implements the architecture described in:
    Zhou, et al. "TinyHAR: A Lightweight Deep Learning Model Designed for Human Activity Recognition"
    ISWC '22. [cite: 8, 47]

    Attributes:
        input_channels (int): Number of sensor channels (C).
        window_size (int): Length of the temporal sliding window (T).
        num_classes (int): Number of activity classes to predict.
        num_filters (int): Number of filters in the conv layers (F). Default: 28 (from paper [cite: 348]).
        lstm_units (int): Hidden size for the LSTM layer. Usually 2 * F in the paper's logic.
    """

    def __init__(
        self,
        input_channels: int,
        window_size: int,
        num_classes: int,
        num_filters: int = 28,
        cross_channel_interaction_heads: int = 4,
        dropout: float = 0.2,
    ):
        super(TinyHAR, self).__init__()

        self.input_channels = input_channels
        self.window_size = window_size
        self.num_filters = num_filters

        # --- 1. Individual Convolutional Subnet  ---
        # "All four convolutional layers have the same number of filters F."
        # Stride is 2 for all layers to reduce temporal dimension.
        # Kernel size is 5x1 (1D along temporal).

        self.conv_subnet = nn.Sequential(
            self._make_conv_block(1, num_filters),  # Layer 1
            self._make_conv_block(num_filters, num_filters),  # Layer 2
            self._make_conv_block(num_filters, num_filters),  # Layer 3
            self._make_conv_block(num_filters, num_filters),  # Layer 4
        )

        # Calculate the reduced temporal dimension T* after 4 layers of stride 2
        # Assuming padding=2 ('same' specific behavior) or sufficient padding to avoid losing too much signal.
        # The paper doesn't specify exact padding, typically Conv1d with k=5, s=2 requires padding to maintain flow.
        # We calculate the output size dynamically or assume standard reduction approx T / 16.
        self._flattened_temporal_dim = self._get_conv_output_dim(window_size)

        # --- 2. Transformer Encoder: Cross-Channel Info Interaction [cite: 128, 130] ---
        # Interaction performed across the sensor channel dimension (C).
        # Input to transformer: Sequence length = C, Embedding dim = F.
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=num_filters,
                nhead=cross_channel_interaction_heads,
                dim_feedforward=num_filters * 2,  # Common default, scalable
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=1,
        )

        # --- 3. Cross-Channel Info Fusion [cite: 167, 170] ---
        # Flattens C and F, projects to F* = 2F.
        # Input dim: C * F
        # Output dim: F* (2 * F)
        self.fusion_dim = 2 * num_filters
        self.fusion_layer = nn.Linear(input_channels * num_filters, self.fusion_dim)

        # --- 4. Global Temporal Info Extraction (LSTM)  ---
        # One LSTM layer.
        self.lstm = nn.LSTM(
            input_size=self.fusion_dim,
            hidden_size=self.fusion_dim,
            num_layers=1,
            batch_first=True,
        )

        # --- 5. Temporal Attention: Global Info Enhancement [cite: 173, 177] ---
        # Attention weights calculation
        self.attention_layer = nn.Sequential(nn.Linear(self.fusion_dim, 1), nn.Tanh())

        # Trainable multiplier parameter gamma [cite: 177]
        self.gamma = nn.Parameter(torch.zeros(1))

        # Prediction
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def _make_conv_block(self, in_c: int, out_c: int) -> nn.Sequential:
        """Helper to create Conv -> ReLU -> BN block."""
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(out_c),
        )

    def _get_conv_output_dim(self, input_size: int) -> int:
        """Calculates temporal dimension after 4 conv layers with stride 2."""
        size = input_size
        for _ in range(4):
            # Formula: floor((L_in + 2*padding - dilation*(kernel-1) - 1)/stride + 1)
            # k=5, p=2, s=2, d=1 -> floor((L + 4 - 4 - 1)/2 + 1) = floor((L-1)/2 + 1)
            size = (size + 4 - 5) // 2 + 1
        return size

    def encode(self, x: Tensor) -> Tensor:
        """
        Encoder function to extract features before classification.
        Useful for feature extraction tasks.

        Args:
            x (Tensor): Input data of shape (Batch, Time, Channels)

        Returns:
            Tensor: Encoded features of shape (Batch, Feature_Dim)
        """
        # Ensure input is (Batch, Time, Channels)
        if x.shape[1] == self.input_channels and x.shape[2] == self.window_size:
            x = x.permute(0, 2, 1)  # Swap to (B, T, C)

        B, T, C = x.shape

        # --- 1. Individual Convolutional Subnet ---
        # Reshape to treat every sensor channel as an independent sample for the ConvNet
        # (B, T, C) -> (B * C, 1, T)
        x_reshaped = x.permute(0, 2, 1).reshape(B * C, 1, T)

        # Apply ConvNet (Shared weights across all channels)
        # Output: (B*C, F, T*)
        features_conv = self.conv_subnet(x_reshaped)

        _, F_dim, T_star = features_conv.shape

        # --- 2. Transformer: Cross-Channel Interaction ---
        # Reshape to (B, T*, C, F) to apply transformer over C
        # The transformer needs a sequence of length C.
        # We treat (B * T*) as the batch size for the transformer.

        # (B*C, F, T*) -> (B, C, F, T*) -> (B, T*, C, F)
        features_trans_in = features_conv.reshape(B, C, F_dim, T_star).permute(
            0, 3, 1, 2
        )

        # Flatten Batch and Time for Transformer processing
        # (B*T*, C, F)
        features_trans_in_flat = features_trans_in.reshape(B * T_star, C, F_dim)

        # Apply Transformer Encoder
        features_trans_out = self.transformer_encoder(features_trans_in_flat)

        # --- 3. Cross-Channel Fusion ---
        # Flatten C and F dimensions: (B*T*, C*F)
        features_fused_in = features_trans_out.reshape(B * T_star, C * F_dim)

        # Apply bottleneck FC layer: (B*T*, C*F) -> (B*T*, F*)
        # F* = 2 * F_dim
        features_fused = self.fusion_layer(features_fused_in)

        # Reshape back to temporal sequence for LSTM: (B, T*, F*)
        features_seq = features_fused.reshape(B, T_star, self.fusion_dim)

        # --- 4. Global Temporal Info Extraction (LSTM) ---
        # Input: (B, T*, F*)
        # Output: (B, T*, F*) (Hidden states at all steps)
        lstm_out, _ = self.lstm(features_seq)

        # --- 5. Temporal Attention & Prediction ---
        # Calculate attention scores: (B, T*, 1)
        att_weights = F.softmax(self.attention_layer(lstm_out), dim=1)

        # Calculate global context c by weighted sum: (B, F*)
        # Sum over T* dimension
        global_context = torch.sum(lstm_out * att_weights, dim=1)

        # Get feature at last time step x_T*: (B, F*)
        last_step_feature = lstm_out[:, -1, :]

        # Combine: x_last + gamma * c [cite: 176, 177]
        final_feature = last_step_feature + (self.gamma * global_context)

        return final_feature

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): Input data of shape (Batch, Time, Channels)
                              or (Batch, Channels, Time).
                              Paper denotes input as T x C x F (where F=1).
                              Here we adhere to (Batch, Time, Channels).

        Returns:
            Tensor: Class logits of shape (Batch, Num_Classes).
        """

        final_feature = self.encode(x)

        # Final classification
        logits: Tensor = self.classifier(final_feature)

        return logits


# --- Example Usage ---
if __name__ == "__main__":
    # Example hyperparameters based on paper context (e.g., PAMAP2 dataset [cite: 165])
    # T=100 (approx 3s at 33Hz), C=18 sensors, Classes=12
    BATCH_SIZE = 32
    WINDOW_SIZE = 100
    CHANNELS = 18
    CLASSES = 12

    model = TinyHAR(
        input_channels=CHANNELS,
        window_size=WINDOW_SIZE,
        num_classes=CLASSES,
        num_filters=28,  # [cite: 348]
    )

    # Random input tensor (Batch, Time, Sensors)
    input_data = torch.randn(BATCH_SIZE, WINDOW_SIZE, CHANNELS)

    output = model(input_data)

    print(f"Model Architecture:\n{model}")
    print(f"\nInput shape: {input_data.shape}")
    print(f"Output shape: {output.shape}")  # Expected: (32, 12)

    # Verify parameter count to ensure "Tiny" nature
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Trainable Parameters: {total_params}")
