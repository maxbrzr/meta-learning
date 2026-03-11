import torch
import torch.nn as nn
from torch import Tensor


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
        l_emb = l_emb.unsqueeze(0).expand(Batch, -1)
        h = torch.cat([z, l_emb], dim=1)
        h = self.mlp(h)
        # (B, D_hnet)

        # Generate flattened matrices
        A_flat = self.heads_A[layer_idx](h)
        B_flat = self.heads_B[layer_idx](h)

        return A_flat, B_flat


class ClassAwareT2LHypernetwork(nn.Module):
    """
    Class-aware hypernetwork.
    Consumes class embeddings (B, num_classes, hidden_dim) and produces LoRA
    parameters per target layer.
    """

    def __init__(
        self,
        class_embedding_dim: int,
        layer_configs: list,
        r: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()
        self.r = r
        self.num_layers = len(layer_configs)

        # Layer-specific query for attending over class embeddings.
        self.layer_query = nn.Embedding(self.num_layers, class_embedding_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=class_embedding_dim, num_heads=num_heads, batch_first=True
        )

        # Shared backbone for head generation.
        self.mlp = nn.Sequential(
            nn.Linear(class_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.heads_A = nn.ModuleList()
        self.heads_B = nn.ModuleList()

        for config in layer_configs:
            in_c, out_c, k = config["in_c"], config["out_c"], config["k"]

            head_a = nn.Linear(128, r * in_c * k)
            head_b = nn.Linear(128, out_c * r)

            # Bias-HyperInit, same policy as task-level hypernetwork.
            nn.init.zeros_(head_a.weight)
            bound = 1.0 / (in_c * k)
            nn.init.uniform_(head_a.bias, -bound, bound)

            nn.init.zeros_(head_b.weight)
            nn.init.zeros_(head_b.bias)

            self.heads_A.append(head_a)
            self.heads_B.append(head_b)

    def forward(self, class_embeddings: Tensor, layer_idx: int):
        # class_embeddings: (Batch, num_classes, hidden_dim)
        batch_size = class_embeddings.shape[0]

        query = self.layer_query(
            torch.tensor(layer_idx, device=class_embeddings.device)
        )
        query = query.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, -1)
        context, _ = self.cross_attention(query, class_embeddings, class_embeddings)
        context = context.squeeze(1)
        # (B, H)

        h = self.mlp(context)
        # (B, D_hnet)
        A_flat = self.heads_A[layer_idx](h)
        # (B, r * in_c * k)
        B_flat = self.heads_B[layer_idx](h)
        # (B, out_c * r)
        return A_flat, B_flat
