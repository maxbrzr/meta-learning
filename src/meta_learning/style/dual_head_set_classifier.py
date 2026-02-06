from typing import Any, Tuple

import torch
import torch.nn as nn
from torch.autograd import Function

from meta_learning.style.set_encoder import SetEncoder


class GradientReversalFn(Function):
    """
    Gradient Reversal Layer function.
    Forward: Identity
    Backward: Negates the gradient multiplied by alpha.
    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:  # type: ignore
        output = grad_output.neg() * ctx.alpha
        # Return gradients for inputs: (x, alpha).
        # alpha is a float, so its gradient is None.
        return output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.alpha)  # type: ignore


class DualHeadSetClassifier(nn.Module):
    def __init__(
        self,
        set_encoder: SetEncoder,
        feature_dim: int,
        num_subjects: int,
        num_activities: int,
        grl_alpha: float = 1.0,
    ):
        """
        Args:
            set_encoder: An instance of the SetEncoder class.
            feature_dim: The dimensionality of the encoder output (mu).
            num_subjects: Number of classes for the primary task head.
            num_activities: Number of classes for the adversarial head (with GRL).
            grl_alpha: The scaling factor for the reversed gradient.
        """
        super(DualHeadSetClassifier, self).__init__()

        self.set_encoder = set_encoder

        # 1. Primary Task Head (Standard)
        self.head_task = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, num_subjects),
        )

        # 2. Adversarial Head (With Gradient Reversal)
        self.grl = GradientReversalLayer(alpha=grl_alpha)
        self.head_adv = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, num_activities),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, context_size, time, channels)

        # Pass through the set encoder
        # We only care about mu as requested
        mu, sigma = self.set_encoder(x)
        # mu shape: (batch_size, feature_dim)

        # --- Head 1: Primary Task ---
        logits_task = self.head_task(mu)

        # --- Head 2: Adversarial Task (GRL) ---
        # Apply gradient reversal to the features before the adversarial head
        mu_reversed = self.grl(mu)
        logits_adv = self.head_adv(mu_reversed)

        return logits_task, logits_adv, mu, sigma
