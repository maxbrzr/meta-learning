from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn

from meta_learning.models.tiny_har import TinyHAR


class SetEncoder(ABC, nn.Module):
    def __init__(self, encoder: TinyHAR, feature_dim: int):
        super(SetEncoder, self).__init__()

        self.encoder = encoder

        # mlp to map to mu and sigma
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )

    @abstractmethod
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, context_size, feature_dim) -> (batch_size, feature_dim)
        pass

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (batch_size, context_size, time, channels)
        batch_size, context_size, time, channels = x.shape

        x = x.view(-1, time, channels)
        # (batch_size * context_size, time, channels))

        features = self.encoder.encode(x)  # type: ignore
        # (batch_size * context_size, feature_dim)

        features = features.view(batch_size, context_size, -1)
        # (batch_size, context_size, feature_dim)

        aggregated = self.aggregate(features)
        # (batch_size, feature_dim)

        proj_out = self.proj(aggregated)
        # (batch_size, feature_dim * 2)

        mu, logvar = proj_out.chunk(2, dim=-1)
        # (batch_size, feature_dim), (batch_size, feature_dim)

        return mu, logvar


class MeanSetEncoder(SetEncoder):
    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class MeanAttentiveSetEncoder(SetEncoder):
    def __init__(self, encoder: TinyHAR, feature_dim: int, num_heads: int = 4):
        super(MeanAttentiveSetEncoder, self).__init__(encoder, feature_dim)

        self.layer = nn.TransformerEncoderLayer(
            feature_dim, num_heads, batch_first=True
        )

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, context_size, feature_dim)
        x = self.layer(x)
        # (batch_size, context_size, feature_dim)

        return x.mean(dim=1)


class QueryAttentiveSetEncoder(SetEncoder):
    def __init__(self, encoder: TinyHAR, feature_dim: int, num_heads: int = 4):
        super(QueryAttentiveSetEncoder, self).__init__(encoder, feature_dim)

        self.attention = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)

        self.query = nn.Parameter(torch.empty(1, 1, feature_dim))
        torch.nn.init.xavier_normal_(self.query)

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, context_size, feature_dim)
        batch_size = x.size(0)

        # use the learnable parameter as the query
        # (1, 1, feature_dim) -> (batch_size, 1, feature_dim)
        query = self.query.expand(batch_size, -1, -1)

        attn_output, _ = self.attention(query, x, x)
        # (batch_size, 1, feature_dim)

        return attn_output.squeeze(1)


class BayesianSetEncoder(SetEncoder):
    """
    Implements Bayesian Context Aggregation (BA) as described in Volpp et al. (2021).

    In this encoder, context aggregation is treated as a Bayesian inference problem.
    Each context point is mapped to a latent observation with a mean and variance,
    which are then aggregated via closed-form Gaussian conditioning to produce
    the posterior distribution of the latent variable z.
    """

    def __init__(self, encoder: TinyHAR, feature_dim: int):
        super(BayesianSetEncoder, self).__init__(encoder, feature_dim)

        # In Bayesian Aggregation, the aggregation step itself yields the
        # distribution parameters (mu and sigma). Therefore, we replace the
        # separate projection layer (self.proj) from the base class with Identity,
        # as the aggregation result already has the correct shape and semantics.
        # [cite: 35, 115]
        self.proj = nn.Identity()

        # Network to map each context point feature to latent observation parameters
        # r_n (mean) and log(sigma^2_r_n) (log variance) [cite: 118]
        self.element_mapper = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim * 2),
        )

        # Define the prior distribution N(0, I) for the latent variable z.
        # We use buffers so they move with the device but are not trainable parameters.
        # Prior precision is 1.0 (inverse of variance 1.0).
        self.prior_mu = nn.Buffer(torch.zeros(1, feature_dim))
        self.prior_precision = nn.Buffer(torch.ones(1, feature_dim))

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregates context features using Bayesian updates.

        Args:
            x: (batch_size, context_size, feature_dim)

        Returns:
            (batch_size, feature_dim * 2) containing concatenated mu_z and logvar_z
        """
        # 1. Map features to latent observations r_n and log_var_n [cite: 118]
        # Shape: (batch_size, context_size, feature_dim * 2)
        latent_obs = self.element_mapper(x)

        # Split into mean (r_n) and log variance
        r_n, log_var_n = latent_obs.chunk(2, dim=-1)

        # 2. Compute precisions for each context point
        # precision = 1 / variance
        # Softplus or Exp can be used; standard implementations often use exp of log_var.
        prec_n = torch.exp(-log_var_n)  # (batch, context, dim)

        # 3. Aggregate Precisions (Eq. 8a)
        # (sigma_z^2)^-1 = (sigma_z,0^2)^-1 + sum((sigma_r_n^2)^-1)
        # Sum precisions over the context set dimension (dim=1)
        sum_prec_n = prec_n.sum(dim=1)  # (batch, dim)
        prec_z = self.prior_precision + sum_prec_n

        # Compute posterior variance and log variance
        var_z = 1.0 / prec_z
        log_var_z = torch.log(var_z)

        # 4. Aggregate Means (Eq. 8b)
        # mu_z = mu_z,0 + sigma_z^2 * sum(sigma_r_n^-2 * (r_n - mu_z,0))
        # Since prior mean is 0, this simplifies to: mu_z = var_z * sum(prec_n * r_n)

        # Weighted sum of observations (precision weighted)
        weighted_obs = prec_n * r_n
        sum_weighted_obs = weighted_obs.sum(dim=1)  # (batch, dim)

        # Combine with prior mean (weighted by prior precision)
        # prior_term = self.prior_precision * self.prior_mu # = 0 since mu=0

        mu_z = var_z * sum_weighted_obs

        # Concatenate mu and logvar to match the output expected by self.proj
        # (which is Identity) in the base forward method.
        return torch.cat([mu_z, log_var_z], dim=-1)
