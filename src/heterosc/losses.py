"""Routing regularisers and routing statistics (manuscript Eq. 6-11)."""
from __future__ import annotations

import math

import numpy as np
import torch


def diversity_loss(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cell-annotation diversity term L_div (Eq. 6-7).

    Negative entropy of the *batch-mean* routing distribution; minimising it spreads usage
    across experts over a batch while leaving per-cell routing unconstrained.
    """
    mu = alpha.mean(dim=0)
    return (mu * torch.log(mu + eps)).sum()


def routing_entropy(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean *per-input* routing entropy H(alpha_i) (Eq. 10), used for drug response.

    Adding ``lambda_H * routing_entropy(alpha)`` to the loss (Eq. 11) pushes each input
    towards a selective expert allocation.
    """
    return -(alpha * torch.log(alpha + eps)).sum(dim=-1).mean()


def normalized_entropy(mean_alpha, eps: float = 1e-12) -> float:
    """Normalised Shannon entropy of the mean routing distribution (Eq. 9).

    0 = routing concentrated on a single expert, 1 = perfectly uniform usage. Undefined (NaN) for a single expert.
    """
    a = np.asarray(mean_alpha, dtype=np.float64)
    if len(a) < 2:
        return float("nan")
    a = a / a.sum()
    return float(-(a * np.log(a + eps)).sum() / math.log(len(a)))
