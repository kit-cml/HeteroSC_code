"""Core HeteroSC building blocks (manuscript Eq. 1-5).

Frozen foundation-model (FM) embeddings  ->  expert-specific adapters  ->  AdaptiveRouter
->  routing-weighted fusion.  Only adapters, router and the task head are trainable;
the FM backbones are never touched.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterMLP(nn.Module):
    """Expert-specific adapter f_k (Eq. 1): 3-layer gradual bottleneck.

    input_dim -> max(4*d, input_dim//2) -> 2*d -> d   with LayerNorm + GELU + Dropout.
    """

    def __init__(self, input_dim: int, d_shared: int, dropout: float = 0.15):
        super().__init__()
        mid = max(d_shared * 4, input_dim // 2)
        mid2 = d_shared * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, mid), nn.LayerNorm(mid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid, mid2), nn.LayerNorm(mid2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mid2, d_shared), nn.LayerNorm(d_shared),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaptiveRouter(nn.Module):
    """Residual-MLP router (Eq. 2-3).

    concat(h_1..h_K) in R^{K*d} -> proj -> +block1 (residual) -> block2 -> K logits -> softmax.
    Linear layers use a small Xavier gain (0.1) so routing starts close to uniform.
    """

    def __init__(self, n_experts: int, d_shared: int, d_hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.proj = nn.Linear(n_experts * d_shared, d_hidden)
        self.ln0 = nn.LayerNorm(d_hidden)
        self.block1 = nn.Sequential(
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Linear(d_hidden, d_hidden // 2), nn.LayerNorm(d_hidden // 2), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Linear(d_hidden // 2, n_experts)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_bars: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat(h_bars, dim=-1)
        h = F.gelu(self.ln0(self.proj(x)))
        h = h + self.block1(h)
        h = self.block2(h)
        logits = self.out(h)
        return F.softmax(logits, dim=-1), logits


class HeteroSCFusion(nn.Module):
    """Adapters + router + weighted sum (Eq. 1-5).

    Works for any "unit of routing": cells (annotation), cell lines (drug response) or genes
    (perturbation) -- just pass tensors of shape (N, dim_k) for each expert k.

    Args:
        expert_dims: ordered mapping ``{expert_name: native_embedding_dim}``.
        d_shared:    shared latent dimension d.
        fusion:      ``"adaptive"`` (HeteroSC) or ``"uniform"`` (ablation: alpha_k = 1/K,
                     same adapters, no router) -- see manuscript Section 3.4.
    """

    def __init__(
        self,
        expert_dims: Dict[str, int],
        d_shared: int = 512,
        d_router_hidden: int = 256,
        dropout_adapter: float = 0.15,
        dropout_router: float = 0.15,
        fusion: Literal["adaptive", "uniform"] = "adaptive",
    ):
        super().__init__()
        if fusion not in ("adaptive", "uniform"):
            raise ValueError(f"fusion must be 'adaptive' or 'uniform', got {fusion!r}")
        self.expert_names = list(expert_dims.keys())
        self.n_experts = len(self.expert_names)
        self.d_shared = d_shared
        self.fusion = fusion
        self.adapters = nn.ModuleDict(
            {n: AdapterMLP(d, d_shared, dropout_adapter) for n, d in expert_dims.items()})
        self.router = (
            AdaptiveRouter(self.n_experts, d_shared, d_router_hidden, dropout_router)
            if fusion == "adaptive" else None)

    def forward(self, expert_embs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(h_fused (N, d), alpha (N, K))``."""
        h_bars = [self.adapters[n](expert_embs[n]) for n in self.expert_names]
        if self.fusion == "adaptive":
            alpha, _ = self.router(h_bars)
        else:
            n = h_bars[0].shape[0]
            alpha = h_bars[0].new_full((n, self.n_experts), 1.0 / self.n_experts)
        stacked = torch.stack(h_bars, dim=1)                 # (N, K, d)   Eq. 4
        h_fused = (alpha.unsqueeze(-1) * stacked).sum(dim=1)  # (N, d)      Eq. 5
        return h_fused, alpha
