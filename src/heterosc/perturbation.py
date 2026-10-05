"""Gene perturbation prediction with HeteroSC on top of GEARS (manuscript Section 2.4).

Routing unit = gene. Four frozen foundation models each provide a (N_genes x dim_k) gene-embedding table aligned to
the GEARS gene order. HeteroSC fuses them with expert-specific adapters and an AdaptiveRouter, and the fused matrix
replaces GEARS' own gene-embedding layer.

Two ways to use the fused embedding:

* **End-to-end fusion** (``attach_heterosc_to_gears``): the GEARS gene-embedding module is *replaced* by
  :class:`HeteroSCGeneEmbedding`; adapters + router are trained through the GEARS objective while everything else in
  GEARS is frozen. Used for the Fusion Ladder (all 15 expert combinations) and the adaptive-vs-uniform ablation.
* **Initialisation** (``export_fused_matrix``): after the fusion module has been trained, its fused (N_genes x d) matrix
  is exported and given to a fresh GEARS model as the initial gene-embedding matrix (frozen or unfrozen).

IMPORTANT: if the fusion module is *not* trained first, ``export_fused_matrix`` returns a fixed random projection of
the foundation-model embeddings (the adapters are randomly initialised). Say which one you used when you report results.

The foundation-model backbones are never fine-tuned.
"""
from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .modules import HeteroSCFusion


# ----------------------------------------------------------------------------- aligning FM gene tables to GEARS
def align_gene_table(
    gene_names: Sequence[str],
    table: Dict[str, np.ndarray],
    dim: int,
    key_fn=None,
    seed: int = 0,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Build a (n_genes x dim) matrix in GEARS gene order **by gene identity**.

    ``table`` maps a gene key (symbol or Ensembl id, whatever the foundation model uses) to its embedding.
    ``key_fn`` converts a GEARS gene symbol to that key (e.g. symbol -> Ensembl for Geneformer). Genes absent from the
    model's vocabulary get N(0,1) random vectors, which is also the default GEARS initialisation.
    Returns ``(matrix, found_mask)``; always report ``found_mask.mean()`` (vocabulary coverage) per model.
    """
    g = torch.Generator().manual_seed(seed)
    mat = torch.randn(len(gene_names), dim, generator=g)
    found = np.zeros(len(gene_names), dtype=bool)
    for i, name in enumerate(gene_names):
        key = key_fn(name) if key_fn is not None else name
        vec = table.get(key)
        if vec is not None:
            mat[i] = torch.as_tensor(np.asarray(vec), dtype=torch.float32)
            found[i] = True
    return mat, found


# ----------------------------------------------------------------------------- the trainable embedding module
class HeteroSCGeneEmbedding(nn.Module):
    """Drop-in replacement for ``GEARS_Model.gene_emb`` (an ``nn.Embedding(num_genes, hidden, max_norm=True)``).

    The raw foundation-model tables are *buffers* (never trained). Adapters + router are the only parameters.
    ``forward(indices)`` mirrors ``nn.Embedding``: it fuses all genes and indexes the result, so the GEARS loss
    back-propagates into the adapters and router.
    """

    def __init__(self, expert_tables: Dict[str, torch.Tensor], d_shared: int = 512, d_router_hidden: int = 256,
                 dropout_adapter: float = 0.15, dropout_router: float = 0.15, fusion: str = "adaptive",
                 max_norm: Optional[float] = 1.0):
        super().__init__()
        self.expert_names = list(expert_tables)
        self.max_norm = max_norm
        for name, t in expert_tables.items():
            self.register_buffer(f"raw_{name}", t.detach().float().clone())
        self.fusion = HeteroSCFusion({n: t.shape[1] for n, t in expert_tables.items()}, d_shared, d_router_hidden,
                                     dropout_adapter, dropout_router, fusion)

    def compute_fused(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(h_fused (n_genes, d), alpha (n_genes, K))`` for all genes."""
        return self.fusion({n: getattr(self, f"raw_{n}") for n in self.expert_names})

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        out = self.compute_fused()[0][indices]
        if self.max_norm is not None:                                   # differentiable stand-in for nn.Embedding(max_norm)
            scale = torch.clamp(self.max_norm / (out.norm(dim=1, keepdim=True) + 1e-7), max=1.0)
            out = out * scale
        return out

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def attach_heterosc_to_gears(gears_model, expert_tables: Dict[str, torch.Tensor], d_shared: int = 512,
                             fusion: str = "adaptive", device: str = "cuda", **kw) -> HeteroSCGeneEmbedding:
    """Replace ``gears_model.model.gene_emb`` and freeze every other GEARS parameter.

    ``gears_model`` is an *initialised* ``gears.GEARS`` object (``model_initialize(hidden_size=d_shared,
    use_pretrained=False)``), ideally with the weights of a GEARS-default run loaded (stage 1). After this call,
    ``gears_model.train(...)`` optimises only the adapters and the router.
    """
    module = HeteroSCGeneEmbedding(expert_tables, d_shared, fusion=fusion, **kw).to(device)
    gears_model.model.gene_emb = module
    for name, p in gears_model.model.named_parameters():
        p.requires_grad = name.startswith("gene_emb.")
    return module


@torch.no_grad()
def export_fused_matrix(module: HeteroSCGeneEmbedding) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused gene matrix (to use as GEARS initial embedding) and the per-gene routing weights."""
    was_training = module.training
    module.eval()
    h, alpha = module.compute_fused()
    module.train(was_training)
    return h.cpu(), alpha.cpu()


# ----------------------------------------------------------------------------- Fusion Ladder
def expert_combinations(names: Sequence[str]) -> List[Tuple[str, ...]]:
    """All non-empty subsets of the experts (15 for 4 experts), ordered by size: the Fusion Ladder."""
    return [c for k in range(1, len(names) + 1) for c in itertools.combinations(names, k)]
