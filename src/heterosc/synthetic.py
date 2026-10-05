"""Synthetic data for the CPU quickstart.

``make_context_experts`` simulates *complementary* foundation models: every cell belongs to a hidden
context (think tissue or batch). The expert that is "in-domain" for that context is clean and also
carries a context signature in its embedding; the other experts are noisy and carry no signature.
A router can therefore learn to trust whichever expert shows its signature, whereas uniform fusion
always mixes in the noisy experts.

This illustrates the mechanism only. It is not evidence about real foundation models, and the
adaptive-versus-uniform gap on this toy data depends on the settings below.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

DEFAULT_DIMS = {"geneformer": 768, "scgpt": 512, "cellfm": 1536, "scfoundation": 768}


def make_context_experts(
    n: int = 3000,
    n_classes: int = 6,
    dims: Optional[Dict[str, int]] = None,
    signal: float = 1.0,
    noise_clean: float = 1.0,
    noise_dirty: float = 4.0,
    context_strength: float = 1.5,
    seed: int = 0,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Returns ``(embeddings, labels, split, context)``; ``split`` in {train, val, test} (70/15/15)."""
    dims = dims or DEFAULT_DIMS
    rng = np.random.default_rng(seed)
    names = list(dims)
    k = len(names)
    labels = rng.integers(0, n_classes, n)
    context = rng.integers(0, k, n)                                   # context c -> expert c is the clean one
    emb: Dict[str, np.ndarray] = {}
    for j, name in enumerate(names):
        d = dims[name]
        class_centers = rng.normal(size=(n_classes, d)) / np.sqrt(d) * 8
        ctx_centers = rng.normal(size=(k, d)) / np.sqrt(d) * 8
        sigma = np.where(context == j, noise_clean, noise_dirty)[:, None] / np.sqrt(d) * 8
        in_domain = (context == j)[:, None]
        emb[name] = (signal * class_centers[labels] + context_strength * in_domain * ctx_centers[j][None, :]
                     + sigma * rng.normal(size=(n, d))).astype(np.float32)
    split = np.array(["train"] * n, dtype=object)
    perm = rng.permutation(n)
    split[perm[: int(0.15 * n)]] = "val"
    split[perm[int(0.15 * n): int(0.30 * n)]] = "test"
    return emb, labels, split, context
