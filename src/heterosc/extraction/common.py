"""Shared I/O for the embedding-extraction functions.

Every extractor writes one AnnData file: ``X`` = (n_cells x dim) frozen embedding, ``obs['cell_id']``, plus any metadata
columns you pass (e.g. ``cell_type``, ``master_split``). The foundation-model backbones are always frozen
(``requires_grad=False``, ``torch.no_grad()``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def save_embeddings(path, X: np.ndarray, cell_ids: Sequence[str], obs: Optional[pd.DataFrame] = None) -> Path:
    import anndata as ad

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = pd.DataFrame(index=pd.Index([str(c) for c in cell_ids], name=None))
    if obs is not None:
        meta = obs.copy(); meta.index = [str(c) for c in cell_ids]
    meta["cell_id"] = [str(c) for c in cell_ids]
    ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=meta).write_h5ad(path)
    return path


def load_embeddings(path):
    """Returns ``(X, obs)``."""
    import anndata as ad

    a = ad.read_h5ad(path)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    return X.astype(np.float32), a.obs.copy()
