"""scFoundation frozen embeddings. Environment: ``envs/scfoundation.yml``.

Cell embedding = max-pool over the encoder output tokens (768-d). Gene embedding = positional (gene-index) embedding table.
Ported from the authors' notebook; needs the scFoundation repository and ``models.ckpt`` (not exercised by unit tests).

scFoundation indexes genes by position in its fixed 19,264-gene list. Your expression matrix MUST therefore be re-ordered
to that list (zeros for genes the model does not know). ``align_to_gene_list`` does this; skipping it assigns the wrong
gene identity to every column.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common import save_embeddings


def load_gene_list(gene_index_tsv, symbol_col: str = "gene_name") -> list:
    """The 19,264 gene symbols in scFoundation order (``OS_scRNA_gene_index.19264.tsv`` in the scFoundation repo)."""
    return pd.read_csv(gene_index_tsv, sep="\t")[symbol_col].astype(str).tolist()


def align_to_gene_list(X: np.ndarray, var_names, gene_list) -> np.ndarray:
    """Re-order columns of ``X`` (cells x genes) to ``gene_list``; unknown genes become zeros."""
    pos = {g: i for i, g in enumerate(var_names)}
    out = np.zeros((X.shape[0], len(gene_list)), dtype=np.float32)
    hit = 0
    for j, g in enumerate(gene_list):
        i = pos.get(g)
        if i is not None:
            out[:, j] = X[:, i]; hit += 1
    print(f"scFoundation gene coverage: {hit}/{len(var_names)} of your genes are in the model's list")
    return out


def extract_scfoundation_cells(adata_path, out_path, scf_repo, gene_index_tsv, batch_size=16,
                               extra_obs=("cell_type", "master_split")):
    import anndata as ad
    import torch

    repo = Path(scf_repo)
    sys.path.insert(0, str(repo / "model"))
    from load import gatherData, load_model_frommmf

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model_frommmf(str(repo / "model" / "models" / "models.ckpt"), key="cell")
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    a = ad.read_h5ad(adata_path)
    if "cell_id" not in a.obs:
        a.obs["cell_id"] = a.obs_names.astype(str)
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = align_to_gene_list(X, a.var_names.astype(str), load_gene_list(gene_index_tsv))

    @torch.no_grad()
    def encode(x):
        labels = x > 0
        xg, pad = gatherData(x, labels, config["pad_token_id"])
        ids = torch.arange(x.shape[1], device=x.device).repeat(x.shape[0], 1)
        pos, _ = gatherData(ids, labels, config["pad_token_id"])
        h = model.token_emb(torch.unsqueeze(xg, 2).float(), output_weight=0) + model.pos_emb(pos)
        return torch.max(model.encoder(h, pad), dim=1)[0]

    out = [encode(torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(dev)).float().cpu().numpy()
           for i in range(0, X.shape[0], batch_size)]
    obs = a.obs[[c for c in extra_obs if c in a.obs]] if any(c in a.obs for c in extra_obs) else None
    return save_embeddings(out_path, np.concatenate(out), a.obs["cell_id"].values, obs)


def scfoundation_gene_table(scf_repo, gene_index_tsv) -> Dict[str, np.ndarray]:
    """``{gene symbol: positional embedding}`` for the 19,264 genes (embedding row = position in the gene list)."""
    sys.path.insert(0, str(Path(scf_repo) / "model"))
    from load import load_model_frommmf

    model, _ = load_model_frommmf(str(Path(scf_repo) / "model" / "models" / "models.ckpt"), key="cell")
    W = model.pos_emb.weight.detach().cpu().numpy()
    genes = load_gene_list(gene_index_tsv)
    return {g: W[i] for i, g in enumerate(genes) if i < W.shape[0]}
