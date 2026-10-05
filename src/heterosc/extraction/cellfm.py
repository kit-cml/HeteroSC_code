"""CellFM frozen embeddings. Environment: ``envs/cellfm.yml`` (CellFM-torch + MindSpore for reading the checkpoint).

Cell embedding = CLS token (1,536-d). Ported from the authors' notebook; needs the CellFM-torch repository and the 80M
checkpoint (not exercised by unit tests). CAUTION: the authors' own notes say frozen CellFM embeddings may be less
informative than the other experts; report that as a limitation rather than hiding it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .common import save_embeddings


def _patch_config(cfg):
    defaults = {"ecs": False, "ecs_threshold": 0.8, "do_mvc": False, "do_dab": False, "use_batch_labels": False,
                "domain_spec_batchnorm": False, "num_batch_labels": 1, "input_emb_style": "continuous",
                "cell_emb_style": "cls", "mvc_decoder_style": "inner product", "pad_token_id": 0, "pad_value": 0,
                "mask_value": -1, "dropout": getattr(cfg, "dropout", 0.2)}
    for k, v in defaults.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def extract_cellfm_cells(adata_path, out_path, cellfm_repo, ckpt="pretrained model/CellFM_80M_weight.ckpt", batch_size=8,
                         extra_obs=("cell_type", "master_split")):
    import anndata as ad
    import torch
    from scipy.sparse import issparse

    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from mindspore.train.serialization import load_checkpoint as ms_load_checkpoint

    repo = Path(cellfm_repo); os.chdir(repo); sys.path.insert(0, str(repo))
    from layers.utils import Config_80M, Prepare, SCrna, build_dataset
    from model import Finetune_Cell_FM

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    a = ad.read_h5ad(adata_path)
    a.var.index.name = None
    a.var_names = a.var_names.astype(str); a.var_names_make_unique()
    if "cell_id" not in a.obs:
        a.obs["cell_id"] = a.obs_names.astype(str)
    a.obs["celltype"], a.obs["celltype_id"], a.obs["feat"] = "x", 0, 0                  # dummy labels: frozen inference only
    a.obs["split"], a.obs["str_batch"], a.obs["batch_id"], a.obs["train"] = "train", "batch_0", 0, 0
    X = a.X.toarray() if issparse(a.X) else np.asarray(a.X)
    X = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0); X[X < 0] = 0
    a.X = X

    cfg = _patch_config(Config_80M())
    cfg.ckpt_path, cfg.device, cfg.add_zero, cfg.pad_zero, cfg.use_bs, cfg.mask_ratio = str(repo / ckpt), dev, True, True, batch_size, 0.0
    cfg.dataset, cfg.feature_col, cfg.num_cls = "embedding", "celltype", 1
    net = Finetune_Cell_FM(cfg)
    net.extractor.ms_ckpt = ms_load_checkpoint(cfg.ckpt_path); net.extractor.load_weight()
    net = net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad = False

    loader = build_dataset(SCrna(a, mode="train"), prep=Prepare(cfg.nonz_len, pad=0, mask_ratio=0.0), batch_size=batch_size,
                           pad_zero=cfg.pad_zero, drop=False, shuffle=False)
    out = []
    with torch.no_grad():
        for b in loader:
            _, cls = net.extractor(raw_nzdata=b["raw_nzdata"].to(dev), dw_nzdata=b["dw_nzdata"].to(dev), ST_feat=b["ST_feat"].to(dev),
                                   nonz_gene=b["nonz_gene"].long().to(dev), mask_gene=b["mask_gene"].to(dev), zero_idx=b["zero_idx"].to(dev))
            out.append(cls.float().cpu().numpy())
    emb = np.concatenate(out)
    obs = a.obs[[c for c in extra_obs if c in a.obs]].iloc[: len(emb)] if any(c in a.obs for c in extra_obs) else None
    return save_embeddings(out_path, emb, a.obs["cell_id"].values[: len(emb)], obs)


def cellfm_gene_table(net, gene_names: Sequence[str], expected_dim: int = 1536) -> Dict[str, np.ndarray]:
    """``{gene: embedding}`` from CellFM's gene-embedding parameter.

    ``gene_names`` must list the genes in the SAME ORDER as the rows of CellFM's gene-embedding matrix (take it from the
    CellFM repository's gene list). The function refuses to guess: assigning rows to genes by position without knowing
    the vocabulary order gives arbitrary vectors.
    """
    cands = [(n, p) for n, p in net.named_parameters() if p.ndim == 2 and p.shape[1] == expected_dim and "gene_emb" in n.lower()]
    if not cands:
        raise RuntimeError(f"no 2-D parameter with 'gene_emb' in its name and width {expected_dim}; inspect net.named_parameters()")
    name, W = max(cands, key=lambda t: t[1].shape[0])
    W = W.detach().cpu().numpy()
    if len(gene_names) > W.shape[0]:
        raise ValueError(f"{len(gene_names)} gene names but the embedding table '{name}' has only {W.shape[0]} rows")
    return {g: W[i] for i, g in enumerate(gene_names)}
