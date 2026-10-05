"""scGPT (whole-human) frozen embeddings. Environment: ``envs/scgpt.yml``.

Cell embedding = CLS embedding (512-d), L2-normalised. Gene embedding = the encoder token-embedding table.
Ported from the authors' notebook; needs the scGPT repository and the whole-human checkpoint (not exercised by unit tests).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from .common import save_embeddings


def _load_weights_with_wqkv_conversion(model, ckpt_path, raise_on_critical_skip=True):
    import torch

    ck = torch.load(ckpt_path, map_location="cpu")
    sd = ck.get("model_state_dict", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    conv = {}
    for k, v in sd.items():
        conv[k.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight").replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")] = v
    own = model.state_dict()
    ok = {k: v for k, v in conv.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in conv if k not in ok]
    own.update(ok); model.load_state_dict(own)
    critical = [k for k in skipped if "transformer_encoder" in k and "self_attn" in k]
    print(f"scGPT weights: loaded {len(ok)}, skipped {len(skipped)}")
    if critical and raise_on_critical_skip:
        raise RuntimeError(f"attention weights not loaded: {critical[:5]}")
    return model


def extract_scgpt_cells(adata_path, out_path, scgpt_repo, model_dir="pretrained model", batch_size=16, max_seq_len=3001,
                        n_bins=51, data_is_raw=True, amp=True, extra_obs=("cell_type", "master_split")):
    """``adata_path``: .h5ad with raw counts, gene *symbols* as var_names, ``obs['cell_id']``."""
    import anndata as ad
    import torch
    from scipy.sparse import issparse

    repo = Path(scgpt_repo); sys.path.insert(0, str(repo))
    from scgpt.model import TransformerModel
    from scgpt.preprocess import Preprocessor
    from scgpt.tokenizer import tokenize_and_pad_batch
    from scgpt.tokenizer.gene_tokenizer import GeneVocab

    mdir = repo / model_dir
    pad = "<pad>"
    vocab = GeneVocab.from_file(mdir / "vocab.json")
    for s in (pad, "<cls>", "<eoc>"):
        if s not in vocab:
            vocab.append_token(s)
    vocab.set_default_index(vocab[pad])
    cfg = json.load(open(mdir / "args.json"))
    emb = int(cfg.get("embsize", cfg.get("layer_size", 512))); nlayers = int(cfg.get("nlayers", 12))
    model = TransformerModel(
        len(vocab), emb, int(cfg.get("nheads", cfg.get("nhead", 8))), int(cfg.get("d_hid", emb)), nlayers,
        nlayers_cls=int(cfg.get("n_layers_cls", 3)), n_cls=2, vocab=vocab, dropout=0.0, pad_token=pad, pad_value=-2,
        do_mvc=False, do_dab=False, use_batch_labels=False, num_batch_labels=1, domain_spec_batchnorm=False,
        input_emb_style=cfg.get("input_emb_style", "continuous"), n_input_bins=n_bins, cell_emb_style="cls",
        mvc_decoder_style="inner product", ecs_threshold=0.0, explicit_zero_prob=False, use_fast_transformer=False,
        fast_transformer_backend="linear", pre_norm=False)
    _load_weights_with_wqkv_conversion(model, mdir / "best_model.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    a = ad.read_h5ad(adata_path)
    a.var.index.name = None
    a.var["gene_name"] = a.var_names.astype(str).values
    _, first = np.unique(a.var["gene_name"].values, return_index=True)
    a = a[:, sorted(first)].copy()
    keep = np.array([g in vocab for g in a.var["gene_name"]])
    print(f"scGPT vocabulary match: {keep.sum()}/{len(keep)} genes")
    a = a[:, keep].copy()
    if "cell_id" not in a.obs:
        a.obs["cell_id"] = a.obs_names.astype(str)
    Preprocessor(use_key="X", filter_gene_by_counts=False, filter_cell_by_counts=False, normalize_total=1e4,
                 result_normed_key="X_normed", log1p=data_is_raw, result_log1p_key="X_log1p", subset_hvg=False,
                 hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger", binning=n_bins, result_binned_key="X_binned")(a, batch_key=None)
    counts = a.layers["X_binned"]; counts = counts.A if issparse(counts) else np.asarray(counts)
    gene_ids = np.array(vocab(a.var["gene_name"].tolist()), dtype=int)
    tok = tokenize_and_pad_batch(counts, gene_ids, max_len=max_seq_len, vocab=vocab, pad_token=pad, pad_value=-2,
                                 append_cls=True, include_zero_gene=False)
    genes, values = tok["genes"], tok["values"]
    with torch.no_grad(), torch.amp.autocast(device_type=dev, enabled=amp and dev == "cuda"):
        cell = model.encode_batch(genes, values.float(), src_key_padding_mask=genes.eq(vocab[pad]), batch_size=batch_size,
                                  batch_labels=None, time_step=0, return_np=True)
    cell = cell / np.maximum(np.linalg.norm(cell, axis=1, keepdims=True), 1e-12)
    obs = a.obs[[c for c in extra_obs if c in a.obs]] if any(c in a.obs for c in extra_obs) else None
    return save_embeddings(out_path, cell, a.obs["cell_id"].values, obs)


def scgpt_gene_table(scgpt_repo, model_dir="pretrained model") -> Dict[str, np.ndarray]:
    """``{gene symbol: input embedding}`` from the encoder token-embedding table of the checkpoint."""
    import torch

    mdir = Path(scgpt_repo) / model_dir
    vocab = json.load(open(mdir / "vocab.json"))                                  # {symbol: token id}
    ck = torch.load(mdir / "best_model.pt", map_location="cpu")
    sd = ck.get("model_state_dict", ck) if isinstance(ck, dict) else ck
    key = next(k for k in ("encoder.embedding.weight", "gene_encoder.embedding.weight", "embedding.weight")
               if k in {kk.replace("module.", "") for kk in sd})
    W = {k.replace("module.", ""): v for k, v in sd.items()}[key].detach().cpu().numpy()
    return {g: W[i] for g, i in vocab.items() if i < W.shape[0]}
