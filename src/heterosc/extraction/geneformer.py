"""Geneformer (V2-104M) frozen embeddings. Environment: ``envs/geneformer.yml``.

Cell embedding = CLS token of the last hidden layer (768-d). Gene embedding = input word-embedding table.
Ported from the authors' notebook; needs the Geneformer repository and checkpoint (not exercised by unit tests).
    git lfs install && git clone https://huggingface.co/ctheodoris/Geneformer
"""
from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .common import save_embeddings


def extract_geneformer_cells(adata_path, out_path, geneformer_repo, model_dir="Geneformer-V2-104M", batch_size=4,
                             model_input_size=4096, work_dir: Optional[Path] = None, extra_obs=("cell_type", "master_split")):
    """``adata_path``: .h5ad with raw counts, ``var['ensembl_id']`` and ``obs['cell_id']`` (see scripts/prepare_annotation_data.py)."""
    import anndata as ad
    import torch
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification

    repo = Path(geneformer_repo); pkg = repo / "geneformer"
    work = Path(work_dir or Path(out_path).parent / "geneformer_work"); work.mkdir(parents=True, exist_ok=True)
    pad_id = pickle.load(open(pkg / "token_dictionary_gc104M.pkl", "rb")).get("<pad>", 0)

    adata = ad.read_h5ad(adata_path)
    adata = adata[:, adata.var["ensembl_id"].notna()].copy()
    _, first = np.unique(adata.var["ensembl_id"].values, return_index=True)            # unique Ensembl ids only
    adata = adata[:, sorted(first)].copy()
    adata.var["ensembl_id"] = adata.var["ensembl_id"].astype(str)
    adata.var.index.name = None
    if "cell_id" not in adata.obs:
        adata.obs["cell_id"] = adata.obs_names.astype(str)
    if "n_counts" not in adata.obs:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    attrs = {"cell_id": "cell_id", **{c: c for c in extra_obs if c in adata.obs}}
    (work / "in").mkdir(exist_ok=True); (work / "tok").mkdir(exist_ok=True)
    adata.write_h5ad(work / "in" / "input.h5ad")

    TranscriptomeTokenizer(
        custom_attr_name_dict=attrs, nproc=1, chunk_size=512, model_input_size=model_input_size, special_token=True,
        collapse_gene_ids=True, use_h5ad_index=False, keep_counts=False, model_version="V2",
        gene_median_file=pkg / "gene_median_dictionary_gc104M.pkl", token_dictionary_file=pkg / "token_dictionary_gc104M.pkl",
        gene_mapping_file=pkg / "ensembl_mapping_dict_gc104M.pkl",
    ).tokenize_data(data_directory=work / "in", output_directory=work / "tok", output_prefix="gf", file_format="h5ad")
    ds = load_from_disk(str(work / "tok" / "gf.dataset"))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(str(repo / model_dir), num_labels=2, ignore_mismatched_sizes=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False

    def collate(feats):
        ids = pad_sequence([torch.tensor(f["input_ids"], dtype=torch.long) for f in feats], batch_first=True, padding_value=pad_id)
        out = {"input_ids": ids, "attention_mask": ids.ne(pad_id).long(), "_cell_id": [f["cell_id"] for f in feats]}
        for c in extra_obs:
            if feats and c in feats[0]:
                out[f"_{c}"] = [f[c] for f in feats]
        return out

    embs, ids, meta = [], [], {c: [] for c in extra_obs}
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate):
            ids += b.pop("_cell_id")
            for c in extra_obs:
                meta[c] += b.pop(f"_{c}", [])
            inp = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                h = model(**inp, output_hidden_states=True, return_dict=True).hidden_states[-1][:, 0, :]   # CLS token
            embs.append(h.float().cpu().numpy())
    del model; gc.collect(); torch.cuda.empty_cache()
    import pandas as pd
    obs = pd.DataFrame({c: v for c, v in meta.items() if v})
    return save_embeddings(out_path, np.concatenate(embs), ids, obs if len(obs.columns) else None)


def geneformer_gene_table(geneformer_repo, model_dir="Geneformer-V2-104M") -> Dict[str, np.ndarray]:
    """``{Ensembl id: input embedding}`` from the model's word-embedding table (keys are Ensembl ids)."""
    import torch
    from transformers import AutoModel

    pkg = Path(geneformer_repo) / "geneformer"
    token_dict = pickle.load(open(pkg / "token_dictionary_gc104M.pkl", "rb"))             # {ensembl id: token id}
    W = AutoModel.from_pretrained(str(Path(geneformer_repo) / model_dir)).get_input_embeddings().weight.detach().cpu().numpy()
    return {g: W[i] for g, i in token_dict.items() if isinstance(g, str) and g.startswith("ENSG") and i < W.shape[0]}


def symbol_to_ensembl(geneformer_repo):
    """Symbol -> Ensembl function from Geneformer's own dictionary (use as ``key_fn`` in align_gene_table)."""
    d = pickle.load(open(Path(geneformer_repo) / "geneformer" / "gene_name_id_dict_gc104M.pkl", "rb"))
    return d.get
