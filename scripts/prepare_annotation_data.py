#!/usr/bin/env python
"""Standardise a labelled scRNA-seq .h5ad and create the *master split* shared by all experts.

Mirrors manuscript Section 2.2 / project notebook ``Multi_Dataset_Preprocessing_HeteroSC``:
  * labels -> ``obs['cell_type']``; cell types with < ``--min-cells`` cells are removed
  * ``var['ensembl_id']`` filled (needed by Geneformer) from an existing column, ENSG-style
    var_names, or an optional symbol->Ensembl pickle (Geneformer ``gene_name_id_dict_gc104M.pkl``)
  * stratified 70/15/15 train/val/test split, seed 42  ->  ``obs['master_split']``
The SAME split file is then used by every foundation-model extractor.

Usage:
  python scripts/prepare_annotation_data.py --input data/raw/Pancrm.h5ad --name Pancreas \
      --label-col cell_type --out data/annotation
"""
import argparse
import pickle
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ENSG = re.compile(r"^(ENSG\d+)(\.\d+)?$")


def map_ensembl(adata, symbol_to_ensembl=None):
    if "ensembl_id" in adata.var.columns:
        return adata
    if "gene_ids" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["gene_ids"].astype(str)
        return adata
    mapped = []
    for name in adata.var_names:
        m = ENSG.match(name)
        mapped.append(m.group(1) if m else (symbol_to_ensembl or {}).get(name, np.nan))
    adata.var["ensembl_id"] = mapped
    print(f"[ensembl] mapped {pd.Series(mapped).notna().sum()}/{adata.n_vars} genes")
    return adata


def master_split(labels, seed=42, train_frac=0.7, val_frac=0.15):
    idx = np.arange(len(labels))
    rest = 1 - train_frac
    tr, rest_idx = train_test_split(idx, test_size=rest, stratify=labels, random_state=seed)
    va, _ = train_test_split(rest_idx, test_size=1 - val_frac / rest,
                             stratify=labels[rest_idx], random_state=seed)
    split = np.array(["test"] * len(labels), dtype=object)
    split[tr], split[va] = "train", "val"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--label-col", default="cell_type")
    ap.add_argument("--min-cells", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symbol-to-ensembl", type=Path, default=None,
                    help="optional pickle {gene symbol: ENSG id}")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    adata = ad.read_h5ad(a.input)
    adata.obs["cell_type"] = adata.obs[a.label_col].astype(str)
    counts = adata.obs["cell_type"].value_counts()
    keep = counts[counts >= a.min_cells].index
    adata = adata[adata.obs["cell_type"].isin(keep)].copy()

    sym = pickle.load(open(a.symbol_to_ensembl, "rb")) if a.symbol_to_ensembl else None
    adata = map_ensembl(adata, sym)
    if "n_counts" not in adata.obs.columns:
        adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).reshape(-1)
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["master_split"] = master_split(adata.obs["cell_type"].values, a.seed)

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.name}_ready_master_split.h5ad"
    adata.write_h5ad(path)
    print(f"{a.name}: {adata.n_obs} cells, {adata.obs['cell_type'].nunique()} cell types -> {path}")
    print(adata.obs["master_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
