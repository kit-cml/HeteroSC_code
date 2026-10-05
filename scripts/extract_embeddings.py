#!/usr/bin/env python
"""Extract frozen cell embeddings from ONE foundation model. Run in that model's own environment (see envs/).

  python scripts/extract_embeddings.py --model geneformer --adata data/annotation/processed/Myeloid_ready_master_split.h5ad \
      --repo checkpoints/Geneformer --out outputs/embeddings/Myeloid/geneformer/Myeloid_geneformer_with_embeddings.h5ad
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["geneformer", "scgpt", "scfoundation", "cellfm"])
    ap.add_argument("--adata", required=True, type=Path)
    ap.add_argument("--repo", required=True, type=Path, help="local clone of the foundation model's repository")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gene-index", type=Path, help="scFoundation only: OS_scRNA_gene_index.19264.tsv")
    ap.add_argument("--batch-size", type=int, default=None)
    a = ap.parse_args()
    kw = {} if a.batch_size is None else {"batch_size": a.batch_size}
    if a.model == "geneformer":
        from heterosc.extraction.geneformer import extract_geneformer_cells as f
        f(a.adata, a.out, a.repo, **kw)
    elif a.model == "scgpt":
        from heterosc.extraction.scgpt import extract_scgpt_cells as f
        f(a.adata, a.out, a.repo, **kw)
    elif a.model == "scfoundation":
        from heterosc.extraction.scfoundation import extract_scfoundation_cells as f
        if a.gene_index is None:
            ap.error("--gene-index is required for scfoundation")
        f(a.adata, a.out, a.repo, a.gene_index, **kw)
    else:
        from heterosc.extraction.cellfm import extract_cellfm_cells as f
        f(a.adata, a.out, a.repo, **kw)
    print("saved ->", a.out)


if __name__ == "__main__":
    main()
