#!/usr/bin/env python
"""Leave-drug-out cross-validation for one cell encoder (heterosc | uniform | geneformer | scgpt | cellfm | scfoundation).

  python scripts/train_drug_response.py --data data/drug_response --emb outputs/drug_response_embeddings \
      --cell-encoder heterosc --drug-mode gcn --out outputs/drug_response/heterosc_gcn
Each finished fold stores the MD5 of the fold table; results made with a different table are recomputed, never reused.
"""
import argparse
from pathlib import Path

from heterosc.data import keep_complete_cell_lines, load_cell_line_embeddings, load_drug_tables
from heterosc.drug_response import DrugFeatureStore, DrugResponseConfig, assert_leave_drug_out, run_cv

EXPERTS = ["geneformer", "scgpt", "cellfm", "scfoundation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--emb", required=True, type=Path)
    ap.add_argument("--cell-encoder", default="heterosc")
    ap.add_argument("--drug-mode", default="gcn", choices=["gcn", "fp", "ensemble"])
    ap.add_argument("--folds", type=int, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    df, graphs, fp = load_drug_tables(a.data)
    n = assert_leave_drug_out(df)
    emb = load_cell_line_embeddings(a.emb, EXPERTS)
    df = keep_complete_cell_lines(df, emb)
    dims = {e: len(next(iter(emb[e].values()))) for e in EXPERTS}
    cfg = DrugResponseConfig(cell_encoder=a.cell_encoder, drug_mode=a.drug_mode, expert_dims=dims)
    store = DrugFeatureStore(graphs, fp)
    res = run_cv(cfg, df, emb, store, a.out, folds=a.folds)
    print(res[["fold", "pcc_per_drug", "pcc_pooled", "mse"]].to_string(index=False))
    print(f"mean per-drug PCC over {len(res)} of {n} folds: {res['pcc_per_drug'].mean():.4f}")


if __name__ == "__main__":
    main()
