#!/usr/bin/env python
"""Train HeteroSC (or the uniform-fusion ablation) on pre-extracted expert embeddings, and report
fine-tuned + zero-shot kNN results.

Usage:
  python scripts/train_annotation.py --emb-root outputs/embeddings --dataset Pancreas \
      --experts geneformer scgpt cellfm scfoundation --out outputs/heterosc --fusion adaptive
"""
import argparse
import json
from pathlib import Path

import numpy as np

from heterosc import (AnnotationConfig, AnnotationTrainer, HeteroSCClassifier, knn_probe,
                      load_dataset, make_loaders)
from heterosc.data import l2_normalize
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-root", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experts", nargs="+", default=["geneformer", "scgpt", "cellfm", "scfoundation"])
    ap.add_argument("--fusion", choices=["adaptive", "uniform"], default="adaptive")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    emb, y, split, classes = load_dataset(a.emb_root, a.dataset, a.experts)
    # embeddings are L2-normalised before fusion, as in the original notebook
    emb = {k: l2_normalize(v) for k, v in emb.items()}
    cfg = AnnotationConfig(expert_dims={k: v.shape[1] for k, v in emb.items()},
                           num_classes=len(classes), fusion=a.fusion, epochs=a.epochs, seed=a.seed)
    tr, va, te = make_loaders(emb, y, split, cfg.batch_size)
    model = HeteroSCClassifier(cfg)
    out_dir = a.out / a.dataset / a.fusion
    res = AnnotationTrainer(model, cfg, out_dir).fit(tr, va, te, y[split == "train"])

    # zero-shot kNN in the fused representation space (5 NN, cosine, distance-weighted)
    dev = cfg.device
    fused = {}
    for s in ("train", "test"):
        e = {k: torch.tensor(v[split == s]).to(dev) for k, v in emb.items()}
        fused[s] = model.fused_embeddings(e).cpu().numpy()
    knn = knn_probe(fused["train"], y[split == "train"], fused["test"], y[split == "test"])
    (out_dir / "zero_shot_knn.json").write_text(json.dumps(knn, indent=2))
    print("zero-shot kNN:", knn)


if __name__ == "__main__":
    main()
