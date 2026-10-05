"""Cell-type annotation with HeteroSC (manuscript Section 2.3).

Adapted from the project notebook ``HeteroSC_MultiDataset_7Dataset_4Expert``.  Training recipe:
class-weighted cross-entropy + label smoothing + diversity regulariser (Eq. 6-8), linear warm-up +
cosine LR, optional mixup, AdamW, gradient clipping, composite early stopping and top-K checkpoint
averaging ("model soup").
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .losses import diversity_loss, normalized_entropy
from .modules import HeteroSCFusion
from .utils import set_seed

DEFAULT_EXPERT_DIMS = {"geneformer": 768, "scgpt": 512, "cellfm": 1536, "scfoundation": 768}


@dataclass
class AnnotationConfig:
    expert_dims: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_EXPERT_DIMS))
    num_classes: int = 10
    fusion: str = "adaptive"            # "adaptive" | "uniform"
    # architecture
    d_shared: int = 512
    d_router_hidden: int = 256
    d_predictor_hidden: int = 512
    dropout_adapter: float = 0.15
    dropout_router: float = 0.15
    dropout_predictor: float = 0.25
    # optimisation
    epochs: int = 150
    batch_size: int = 32
    lr: float = 5e-4
    lr_min: float = 5e-6
    warmup_epochs: int = 15
    weight_decay: float = 1e-4
    patience: int = 25
    top_k_average: int = 3
    # loss / regularisation
    label_smoothing: float = 0.08
    lambda_div: float = 0.05            # weight of the diversity term (lambda_div in Eq. 8)
    cw_method: str = "sqrt_inverse"     # class weights: sqrt_inverse | inverse | balanced | none
    cw_max_ratio: float = 5.0
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    es_composite_w: float = 0.5         # early stopping on w*macroF1 + (1-w)*accuracy
    seed: int = 42
    device: str = "auto"

    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


class GlobalPredictor(nn.Module):
    """Task head: d -> hidden -> hidden/2 -> num_classes."""

    def __init__(self, d_shared: int, d_hidden: int, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_shared, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class HeteroSCClassifier(nn.Module):
    """HeteroSCFusion + GlobalPredictor.  ``forward`` -> (logits, alpha, h_fused)."""

    def __init__(self, cfg: AnnotationConfig):
        super().__init__()
        self.cfg = cfg
        self.fusion = HeteroSCFusion(
            cfg.expert_dims, cfg.d_shared, cfg.d_router_hidden,
            cfg.dropout_adapter, cfg.dropout_router, cfg.fusion)
        self.predictor = GlobalPredictor(
            cfg.d_shared, cfg.d_predictor_hidden, cfg.num_classes, cfg.dropout_predictor)
        self.expert_names = self.fusion.expert_names
        self.n_experts = self.fusion.n_experts

    def forward(self, expert_embs: Dict[str, torch.Tensor]):
        h_fused, alpha = self.fusion(expert_embs)
        return self.predictor(h_fused), alpha, h_fused

    @torch.no_grad()
    def fused_embeddings(self, expert_embs: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.eval()
        return self.fusion(expert_embs)[0]


# ----------------------------------------------------------------------------- data
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: Dict[str, np.ndarray], labels: np.ndarray, indices: np.ndarray):
        self.embeddings = {k: torch.tensor(v[indices], dtype=torch.float32) for k, v in embeddings.items()}
        self.labels = torch.tensor(labels[indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        return {k: v[i] for k, v in self.embeddings.items()}, self.labels[i]


def _collate(batch):
    embs, labels = zip(*batch)
    return {k: torch.stack([b[k] for b in embs]) for k in embs[0]}, torch.stack(labels)


def make_loaders(embeddings, labels, split, batch_size: int = 32):
    """Build train/val/test loaders from a ``split`` array of {'train','val','test'}."""
    idx = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
    for s, i in idx.items():
        if len(i) == 0:
            raise ValueError(f"No cells in split '{s}'.")
    mk = lambda s, shuffle: DataLoader(
        EmbeddingDataset(embeddings, labels, idx[s]), batch_size=batch_size,
        shuffle=shuffle, collate_fn=_collate, num_workers=0)
    return mk("train", True), mk("val", False), mk("test", False)


# ----------------------------------------------------------------------------- utilities
def build_class_weights(labels: np.ndarray, num_classes: int,
                        method: Optional[str] = "sqrt_inverse", max_ratio: Optional[float] = 5.0) -> np.ndarray:
    if method in (None, "none"):
        return np.ones(num_classes, dtype=np.float32)
    counts = np.maximum(np.bincount(labels, minlength=num_classes).astype(float), 1)
    if method == "inverse":
        w = 1.0 / counts
    elif method == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    elif method == "balanced":
        w = len(labels) / (num_classes * counts)
    else:
        raise ValueError(f"Unknown cw_method {method!r}")
    w = w / w.sum() * num_classes
    if max_ratio is not None:
        w = np.clip(w, w.min(), w.min() * max_ratio)
        w = w / w.sum() * num_classes
    return w.astype(np.float32)


def lr_at(epoch: int, cfg: AnnotationConfig) -> float:
    """Linear warm-up then cosine decay (epoch is 0-based)."""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(cfg.warmup_epochs, 1)
    progress = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1)
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1 + np.cos(np.pi * progress))


def mixup_batch(emb: Dict[str, torch.Tensor], labels: torch.Tensor, alpha: float = 0.4):
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)
    perm = torch.randperm(labels.size(0), device=labels.device)
    return {k: lam * v + (1 - lam) * v[perm] for k, v in emb.items()}, labels, labels[perm], lam


def average_state_dicts(states: List[dict]) -> dict:
    return {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model: HeteroSCClassifier, loader: DataLoader, criterion: nn.Module, device: str) -> dict:
    model.eval()
    loss_sum, n, preds, labels, alphas = 0.0, 0, [], [], []
    for emb, y in loader:
        emb = {k: v.to(device) for k, v in emb.items()}
        y = y.to(device)
        logits, alpha, _ = model(emb)
        loss_sum += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        preds.append(logits.argmax(-1).cpu().numpy())
        labels.append(y.cpu().numpy())
        alphas.append(alpha.cpu().numpy())
    preds, labels, alphas = np.concatenate(preds), np.concatenate(labels), np.concatenate(alphas)
    mean_alpha = alphas.mean(axis=0)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        "loss": loss_sum / max(n, 1),
        "accuracy": float(acc),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "expert_utilization": mean_alpha.tolist(),                 # mean routing weight per expert
        "normalized_entropy": normalized_entropy(mean_alpha),      # Eq. 9
        "preds": preds, "labels": labels, "alphas": alphas,
    }


# ----------------------------------------------------------------------------- trainer
class AnnotationTrainer:
    def __init__(self, model: HeteroSCClassifier, cfg: AnnotationConfig, output_dir=None):
        self.model, self.cfg, self.device = model, cfg, cfg.device
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

    def _criterion(self, train_labels: np.ndarray) -> nn.CrossEntropyLoss:
        w = build_class_weights(train_labels, self.cfg.num_classes, self.cfg.cw_method, self.cfg.cw_max_ratio)
        return nn.CrossEntropyLoss(weight=torch.tensor(w, device=self.device),
                                   label_smoothing=self.cfg.label_smoothing)

    def _train_epoch(self, loader, optimizer, criterion) -> float:
        self.model.train()
        total = 0.0
        for emb, y in loader:
            emb = {k: v.to(self.device) for k, v in emb.items()}
            y = y.to(self.device)
            optimizer.zero_grad()
            if self.cfg.use_mixup and np.random.rand() > 0.5:
                mixed, ya, yb, lam = mixup_batch(emb, y, self.cfg.mixup_alpha)
                logits, alpha, _ = self.model(mixed)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits, alpha, _ = self.model(emb)
                loss = criterion(logits, y)
            if self.model.fusion.fusion == "adaptive":          # nothing to regularise for uniform fusion
                loss = loss + self.cfg.lambda_div * diversity_loss(alpha)   # Eq. 8
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        return total / len(loader)

    def fit(self, train_loader, val_loader, test_loader, train_labels: np.ndarray,
            verbose: bool = True, log_every: int = 10) -> dict:
        cfg = self.cfg
        set_seed(cfg.seed)
        criterion = self._criterion(train_labels)
        opt = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, bad, top, hist = -1.0, 0, [], []
        t0 = time.time()
        for epoch in range(1, cfg.epochs + 1):
            lr = lr_at(epoch - 1, cfg)
            for g in opt.param_groups:
                g["lr"] = lr
            tr_loss = self._train_epoch(train_loader, opt, criterion)
            vl = evaluate(self.model, val_loader, criterion, self.device)
            score = cfg.es_composite_w * vl["macro_f1"] + (1 - cfg.es_composite_w) * vl["accuracy"]
            hist.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss, "val_loss": vl["loss"],
                         "val_macro_f1": vl["macro_f1"], "val_accuracy": vl["accuracy"],
                         "val_score": score, "norm_entropy": vl["normalized_entropy"]})
            top.append((score, {k: v.detach().clone() for k, v in self.model.state_dict().items()}))
            top = sorted(top, key=lambda t: -t[0])[: cfg.top_k_average]
            if score > best:
                best, bad = score, 0
            else:
                bad += 1
            if verbose and (epoch == 1 or epoch % log_every == 0):
                print(f"ep {epoch:3d} | lr {lr:.1e} | val F1 {vl['macro_f1']:.4f} acc {vl['accuracy']:.4f} "
                      f"| H_norm {vl['normalized_entropy']:.3f} | alpha {np.round(vl['expert_utilization'], 3)}")
            if bad >= cfg.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch}")
                break
        self.model.load_state_dict(average_state_dicts([s for _, s in top]))   # top-K "model soup"
        test = evaluate(self.model, test_loader, criterion, self.device)
        if verbose:
            print(f"TEST  acc {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f} | "
                  f"alpha {np.round(test['expert_utilization'], 3)} | H_norm {test['normalized_entropy']:.3f}")
        out = {"test": test, "history": pd.DataFrame(hist), "train_time_sec": time.time() - t0}
        if self.output_dir:
            self._save(out)
        return out

    def _save(self, out: dict) -> None:
        self.cfg.save(self.output_dir / "config.json")
        out["history"].to_csv(self.output_dir / "train_history.csv", index=False)
        torch.save(self.model.state_dict(), self.output_dir / "model_weights.pt")
        t = {k: v for k, v in out["test"].items() if k not in ("preds", "labels", "alphas")}
        t["train_time_sec"] = out["train_time_sec"]
        (self.output_dir / "test_metrics.json").write_text(json.dumps(t, indent=2))
        np.save(self.output_dir / "test_routing_alpha.npy", out["test"]["alphas"])
