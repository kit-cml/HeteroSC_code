"""Unseen-drug response prediction with HeteroSC (manuscript Section 2.5).

Cell-line transcriptomic representation  = HeteroSC fusion of four frozen foundation-model embeddings
                                           (or one expert with a plain projection, for the baselines)
Drug representation                      = edge-gated GCN | MACCS+ECFP4 fingerprint MLP | gated ensemble
Prediction                               = MLP on [cell, drug] -> AUC

Validation protocol: **10-fold leave-drug-out**. Every drug sits in exactly one fold, so a drug in the test fold is
never seen in training. ``assert_leave_drug_out`` refuses to run on a table that violates this.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .losses import routing_entropy
from .modules import HeteroSCFusion
from .utils import get_device, set_seed, warmup_cosine

NODE_DIM, EDGE_DIM, MAX_ATOMS = 67, 10, 100
DEFAULT_EXPERT_DIMS = {"geneformer": 768, "scgpt": 512, "cellfm": 1536, "scfoundation": 768}


# ----------------------------------------------------------------------------- data checks
def assert_leave_drug_out(df: pd.DataFrame) -> int:
    """Raise unless every drug belongs to exactly one fold. Returns the number of folds."""
    per_drug = df.groupby("drug_id")["fold"].nunique()
    if (per_drug > 1).any():
        raise ValueError(
            f"{int((per_drug > 1).sum())} of {len(per_drug)} drugs appear in more than one fold: this is not a "
            "leave-drug-out split, test drugs would be seen in training. Use the `fold` column of "
            "response_table_merged_leave_drug_out.csv (Zenodo) or build one with make_leave_drug_out_folds().")
    return int(df["fold"].nunique())


def make_leave_drug_out_folds(df: pd.DataFrame, n_folds: int = 10, seed: int = 42) -> pd.DataFrame:
    """Assign each *drug* (not each pair) to one of ``n_folds`` folds: KFold over the sorted drug list."""
    from sklearn.model_selection import KFold

    drugs = sorted(df["drug_id"].astype(str).unique())
    to_fold = {}
    for k, (_, test_idx) in enumerate(KFold(n_folds, shuffle=True, random_state=seed).split(drugs)):
        for i in test_idx:
            to_fold[drugs[i]] = k
    out = df.drop(columns=[c for c in ["fold"] if c in df.columns]).copy()
    out["fold"] = out["drug_id"].astype(str).map(to_fold)
    assert_leave_drug_out(out)
    return out


def table_fingerprint(df: pd.DataFrame) -> str:
    """MD5 of the (drug_id, fold) assignment; stored with results so stale results are never reused silently."""
    pairs = df[["drug_id", "fold"]].drop_duplicates().sort_values(["drug_id", "fold"])
    return hashlib.md5(pairs.to_csv(index=False).encode()).hexdigest()


# ----------------------------------------------------------------------------- drug features
class DrugFeatureStore:
    """Padded molecular graphs and fingerprints, computed once per drug."""

    def __init__(self, graphs: Optional[dict] = None, fp_df: Optional[pd.DataFrame] = None, max_atoms: int = MAX_ATOMS):
        self.graphs, self.fp_df, self.max_atoms = graphs, fp_df, max_atoms
        self._cache: Dict[str, Tuple[np.ndarray, ...]] = {}
        self.fp_dim = None if fp_df is None else int(fp_df.shape[1])

    def graph(self, drug_id: str):
        if drug_id not in self._cache:
            g = self.graphs[drug_id]
            nf = np.asarray(g["node_features"], dtype=np.float32)
            n = min(nf.shape[0], self.max_atoms)
            node = np.zeros((self.max_atoms, NODE_DIM), np.float32); node[:n] = nf[:n]
            mask = np.zeros((self.max_atoms, 1), np.float32); mask[:n] = 1.0
            adj = np.zeros((self.max_atoms, self.max_atoms), np.float32)
            edge = np.zeros((self.max_atoms, self.max_atoms, EDGE_DIM), np.float32)
            for (i, j), feat in zip(g["edge_index"], g["edge_features"]):
                if i < n and j < n:
                    adj[i, j] = 1.0
                    edge[i, j] = feat
            self._cache[drug_id] = (node, adj, edge, mask)
        return self._cache[drug_id]

    def fp(self, drug_id: str) -> np.ndarray:
        return self.fp_df.loc[drug_id].values.astype(np.float32)


# ----------------------------------------------------------------------------- drug encoders
class EdgeGatedGCNLayer(nn.Module):
    """Message passing with a learned scalar gate per bond (edge features -> sigmoid) and a residual connection."""

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = EDGE_DIM, dropout: float = 0.15):
        super().__init__()
        self.node_lin = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Sequential(nn.Linear(edge_dim, 1), nn.Sigmoid())
        self.ln = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, h, adj_mask, edge_feat, node_mask):
        w = self.edge_gate(edge_feat).squeeze(-1) * adj_mask             # (B,N,N)
        msg = torch.bmm(w, self.node_lin(h))                             # (B,N,out)
        out = self.dropout(F.gelu(self.ln(msg))) + self.residual(h)
        return out * node_mask


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, h, node_mask):
        logits = self.score(h).masked_fill(node_mask == 0, float("-inf"))
        return (F.softmax(logits, dim=1) * h).sum(dim=1)


class GCNDrugEncoder(nn.Module):
    """3 edge-gated GCN layers + attention pooling -> d_shared."""

    def __init__(self, d_shared: int = 512, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.l1 = EdgeGatedGCNLayer(NODE_DIM, hidden, dropout=dropout)
        self.l2 = EdgeGatedGCNLayer(hidden, hidden, dropout=dropout)
        self.l3 = EdgeGatedGCNLayer(hidden, hidden, dropout=dropout)
        self.pool = AttentionPool(hidden)
        self.proj = nn.Sequential(nn.Linear(hidden, d_shared), nn.LayerNorm(d_shared))

    def forward(self, node_feat, adj_mask, edge_feat, node_mask):
        h = self.l3(self.l2(self.l1(node_feat, adj_mask, edge_feat, node_mask), adj_mask, edge_feat, node_mask),
                    adj_mask, edge_feat, node_mask)
        return self.proj(self.pool(h, node_mask))


class FPDrugEncoder(nn.Module):
    """Residual MLP on MACCS (167) + ECFP4 (2048) fingerprints."""

    def __init__(self, fp_dim: int, d_shared: int = 512, hidden: int = 1024, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Linear(fp_dim, hidden)
        self.b1 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.b2 = nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.out = nn.Sequential(nn.Linear(hidden, d_shared), nn.LayerNorm(d_shared))

    def forward(self, fp):
        h = F.gelu(self.in_proj(fp))
        h = h + self.b1(h)
        h = h + self.b2(h)
        return self.out(h)


class EnsembleDrugEncoder(nn.Module):
    """Per-molecule gated mix of the GCN and fingerprint representations."""

    def __init__(self, fp_dim: int, d_shared: int = 512, dropout: float = 0.15):
        super().__init__()
        self.gcn = GCNDrugEncoder(d_shared, dropout=dropout)
        self.fp = FPDrugEncoder(fp_dim, d_shared, dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_shared, d_shared), nn.Sigmoid())
        self.out = nn.Sequential(nn.Linear(d_shared, d_shared), nn.LayerNorm(d_shared))

    def forward(self, node_feat, adj_mask, edge_feat, node_mask, fp):
        a, b = self.gcn(node_feat, adj_mask, edge_feat, node_mask), self.fp(fp)
        g = self.gate(torch.cat([a, b], dim=-1))
        return self.out(g * a + (1 - g) * b)


# ----------------------------------------------------------------------------- full model
class PredictionHead(nn.Module):
    def __init__(self, d_shared: int = 512, hidden: int = 256, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d_shared, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, cell, drug):
        return self.net(torch.cat([cell, drug], dim=-1)).squeeze(-1)


class DrugResponseModel(nn.Module):
    """``cell_encoder``: ``"heterosc"`` (adaptive fusion), ``"uniform"`` (alpha=1/K ablation) or the name of a single
    expert (baseline: Linear + LayerNorm projection, no router). ``drug_mode``: ``gcn`` | ``fp`` | ``ensemble``."""

    def __init__(self, cell_encoder: str, drug_mode: str, expert_dims: Dict[str, int],
                 fp_dim: Optional[int] = None, d_shared: int = 512):
        super().__init__()
        self.cell_encoder, self.drug_mode = cell_encoder, drug_mode
        if cell_encoder in ("heterosc", "uniform"):
            self.cell = HeteroSCFusion(expert_dims, d_shared, fusion="adaptive" if cell_encoder == "heterosc" else "uniform")
        elif cell_encoder in expert_dims:
            self.cell = nn.Sequential(nn.Linear(expert_dims[cell_encoder], d_shared), nn.LayerNorm(d_shared))
        else:
            raise ValueError(f"cell_encoder must be 'heterosc', 'uniform' or one of {list(expert_dims)}")
        if drug_mode == "gcn":
            self.drug = GCNDrugEncoder(d_shared)
        elif drug_mode == "fp":
            self.drug = FPDrugEncoder(fp_dim, d_shared)
        elif drug_mode == "ensemble":
            self.drug = EnsembleDrugEncoder(fp_dim, d_shared)
        else:
            raise ValueError(drug_mode)
        self.head = PredictionHead(d_shared)

    def forward(self, b: Dict[str, torch.Tensor]):
        if self.cell_encoder in ("heterosc", "uniform"):
            cell, alpha = self.cell({k[4:]: v for k, v in b.items() if k.startswith("emb_")})
        else:
            cell, alpha = self.cell(b[f"emb_{self.cell_encoder}"]), None
        if self.drug_mode == "gcn":
            drug = self.drug(b["node"], b["adj"], b["edge"], b["mask"])
        elif self.drug_mode == "fp":
            drug = self.drug(b["fp"])
        else:
            drug = self.drug(b["node"], b["adj"], b["edge"], b["mask"], b["fp"])
        return self.head(cell, drug), alpha


# ----------------------------------------------------------------------------- dataset
class PairDataset(Dataset):
    """Rows of the response table -> tensors (cell-line embeddings per expert + drug features + AUC)."""

    def __init__(self, df: pd.DataFrame, cell_emb: Dict[str, Dict[str, np.ndarray]], store: DrugFeatureStore, drug_mode: str):
        self.cell = df["cell_line_id"].astype(str).values
        self.drug = df["drug_id"].astype(str).values
        self.auc = df["auc"].values.astype(np.float32)
        self.cell_emb, self.store, self.drug_mode = cell_emb, store, drug_mode

    def __len__(self):
        return len(self.auc)

    def __getitem__(self, i):
        return self.cell[i], self.drug[i], self.auc[i]

    def collate(self, batch):
        cells, drugs, auc = zip(*batch)
        out = {f"emb_{n}": torch.tensor(np.stack([e[c] for c in cells])) for n, e in self.cell_emb.items()}
        out["auc"] = torch.tensor(np.array(auc))
        if self.drug_mode in ("gcn", "ensemble"):
            g = [self.store.graph(d) for d in drugs]
            for j, key in enumerate(["node", "adj", "edge", "mask"]):
                out[key] = torch.tensor(np.stack([x[j] for x in g]))
        if self.drug_mode in ("fp", "ensemble"):
            out["fp"] = torch.tensor(np.stack([self.store.fp(d) for d in drugs]))
        out["_drugs"] = list(drugs)
        return out


# ----------------------------------------------------------------------------- metrics
def compute_metrics(pred_df: pd.DataFrame) -> Dict[str, float]:
    """Pooled PCC/SRCC/MSE and the primary metric: mean per-drug PCC across cell lines (drugs with >= 3 pairs)."""
    t, p = pred_df["true_auc"].values, pred_df["pred_auc"].values
    per_drug = []
    for _, g in pred_df.groupby("drug_id"):
        if len(g) >= 3:
            r = pearsonr(g["true_auc"], g["pred_auc"])[0]
            if not np.isnan(r):
                per_drug.append(r)
    return {"pcc_pooled": float(pearsonr(t, p)[0]), "srcc_pooled": float(spearmanr(t, p)[0]),
            "mse": float(mean_squared_error(t, p)),
            "pcc_per_drug": float(np.mean(per_drug)) if per_drug else float("nan"),
            "n_drugs_evaluated": len(per_drug)}


# ----------------------------------------------------------------------------- training
@dataclass
class DrugResponseConfig:
    cell_encoder: str = "heterosc"           # heterosc | uniform | geneformer | scgpt | cellfm | scfoundation
    drug_mode: str = "gcn"                   # gcn | fp | ensemble
    d_shared: int = 512
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    entropy_lambda: float = 0.05             # lambda_H in Eq. 11 (adaptive routing only)
    val_drug_fraction: float = 0.10          # drugs held out of training for early stopping
    seed: int = 42
    device: str = "auto"
    expert_dims: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_EXPERT_DIMS))


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _run_epoch(model, loader, cfg, device, optimizer=None, entropy_lambda=0.0):
    train = optimizer is not None
    model.train(train)
    rows, total = [], 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            b = _to_device(batch, device)
            pred, alpha = model(b)
            loss = F.mse_loss(pred, b["auc"])
            if alpha is not None and entropy_lambda > 0:
                loss = loss + entropy_lambda * routing_entropy(alpha)       # Eq. 10-11
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            total += loss.item() * len(pred)
            rows.extend(zip(batch["_drugs"], b["auc"].cpu().numpy(), pred.detach().cpu().numpy()))
    df = pd.DataFrame(rows, columns=["drug_id", "true_auc", "pred_auc"])
    m = compute_metrics(df); m["loss"] = total / len(df)
    return m, df


def train_fold(cfg: DrugResponseConfig, fold: int, response_df: pd.DataFrame,
               cell_emb: Dict[str, Dict[str, np.ndarray]], store: DrugFeatureStore,
               out_dir: Optional[Path] = None, overwrite: bool = False, verbose: bool = True) -> Dict:
    """Train one leave-drug-out fold and evaluate on its held-out drugs.

    The validation set is a random 10% of the *training drugs* (so early stopping also measures generalisation to
    unseen drugs). The test fold is never used for model selection.
    """
    assert_leave_drug_out(response_df)
    fp = table_fingerprint(response_df)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        mpath = out_dir / f"fold{fold}_test_metrics.json"
        if mpath.exists() and not overwrite:
            old = json.loads(mpath.read_text())
            if old.get("fold_table_md5") == fp:
                if verbose:
                    print(f"[fold {fold}] reusing finished result (same fold table)")
                return old
            if verbose:
                print(f"[fold {fold}] existing result was made with a DIFFERENT fold table -> recomputing")
    device = get_device(cfg.device)
    set_seed(cfg.seed + fold)
    test_df = response_df[response_df["fold"] == fold]
    trainval = response_df[response_df["fold"] != fold]
    drugs = np.array(sorted(trainval["drug_id"].unique()))
    val_drugs = np.random.RandomState(cfg.seed + fold).choice(drugs, max(1, int(cfg.val_drug_fraction * len(drugs))), replace=False)
    val_df = trainval[trainval["drug_id"].isin(val_drugs)]
    train_df = trainval[~trainval["drug_id"].isin(val_drugs)]
    assert not (set(test_df["drug_id"]) & set(train_df["drug_id"])), "drug leakage between train and test"

    names = list(cfg.expert_dims)
    mk = lambda d, shuffle: (lambda ds: DataLoader(ds, cfg.batch_size, shuffle=shuffle, collate_fn=ds.collate))(
        PairDataset(d, {n: cell_emb[n] for n in names}, store, cfg.drug_mode))
    tr, va, te = mk(train_df, True), mk(val_df, False), mk(test_df, False)
    model = DrugResponseModel(cfg.cell_encoder, cfg.drug_mode, cfg.expert_dims, store.fp_dim, cfg.d_shared).to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: warmup_cosine(e, 1.0, 0.0, cfg.warmup_epochs, cfg.max_epochs))
    lam = cfg.entropy_lambda if cfg.cell_encoder == "heterosc" else 0.0
    if verbose:
        print(f"[fold {fold}] {cfg.cell_encoder}|{cfg.drug_mode}  train {len(train_df):,} / val {len(val_df):,} / test {len(test_df):,} pairs "
              f"({train_df.drug_id.nunique()} / {val_df.drug_id.nunique()} / {test_df.drug_id.nunique()} drugs)")
    best, best_state, bad, hist, t0 = -np.inf, None, 0, [], time.time()
    for ep in range(cfg.max_epochs):
        trm, _ = _run_epoch(model, tr, cfg, device, opt, lam)
        vam, _ = _run_epoch(model, va, cfg, device, None, lam)
        sched.step()
        hist.append({"epoch": ep + 1, "train_loss": trm["loss"], "val_loss": vam["loss"], "val_pcc_per_drug": vam["pcc_per_drug"]})
        if vam["pcc_per_drug"] > best:
            best, bad = vam["pcc_per_drug"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and ((ep + 1) % 5 == 0 or bad == 0):
            print(f"  ep {ep + 1:3d} train {trm['loss']:.4f} val {vam['loss']:.4f} val per-drug PCC {vam['pcc_per_drug']:.4f} (best {best:.4f}) {time.time() - t0:.0f}s")
        if bad >= cfg.patience:
            break
    model.load_state_dict(best_state)
    tm, preds = _run_epoch(model, te, cfg, device, None, 0.0)
    if verbose:
        print(f"[fold {fold}] TEST per-drug PCC {tm['pcc_per_drug']:.4f} | pooled PCC {tm['pcc_pooled']:.4f} | MSE {tm['mse']:.4f}")
    result = {**tm, "fold": fold, "fold_table_md5": fp, "cell_encoder": cfg.cell_encoder, "drug_mode": cfg.drug_mode,
              "n_test_drugs": int(test_df["drug_id"].nunique()), "best_val_pcc_per_drug": float(best)}
    if out_dir is not None:
        torch.save(model.state_dict(), out_dir / f"fold{fold}_model.pt")
        pd.DataFrame(hist).to_csv(out_dir / f"fold{fold}_history.csv", index=False)
        preds.to_csv(out_dir / f"fold{fold}_predictions.csv", index=False)
        (out_dir / f"fold{fold}_test_metrics.json").write_text(json.dumps(result, indent=2))
    return result


def run_cv(cfg: DrugResponseConfig, response_df, cell_emb, store, out_dir=None, folds: Optional[Sequence[int]] = None,
           overwrite: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Leave-drug-out cross-validation over ``folds`` (default: all). Returns one row per fold."""
    n = assert_leave_drug_out(response_df)
    rows = [train_fold(cfg, f, response_df, cell_emb, store, out_dir, overwrite, verbose) for f in (folds if folds is not None else range(n))]
    return pd.DataFrame(rows)


@torch.no_grad()
def collect_routing(model: DrugResponseModel, df: pd.DataFrame, cell_emb, store, cfg: DrugResponseConfig) -> pd.DataFrame:
    """Routing weights per cell-line/drug pair (adaptive or uniform HeteroSC models only)."""
    device = get_device(cfg.device)
    ds = PairDataset(df, {n: cell_emb[n] for n in cfg.expert_dims}, store, cfg.drug_mode)
    model.eval().to(device)
    alphas = []
    for batch in DataLoader(ds, cfg.batch_size, collate_fn=ds.collate):
        _, a = model(_to_device(batch, device))
        alphas.append(a.cpu().numpy())
    out = pd.DataFrame(np.concatenate(alphas), columns=list(cfg.expert_dims))
    out.insert(0, "cell_line_id", ds.cell); out.insert(1, "drug_id", ds.drug)
    return out
