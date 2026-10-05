"""Loading and aligning per-expert embeddings.

Convention (same as the original notebooks): every expert writes
``{root}/{dataset}/{expert}/{dataset}_{expert}_with_embeddings.h5ad`` where ``adata.X`` is the
(N_cells x embedding_dim) embedding and ``adata.obs`` has ``cell_id``, ``cell_type`` and
``master_split`` (train / val / test).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def load_expert_h5ad(root: Path, dataset: str, expert: str) -> Tuple[np.ndarray, pd.DataFrame]:
    import anndata as ad

    path = Path(root) / dataset / expert / f"{dataset}_{expert}_with_embeddings.h5ad"
    if not path.exists():
        raise FileNotFoundError(f"[{dataset}/{expert}] embedding file not found: {path}")
    adata = ad.read_h5ad(path)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    obs = adata.obs.copy()
    if "cell_id" not in obs.columns:
        obs["cell_id"] = obs.index.astype(str)
    obs["cell_id"] = obs["cell_id"].astype(str)
    return X.astype(np.float32), obs


def align_experts(
    per_expert: Dict[str, Tuple[np.ndarray, pd.DataFrame]],
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Keep only cells with a valid embedding from *all* experts, in a common order.

    Matches manuscript Section 2.2: "restricted to cells for which valid embeddings were
    available from all four models before HeteroSC fusion".
    """
    names = list(per_expert.keys())
    common = None
    for n in names:
        ids = set(per_expert[n][1]["cell_id"])
        common = ids if common is None else common & ids
    first_obs = per_expert[names[0]][1]
    ordered = [c for c in first_obs["cell_id"] if c in common]
    if not ordered:
        raise ValueError("No cell_id is shared by all experts.")

    embeddings: Dict[str, np.ndarray] = {}
    for n in names:
        X, obs = per_expert[n]
        pos = pd.Series(np.arange(len(obs)), index=obs["cell_id"].values)
        if pos.index.has_duplicates:
            raise ValueError(f"Duplicate cell_id in expert '{n}'.")
        embeddings[n] = X[pos.loc[ordered].values]

    obs0 = first_obs.set_index("cell_id").loc[ordered].reset_index()
    for n in names[1:]:
        o = per_expert[n][1].set_index("cell_id").loc[ordered]
        if not (o["cell_type"].astype(str).values == obs0["cell_type"].astype(str).values).all():
            raise ValueError(f"cell_type labels disagree between '{names[0]}' and '{n}'.")
    return embeddings, obs0


def load_dataset(
    root: Path, dataset: str, experts: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str]]:
    """Load + align all experts for one dataset.

    Returns ``(embeddings, labels, split, class_names)`` with integer labels
    (alphabetical class order) and ``split`` in {train, val, test}.
    """
    per = {e: load_expert_h5ad(root, dataset, e) for e in experts}
    emb, obs = align_experts(per)
    classes = sorted(obs["cell_type"].astype(str).unique().tolist())
    c2i = {c: i for i, c in enumerate(classes)}
    labels = obs["cell_type"].astype(str).map(c2i).values.astype(np.int64)
    split = obs["master_split"].astype(str).values
    return emb, labels, split, classes


from .utils import l2_normalize  # noqa: E402,F401  (re-exported for backwards compatibility)


# ----------------------------------------------------------------------------- drug response tables
def load_drug_tables(data_dir: Path):
    """Load the drug-response tables from the Zenodo ``drug_response`` folder.

    Returns ``(response_df, drug_graphs, fp_df)``. ``response_df`` comes from the leave-drug-out table.
    """
    import pickle

    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "response_table_merged_leave_drug_out.csv", dtype={"drug_id": str, "cell_line_id": str})
    with open(data_dir / "drug_graph_features_merged.pkl", "rb") as f:        # trusted source only (pickle)
        graphs = pickle.load(f)
    fp = pd.read_csv(data_dir / "drug_fingerprint_merged.csv", dtype={"drug_id": str}).set_index("drug_id")
    return df, graphs, fp


def load_cell_line_embeddings(root: Path, experts: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Cell-line embeddings written by the extraction functions:
    ``{root}/{expert}/cellline_{expert}_embeddings.h5ad`` -> ``{expert: {cell_line_id: vector}}``."""
    import anndata as ad

    out = {}
    for e in experts:
        a = ad.read_h5ad(Path(root) / e / f"cellline_{e}_embeddings.h5ad")
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        out[e] = dict(zip(a.obs["cell_id"].astype(str), X.astype(np.float32)))
    return out


def keep_complete_cell_lines(response_df: pd.DataFrame, cell_emb: Dict[str, Dict[str, np.ndarray]]) -> pd.DataFrame:
    """Keep only pairs whose cell line has an embedding from every expert."""
    common = set.intersection(*[set(d) for d in cell_emb.values()])
    return response_df[response_df["cell_line_id"].astype(str).isin(common)].reset_index(drop=True)
