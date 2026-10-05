"""Tests for expert alignment and the zero-shot kNN probe."""
import numpy as np
import pandas as pd
import pytest

from heterosc.data import align_experts
from heterosc.knn import knn_probe


class _Knn:                      # keeps the test bodies below unchanged
    knn_probe = staticmethod(knn_probe)


class _Data:
    align_experts = staticmethod(align_experts)


knn, data = _Knn, _Data


def _expert(ids, labels, dim, seed):
    rng = np.random.default_rng(seed)
    obs = pd.DataFrame({"cell_id": ids, "cell_type": labels, "master_split": "train"})
    return rng.normal(size=(len(ids), dim)).astype(np.float32), obs


def test_align_keeps_only_common_cells_in_common_order():
    labels = {f"c{i}": "A" if i % 2 else "B" for i in range(6)}
    ids_a = [f"c{i}" for i in range(6)]
    ids_b = ["c5", "c3", "c0", "c1", "c2"]                      # shuffled, c4 missing
    per = {"a": _expert(ids_a, [labels[i] for i in ids_a], 4, 0),
           "b": _expert(ids_b, [labels[i] for i in ids_b], 7, 1)}
    emb, obs = data.align_experts(per)
    assert list(obs["cell_id"]) == ["c0", "c1", "c2", "c3", "c5"]
    assert emb["a"].shape == (5, 4) and emb["b"].shape == (5, 7)
    np.testing.assert_allclose(emb["b"][0], per["b"][0][2])      # c0 sits at row 2 in expert b


def test_align_rejects_label_mismatch():
    ids = ["c0", "c1"]
    per = {"a": _expert(ids, ["A", "B"], 3, 0), "b": _expert(ids, ["A", "A"], 3, 1)}
    with pytest.raises(ValueError):
        data.align_experts(per)


def test_knn_probe_separable_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[5, 0], [0, 5], [-5, -5]], dtype=float)
    X = np.concatenate([c + rng.normal(scale=0.3, size=(40, 2)) for c in centers])
    y = np.repeat([0, 1, 2], 40)
    idx = rng.permutation(len(y))
    tr, te = idx[:90], idx[90:]
    r = knn.knn_probe(X[tr], y[tr], X[te], y[te])
    assert r["accuracy"] > 0.95 and r["n_neighbors"] == 5 and r["metric"] == "cosine"
