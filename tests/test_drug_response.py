import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from heterosc.drug_response import (DrugFeatureStore, DrugResponseConfig, DrugResponseModel, EDGE_DIM, NODE_DIM,
                                    assert_leave_drug_out, collect_routing, make_leave_drug_out_folds, run_cv,
                                    table_fingerprint, train_fold)

DIMS = {"a": 24, "b": 16, "c": 40}


def _toy(n_drugs=30, n_cells=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = [(f"c{c}", f"d{d}", float(rng.random())) for d in range(n_drugs) for c in range(n_cells)]
    df = pd.DataFrame(rows, columns=["cell_line_id", "drug_id", "auc"])
    df = make_leave_drug_out_folds(df, n_folds=5, seed=1)
    graphs = {}
    for d in range(n_drugs):
        n = int(rng.integers(5, 12))
        ei = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
        graphs[f"d{d}"] = {"node_features": rng.normal(size=(n, NODE_DIM)), "edge_index": ei,
                           "edge_features": rng.normal(size=(len(ei), EDGE_DIM))}
    fp = pd.DataFrame(rng.integers(0, 2, (n_drugs, 50)), index=[f"d{d}" for d in range(n_drugs)])
    emb = {k: {f"c{c}": rng.normal(size=dim).astype(np.float32) for c in range(n_cells)} for k, dim in DIMS.items()}
    return df, DrugFeatureStore(graphs, fp), emb


def test_leave_drug_out_guard():
    df, _, _ = _toy()
    assert assert_leave_drug_out(df) == 5
    bad = df.copy(); bad["fold"] = np.random.default_rng(0).integers(0, 5, len(bad))      # random per pair
    with pytest.raises(ValueError, match="not a leave-drug-out"):
        assert_leave_drug_out(bad)


def test_folds_are_per_drug_and_balanced():
    df, _, _ = _toy(n_drugs=50)
    df = make_leave_drug_out_folds(df, n_folds=10)
    assert df.groupby("drug_id")["fold"].nunique().max() == 1
    assert sorted(df["fold"].unique()) == list(range(10))
    assert df.groupby("fold")["drug_id"].nunique().between(5, 5).all()


@pytest.mark.parametrize("cell_enc,drug_mode", [("heterosc", "gcn"), ("uniform", "fp"), ("heterosc", "ensemble"), ("a", "gcn")])
def test_model_forward_all_modes(cell_enc, drug_mode):
    df, store, emb = _toy()
    cfg = DrugResponseConfig(cell_encoder=cell_enc, drug_mode=drug_mode, expert_dims=DIMS, d_shared=16, batch_size=8, device="cpu")
    from heterosc.drug_response import PairDataset
    ds = PairDataset(df.head(8), emb, store, drug_mode)
    batch = ds.collate([ds[i] for i in range(8)])
    m = DrugResponseModel(cell_enc, drug_mode, DIMS, store.fp_dim, 16)
    pred, alpha = m({k: v for k, v in batch.items() if torch.is_tensor(v)})
    assert pred.shape == (8,)
    assert (alpha is None) == (cell_enc not in ("heterosc", "uniform"))


def test_train_fold_no_leakage_and_result_reuse(tmp_path):
    df, store, emb = _toy()
    cfg = DrugResponseConfig(expert_dims=DIMS, d_shared=16, batch_size=64, max_epochs=3, patience=3, warmup_epochs=1, device="cpu")
    r = train_fold(cfg, 0, df, emb, store, tmp_path, verbose=False)
    assert r["n_test_drugs"] == df[df.fold == 0].drug_id.nunique()
    assert (tmp_path / "fold0_predictions.csv").exists()
    again = train_fold(cfg, 0, df, emb, store, tmp_path, verbose=False)                # same table -> reused
    assert again["fold_table_md5"] == r["fold_table_md5"]
    df2 = make_leave_drug_out_folds(df, n_folds=5, seed=999)                          # different fold table
    assert table_fingerprint(df2) != table_fingerprint(df)
    r2 = train_fold(cfg, 0, df2, emb, store, tmp_path, verbose=False)                 # must recompute, not reuse
    assert r2["fold_table_md5"] == table_fingerprint(df2)


def test_run_cv_and_routing(tmp_path):
    df, store, emb = _toy()
    cfg = DrugResponseConfig(expert_dims=DIMS, d_shared=16, batch_size=64, max_epochs=2, patience=2, warmup_epochs=1, device="cpu")
    res = run_cv(cfg, df, emb, store, folds=[0, 1], verbose=False)
    assert len(res) == 2 and {"pcc_per_drug", "pcc_pooled", "mse"} <= set(res.columns)
    m = DrugResponseModel("heterosc", "gcn", DIMS, store.fp_dim, 16)
    a = collect_routing(m, df.head(40), emb, store, cfg)
    assert np.allclose(a[list(DIMS)].sum(axis=1), 1.0, atol=1e-5)
