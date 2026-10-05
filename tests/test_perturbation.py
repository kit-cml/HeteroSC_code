import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from heterosc.perturbation import (HeteroSCGeneEmbedding, align_gene_table, attach_heterosc_to_gears,
                                   expert_combinations, export_fused_matrix)


def test_align_by_identity_not_position():
    table = {"B": np.array([1.0, 1.0]), "A": np.array([2.0, 2.0])}              # vocabulary order differs from GEARS order
    mat, found = align_gene_table(["A", "X", "B"], table, dim=2)
    assert torch.allclose(mat[0], torch.tensor([2.0, 2.0])) and torch.allclose(mat[2], torch.tensor([1.0, 1.0]))
    assert found.tolist() == [True, False, True]                                  # X is random-initialised, flagged missing


def test_align_with_key_function():
    mat, found = align_gene_table(["TP53"], {"ENSG1": np.ones(3)}, 3, key_fn={"TP53": "ENSG1"}.get)
    assert found[0] and torch.allclose(mat[0], torch.ones(3))


def test_ladder_has_15_combinations():
    c = expert_combinations(["a", "b", "c", "d"])
    assert len(c) == 15 and len({tuple(x) for x in c}) == 15 and len(c[0]) == 1 and len(c[-1]) == 4


def _tables(n=50):
    return {"a": torch.randn(n, 24), "b": torch.randn(n, 16), "c": torch.randn(n, 40)}


def test_embedding_matches_nn_embedding_interface_and_trains_adapters_only():
    m = HeteroSCGeneEmbedding(_tables(), d_shared=32, d_router_hidden=16)
    idx = torch.arange(50).repeat(3)                                              # GEARS calls it with arange(num_genes).repeat(B)
    out = m(idx)
    assert out.shape == (150, 32) and out.norm(dim=1).max() <= 1.0 + 1e-4         # max_norm respected
    out.sum().backward()
    assert all(p.grad is not None for p in m.parameters())
    assert not any(n.startswith("raw_") for n, _ in m.named_parameters())         # FM tables are buffers, never trained


def test_attach_freezes_everything_but_gene_emb():
    class FakeGears(nn.Module):
        def __init__(self):
            super().__init__(); self.gene_emb = nn.Embedding(50, 32); self.other = nn.Linear(32, 4)

    class Wrapper:
        def __init__(self): self.model = FakeGears()

    g = Wrapper()
    mod = attach_heterosc_to_gears(g, _tables(), d_shared=32, device="cpu", d_router_hidden=16)
    trainable = {n for n, p in g.model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("gene_emb.") for n in trainable)
    assert not g.model.other.weight.requires_grad and g.model.gene_emb is mod


def test_export_fused_matrix_and_uniform_alpha():
    m = HeteroSCGeneEmbedding(_tables(), d_shared=32, d_router_hidden=16, fusion="uniform")
    h, a = export_fused_matrix(m)
    assert h.shape == (50, 32) and torch.allclose(a, torch.full_like(a, 1 / 3))
