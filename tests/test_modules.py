"""Shape / property tests for the PyTorch modules (require torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heterosc import (AnnotationConfig, HeteroSCClassifier, HeteroSCFusion, diversity_loss,
                      normalized_entropy, routing_entropy)

DIMS = {"geneformer": 768, "scgpt": 512, "cellfm": 1536, "scfoundation": 768}


def _batch(n=16):
    return {k: torch.randn(n, d) for k, d in DIMS.items()}


def test_fusion_shapes_and_simplex():
    f = HeteroSCFusion(DIMS, d_shared=64, d_router_hidden=32).eval()
    h, a = f(_batch())
    assert h.shape == (16, 64) and a.shape == (16, 4)
    assert torch.allclose(a.sum(-1), torch.ones(16), atol=1e-5) and (a >= 0).all()


def test_uniform_fusion_is_exactly_quarter_and_has_no_router():
    f = HeteroSCFusion(DIMS, d_shared=64, fusion="uniform").eval()
    _, a = f(_batch())
    assert f.router is None
    assert torch.allclose(a, torch.full_like(a, 0.25))


def test_router_starts_near_uniform():
    f = HeteroSCFusion(DIMS, d_shared=64, d_router_hidden=32).eval()
    _, a = f(_batch(64))
    assert (a.mean(0) - 0.25).abs().max() < 0.1


def test_diversity_loss_minimised_when_balanced():
    balanced = torch.full((8, 4), 0.25)
    collapsed = torch.tensor([[1.0, 0, 0, 0]] * 8)
    assert diversity_loss(balanced) < diversity_loss(collapsed)
    assert abs(diversity_loss(balanced).item() + np.log(4)) < 1e-4     # = -log K


def test_routing_entropy_per_input():
    uniform = torch.full((8, 4), 0.25)
    peaked = torch.tensor([[0.97, 0.01, 0.01, 0.01]] * 8)
    assert routing_entropy(peaked) < routing_entropy(uniform)


def test_normalized_entropy_bounds():
    assert abs(normalized_entropy([0.25] * 4) - 1.0) < 1e-6
    assert normalized_entropy([1, 0, 0, 0]) < 1e-6


def test_classifier_forward_and_backward():
    cfg = AnnotationConfig(expert_dims=DIMS, num_classes=5, d_shared=64, d_router_hidden=32,
                           d_predictor_hidden=64, device="cpu")
    m = HeteroSCClassifier(cfg)
    logits, alpha, h = m(_batch())
    assert logits.shape == (16, 5) and alpha.shape == (16, 4) and h.shape == (16, 64)
    (logits.sum() + diversity_loss(alpha)).backward()
    assert all(p.grad is not None for p in m.fusion.router.parameters())


def test_normalized_entropy_single_expert_is_nan():
    assert np.isnan(normalized_entropy([1.0]))


def test_single_expert_classifier_runs():
    cfg = AnnotationConfig(expert_dims={"a": 20}, num_classes=3, fusion="uniform", d_shared=16, d_router_hidden=8,
                           d_predictor_hidden=16, device="cpu", epochs=2, warmup_epochs=1)
    m = HeteroSCClassifier(cfg)
    logits, alpha, _ = m({"a": torch.randn(5, 20)})
    assert logits.shape == (5, 3) and torch.allclose(alpha, torch.ones(5, 1))
