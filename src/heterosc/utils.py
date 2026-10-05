"""Small shared helpers."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (applied to every expert's embedding before fusion)."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def warmup_cosine(epoch: int, base_lr: float, min_lr: float, warmup: int, total: int) -> float:
    """Linear warm-up followed by cosine decay (``epoch`` is 0-based)."""
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + float(np.cos(np.pi * progress)))
