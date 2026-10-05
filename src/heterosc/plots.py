"""Small plotting helpers for routing diagnostics (matplotlib only)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_routing_by_group(alpha: np.ndarray, groups: Sequence, expert_names: Sequence[str], group_names: Optional[Sequence[str]] = None,
                          ax=None, title: str = "Mean routing weight per group"):
    """Grouped bars of the mean routing weight of each expert within each group (cell type, dataset, ...).
    The dashed line is the uniform weight 1/K."""
    plt = _plt()
    alpha, groups = np.asarray(alpha), np.asarray(groups)
    labels = list(group_names) if group_names is not None else sorted(set(groups.tolist()))
    keys = sorted(set(groups.tolist()))
    means = np.array([alpha[groups == k].mean(0) for k in keys])
    if ax is None:
        _, ax = plt.subplots(figsize=(max(5, 0.9 * len(keys) + 2), 3.5))
    w = 0.8 / len(expert_names)
    for j, e in enumerate(expert_names):
        ax.bar(np.arange(len(keys)) + j * w, means[:, j], width=w, label=e)
    ax.axhline(1 / len(expert_names), ls="--", c="k", lw=1, label="uniform")
    ax.set_xticks(np.arange(len(keys)) + 0.4 - w / 2); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean routing weight"); ax.set_title(title); ax.legend(fontsize=8, ncol=3, frameon=False)
    return ax


def plot_training_history(history, columns: Sequence[str] = ("train_loss", "val_loss"), ax=None, title: str = "Training history"):
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    for c in columns:
        if c in history:
            ax.plot(history["epoch"], history[c], label=c)
    ax.set_xlabel("epoch"); ax.set_title(title); ax.legend(frameon=False)
    return ax


def plot_metric_comparison(results: Dict[str, Sequence[float]], metric_name: str = "metric", ax=None, title: Optional[str] = None):
    """Box + points of a metric per model across folds/seeds (e.g. per-drug PCC over the 10 leave-drug-out folds)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, 1.2 * len(results)), 3.5))
    names = list(results)
    ax.boxplot([results[n] for n in names], labels=names, showmeans=True)
    for i, n in enumerate(names):
        ax.scatter(np.full(len(results[n]), i + 1) + np.random.default_rng(0).normal(0, 0.04, len(results[n])), results[n], s=12, c="k", alpha=.6, zorder=3)
    ax.set_ylabel(metric_name)
    if title:
        ax.set_title(title)
    return ax
