"""Zero-shot kNN evaluation in a frozen representation space (manuscript Section 2.3).

Five nearest neighbours, cosine distance, distance-weighted voting, no parametric head.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier


def knn_probe(emb_train: np.ndarray, y_train: np.ndarray,
              emb_test: np.ndarray, y_test: np.ndarray, n_neighbors: int = 5) -> Dict[str, float]:
    k = min(n_neighbors, len(y_train))
    knn = KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance")
    knn.fit(emb_train, y_train)
    pred = knn.predict(emb_test)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "n_neighbors": int(k),
        "metric": "cosine",
    }
