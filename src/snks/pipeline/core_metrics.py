"""Compact, seed-aware metrics for learning-core hypothesis tests."""

from __future__ import annotations

import numpy as np


def normalized_auc(steps: list[float], scores: list[float]) -> float:
    """Return trapezoidal area divided by the observed step span."""
    x = np.asarray(steps, dtype=float)
    y = np.asarray(scores, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("steps and scores need equally sized sequences of at least two points")
    span = x[-1] - x[0]
    if span <= 0:
        raise ValueError("steps must be strictly increasing overall")
    if np.any(np.diff(x) <= 0):
        raise ValueError("steps must be strictly increasing")
    return float(np.trapezoid(y, x) / span)


def paired_cluster_interval(
    left: list[float], right: list[float], seed: int, n_boot: int = 10_000, alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap a paired per-training-seed difference interval."""
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if len(a) != len(b) or len(a) < 5:
        raise ValueError("paired bootstrap requires at least 5 training seeds")
    if n_boot < 1 or not 0 < alpha < 1:
        raise ValueError("n_boot must be positive and alpha must be in (0, 1)")
    differences = a - b
    rng = np.random.default_rng(seed)
    draws = differences[rng.integers(0, len(differences), size=(n_boot, len(differences)))].mean(axis=1)
    return tuple(float(item) for item in np.quantile(draws, (alpha / 2, 1 - alpha / 2)))


def prediction_metrics(records: list[dict[str, float]]) -> dict[str, float]:
    """Average named scalar prediction losses from already-reduced records."""
    if not records:
        return {}
    keys = set().union(*(record.keys() for record in records))
    return {key: float(np.mean([record[key] for record in records if key in record])) for key in sorted(keys)}
