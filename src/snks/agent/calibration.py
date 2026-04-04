"""Stage 65: Calibration tracker for uncertainty measurement.

Collects (confidence, was_correct) pairs and computes calibration metrics:
- Brier score (mean squared error between confidence and correctness)
- Calibration curve (accuracy per confidence bucket)
- Spearman correlation (confidence ~ correctness monotonic relationship)
"""

from __future__ import annotations

import math


class CalibrationTracker:
    """Tracks prediction confidence vs actual correctness."""

    def __init__(self, n_buckets: int = 5):
        self.predictions: list[tuple[float, bool]] = []
        self.n_buckets = n_buckets

    def record(self, confidence: float, predicted: str, actual: str) -> None:
        correct = predicted == actual
        self.predictions.append((confidence, correct))

    def brier_score(self) -> float:
        """Mean squared error between confidence and correctness.

        Perfect calibration → 0.0. Random → 0.25. Always wrong at conf=1 → 1.0.
        """
        if not self.predictions:
            return 1.0
        return sum((c - int(ok)) ** 2
                   for c, ok in self.predictions) / len(self.predictions)

    def calibration_curve(self) -> list[tuple[float, float, int]]:
        """Returns (mean_confidence, accuracy, count) per bucket."""
        buckets: list[list[tuple[float, bool]]] = [
            [] for _ in range(self.n_buckets)
        ]
        for conf, correct in self.predictions:
            idx = min(int(conf * self.n_buckets), self.n_buckets - 1)
            buckets[idx].append((conf, correct))

        curve = []
        for bucket in buckets:
            if bucket:
                mean_conf = sum(c for c, _ in bucket) / len(bucket)
                accuracy = sum(int(ok) for _, ok in bucket) / len(bucket)
                curve.append((mean_conf, accuracy, len(bucket)))
        return curve

    def correlation(self) -> float:
        """Spearman rank correlation between confidence and correctness."""
        if len(self.predictions) < 10:
            return 0.0
        try:
            from scipy.stats import spearmanr
            confs = [c for c, _ in self.predictions]
            correct = [int(ok) for _, ok in self.predictions]
            rho, _ = spearmanr(confs, correct)
            return rho if rho == rho else 0.0  # NaN guard
        except ImportError:
            # Fallback: point-biserial approximation
            return self._fallback_correlation()

    def _fallback_correlation(self) -> float:
        """Simple correlation without scipy."""
        n = len(self.predictions)
        if n < 10:
            return 0.0
        confs = [c for c, _ in self.predictions]
        correct = [int(ok) for _, ok in self.predictions]
        mean_c = sum(confs) / n
        mean_ok = sum(correct) / n
        cov = sum((c - mean_c) * (ok - mean_ok)
                  for c, ok in zip(confs, correct)) / n
        std_c = math.sqrt(sum((c - mean_c) ** 2 for c in confs) / n)
        std_ok = math.sqrt(sum((ok - mean_ok) ** 2 for ok in correct) / n)
        if std_c < 1e-9 or std_ok < 1e-9:
            return 0.0
        return cov / (std_c * std_ok)

    def summary(self) -> dict:
        return {
            "n_predictions": len(self.predictions),
            "brier_score": round(self.brier_score(), 4),
            "correlation": round(self.correlation(), 4),
            "calibration_curve": [
                {"conf": round(c, 2), "acc": round(a, 2), "n": n}
                for c, a, n in self.calibration_curve()
            ],
        }
