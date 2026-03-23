"""Tests for SKS metrics (Stage 3)."""

import numpy as np
import torch
import pytest

from snks.sks.metrics import compute_nmi, sks_stability, sks_separability


class TestComputeNMI:
    """compute_nmi — Normalized Mutual Information."""

    def test_perfect_match(self) -> None:
        """Identical labels → NMI = 1.0."""
        labels = np.array([0, 0, 1, 1, 2, 2])
        assert compute_nmi(labels, labels) == pytest.approx(1.0)

    def test_permuted_labels(self) -> None:
        """Permuted labels (same clustering) → NMI = 1.0."""
        pred = np.array([0, 0, 1, 1, 2, 2])
        true = np.array([2, 2, 0, 0, 1, 1])
        assert compute_nmi(pred, true) == pytest.approx(1.0)

    def test_random_low(self) -> None:
        """Uncorrelated labels → NMI ≈ 0."""
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 5, size=1000)
        true = rng.randint(0, 5, size=1000)
        assert compute_nmi(pred, true) < 0.1

    def test_single_cluster(self) -> None:
        """All same label → NMI = 0 (no information)."""
        pred = np.array([0, 0, 0, 0])
        true = np.array([0, 1, 0, 1])
        assert compute_nmi(pred, true) == pytest.approx(0.0)


class TestSKSStability:
    """sks_stability — Jaccard-based stability."""

    def test_identical(self) -> None:
        """Same clusters → stability = 1.0."""
        c = [{0, 1, 2}, {3, 4, 5}]
        assert sks_stability(c, c) == pytest.approx(1.0)

    def test_disjoint(self) -> None:
        """Completely different clusters → stability = 0.0."""
        c1 = [{0, 1, 2}]
        c2 = [{3, 4, 5}]
        assert sks_stability(c1, c2) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """Partial overlap → 0 < stability < 1."""
        c1 = [{0, 1, 2, 3}]
        c2 = [{2, 3, 4, 5}]
        s = sks_stability(c1, c2)
        assert 0.0 < s < 1.0

    def test_empty_returns_zero(self) -> None:
        """Empty input → 0."""
        assert sks_stability([], [{0, 1}]) == pytest.approx(0.0)
        assert sks_stability([{0, 1}], []) == pytest.approx(0.0)
        assert sks_stability([], []) == pytest.approx(0.0)


class TestSKSSeparability:
    """sks_separability — inter/intra cluster distance ratio."""

    def test_well_separated(self) -> None:
        """Well-separated clusters → high separability."""
        states = torch.zeros(20, 8)
        states[:10, 0] = 0.0  # group A: phase 0
        states[10:, 0] = 3.0  # group B: phase 3
        clusters = [{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 11, 12, 13, 14, 15, 16, 17, 18, 19}]
        s = sks_separability(clusters, states)
        assert s > 1.0

    def test_overlapping_low(self) -> None:
        """Overlapping phases → low separability."""
        states = torch.zeros(20, 8)
        states[:, 0] = torch.randn(20) * 0.01  # all phases nearly same
        clusters = [set(range(10)), set(range(10, 20))]
        s = sks_separability(clusters, states)
        assert s < 1.5

    def test_single_cluster(self) -> None:
        """Single cluster → returns 0 (no inter-cluster distance)."""
        states = torch.zeros(10, 8)
        clusters = [set(range(10))]
        s = sks_separability(clusters, states)
        assert s == pytest.approx(0.0)

    def test_empty(self) -> None:
        states = torch.zeros(10, 8)
        assert sks_separability([], states) == pytest.approx(0.0)
