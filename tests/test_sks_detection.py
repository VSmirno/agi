"""Tests for SKS detection (Stage 3)."""

import math

import torch
import pytest

from snks.sks.detection import (
    phase_coherence_matrix,
    cofiring_coherence_matrix,
    detect_sks,
)


class TestPhaseCoherenceMatrix:
    """phase_coherence_matrix — Kuramoto phase coherence."""

    def test_output_shapes(self) -> None:
        """Returns (K,K) coherence and (K,) indices."""
        states = torch.randn(200, 8)
        coh, idx = phase_coherence_matrix(states, top_k=50)
        assert coh.shape == (50, 50)
        assert idx.shape == (50,)

    def test_top_k_clamps_to_n(self) -> None:
        """top_k > N → uses N."""
        states = torch.randn(30, 8)
        coh, idx = phase_coherence_matrix(states, top_k=100)
        assert coh.shape == (30, 30)

    def test_values_range(self) -> None:
        """Coherence in [-1, 1]."""
        states = torch.randn(100, 8)
        coh, _ = phase_coherence_matrix(states, top_k=50)
        assert coh.min() >= -1.0 - 1e-6
        assert coh.max() <= 1.0 + 1e-6

    def test_symmetric(self) -> None:
        """Coherence matrix is symmetric."""
        states = torch.randn(100, 8)
        coh, _ = phase_coherence_matrix(states, top_k=50)
        assert torch.allclose(coh, coh.T, atol=1e-6)

    def test_diagonal_ones(self) -> None:
        """Self-coherence = 1."""
        states = torch.randn(100, 8)
        coh, _ = phase_coherence_matrix(states, top_k=50)
        assert torch.allclose(coh.diag(), torch.ones(50), atol=1e-6)

    def test_synchronized_phases_high_coherence(self) -> None:
        """Nodes with same phase → coherence ≈ 1."""
        states = torch.zeros(50, 8)
        states[:, 0] = 1.5  # all same phase
        states[:, 1] = 1.0  # amplitude for activity selection
        coh, _ = phase_coherence_matrix(states, top_k=50)
        assert coh.min() > 0.99

    def test_opposite_phases_negative_coherence(self) -> None:
        """Two groups with π phase difference → coherence ≈ -1."""
        states = torch.zeros(40, 8)
        states[:20, 0] = 0.0
        states[20:, 0] = math.pi
        states[:, 1] = 1.0
        coh, _ = phase_coherence_matrix(states, top_k=40)
        # Cross-group coherence should be near -1
        cross = coh[:20, 20:]
        assert cross.mean() < -0.9


class TestCofiringCoherenceMatrix:
    """cofiring_coherence_matrix — FHN co-firing correlation."""

    def test_output_shapes(self) -> None:
        states = torch.zeros(200, 8)
        fired = torch.randint(0, 2, (50, 200), dtype=torch.bool)
        coh, idx = cofiring_coherence_matrix(fired, top_k=30)
        assert coh.shape == (30, 30)
        assert idx.shape == (30,)

    def test_values_range(self) -> None:
        fired = torch.randint(0, 2, (100, 50), dtype=torch.bool)
        coh, _ = cofiring_coherence_matrix(fired, top_k=30)
        assert coh.min() >= -1.0 - 1e-6
        assert coh.max() <= 1.0 + 1e-6

    def test_symmetric(self) -> None:
        fired = torch.randint(0, 2, (100, 50), dtype=torch.bool)
        coh, _ = cofiring_coherence_matrix(fired, top_k=30)
        assert torch.allclose(coh, coh.T, atol=1e-6)

    def test_coactive_nodes_high_coherence(self) -> None:
        """Nodes that always fire together → high coherence."""
        torch.manual_seed(42)
        fired = torch.zeros(100, 20, dtype=torch.bool)
        # Group A: nodes 0-9 fire on random subset of steps (correlated)
        pattern = torch.rand(100) > 0.7  # ~30 steps
        for n in range(10):
            fired[:, n] = pattern
        # Group B: nodes 10-19 fire independently (uncorrelated)
        for n in range(10, 20):
            fired[:, n] = torch.rand(100) > 0.7
        coh, idx = cofiring_coherence_matrix(fired, top_k=20)
        # Find two nodes from group A in idx
        group_a = set(range(10))
        a_positions = [i for i, v in enumerate(idx.tolist()) if v in group_a]
        if len(a_positions) >= 2:
            i, j = a_positions[0], a_positions[1]
            assert coh[i, j] > 0.9


class TestDetectSKS:
    """detect_sks — DBSCAN clustering on coherence."""

    def test_returns_list_of_sets(self) -> None:
        coh = torch.eye(20)
        result = detect_sks(coh, min_samples=2, min_size=2)
        assert isinstance(result, list)
        for cluster in result:
            assert isinstance(cluster, set)

    def test_no_overlap(self) -> None:
        """Clusters don't share nodes."""
        coh = torch.eye(50)
        coh[:10, :10] = 1.0
        coh[10:20, 10:20] = 1.0
        result = detect_sks(coh, eps=0.3, min_samples=3, min_size=3)
        all_nodes: set[int] = set()
        for cluster in result:
            assert len(all_nodes & cluster) == 0
            all_nodes |= cluster

    def test_min_size_filter(self) -> None:
        """Clusters smaller than min_size are discarded."""
        coh = torch.eye(30)
        coh[:15, :15] = 1.0  # one big cluster
        coh[15:18, 15:18] = 1.0  # one small cluster (3 nodes)
        result = detect_sks(coh, eps=0.3, min_samples=2, min_size=5)
        for cluster in result:
            assert len(cluster) >= 5

    def test_synchronized_group_forms_cluster(self) -> None:
        """Fully coherent group → single cluster."""
        coh = torch.eye(30)
        coh[:20, :20] = 1.0
        result = detect_sks(coh, eps=0.3, min_samples=5, min_size=5)
        assert len(result) >= 1
        biggest = max(result, key=len)
        assert len(biggest) >= 15

    def test_two_groups_two_clusters(self) -> None:
        """Two coherent groups → two clusters."""
        coh = torch.zeros(40, 40)
        coh[:20, :20] = 1.0
        coh[20:, 20:] = 1.0
        # Cross-group = 0 → distance = 1
        result = detect_sks(coh, eps=0.3, min_samples=5, min_size=5)
        assert len(result) == 2

    def test_random_noise_few_clusters(self) -> None:
        """Random coherence → few or no clusters."""
        coh = torch.randn(50, 50) * 0.1
        coh = (coh + coh.T) / 2
        coh.fill_diagonal_(1.0)
        result = detect_sks(coh, eps=0.3, min_samples=5, min_size=5)
        # Random noise shouldn't form many stable clusters
        assert len(result) <= 5
