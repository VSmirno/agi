"""Tests for STDP learning rule."""

import torch

from snks.daf.graph import SparseDafGraph
from snks.daf.stdp import STDP, STDPResult
from snks.daf.types import DafConfig


def _make_single_edge_graph(device, strength=0.5):
    """Helper: single edge 0→1."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.int64, device=device)
    edge_attr = torch.tensor([[strength, 0.0, 0.0, 0.0]], device=device)
    return SparseDafGraph(edge_index, edge_attr, num_nodes=10, device=device)


class TestSTDPPotentiation:
    def test_pre_before_post_strengthens(self, device):
        """Pre fires before post → Δt > 0 → potentiation."""
        T, N = 20, 10
        fired = torch.zeros(T, N, dtype=torch.bool, device=device)
        fired[5, 0] = True  # pre at t=5
        fired[10, 1] = True  # post at t=10

        g = _make_single_edge_graph(device, strength=0.5)
        cfg = DafConfig(num_nodes=N)
        result = STDP(cfg).apply(g, fired)

        assert result.edges_potentiated >= 1
        assert g.get_strength()[0] > 0.5


class TestSTDPDepression:
    def test_post_before_pre_weakens(self, device):
        """Post fires before pre → Δt < 0 → depression."""
        T, N = 20, 10
        fired = torch.zeros(T, N, dtype=torch.bool, device=device)
        fired[10, 0] = True  # pre late
        fired[5, 1] = True  # post early

        g = _make_single_edge_graph(device, strength=0.5)
        cfg = DafConfig(num_nodes=N)
        result = STDP(cfg).apply(g, fired)

        assert result.edges_depressed >= 1
        assert g.get_strength()[0] < 0.5


class TestSTDPBounds:
    def test_weights_stay_bounded(self, device):
        """After many STDP updates, weights stay in [w_min, w_max]."""
        N = 100
        cfg = DafConfig(num_nodes=N, avg_degree=5)
        g = SparseDafGraph.random_sparse(N, 5, device, seed=42)
        stdp = STDP(cfg)

        for _ in range(50):
            fired = torch.rand(20, N, device=device) > 0.7
            stdp.apply(g, fired)

        w = g.get_strength()
        assert w.min() >= cfg.stdp_w_min - 1e-6
        assert w.max() <= cfg.stdp_w_max + 1e-6


class TestSTDPNoSpikes:
    def test_no_spikes_only_homeostatic(self, device):
        """No spikes → only homeostatic term applies."""
        N = 10
        fired = torch.zeros(20, N, dtype=torch.bool, device=device)
        g = _make_single_edge_graph(device, strength=0.3)
        cfg = DafConfig(num_nodes=N)
        stdp = STDP(cfg)

        w_before = g.get_strength()[0].item()
        result = stdp.apply(g, fired)
        w_after = g.get_strength()[0].item()

        assert result.edges_potentiated == 0
        assert result.edges_depressed == 0
        # Homeostatic: λ * (w_target - w), w_target=0.5, w=0.3 → dw > 0
        assert w_after > w_before


class TestSTDPResult:
    def test_result_fields(self, device):
        N = 50
        cfg = DafConfig(num_nodes=N, avg_degree=5)
        g = SparseDafGraph.random_sparse(N, 5, device, seed=42)
        fired = torch.rand(20, N, device=device) > 0.5
        result = STDP(cfg).apply(g, fired)

        assert isinstance(result, STDPResult)
        assert isinstance(result.edges_potentiated, int)
        assert isinstance(result.edges_depressed, int)
        assert isinstance(result.mean_weight_change, float)
