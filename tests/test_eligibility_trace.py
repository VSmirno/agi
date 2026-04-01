"""Tests for Stage 41: Eligibility Trace — temporal credit assignment."""

import math

import pytest
import torch

from snks.daf.eligibility import EligibilityTrace
from snks.daf.graph import SparseDafGraph
from snks.daf.stdp import STDP, STDPResult
from snks.daf.types import DafConfig


def _make_graph(n_nodes: int = 100, avg_degree: int = 10, seed: int = 42) -> SparseDafGraph:
    """Create a small test graph."""
    return SparseDafGraph.random_sparse(n_nodes, avg_degree, device=torch.device("cpu"), seed=seed)


class TestEligibilityTraceBasic:
    """Basic trace accumulation and decay."""

    def test_accumulate_initializes_trace(self):
        trace = EligibilityTrace(decay=0.9)
        dw = torch.randn(50)
        trace.accumulate(dw)
        assert trace._trace is not None
        assert trace._trace.shape == (50,)
        assert torch.allclose(trace._trace, dw)

    def test_accumulate_decays_previous(self):
        trace = EligibilityTrace(decay=0.9)
        dw1 = torch.ones(50)
        dw2 = torch.ones(50) * 2.0
        trace.accumulate(dw1)
        trace.accumulate(dw2)
        # e = 0.9 * 1.0 + 2.0 = 2.9
        expected = 0.9 * 1.0 + 2.0
        assert torch.allclose(trace._trace, torch.full((50,), expected))

    def test_accumulate_20_steps(self):
        """Trace after 20 steps should still be non-zero."""
        trace = EligibilityTrace(decay=0.92)
        for _ in range(20):
            trace.accumulate(torch.ones(100) * 0.01)
        assert trace.trace_magnitude > 0
        assert trace._steps_accumulated == 20

    def test_trace_decay_over_20_steps(self):
        """Signal from step 0 should be ~19% after 20 steps with λ=0.92."""
        trace = EligibilityTrace(decay=0.92)
        # Step 0: inject signal
        trace.accumulate(torch.ones(10))
        # Steps 1-19: inject zero (just decay)
        for _ in range(19):
            trace.accumulate(torch.zeros(10))
        # Original signal decayed: 0.92^19 ≈ 0.20
        expected_decay = 0.92 ** 19
        assert torch.allclose(trace._trace, torch.full((10,), expected_decay), atol=1e-5)

    def test_reset_zeros_trace(self):
        trace = EligibilityTrace(decay=0.9)
        trace.accumulate(torch.ones(50))
        trace.reset()
        assert trace.trace_magnitude == 0.0
        assert trace._steps_accumulated == 0

    def test_effective_window(self):
        """λ=0.92 → effective window ~35 steps (0.92^35 ≈ 0.05)."""
        trace = EligibilityTrace(decay=0.92)
        expected = int(math.log(0.05) / math.log(0.92))
        assert trace.effective_window == expected
        assert trace.effective_window >= 20  # must be >= 20 (gate criteria)


class TestEligibilityTraceReward:
    """Reward modulation through trace."""

    def test_apply_reward_modifies_weights(self):
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.9, reward_lr=0.5)

        # Accumulate some dw
        dw = torch.ones(n_edges) * 0.01
        trace.accumulate(dw)

        # Record weights before
        w_before = graph.get_strength().clone()

        # Apply positive reward
        n_mod = trace.apply_reward(1.0, graph, w_min=0.0, w_max=1.0)

        w_after = graph.get_strength()
        assert n_mod > 0
        assert not torch.allclose(w_before, w_after)

    def test_positive_reward_increases_traced_weights(self):
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.9, reward_lr=0.5)

        # Set weights to middle
        graph.set_strength(torch.full((n_edges,), 0.5))

        # Positive dw (potentiation trace)
        trace.accumulate(torch.ones(n_edges) * 0.1)

        w_before = graph.get_strength().clone()
        trace.apply_reward(1.0, graph, w_min=0.0, w_max=1.0)
        w_after = graph.get_strength()

        # Weights should increase
        assert (w_after >= w_before).all()

    def test_negative_reward_decreases_traced_weights(self):
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.9, reward_lr=0.5)

        graph.set_strength(torch.full((n_edges,), 0.5))
        trace.accumulate(torch.ones(n_edges) * 0.1)

        w_before = graph.get_strength().clone()
        trace.apply_reward(-1.0, graph, w_min=0.0, w_max=1.0)
        w_after = graph.get_strength()

        assert (w_after <= w_before).all()

    def test_zero_reward_no_change(self):
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.9, reward_lr=0.5)
        trace.accumulate(torch.ones(n_edges) * 0.1)

        w_before = graph.get_strength().clone()
        n_mod = trace.apply_reward(0.0, graph, w_min=0.0, w_max=1.0)
        assert n_mod == 0
        assert torch.allclose(w_before, graph.get_strength())

    def test_reward_clamped(self):
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.9, reward_lr=10.0)

        graph.set_strength(torch.full((n_edges,), 0.9))
        trace.accumulate(torch.ones(n_edges) * 1.0)
        trace.apply_reward(1.0, graph, w_min=0.0, w_max=1.0)

        w = graph.get_strength()
        assert w.max() <= 1.0
        assert w.min() >= 0.0

    def test_long_range_credit(self):
        """Reward at step 15 should still credit step 0's STDP changes."""
        graph = _make_graph()
        n_edges = graph.edge_attr.shape[0]
        trace = EligibilityTrace(decay=0.92, reward_lr=0.5)

        graph.set_strength(torch.full((n_edges,), 0.5))

        # Step 0: significant STDP
        trace.accumulate(torch.ones(n_edges) * 0.1)
        # Steps 1-14: small STDP
        for _ in range(14):
            trace.accumulate(torch.randn(n_edges) * 0.001)

        w_before = graph.get_strength().clone()
        trace.apply_reward(1.0, graph, w_min=0.0, w_max=1.0)
        w_after = graph.get_strength()

        # Should have changed
        delta = (w_after - w_before).abs().mean()
        assert delta > 1e-4, f"Long-range credit too weak: {delta}"


class TestEligibilityTraceEdgeCases:
    """Edge cases and robustness."""

    def test_no_trace_reward_noop(self):
        graph = _make_graph()
        trace = EligibilityTrace()
        n_mod = trace.apply_reward(1.0, graph, 0.0, 1.0)
        assert n_mod == 0

    def test_shape_mismatch_resets(self):
        """If graph edge count changes (structural pruning), trace resets."""
        graph = _make_graph()
        trace = EligibilityTrace()
        # Accumulate with wrong size
        trace.accumulate(torch.ones(999))
        n_mod = trace.apply_reward(1.0, graph, 0.0, 1.0)
        assert n_mod == 0  # reset due to mismatch

    def test_accumulate_new_size_reinitializes(self):
        trace = EligibilityTrace()
        trace.accumulate(torch.ones(50))
        assert trace._trace.shape == (50,)
        trace.accumulate(torch.ones(100))
        assert trace._trace.shape == (100,)

    def test_stats(self):
        trace = EligibilityTrace(decay=0.92)
        trace.accumulate(torch.ones(50))
        s = trace.stats
        assert s["steps_accumulated"] == 1
        assert s["trace_magnitude"] > 0
        assert s["effective_window"] >= 20


class TestSTDPReturnsRawDw:
    """STDP.apply() must return raw dw (before homeostasis) in STDPResult."""

    def test_stdp_result_has_dw_field(self):
        result = STDPResult(edges_potentiated=0, edges_depressed=0,
                           mean_weight_change=0.0, dw=None)
        assert hasattr(result, "dw")

    def test_rate_stdp_returns_dw(self):
        config = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="kuramoto",
                          stdp_mode="rate")
        graph = _make_graph(n_nodes=50, avg_degree=5)
        stdp = STDP(config)

        fired = torch.rand(10, 50) > 0.7
        result = stdp.apply(graph, fired)
        assert result.dw is not None
        assert result.dw.shape == (graph.edge_attr.shape[0],)

    def test_timing_stdp_returns_dw(self):
        config = DafConfig(num_nodes=50, avg_degree=5, oscillator_model="fhn",
                          stdp_mode="timing")
        graph = _make_graph(n_nodes=50, avg_degree=5)
        stdp = STDP(config)

        fired = torch.rand(10, 50) > 0.7
        result = stdp.apply(graph, fired)
        assert result.dw is not None
        assert result.dw.shape == (graph.edge_attr.shape[0],)

    def test_dw_excludes_homeostasis(self):
        """dw should be pure STDP without homeostatic regularization."""
        config = DafConfig(num_nodes=50, avg_degree=5, stdp_mode="rate",
                          homeostasis_lambda=0.1)  # strong homeostasis
        graph = _make_graph(n_nodes=50, avg_degree=5)
        stdp = STDP(config)

        # All neurons fire: dw = a_plus * (rate_product - baseline)
        # With all firing: rate=1.0, product=1.0, baseline=1.0 → dw_stdp = 0
        # homeostasis would add 0.1*(0.5-w) which is nonzero
        fired = torch.ones(10, 50, dtype=torch.bool)
        result = stdp.apply(graph, fired)

        # dw should be close to zero (no net STDP for uniform firing)
        # if homeostasis leaked in, magnitude would be ~0.1*(0.5-w) ≈ 0.05
        assert result.dw.abs().max() < 0.02, "dw should exclude homeostasis"


class TestCycleResultHasSTDP:
    """CycleResult should expose stdp_result from engine."""

    def test_cycle_result_has_stdp_field(self):
        from snks.pipeline.runner import CycleResult
        # Check the dataclass has the field
        import dataclasses
        field_names = [f.name for f in dataclasses.fields(CycleResult)]
        assert "stdp_result" in field_names


class TestDafCausalModelWithEligibility:
    """DafCausalModel with eligibility trace integration."""

    def test_accumulate_stdp_method_exists(self):
        from snks.agent.daf_causal_model import DafCausalModel
        assert hasattr(DafCausalModel, "accumulate_stdp")

    def test_eligibility_trace_attribute(self):
        from snks.agent.daf_causal_model import DafCausalModel
        config = DafConfig(num_nodes=50, avg_degree=5)
        from snks.daf.engine import DafEngine
        engine = DafEngine(config)
        model = DafCausalModel(engine)
        assert hasattr(model, "_eligibility")

    def test_after_action_uses_eligibility(self):
        """After action with reward should modulate via eligibility trace."""
        from snks.agent.daf_causal_model import DafCausalModel
        config = DafConfig(num_nodes=50, avg_degree=5)
        from snks.daf.engine import DafEngine
        engine = DafEngine(config)
        model = DafCausalModel(engine, trace_decay=0.92, trace_reward_lr=0.5)

        # Accumulate some STDP signal
        n_edges = engine.graph.edge_attr.shape[0]
        dw = torch.ones(n_edges) * 0.01
        model.accumulate_stdp(STDPResult(
            edges_potentiated=100, edges_depressed=50,
            mean_weight_change=0.01, dw=dw,
        ))

        # Prepare before_action (needed for _trace)
        model.before_action(0, set())

        w_before = engine.graph.get_strength().clone()
        model.after_action(1.0)
        w_after = engine.graph.get_strength()

        assert not torch.allclose(w_before, w_after), "Eligibility trace should modulate weights"
