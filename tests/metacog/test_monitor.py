"""Tests for MetacogMonitor."""

import pytest
from dataclasses import dataclass

from snks.metacog.monitor import MetacogMonitor, MetacogState
from snks.daf.types import MetacogConfig, DafConfig


# Minimal mocks for types defined in other modules
@dataclass
class MockGWSState:
    winner_id: int
    winner_nodes: set
    winner_size: int
    winner_score: float
    dominance: float


@dataclass
class MockCycleResult:
    sks_clusters: dict
    n_sks: int
    mean_prediction_error: float
    n_spikes: int
    cycle_time_ms: float


def make_monitor(**kwargs) -> MetacogMonitor:
    return MetacogMonitor(MetacogConfig(**kwargs))


def make_gws(winner_nodes: set, dominance: float) -> MockGWSState:
    return MockGWSState(
        winner_id=0,
        winner_nodes=winner_nodes,
        winner_size=len(winner_nodes),
        winner_score=float(len(winner_nodes)),
        dominance=dominance,
    )


def make_cycle(mean_pe: float = 0.1) -> MockCycleResult:
    return MockCycleResult(
        sks_clusters={}, n_sks=0,
        mean_prediction_error=mean_pe,
        n_spikes=0, cycle_time_ms=0.0,
    )


class TestConfidenceFormula:
    def test_confidence_formula(self):
        """confidence = alpha*dom + beta*stab + gamma*(1-pe_norm)."""
        # alpha=beta=gamma=1/3, dominance=1.0, stability=1.0 (prev same), pred_error=0.0
        monitor = make_monitor(alpha=1/3, beta=1/3, gamma=1/3)
        gws = make_gws({1, 2, 3}, dominance=1.0)
        cycle = make_cycle(mean_pe=0.0)

        # First cycle: stability=0.0
        state1 = monitor.update(gws, cycle)
        assert state1.stability == pytest.approx(0.0)
        # confidence = 1/3 * 1.0 + 1/3 * 0.0 + 1/3 * 1.0 = 2/3
        assert state1.confidence == pytest.approx(2/3, abs=1e-6)

        # Second cycle: same winner -> stability=1.0
        state2 = monitor.update(gws, cycle)
        assert state2.stability == pytest.approx(1.0)
        # confidence = 1/3 * 1.0 + 1/3 * 1.0 + 1/3 * 1.0 = 1.0
        assert state2.confidence == pytest.approx(1.0, abs=1e-6)

    def test_stability_first_cycle(self):
        """stability=0.0 on first cycle (no previous winner)."""
        monitor = make_monitor()
        gws = make_gws({1, 2, 3}, dominance=0.5)
        state = monitor.update(gws, make_cycle())
        assert state.stability == pytest.approx(0.0)

    def test_stability_identical_winner(self):
        """stability=1.0 when winner nodes are identical."""
        monitor = make_monitor()
        gws = make_gws({1, 2, 3}, dominance=0.5)
        monitor.update(gws, make_cycle())  # first cycle
        state = monitor.update(gws, make_cycle())  # second with same nodes
        assert state.stability == pytest.approx(1.0)

    def test_stability_disjoint_winner(self):
        """stability=0.0 when winner nodes are fully disjoint."""
        monitor = make_monitor()
        gws1 = make_gws({1, 2, 3}, dominance=0.5)
        gws2 = make_gws({4, 5, 6}, dominance=0.5)
        monitor.update(gws1, make_cycle())
        state = monitor.update(gws2, make_cycle())
        assert state.stability == pytest.approx(0.0)

    def test_pred_error_normalization(self):
        """pred_error_norm in [0, 1]."""
        monitor = make_monitor(alpha=0.0, beta=0.0, gamma=1.0)
        gws = make_gws({1, 2}, dominance=1.0)
        # First: max_pred_error initialized to 1.0, pred_error=0.5 -> norm=0.5
        state = monitor.update(gws, make_cycle(mean_pe=0.5))
        # confidence = gamma * (1 - 0.5/1.0) = 0.5
        assert state.confidence == pytest.approx(0.5, abs=1e-6)

        # Second: pred_error=1.5 -> updates max to 1.5, norm=1.0
        state2 = monitor.update(gws, make_cycle(mean_pe=1.5))
        # confidence = gamma * (1 - 1.5/1.5) = 0.0
        assert state2.confidence == pytest.approx(0.0, abs=1e-6)

        # Third: pred_error=0.0 -> norm=0.0/1.5=0.0, confidence=1.0
        state3 = monitor.update(gws, make_cycle(mean_pe=0.0))
        assert state3.confidence == pytest.approx(1.0, abs=1e-6)

    def test_gws_none_returns_zero_confidence(self):
        """gws_state=None -> confidence=0.0, all components=0.0."""
        monitor = make_monitor()
        state = monitor.update(None, make_cycle())
        assert state.confidence == pytest.approx(0.0)
        assert state.dominance == pytest.approx(0.0)
        assert state.stability == pytest.approx(0.0)
