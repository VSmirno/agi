"""Tests for MetacogState + MetacogMonitor Stage 9 extensions."""

from __future__ import annotations

import pytest

from snks.daf.types import MetacogConfig
from snks.gws.workspace import GWSState
from snks.metacog.monitor import MetacogMonitor, MetacogState


def make_gws(winner_id: int = 0, size: int = 10) -> GWSState:
    return GWSState(
        winner_id=winner_id,
        winner_nodes=set(range(size)),
        winner_size=size,
        winner_score=float(size),
        dominance=1.0,
    )


class _Proxy:
    """Minimal proxy simulating CycleResult for MetacogMonitor.update()."""
    def __init__(self, mean_pe: float = 0.1, winner_pe: float = 0.0) -> None:
        self.mean_prediction_error = mean_pe
        self.winner_pe = winner_pe


class TestMetacogStateFields:

    def test_metacog_state_has_winner_pe(self) -> None:
        state = MetacogState(
            confidence=0.5, dominance=0.5, stability=0.5,
            pred_error=0.1, winner_pe=0.2,
        )
        assert state.winner_pe == 0.2

    def test_metacog_state_winner_pe_default_zero(self) -> None:
        state = MetacogState(
            confidence=0.5, dominance=0.5, stability=0.5, pred_error=0.1,
        )
        assert state.winner_pe == 0.0

    def test_metacog_state_has_winner_nodes(self) -> None:
        nodes = {1, 2, 3}
        state = MetacogState(
            confidence=0.5, dominance=0.5, stability=0.5,
            pred_error=0.1, winner_nodes=nodes,
        )
        assert state.winner_nodes == nodes

    def test_metacog_state_winner_nodes_default_empty(self) -> None:
        state = MetacogState(
            confidence=0.5, dominance=0.5, stability=0.5, pred_error=0.1,
        )
        assert state.winner_nodes == set()


class TestMetacogMonitorWinnerPE:

    def test_monitor_uses_winner_pe_when_nonzero(self) -> None:
        cfg = MetacogConfig(alpha=0.0, beta=0.0, gamma=1.0)
        monitor = MetacogMonitor(cfg)
        gws = make_gws()

        # winner_pe=0.5 → confidence = 1 - 0.5 = 0.5
        proxy = _Proxy(mean_pe=0.0, winner_pe=0.5)
        state = monitor.update(gws, proxy)
        assert abs(state.confidence - 0.5) < 0.05

    def test_monitor_fallback_when_winner_pe_zero(self) -> None:
        cfg = MetacogConfig(alpha=0.0, beta=0.0, gamma=1.0)
        monitor = MetacogMonitor(cfg)
        gws = make_gws()

        # winner_pe=0.0 → fallback на pred_error_norm
        proxy = _Proxy(mean_pe=0.0, winner_pe=0.0)
        state = monitor.update(gws, proxy)
        # pred_error=0.0 → pred_error_norm=0.0 → confidence = 1 - 0.0 = 1.0
        assert abs(state.confidence - 1.0) < 0.05

    def test_monitor_winner_pe_stored_in_state(self) -> None:
        monitor = MetacogMonitor()
        gws = make_gws()
        proxy = _Proxy(winner_pe=0.3)
        state = monitor.update(gws, proxy)
        assert abs(state.winner_pe - 0.3) < 1e-5

    def test_monitor_passes_winner_nodes_to_state(self) -> None:
        monitor = MetacogMonitor()
        winner_nodes = set(range(10))
        gws = make_gws(size=10)
        proxy = _Proxy()
        state = monitor.update(gws, proxy)
        assert state.winner_nodes == winner_nodes

    def test_monitor_winner_nodes_empty_when_no_gws(self) -> None:
        monitor = MetacogMonitor()
        proxy = _Proxy()
        state = monitor.update(None, proxy)
        assert state.winner_nodes == set()

    def test_get_broadcast_currents_delegates_to_policy(self) -> None:
        from snks.metacog.policies import BroadcastPolicy
        cfg = MetacogConfig(policy="broadcast", policy_strength=1.0, broadcast_threshold=0.5)
        monitor = MetacogMonitor(cfg)
        gws = make_gws(size=5)
        proxy = _Proxy(winner_pe=0.1)  # low PE → high confidence
        state = monitor.update(gws, proxy)
        monitor.apply_policy(state, __import__("snks.daf.types", fromlist=["DafConfig"]).DafConfig())
        result = monitor.get_broadcast_currents(n_nodes=100)
        # BroadcastPolicy should return currents for winner_nodes
        assert result is not None
