"""Tests for BroadcastPolicy (Stage 9)."""

import torch
import pytest

from snks.daf.types import DafConfig
from snks.metacog.monitor import MetacogState
from snks.metacog.policies import NullPolicy, NoisePolicy, STDPPolicy, BroadcastPolicy


N_NODES = 100


def make_state(confidence: float = 0.8, winner_nodes: set | None = None) -> MetacogState:
    nodes = {10, 20, 30} if winner_nodes is None else winner_nodes
    return MetacogState(
        confidence=confidence,
        dominance=0.5,
        stability=0.5,
        pred_error=0.1,
        winner_nodes=nodes,
    )


class TestNullPolicyBroadcast:

    def test_null_policy_get_broadcast_returns_none(self) -> None:
        policy = NullPolicy()
        result = policy.get_broadcast_currents(N_NODES)
        assert result is None

    def test_noise_policy_get_broadcast_returns_none(self) -> None:
        policy = NoisePolicy(strength=1.0)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is None

    def test_stdp_policy_get_broadcast_returns_none(self) -> None:
        policy = STDPPolicy(strength=1.0)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is None


class TestBroadcastPolicy:

    def test_broadcast_below_threshold_no_currents(self) -> None:
        policy = BroadcastPolicy(strength=1.0, threshold=0.6)
        config = DafConfig()
        state = make_state(confidence=0.3)
        policy.apply(state, config)
        assert policy.get_broadcast_currents(N_NODES) is None

    def test_broadcast_above_threshold_returns_currents(self) -> None:
        policy = BroadcastPolicy(strength=1.0, threshold=0.6)
        config = DafConfig()
        state = make_state(confidence=0.8, winner_nodes={10, 20, 30})
        policy.apply(state, config)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is not None
        assert result.shape == (N_NODES,)

    def test_winner_nodes_get_current(self) -> None:
        winner_nodes = {10, 20, 30}
        policy = BroadcastPolicy(strength=1.0, threshold=0.5)
        config = DafConfig()
        state = make_state(confidence=0.9, winner_nodes=winner_nodes)
        policy.apply(state, config)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is not None
        for node in winner_nodes:
            assert result[node].item() > 0.0, f"Node {node} should have positive current"
        # Non-winner nodes should be zero
        non_winner = set(range(N_NODES)) - winner_nodes
        for node in list(non_winner)[:10]:
            assert result[node].item() == 0.0

    def test_broadcast_strength_scales_current(self) -> None:
        policy1 = BroadcastPolicy(strength=1.0, threshold=0.5)
        policy2 = BroadcastPolicy(strength=2.0, threshold=0.5)
        config = DafConfig()
        state = make_state(confidence=0.9, winner_nodes={5})
        policy1.apply(state, config)
        policy2.apply(state, config)
        r1 = policy1.get_broadcast_currents(N_NODES)
        r2 = policy2.get_broadcast_currents(N_NODES)
        assert r1 is not None and r2 is not None
        assert abs(r2[5].item() / r1[5].item() - 2.0) < 1e-5

    def test_broadcast_clears_after_get(self) -> None:
        policy = BroadcastPolicy(strength=1.0, threshold=0.5)
        config = DafConfig()
        state = make_state(confidence=0.9)
        policy.apply(state, config)
        _ = policy.get_broadcast_currents(N_NODES)   # первый вызов
        result = policy.get_broadcast_currents(N_NODES)  # второй вызов
        assert result is None, "BroadcastPolicy должна сбросить pending после первого get"

    def test_broadcast_at_exact_threshold(self) -> None:
        policy = BroadcastPolicy(strength=1.0, threshold=0.6)
        config = DafConfig()
        state = make_state(confidence=0.6)  # точно на пороге
        policy.apply(state, config)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is not None, "confidence == threshold должно давать broadcast"

    def test_broadcast_empty_winner_nodes_no_currents(self) -> None:
        policy = BroadcastPolicy(strength=1.0, threshold=0.5)
        config = DafConfig()
        state = make_state(confidence=0.9, winner_nodes=set())
        policy.apply(state, config)
        result = policy.get_broadcast_currents(N_NODES)
        assert result is None, "Пустые winner_nodes не должны давать broadcast"
