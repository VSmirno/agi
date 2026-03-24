"""Tests for agent/simulation.py — MentalSimulator."""

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.agent.simulation import MentalSimulator
from snks.daf.types import CausalAgentConfig


def make_simulator(**kwargs) -> tuple[CausalWorldModel, MentalSimulator]:
    config = CausalAgentConfig(**kwargs)
    model = CausalWorldModel(config)
    sim = MentalSimulator(model)
    return model, sim


class TestMentalSimulator:
    def test_simulate_empty_sequence(self):
        model, sim = make_simulator()
        result = sim.simulate({1, 2}, [])
        assert result == []

    def test_simulate_single_step(self):
        model, sim = make_simulator(causal_min_observations=1)
        # Teach: action 2 in context {1} produces {1, 10}
        model.observe_transition({1}, action=2, post_sks={1, 10})
        trajectory = sim.simulate({1}, [2])
        assert len(trajectory) == 1
        predicted_sks, confidence = trajectory[0]
        assert 10 in predicted_sks

    def test_simulate_chain(self):
        model, sim = make_simulator(causal_min_observations=1)
        # Teach chain: {1} + action 0 → {1, 2}, then {1, 2} + action 1 → {1, 2, 3}
        model.observe_transition({1}, action=0, post_sks={1, 2})
        model.observe_transition({1, 2}, action=1, post_sks={1, 2, 3})
        trajectory = sim.simulate({1}, [0, 1])
        assert len(trajectory) == 2
        assert 2 in trajectory[0][0]
        assert 3 in trajectory[1][0]

    def test_find_plan_trivial(self):
        model, sim = make_simulator()
        # Goal already reached
        plan = sim.find_plan({1, 2}, {1, 2})
        assert plan == []

    def test_find_plan_single_step(self):
        model, sim = make_simulator(causal_min_observations=1)
        model.observe_transition({1}, action=2, post_sks={1, 10})
        plan = sim.find_plan({1}, {10}, n_actions=5, min_confidence=0.0)
        assert plan is not None
        assert 2 in plan

    def test_find_plan_not_found(self):
        model, sim = make_simulator(causal_min_observations=1)
        # No causal links — can't reach anything
        plan = sim.find_plan({1}, {999}, max_depth=3, n_actions=5)
        assert plan is None

    def test_find_plan_multi_step(self):
        model, sim = make_simulator(causal_min_observations=1)
        # Chain: {1} + a0 → {1,2}, {1,2} + a1 → {1,2,3}
        model.observe_transition({1}, action=0, post_sks={1, 2})
        model.observe_transition({1, 2}, action=1, post_sks={1, 2, 3})

        plan = sim.find_plan({1}, {3}, n_actions=5, min_confidence=0.0)
        assert plan is not None
        assert len(plan) == 2
