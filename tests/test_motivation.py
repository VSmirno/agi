"""Tests for agent/motivation.py — IntrinsicMotivation."""

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.agent.motivation import IntrinsicMotivation
from snks.daf.types import CausalAgentConfig


def make_components(**kwargs):
    config = CausalAgentConfig(**kwargs)
    model = CausalWorldModel(config)
    motivation = IntrinsicMotivation(config)
    return model, motivation


class TestIntrinsicMotivation:
    def test_select_action_returns_valid(self):
        model, motivation = make_components(curiosity_epsilon=0.0)
        action = motivation.select_action({1, 2}, model, n_actions=5)
        assert 0 <= action < 5

    def test_epsilon_greedy(self):
        """With epsilon=1.0 all actions should be random (uniform)."""
        model, motivation = make_components(curiosity_epsilon=1.0)
        actions = [motivation.select_action({1}, model, n_actions=5) for _ in range(100)]
        unique_actions = set(actions)
        # Should see multiple different actions with high epsilon
        assert len(unique_actions) > 1

    def test_novelty_decreases_with_visits(self):
        model, motivation = make_components(curiosity_epsilon=0.0)
        ctx = {1, 2}
        # Visit action 0 many times
        for _ in range(50):
            motivation.update(ctx, action=0, prediction_error=0.1)

        count = motivation.get_visit_count(ctx, action=0)
        assert count == 50

    def test_prefers_unexplored_actions(self):
        model, motivation = make_components(curiosity_epsilon=0.0)
        ctx = {1}
        # Heavily visit action 0
        for _ in range(100):
            motivation.update(ctx, action=0, prediction_error=0.01)

        # Now select — should prefer actions other than 0
        actions = [motivation.select_action(ctx, model, n_actions=5) for _ in range(20)]
        # Action 0 should be rare since it has high visit count
        action_0_count = actions.count(0)
        assert action_0_count < 15  # not always action 0

    def test_update_records_error(self):
        model, motivation = make_components()
        motivation.update({1}, action=2, prediction_error=0.5)
        assert motivation.get_visit_count({1}, action=2) == 1
