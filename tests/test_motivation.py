"""Tests for agent/motivation.py — IntrinsicMotivation."""

import pytest
import torch

from snks.agent.agent import _perceptual_hash
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

    def test_learning_progress_decays_for_repeated_actions(self):
        """Learning progress → 0 when prediction error stabilises."""
        _, motivation = make_components(curiosity_epsilon=0.0, curiosity_decay=0.9)
        # Use perceptual hash IDs (>= 10000) so _stable_context works
        ctx = {10001, 10002}
        # Simulate constant prediction error (already learned)
        for _ in range(20):
            motivation.update(ctx, action=0, prediction_error=0.1)

        from snks.agent.motivation import _stable_context
        key = (_stable_context(ctx), 0)
        lp = motivation._learning_progress[key]
        # After many updates with same error, LP should be near 0
        assert lp < 0.15, f"Expected LP < 0.15 after stable error, got {lp}"

    def test_forward_preferred_over_turn_after_learning(self):
        """After turns are learned, forward into new state should win."""
        model, motivation = make_components(curiosity_epsilon=0.0, curiosity_decay=0.9)
        ctx = {10001, 10002}
        # Simulate: turns (actions 0,1) learned — constant low error
        for _ in range(30):
            motivation.update(ctx, action=0, prediction_error=0.1)
            motivation.update(ctx, action=1, prediction_error=0.1)
        # Forward (action 2) — never tried → LP defaults to 1.0
        action = motivation.select_action(ctx, model, n_actions=5)
        assert action != 0 and action != 1, (
            f"Expected non-turn action, got {action}"
        )


class TestPerceptualHash:
    def test_rotation_invariant(self):
        """Same image rotated 90° should produce identical hash."""
        torch.manual_seed(42)
        img = torch.rand(1, 1, 64, 64)
        # Rotate 90° clockwise
        img_rot = torch.rot90(img, k=1, dims=[-2, -1])

        hash_orig = _perceptual_hash(img)
        hash_rot = _perceptual_hash(img_rot)
        assert hash_orig == hash_rot, (
            f"Hash changed after rotation: {len(hash_orig ^ hash_rot)} IDs differ"
        )

    def test_different_images_different_hash(self):
        """Clearly different images should produce different hashes."""
        black = torch.zeros(1, 1, 64, 64)
        white = torch.ones(1, 1, 64, 64)
        assert _perceptual_hash(black) != _perceptual_hash(white)

    def test_returns_set_of_ints(self):
        img = torch.rand(1, 1, 64, 64)
        result = _perceptual_hash(img)
        assert isinstance(result, set)
        assert all(isinstance(x, int) for x in result)
        # IDs should be in the expected range
        assert all(x >= 10000 for x in result)
