"""Unit tests for Stage 11: StochasticSimulator.

All tests are CPU-only and complete in < 5 seconds total.
"""

from __future__ import annotations

import pytest

from snks.daf.types import CausalAgentConfig
from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(min_obs: int = 1) -> CausalWorldModel:
    """Return a fresh CausalWorldModel with a low observation threshold."""
    cfg = CausalAgentConfig()
    cfg.causal_min_observations = min_obs
    return CausalWorldModel(cfg)


def _model_with_single_effect(n_obs: int = 5) -> CausalWorldModel:
    """Model where action=0 in context {0} always produces effect {0,1}."""
    model = _make_model(min_obs=1)
    for _ in range(n_obs):
        model.observe_transition(pre_sks={0}, action=0, post_sks={0, 1})
    return model


def _model_with_two_effects() -> CausalWorldModel:
    """Model where action=0 in context {0} has two possible effects.

    effect A ({0,1}): observed 3 times  → confidence ~0.3
    effect B ({0,2}): observed 7 times  → confidence ~0.7
    """
    model = _make_model(min_obs=1)
    for _ in range(3):
        model.observe_transition(pre_sks={0}, action=0, post_sks={0, 1})
    for _ in range(7):
        model.observe_transition(pre_sks={0}, action=0, post_sks={0, 2})
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSampleEffect:

    def test_sample_effect_empty_model_returns_empty(self):
        """CausalWorldModel with no observations → (set(), 0.0)."""
        model = _make_model()
        sim = StochasticSimulator(model, seed=0)
        effect, conf = sim.sample_effect(context={0}, action=0)
        assert effect == set()
        assert conf == 0.0

    def test_sample_effect_single_effect_deterministic(self):
        """Model with one known effect → sample_effect always returns that effect."""
        model = _model_with_single_effect(n_obs=5)
        sim = StochasticSimulator(model, seed=42)
        for _ in range(10):
            effect, conf = sim.sample_effect(context={0}, action=0)
            # The effect is the symmetric difference: {0,1} ^ {0} = {1}
            assert isinstance(effect, set)
            assert conf > 0.0

    def test_sample_effect_temperature_zero_argmax(self):
        """Two effects with different confidences; temperature=0 → always higher conf."""
        model = _model_with_two_effects()
        sim = StochasticSimulator(model, seed=0)

        # Determine which effect has the highest confidence
        effects = model.get_all_effects_for_action({0}, 0)
        assert len(effects) == 2
        best_effect, best_conf = max(effects, key=lambda x: x[1])

        # At temperature=0 we must always get the best effect
        for _ in range(20):
            sampled_effect, sampled_conf = sim.sample_effect(
                context={0}, action=0, temperature=0.0
            )
            assert sampled_effect == best_effect
            assert sampled_conf == pytest.approx(best_conf, abs=1e-9)


class TestRollout:

    def test_rollout_returns_trajectory(self):
        """rollout() returns a trajectory of exactly len(action_sequence) steps."""
        model = _model_with_single_effect(n_obs=5)
        sim = StochasticSimulator(model, seed=7)

        action_seq = [0, 0, 0]
        traj, total_cost = sim.rollout(
            initial_sks={0}, action_sequence=action_seq, temperature=0.0
        )

        assert len(traj) == len(action_seq)
        # Each element is (set, float)
        for state, conf in traj:
            assert isinstance(state, set)
            assert isinstance(conf, float)
        assert isinstance(total_cost, float)
        assert total_cost >= 0.0

    def test_rollout_empty_action_sequence(self):
        """rollout() with empty action sequence returns empty trajectory and 0 cost."""
        model = _make_model()
        sim = StochasticSimulator(model, seed=0)
        traj, cost = sim.rollout(initial_sks={0}, action_sequence=[])
        assert traj == []
        assert cost == 0.0


class TestFindPlanStochastic:

    def test_find_plan_goal_already_reached(self):
        """If current_sks already contains goal_sks → returns ([], 1.0)."""
        model = _make_model()
        sim = StochasticSimulator(model, seed=0)
        plan, rate = sim.find_plan_stochastic(
            current_sks={0, 1}, goal_sks={1}
        )
        assert plan == []
        assert rate == pytest.approx(1.0)

    def test_find_plan_stochastic_simple_goal(self):
        """Action 0 reliably moves {0} → {0,1}; goal={1} should be found."""
        model = _make_model(min_obs=1)
        for _ in range(10):
            model.observe_transition(pre_sks={0}, action=0, post_sks={0, 1})

        sim = StochasticSimulator(model, seed=0)
        plan, success_rate = sim.find_plan_stochastic(
            current_sks={0},
            goal_sks={1},
            n_actions=3,
            n_samples=16,
            temperature=0.5,
            max_depth=5,
            min_confidence=0.1,
        )
        assert plan is not None, "Expected a plan to be found"
        assert len(plan) >= 1
        assert success_rate > 0.0

    def test_find_plan_stochastic_returns_none_if_unreachable(self):
        """Goal SKS that no action can ever reach → returns (None, ...)."""
        model = _make_model(min_obs=1)
        # Only teach action 0: {0} → {0, 1}
        for _ in range(5):
            model.observe_transition(pre_sks={0}, action=0, post_sks={0, 1})

        sim = StochasticSimulator(model, seed=0)
        # goal {99} is never reachable
        plan, rate = sim.find_plan_stochastic(
            current_sks={0},
            goal_sks={99},
            n_actions=2,
            n_samples=4,
            temperature=1.0,
            max_depth=3,
            min_confidence=0.3,
        )
        assert plan is None

    def test_deterministic_step_not_random(self):
        """Same seed + same model → same plan (reproducibility)."""
        model = _make_model(min_obs=1)
        for _ in range(10):
            model.observe_transition(pre_sks={0}, action=0, post_sks={0, 1})
        for _ in range(10):
            model.observe_transition(pre_sks={0}, action=1, post_sks={0, 2})

        def _run(seed: int):
            sim = StochasticSimulator(model, seed=seed)
            plan, rate = sim.find_plan_stochastic(
                current_sks={0},
                goal_sks={1},
                n_actions=3,
                n_samples=8,
                temperature=1.0,
                max_depth=5,
                min_confidence=0.1,
            )
            return plan, rate

        plan_a, rate_a = _run(seed=123)
        plan_b, rate_b = _run(seed=123)
        assert plan_a == plan_b
        assert rate_a == pytest.approx(rate_b)
