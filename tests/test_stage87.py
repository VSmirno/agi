"""Stage 87 unit tests: DeathHypothesis + HypothesisTracker + CuriosityStimulus."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from snks.agent.death_hypothesis import (
    DeathHypothesis,
    HypothesisTracker,
    _HYPOTHESIS_THRESHOLDS,
)
from snks.agent.stimuli import CuriosityStimulus, StimuliLayer, SurvivalAversion
from snks.agent.post_mortem import PostMortemLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trajectory(vital_values: list[float], vital: str = "food", confidences: list[float] | None = None):
    """Create a minimal VectorTrajectory-like mock for testing."""
    states = [MagicMock(body={vital: v}) for v in vital_values]
    traj = MagicMock()
    traj.states = states
    traj.confidences = confidences or [0.8] * len(vital_values)
    traj.avg_surprise.return_value = 1.0 - sum(traj.confidences) / max(1, len(traj.confidences))
    return traj


# ---------------------------------------------------------------------------
# DeathHypothesis
# ---------------------------------------------------------------------------

class TestDeathHypothesis:
    def test_not_verifiable_below_threshold(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=1, n_observed=2)
        assert not h.is_verifiable

    def test_verifiable_enough_observations(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=2, n_observed=3)
        assert h.is_verifiable

    def test_support_rate(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=2, n_observed=4)
        assert h.support_rate == pytest.approx(0.5)

    def test_support_rate_zero_observed(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        assert h.support_rate == 0.0

    def test_death_relevance_at_threshold(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        traj = _make_trajectory([3.0, 3.0], vital="food")
        # proximity = 1.0 → relevance = 2.0
        assert h.death_relevance(traj) == pytest.approx(2.0)

    def test_death_relevance_far_from_threshold(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        traj = _make_trajectory([9.0, 8.0], vital="food")
        # min_vital=8.0, |8-3|=5 > 3 → proximity=0 → relevance=1.0
        assert h.death_relevance(traj) == pytest.approx(1.0)

    def test_death_relevance_slightly_above_threshold(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        traj = _make_trajectory([4.5, 3.0], vital="food")
        # min_vital=3.0, proximity=1.0 → relevance=2.0
        assert h.death_relevance(traj) == pytest.approx(2.0)

    def test_death_relevance_empty_trajectory(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        traj = MagicMock()
        traj.states = []
        assert h.death_relevance(traj) == 1.0

    def test_str_representation(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=2, n_observed=4)
        s = str(h)
        assert "zombie" in s
        assert "food" in s
        assert "3.0" in s


# ---------------------------------------------------------------------------
# HypothesisTracker
# ---------------------------------------------------------------------------

class TestHypothesisTracker:
    def test_no_hypothesis_before_records(self):
        tracker = HypothesisTracker()
        assert tracker.active_hypothesis() is None
        assert tracker.n_verifiable() == 0

    def test_record_alive_episode_ignored(self):
        tracker = HypothesisTracker()
        tracker.record({}, {"food": 1.0})
        assert tracker.active_hypothesis() is None

    def test_hypothesis_forms_after_enough_deaths(self):
        tracker = HypothesisTracker()
        # 3 zombie deaths with low food → hypothesis: zombie + food < 3
        for _ in range(3):
            tracker.record({"zombie": 1.0}, {"food": 1.5, "health": 3.0})
        assert tracker.n_verifiable() >= 1
        h = tracker.active_hypothesis()
        assert h is not None
        assert h.cause == "zombie"

    def test_hypothesis_not_verifiable_with_only_two_deaths(self):
        tracker = HypothesisTracker()
        for _ in range(2):
            tracker.record({"zombie": 1.0}, {"food": 1.5})
        assert tracker.n_verifiable() == 0

    def test_dominant_cause_used(self):
        tracker = HypothesisTracker()
        # Starvation dominates
        for _ in range(3):
            tracker.record({"starvation": 0.7, "zombie": 0.3}, {"food": 0.2})
        h = tracker.active_hypothesis()
        assert h is not None
        assert h.cause == "starvation"

    def test_all_hypotheses_returned(self):
        tracker = HypothesisTracker()
        tracker.record({"zombie": 1.0}, {"food": 1.0, "health": 3.0})
        all_h = tracker.all_hypotheses()
        # Should have entries for each (cause, vital) combination
        assert len(all_h) > 0
        causes = {h.cause for h in all_h}
        assert "zombie" in causes

    def test_mixed_vitals_forms_most_supported(self):
        tracker = HypothesisTracker()
        # 3 zombie deaths: all with low food, none with low health
        for _ in range(3):
            tracker.record({"zombie": 1.0}, {"food": 1.0, "health": 7.0})
        h = tracker.active_hypothesis()
        assert h is not None
        assert h.vital == "food"  # food < 3 is the correlated condition

    def test_surviving_episodes_do_not_count(self):
        tracker = HypothesisTracker()
        # 2 deaths + many survivals — still insufficient
        for _ in range(10):
            tracker.record({}, {})   # survived
        tracker.record({"zombie": 1.0}, {"food": 1.0})
        tracker.record({"zombie": 1.0}, {"food": 1.0})
        assert tracker.n_verifiable() == 0


# ---------------------------------------------------------------------------
# CuriosityStimulus
# ---------------------------------------------------------------------------

class TestCuriosityStimulus:
    def test_no_hypothesis_uses_pure_surprise(self):
        stim = CuriosityStimulus(weight=1.0, hypothesis=None)
        traj = _make_trajectory([5.0], confidences=[0.5])
        # avg_surprise = 0.5, relevance = 1.0
        assert stim.evaluate(traj) == pytest.approx(0.5)

    def test_with_hypothesis_multiplies_relevance(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        stim = CuriosityStimulus(weight=1.0, hypothesis=h)
        # Trajectory at threshold → relevance = 2.0
        traj = _make_trajectory([3.0], confidences=[0.5])
        # avg_surprise = 0.5, relevance = 2.0 → score = 1.0
        assert stim.evaluate(traj) == pytest.approx(1.0)

    def test_with_hypothesis_far_from_threshold(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        stim = CuriosityStimulus(weight=1.0, hypothesis=h)
        # Trajectory far above threshold → relevance = 1.0
        traj = _make_trajectory([9.0], confidences=[0.5])
        assert stim.evaluate(traj) == pytest.approx(0.5)

    def test_zero_surprise_gives_zero_score(self):
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0)
        stim = CuriosityStimulus(weight=1.0, hypothesis=h)
        traj = _make_trajectory([3.0], confidences=[1.0])  # no surprise
        assert stim.evaluate(traj) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PostMortemLearner.build_stimuli integration
# ---------------------------------------------------------------------------

class TestPostMortemLearnerWithHypothesis:
    def test_build_stimuli_without_hypothesis(self):
        learner = PostMortemLearner()
        layer = learner.build_stimuli(["food", "health"])
        assert len(layer.stimuli) == 2  # SurvivalAversion + HomeostasisStimulus

    def test_build_stimuli_with_hypothesis_adds_curiosity(self):
        learner = PostMortemLearner()
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=2, n_observed=3)
        layer = learner.build_stimuli(["food", "health"], hypothesis=h)
        assert len(layer.stimuli) == 3  # + CuriosityStimulus
        assert any(isinstance(s, CuriosityStimulus) for s in layer.stimuli)

    def test_curiosity_stimulus_has_hypothesis(self):
        learner = PostMortemLearner()
        h = DeathHypothesis(cause="zombie", vital="food", threshold=3.0,
                            n_supporting=2, n_observed=3)
        layer = learner.build_stimuli(["food"], hypothesis=h)
        curiosity = next(s for s in layer.stimuli if isinstance(s, CuriosityStimulus))
        assert curiosity.hypothesis is h


# ---------------------------------------------------------------------------
# Integration: tracker → hypothesis → curiosity reward
# ---------------------------------------------------------------------------

class TestEndToEndHypothesisCuriosity:
    def test_hypothesis_guides_curiosity_reward(self):
        """After N zombie deaths with low food, curiosity rewards low-food trajectories."""
        tracker = HypothesisTracker()
        for _ in range(3):
            tracker.record({"zombie": 1.0}, {"food": 1.5, "health": 5.0})

        h = tracker.active_hypothesis()
        assert h is not None

        stim = CuriosityStimulus(weight=0.1, hypothesis=h)

        # Low-food trajectory (near threshold) gets higher curiosity reward
        low_food_traj = _make_trajectory([2.0, 2.5], vital="food", confidences=[0.5, 0.5])
        high_food_traj = _make_trajectory([8.0, 8.5], vital="food", confidences=[0.5, 0.5])

        assert stim.evaluate(low_food_traj) > stim.evaluate(high_food_traj)
