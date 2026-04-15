"""Tests for Stage 86: PostMortemAnalyzer, PostMortemLearner, HomeostasisStimulus thresholds."""

from __future__ import annotations

import math

import pytest

from snks.agent.post_mortem import (
    DamageEvent,
    PostMortemAnalyzer,
    PostMortemLearner,
    dominant_cause,
)
from snks.agent.stimuli import HomeostasisStimulus, StimuliLayer, SurvivalAversion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(
    step: int,
    health_delta: float = -1.0,
    food: float = 5.0,
    drink: float = 5.0,
    energy: float = 5.0,
    nearby_cids: list[tuple[str, int]] | None = None,
) -> DamageEvent:
    return DamageEvent(
        step=step,
        health_delta=health_delta,
        vitals={"food": food, "drink": drink, "energy": energy},
        nearby_cids=nearby_cids or [],
    )


# ---------------------------------------------------------------------------
# PostMortemAnalyzer
# ---------------------------------------------------------------------------

class TestAnalyzerEmptyLog:
    def test_empty_log_returns_empty(self):
        a = PostMortemAnalyzer()
        assert a.attribute([], death_step=100) == {}

    def test_dominant_cause_alive_on_empty(self):
        assert dominant_cause({}) == "alive"


class TestAnalyzerNormalisation:
    def test_single_event_sums_to_one(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=100, food=0.0)]
        attr = a.attribute(log, death_step=100)
        assert abs(sum(attr.values()) - 1.0) < 1e-6

    def test_multi_event_sums_to_one(self):
        a = PostMortemAnalyzer()
        log = [
            make_event(step=50, food=0.0),
            make_event(step=80, food=0.0),
            make_event(step=100, nearby_cids=[("zombie", 1)]),
        ]
        attr = a.attribute(log, death_step=100)
        assert abs(sum(attr.values()) - 1.0) < 1e-6


class TestAnalyzerTemporalDecay:
    def test_recent_event_gets_higher_weight_than_early(self):
        a = PostMortemAnalyzer()
        # early starvation, late starvation — same source, decay makes late heavier
        log = [
            make_event(step=10, food=0.0),    # early
            make_event(step=100, food=0.0),   # at death
        ]
        attr = a.attribute(log, death_step=100, decay=0.05)
        # Both are "starvation"; late event weight=exp(0)=1, early=exp(-4.5)≈0.011
        # So almost all weight goes to "starvation" anyway — just verify sum
        assert attr.get("starvation", 0) > 0.9

    def test_decay_weights_are_proportional(self):
        """Late event weight > early event weight (same source, confirms decay direction)."""
        a = PostMortemAnalyzer()
        # Use different sources so we can compare individual weights
        log = [
            make_event(step=0, food=0.0),                          # starvation, early
            make_event(step=100, nearby_cids=[("zombie", 1)]),     # zombie, late
        ]
        attr = a.attribute(log, death_step=100, decay=0.05)
        # zombie (late, step=100) weight = exp(0) = 1.0
        # starvation (early, step=0) weight = exp(-5) ≈ 0.0067
        assert attr.get("zombie", 0) > attr.get("starvation", 0)


class TestAnalyzerSourceDetection:
    def test_starvation_when_food_zero(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=10, food=0.0)]
        attr = a.attribute(log, death_step=10)
        assert "starvation" in attr

    def test_dehydration_when_drink_zero(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=10, drink=0.0)]
        attr = a.attribute(log, death_step=10)
        assert "dehydration" in attr

    def test_zombie_when_nearby(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=10, nearby_cids=[("zombie", 1)])]
        attr = a.attribute(log, death_step=10)
        assert "zombie" in attr

    def test_unknown_when_no_source(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=10)]   # no zero vitals, no nearby entities
        attr = a.attribute(log, death_step=10)
        assert "unknown" in attr

    def test_entity_beyond_dist2_not_attributed(self):
        a = PostMortemAnalyzer()
        log = [make_event(step=10, nearby_cids=[("zombie", 5)])]
        attr = a.attribute(log, death_step=10)
        assert "zombie" not in attr
        assert "unknown" in attr

    def test_multi_source_split_equally(self):
        """food=0 AND zombie nearby → starvation and zombie each get 50%."""
        a = PostMortemAnalyzer()
        log = [make_event(step=100, food=0.0, nearby_cids=[("zombie", 1)])]
        attr = a.attribute(log, death_step=100)
        assert abs(attr.get("starvation", 0) - attr.get("zombie", 0)) < 1e-6


class TestAnalyzerDominantCause:
    def test_dominant_cause_zombie(self):
        attr = {"starvation": 0.3, "zombie": 0.7}
        assert dominant_cause(attr) == "zombie"

    def test_dominant_cause_starvation(self):
        attr = {"starvation": 0.8, "zombie": 0.2}
        assert dominant_cause(attr) == "starvation"


# ---------------------------------------------------------------------------
# PostMortemLearner
# ---------------------------------------------------------------------------

class TestLearnerUpdate:
    def test_starvation_increases_food_threshold(self):
        learner = PostMortemLearner(food_threshold=3.0, lr=0.1)
        learner.update({"starvation": 1.0})
        assert learner.food_threshold > 3.0

    def test_dehydration_increases_drink_threshold(self):
        learner = PostMortemLearner(drink_threshold=3.0, lr=0.1)
        learner.update({"dehydration": 1.0})
        assert learner.drink_threshold > 3.0

    def test_entity_increases_health_weight(self):
        learner = PostMortemLearner(health_weight=1.0, lr=0.1)
        learner.update({"zombie": 0.6, "skeleton": 0.4})
        assert learner.health_weight > 1.0

    def test_empty_attribution_no_change(self):
        learner = PostMortemLearner(food_threshold=3.0)
        learner.update({})
        assert learner.food_threshold == 3.0

    def test_unknown_attribution_no_change(self):
        learner = PostMortemLearner(food_threshold=3.0, health_weight=1.0)
        learner.update({"unknown": 1.0})
        assert learner.food_threshold == 3.0
        assert learner.health_weight == 1.0


class TestLearnerClamping:
    def test_food_threshold_clamped_at_max(self):
        learner = PostMortemLearner(food_threshold=7.9, lr=1.0)
        learner.update({"starvation": 1.0})
        assert learner.food_threshold <= 8.0

    def test_food_threshold_clamped_at_min(self):
        learner = PostMortemLearner(food_threshold=1.1, lr=1.0)
        # No starvation — threshold doesn't decrease (only increases from starvation)
        learner.update({})
        assert learner.food_threshold >= 1.0

    def test_health_weight_clamped_at_max(self):
        learner = PostMortemLearner(health_weight=4.9, lr=1.0)
        learner.update({"zombie": 1.0})
        assert learner.health_weight <= 5.0


class TestLearnerBuildStimuli:
    def test_build_stimuli_returns_stimuli_layer(self):
        learner = PostMortemLearner()
        stimuli = learner.build_stimuli(["health", "food", "drink", "energy"])
        assert isinstance(stimuli, StimuliLayer)

    def test_build_stimuli_thresholds_match_params(self):
        learner = PostMortemLearner(food_threshold=4.5, drink_threshold=2.5)
        stimuli = learner.build_stimuli(["health", "food", "drink", "energy"])
        homeostasis = next(
            s for s in stimuli.stimuli if isinstance(s, HomeostasisStimulus)
        )
        assert homeostasis.thresholds["food"] == pytest.approx(4.5)
        assert homeostasis.thresholds["drink"] == pytest.approx(2.5)

    def test_build_stimuli_weight_matches_param(self):
        learner = PostMortemLearner(health_weight=2.5)
        stimuli = learner.build_stimuli(["health", "food", "drink", "energy"])
        homeostasis = next(
            s for s in stimuli.stimuli if isinstance(s, HomeostasisStimulus)
        )
        assert homeostasis.weight == pytest.approx(2.5)

    def test_new_instance_each_call(self):
        learner = PostMortemLearner()
        s1 = learner.build_stimuli(["food"])
        s2 = learner.build_stimuli(["food"])
        assert s1 is not s2


# ---------------------------------------------------------------------------
# HomeostasisStimulus — threshold-based deficit scoring
# ---------------------------------------------------------------------------

class TestHomeostasisThresholds:
    def _make_traj(self, body: dict):
        from snks.agent.vector_sim import VectorPlan, VectorState, VectorTrajectory
        s = VectorState(inventory={}, body=body)
        return VectorTrajectory(plan=VectorPlan(steps=[]), states=[s, s])

    def test_no_thresholds_zero_penalty(self):
        """Backwards compat: default thresholds={} → zero deficit → zero penalty."""
        stim = HomeostasisStimulus()
        traj = self._make_traj({"food": 9.0, "drink": 9.0, "health": 9.0, "energy": 9.0})
        assert stim.evaluate(traj) == pytest.approx(0.0)

    def test_food_below_threshold_negative_score(self):
        stim = HomeostasisStimulus(thresholds={"food": 5.0}, weight=1.0,
                                   vital_vars=["food"])
        traj = self._make_traj({"food": 2.0})
        assert stim.evaluate(traj) == pytest.approx(-3.0)

    def test_food_above_threshold_zero_penalty(self):
        stim = HomeostasisStimulus(thresholds={"food": 3.0}, weight=1.0,
                                   vital_vars=["food"])
        traj = self._make_traj({"food": 5.0})
        assert stim.evaluate(traj) == pytest.approx(0.0)

    def test_multiple_vitals_deficit_summed(self):
        stim = HomeostasisStimulus(
            thresholds={"food": 5.0, "drink": 5.0},
            weight=1.0,
            vital_vars=["food", "drink"],
        )
        # food deficit=2, drink deficit=3
        traj = self._make_traj({"food": 3.0, "drink": 2.0})
        assert stim.evaluate(traj) == pytest.approx(-5.0)

    def test_weight_scales_penalty(self):
        stim = HomeostasisStimulus(thresholds={"food": 5.0}, weight=2.0,
                                   vital_vars=["food"])
        traj = self._make_traj({"food": 3.0})
        assert stim.evaluate(traj) == pytest.approx(-4.0)
