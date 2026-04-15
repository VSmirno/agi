"""Tests for Stage 84 StimuliLayer (Category 4 — IDEOLOGY v2)."""

from __future__ import annotations

import pytest

from snks.agent.stimuli import (
    HomeostasisStimulus,
    StimuliLayer,
    SurvivalAversion,
)
from snks.agent.vector_sim import (
    VectorPlan,
    VectorState,
    VectorTrajectory,
    score_trajectory,
)


def make_trajectory(terminated: bool = False, body: dict | None = None) -> VectorTrajectory:
    state = VectorState(
        inventory={"wood": 0},
        body=body or {"health": 5.0, "food": 4.0, "drink": 3.0, "energy": 9.0},
    )
    return VectorTrajectory(
        plan=VectorPlan(steps=[]),
        states=[state],
        terminated=terminated,
    )


# ---------------------------------------------------------------------------
# SurvivalAversion
# ---------------------------------------------------------------------------

class TestSurvivalAversion:
    def test_terminated_returns_penalty(self):
        traj = make_trajectory(terminated=True)
        s = SurvivalAversion(weight=1000.0)
        assert s.evaluate(traj) == -1000.0

    def test_alive_returns_zero(self):
        traj = make_trajectory(terminated=False)
        s = SurvivalAversion()
        assert s.evaluate(traj) == 0.0

    def test_custom_weight(self):
        traj = make_trajectory(terminated=True)
        s = SurvivalAversion(weight=500.0)
        assert s.evaluate(traj) == -500.0


# ---------------------------------------------------------------------------
# HomeostasisStimulus
# ---------------------------------------------------------------------------

class TestHomeostasisStimulus:
    def test_no_thresholds_zero_penalty(self):
        # Default thresholds={} → no deficit → zero score regardless of vitals.
        traj = make_trajectory(body={"health": 5.0, "food": 4.0, "drink": 3.0, "energy": 9.0})
        s = HomeostasisStimulus(vital_vars=["health", "food", "drink", "energy"])
        assert s.evaluate(traj) == pytest.approx(0.0)

    def test_deficit_below_threshold_negative(self):
        # food=2, threshold=5 → deficit=3 → score=-3
        traj = make_trajectory(body={"food": 2.0})
        s = HomeostasisStimulus(weight=1.0, vital_vars=["food"],
                                thresholds={"food": 5.0})
        assert s.evaluate(traj) == pytest.approx(-3.0)

    def test_weight_scales_deficit(self):
        traj = make_trajectory(body={"food": 3.0})
        s = HomeostasisStimulus(weight=2.0, vital_vars=["food"],
                                thresholds={"food": 5.0})
        assert s.evaluate(traj) == pytest.approx(-4.0)

    def test_empty_final_state_returns_zero(self):
        traj = VectorTrajectory(plan=VectorPlan(steps=[]), states=[])
        s = HomeostasisStimulus()
        assert s.evaluate(traj) == 0.0

    def test_vital_above_threshold_zero_penalty(self):
        traj = make_trajectory(body={"food": 7.0})
        s = HomeostasisStimulus(vital_vars=["food"], thresholds={"food": 3.0})
        assert s.evaluate(traj) == pytest.approx(0.0)

    def test_missing_vital_defaults_zero_body(self):
        # vital missing from body → body.get(v, 0.0) = 0, threshold=0 → deficit=0
        traj = make_trajectory(body={"health": 5.0})
        s = HomeostasisStimulus(vital_vars=["health", "food", "drink", "energy"])
        assert s.evaluate(traj) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# StimuliLayer
# ---------------------------------------------------------------------------

class TestStimuliLayer:
    def test_empty_layer_returns_zero(self):
        traj = make_trajectory()
        layer = StimuliLayer(stimuli=[])
        assert layer.evaluate(traj) == 0.0

    def test_single_stimulus(self):
        traj = make_trajectory(terminated=True)
        layer = StimuliLayer(stimuli=[SurvivalAversion(weight=100.0)])
        assert layer.evaluate(traj) == pytest.approx(-100.0)

    def test_two_stimuli_sum(self):
        # terminated=False → SurvivalAversion=0
        # HomeostasisStimulus with thresholds: food deficit=1 (threshold=5, food=4)
        traj = make_trajectory(
            terminated=False,
            body={"health": 5.0, "food": 4.0, "drink": 3.0, "energy": 9.0},
        )
        layer = StimuliLayer(stimuli=[
            SurvivalAversion(weight=1000.0),
            HomeostasisStimulus(weight=1.0, vital_vars=["food"],
                                thresholds={"food": 5.0}),
        ])
        # 0.0 + (-1.0) = -1.0
        assert layer.evaluate(traj) == pytest.approx(-1.0)

    def test_death_penalty_dominates(self):
        traj = make_trajectory(
            terminated=True,
            body={"health": 0.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        )
        layer = StimuliLayer(stimuli=[
            SurvivalAversion(weight=1000.0),
            HomeostasisStimulus(weight=1.0),
        ])
        assert layer.evaluate(traj) < 0


# ---------------------------------------------------------------------------
# score_trajectory integration
# ---------------------------------------------------------------------------

class TestScoreTrajectoryWithStimuli:
    def test_with_stimuli_returns_3tuple(self):
        traj = make_trajectory()
        layer = StimuliLayer(stimuli=[SurvivalAversion()])
        score = score_trajectory(traj, stimuli=layer)
        assert len(score) == 3

    def test_stimuli_score_at_position_0(self):
        traj = make_trajectory(terminated=True)
        layer = StimuliLayer(stimuli=[SurvivalAversion(weight=999.0)])
        score = score_trajectory(traj, stimuli=layer)
        assert score[0] == pytest.approx(-999.0)

    def test_without_stimuli_survived_at_position_0(self):
        alive = make_trajectory(terminated=False)
        dead = make_trajectory(terminated=True)
        s_alive = score_trajectory(alive)
        s_dead = score_trajectory(dead)
        assert s_alive > s_dead  # (1, ...) > (0, ...)

    def test_without_stimuli_returns_3tuple(self):
        traj = make_trajectory()
        score = score_trajectory(traj, stimuli=None)
        assert len(score) == 3

    def test_alive_beats_dead_with_stimuli(self):
        alive = make_trajectory(
            terminated=False,
            body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        )
        dead = make_trajectory(terminated=True, body={"health": 0.0})
        layer = StimuliLayer(stimuli=[SurvivalAversion(), HomeostasisStimulus()])
        s_alive = score_trajectory(alive, stimuli=layer)
        s_dead = score_trajectory(dead, stimuli=layer)
        assert s_alive > s_dead
