"""Tests for OutcomeStimulus — the planner-side reader of the world-model outcome role."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from snks.agent.stimuli import OutcomeStimulus
from snks.agent.vector_world_model import VectorWorldModel


SMOKE_DIM = 8192
SMOKE_LOC = 10000


@dataclass
class _PlanStep:
    action: str
    target: str


@dataclass
class _Plan:
    steps: list
    origin: str = ""


@dataclass
class _Traj:
    plan: _Plan


def _train(model: VectorWorldModel, concept: str, action: str,
           survived: bool, damage: int, died_to: str | None) -> None:
    for _ in range(5):
        model.learn_outcome(concept, action, {
            "survived_h": survived,
            "damage_h": damage,
            "died_to": died_to,
        })


def _traj(action: str, target: str) -> _Traj:
    return _Traj(plan=_Plan(steps=[_PlanStep(action=action, target=target)]))


def test_no_model_returns_zero() -> None:
    stim = OutcomeStimulus(model=None, weight=1.0)
    assert stim.evaluate(_traj("do", "tree")) == 0.0


def test_unknown_pair_returns_zero() -> None:
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=23)
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "grass",
    )
    assert stim.evaluate(_traj("do", "tree")) == 0.0


def test_positive_recall_yields_positive_signal() -> None:
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=29)
    _train(model, "tree", "do", survived=True, damage=0, died_to=None)
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "grass",
    )
    score = stim.evaluate(_traj("do", "tree"))
    assert score > 0.0, f"survived-true context should produce positive signal, got {score}"


def test_death_recall_yields_strong_negative() -> None:
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=31)
    _train(model, "zombie", "do", survived=False, damage=9, died_to="zombie")
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "grass",
    )
    score = stim.evaluate(_traj("do", "zombie"))
    # died_penalty=3 + death_cause_penalty=5 + damage*0.25*9 ≈ 10.25 negative.
    assert score < -1.0, f"death context should produce strong negative signal, got {score}"


def test_per_candidate_differentiation() -> None:
    """Two candidates in the same context produce DIFFERENT signals."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=37)
    _train(model, "tree", "do", survived=True, damage=0, died_to=None)
    _train(model, "zombie", "do", survived=False, damage=9, died_to="zombie")
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "grass",
    )
    tree_score = stim.evaluate(_traj("do", "tree"))
    zomb_score = stim.evaluate(_traj("do", "zombie"))
    # The previous bundled-context substrate failed because tree_score == zomb_score.
    # This is the core property of the new design.
    assert tree_score > 0.0 > zomb_score, (
        f"Per-candidate signals must differ: tree={tree_score:.3f}, zombie={zomb_score:.3f}"
    )
    assert abs(tree_score - zomb_score) > 1.0, (
        f"Signals must differ substantially, got delta={abs(tree_score - zomb_score):.3f}"
    )


def test_motion_plan_uses_near_concept() -> None:
    """A `move_left` plan queries the provided near concept, not 'self'."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=41)
    _train(model, "lava", "move_left", survived=False, damage=9, died_to="lava")
    _train(model, "grass", "move_left", survived=True, damage=0, died_to=None)
    near = {"value": "lava"}

    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: near["value"],
    )
    move_left_self = _traj("move_left", "self")

    # Facing lava → strong negative.
    near["value"] = "lava"
    assert stim.evaluate(move_left_self) < -1.0

    # Facing grass → positive.
    near["value"] = "grass"
    assert stim.evaluate(move_left_self) > 0.0


def test_baseline_plan_uses_near_concept() -> None:
    """An empty plan (baseline) queries `(near, 'noop')`."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=43)
    _train(model, "lava", "noop", survived=False, damage=9, died_to="lava")
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "lava",
    )
    baseline = _Traj(plan=_Plan(steps=[]))
    assert stim.evaluate(baseline) < -1.0


def test_weight_scales_signal_linearly() -> None:
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=47)
    _train(model, "tree", "do", survived=True, damage=0, died_to=None)
    base = OutcomeStimulus(
        model=model, weight=1.0, near_concept_provider=lambda: "grass",
    ).evaluate(_traj("do", "tree"))
    doubled = OutcomeStimulus(
        model=model, weight=2.0, near_concept_provider=lambda: "grass",
    ).evaluate(_traj("do", "tree"))
    assert abs(doubled - 2.0 * base) < 1e-5, (
        f"weight should scale signal linearly: base={base:.4f}, doubled={doubled:.4f}"
    )
