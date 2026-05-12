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


def test_survived_recall_contributes_zero() -> None:
    """A survived recall produces no signal — the stimulus is death-only.

    The previous design boosted known-safe pairs which systematically
    pulled the planner away from never-tried pairs (e.g. crafting).
    Survival is the default expectation, so it contributes nothing; only
    death warnings shift ranking.
    """
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=29)
    _train(model, "tree", "do", survived=True, damage=0, died_to=None)
    stim = OutcomeStimulus(
        model=model,
        weight=1.0,
        near_concept_provider=lambda: "grass",
    )
    score = stim.evaluate(_traj("do", "tree"))
    assert score == 0.0, f"survived-true recall should contribute zero, got {score}"


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
    """Two candidates in the same context produce DIFFERENT signals.

    Under the death-only semantics: a survived recall is 0, a death
    recall is strongly negative. The differentiation property (which the
    previous bundled-context substrate lacked) holds when one of the two
    candidates has a death recall — `0 vs -10` reorders ranking exactly
    the same way `+1 vs -10` would, while keeping never-tried candidates
    on equal footing with known-safe ones.
    """
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
    assert tree_score == 0.0
    assert zomb_score < -1.0, f"zombie death recall should be strongly negative, got {zomb_score}"


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

    # Facing lava → strong negative (death recall).
    near["value"] = "lava"
    assert stim.evaluate(move_left_self) < -1.0

    # Facing grass → zero (survived recall does not contribute).
    near["value"] = "grass"
    assert stim.evaluate(move_left_self) == 0.0


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
    """Weight linearly scales the (negative) death-recall signal."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=47)
    _train(model, "zombie", "do", survived=False, damage=9, died_to="zombie")
    base = OutcomeStimulus(
        model=model, weight=1.0, near_concept_provider=lambda: "grass",
    ).evaluate(_traj("do", "zombie"))
    doubled = OutcomeStimulus(
        model=model, weight=2.0, near_concept_provider=lambda: "grass",
    ).evaluate(_traj("do", "zombie"))
    assert abs(doubled - 2.0 * base) < 1e-5, (
        f"weight should scale signal linearly: base={base:.4f}, doubled={doubled:.4f}"
    )
