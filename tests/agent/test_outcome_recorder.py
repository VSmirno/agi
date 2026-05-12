"""Integration test for _OutcomeRecorder lifecycle helper in vector_mpc_agent."""

from __future__ import annotations

from dataclasses import dataclass

from snks.agent.stimuli import resolve_outcome_pair
from snks.agent.vector_mpc_agent import _OutcomeRecorder
from snks.agent.vector_world_model import VectorWorldModel


SMOKE_DIM = 8192
SMOKE_LOC = 10000


@dataclass
class _PlanStep:
    action: str
    target: str


def test_push_flush_due_writes_outcome() -> None:
    """A pushed snapshot whose horizon elapses gets written via learn_outcome."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=53)
    rec = _OutcomeRecorder(model=model, horizon=3, resolver=resolve_outcome_pair)

    # Decision at step 0: chose `single:tree:do`, full health.
    rec.push(step=0, plan_steps=[_PlanStep("do", "tree")],
             near="grass", health_now=9.0)
    # Horizon is 3 — flush at step 3, not before.
    assert rec.flush_due(current_step=2, health_now=9.0) == 0
    assert rec.flush_due(current_step=3, health_now=9.0) == 1

    # Predict_outcome should now retrieve the alive outcome.
    decoded, conf = model.predict_outcome("tree", "do")
    assert decoded is not None, f"Expected recall after flush, got conf={conf:.3f}"
    assert decoded["survived_h"] is True


def test_flush_on_death_writes_died_to() -> None:
    """Pending snapshots at death flush with (survived=False, died_to=cause)."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=59)
    rec = _OutcomeRecorder(model=model, horizon=5, resolver=resolve_outcome_pair)

    rec.push(step=10, plan_steps=[_PlanStep("do", "zombie")],
             near="grass", health_now=9.0)
    # Death happens at step 13, before horizon elapsed.
    flushed = rec.flush_on_death(health_now=0.0, died_to="zombie")
    assert flushed == 1

    decoded, conf = model.predict_outcome("zombie", "do")
    assert decoded is not None
    assert decoded["survived_h"] is False
    assert decoded["died_to"] == "zombie"


def test_motion_plan_uses_near_concept_for_recording() -> None:
    """A motion-plan push writes under (near_concept, move_*)."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=61)
    rec = _OutcomeRecorder(model=model, horizon=2, resolver=resolve_outcome_pair)

    rec.push(step=0, plan_steps=[_PlanStep("move_left", "self")],
             near="lava", health_now=9.0)
    # Health drops 8 over horizon — agent stepped into lava.
    rec.flush_due(current_step=2, health_now=1.0)

    decoded, _ = model.predict_outcome("lava", "move_left")
    assert decoded is not None
    assert decoded["damage_h"] >= 5, decoded


def test_baseline_plan_records_against_near_noop() -> None:
    """An empty plan (baseline) records under (near, 'noop')."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=67)
    rec = _OutcomeRecorder(model=model, horizon=1, resolver=resolve_outcome_pair)
    rec.push(step=0, plan_steps=[], near="grass", health_now=9.0)
    rec.flush_due(current_step=1, health_now=9.0)
    decoded, _ = model.predict_outcome("grass", "noop")
    assert decoded is not None


def test_unresolvable_plan_is_skipped() -> None:
    """A motion plan with no near concept available is silently skipped."""
    model = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=71)
    rec = _OutcomeRecorder(model=model, horizon=1, resolver=resolve_outcome_pair)
    rec.push(step=0, plan_steps=[_PlanStep("move_left", "self")],
             near=None, health_now=9.0)
    # Nothing pushed; flush_due returns 0.
    assert rec.flush_due(current_step=1, health_now=9.0) == 0
