"""Stage 77a Commit 1: Unit tests for forward_sim_types dataclasses.

Tests are pure data-layer: constructors, equality, helper methods.
No simulation logic — that's Commit 4.
"""

from __future__ import annotations

import pytest

from snks.agent.concept_store import CausalLink
from snks.agent.forward_sim_types import (
    EFFECT_KINDS,
    EVENT_KINDS,
    FAILURE_KINDS,
    DynamicEntity,
    Failure,
    Plan,
    PlannedStep,
    RuleEffect,
    SimEvent,
    SimState,
    StatefulCondition,
    Trajectory,
)


# ---------------------------------------------------------------------------
# StatefulCondition
# ---------------------------------------------------------------------------


def _mkstate(body: dict[str, float]) -> SimState:
    """Minimal SimState for StatefulCondition tests."""
    return SimState(
        inventory={},
        body=body,
        player_pos=(0, 0),
        dynamic_entities=[],
        spatial_map=None,
        last_action=None,
        step=0,
    )


class TestStatefulCondition:
    def test_greater_than_true(self):
        cond = StatefulCondition(var="food", op=">", threshold=0)
        assert cond.satisfied(_mkstate({"food": 5.0})) is True

    def test_greater_than_false(self):
        cond = StatefulCondition(var="food", op=">", threshold=0)
        assert cond.satisfied(_mkstate({"food": 0.0})) is False

    def test_less_than(self):
        cond = StatefulCondition(var="food", op="<", threshold=3)
        assert cond.satisfied(_mkstate({"food": 2.0})) is True
        assert cond.satisfied(_mkstate({"food": 3.0})) is False

    def test_equal(self):
        cond = StatefulCondition(var="food", op="==", threshold=0)
        assert cond.satisfied(_mkstate({"food": 0.0})) is True
        assert cond.satisfied(_mkstate({"food": 1.0})) is False

    def test_greater_or_equal(self):
        cond = StatefulCondition(var="drink", op=">=", threshold=5)
        assert cond.satisfied(_mkstate({"drink": 5.0})) is True
        assert cond.satisfied(_mkstate({"drink": 6.0})) is True
        assert cond.satisfied(_mkstate({"drink": 4.0})) is False

    def test_less_or_equal(self):
        cond = StatefulCondition(var="health", op="<=", threshold=2)
        assert cond.satisfied(_mkstate({"health": 2.0})) is True
        assert cond.satisfied(_mkstate({"health": 1.0})) is True
        assert cond.satisfied(_mkstate({"health": 3.0})) is False

    def test_unknown_op_raises(self):
        cond = StatefulCondition(var="food", op="!=", threshold=0)
        with pytest.raises(ValueError, match="unknown op"):
            cond.satisfied(_mkstate({"food": 5.0}))

    def test_missing_var_defaults_to_zero(self):
        cond = StatefulCondition(var="missing", op=">", threshold=0)
        # Missing vars default to 0.0 in body.get, so `> 0` is False
        assert cond.satisfied(_mkstate({"food": 5.0})) is False


# ---------------------------------------------------------------------------
# RuleEffect
# ---------------------------------------------------------------------------


class TestRuleEffect:
    def test_gather_effect(self):
        effect = RuleEffect(kind="gather", inventory_delta={"wood": 1})
        assert effect.kind == "gather"
        assert effect.inventory_delta == {"wood": 1}
        assert effect.body_delta == {}
        assert effect.scene_remove is None

    def test_remove_effect(self):
        effect = RuleEffect(kind="remove", scene_remove="zombie")
        assert effect.scene_remove == "zombie"
        assert effect.inventory_delta == {}

    def test_movement_effect(self):
        effect = RuleEffect(kind="movement", movement_behavior="chase_player")
        assert effect.movement_behavior == "chase_player"

    def test_spatial_effect_with_range(self):
        effect = RuleEffect(
            kind="spatial",
            body_delta={"health": -2.0},
            spatial_range=1,
        )
        assert effect.spatial_range == 1
        assert effect.body_delta == {"health": -2.0}

    def test_stateful_effect(self):
        cond = StatefulCondition(var="food", op=">", threshold=0)
        effect = RuleEffect(
            kind="stateful",
            body_delta={"health": 0.1},
            stateful_condition=cond,
        )
        assert effect.stateful_condition is cond

    def test_body_rate_effect(self):
        effect = RuleEffect(
            kind="body_rate",
            body_rate=-0.04,
            body_rate_variable="food",
        )
        assert effect.body_rate == -0.04
        assert effect.body_rate_variable == "food"

    def test_place_effect(self):
        effect = RuleEffect(
            kind="place",
            world_place=("table", "adjacent_empty"),
            inventory_delta={"wood": -2},
        )
        assert effect.world_place == ("table", "adjacent_empty")
        assert effect.inventory_delta == {"wood": -2}

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="RuleEffect.kind must be one of"):
            RuleEffect(kind="invalid_kind")

    def test_all_declared_kinds_valid(self):
        # Sanity: constructing each declared kind doesn't raise
        for kind in EFFECT_KINDS:
            RuleEffect(kind=kind)


# ---------------------------------------------------------------------------
# DynamicEntity
# ---------------------------------------------------------------------------


class TestDynamicEntity:
    def test_construct(self):
        e = DynamicEntity(concept_id="zombie", pos=(5, 10))
        assert e.concept_id == "zombie"
        assert e.pos == (5, 10)

    def test_equality(self):
        e1 = DynamicEntity(concept_id="zombie", pos=(5, 10))
        e2 = DynamicEntity(concept_id="zombie", pos=(5, 10))
        assert e1 == e2

    def test_inequality_different_pos(self):
        e1 = DynamicEntity(concept_id="zombie", pos=(5, 10))
        e2 = DynamicEntity(concept_id="zombie", pos=(5, 11))
        assert e1 != e2


# ---------------------------------------------------------------------------
# SimState
# ---------------------------------------------------------------------------


class TestSimState:
    def _make(self):
        return SimState(
            inventory={"wood": 2},
            body={"health": 9.0, "food": 7.0, "drink": 8.0, "energy": 9.0},
            player_pos=(10, 20),
            dynamic_entities=[DynamicEntity("zombie", (15, 20))],
            spatial_map=None,
            last_action="move_right",
            step=5,
        )

    def test_copy_is_deep_for_dicts(self):
        s1 = self._make()
        s2 = s1.copy()
        s2.inventory["wood"] = 99
        s2.body["health"] = 0
        # Original untouched
        assert s1.inventory["wood"] == 2
        assert s1.body["health"] == 9.0

    def test_copy_is_deep_for_entities(self):
        s1 = self._make()
        s2 = s1.copy()
        s2.dynamic_entities[0].pos = (100, 100)
        # Original entity position untouched
        assert s1.dynamic_entities[0].pos == (15, 20)

    def test_copy_shares_spatial_map(self):
        """spatial_map is intentionally a shared reference (read-only in rollout)."""
        s1 = self._make()
        s2 = s1.copy()
        assert s2.spatial_map is s1.spatial_map  # identity, not copy

    def test_is_dead_false_with_full_body(self):
        s = self._make()
        vital_mins = {"health": 0}
        assert s.is_dead(vital_mins) is False

    def test_is_dead_true_on_zero_health(self):
        s = self._make()
        s.body["health"] = 0.0
        vital_mins = {"health": 0}
        assert s.is_dead(vital_mins) is True

    def test_is_dead_true_on_negative(self):
        s = self._make()
        s.body["health"] = -0.5
        vital_mins = {"health": 0}
        assert s.is_dead(vital_mins) is True

    def test_is_dead_respects_nonzero_vital_min(self):
        s = self._make()
        s.body["health"] = 2.0
        vital_mins = {"health": 3}
        # health (2) <= vital_min (3) → dead
        assert s.is_dead(vital_mins) is True

    def test_non_vital_var_at_zero_not_dead(self):
        """food=0 is not death in Crafter — only vital vars count."""
        s = self._make()
        s.body["food"] = 0.0
        # Only health is vital; food=0 without health=0 → not dead
        vital_mins = {"health": 0}
        assert s.is_dead(vital_mins) is False

    def test_empty_vital_mins_never_dead(self):
        s = self._make()
        s.body["health"] = -100.0  # catastrophically low
        # But if nothing is marked vital → not "dead" per this semantics
        assert s.is_dead({}) is False


# ---------------------------------------------------------------------------
# SimEvent
# ---------------------------------------------------------------------------


class TestSimEvent:
    def test_construct_body_delta(self):
        e = SimEvent(
            step=10,
            kind="body_delta",
            var="health",
            amount=-2.0,
            source="zombie",
        )
        assert e.step == 10
        assert e.var == "health"
        assert e.amount == -2.0
        assert e.source == "zombie"

    def test_construct_rule_applied(self):
        e = SimEvent(
            step=5,
            kind="rule_applied",
            var=None,
            amount=0.0,
            source="do:tree",
        )
        assert e.kind == "rule_applied"
        assert e.var is None


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------


class TestTrajectory:
    def _empty_state(self):
        return SimState(
            inventory={},
            body={"health": 9.0, "food": 9.0},
            player_pos=(0, 0),
            dynamic_entities=[],
            spatial_map=None,
            last_action=None,
            step=0,
        )

    def test_failure_step_none_on_clean_series(self):
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series={"health": [9.0, 8.5, 8.0]},
            events=[],
            final_state=self._empty_state(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        assert traj.failure_step("health") is None

    def test_failure_step_returns_first_zero(self):
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series={"health": [9.0, 5.0, 2.0, 0.0, -1.0]},
            events=[],
            final_state=self._empty_state(),
            terminated=True,
            terminated_reason="body_dead",
            plan_progress=0,
        )
        assert traj.failure_step("health") == 3

    def test_failure_step_missing_var(self):
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series={"health": [9.0]},
            events=[],
            final_state=self._empty_state(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        assert traj.failure_step("food") is None

    def test_tick_count(self):
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series={"health": [9, 8, 7], "food": [9, 8, 7]},
            events=[],
            final_state=self._empty_state(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        assert traj.tick_count() == 3

    def test_tick_count_empty(self):
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series={},
            events=[],
            final_state=self._empty_state(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        assert traj.tick_count() == 0


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------


class TestFailure:
    def test_var_depleted(self):
        f = Failure(kind="var_depleted", var="health", cause=None, step=9)
        assert f.kind == "var_depleted"
        assert f.var == "health"
        assert f.severity == 1.0  # default

    def test_attributed_to(self):
        f = Failure(
            kind="attributed_to",
            var=None,
            cause="zombie",
            step=5,
            severity=2.0,
        )
        assert f.cause == "zombie"
        assert f.severity == 2.0

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="Failure.kind must be one of"):
            Failure(kind="nonsense", var=None, cause=None, step=0)

    def test_all_declared_kinds_valid(self):
        for kind in FAILURE_KINDS:
            Failure(kind=kind, var=None, cause=None, step=0)


# ---------------------------------------------------------------------------
# PlannedStep and Plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_planned_step_construct(self):
        link = CausalLink(action="do", kind="action_triggered")
        step = PlannedStep(action="do", target="tree", near=None, rule=link)
        assert step.action == "do"
        assert step.target == "tree"
        assert step.rule is link

    def test_plan_empty(self):
        plan = Plan(steps=[])
        assert plan.steps == []
        assert plan.origin == "unknown"

    def test_plan_with_origin(self):
        step = PlannedStep(action="inertia", target=None, near=None, rule=None)
        plan = Plan(steps=[step], origin="baseline")
        assert plan.origin == "baseline"
        assert len(plan.steps) == 1


# ---------------------------------------------------------------------------
# CausalLink structured effect
# ---------------------------------------------------------------------------


class TestCausalLinkStructured:
    """CausalLink post-Stage 77a: structured `effect: RuleEffect`, no `result`."""

    def test_effect_field(self):
        effect = RuleEffect(kind="gather", inventory_delta={"wood": 1})
        link = CausalLink(
            action="do",
            kind="action_triggered",
            effect=effect,
            concept="tree",
        )
        assert link.effect is effect
        assert link.effect.kind == "gather"
        assert link.concept == "tree"

    def test_default_kind(self):
        link = CausalLink(action="do")
        assert link.kind == "action_triggered"
