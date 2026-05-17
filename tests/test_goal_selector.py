"""Tests for Stage 85: GoalSelector and Goal.progress()."""

from __future__ import annotations

from pathlib import Path

import pytest

from snks.agent.goal_selector import Goal, GoalSelector
from snks.agent.vector_sim import DynamicEntityState, VectorPlan, VectorState, VectorTrajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trajectory(
    inv_start: dict | None = None,
    inv_end: dict | None = None,
    body_start: dict | None = None,
    body_end: dict | None = None,
    confidences: list[float] | None = None,
) -> VectorTrajectory:
    s0 = VectorState(
        inventory=inv_start or {},
        body=body_start or {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
    )
    s1 = VectorState(
        inventory=inv_end or inv_start or {},
        body=body_end or body_start or {"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
    )
    return VectorTrajectory(
        plan=VectorPlan(steps=[]),
        states=[s0, s1],
        confidences=confidences or [],
    )


def make_state(body: dict | None = None, inventory: dict | None = None) -> VectorState:
    return VectorState(
        inventory=inventory or {},
        body=body or {"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        spatial_map=None,
    )


TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


@pytest.fixture
def textbook():
    from snks.agent.crafter_textbook import CrafterTextbook
    return CrafterTextbook(str(TEXTBOOK_PATH))


@pytest.fixture
def selector(textbook):
    return GoalSelector(textbook)


# ---------------------------------------------------------------------------
# Goal.progress — vital deltas
# ---------------------------------------------------------------------------

class TestGoalProgressVitals:
    def test_find_cow_positive_food_delta(self):
        traj = make_trajectory(
            body_start={"health": 5.0, "food": 3.0, "drink": 5.0, "energy": 5.0},
            body_end={"health": 5.0, "food": 8.0, "drink": 5.0, "energy": 5.0},
        )
        assert Goal("find_cow").progress(traj) == pytest.approx(5.0)

    def test_find_cow_negative_food_delta_clamped(self):
        traj = make_trajectory(
            body_start={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            body_end={"health": 5.0, "food": 2.0, "drink": 5.0, "energy": 5.0},
        )
        assert Goal("find_cow").progress(traj) == pytest.approx(0.0)

    def test_find_water_positive_drink_delta(self):
        traj = make_trajectory(
            body_start={"health": 5.0, "food": 5.0, "drink": 2.0, "energy": 5.0},
            body_end={"health": 5.0, "food": 5.0, "drink": 7.0, "energy": 5.0},
        )
        assert Goal("find_water").progress(traj) == pytest.approx(5.0)

    def test_sleep_positive_energy_delta(self):
        traj = make_trajectory(
            body_start={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 2.0},
            body_end={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 7.0},
        )
        assert Goal("sleep").progress(traj) == pytest.approx(5.0)

    def test_fight_zombie_progresses_on_targeted_do_plan(self):
        from snks.agent.vector_sim import VectorPlanStep
        traj = make_trajectory(
            body_start={"health": 3.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            body_end={"health": 7.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        )
        traj.plan = VectorPlan(steps=[VectorPlanStep(action="do", target="zombie")])
        assert Goal("fight_zombie").progress(traj) == pytest.approx(1.0)

    def test_fight_zombie_does_not_progress_from_self_healing(self):
        traj = make_trajectory(
            body_start={"health": 3.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            body_end={"health": 7.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        )
        assert Goal("fight_zombie").progress(traj) == pytest.approx(0.0)

    def test_single_state_trajectory_returns_zero(self):
        s = make_state()
        traj = VectorTrajectory(plan=VectorPlan(steps=[]), states=[s])
        assert Goal("find_cow").progress(traj) == 0.0


# ---------------------------------------------------------------------------
# Goal.progress — inventory
# ---------------------------------------------------------------------------

class TestGoalProgressInventory:
    def test_gather_wood_positive_delta(self):
        traj = make_trajectory(
            inv_start={"wood": 0},
            inv_end={"wood": 3},
        )
        assert Goal("gather_wood").progress(traj) == pytest.approx(3.0)

    def test_gather_wood_no_gain_returns_zero(self):
        traj = make_trajectory(inv_start={"wood": 2}, inv_end={"wood": 2})
        assert Goal("gather_wood").progress(traj) == pytest.approx(0.0)

    def test_craft_wood_sword_gained(self):
        traj = make_trajectory(
            inv_start={"wood": 2, "wood_sword": 0},
            inv_end={"wood": 1, "wood_sword": 1},
        )
        assert Goal("craft_wood_sword").progress(traj) == pytest.approx(1.0)

    def test_craft_wood_sword_not_gained(self):
        traj = make_trajectory(
            inv_start={"wood_sword": 0},
            inv_end={"wood_sword": 0},
        )
        assert Goal("craft_wood_sword").progress(traj) == pytest.approx(0.0)

    def test_craft_wood_sword_already_had_sword(self):
        # item_gained requires start=0 → end>0; if already had it, returns False
        traj = make_trajectory(
            inv_start={"wood_sword": 1},
            inv_end={"wood_sword": 2},
        )
        assert Goal("craft_wood_sword").progress(traj) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Goal.progress — explore (confidence-based)
# ---------------------------------------------------------------------------

class TestGoalProgressExplore:
    def test_low_confidence_high_surprise(self):
        traj = make_trajectory(confidences=[0.1, 0.2, 0.1])
        # avg_confidence = 0.133 → surprise = 0.867
        prog = Goal("explore").progress(traj)
        assert prog == pytest.approx(1.0 - (0.1 + 0.2 + 0.1) / 3, rel=1e-4)

    def test_high_confidence_low_surprise(self):
        traj = make_trajectory(confidences=[0.9, 0.95, 0.85])
        prog = Goal("explore").progress(traj)
        assert prog < 0.2

    def test_empty_confidences_returns_zero(self):
        traj = make_trajectory(confidences=[])
        assert Goal("explore").progress(traj) == 0.0

    def test_self_action_trajectory_returns_zero(self):
        """Sleep plan (target=self) must not count as exploration — avoids sleeping at full vitals."""
        from snks.agent.vector_sim import VectorPlanStep
        s0 = VectorState(inventory={}, body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        s1 = VectorState(inventory={}, body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        plan = VectorPlan(steps=[VectorPlanStep(action="sleep", target="self")])
        traj = VectorTrajectory(plan=plan, states=[s0, s1], confidences=[0.5])
        assert Goal("explore").progress(traj) == 0.0


# ---------------------------------------------------------------------------
# GoalSelector.select — vital-based priorities
# ---------------------------------------------------------------------------

class TestGoalSelectorSelect:
    def test_full_vitals_returns_explore(self, selector):
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        goal = selector.select(state)
        assert goal.id == "gather_wood"

    def test_low_food_returns_find_cow(self, selector):
        state = make_state(body={"health": 9.0, "food": 2.0, "drink": 9.0, "energy": 9.0})
        goal = selector.select(state)
        assert goal.id == "find_cow"

    def test_low_drink_returns_find_water(self, selector):
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 2.0, "energy": 9.0})
        goal = selector.select(state)
        assert goal.id == "find_water"

    def test_low_energy_returns_sleep(self, selector):
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 2.0})
        goal = selector.select(state)
        assert goal.id == "sleep"

    def test_critical_health_beats_low_food(self, selector):
        # health < 2 → critical → find_cow (not low food rule which also returns find_cow,
        # but critical health is checked first at priority 2)
        state = make_state(body={"health": 1.0, "food": 2.5, "drink": 9.0, "energy": 9.0})
        goal = selector.select(state)
        # health < 2 triggers critical health → find_cow
        assert goal.id == "find_cow"

    def test_no_spatial_map_no_entity_threat(self, selector):
        # No spatial map → entity threats inactive → falls through to vitals
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        assert state.spatial_map is None
        goal = selector.select(state)
        assert goal.id == "gather_wood"

    def test_dynamic_arrow_overrides_proactive_gather_goal(self, selector):
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 0, "wood_sword": 1},
        )
        state.dynamic_entities = [
            DynamicEntityState(concept_id="arrow", position=(9, 10), velocity=(1, 0))
        ]
        goal = selector.select(state)
        assert goal.id == "fight_skeleton"

    def test_dynamic_zombie_overrides_proactive_gather_goal(self, selector):
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 0, "wood_sword": 1},
        )
        state.dynamic_entities = [
            DynamicEntityState(concept_id="zombie", position=(11, 10), velocity=(-1, 0))
        ]
        goal = selector.select(state)
        assert goal.id == "fight_zombie"
        assert goal.requested_capability == "armed_melee"
        assert goal.reason == "dynamic_threat_present"

    def test_dynamic_zombie_without_weapon_returns_craft_with_parent_goal(self, selector):
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 5, "wood_sword": 0},
        )
        state.dynamic_entities = [
            DynamicEntityState(concept_id="zombie", position=(11, 10), velocity=(-1, 0))
        ]

        goal = selector.select(state)

        assert goal.id == "craft_wood_sword"
        assert goal.parent_goal == "fight_zombie"
        assert goal.requested_capability == "armed_melee"
        assert goal.blocked_by == "missing:wood_sword"
        assert goal.reason == "required_weapon_missing"

    def test_low_drink_overrides_dynamic_zombie(self, selector):
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 2.0, "energy": 9.0},
            inventory={"wood": 5, "wood_sword": 1},
        )
        state.dynamic_entities = [
            DynamicEntityState(concept_id="zombie", position=(11, 10), velocity=(-1, 0))
        ]

        goal = selector.select(state)

        assert goal.id == "find_water"
        assert goal.reason == "low_drink"

    def test_dynamic_goals_can_be_disabled(self, textbook):
        selector = GoalSelector(textbook, allow_dynamic_entity_goals=False)
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 0, "wood_sword": 0},
        )
        state.dynamic_entities = [
            DynamicEntityState(concept_id="arrow", position=(9, 10), velocity=(1, 0))
        ]
        goal = selector.select(state)
        assert goal.id == "gather_wood"


# ---------------------------------------------------------------------------
# GoalSelector — textbook derivation sanity check
# ---------------------------------------------------------------------------

class TestGoalSelectorTextbookDerivation:
    def test_threats_list_nonempty(self, selector):
        """Entity threats + critical + 3 vital + proactive crafting."""
        assert len(selector._threats) >= 6

    def test_proactive_crafting_triggers_gather_wood(self, selector):
        """No wood AND no wood_sword → gather_wood goal."""
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 0, "wood_sword": 0},
        )
        goal = selector.select(state)
        assert goal.id == "gather_wood"

    def test_proactive_crafting_inactive_when_has_sword(self, selector):
        """Current textbook-derived priorities still keep a proactive gather goal active."""
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood_sword": 1},
        )
        goal = selector.select(state)
        assert goal.id == "gather_wood"

    def test_proactive_crafting_inactive_when_has_enough_wood(self, selector):
        """With enough wood, selector currently falls through to another proactive gather goal."""
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 5, "wood_sword": 0},
        )
        goal = selector.select(state)
        assert goal.id == "gather_stone_item"

    def test_proactive_crafting_still_active_with_partial_wood(self, selector):
        """Has wood=2 (< chain_cost=5) → still gathering needed."""
        state = make_state(
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            inventory={"wood": 2, "wood_sword": 0},
        )
        goal = selector.select(state)
        assert goal.id == "gather_wood"

    def test_goals_block_loaded(self, textbook):
        assert textbook.goals_block.get("primary") == "survive"


# ---------------------------------------------------------------------------
# Phase 1 — goal-conditioned frontier exploration: target_concept derivation
# ---------------------------------------------------------------------------

class TestGoalTargetDerivation:
    def test_find_water_resolves_to_water_via_textbook(self, selector):
        targets = selector._goal_targets
        assert targets.get("find_water") == "water"
        assert targets.get("find_drink") == "water"

    def test_find_cow_resolves_to_cow_via_textbook(self, selector):
        targets = selector._goal_targets
        assert targets.get("find_cow") == "cow"
        assert targets.get("find_food") == "cow"

    def test_fight_zombie_and_skeleton_resolve_via_remove_entity(self, selector):
        targets = selector._goal_targets
        assert targets.get("fight_zombie") == "zombie"
        assert targets.get("fight_skeleton") == "skeleton"

    def test_gather_wood_resolves_to_tree_via_inventory_delta(self, selector):
        targets = selector._goal_targets
        assert targets.get("gather_wood") == "tree"
        assert targets.get("gather_tree") == "tree"

    def test_select_attaches_target_concept_on_vital_goal(self, selector):
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 2.0, "energy": 9.0})
        goal = selector.select(state)
        assert goal.id == "find_water"
        assert goal.target_concept == "water"

    def test_select_attaches_target_concept_on_critical_health(self, selector):
        state = make_state(body={"health": 1.0, "food": 5.0, "drink": 5.0, "energy": 5.0})
        goal = selector.select(state)
        assert goal.id == "find_cow"
        assert goal.target_concept == "cow"

    def test_select_attaches_target_concept_on_dynamic_threat_fight(self, selector):
        state = make_state(inventory={"wood_sword": 1})
        state.dynamic_entities = [DynamicEntityState(concept_id="zombie", position=(11, 10))]
        goal = selector.select(state)
        assert goal.id == "fight_zombie"
        assert goal.target_concept == "zombie"

    def test_explore_goal_has_no_target_concept(self, selector):
        state = make_state()
        goal = selector.select(state)
        # default fallback when nothing else matches; even if proactive
        # crafting fires, target_concept must still be either None or a
        # textbook-resolvable concept — never silently invented.
        if goal.target_concept is not None:
            assert goal.target_concept in selector._goal_targets.values()


# ---------------------------------------------------------------------------
# Phase 1 — Goal.progress frontier epsilon
# ---------------------------------------------------------------------------

class TestFrontierProgressEpsilon:
    def _frontier_traj(self, target: str) -> VectorTrajectory:
        from snks.agent.vector_sim import VectorPlanStep
        plan = VectorPlan(
            steps=[VectorPlanStep(action="frontier_seek", target=target)],
            origin=f"frontier:{target}",
        )
        s = VectorState(inventory={}, body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0})
        return VectorTrajectory(plan=plan, states=[s, s], confidences=[])

    def test_find_water_with_frontier_water_plan_returns_epsilon(self):
        from snks.agent.goal_selector import FRONTIER_PROGRESS_EPSILON
        goal = Goal("find_water", target_concept="water")
        traj = self._frontier_traj("water")
        assert goal.progress(traj) == pytest.approx(FRONTIER_PROGRESS_EPSILON)

    def test_find_water_with_frontier_cow_plan_returns_zero(self):
        goal = Goal("find_water", target_concept="water")
        traj = self._frontier_traj("cow")
        assert goal.progress(traj) == 0.0

    def test_find_water_with_baseline_plan_returns_zero(self):
        goal = Goal("find_water", target_concept="water")
        s = VectorState(inventory={}, body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0})
        traj = VectorTrajectory(plan=VectorPlan(steps=[], origin="baseline"), states=[s, s], confidences=[])
        assert goal.progress(traj) == 0.0

    def test_real_drink_delta_dominates_frontier_epsilon(self):
        from snks.agent.vector_sim import VectorPlanStep
        from snks.agent.goal_selector import FRONTIER_PROGRESS_EPSILON
        goal = Goal("find_water", target_concept="water")
        plan = VectorPlan(
            steps=[VectorPlanStep(action="do", target="water")],
            origin="single:water:do",
        )
        s0 = VectorState(inventory={}, body={"health": 5.0, "food": 5.0, "drink": 2.0, "energy": 5.0})
        s1 = VectorState(inventory={}, body={"health": 5.0, "food": 5.0, "drink": 7.0, "energy": 5.0})
        traj = VectorTrajectory(plan=plan, states=[s0, s1], confidences=[])
        assert goal.progress(traj) > FRONTIER_PROGRESS_EPSILON

    def test_fight_zombie_frontier_zombie_plan_returns_epsilon(self):
        from snks.agent.goal_selector import FRONTIER_PROGRESS_EPSILON
        goal = Goal("fight_zombie", target_concept="zombie")
        traj = self._frontier_traj("zombie")
        assert goal.progress(traj) == pytest.approx(FRONTIER_PROGRESS_EPSILON)

    def test_target_concept_none_disables_epsilon(self):
        goal = Goal("find_water")  # no target_concept set
        traj = self._frontier_traj("water")
        assert goal.progress(traj) == 0.0

    def test_to_trace_includes_target_concept(self):
        goal = Goal("find_water", target_concept="water")
        trace = goal.to_trace()
        assert trace["target_concept"] == "water"
        assert trace["id"] == "find_water"


# ---------------------------------------------------------------------------
# Phase 2A — distance-based fight priority
# ---------------------------------------------------------------------------

class TestFightPriorityByDistance:
    def test_nearest_zombie_wins_over_far_skeleton(self, selector):
        state = make_state(inventory={"wood_sword": 1})
        state.dynamic_entities = [
            DynamicEntityState(concept_id="skeleton", position=(20, 10)),
            DynamicEntityState(concept_id="zombie", position=(11, 10)),
        ]
        goal = selector.select(state)
        assert goal.id == "fight_zombie"
        assert goal.target_concept == "zombie"

    def test_nearest_skeleton_wins_over_far_zombie(self, selector):
        state = make_state(inventory={"wood_sword": 1})
        state.dynamic_entities = [
            DynamicEntityState(concept_id="zombie", position=(20, 10)),
            DynamicEntityState(concept_id="skeleton", position=(11, 10)),
        ]
        goal = selector.select(state)
        assert goal.id == "fight_skeleton"

    def test_arrow_routes_to_fight_skeleton(self, selector):
        state = make_state(inventory={"wood_sword": 1})
        state.dynamic_entities = [
            DynamicEntityState(concept_id="arrow", position=(11, 10)),
        ]
        goal = selector.select(state)
        assert goal.id == "fight_skeleton"

    def test_unarmed_promotes_craft_subgoal_with_fight_parent(self, selector):
        state = make_state(inventory={"wood": 5})  # has crafting material, no sword
        state.dynamic_entities = [
            DynamicEntityState(concept_id="zombie", position=(11, 10)),
        ]
        goal = selector.select(state)
        assert goal.id == "craft_wood_sword"
        assert goal.parent_goal == "fight_zombie"
