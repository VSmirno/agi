"""Tests for Stage 85: GoalSelector and Goal.progress()."""

from __future__ import annotations

from pathlib import Path

import pytest

from snks.agent.goal_selector import Goal, GoalSelector
from snks.agent.vector_sim import VectorPlan, VectorState, VectorTrajectory


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

    def test_fight_zombie_positive_health_delta(self):
        traj = make_trajectory(
            body_start={"health": 3.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            body_end={"health": 7.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        )
        assert Goal("fight_zombie").progress(traj) == pytest.approx(4.0)

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


# ---------------------------------------------------------------------------
# GoalSelector.select — vital-based priorities
# ---------------------------------------------------------------------------

class TestGoalSelectorSelect:
    def test_full_vitals_returns_explore(self, selector):
        state = make_state(body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        goal = selector.select(state)
        assert goal.id == "explore"

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
        assert goal.id == "explore"


# ---------------------------------------------------------------------------
# GoalSelector — textbook derivation sanity check
# ---------------------------------------------------------------------------

class TestGoalSelectorTextbookDerivation:
    def test_threats_list_nonempty(self, selector):
        """At least zombie + skeleton + 3 vital threats."""
        assert len(selector._threats) >= 5

    def test_goals_block_loaded(self, textbook):
        assert textbook.goals_block.get("primary") == "survive"
