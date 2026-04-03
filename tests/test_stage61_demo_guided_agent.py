"""Stage 61: Tests for Demo-Guided Agent.

Tests the demo-guided agent: causal planning, subgoal execution,
color selection, and state machine transitions.
TDD: tests written before/alongside implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.causal_world_model import CausalWorldModel
from snks.agent.demo_guided_agent import (
    ACT_LEFT,
    ACT_PICKUP,
    ACT_RIGHT,
    ACT_TOGGLE,
    COLOR_IDS,
    CausalPlanner,
    DemoGuidedAgent,
    ExecutableSubgoal,
    ExecutorState,
    SubgoalExecutor,
)
from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    OBJ_DOOR,
    OBJ_EMPTY,
    OBJ_GOAL,
    OBJ_KEY,
    OBJ_WALL,
    SpatialMap,
)


def _make_trained_model(colors: list[str] | None = None) -> CausalWorldModel:
    """Create a CausalWorldModel trained on standard colors."""
    if colors is None:
        colors = ["red", "green", "blue"]
    model = CausalWorldModel(dim=512, seed=42)
    model.learn_all_rules(colors)
    return model


def _make_spatial_map_with_objects(
    width: int = 8, height: int = 8,
    key_pos: tuple[int, int] | None = None,
    key_color: int = 0,
    door_pos: tuple[int, int] | None = None,
    door_color: int = 0,
    door_state: int = 2,  # locked
    goal_pos: tuple[int, int] | None = None,
) -> SpatialMap:
    """Create a SpatialMap pre-populated with objects."""
    sm = SpatialMap(width, height)
    # Fill with empty explored cells
    sm.grid[:, :, 0] = OBJ_EMPTY
    sm.grid[:, :, 1] = 0
    sm.grid[:, :, 2] = 0
    sm.explored[:] = True

    # Add walls around border
    sm.grid[0, :, 0] = OBJ_WALL
    sm.grid[-1, :, 0] = OBJ_WALL
    sm.grid[:, 0, 0] = OBJ_WALL
    sm.grid[:, -1, 0] = OBJ_WALL

    if key_pos is not None:
        sm.grid[key_pos[0], key_pos[1], 0] = OBJ_KEY
        sm.grid[key_pos[0], key_pos[1], 1] = key_color

    if door_pos is not None:
        sm.grid[door_pos[0], door_pos[1], 0] = OBJ_DOOR
        sm.grid[door_pos[0], door_pos[1], 1] = door_color
        sm.grid[door_pos[0], door_pos[1], 2] = door_state

    if goal_pos is not None:
        sm.grid[goal_pos[0], goal_pos[1], 0] = OBJ_GOAL

    return sm


class TestCausalPlanner:
    """Test CausalPlanner generates correct subgoals."""

    def test_generates_subgoals_for_locked_door(self):
        """Plan for pass_locked_door should produce 4 subgoals."""
        model = _make_trained_model()
        planner = CausalPlanner(model)
        sm = _make_spatial_map_with_objects(
            key_pos=(3, 2), key_color=0,  # red key
            door_pos=(3, 4), door_color=0,  # red door
            goal_pos=(3, 6),
        )

        subgoals = planner.plan("pass_locked_door", "red", sm)

        assert len(subgoals) == 4
        assert subgoals[0].name == "find_key"
        assert subgoals[1].name == "pickup_key"
        assert subgoals[1].action_at_target == ACT_PICKUP
        assert subgoals[2].name == "open_door"
        assert subgoals[2].action_at_target == ACT_TOGGLE
        assert subgoals[3].name == "pass_through"

    def test_subgoal_positions_from_map(self):
        """Subgoal targets should be populated from spatial map."""
        model = _make_trained_model()
        planner = CausalPlanner(model)
        sm = _make_spatial_map_with_objects(
            key_pos=(2, 3), key_color=0,
            door_pos=(4, 3), door_color=0,
            goal_pos=(6, 3),
        )

        subgoals = planner.plan("pass_locked_door", "red", sm)

        assert subgoals[1].target_pos == (2, 3)  # key pos
        assert subgoals[2].target_pos == (4, 3)  # door pos
        assert subgoals[3].target_pos == (6, 3)  # goal pos

    def test_replan_fills_missing_positions(self):
        """Replan should fill in target_pos as objects are discovered."""
        model = _make_trained_model()
        planner = CausalPlanner(model)
        sm = _make_spatial_map_with_objects(
            door_pos=(4, 3), door_color=0,
            # No key or goal yet
        )

        subgoals = planner.plan("pass_locked_door", "red", sm)
        # Key and goal positions should be None
        assert subgoals[0].target_pos is None  # find_key
        assert subgoals[1].target_pos is None  # pickup_key

        # Now "discover" the key
        sm.grid[2, 3, 0] = OBJ_KEY
        sm.grid[2, 3, 1] = 0  # red
        planner.replan(subgoals, sm)

        assert subgoals[0].target_pos == (2, 3)
        assert subgoals[1].target_pos == (2, 3)


class TestColorSelection:
    """Test causal model-driven color selection."""

    def test_correct_color_match(self):
        """Model should match same-color key to door."""
        model = _make_trained_model(["red", "green", "blue"])

        assert model.query_color_match("red", "red") is True
        assert model.query_color_match("blue", "blue") is True
        assert model.query_color_match("red", "blue") is False
        assert model.query_color_match("green", "red") is False

    def test_unseen_color_generalization(self):
        """Model should generalize to unseen colors via identity property."""
        model = _make_trained_model(["red", "green", "blue"])

        # Purple was not in training set
        assert model.query_color_match("purple", "purple") is True
        assert model.query_color_match("purple", "red") is False

    def test_precondition_returns_correct_color(self):
        """query_precondition should return matching key color."""
        model = _make_trained_model(["red", "green", "blue"])

        assert model.query_precondition("open", "red") == "red"
        assert model.query_precondition("open", "blue") == "blue"

    def test_planner_uses_correct_key_color(self):
        """Planner should select key of correct color for LockedRoom."""
        model = _make_trained_model(["red", "green", "blue"])
        planner = CausalPlanner(model)

        # Map has red key at (2,2), blue key at (2,5), blue door at (4,3)
        sm = _make_spatial_map_with_objects(
            door_pos=(4, 3), door_color=2,  # blue door
            goal_pos=(6, 3),
        )
        # Add red key
        sm.grid[2, 2, 0] = OBJ_KEY
        sm.grid[2, 2, 1] = 0  # red
        # Add blue key
        sm.grid[2, 5, 0] = OBJ_KEY
        sm.grid[2, 5, 1] = 2  # blue

        subgoals = planner.plan("pass_locked_door", "blue", sm)

        # pickup_key should target the blue key
        assert subgoals[1].target_pos == (2, 5)
        assert subgoals[1].target_color == 2  # blue


class TestSubgoalExecutor:
    """Test SubgoalExecutor state machine."""

    def _make_executor(self, width: int = 8, height: int = 8) -> tuple[SubgoalExecutor, SpatialMap]:
        sm = _make_spatial_map_with_objects(width, height)
        explorer = FrontierExplorer()
        pathfinder = GridPathfinder()
        executor = SubgoalExecutor(sm, explorer, pathfinder)
        return executor, sm

    def test_explore_when_target_unknown(self):
        """Should be in EXPLORE state when target_pos is None."""
        executor, sm = self._make_executor()
        sg = ExecutableSubgoal(
            name="find_key", target_pos=None,
            action_at_target=None, precondition=None,
            target_obj_type=OBJ_KEY,
        )
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        executor._update_state(sg, 3, 3, obs)
        assert executor.state == ExecutorState.EXPLORE

    def test_navigate_when_target_far(self):
        """Should be in NAVIGATE state when target is far."""
        executor, sm = self._make_executor()
        sg = ExecutableSubgoal(
            name="pickup_key", target_pos=(2, 2),
            action_at_target=ACT_PICKUP, precondition="adjacent",
            target_obj_type=OBJ_KEY,
        )
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        executor._update_state(sg, 5, 5, obs)
        assert executor.state == ExecutorState.NAVIGATE

    def test_interact_when_adjacent(self):
        """Should be in INTERACT state when adjacent to target."""
        executor, sm = self._make_executor()
        sg = ExecutableSubgoal(
            name="pickup_key", target_pos=(3, 3),
            action_at_target=ACT_PICKUP, precondition="adjacent",
            target_obj_type=OBJ_KEY,
        )
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        executor._update_state(sg, 3, 4, obs)  # adjacent
        assert executor.state == ExecutorState.INTERACT

    def test_subgoal_achievement_pickup(self):
        """pickup_key should be achieved when has_key is True."""
        executor, sm = self._make_executor()
        sg = ExecutableSubgoal(
            name="pickup_key", target_pos=(3, 3),
            action_at_target=ACT_PICKUP, precondition="adjacent",
        )
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        assert executor.is_subgoal_achieved(sg, obs, has_key=False, door_open=False) is False
        assert executor.is_subgoal_achieved(sg, obs, has_key=True, door_open=False) is True

    def test_subgoal_achievement_open_door(self):
        """open_door should be achieved when door_open is True."""
        executor, sm = self._make_executor()
        sg = ExecutableSubgoal(
            name="open_door", target_pos=(4, 3),
            action_at_target=ACT_TOGGLE, precondition="has_key",
        )
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        assert executor.is_subgoal_achieved(sg, obs, has_key=True, door_open=False) is False
        assert executor.is_subgoal_achieved(sg, obs, has_key=True, door_open=True) is True


class TestPreconditionChecks:
    """Test that precondition checks work correctly."""

    def test_cannot_open_without_key(self):
        """Causal model should say can't open without key."""
        model = _make_trained_model()
        assert model.query_can_act("open", has_key=False) is False
        assert model.query_can_act("open", has_key=True) is True

    def test_cannot_plan_without_training(self):
        """Untrained model should produce empty/failed plans."""
        model = CausalWorldModel(dim=512, seed=42)
        # Without learning, query_chain should fail
        chain = model.query_chain("pass_locked_door", "red")
        # Untrained SDMs have zero rewards — chain should fail at some step
        assert chain == ["cannot_plan"] or chain == []


class TestDemoGuidedAgentUnit:
    """Unit tests for DemoGuidedAgent without MiniGrid."""

    def test_learn_from_demos(self):
        """Agent should learn rules and be marked as trained."""
        agent = DemoGuidedAgent(grid_width=8, grid_height=8)
        assert agent._trained is False
        agent.learn_from_demos(["red", "green", "blue"])
        assert agent._trained is True
        # Verify causal model works
        assert agent.causal_model.query_color_match("red", "red") is True

    def test_stats_initial(self):
        """Initial stats should be zeros."""
        agent = DemoGuidedAgent(grid_width=8, grid_height=8)
        stats = agent.get_stats()
        assert stats["explore_steps"] == 0
        assert stats["execute_steps"] == 0
        assert stats["plan_ready"] is False

    def test_reset_clears_state(self):
        """Reset should clear all episode state."""
        agent = DemoGuidedAgent(grid_width=8, grid_height=8)
        agent.learn_from_demos(["red", "green", "blue"])
        agent.explore_steps = 10
        agent._has_key = True
        agent.plan_ready = True

        agent.reset()

        assert agent.explore_steps == 0
        assert agent._has_key is False
        assert agent.plan_ready is False
        # Training should persist across resets
        assert agent._trained is True
