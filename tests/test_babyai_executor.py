"""Tests for Stage 24c: GridPerception, GridNavigator, BabyAIExecutor."""

from __future__ import annotations

import pytest
import torch

from snks.language.grid_perception import (
    GridPerception,
    GridObject,
    SKS_AGENT,
    INTERACTIVE_OBJECTS,
)
from snks.language.grid_navigator import (
    GridNavigator,
    _bfs,
    _turn_actions,
    _direction_to,
    ACT_LEFT,
    ACT_RIGHT,
    ACT_FORWARD,
)
from snks.language.grounding_map import GroundingMap


# ─── Mock MiniGrid objects ───────────────────────────────────────────────


class MockCell:
    """Mock MiniGrid cell."""

    def __init__(self, obj_type: str, color: str = "red", is_open: bool = False):
        self.type = obj_type
        self.color = color
        self.is_open = is_open


class MockGrid:
    """Mock MiniGrid grid."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells: dict[tuple[int, int], MockCell | None] = {}

    def get(self, x: int, y: int) -> MockCell | None:
        return self._cells.get((x, y))

    def set(self, x: int, y: int, cell: MockCell | None) -> None:
        self._cells[(x, y)] = cell


def make_walled_grid(w: int, h: int) -> MockGrid:
    """Create a grid with walls on borders, empty inside."""
    grid = MockGrid(w, h)
    for x in range(w):
        grid.set(x, 0, MockCell("wall"))
        grid.set(x, h - 1, MockCell("wall"))
    for y in range(h):
        grid.set(0, y, MockCell("wall"))
        grid.set(w - 1, y, MockCell("wall"))
    return grid


# ─── GridPerception tests ────────────────────────────────────────────────


class TestGridPerception:
    def test_register_object(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        sks_id = perc.register_object("key", "red")
        assert sks_id >= 100
        assert gmap.word_to_sks("red key") == sks_id
        assert gmap.sks_to_word(sks_id) == "red key"

    def test_register_bare_noun(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        sks_id = perc.register_object("ball", "blue")
        assert gmap.word_to_sks("ball") == sks_id

    def test_register_color_as_attr(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        perc.register_object("key", "red")
        assert gmap.word_to_sks("red") is not None

    def test_stable_ids(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        id1 = perc.register_object("key", "red")
        id2 = perc.register_object("key", "red")
        assert id1 == id2

    def test_different_objects_different_ids(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        id1 = perc.register_object("key", "red")
        id2 = perc.register_object("ball", "blue")
        assert id1 != id2

    def test_perceive(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(6, 6)
        grid.set(2, 2, MockCell("key", "red"))
        grid.set(3, 3, MockCell("ball", "blue"))

        active = perc.perceive(grid, (1, 1), 0)
        assert SKS_AGENT in active
        assert len(active) >= 3  # agent + key + ball

    def test_perceive_skips_walls(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)

        active = perc.perceive(grid, (2, 2), 0)
        # Only agent, no interactive objects.
        assert active == {SKS_AGENT}

    def test_find_object_composite(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(6, 6)
        grid.set(2, 2, MockCell("key", "red"))
        grid.set(3, 3, MockCell("key", "blue"))

        perc.perceive(grid, (1, 1), 0)
        obj = perc.find_object("red key")
        assert obj is not None
        assert obj.color == "red"
        assert obj.pos == (2, 2)

    def test_find_object_bare_noun(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(6, 6)
        grid.set(2, 2, MockCell("ball", "green"))

        perc.perceive(grid, (1, 1), 0)
        obj = perc.find_object("ball")
        assert obj is not None
        assert obj.obj_type == "ball"

    def test_find_object_by_sks(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(6, 6)
        grid.set(2, 2, MockCell("key", "red"))

        perc.perceive(grid, (1, 1), 0)
        sks_id = perc.register_object("key", "red")
        obj = perc.find_object_by_sks(sks_id)
        assert obj is not None
        assert obj.pos == (2, 2)


# ─── GridNavigator tests ────────────────────────────────────────────────


class TestGridNavigator:
    def test_turn_actions_same_dir(self):
        assert _turn_actions(0, 0) == []

    def test_turn_actions_right(self):
        assert _turn_actions(0, 1) == [ACT_RIGHT]

    def test_turn_actions_opposite(self):
        turns = _turn_actions(0, 2)
        assert len(turns) == 2

    def test_turn_actions_left(self):
        assert _turn_actions(1, 0) == [ACT_LEFT]

    def test_bfs_simple(self):
        grid = make_walled_grid(5, 5)
        path = _bfs(grid, (1, 1), (3, 1))
        assert path is not None
        assert path[0] == (1, 1)
        assert path[-1] == (3, 1)

    def test_bfs_no_path(self):
        grid = make_walled_grid(5, 5)
        # Block the only path.
        grid.set(2, 1, MockCell("wall"))
        grid.set(2, 2, MockCell("wall"))
        grid.set(2, 3, MockCell("wall"))
        path = _bfs(grid, (1, 1), (3, 1))
        assert path is None

    def test_bfs_same_position(self):
        grid = make_walled_grid(5, 5)
        path = _bfs(grid, (2, 2), (2, 2))
        assert path == [(2, 2)]

    def test_plan_path_straight(self):
        nav = GridNavigator()
        grid = make_walled_grid(5, 5)
        # Agent at (1,2) facing right (dir=0), target at (3,2).
        actions = nav.plan_path(grid, (1, 2), 0, (3, 2))
        assert ACT_FORWARD in actions
        assert len(actions) == 2  # forward, forward

    def test_plan_path_with_turn(self):
        nav = GridNavigator()
        grid = make_walled_grid(5, 5)
        # Agent at (2,1) facing right (dir=0), target at (2,3) — need to go down.
        actions = nav.plan_path(grid, (2, 1), 0, (2, 3))
        assert ACT_RIGHT in actions  # turn to face down
        assert ACT_FORWARD in actions

    def test_plan_path_stop_adjacent(self):
        nav = GridNavigator()
        grid = make_walled_grid(6, 6)
        grid.set(3, 2, MockCell("key", "red"))
        # Agent at (1,2) facing right, target at (3,2). Stop adjacent = (2,2).
        actions = nav.plan_path(grid, (1, 2), 0, (3, 2), stop_adjacent=True)
        # Should navigate to (2,2) and face right toward (3,2).
        assert ACT_FORWARD in actions

    def test_plan_path_already_there(self):
        nav = GridNavigator()
        grid = make_walled_grid(5, 5)
        actions = nav.plan_path(grid, (2, 2), 0, (2, 2))
        assert actions == []


# ─── BabyAIExecutor tests ───────────────────────────────────────────────


class MockEnv:
    """Mock MiniGrid environment for testing BabyAIExecutor."""

    def __init__(self, grid: MockGrid, agent_pos: tuple[int, int], agent_dir: int = 0):
        self._grid = grid
        self._agent_pos = list(agent_pos)
        self._agent_dir = agent_dir
        self._steps = 0
        self._target_pos: tuple[int, int] | None = None
        self.unwrapped = self

    @property
    def grid(self):
        return self._grid

    @property
    def agent_pos(self):
        return self._agent_pos

    @property
    def agent_dir(self):
        return self._agent_dir

    def set_target(self, pos: tuple[int, int]):
        """Set the target position — reward given when agent faces it from adjacent."""
        self._target_pos = pos

    def step(self, action: int):
        """Execute action, return (obs, reward, terminated, truncated, info)."""
        self._steps += 1
        dir_vec = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

        if action == 0:  # left
            self._agent_dir = (self._agent_dir - 1) % 4
        elif action == 1:  # right
            self._agent_dir = (self._agent_dir + 1) % 4
        elif action == 2:  # forward
            dx, dy = dir_vec[self._agent_dir]
            nx, ny = self._agent_pos[0] + dx, self._agent_pos[1] + dy
            cell = self._grid.get(nx, ny)
            if cell is None or cell.type in ("empty", "floor"):
                self._agent_pos = [nx, ny]

        # Check if agent is adjacent to and facing target.
        reward = 0.0
        terminated = False
        if self._target_pos is not None:
            dx, dy = dir_vec[self._agent_dir]
            facing = (self._agent_pos[0] + dx, self._agent_pos[1] + dy)
            if facing == self._target_pos:
                # Agent faces target from adjacent cell.
                if action in (2, 3, 5, 6):  # forward, pickup, toggle, done
                    reward = 0.9
                    terminated = True

        obs = {"image": None, "direction": self._agent_dir, "mission": ""}
        return obs, reward, terminated, False, {}


class TestBabyAIExecutor:
    def _make_simple_env(self):
        """5x5 walled grid, red key at (3,2), agent at (1,2) facing right."""
        grid = make_walled_grid(5, 5)
        grid.set(3, 2, MockCell("key", "red"))
        env = MockEnv(grid, (1, 2), 0)
        env.set_target((3, 2))
        return env

    def test_parse_goto(self):
        from snks.language.babyai_executor import BabyAIExecutor

        env = self._make_simple_env()
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute("go to the red key")
        assert result.parsed_action == "go to"
        assert result.parsed_object == "key"

    def test_goto_success(self):
        from snks.language.babyai_executor import BabyAIExecutor

        env = self._make_simple_env()
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute("go to the red key")
        assert result.success
        assert result.reward > 0
        assert result.steps_taken > 0

    def test_pickup_success(self):
        from snks.language.babyai_executor import BabyAIExecutor

        env = self._make_simple_env()
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute("pick up the red key")
        assert result.success
        assert result.parsed_action == "pick up"

    def test_object_not_found(self):
        from snks.language.babyai_executor import BabyAIExecutor

        env = self._make_simple_env()
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute("go to the purple ball")
        assert not result.success
        assert "not found" in result.error

    def test_no_action_parsed(self):
        from snks.language.babyai_executor import BabyAIExecutor

        env = self._make_simple_env()
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute("")
        assert not result.success
        assert "no action" in result.error

    def test_extract_roles(self):
        from snks.language.babyai_executor import BabyAIExecutor
        from snks.language.chunker import Chunk

        action, obj, attr = BabyAIExecutor._extract_roles([
            Chunk("pick up", "ACTION"),
            Chunk("red", "ATTR"),
            Chunk("key", "OBJECT"),
        ])
        assert action == "pick up"
        assert obj == "key"
        assert attr == "red"
