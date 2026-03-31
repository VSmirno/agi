"""Tests for Stage 25: GoalAgent, CausalLearner, BlockingAnalyzer."""

from __future__ import annotations

import pytest

from snks.agent.causal_model import CausalWorldModel, _split_context, _STATE_SKS_RANGE
from snks.daf.types import CausalAgentConfig
from snks.language.blocking_analyzer import BlockingAnalyzer, Blocker, SubGoal
from snks.language.causal_learner import CausalLearner
from snks.language.goal_agent import GoalAgent, EpisodeResult
from snks.language.grid_navigator import GridNavigator, PathStatus, PathResult
from snks.language.grid_perception import (
    GridPerception,
    SKS_AGENT,
    SKS_KEY_PRESENT,
    SKS_KEY_HELD,
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
    SKS_GOAL_PRESENT,
)
from snks.language.grounding_map import GroundingMap


# ─── Mock MiniGrid objects ───────────────────────────────────────────


class MockCell:
    def __init__(self, obj_type: str, color: str = "yellow", is_open: bool = False, is_locked: bool = False):
        self.type = obj_type
        self.color = color
        self.is_open = is_open
        self.is_locked = is_locked


class MockGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._cells: dict[tuple[int, int], MockCell | None] = {}

    def get(self, x: int, y: int) -> MockCell | None:
        return self._cells.get((x, y))

    def set(self, x: int, y: int, cell: MockCell | None) -> None:
        self._cells[(x, y)] = cell


def make_walled_grid(w: int, h: int) -> MockGrid:
    grid = MockGrid(w, h)
    for x in range(w):
        grid.set(x, 0, MockCell("wall"))
        grid.set(x, h - 1, MockCell("wall"))
    for y in range(h):
        grid.set(0, y, MockCell("wall"))
        grid.set(w - 1, y, MockCell("wall"))
    return grid


class DoorKeyMockEnv:
    """Mock DoorKey environment with pickup/toggle state machine."""

    def __init__(self, grid: MockGrid, agent_pos: tuple[int, int], agent_dir: int = 0):
        self._grid = grid
        self._agent_pos = list(agent_pos)
        self._agent_dir = agent_dir
        self.carrying = None
        self._steps = 0
        self._goal_pos: tuple[int, int] | None = None
        self._key_pos: tuple[int, int] | None = None
        self._door_pos: tuple[int, int] | None = None
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

    def set_key(self, pos: tuple[int, int], color: str = "yellow"):
        self._key_pos = pos
        self._grid.set(pos[0], pos[1], MockCell("key", color))

    def set_door(self, pos: tuple[int, int], color: str = "yellow", locked: bool = True):
        self._door_pos = pos
        self._grid.set(pos[0], pos[1], MockCell("door", color, is_open=False, is_locked=locked))

    def set_goal(self, pos: tuple[int, int]):
        self._goal_pos = pos
        self._grid.set(pos[0], pos[1], MockCell("goal", "green"))

    def step(self, action: int):
        self._steps += 1
        dir_vec = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}
        reward = 0.0
        terminated = False

        if action == 0:  # left
            self._agent_dir = (self._agent_dir - 1) % 4
        elif action == 1:  # right
            self._agent_dir = (self._agent_dir + 1) % 4
        elif action == 2:  # forward
            dx, dy = dir_vec[self._agent_dir]
            nx, ny = self._agent_pos[0] + dx, self._agent_pos[1] + dy
            cell = self._grid.get(nx, ny)
            if cell is None or cell.type in ("empty", "floor", "goal"):
                self._agent_pos = [nx, ny]
                # Check if reached goal.
                if self._goal_pos and (nx, ny) == self._goal_pos:
                    reward = 1.0 - 0.9 * (self._steps / 200)
                    terminated = True
            elif cell.type == "door" and cell.is_open:
                self._agent_pos = [nx, ny]
        elif action == 3:  # pickup
            dx, dy = dir_vec[self._agent_dir]
            fx, fy = self._agent_pos[0] + dx, self._agent_pos[1] + dy
            cell = self._grid.get(fx, fy)
            if cell and cell.type == "key" and self.carrying is None:
                self.carrying = cell
                self._grid.set(fx, fy, None)
        elif action == 5:  # toggle
            dx, dy = dir_vec[self._agent_dir]
            fx, fy = self._agent_pos[0] + dx, self._agent_pos[1] + dy
            cell = self._grid.get(fx, fy)
            if cell and cell.type == "door":
                if cell.is_locked and self.carrying and self.carrying.type == "key":
                    cell.is_locked = False
                    cell.is_open = True
                elif not cell.is_locked:
                    cell.is_open = not cell.is_open

        obs = {"image": None, "direction": self._agent_dir, "mission": ""}
        return obs, reward, terminated, False, {}


def make_doorkey_env():
    """Standard 5x5 DoorKey mock with dividing wall + locked door.

    Layout (5x5):
    WWWWW
    W.W.W
    W.D.W   D=locked door at (2,2)
    WKW.W   K=key at (1,3), wall at (2,1) and (2,3)
    WWWWW
    Goal at (3,1). Agent at (1,1) facing right.
    Wall column at x=2 with door at (2,2) — blocks all paths.
    """
    grid = make_walled_grid(5, 5)
    # Dividing wall at x=2, with door only at y=2.
    grid.set(2, 1, MockCell("wall"))
    grid.set(2, 3, MockCell("wall"))
    env = DoorKeyMockEnv(grid, (1, 1), 0)
    env.set_key((1, 3), "yellow")
    env.set_door((2, 2), "yellow", locked=True)
    env.set_goal((3, 1))
    return env


# ─── _split_context tests ───────────────────────────────────────────


class TestSplitContext:
    def test_state_sks_preserved(self):
        result = _split_context({51, 7, 10001}, 64)
        assert 51 in result  # state SKS preserved
        assert 10001 in result  # perceptual hash preserved
        assert 7 in result  # 7 % 64 = 7

    def test_state_range_not_coarsened(self):
        # SKS 51 should NOT be coarsened even with small n_bins
        result = _split_context({51}, 32)
        assert 51 in result  # preserved, not 51%32=19


# ─── CausalLearner tests ────────────────────────────────────────────


class TestCausalLearner:
    def _make_model(self):
        config = CausalAgentConfig(causal_min_observations=1)
        return CausalWorldModel(config)

    def test_before_after_records_transition(self):
        model = self._make_model()
        learner = CausalLearner(model)

        learner.before_action({SKS_KEY_PRESENT})
        learner.after_action(3, {SKS_KEY_HELD})

        effect, conf = model.predict_effect({SKS_KEY_PRESENT}, 3)
        assert len(effect) > 0

    def test_after_without_before_is_noop(self):
        model = self._make_model()
        learner = CausalLearner(model)
        learner.after_action(3, {SKS_KEY_HELD})
        assert model.n_links == 0

    def test_multiple_observations(self):
        model = self._make_model()
        learner = CausalLearner(model)

        for _ in range(5):
            learner.before_action({SKS_KEY_PRESENT})
            learner.after_action(3, {SKS_KEY_HELD})

        effect, conf = model.predict_effect({SKS_KEY_PRESENT}, 3)
        assert conf > 0

    def test_state_sks_in_causal_links(self):
        model = self._make_model()
        learner = CausalLearner(model)

        learner.before_action({SKS_KEY_HELD, SKS_DOOR_LOCKED})
        learner.after_action(5, {SKS_DOOR_OPEN})  # toggle

        # query_by_effect should find this link.
        results = model.query_by_effect(frozenset({SKS_DOOR_OPEN}))
        assert len(results) > 0
        action, context, confidence = results[0]
        assert action == 5  # toggle


# ─── BlockingAnalyzer tests ──────────────────────────────────────────


class TestBlockingAnalyzer:
    def test_find_blocker_locked_door(self):
        grid = make_walled_grid(5, 5)
        # Full dividing wall with locked door — no way around.
        grid.set(2, 1, MockCell("wall"))
        grid.set(2, 2, MockCell("door", "yellow", is_locked=True))
        grid.set(2, 3, MockCell("wall"))

        analyzer = BlockingAnalyzer()
        blocker = analyzer.find_blocker(grid, (1, 2), (3, 2))
        assert blocker is not None
        assert blocker.cell_type == "door"
        assert blocker.state == "locked"

    def test_find_blocker_clear_path(self):
        grid = make_walled_grid(5, 5)

        analyzer = BlockingAnalyzer()
        blocker = analyzer.find_blocker(grid, (1, 1), (3, 1))
        assert blocker is None

    def test_suggest_resolution_with_model(self):
        config = CausalAgentConfig(causal_min_observations=1)
        model = CausalWorldModel(config)

        # Seed: pickup produces key_held.
        model.observe_transition(
            {SKS_KEY_PRESENT}, 3, {SKS_KEY_HELD},
        )
        # Seed: toggle with key_held produces door_open.
        model.observe_transition(
            {SKS_KEY_HELD, SKS_DOOR_LOCKED}, 5, {SKS_DOOR_OPEN},
        )

        blocker = Blocker("door", "yellow", (2, 2), "locked", SKS_DOOR_LOCKED)
        analyzer = BlockingAnalyzer()
        resolution = analyzer.suggest_resolution(blocker, model, set())

        assert resolution is not None
        assert resolution.action == "toggle"
        assert resolution.prerequisite is not None
        assert resolution.prerequisite.action == "pickup"

    def test_suggest_resolution_empty_model(self):
        config = CausalAgentConfig(causal_min_observations=1)
        model = CausalWorldModel(config)

        blocker = Blocker("door", "yellow", (2, 2), "locked", SKS_DOOR_LOCKED)
        analyzer = BlockingAnalyzer()
        resolution = analyzer.suggest_resolution(blocker, model, set())
        assert resolution is None


# ─── PathResult tests ────────────────────────────────────────────────


class TestPathResult:
    def test_blocked(self):
        grid = make_walled_grid(5, 5)
        # Full dividing wall.
        grid.set(2, 1, MockCell("wall"))
        grid.set(2, 2, MockCell("door", "yellow", is_locked=True))
        grid.set(2, 3, MockCell("wall"))

        nav = GridNavigator()
        result = nav.plan_path_ex(grid, (1, 2), 0, (3, 2))
        assert result.status == PathStatus.BLOCKED
        assert result.actions == []

    def test_already_there(self):
        grid = make_walled_grid(5, 5)
        nav = GridNavigator()
        result = nav.plan_path_ex(grid, (2, 2), 0, (2, 2))
        assert result.status == PathStatus.ALREADY_THERE

    def test_ok(self):
        grid = make_walled_grid(5, 5)
        nav = GridNavigator()
        result = nav.plan_path_ex(grid, (1, 2), 0, (3, 2))
        assert result.status == PathStatus.OK
        assert len(result.actions) > 0


# ─── GridPerception state predicates tests ───────────────────────────


class TestStatePredicates:
    def test_key_present(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)
        grid.set(2, 2, MockCell("key", "yellow"))

        active = perc.perceive(grid, (1, 1), 0)
        assert SKS_KEY_PRESENT in active
        assert SKS_KEY_HELD not in active

    def test_key_held(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)

        carrying = MockCell("key", "yellow")
        active = perc.perceive(grid, (1, 1), 0, carrying=carrying)
        assert SKS_KEY_HELD in active

    def test_door_locked(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)
        grid.set(2, 2, MockCell("door", "yellow", is_locked=True))

        active = perc.perceive(grid, (1, 1), 0)
        assert SKS_DOOR_LOCKED in active
        assert SKS_DOOR_OPEN not in active

    def test_door_open(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)
        grid.set(2, 2, MockCell("door", "yellow", is_open=True))

        active = perc.perceive(grid, (1, 1), 0)
        assert SKS_DOOR_OPEN in active

    def test_goal_present(self):
        gmap = GroundingMap()
        perc = GridPerception(gmap)
        grid = make_walled_grid(5, 5)
        grid.set(3, 3, MockCell("goal", "green"))

        active = perc.perceive(grid, (1, 1), 0)
        assert SKS_GOAL_PRESENT in active


# ─── GoalAgent tests ────────────────────────────────────────────────


class TestGoalAgent:
    def test_doorkey_with_prelearned_model(self):
        env = make_doorkey_env()

        config = CausalAgentConfig(causal_min_observations=1)
        model = CausalWorldModel(config)

        # Pre-seed causal links.
        model.observe_transition({SKS_KEY_PRESENT}, 3, {SKS_KEY_HELD})
        model.observe_transition(
            {SKS_KEY_HELD, SKS_DOOR_LOCKED}, 5, {SKS_DOOR_OPEN},
        )

        agent = GoalAgent(env, causal_model=model)
        result = agent.run_episode(
            "use the key to open the door and then get to the goal",
            max_steps=200,
        )
        assert result.success
        assert result.reward > 0
        assert len(result.subgoals_identified) > 0

    def test_subgoal_chain_correct(self):
        env = make_doorkey_env()

        config = CausalAgentConfig(causal_min_observations=1)
        model = CausalWorldModel(config)
        model.observe_transition({SKS_KEY_PRESENT}, 3, {SKS_KEY_HELD})
        model.observe_transition(
            {SKS_KEY_HELD, SKS_DOOR_LOCKED}, 5, {SKS_DOOR_OPEN},
        )

        agent = GoalAgent(env, causal_model=model)
        result = agent.run_episode("get to the goal", max_steps=200)

        # Subgoals should include pickup and toggle.
        subgoal_text = " ".join(result.subgoals_identified)
        assert "pickup" in subgoal_text
        assert "toggle" in subgoal_text

    def test_exploration_discovers_links(self):
        env = make_doorkey_env()

        # Empty model — agent must explore.
        agent = GoalAgent(env)
        result = agent.run_episode("get to the goal", max_steps=200)

        # After exploration, model should have learned something.
        links = agent.causal_model.get_causal_links(min_confidence=0.0)
        assert len(links) > 0
        assert result.explored

    def test_flatten_subgoals(self):
        inner = SubGoal("pickup", "key", SKS_KEY_HELD, None)
        outer = SubGoal("toggle", "door", SKS_DOOR_OPEN, inner)

        flat = GoalAgent._flatten_subgoals(outer)
        assert len(flat) == 2
        assert flat[0].action == "pickup"
        assert flat[1].action == "toggle"
