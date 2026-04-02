"""Stage 46: Subgoal Planning — unit tests (TDD).

Tests for SubgoalExtractor, PlanGraph, SubgoalNavigator, SubgoalPlanningAgent.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.agent.vsa_world_model import (
    SDMMemory,
    VSACodebook,
    VSAEncoder,
    WorldModelConfig,
)
from snks.agent.subgoal_planning import (
    PlanGraph,
    Subgoal,
    SubgoalConfig,
    SubgoalExtractor,
    SubgoalNavigator,
    SubgoalPlanningAgent,
    TraceStep,
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

class _DoorKeyEnv:
    """Simplified DoorKey-5x5 with blocking wall for testing.

    Layout (inner 5x5):
      Row 0: . A . K .   (agent at 0,1, key at 0,3)
      Row 1: . . . . .
      Row 2: W W D W W   (wall with door at 2,2)
      Row 3: . . . . .
      Row 4: . . . G .   (goal at 4,3)
    """

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)
        self.size = 5
        self.n_actions = 7
        self.max_steps = 200
        self.wall_positions = [[2, 0], [2, 1], [2, 3], [2, 4]]
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.agent_pos = [0, 1]
        self.agent_dir = 1
        self.key_pos = [0, 3]
        self.has_key = False
        self.door_pos = [2, 2]
        self.door_open = False
        self.goal_pos = [4, 3]
        self.steps = 0
        self.key_picked = False
        return self._obs()

    def _is_wall(self, r: int, c: int) -> bool:
        return [r, c] in self.wall_positions

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = 0.0
        if action == 0:
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                if self._is_wall(nr, nc):
                    pass
                elif [nr, nc] == self.door_pos and not self.door_open:
                    pass
                else:
                    self.agent_pos = [nr, nc]
        elif action == 3:
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
                self.key_picked = True
        elif action == 5:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == self.door_pos and self.has_key and not self.door_open:
                self.door_open = True
        terminated = False
        if self.agent_pos == self.goal_pos:
            reward = 1.0 - 0.9 * (self.steps / self.max_steps)
            terminated = True
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {}

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        for i in range(7):
            obs[0, i, 0] = 2; obs[6, i, 0] = 2
            obs[i, 0, 0] = 2; obs[i, 6, 0] = 2
        for wr, wc in self.wall_positions:
            obs[wr + 1, wc + 1, 0] = 2
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        if not self.key_picked:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5; obs[kr, kc, 1] = 1
        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2
        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


def _make_obs(agent_pos=(3, 3), agent_dir=0, key_pos=None, key_color=1,
              door_pos=None, door_state=2, goal_pos=(4, 4),
              has_key=False) -> np.ndarray:
    """Create a MiniGrid-like symbolic obs (7x7x3)."""
    obs = np.zeros((7, 7, 3), dtype=np.int64)
    # Walls
    for i in range(7):
        obs[0, i, 0] = 2
        obs[6, i, 0] = 2
        obs[i, 0, 0] = 2
        obs[i, 6, 0] = 2

    ar, ac = agent_pos
    obs[ar, ac, 0] = 10
    obs[ar, ac, 2] = agent_dir
    if has_key:
        obs[ar, ac, 1] = 5  # carrying indicator

    if key_pos is not None:
        kr, kc = key_pos
        obs[kr, kc, 0] = 5
        obs[kr, kc, 1] = key_color

    if door_pos is not None:
        dr, dc = door_pos
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = door_state

    if goal_pos is not None:
        gr, gc = goal_pos
        obs[gr, gc, 0] = 8

    return obs


def _make_doorkey_trace() -> list[TraceStep]:
    """Create a realistic successful DoorKey trace with blocking wall.

    Layout (obs coords): agent(1,2), key(1,4), wall row 3, door(3,3), goal(5,4).
    Sequence: navigate to key → pickup → navigate to door → toggle → go through → goal.
    """
    trace = []

    # Step 1: Agent at (1,2), moving toward key at (1,4)
    obs_before = _make_obs(agent_pos=(1, 2), key_pos=(1, 4), door_pos=(3, 3), goal_pos=(5, 4))
    obs_after = _make_obs(agent_pos=(1, 3), key_pos=(1, 4), door_pos=(3, 3), goal_pos=(5, 4))
    trace.append(TraceStep(obs_before, 2, obs_after, 0.0))

    # Step 2: Reach key position
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(1, 4), key_pos=(1, 4), door_pos=(3, 3), goal_pos=(5, 4))
    trace.append(TraceStep(obs_before, 2, obs_after, 0.0))

    # Step 3: Pickup key — key disappears
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(1, 4), key_pos=None, door_pos=(3, 3), goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 3, obs_after, 0.0))

    # Step 4: Navigate toward door (turn + move)
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(2, 3), key_pos=None, door_pos=(3, 3), goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 2, obs_after, 0.0))

    # Step 5: Toggle door — door opens
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(2, 3), key_pos=None, door_pos=(3, 3), door_state=0, goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 5, obs_after, 0.0))

    # Step 6: Move through door
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(3, 3), key_pos=None, door_pos=(3, 3), door_state=0, goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 2, obs_after, 0.0))

    # Step 7: Navigate to goal
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(4, 4), key_pos=None, door_pos=(3, 3), door_state=0, goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 2, obs_after, 0.0))

    # Step 8: Reach goal
    obs_before = obs_after
    obs_after = _make_obs(agent_pos=(5, 4), key_pos=None, door_pos=(3, 3), door_state=0, goal_pos=(5, 4), has_key=True)
    trace.append(TraceStep(obs_before, 2, obs_after, 1.0))

    return trace


# ──────────────────────────────────────────────
# SubgoalExtractor
# ──────────────────────────────────────────────

class TestSubgoalExtractor:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.enc = VSAEncoder(self.cb)
        self.extractor = SubgoalExtractor(self.cb, self.enc)

    def test_extract_detects_pickup_key(self):
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert "pickup_key" in names

    def test_extract_detects_open_door(self):
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert "open_door" in names

    def test_extract_detects_reach_goal(self):
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert "reach_goal" in names

    def test_extract_correct_order(self):
        """Subgoals must be ordered: pickup_key → open_door → reach_goal."""
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert names.index("pickup_key") < names.index("open_door")
        assert names.index("open_door") < names.index("reach_goal")

    def test_extract_has_target_states(self):
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        for sg in subgoals:
            assert sg.target_state.shape == (512,)
            assert sg.precondition_state.shape == (512,)

    def test_extract_no_duplicates(self):
        trace = _make_doorkey_trace()
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert len(names) == len(set(names))

    def test_extract_empty_trace(self):
        subgoals = self.extractor.extract([])
        assert subgoals == []

    def test_extract_trace_no_key_event(self):
        """Trace where agent goes straight to goal (no key/door) → only reach_goal."""
        obs1 = _make_obs(agent_pos=(3, 3), goal_pos=(4, 4))
        obs2 = _make_obs(agent_pos=(4, 4), goal_pos=(4, 4))
        trace = [TraceStep(obs1, 2, obs2, 1.0)]
        subgoals = self.extractor.extract(trace)
        names = [s.name for s in subgoals]
        assert "reach_goal" in names
        assert "pickup_key" not in names


# ──────────────────────────────────────────────
# PlanGraph
# ──────────────────────────────────────────────

class TestPlanGraph:
    def _make_subgoals(self) -> list[Subgoal]:
        dummy = torch.zeros(512)
        return [
            Subgoal("pickup_key", dummy, dummy, "symbolic"),
            Subgoal("open_door", dummy, dummy, "symbolic"),
            Subgoal("reach_goal", dummy, dummy, "symbolic"),
        ]

    def test_current_subgoal_starts_at_first(self):
        sg = self._make_subgoals()
        plan = PlanGraph(sg)
        assert plan.current_subgoal().name == "pickup_key"

    def test_advance_moves_to_next(self):
        sg = self._make_subgoals()
        plan = PlanGraph(sg)
        done = plan.advance()
        assert not done
        assert plan.current_subgoal().name == "open_door"

    def test_advance_to_end(self):
        sg = self._make_subgoals()
        plan = PlanGraph(sg)
        plan.advance()  # → open_door
        plan.advance()  # → reach_goal
        done = plan.advance()  # → done
        assert done
        assert plan.current_subgoal() is None

    def test_reset(self):
        sg = self._make_subgoals()
        plan = PlanGraph(sg)
        plan.advance()
        plan.advance()
        plan.reset()
        assert plan.current_subgoal().name == "pickup_key"

    def test_empty_plan(self):
        plan = PlanGraph([])
        assert plan.current_subgoal() is None
        assert plan.advance() is True


# ──────────────────────────────────────────────
# SubgoalNavigator
# ──────────────────────────────────────────────

class TestSubgoalNavigator:
    def setup_method(self):
        self.cb = VSACodebook(dim=512, seed=42)
        self.enc = VSAEncoder(self.cb)
        self.sdm = SDMMemory(n_locations=1000, dim=512, seed=42)
        self.nav = SubgoalNavigator(self.sdm, self.cb, self.enc, n_actions=7)

    def test_select_returns_valid_action(self):
        state = torch.randint(0, 2, (512,), dtype=torch.float32)
        dummy_target = torch.randint(0, 2, (512,), dtype=torch.float32)
        sg = Subgoal("test", dummy_target, state, "symbolic")
        action = self.nav.select(state, sg)
        assert 0 <= action < 7

    def test_select_prefers_action_from_trace_segment(self):
        """If trace segment has matching state, navigator should replay that action."""
        from snks.agent.subgoal_planning import SymbolicState

        # Set up trace segment: at position (2,3) facing down, do action 2
        segment = [
            (SymbolicState(2, 3, 1, False, False), 2),  # at (2,3), dir=down, action=forward
            (SymbolicState(3, 3, 1, False, False), 2),  # at (3,3), dir=down, action=forward
        ]
        nav = SubgoalNavigator(self.sdm, self.cb, self.enc, n_actions=7, epsilon=0.0)
        nav.set_trace_segments({"test_sg": segment})

        # Current obs: agent at (2,3), same as trace
        obs = _make_obs(agent_pos=(2, 3), agent_dir=1, key_pos=(1, 4), door_pos=(3, 3), goal_pos=(5, 4))
        state = self.enc.encode(obs)
        dummy_target = torch.zeros(512)
        sg = Subgoal("test_sg", dummy_target, state, "symbolic")

        actions = [nav.select(state, sg, current_obs=obs) for _ in range(10)]
        assert actions.count(2) >= 9, f"Expected action 2 from trace, got {actions}"

    def test_is_achieved_pickup_key(self):
        """Key not visible → pickup_key achieved."""
        obs_with_key = _make_obs(agent_pos=(2, 2), key_pos=(2, 4))
        obs_no_key = _make_obs(agent_pos=(2, 4), key_pos=None, has_key=True)
        dummy = torch.zeros(512)
        sg = Subgoal("pickup_key", dummy, dummy, "symbolic")
        assert not self.nav.is_achieved(obs_with_key, sg)
        assert self.nav.is_achieved(obs_no_key, sg)

    def test_is_achieved_open_door(self):
        """Door state=0 → open_door achieved."""
        obs_locked = _make_obs(door_pos=(3, 3), door_state=2)
        obs_open = _make_obs(door_pos=(3, 3), door_state=0)
        dummy = torch.zeros(512)
        sg = Subgoal("open_door", dummy, dummy, "symbolic")
        assert not self.nav.is_achieved(obs_locked, sg)
        assert self.nav.is_achieved(obs_open, sg)


# ──────────────────────────────────────────────
# SubgoalPlanningAgent (integration)
# ──────────────────────────────────────────────

class TestSubgoalPlanningAgent:
    def test_creates_successfully(self):
        config = SubgoalConfig(dim=512, n_locations=1000, n_actions=7)
        agent = SubgoalPlanningAgent(config)
        assert agent is not None
        assert agent.extractor is not None
        assert agent.navigator is not None

    def test_explore_phase_collects_traces(self):
        """During explore, agent should collect trace data."""
        config = SubgoalConfig(
            dim=512, n_locations=1000, n_actions=7,
            explore_episodes=5,
        )
        agent = SubgoalPlanningAgent(config)

        class SimpleEnv:
            def __init__(self):
                self.steps = 0
            def reset(self, seed=None):
                self.steps = 0
                return _make_obs(agent_pos=(2, 2), key_pos=(2, 4),
                                 door_pos=(3, 3), goal_pos=(4, 4))
            def step(self, action):
                self.steps += 1
                done = self.steps >= 3
                obs = _make_obs(agent_pos=(4, 4), goal_pos=(4, 4)) if done else \
                    _make_obs(agent_pos=(2, 3), key_pos=(2, 4),
                              door_pos=(3, 3), goal_pos=(4, 4))
                return obs, 1.0 if done else 0.0, done, False, {}

        env = SimpleEnv()
        for _ in range(5):
            agent.run_episode(env, max_steps=20)
        # Should have collected traces
        assert len(agent._successful_traces) > 0

    def test_plan_phase_builds_plan(self):
        """After explore with successful traces, plan phase should build a PlanGraph."""
        config = SubgoalConfig(
            dim=512, n_locations=1000, n_actions=7,
            explore_episodes=2,
        )
        agent = SubgoalPlanningAgent(config)

        # Manually add a successful trace
        trace = _make_doorkey_trace()
        agent._successful_traces.append(trace)
        agent._episode_count = 3  # past explore phase

        # Run a plan episode
        class DummyEnv:
            def __init__(self):
                self.steps = 0
            def reset(self, seed=None):
                self.steps = 0
                return _make_obs(agent_pos=(2, 2), key_pos=(2, 4),
                                 door_pos=(3, 3), goal_pos=(4, 4))
            def step(self, action):
                self.steps += 1
                done = self.steps >= 10
                return _make_obs(agent_pos=(2, 3)), 0.0, done, False, {}

        env = DummyEnv()
        agent.run_episode(env, max_steps=20)
        assert agent.plan is not None
        assert len(agent.plan.subgoals) >= 2  # at least pickup_key and reach_goal

    def test_full_episode_with_doorkey_env(self):
        """Integration test: inject successful trace, verify plan builds and agent runs."""
        config = SubgoalConfig(
            dim=512, n_locations=2000, n_actions=7,
            explore_episodes=2,
            epsilon=0.2,
            min_confidence=0.01,
        )
        agent = SubgoalPlanningAgent(config)

        env = _DoorKeyEnv(seed=42)

        # Run 2 explore episodes (won't succeed, but fills SDM)
        for _ in range(2):
            agent.run_episode(env, max_steps=200)

        # Inject a known successful trace so plan phase has data
        trace = _make_doorkey_trace()
        agent._successful_traces.append(trace)

        # Run plan episodes — should build plan and attempt subgoal navigation
        for _ in range(5):
            agent.run_episode(env, max_steps=200)

        # Verify plan was built with correct subgoals
        assert agent.plan is not None, "Plan should have been built from injected trace"
        names = [s.name for s in agent.plan.subgoals]
        assert "pickup_key" in names, f"Expected pickup_key in plan, got {names}"
        assert "reach_goal" in names, f"Expected reach_goal in plan, got {names}"
