"""Tests for Stage 52: IntegrationAgent — language-guided MultiRoom + DoorKey."""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.integration_agent import IntegrationAgent


# --- Minimal MultiRoom env for testing ---


class MiniMultiRoomEnv:
    """Minimal 3-room env for unit tests.

    Layout (9x9 internal, 11x11 with walls):
    Room 0: cols 1-3, door at (3, 4)
    Room 1: cols 5-7, door at (3, 8)
    Room 2: cols 9-11 (goal in room 2)

    Simplified: 3 rooms in a row, doors between them.
    """

    def __init__(self):
        self.grid_h = 7
        self.grid_w = 13
        self.agent_pos = [1, 1]
        self.agent_dir = 0  # right
        self.goal_pos = [3, 11]
        self.doors = {
            (3, 4): False,   # door 1: closed
            (3, 8): False,   # door 2: closed
        }
        self.steps = 0
        self.max_steps = 300

    def reset(self, seed=None) -> np.ndarray:
        self.agent_pos = [1, 1]
        self.agent_dir = 0
        self.doors = {(3, 4): False, (3, 8): False}
        self.steps = 0
        return self._obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.steps += 1
        reward = 0.0

        if action == 0:  # turn left
            self.agent_dir = (self.agent_dir - 1) % 4
        elif action == 1:  # turn right
            self.agent_dir = (self.agent_dir + 1) % 4
        elif action == 2:  # forward
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            nr, nc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if 0 <= nr < self.grid_h and 0 <= nc < self.grid_w:
                if not self._is_blocked(nr, nc):
                    self.agent_pos = [nr, nc]
        elif action == 5:  # toggle
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            door_key = (fr, fc)
            if door_key in self.doors and not self.doors[door_key]:
                self.doors[door_key] = True

        terminated = self.agent_pos == list(self.goal_pos)
        if terminated:
            reward = 1.0
        truncated = self.steps >= self.max_steps

        return self._obs(), reward, terminated, truncated, {}

    def _is_blocked(self, r: int, c: int) -> bool:
        if r == 0 or r == self.grid_h - 1 or c == 0 or c == self.grid_w - 1:
            return True  # outer walls
        # Internal walls at cols 4 and 8
        if c == 4 and (r, c) not in self.doors:
            return True
        if c == 8 and (r, c) not in self.doors:
            return True
        if c == 4 and (r, c) in self.doors and not self.doors[(r, c)]:
            return True
        if c == 8 and (r, c) in self.doors and not self.doors[(r, c)]:
            return True
        return False

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.grid_h, self.grid_w, 3), dtype=np.int64)
        # Outer walls
        for r in range(self.grid_h):
            obs[r, 0, 0] = 2
            obs[r, self.grid_w - 1, 0] = 2
        for c in range(self.grid_w):
            obs[0, c, 0] = 2
            obs[self.grid_h - 1, c, 0] = 2
        # Internal walls at cols 4 and 8
        for r in range(self.grid_h):
            if (r, 4) not in self.doors:
                obs[r, 4, 0] = 2
            if (r, 8) not in self.doors:
                obs[r, 8, 0] = 2
        # Doors
        for (dr, dc), is_open in self.doors.items():
            obs[dr, dc, 0] = 4
            obs[dr, dc, 2] = 0 if is_open else 1  # 1=closed (unlocked)
        # Goal
        obs[self.goal_pos[0], self.goal_pos[1], 0] = 8
        # Agent
        ar, ac = self.agent_pos
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        return obs

    @property
    def unwrapped(self):
        return self


# --- Minimal DoorKey env (from Stage 51 tests) ---


class MiniDoorKeyEnv:
    """Minimal DoorKey-5x5 for unit tests. Fixed layout."""

    def __init__(self):
        self.size = 5
        self.agent_pos = [0, 0]
        self.agent_dir = 0
        self.key_pos = [0, 2]
        self.has_key = False
        self.door_pos = [2, 2]
        self.door_open = False
        self.goal_pos = [3, 2]
        self.wall_row = 2
        self.steps = 0

    def reset(self, seed=None) -> np.ndarray:
        self.agent_pos = [0, 0]
        self.agent_dir = 0
        self.has_key = False
        self.door_open = False
        self.steps = 0
        return self._obs()

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
                if not self._is_blocked(nr, nc):
                    self.agent_pos = [nr, nc]
        elif action == 3:
            if self.agent_pos == list(self.key_pos) and not self.has_key:
                self.has_key = True
        elif action == 5:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == list(self.door_pos) and self.has_key and not self.door_open:
                self.door_open = True
        terminated = self.agent_pos == list(self.goal_pos)
        if terminated:
            reward = 1.0
        truncated = self.steps >= 100
        return self._obs(), reward, terminated, truncated, {}

    def _is_blocked(self, r, c):
        if r == self.wall_row and [r, c] != list(self.door_pos):
            return True
        if [r, c] == list(self.door_pos) and not self.door_open:
            return True
        return False

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        for i in range(7):
            obs[0, i, 0] = 2; obs[6, i, 0] = 2
            obs[i, 0, 0] = 2; obs[i, 6, 0] = 2
        for c in range(self.size):
            if c != self.door_pos[1]:
                obs[self.wall_row + 1, c + 1, 0] = 2
        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2
        if not self.has_key:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5; obs[kr, kc, 1] = 1
        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10; obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


@pytest.fixture
def agent():
    return IntegrationAgent()


@pytest.fixture
def multiroom_env():
    return MiniMultiRoomEnv()


@pytest.fixture
def doorkey_env():
    return MiniDoorKeyEnv()


# --- Instruction parsing ---


class TestInstructionParsing:
    def test_goto_goal(self, agent):
        subgoals = agent.parse_instruction("go to the goal")
        assert "reach_goal" in subgoals

    def test_open_door_then_goal(self, agent):
        subgoals = agent.parse_instruction(
            "open the door then go to the goal"
        )
        assert subgoals == ["open_door", "reach_goal"]

    def test_full_doorkey_instruction(self, agent):
        subgoals = agent.parse_instruction(
            "pick up the key then open the door then go to the goal"
        )
        assert subgoals == ["pickup_key", "open_door", "reach_goal"]


# --- Environment detection ---


class TestEnvDetection:
    def test_detect_multiroom(self, agent, multiroom_env):
        obs = multiroom_env.reset()
        env_type = agent.detect_env_type(obs)
        assert env_type == "multiroom"

    def test_detect_doorkey(self, agent, doorkey_env):
        obs = doorkey_env.reset()
        env_type = agent.detect_env_type(obs)
        assert env_type == "doorkey"


# --- MultiRoom episodes ---


class TestMultiRoomEpisode:
    def test_goto_goal_success(self, agent, multiroom_env):
        success, steps = agent.run_episode(
            multiroom_env, "go to the goal"
        )
        assert success is True
        assert steps < 300

    def test_open_doors_goto_goal(self, agent, multiroom_env):
        success, steps = agent.run_episode(
            multiroom_env, "open the door then go to the goal"
        )
        assert success is True

    def test_10_episodes_multiroom(self, agent, multiroom_env):
        successes = 0
        for _ in range(10):
            s, _ = agent.run_episode(multiroom_env, "go to the goal")
            if s:
                successes += 1
        assert successes >= 9


# --- DoorKey regression ---


class TestDoorKeyRegression:
    def test_doorkey_full_instruction(self, agent, doorkey_env):
        success, steps = agent.run_episode(
            doorkey_env,
            "pick up the key then open the door then go to the goal",
        )
        assert success is True
        assert steps < 100

    def test_10_doorkey_episodes(self, agent, doorkey_env):
        successes = 0
        for _ in range(10):
            s, _ = agent.run_episode(
                doorkey_env,
                "pick up the key then open the door then go to the goal",
            )
            if s:
                successes += 1
        assert successes >= 9


# --- Edge cases ---


class TestEdgeCases:
    def test_unknown_instruction(self, agent, multiroom_env):
        success, steps = agent.run_episode(multiroom_env, "fly to mars")
        assert success is False

    def test_empty_instruction(self, agent, multiroom_env):
        success, steps = agent.run_episode(multiroom_env, "")
        assert success is False
