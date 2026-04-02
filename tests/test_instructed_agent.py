"""Tests for Stage 51: InstructedAgent — language-guided planning in MiniGrid."""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.instructed_agent import InstructedAgent


# --- Minimal DoorKey env for testing ---

class MiniDoorKeyEnv:
    """Minimal DoorKey-5x5 for unit tests. Fixed layout."""

    def __init__(self):
        self.size = 5
        self.n_actions = 7
        self.max_steps = 100
        self.agent_pos = [0, 0]
        self.agent_dir = 0
        self.key_pos = [0, 2]
        self.has_key = False
        self.door_pos = [2, 2]
        self.door_open = False
        self.goal_pos = [3, 2]
        self.wall_row = 2
        self.steps = 0

    def reset(self) -> np.ndarray:
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
            if self.agent_pos == self.key_pos and not self.has_key:
                self.has_key = True
        elif action == 5:
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][self.agent_dir]
            fr, fc = self.agent_pos[0] + dr, self.agent_pos[1] + dc
            if [fr, fc] == self.door_pos and self.has_key and not self.door_open:
                self.door_open = True

        terminated = self.agent_pos == self.goal_pos
        if terminated:
            reward = 1.0
        truncated = self.steps >= self.max_steps

        return self._obs(), reward, terminated, truncated, {}

    def _is_blocked(self, r: int, c: int) -> bool:
        # Wall row except door
        if r == self.wall_row and [r, c] != self.door_pos:
            return True
        if [r, c] == self.door_pos and not self.door_open:
            return True
        return False

    def _obs(self) -> np.ndarray:
        obs = np.zeros((7, 7, 3), dtype=np.int64)
        for i in range(7):
            obs[0, i, 0] = 2; obs[6, i, 0] = 2
            obs[i, 0, 0] = 2; obs[i, 6, 0] = 2
        # Walls
        for c in range(self.size):
            if c != self.door_pos[1]:
                obs[self.wall_row + 1, c + 1, 0] = 2
        # Door
        dr, dc = self.door_pos[0] + 1, self.door_pos[1] + 1
        obs[dr, dc, 0] = 4
        obs[dr, dc, 2] = 0 if self.door_open else 2
        # Key
        if not self.has_key:
            kr, kc = self.key_pos[0] + 1, self.key_pos[1] + 1
            obs[kr, kc, 0] = 5; obs[kr, kc, 1] = 1
        # Goal
        gr, gc = self.goal_pos[0] + 1, self.goal_pos[1] + 1
        obs[gr, gc, 0] = 8
        # Agent
        ar, ac = self.agent_pos[0] + 1, self.agent_pos[1] + 1
        obs[ar, ac, 0] = 10
        obs[ar, ac, 2] = self.agent_dir
        if self.has_key:
            obs[ar, ac, 1] = 5
        return obs


@pytest.fixture
def agent():
    return InstructedAgent()


@pytest.fixture
def env():
    return MiniDoorKeyEnv()


# --- Instruction parsing ---


class TestInstructionParsing:
    def test_full_doorkey_instruction(self, agent):
        subgoals = agent.set_instruction(
            "pick up the key then open the door then go to the goal"
        )
        assert subgoals == ["pickup_key", "open_door", "reach_goal"]

    def test_partial_instruction_key_only(self, agent):
        subgoals = agent.set_instruction("pick up the key")
        assert subgoals == ["pickup_key"]

    def test_partial_instruction_key_and_door(self, agent):
        subgoals = agent.set_instruction("pick up the key then open the door")
        assert subgoals == ["pickup_key", "open_door"]

    def test_empty_instruction(self, agent):
        subgoals = agent.set_instruction("fly to the moon")
        assert subgoals == []


# --- Plan building ---


class TestPlanBuilding:
    def test_build_plan_from_obs(self, agent, env):
        obs = env.reset()
        agent.set_instruction("pick up the key then open the door then go to the goal")
        ok = agent.build_plan(obs)
        assert ok is True
        assert agent.plan is not None
        assert len(agent.plan.subgoals) == 3

    def test_build_plan_partial(self, agent, env):
        obs = env.reset()
        agent.set_instruction("pick up the key")
        ok = agent.build_plan(obs)
        assert ok is True
        assert len(agent.plan.subgoals) == 1
        assert agent.plan.subgoals[0].name == "pickup_key"

    def test_build_plan_no_instruction_fails(self, agent, env):
        obs = env.reset()
        ok = agent.build_plan(obs)
        assert ok is False


# --- Full episode ---


class TestEpisode:
    def test_full_doorkey_episode(self, agent, env):
        success, steps = agent.run_episode(
            env, "pick up the key then open the door then go to the goal"
        )
        assert success is True
        assert steps < 100

    def test_key_only_episode(self, agent, env):
        """Agent picks up key and then episode ends (no more subgoals)."""
        success, steps = agent.run_episode(env, "pick up the key")
        # Success = all subgoals achieved (key picked up)
        assert success is True

    def test_unknown_instruction_fails(self, agent, env):
        success, steps = agent.run_episode(env, "fly to the moon")
        assert success is False


# --- Multiple episodes (robustness) ---


class TestMultipleEpisodes:
    def test_10_episodes_fixed_layout(self, agent, env):
        """Same layout, same instruction, 10 episodes."""
        successes = 0
        for _ in range(10):
            s, _ = agent.run_episode(
                env, "pick up the key then open the door then go to the goal"
            )
            if s:
                successes += 1
        assert successes >= 9  # Should be near 100% on fixed layout
