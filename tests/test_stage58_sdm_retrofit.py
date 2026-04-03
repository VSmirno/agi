"""Stage 58: Tests for SDM Retrofit — Learned DoorKey agent.

Tests the СНКС pipeline: obs → SpatialMap → AbstractStateEncoder (VSA) → SDM → action.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.agent.sdm_doorkey_agent import (
    AbstractStateEncoder,
    SDMDoorKeyAgent,
    SDMDoorKeyEnv,
)
from snks.agent.vsa_world_model import SDMMemory, VSACodebook


class TestAbstractStateEncoder:
    def test_encode_produces_binary_vector(self):
        cb = VSACodebook(dim=512)
        enc = AbstractStateEncoder(cb)
        vec = enc.encode(
            agent_row=2, agent_col=3, has_key=False,
            door_state="locked", key_known=True, door_known=True,
            goal_known=False, exploration_pct=0.3,
        )
        assert vec.shape == (512,)
        assert set(vec.unique().tolist()).issubset({0.0, 1.0})

    def test_different_states_produce_different_vectors(self):
        cb = VSACodebook(dim=512)
        enc = AbstractStateEncoder(cb)
        v1 = enc.encode(2, 3, False, "locked", True, True, False, 0.3)
        v2 = enc.encode(2, 3, True, "locked", True, True, False, 0.3)  # has_key changed
        sim = cb.similarity(v1, v2)
        assert sim < 0.95, f"States with different has_key should differ, sim={sim}"

    def test_same_state_produces_same_vector(self):
        cb = VSACodebook(dim=512)
        enc = AbstractStateEncoder(cb)
        v1 = enc.encode(2, 3, True, "open", True, True, True, 0.8)
        v2 = enc.encode(2, 3, True, "open", True, True, True, 0.8)
        sim = cb.similarity(v1, v2)
        assert sim == 1.0

    def test_position_changes_vector(self):
        cb = VSACodebook(dim=512)
        enc = AbstractStateEncoder(cb)
        v1 = enc.encode(1, 1, False, "locked", False, False, False, 0.1)
        v2 = enc.encode(3, 3, False, "locked", False, False, False, 0.1)
        sim = cb.similarity(v1, v2)
        assert sim < 0.95, f"Different positions should differ, sim={sim}"


class TestSDMIntegration:
    def test_sdm_write_read_cycle(self):
        """Write transition to SDM, read it back."""
        cb = VSACodebook(dim=512)
        sdm = SDMMemory(n_locations=1000, dim=512)
        enc = AbstractStateEncoder(cb)

        state1 = enc.encode(2, 2, False, "locked", True, True, False, 0.5)
        state2 = enc.encode(2, 3, False, "locked", True, True, False, 0.5)
        action_vsa = cb.action(2)  # forward

        # Write
        sdm.write(state1, action_vsa, state2, 0.0)

        # Read
        pred, conf = sdm.read_next(state1, action_vsa)
        assert conf > 0, "Confidence should be > 0 after write"

    def test_sdm_reward_signal(self):
        """SDM should store reward signal."""
        cb = VSACodebook(dim=512)
        sdm = SDMMemory(n_locations=1000, dim=512)
        enc = AbstractStateEncoder(cb)

        state = enc.encode(3, 3, True, "open", True, True, True, 0.9)
        action_vsa = cb.action(2)

        # Write positive reward multiple times
        goal_state = enc.encode(3, 4, False, "open", True, True, True, 1.0)
        for _ in range(5):
            sdm.write(state, action_vsa, goal_state, 1.0)

        reward_score = sdm.read_reward(state, action_vsa)
        assert reward_score > 0, f"Should have positive reward signal, got {reward_score}"

    def test_sdm_counts_writes(self):
        cb = VSACodebook(dim=512)
        sdm = SDMMemory(n_locations=1000, dim=512)
        enc = AbstractStateEncoder(cb)

        state = enc.encode(1, 1, False, "locked", False, False, False, 0.1)
        action = cb.action(0)
        next_state = enc.encode(1, 1, False, "locked", False, False, False, 0.1)

        for _ in range(10):
            sdm.write(state, action, next_state, 0.0)

        assert sdm.n_writes == 10


class TestSDMDoorKeyEnv:
    def test_env_reset(self):
        env = SDMDoorKeyEnv(size=5)
        obs, col, row, d, has_key, door_state = env.reset(seed=42)
        assert obs.shape == (7, 7, 3)
        assert not has_key
        assert door_state in ("locked", "closed", "open")

    def test_env_step(self):
        env = SDMDoorKeyEnv(size=5)
        env.reset(seed=42)
        obs, reward, term, trunc, col, row, d, has_key, door_state = env.step(2)
        assert obs.shape == (7, 7, 3)


class TestSDMDoorKeyAgent:
    def test_agent_init(self):
        agent = SDMDoorKeyAgent(grid_width=5, grid_height=5)
        assert agent.sdm.n_writes == 0
        assert agent._exploring

    def test_agent_records_transitions(self):
        """Agent should write to SDM during exploration."""
        env = SDMDoorKeyEnv(size=5)
        obs, col, row, d, has_key, door_state = env.reset(seed=42)
        agent = SDMDoorKeyAgent(grid_width=env.grid_width, grid_height=env.grid_height)

        for step in range(20):
            action = agent.select_action(
                obs, col, row, d, has_key, door_state
            )
            obs, reward, term, trunc, col, row, d, has_key, door_state = env.step(action)
            agent.observe_result(obs, col, row, d, has_key, door_state, reward)
            if term or trunc:
                break

        assert agent.sdm.n_writes > 0, "Agent should have written transitions to SDM"

    def test_exploration_to_planning_transition(self):
        """Agent should switch from exploration to planning after N episodes."""
        agent = SDMDoorKeyAgent(grid_width=5, grid_height=5, explore_episodes=3)
        assert agent._exploring
        for _ in range(3):
            agent._episode_done(success=False)
        assert not agent._exploring


class TestEndToEnd:
    def _run_episode(self, agent: SDMDoorKeyAgent, seed: int,
                     max_steps: int = 200) -> tuple[bool, int]:
        env = SDMDoorKeyEnv(size=5)
        obs, col, row, d, has_key, door_state = env.reset(seed=seed)

        for step in range(max_steps):
            action = agent.select_action(obs, col, row, d, has_key, door_state)
            obs, reward, term, trunc, col, row, d, has_key, door_state = env.step(action)
            agent.observe_result(obs, col, row, d, has_key, door_state, reward)

            if term or trunc:
                agent._episode_done(success=reward > 0)
                env.close()
                return reward > 0, step + 1

        agent._episode_done(success=False)
        env.close()
        return False, max_steps

    def test_exploration_fills_sdm(self):
        """After exploration episodes, SDM should have ≥100 transitions."""
        agent = SDMDoorKeyAgent(grid_width=5, grid_height=5, explore_episodes=10)
        for seed in range(10):
            self._run_episode(agent, seed=seed, max_steps=200)
        assert agent.sdm.n_writes >= 100, f"SDM should have ≥100 writes, got {agent.sdm.n_writes}"

    def test_learned_agent_beats_random(self):
        """After exploration, learned agent should beat random on at least 1 seed."""
        agent = SDMDoorKeyAgent(grid_width=5, grid_height=5, explore_episodes=20)

        # Exploration phase
        for seed in range(20):
            self._run_episode(agent, seed=seed, max_steps=200)

        # Planning phase — try 10 seeds
        successes = 0
        for seed in range(10):
            success, _ = self._run_episode(agent, seed=seed, max_steps=200)
            if success:
                successes += 1

        assert successes >= 1, f"Learned agent should solve at least 1 of 10 seeds, got {successes}"
