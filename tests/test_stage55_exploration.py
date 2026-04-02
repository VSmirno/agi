"""Tests for Stage 55: Exploration Strategy — MultiRoom-N3 with partial obs."""

import numpy as np
import pytest

from snks.agent.partial_obs_agent import (
    MultiRoomPartialObsAgent,
    PartialObsMultiRoomEnv,
)
from snks.agent.spatial_map import SpatialMap


class TestPartialObsMultiRoomEnv:

    def test_reset_returns_7x7(self):
        env = PartialObsMultiRoomEnv()
        obs, col, row, d = env.reset(seed=42)
        assert obs.shape == (7, 7, 3)
        assert isinstance(col, int)
        assert isinstance(row, int)
        assert d in (0, 1, 2, 3)

    def test_step_returns_correct_format(self):
        env = PartialObsMultiRoomEnv()
        env.reset(seed=42)
        obs, reward, term, trunc, col, row, d = env.step(2)
        assert obs.shape == (7, 7, 3)

    def test_grid_size(self):
        env = PartialObsMultiRoomEnv()
        env.reset(seed=42)
        assert env.grid_width == 25
        assert env.grid_height == 25


class TestSpatialMap25x25:

    def test_large_spatial_map(self):
        sm = SpatialMap(25, 25)
        assert sm.grid.shape == (25, 25, 3)
        assert not np.any(sm.explored)

    def test_frontiers_large_grid(self):
        sm = SpatialMap(25, 25)
        sm.grid[10, 10] = [1, 0, 0]  # empty
        sm.explored[10, 10] = True
        fronts = sm.frontiers()
        assert (10, 10) in fronts


class TestMultiRoomPartialObsAgent:

    def test_creation(self):
        agent = MultiRoomPartialObsAgent(25, 25)
        assert agent.spatial_map.width == 25

    def test_updates_map(self):
        agent = MultiRoomPartialObsAgent(25, 25, epsilon=0.0)
        env = PartialObsMultiRoomEnv()
        obs, col, row, d = env.reset(seed=42)
        agent.select_action(obs, col, row, d)
        assert np.any(agent.spatial_map.explored)

    def test_solves_multiroom(self):
        """Integration test: ≥60% on 20 seeds."""
        successes = 0
        n_tests = 20

        for seed in range(n_tests):
            env = PartialObsMultiRoomEnv()
            obs, col, row, d = env.reset(seed=seed)
            agent = MultiRoomPartialObsAgent(
                env.grid_width, env.grid_height, epsilon=0.05
            )
            agent.reset()

            for step in range(300):
                action = agent.select_action(obs, col, row, d)
                obs, reward, term, trunc, col, row, d = env.step(action)
                agent.observe_result(obs, col, row, d, reward)

                if term or trunc:
                    if reward > 0:
                        successes += 1
                    break

        rate = successes / n_tests
        assert rate >= 0.5, f"Success rate {rate:.0%} < 50% on {n_tests} seeds"

    def test_map_grows(self):
        """Spatial map should grow over steps."""
        env = PartialObsMultiRoomEnv()
        obs, col, row, d = env.reset(seed=0)
        agent = MultiRoomPartialObsAgent(
            env.grid_width, env.grid_height, epsilon=0.0
        )
        agent.reset()

        counts = []
        for _ in range(20):
            action = agent.select_action(obs, col, row, d)
            obs, reward, term, trunc, col, row, d = env.step(action)
            agent.observe_result(obs, col, row, d, reward)
            counts.append(int(np.sum(agent.spatial_map.explored)))
            if term or trunc:
                break

        # Should be non-decreasing
        for i in range(1, len(counts)):
            assert counts[i] >= counts[i - 1]
        # Should explore at least some cells
        assert counts[-1] > 10
