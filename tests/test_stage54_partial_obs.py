"""Tests for Stage 54: Partial Observability — SpatialMap, FrontierExplorer, PartialObsAgent."""

import numpy as np
import pytest

from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    view_to_world,
    OBJ_DOOR,
    OBJ_EMPTY,
    OBJ_GOAL,
    OBJ_KEY,
    OBJ_WALL,
)
from snks.agent.partial_obs_agent import (
    PartialObsAgent,
    PartialObsDoorKeyEnv,
    ACT_LEFT,
    ACT_RIGHT,
    ACT_FORWARD,
    ACT_PICKUP,
    ACT_TOGGLE,
)


# ─── view_to_world coordinate transform ───


class TestViewToWorld:
    """Verify 7x7 obs → world coordinate transform for all 4 directions."""

    def test_dir0_right_agent_position(self):
        """Agent at obs[3,6] maps to agent world position."""
        wc, wr = view_to_world(3, 6, agent_col=2, agent_row=3, agent_dir=0)
        assert (wc, wr) == (2, 3)

    def test_dir1_down_agent_position(self):
        wc, wr = view_to_world(3, 6, agent_col=1, agent_row=2, agent_dir=1)
        assert (wc, wr) == (1, 2)

    def test_dir2_left_agent_position(self):
        wc, wr = view_to_world(3, 6, agent_col=1, agent_row=3, agent_dir=2)
        assert (wc, wr) == (1, 3)

    def test_dir3_up_agent_position(self):
        wc, wr = view_to_world(3, 6, agent_col=1, agent_row=2, agent_dir=3)
        assert (wc, wr) == (1, 2)

    def test_dir1_known_mapping(self):
        """Verified against MiniGrid seed=42: obs[2,6]=door at world(2,2)."""
        wc, wr = view_to_world(2, 6, agent_col=1, agent_row=2, agent_dir=1)
        assert (wc, wr) == (2, 2)

    def test_dir1_key_mapping(self):
        """seed=42: obs[3,5]=key at world(1,3)."""
        wc, wr = view_to_world(3, 5, agent_col=1, agent_row=2, agent_dir=1)
        assert (wc, wr) == (1, 3)

    def test_dir3_known_mapping(self):
        """seed=99: obs[4,6]=door at world(2,2)."""
        wc, wr = view_to_world(4, 6, agent_col=1, agent_row=2, agent_dir=3)
        assert (wc, wr) == (2, 2)

    def test_dir2_key_mapping(self):
        """seed=7: obs[5,6]=key at world(1,1)."""
        wc, wr = view_to_world(5, 6, agent_col=1, agent_row=3, agent_dir=2)
        assert (wc, wr) == (1, 1)

    def test_symmetry_all_dirs_center(self):
        """obs[3,6] always maps to agent position regardless of direction."""
        for d in range(4):
            wc, wr = view_to_world(3, 6, agent_col=5, agent_row=5, agent_dir=d)
            assert (wc, wr) == (5, 5), f"dir={d}: expected (5,5), got ({wc},{wr})"


# ─── SpatialMap ───


class TestSpatialMap:

    def test_init_all_unknown(self):
        sm = SpatialMap(5, 5)
        assert np.all(sm.grid == -1)
        assert not np.any(sm.explored)

    def test_update_marks_explored(self):
        sm = SpatialMap(5, 5)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        # Make center visible as empty
        obs[3, 6, 0] = OBJ_EMPTY
        sm.update(obs, agent_col=2, agent_row=2, agent_dir=0)
        assert sm.explored[2, 2]

    def test_update_stores_objects(self):
        sm = SpatialMap(5, 5)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        obs[3, 6, 0] = OBJ_KEY
        obs[3, 6, 1] = 4  # yellow
        sm.update(obs, agent_col=2, agent_row=2, agent_dir=0)
        assert sm.grid[2, 2, 0] == OBJ_KEY

    def test_update_ignores_unseen(self):
        sm = SpatialMap(5, 5)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        # obj_type=0 is unseen — should not be written
        sm.update(obs, agent_col=2, agent_row=2, agent_dir=0)
        assert not sm.explored[2, 2]

    def test_accumulation_over_steps(self):
        sm = SpatialMap(5, 5)
        obs1 = np.zeros((7, 7, 3), dtype=np.int8)
        obs1[3, 6, 0] = OBJ_WALL
        sm.update(obs1, agent_col=1, agent_row=1, agent_dir=0)
        assert sm.explored[1, 1]

        obs2 = np.zeros((7, 7, 3), dtype=np.int8)
        obs2[3, 6, 0] = OBJ_EMPTY
        sm.update(obs2, agent_col=3, agent_row=3, agent_dir=0)
        assert sm.explored[3, 3]
        # Both cells should be explored
        assert sm.explored[1, 1] and sm.explored[3, 3]

    def test_find_object(self):
        sm = SpatialMap(5, 5)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        obs[3, 6, 0] = OBJ_KEY
        sm.update(obs, agent_col=2, agent_row=3, agent_dir=0)
        pos = sm.find_object(OBJ_KEY)
        assert pos == (3, 2)

    def test_find_object_not_found(self):
        sm = SpatialMap(5, 5)
        assert sm.find_object(OBJ_KEY) is None

    def test_to_obs_unknown_as_empty(self):
        sm = SpatialMap(5, 5)
        obs = sm.to_obs()
        assert np.all(obs[:, :, 0] == OBJ_EMPTY)

    def test_frontiers_basic(self):
        sm = SpatialMap(5, 5)
        # Mark center as explored empty
        sm.grid[2, 2] = [OBJ_EMPTY, 0, 0]
        sm.explored[2, 2] = True
        fronts = sm.frontiers()
        assert (2, 2) in fronts  # adjacent to unexplored

    def test_frontiers_wall_excluded(self):
        sm = SpatialMap(5, 5)
        # Mark cell as wall
        sm.grid[2, 2] = [OBJ_WALL, 0, 0]
        sm.explored[2, 2] = True
        fronts = sm.frontiers()
        assert (2, 2) not in fronts  # walls are not frontiers

    def test_frontiers_fully_explored(self):
        sm = SpatialMap(3, 3)
        sm.grid[:] = [OBJ_EMPTY, 0, 0]
        sm.explored[:] = True
        fronts = sm.frontiers()
        assert len(fronts) == 0

    def test_reset(self):
        sm = SpatialMap(5, 5)
        sm.grid[2, 2] = [OBJ_KEY, 0, 0]
        sm.explored[2, 2] = True
        sm.reset()
        assert np.all(sm.grid == -1)
        assert not np.any(sm.explored)


# ─── FrontierExplorer ───


class TestFrontierExplorer:

    def test_nearest_frontier_simple(self):
        sm = SpatialMap(5, 5)
        # Explored cells around agent
        for r, c in [(1, 1), (1, 2), (2, 1)]:
            sm.grid[r, c] = [OBJ_EMPTY, 0, 0]
            sm.explored[r, c] = True
        # Cell (2,2) not explored → (1,2) and (2,1) are frontiers

        fe = FrontierExplorer()
        target = fe.nearest_frontier(sm, agent_row=1, agent_col=1)
        assert target is not None
        assert target in [(1, 1), (1, 2), (2, 1)]  # all are frontiers

    def test_no_frontiers(self):
        sm = SpatialMap(3, 3)
        sm.grid[:] = [OBJ_EMPTY, 0, 0]
        sm.explored[:] = True
        fe = FrontierExplorer()
        assert fe.nearest_frontier(sm, 1, 1) is None


# ─── PartialObsDoorKeyEnv ───


class TestPartialObsDoorKeyEnv:

    def test_reset_returns_7x7(self):
        env = PartialObsDoorKeyEnv(size=5, seed=42)
        obs, col, row, d = env.reset()
        assert obs.shape == (7, 7, 3)
        assert isinstance(col, int)
        assert isinstance(row, int)
        assert d in (0, 1, 2, 3)

    def test_step_returns_correct_format(self):
        env = PartialObsDoorKeyEnv(size=5, seed=42)
        env.reset()
        obs, reward, term, trunc, col, row, d = env.step(ACT_FORWARD)
        assert obs.shape == (7, 7, 3)
        assert isinstance(reward, float)

    def test_deterministic_with_seed(self):
        env1 = PartialObsDoorKeyEnv(size=5)
        obs1, c1, r1, d1 = env1.reset(seed=42)

        env2 = PartialObsDoorKeyEnv(size=5)
        obs2, c2, r2, d2 = env2.reset(seed=42)

        assert np.array_equal(obs1, obs2)
        assert (c1, r1, d1) == (c2, r2, d2)


# ─── PartialObsAgent ───


class TestPartialObsAgent:

    def test_agent_creation(self):
        agent = PartialObsAgent(5, 5)
        assert agent.spatial_map.width == 5

    def test_agent_reset(self):
        agent = PartialObsAgent(5, 5)
        agent._has_key = True
        agent.reset()
        assert not agent._has_key

    def test_agent_updates_map(self):
        agent = PartialObsAgent(5, 5, epsilon=0.0)
        env = PartialObsDoorKeyEnv(size=5, seed=42)
        obs, col, row, d = env.reset()
        agent.select_action(obs, col, row, d)
        assert np.any(agent.spatial_map.explored)

    def test_agent_solves_doorkey(self):
        """Integration test: agent should solve DoorKey-5x5 with partial obs."""
        successes = 0
        n_tests = 20

        for seed in range(n_tests):
            env = PartialObsDoorKeyEnv(size=5, seed=seed)
            agent = PartialObsAgent(5, 5, epsilon=0.05)
            agent.reset()

            obs, col, row, d = env.reset()
            max_steps = 200

            for step in range(max_steps):
                action = agent.select_action(obs, col, row, d)

                # Update inventory from env
                agent.update_inventory(env.carrying)

                obs, reward, term, trunc, col, row, d = env.step(action)
                agent.observe_result(obs, col, row, d, reward)

                if term or trunc:
                    if reward > 0:
                        successes += 1
                    break

        rate = successes / n_tests
        assert rate >= 0.6, f"Success rate {rate:.0%} < 60% on {n_tests} seeds"

    def test_map_accumulation_real_env(self):
        """Verify SpatialMap correctly accumulates DoorKey observations."""
        env = PartialObsDoorKeyEnv(size=5, seed=42)
        agent = PartialObsAgent(5, 5, epsilon=0.0)
        agent.reset()

        obs, col, row, d = env.reset()

        # Take a few steps and verify map grows
        explored_counts = []
        for _ in range(10):
            action = agent.select_action(obs, col, row, d)
            agent.update_inventory(env.carrying)
            obs, reward, term, trunc, col, row, d = env.step(action)
            agent.observe_result(obs, col, row, d, reward)
            explored_counts.append(np.sum(agent.spatial_map.explored))
            if term or trunc:
                break

        # Map should grow or stay (never shrink)
        for i in range(1, len(explored_counts)):
            assert explored_counts[i] >= explored_counts[i - 1]

    def test_immediate_pickup(self):
        """If agent faces key and doesn't have it, should pickup."""
        agent = PartialObsAgent(5, 5, epsilon=0.0)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        # Front cell in MiniGrid partial obs: obs[3,5] (view col=3, row=5)
        obs[3, 5, 0] = OBJ_KEY
        action = agent._check_immediate_action(obs, agent_dir=0)
        assert action == ACT_PICKUP

    def test_immediate_toggle_locked_door(self):
        """If agent faces locked door and has key, should toggle."""
        agent = PartialObsAgent(5, 5, epsilon=0.0)
        agent._has_key = True
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        obs[3, 5, 0] = OBJ_DOOR
        obs[3, 5, 2] = 2  # locked
        action = agent._check_immediate_action(obs, agent_dir=0)
        assert action == ACT_TOGGLE

    def test_immediate_toggle_closed_door(self):
        """If agent faces closed door, should toggle."""
        agent = PartialObsAgent(5, 5, epsilon=0.0)
        obs = np.zeros((7, 7, 3), dtype=np.int8)
        obs[3, 5, 0] = OBJ_DOOR
        obs[3, 5, 2] = 1  # closed
        action = agent._check_immediate_action(obs, agent_dir=0)
        assert action == ACT_TOGGLE
