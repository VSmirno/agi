"""Stage 56: Tests for Complex Environment — BabyAI PutNext agent.

Tests mission parsing, multi-object tracking, and PutNext task completion.
"""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.spatial_map import SpatialMap, OBJ_EMPTY, OBJ_WALL


# ── MissionParser tests ──────────────────────────────────────────────

class TestMissionParser:
    def test_parse_basic_mission(self):
        from snks.agent.putnext_agent import MissionParser
        parser = MissionParser()
        src, tgt = parser.parse("put the red ball next to the blue key")
        assert src == (6, 0)  # ball=6, red=0
        assert tgt == (5, 2)  # key=5, blue=2

    def test_parse_box_mission(self):
        from snks.agent.putnext_agent import MissionParser
        parser = MissionParser()
        src, tgt = parser.parse("put the grey box next to the green ball")
        assert src == (7, 5)  # box=7, grey=5
        assert tgt == (6, 1)  # ball=6, green=1

    def test_parse_all_colors(self):
        from snks.agent.putnext_agent import MissionParser
        parser = MissionParser()
        colors = {'red': 0, 'green': 1, 'blue': 2, 'purple': 3, 'yellow': 4, 'grey': 5}
        for color_name, color_id in colors.items():
            src, tgt = parser.parse(f"put the {color_name} ball next to the red key")
            assert src == (6, color_id), f"Failed for {color_name}"

    def test_parse_all_types(self):
        from snks.agent.putnext_agent import MissionParser
        parser = MissionParser()
        types = {'key': 5, 'ball': 6, 'box': 7}
        for type_name, type_id in types.items():
            src, tgt = parser.parse(f"put the red {type_name} next to the blue key")
            assert src == (type_id, 0)

    def test_parse_invalid_mission(self):
        from snks.agent.putnext_agent import MissionParser
        parser = MissionParser()
        with pytest.raises(ValueError):
            parser.parse("go to the red ball")


# ── SpatialMap extension tests ───────────────────────────────────────

class TestSpatialMapExtended:
    def test_find_object_by_type_color(self):
        smap = SpatialMap(5, 5)
        # Place a red ball at (2, 3)
        smap.grid[2, 3, 0] = 6   # ball
        smap.grid[2, 3, 1] = 0   # red
        smap.explored[2, 3] = True

        pos = smap.find_object_by_type_color(6, 0)
        assert pos == (2, 3)

    def test_find_object_by_type_color_not_found(self):
        smap = SpatialMap(5, 5)
        pos = smap.find_object_by_type_color(6, 0)
        assert pos is None

    def test_find_object_distinguishes_colors(self):
        smap = SpatialMap(5, 5)
        # Red ball at (1, 1)
        smap.grid[1, 1, 0] = 6  # ball
        smap.grid[1, 1, 1] = 0  # red
        smap.explored[1, 1] = True
        # Blue ball at (3, 3)
        smap.grid[3, 3, 0] = 6  # ball
        smap.grid[3, 3, 1] = 2  # blue
        smap.explored[3, 3] = True

        assert smap.find_object_by_type_color(6, 0) == (1, 1)  # red ball
        assert smap.find_object_by_type_color(6, 2) == (3, 3)  # blue ball

    def test_find_all_objects(self):
        smap = SpatialMap(5, 5)
        smap.grid[1, 1, 0] = 6  # ball
        smap.grid[1, 1, 1] = 0  # red
        smap.explored[1, 1] = True
        smap.grid[3, 3, 0] = 5  # key
        smap.grid[3, 3, 1] = 2  # blue
        smap.explored[3, 3] = True

        objs = smap.find_all_objects()
        assert len(objs) >= 2
        assert (6, 0, 1, 1) in objs  # (type, color, row, col)
        assert (5, 2, 3, 3) in objs


# ── PutNextAgent state machine tests ─────────────────────────────────

class TestPutNextAgentStateMachine:
    def _make_agent(self):
        from snks.agent.putnext_agent import PutNextAgent
        return PutNextAgent(11, 6, "put the red ball next to the blue key")

    def test_initial_phase(self):
        agent = self._make_agent()
        assert agent.phase == "EXPLORE"

    def test_source_target_parsed(self):
        agent = self._make_agent()
        assert agent.source == (6, 0)  # red ball
        assert agent.target == (5, 2)  # blue key

    def test_transitions_to_goto_source(self):
        agent = self._make_agent()
        # Place both source and target in map
        agent.spatial_map.grid[2, 3, 0] = 6   # ball
        agent.spatial_map.grid[2, 3, 1] = 0   # red
        agent.spatial_map.explored[2, 3] = True
        agent.spatial_map.grid[4, 8, 0] = 5   # key
        agent.spatial_map.grid[4, 8, 1] = 2   # blue
        agent.spatial_map.explored[4, 8] = True
        agent._update_phase()
        assert agent.phase == "GOTO_SOURCE"

    def test_transitions_to_goto_target_when_carrying(self):
        agent = self._make_agent()
        agent.phase = "PICKUP"
        agent._carrying = True
        # Place target
        agent.spatial_map.grid[4, 8, 0] = 5
        agent.spatial_map.grid[4, 8, 1] = 2
        agent.spatial_map.explored[4, 8] = True
        agent._update_phase()
        assert agent.phase == "GOTO_TARGET"


# ── Adjacent cell finding tests ──────────────────────────────────────

class TestDropPlanFinding:
    def test_find_drop_plan(self):
        from snks.agent.putnext_agent import PutNextAgent
        agent = PutNextAgent(5, 5, "put the red ball next to the blue key")

        # Set up map with key at (2, 2) and empty surroundings
        for r in range(5):
            for c in range(5):
                if r == 0 or r == 4 or c == 0 or c == 4:
                    agent.spatial_map.grid[r, c, 0] = OBJ_WALL
                else:
                    agent.spatial_map.grid[r, c, 0] = OBJ_EMPTY
                agent.spatial_map.explored[r, c] = True

        agent.spatial_map.grid[2, 2, 0] = 5  # key
        agent.spatial_map.grid[2, 2, 1] = 2  # blue

        plan = agent._find_drop_plan((2, 2), 1, 1)
        assert plan is not None
        stand, drop, face_dir = plan
        # Drop cell must be adjacent to target
        assert abs(drop[0] - 2) + abs(drop[1] - 2) == 1
        # Drop cell must be empty
        assert agent.spatial_map.grid[drop[0], drop[1], 0] in (OBJ_EMPTY, 1)


# ── PutNextEnv wrapper tests ─────────────────────────────────────────

class TestPutNextEnv:
    def test_env_creation(self):
        from snks.agent.putnext_agent import PutNextEnv
        env = PutNextEnv(env_name='BabyAI-PutNextLocalS5N3-v0')
        obs, col, row, d, mission = env.reset(seed=42)
        assert obs.shape == (7, 7, 3)
        assert isinstance(mission, str)
        assert 'put' in mission
        env.close()

    def test_env_step(self):
        from snks.agent.putnext_agent import PutNextEnv
        env = PutNextEnv(env_name='BabyAI-PutNextLocalS5N3-v0')
        env.reset(seed=42)
        obs, reward, term, trunc, col, row, d = env.step(2)  # forward
        assert obs.shape == (7, 7, 3)
        env.close()

    def test_env_carrying(self):
        from snks.agent.putnext_agent import PutNextEnv
        env = PutNextEnv(env_name='BabyAI-PutNextLocalS5N3-v0')
        env.reset(seed=42)
        assert env.carrying is None or env.carrying is not None  # just test property exists
        env.close()


# ── Integration test: full episode ───────────────────────────────────

class TestPutNextIntegration:
    def test_solve_putnext_local_seed0(self):
        """Smoke test: agent should solve at least one seed of PutNextLocal."""
        from snks.agent.putnext_agent import PutNextAgent, PutNextEnv

        env = PutNextEnv(env_name='BabyAI-PutNextLocalS5N3-v0')

        solved = False
        for seed in range(20):
            obs, col, row, d, mission = env.reset(seed=seed)
            agent = PutNextAgent(
                env.grid_width, env.grid_height, mission, epsilon=0.0
            )

            for step in range(200):
                action = agent.select_action(obs, col, row, d)
                agent.update_carrying(env.carrying_type_color)
                obs, reward, term, trunc, col, row, d = env.step(action)
                agent.observe_result(obs, col, row, d, reward)

                if term:
                    if reward > 0:
                        solved = True
                    break
                if trunc:
                    break

            if solved:
                break

        env.close()
        assert solved, "Agent should solve at least 1 of 20 PutNextLocal seeds"

    def test_solve_multiple_seeds(self):
        """Agent should solve ≥30% of 50 seeds on PutNextLocalS5N3."""
        from snks.agent.putnext_agent import PutNextAgent, PutNextEnv

        env = PutNextEnv(env_name='BabyAI-PutNextLocalS5N3-v0')
        successes = 0
        n_seeds = 50

        for seed in range(n_seeds):
            obs, col, row, d, mission = env.reset(seed=seed)
            agent = PutNextAgent(
                env.grid_width, env.grid_height, mission, epsilon=0.0
            )

            for step in range(200):
                action = agent.select_action(obs, col, row, d)
                agent.update_carrying(env.carrying_type_color)
                obs, reward, term, trunc, col, row, d = env.step(action)
                agent.observe_result(obs, col, row, d, reward)

                if term:
                    if reward > 0:
                        successes += 1
                    break
                if trunc:
                    break

        env.close()
        rate = successes / n_seeds
        assert rate >= 0.30, f"Success rate {rate:.1%} < 30% on {n_seeds} seeds"
