"""Stage 57: Tests for Long Subgoal Chains — KeyCorridor agent.

Tests prerequisite-graph planning with 5+ subgoal chains.
"""

from __future__ import annotations

import numpy as np
import pytest

from snks.agent.keycorridor_agent import (
    ChainPlanner,
    KeyCorridorAgent,
    KeyCorridorEnv,
    MissionAnalyzer,
)
from snks.agent.spatial_map import SpatialMap, OBJ_DOOR, OBJ_KEY, OBJ_WALL


# MiniGrid object/color IDs
OBJ_BALL = 6
OBJ_BOX = 7
COLOR_RED = 0
COLOR_GREEN = 1
COLOR_BLUE = 2
COLOR_PURPLE = 3
COLOR_YELLOW = 4
COLOR_GREY = 5


class TestMissionAnalyzer:
    def test_pick_up_ball(self):
        ma = MissionAnalyzer()
        goal = ma.analyze("pick up the ball")
        assert goal == ("pickup", OBJ_BALL, None)

    def test_pick_up_box(self):
        ma = MissionAnalyzer()
        goal = ma.analyze("pick up the box")
        assert goal == ("pickup", OBJ_BOX, None)

    def test_pick_up_key(self):
        ma = MissionAnalyzer()
        goal = ma.analyze("pick up the key")
        assert goal == ("pickup", OBJ_KEY, None)

    def test_open_door(self):
        ma = MissionAnalyzer()
        goal = ma.analyze("open the door")
        assert goal == ("open", OBJ_DOOR, None)


class TestChainPlanner:
    def _make_map(self, width=10, height=10):
        return SpatialMap(width, height)

    def test_basic_chain_with_locked_door(self):
        """Chain: EXPLORE → GOTO_KEY → PICKUP → GOTO_DOOR → OPEN → GOTO_GOAL → PICKUP."""
        sm = self._make_map()
        # Place objects on the map
        sm.grid[3, 2, :] = [OBJ_KEY, COLOR_PURPLE, 0]  # purple key at (3,2)
        sm.explored[3, 2] = True
        sm.grid[5, 5, :] = [OBJ_DOOR, COLOR_PURPLE, 2]  # locked purple door at (5,5)
        sm.explored[5, 5] = True
        sm.grid[7, 5, :] = [OBJ_BALL, 0, 0]  # ball at (7,5)
        sm.explored[7, 5] = True

        planner = ChainPlanner(("pickup", OBJ_BALL, None), sm)
        chain = planner.build_chain()

        # Should have at least 5 subgoals
        assert len(chain) >= 5
        names = [s.name for s in chain]
        # Must contain key steps in order
        assert "GOTO_KEY" in names
        assert "PICKUP_KEY" in names
        assert "GOTO_LOCKED_DOOR" in names
        assert "OPEN_DOOR" in names
        assert "PICKUP_GOAL" in names
        # Order: key before door before goal
        assert names.index("PICKUP_KEY") < names.index("OPEN_DOOR")
        assert names.index("OPEN_DOOR") < names.index("PICKUP_GOAL")

    def test_chain_no_locked_door(self):
        """If no locked door, chain is shorter: GOTO_GOAL → PICKUP."""
        sm = self._make_map()
        sm.grid[7, 5, :] = [OBJ_BALL, 0, 0]
        sm.explored[7, 5] = True

        planner = ChainPlanner(("pickup", OBJ_BALL, None), sm)
        chain = planner.build_chain()

        names = [s.name for s in chain]
        assert "GOTO_GOAL" in names
        assert "PICKUP_GOAL" in names
        assert "GOTO_KEY" not in names

    def test_chain_explore_when_nothing_found(self):
        """If nothing found yet, chain is just EXPLORE."""
        sm = self._make_map()
        planner = ChainPlanner(("pickup", OBJ_BALL, None), sm)
        chain = planner.build_chain()

        assert len(chain) >= 1
        assert chain[0].name == "EXPLORE"

    def test_chain_updates_on_discovery(self):
        """Chain should be rebuilt when new objects are discovered."""
        sm = self._make_map()
        planner = ChainPlanner(("pickup", OBJ_BALL, None), sm)

        chain1 = planner.build_chain()
        assert chain1[0].name == "EXPLORE"

        # "Discover" a locked door
        sm.grid[5, 5, :] = [OBJ_DOOR, COLOR_GREEN, 2]
        sm.explored[5, 5] = True
        chain2 = planner.build_chain()
        # Now should include GOTO_KEY or EXPLORE for key
        names2 = [s.name for s in chain2]
        assert any(n in names2 for n in ["EXPLORE", "EXPLORE_FOR_KEY", "GOTO_KEY"])

    def test_chain_length_with_locked_door(self):
        """Ensure chain length ≥ 5 when locked door present."""
        sm = self._make_map()
        sm.grid[2, 3, :] = [OBJ_KEY, COLOR_RED, 0]
        sm.explored[2, 3] = True
        sm.grid[5, 5, :] = [OBJ_DOOR, COLOR_RED, 2]
        sm.explored[5, 5] = True
        sm.grid[8, 5, :] = [OBJ_BALL, 0, 0]
        sm.explored[8, 5] = True

        planner = ChainPlanner(("pickup", OBJ_BALL, None), sm)
        chain = planner.build_chain()
        assert len(chain) >= 5, f"Chain too short: {[s.name for s in chain]}"


class TestKeyCorridorAgent:
    def test_agent_init(self):
        agent = KeyCorridorAgent(10, 10, "pick up the ball")
        assert agent.phase == "EXPLORE"
        assert not agent._carrying

    def test_agent_toggle_closed_door(self):
        """Agent should toggle closed unlocked door when facing it."""
        agent = KeyCorridorAgent(10, 10, "pick up the ball")
        # Create a fake 7x7 obs with a closed door in front
        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        obs[3, 5, 0] = OBJ_DOOR  # door in front cell
        obs[3, 5, 2] = 1  # closed (unlocked)
        action = agent.select_action(obs, 5, 5, 0)
        assert action == 5  # ACT_TOGGLE

    def test_agent_pickup_key_when_facing(self):
        """Agent should pickup key when facing it and key is needed."""
        agent = KeyCorridorAgent(10, 10, "pick up the ball")
        # Put a locked door on the map so agent knows it needs a key
        agent.spatial_map.grid[7, 7, :] = [OBJ_DOOR, COLOR_PURPLE, 2]
        agent.spatial_map.explored[7, 7] = True
        agent._locked_door_color = COLOR_PURPLE

        obs = np.zeros((7, 7, 3), dtype=np.uint8)
        obs[3, 5, 0] = OBJ_KEY  # key in front
        obs[3, 5, 1] = COLOR_PURPLE
        action = agent.select_action(obs, 5, 5, 0)
        assert action == 3  # ACT_PICKUP

    def test_agent_updates_carrying_state(self):
        agent = KeyCorridorAgent(10, 10, "pick up the ball")
        agent.update_carrying(OBJ_KEY, COLOR_PURPLE)
        assert agent._carrying
        assert agent._carrying_type == OBJ_KEY
        assert agent._carrying_color == COLOR_PURPLE

    def test_subgoal_count_tracking(self):
        """Verify that agent tracks subgoal completions."""
        agent = KeyCorridorAgent(10, 10, "pick up the ball")
        assert agent.subgoals_completed == 0
        agent._complete_subgoal("EXPLORE")
        assert agent.subgoals_completed == 1
        agent._complete_subgoal("GOTO_KEY")
        assert agent.subgoals_completed == 2


class TestKeyCorridorEnv:
    @pytest.mark.parametrize("env_name", [
        "BabyAI-KeyCorridorS3R3-v0",
        "BabyAI-KeyCorridorS4R3-v0",
    ])
    def test_env_reset(self, env_name):
        env = KeyCorridorEnv(env_name)
        obs, col, row, d, mission = env.reset(seed=42)
        assert obs.shape == (7, 7, 3)
        assert isinstance(mission, str)
        assert "pick up" in mission
        env.close()

    def test_env_step(self):
        env = KeyCorridorEnv("BabyAI-KeyCorridorS3R3-v0")
        env.reset(seed=42)
        obs, reward, term, trunc, col, row, d = env.step(2)  # forward
        assert obs.shape == (7, 7, 3)
        env.close()

    def test_blocked_unlock_pickup_env(self):
        env = KeyCorridorEnv("BabyAI-BlockedUnlockPickup-v0")
        obs, col, row, d, mission = env.reset(seed=42)
        assert "pick up" in mission
        env.close()


class TestEndToEnd:
    """Integration tests — run agent on actual envs with limited seeds."""

    def _run_episode(self, env_name: str, seed: int, max_steps: int = 300) -> tuple[bool, int, int]:
        """Run one episode, return (success, steps, subgoals_completed)."""
        env = KeyCorridorEnv(env_name)
        obs, col, row, d, mission = env.reset(seed=seed)

        agent = KeyCorridorAgent(env.grid_width, env.grid_height, mission)

        for step in range(max_steps):
            # Update carrying state
            ct, cc = env.carrying_type_color or (None, None)
            if ct is not None:
                agent.update_carrying(ct, cc)
            else:
                agent.clear_carrying()

            action = agent.select_action(obs, col, row, d)
            obs, reward, term, trunc, col, row, d = env.step(action)
            agent.observe_result(obs, col, row, d, reward)

            if term or trunc:
                env.close()
                return reward > 0, step + 1, agent.subgoals_completed

        env.close()
        return False, max_steps, agent.subgoals_completed

    def test_keycorridor_s3r3_seed0(self):
        success, steps, subs = self._run_episode("BabyAI-KeyCorridorS3R3-v0", 0, 200)
        assert success, f"Failed seed 0: {steps} steps, {subs} subgoals"
        assert subs >= 5, f"Too few subgoals: {subs}"

    def test_keycorridor_s3r3_seed1(self):
        success, steps, subs = self._run_episode("BabyAI-KeyCorridorS3R3-v0", 1, 200)
        assert success, f"Failed seed 1: {steps} steps, {subs} subgoals"

    def test_keycorridor_s4r3_seed0(self):
        success, steps, subs = self._run_episode("BabyAI-KeyCorridorS4R3-v0", 0, 400)
        assert success, f"Failed seed 0: {steps} steps, {subs} subgoals"
        assert subs >= 5, f"Too few subgoals: {subs}"

    def test_blocked_unlock_seed0(self):
        """BlockedUnlockPickup has blocking ball — requires 9+ subgoals (drop/swap).
        This is beyond Stage 57 scope. Test only that agent makes progress."""
        success, steps, subs = self._run_episode("BabyAI-BlockedUnlockPickup-v0", 0, 400)
        # At minimum agent should find key and pick it up (2 subgoals)
        assert subs >= 2, f"Too few subgoals: {subs}"

    def test_subgoal_chain_minimum_5(self):
        """Verify chain ≥ 5 subgoals on KeyCorridorS4R3."""
        success, steps, subs = self._run_episode("BabyAI-KeyCorridorS4R3-v0", 0, 400)
        assert subs >= 5, f"Subgoal chain too short: {subs}"
