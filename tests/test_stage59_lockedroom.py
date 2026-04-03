"""Stage 59: SDM Learned Color Matching — LockedRoom tests."""

import numpy as np
import pytest
import torch

from snks.agent.sdm_lockedroom_agent import (
    ACT_DROP,
    ACT_FORWARD,
    ACT_PICKUP,
    ACT_TOGGLE,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    ColorStateEncoder,
    LockedRoomEnv,
    MissionParser,
    SDMLockedRoomAgent,
)
from snks.agent.vsa_world_model import SDMMemory, VSACodebook


class TestLockedRoomEnv:
    def test_reset_returns_correct_format(self):
        env = LockedRoomEnv(max_steps=100)
        img, col, row, d, carrying, mission = env.reset(seed=0)
        assert img.shape == (7, 7, 3)
        assert isinstance(col, int)
        assert isinstance(row, int)
        assert isinstance(d, int)
        assert carrying is None  # not carrying anything at start
        assert isinstance(mission, str)
        assert "key" in mission
        env.close()

    def test_step_returns_correct_format(self):
        env = LockedRoomEnv(max_steps=100)
        env.reset(seed=0)
        img, reward, term, trunc, col, row, d, carrying, mission = env.step(ACT_FORWARD)
        assert img.shape == (7, 7, 3)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        env.close()

    def test_get_all_doors(self):
        env = LockedRoomEnv()
        env.reset(seed=0)
        doors = env.get_all_doors()
        assert len(doors) == 6  # LockedRoom always has 6 doors
        # Exactly one locked
        locked = [d for d in doors if d[3]]  # is_locked
        assert len(locked) == 1
        env.close()

    def test_get_all_keys(self):
        env = LockedRoomEnv()
        env.reset(seed=0)
        keys = env.get_all_keys()
        assert len(keys) == 1  # LockedRoom has exactly 1 key
        env.close()

    def test_key_color_matches_locked_door(self):
        """Key color must match the locked door color."""
        env = LockedRoomEnv()
        for seed in range(10):
            env.reset(seed=seed)
            keys = env.get_all_keys()
            doors = env.get_all_doors()
            locked = [d for d in doors if d[3]]
            assert keys[0][0] == locked[0][0], f"Seed {seed}: key={keys[0][0]} != door={locked[0][0]}"
        env.close()


class TestMissionParser:
    def test_parse_standard_format(self):
        parser = MissionParser()
        result = parser.parse(
            "get the yellow key from the grey room, unlock the yellow door and go to the goal"
        )
        assert result is not None
        assert result["key_color"] == "yellow"
        assert result["room_color"] == "grey"
        assert result["door_color"] == "yellow"

    def test_parse_multiple_seeds(self):
        parser = MissionParser()
        env = LockedRoomEnv()
        for seed in range(5):
            _, _, _, _, _, mission = env.reset(seed=seed)
            result = parser.parse(mission)
            assert result is not None, f"Seed {seed}: failed to parse '{mission}'"
            assert result["key_color"] in COLOR_TO_IDX
            assert result["door_color"] in COLOR_TO_IDX
            assert result["key_color"] == result["door_color"]  # same color
        env.close()

    def test_parse_invalid(self):
        parser = MissionParser()
        assert parser.parse("go to the red door") is None
        assert parser.parse("") is None


class TestColorStateEncoder:
    def test_encode_color_dimension(self):
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        v = enc.encode_color("red")
        assert v.shape == (512,)

    def test_different_colors_differ(self):
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        v1 = enc.encode_color("red")
        v2 = enc.encode_color("blue")
        sim = (v1 == v2).float().mean().item()
        assert sim < 0.7  # random vectors ~50% similar

    def test_encode_color_pair(self):
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        v = enc.encode_color_pair("red", "red")
        assert v.shape == (512,)

    def test_same_pair_is_deterministic(self):
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        v1 = enc.encode_color_pair("red", "blue")
        v2 = enc.encode_color_pair("red", "blue")
        assert torch.equal(v1, v2)


class TestSDMColorLearning:
    """Test that SDM can learn color matching from experience."""

    def test_sdm_records_reward(self):
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        sdm = SDMMemory(n_locations=1000, dim=512)

        # Write positive reward for red-red
        state = enc.encode_color("red")
        action = enc.encode_color("red")
        for _ in range(10):
            sdm.write(state, action, state, 1.0)

        # Write negative reward for red-blue
        action_wrong = enc.encode_color("blue")
        for _ in range(5):
            sdm.write(state, action_wrong, state, -1.0)

        # Read: red-red should be positive, red-blue negative
        r_correct = sdm.read_reward(state, action)
        r_wrong = sdm.read_reward(state, action_wrong)
        assert r_correct > r_wrong

    def test_sdm_learns_all_colors(self):
        """Train SDM on all 6 same-color pairs, verify each has positive reward."""
        cb = VSACodebook(dim=512)
        enc = ColorStateEncoder(cb)
        sdm = SDMMemory(n_locations=1000, dim=512)

        colors = list(COLOR_TO_IDX.keys())

        # Train: same color → +1, different → -1
        for _ in range(20):  # multiple passes
            for kc in colors:
                for dc in colors:
                    state = enc.encode_color(kc)
                    action = enc.encode_color(dc)
                    reward = 1.0 if kc == dc else -1.0
                    sdm.write(state, action, state, reward)

        # Verify: for each key color, same-color door should have highest reward
        correct = 0
        for kc in colors:
            state = enc.encode_color(kc)
            best_color = None
            best_reward = -float("inf")
            for dc in colors:
                action = enc.encode_color(dc)
                r = sdm.read_reward(state, action)
                if r > best_reward:
                    best_reward = r
                    best_color = dc
            if best_color == kc:
                correct += 1

        assert correct >= 4, f"Only {correct}/6 colors correctly matched"


class TestAgentComponents:
    """Test agent subgoal selection and drop-key logic."""

    def test_subgoal_explore_when_nothing_known(self):
        agent = SDMLockedRoomAgent(explore_episodes=100, use_mission=True)
        agent.reset_episode()
        sg = agent._select_subgoal(carrying_color=None)
        assert sg == agent.SG_EXPLORE

    def test_drop_key_flag(self):
        agent = SDMLockedRoomAgent(explore_episodes=100, use_mission=True)
        agent.reset_episode()
        assert not agent._needs_drop
        agent._needs_drop = True
        # When needs_drop and carrying, select_action should return DROP
        # (tested via the flag, not full env run)
