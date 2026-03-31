"""Tests for env/multi_room.py — MultiRoomDoorKey environment."""

import pytest

from snks.env.multi_room import MultiRoomDoorKey


class TestMultiRoomDoorKey:
    def test_creates_successfully(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        obs, info = env.reset()
        assert obs is not None

    def test_has_two_doors(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        door_count = sum(
            1 for j in range(env.grid.height) for i in range(env.grid.width)
            if env.grid.get(i, j) is not None and env.grid.get(i, j).type == "door"
        )
        assert door_count == 2

    def test_has_two_keys(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        key_count = sum(
            1 for j in range(env.grid.height) for i in range(env.grid.width)
            if env.grid.get(i, j) is not None and env.grid.get(i, j).type == "key"
        )
        assert key_count == 2

    def test_has_goal(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        goal_found = any(
            env.grid.get(i, j) is not None and env.grid.get(i, j).type == "goal"
            for j in range(env.grid.height) for i in range(env.grid.width)
        )
        assert goal_found

    def test_has_boxes(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        box_count = sum(
            1 for j in range(env.grid.height) for i in range(env.grid.width)
            if env.grid.get(i, j) is not None and env.grid.get(i, j).type == "box"
        )
        assert box_count >= 1

    def test_three_rooms(self):
        """Verify 2 vertical walls creating 3 rooms."""
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        # Count vertical wall segments (excluding outer walls)
        # Walls at x=wall1_x and x=wall2_x
        assert env._wall1_x > 0
        assert env._wall2_x > env._wall1_x

    def test_step_works(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        # Forward action
        obs, reward, terminated, truncated, info = env.step(2)
        assert not (terminated and truncated)

    def test_doors_are_locked(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                obj = env.grid.get(i, j)
                if obj is not None and obj.type == "door":
                    assert obj.is_locked, f"Door at ({i},{j}) should be locked"

    def test_action_space(self):
        """Uses MiniGrid native action space (7 actions)."""
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        assert env.action_space.n == 7

    def test_mission_text(self):
        env = MultiRoomDoorKey(size=10, seed=42)
        env.reset()
        assert "key" in env.mission.lower() or "door" in env.mission.lower()
