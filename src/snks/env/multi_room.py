"""MultiRoomDoorKey: 3-room MiniGrid environment for transfer learning (Stage 26).

Layout: 3 rooms separated by 2 vertical walls, each with a locked door.
Room 1: agent start + yellow key
Room 2: blue key + box distractors
Room 3: goal

Agent must use yellow key to open door 1, enter room 2,
pick up blue key, open door 2, reach goal in room 3.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class MultiRoomDoorKey(MiniGridEnv):
    """3-room environment with 2 locked doors, 2 keys, and distractor boxes."""

    def __init__(
        self,
        size: int = 10,
        max_steps: int = 300,
        seed: int | None = None,
        **kwargs,
    ):
        self._wall1_x: int = 0
        self._wall2_x: int = 0

        mission_space = MissionSpace(
            mission_func=lambda: "use the keys to open the doors and get to the goal"
        )

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Two vertical walls dividing into 3 rooms
        self._wall1_x = width // 3
        self._wall2_x = 2 * width // 3

        # Wall 1 (between room 1 and room 2)
        self.grid.vert_wall(self._wall1_x, 1, height - 2)
        door1 = Door("yellow", is_locked=True)
        door1_y = height // 2
        self.grid.set(self._wall1_x, door1_y, door1)

        # Wall 2 (between room 2 and room 3)
        self.grid.vert_wall(self._wall2_x, 1, height - 2)
        door2 = Door("blue", is_locked=True)
        door2_y = height // 2
        self.grid.set(self._wall2_x, door2_y, door2)

        # Room 1: agent + yellow key
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0  # facing right

        key1_x = self._wall1_x - 2 if self._wall1_x > 2 else 1
        key1_y = height - 2
        self.put_obj(Key("yellow"), key1_x, key1_y)

        # Room 2: blue key + box distractors
        room2_center_x = (self._wall1_x + self._wall2_x) // 2
        key2_y = 2
        self.put_obj(Key("blue"), room2_center_x, key2_y)

        # Distractor boxes in room 2
        box_y = height - 3
        if room2_center_x - 1 > self._wall1_x:
            self.put_obj(Box("red"), room2_center_x - 1, box_y)
        if room2_center_x + 1 < self._wall2_x:
            self.put_obj(Box("green"), room2_center_x + 1, box_y)

        # Room 3: goal
        goal_x = width - 2
        goal_y = height - 2
        self.put_obj(Goal(), goal_x, goal_y)
