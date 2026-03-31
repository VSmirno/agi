"""CardGateWorld: Analogical DoorKey environment (Stage 28).

Same structure as DoorKey-5x5 but uses purple Key ("card") and
purple Door ("gate") instead of yellow key/door.

Used to test analogical transfer: agent trained on key/door can
solve card/gate by structural analogy without retraining.
"""

from __future__ import annotations

import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv


class CardGateWorld(MiniGridEnv):
    """5x5 grid with one purple 'card' (key) and one purple 'gate' (door)."""

    def __init__(
        self,
        size: int = 5,
        max_steps: int = 200,
        **kwargs,
    ):
        mission_space = MissionSpace(
            mission_func=lambda: "use the card to open the gate and get to the goal"
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

        # Vertical wall in the middle.
        wall_x = width // 2
        self.grid.vert_wall(wall_x, 1, height - 2)

        # Purple gate (locked door) in the wall.
        gate = Door("purple", is_locked=True)
        door_y = height // 2
        self.grid.set(wall_x, door_y, gate)

        # Agent in room 1.
        self.agent_pos = np.array([1, 1])
        self.agent_dir = 0  # facing right

        # Purple card (key) in room 1.
        card_x = wall_x - 2 if wall_x > 2 else 1
        card_y = height - 2
        self.put_obj(Key("purple"), card_x, card_y)

        # Goal in room 2.
        self.put_obj(Goal(), width - 2, height - 2)
