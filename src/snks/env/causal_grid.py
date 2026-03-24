"""CausalGridWorld: MiniGrid-based environments for causal learning experiments."""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions as MGActions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Ball, Box, Door, Goal, Key, Lava, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgObsWrapper


class Action(IntEnum):
    """Agent actions for CausalGridWorld."""
    turn_left = 0
    turn_right = 1
    forward = 2
    interact = 3  # push/pickup/toggle
    noop = 4


# Map our Action enum to MiniGrid's action space
_ACTION_MAP = {
    Action.turn_left: MGActions.left,
    Action.turn_right: MGActions.right,
    Action.forward: MGActions.forward,
    Action.interact: MGActions.toggle,  # toggle handles push/pickup/toggle
    Action.noop: MGActions.done,  # done as noop
}


class CausalGridWorld(MiniGridEnv):
    """Base causal grid world with configurable scenarios.

    Extends MiniGrid with support for causal experiments:
    - Scripted object movements (for correlation vs causation tests)
    - Object tracking for causal link verification
    - Episode metrics collection
    """

    def __init__(
        self,
        level: str = "EmptyExplore",
        size: int = 8,
        max_steps: int = 200,
        scripted_objects: bool = False,
        seed: int | None = None,
        **kwargs,
    ):
        self.level = level
        self._scripted_objects = scripted_objects
        self._scripted_ball_pos: tuple[int, int] | None = None
        self._step_counter = 0
        self._visited_cells: set[tuple[int, int]] = set()
        self._object_positions: dict[str, tuple[int, int]] = {}

        mission_space = MissionSpace(mission_func=lambda: "explore and learn causality")

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs,
        )
        self.action_space = gym.spaces.Discrete(5)  # our 5 actions

    def _gen_grid(self, width: int, height: int) -> None:
        """Generate grid based on level name."""
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.level == "EmptyExplore":
            self._gen_empty(width, height)
        elif self.level == "PushBox":
            self._gen_push_box(width, height)
        elif self.level == "PushChain":
            self._gen_push_chain(width, height)
        elif self.level == "BallRoll":
            self._gen_ball_roll(width, height)
        elif self.level == "DoorKey":
            self._gen_door_key(width, height)
        elif self.level == "MultiRoom":
            self._gen_multi_room(width, height)
        else:
            raise ValueError(f"Unknown level: {self.level}")

    def _gen_empty(self, w: int, h: int) -> None:
        self.agent_pos = np.array([w // 2, h // 2])
        self.agent_dir = 0

    def _gen_push_box(self, w: int, h: int) -> None:
        self.agent_pos = np.array([2, 2])
        self.agent_dir = 0  # facing right
        self.put_obj(Box("blue"), 4, 2)
        self._object_positions["box_0"] = (4, 2)

        if self._scripted_objects:
            self.put_obj(Ball("red"), 4, 5)
            self._scripted_ball_pos = (4, 5)
            self._object_positions["ball_scripted"] = (4, 5)

    def _gen_push_chain(self, w: int, h: int) -> None:
        self.agent_pos = np.array([2, 5])
        self.agent_dir = 0  # facing right
        self.put_obj(Box("blue"), 4, 5)
        self.put_obj(Box("green"), 5, 5)
        self.put_obj(Box("purple"), 6, 5)
        self._object_positions["box_0"] = (4, 5)
        self._object_positions["box_1"] = (5, 5)
        self._object_positions["box_2"] = (6, 5)

    def _gen_ball_roll(self, w: int, h: int) -> None:
        self.agent_pos = np.array([2, 4])
        self.agent_dir = 0
        self.put_obj(Ball("yellow"), 4, 4)
        self._object_positions["ball_0"] = (4, 4)

    def _gen_door_key(self, w: int, h: int) -> None:
        self.grid.vert_wall(w // 2, 1, h - 2)
        door = Door("yellow", is_locked=True)
        self.grid.set(w // 2, h // 2, door)
        self._object_positions["door_0"] = (w // 2, h // 2)

        self.agent_pos = np.array([2, 2])
        self.agent_dir = 0
        self.put_obj(Key("yellow"), 2, h - 3)
        self._object_positions["key_0"] = (2, h - 3)

        self.put_obj(Goal(), w - 2, h // 2)
        self._object_positions["goal_0"] = (w - 2, h // 2)

    def _gen_multi_room(self, w: int, h: int) -> None:
        self.grid.vert_wall(w // 2, 1, h - 2)
        door = Door("yellow", is_locked=True)
        self.grid.set(w // 2, h // 2, door)
        self._object_positions["door_0"] = (w // 2, h // 2)

        self.agent_pos = np.array([2, 2])
        self.agent_dir = 0
        self.put_obj(Key("yellow"), 3, h - 3)
        self._object_positions["key_0"] = (3, h - 3)

        self.put_obj(Box("blue"), 3, 4)
        self._object_positions["box_0"] = (3, 4)

        self.put_obj(Box("green"), w - 3, 3)
        self._object_positions["box_1"] = (w - 3, 3)

        self.put_obj(Goal(), w - 2, h - 2)
        self._object_positions["goal_0"] = (w - 2, h - 2)

    def step(self, action: int):
        """Execute action and return (obs, reward, terminated, truncated, info).

        Maps our 5-action space to MiniGrid actions.
        Handles scripted object movements for causal tests.
        """
        self._step_counter += 1

        if action == Action.noop:
            # MiniGrid's done terminates episode — we want true noop
            obs = self.gen_obs()
            reward = 0.0
            terminated = False
            truncated = self._step_counter >= self.max_steps
            info = self._make_info()
            return obs, reward, terminated, truncated, info

        # Custom push mechanics: both interact and forward can push objects
        # interact: explicit push/pickup/toggle
        # forward: if pushable object in front, push it and move agent (Sokoban-style)
        if action == Action.interact:
            self._handle_interact()
        elif action == Action.forward:
            self._handle_forward_push()

        mg_action = _ACTION_MAP[Action(action)]
        obs, reward, terminated, truncated, info = super().step(mg_action)

        # Track visited cells
        pos = tuple(self.agent_pos)
        self._visited_cells.add(pos)

        # Scripted ball movement (moves randomly, NOT caused by agent)
        if self._scripted_objects and self._scripted_ball_pos is not None:
            self._move_scripted_ball()

        info.update(self._make_info())
        return obs, reward, terminated, truncated, info

    def _handle_forward_push(self) -> None:
        """Sokoban-style: walking into a pushable object pushes it."""
        dx, dy = self.dir_vec
        fx, fy = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        obj = self.grid.get(fx, fy)

        if obj is not None and isinstance(obj, (Box, Ball)):
            # Push object one cell further if space is free
            nx, ny = fx + dx, fy + dy
            if (
                0 < nx < self.width - 1
                and 0 < ny < self.height - 1
                and self.grid.get(nx, ny) is None
            ):
                self.grid.set(fx, fy, None)
                self.grid.set(nx, ny, obj)
                for name, pos in self._object_positions.items():
                    if pos == (fx, fy):
                        self._object_positions[name] = (nx, ny)
                        break
                # Now agent can move into (fx, fy) — MiniGrid's forward will handle it

    def _handle_interact(self) -> None:
        """Custom push mechanics for Box and Ball objects.

        Standard MiniGrid toggle doesn't push boxes. This implements:
        - Box: pushed 1 cell in agent's facing direction
        - Ball: pushed and rolls until hitting a wall/object
        """
        # Get cell in front of agent
        dx, dy = self.dir_vec
        fx, fy = self.agent_pos[0] + dx, self.agent_pos[1] + dy

        obj = self.grid.get(fx, fy)
        if obj is None:
            return

        if isinstance(obj, Box):
            # Push box 1 cell in facing direction
            nx, ny = fx + dx, fy + dy
            if (
                0 < nx < self.width - 1
                and 0 < ny < self.height - 1
                and self.grid.get(nx, ny) is None
            ):
                self.grid.set(fx, fy, None)
                self.grid.set(nx, ny, obj)
                # Update tracked positions
                for name, pos in self._object_positions.items():
                    if pos == (fx, fy):
                        self._object_positions[name] = (nx, ny)
                        break

        elif isinstance(obj, Ball):
            # Ball rolls until hitting wall/object
            cx, cy = fx, fy
            self.grid.set(cx, cy, None)
            while True:
                nx, ny = cx + dx, cy + dy
                if (
                    0 < nx < self.width - 1
                    and 0 < ny < self.height - 1
                    and self.grid.get(nx, ny) is None
                    and not (nx == self.agent_pos[0] and ny == self.agent_pos[1])
                ):
                    cx, cy = nx, ny
                else:
                    break
            self.grid.set(cx, cy, obj)
            for name, pos in self._object_positions.items():
                if pos == (fx, fy):
                    self._object_positions[name] = (cx, cy)
                    break

    def _move_scripted_ball(self) -> None:
        """Move scripted ball randomly (not caused by agent action)."""
        bx, by = self._scripted_ball_pos
        # Remove from current position
        obj = self.grid.get(bx, by)
        if obj is not None and isinstance(obj, Ball):
            self.grid.set(bx, by, None)
            # Try random direction
            dx, dy = self.np_random.choice([-1, 0, 1]), self.np_random.choice([-1, 0, 1])
            nx, ny = bx + dx, by + dy
            if (
                0 < nx < self.width - 1
                and 0 < ny < self.height - 1
                and self.grid.get(nx, ny) is None
                and not (nx == self.agent_pos[0] and ny == self.agent_pos[1])
            ):
                self.grid.set(nx, ny, obj)
                self._scripted_ball_pos = (nx, ny)
                self._object_positions["ball_scripted"] = (nx, ny)
            else:
                self.grid.set(bx, by, obj)  # stay put

    def _make_info(self) -> dict:
        return {
            "step": self._step_counter,
            "visited_cells": len(self._visited_cells),
            "agent_pos": tuple(self.agent_pos),
            "agent_dir": self.agent_dir,
            "object_positions": dict(self._object_positions),
        }

    def reset(self, **kwargs):
        self._step_counter = 0
        self._visited_cells = set()
        self._object_positions = {}
        obs, info = super().reset(**kwargs)
        self._visited_cells.add(tuple(self.agent_pos))
        info.update(self._make_info())
        return obs, info

    @property
    def coverage(self) -> float:
        """Fraction of reachable cells visited."""
        total = (self.width - 2) * (self.height - 2)
        return len(self._visited_cells) / max(total, 1)


def make_level(level: str, render_mode: str | None = None, **kwargs) -> gym.Env:
    """Create a CausalGridWorld level wrapped with RGBImgObsWrapper.

    Returns a Gymnasium env where obs is RGB (H, W, 3) uint8.
    """
    sizes = {
        "EmptyExplore": 8,
        "PushBox": 8,
        "PushChain": 10,
        "BallRoll": 8,
        "DoorKey": 8,
        "MultiRoom": 12,
    }
    size = kwargs.pop("size", sizes.get(level, 8))
    env = CausalGridWorld(level=level, size=size, render_mode=render_mode, **kwargs)
    env = RGBImgObsWrapper(env)
    return env
