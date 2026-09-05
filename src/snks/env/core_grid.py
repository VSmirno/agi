"""Small MiniGrid domains for testing cross-ruleset learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions as MiniGridActions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Door, Goal, Key
from minigrid.minigrid_env import MiniGridEnv

from snks.env.core_types import ActionSpec, Observation, Transition


class GridAction(IntEnum):
    turn_left = 0
    turn_right = 1
    forward = 2
    interact = 3
    noop = 4


GRID_ACTIONS = tuple(action.name for action in GridAction)
_MINIGRID_ACTIONS = {
    GridAction.turn_left: MiniGridActions.left,
    GridAction.turn_right: MiniGridActions.right,
    GridAction.forward: MiniGridActions.forward,
    GridAction.interact: MiniGridActions.toggle,
}


@dataclass(frozen=True, slots=True)
class GridRules:
    """Rules that change grid physics without changing the agent schema."""

    consume_key: bool = False
    push_distance: int = 1
    appearance_mode: str = "independent"

    def __post_init__(self) -> None:
        if self.push_distance not in (1, 2):
            raise ValueError("push_distance must be 1 or 2 for the pilot")
        if self.appearance_mode not in ("independent", "fixed"):
            raise ValueError("appearance_mode must be 'independent' or 'fixed'")


class CoreGridWorld(MiniGridEnv):
    """Two compact, partially observed MiniGrid task families."""

    def __init__(
        self,
        family: str,
        rules: GridRules,
        seed: int = 0,
        max_steps: int = 64,
    ) -> None:
        if family not in {"door_key", "push_box"}:
            raise ValueError(f"unknown core grid family: {family}")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.family = family
        self.rules = rules
        self.seed = int(seed)
        self._success = False
        self.door_pos = (4, 3)
        self.key_pos = (2, 5)
        self.box_pos = (3, 3)
        self.goal_pos = (6, 3) if family == "door_key" else (5, 3)

        mission_space = MissionSpace(
            mission_func=lambda: "interact with objects to reach the local goal"
        )
        super().__init__(
            mission_space=mission_space,
            grid_size=8,
            max_steps=max_steps,
            see_through_walls=False,
            render_mode="rgb_array",
        )
        self.action_space = gym.spaces.Discrete(len(GRID_ACTIONS))

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self._success = False

        if self.family == "door_key":
            self.grid.vert_wall(self.door_pos[0], 1, height - 2)
            self.grid.set(*self.door_pos, Door("yellow", is_locked=True))
            self.grid.set(*self.key_pos, Key("yellow"))
            self.grid.set(*self.goal_pos, Goal())
            self.agent_pos = np.array([2, 3])
            self.agent_dir = 0
            return

        box_color = "blue"
        if self.rules.appearance_mode == "independent":
            box_color = str(self.np_random.choice(("blue", "purple", "yellow")))
        self.box_pos = (3, 3)
        self.grid.set(*self.box_pos, Box(box_color))
        self.grid.set(*self.goal_pos, Goal())
        self.agent_pos = np.array([2, 3])
        self.agent_dir = 0

    def reset(self, **kwargs: Any):
        """Reset physics and evaluator state."""
        if kwargs.get("seed") is None:
            kwargs["seed"] = self.seed
        self._success = False
        return super().reset(**kwargs)

    def step(self, action: int):
        """Dispatch one of the five shared actions."""
        try:
            selected = GridAction(int(action))
        except ValueError as exc:
            raise ValueError(f"invalid grid action: {action}") from exc

        if selected is GridAction.noop:
            self.step_count += 1
            reward = 0.0
            info: dict[str, Any] = {}
        elif self.family == "push_box" and selected is GridAction.interact:
            self.step_count += 1
            self._push_box()
            reward = 1.0 if self._success else 0.0
            info = {}
        elif (
            self.family == "door_key"
            and selected is GridAction.interact
            and self._pickup_key()
        ):
            self.step_count += 1
            reward = 0.0
            info = {}
        else:
            door_was_locked = self._door_is_locked()
            _, native_reward, _, _, info = super().step(_MINIGRID_ACTIONS[selected])
            reward = float(native_reward)
            if (
                self.family == "door_key"
                and selected is GridAction.interact
                and door_was_locked
                and not self._door_is_locked()
                and self.rules.consume_key
            ):
                self.carrying = None

        if self.family == "door_key":
            self._success = tuple(int(value) for value in self.agent_pos) == self.goal_pos
        terminated = self._success
        truncated = self.step_count >= self.max_steps and not terminated
        if terminated:
            reward = max(float(reward), 1.0)
        return self.gen_obs(), float(reward), terminated, truncated, info

    def diagnostic_snapshot(self) -> dict[str, Any]:
        """Expose ground truth only to the runner/evaluator."""
        snapshot: dict[str, Any] = {
            "success": self._success,
            "agent_pos": tuple(int(value) for value in self.agent_pos),
            "goal_pos": self.goal_pos,
        }
        if self.family == "door_key":
            door = self.grid.get(*self.door_pos)
            snapshot.update(
                door_open=bool(isinstance(door, Door) and door.is_open),
                carrying_key=isinstance(self.carrying, Key),
            )
        else:
            snapshot["box_pos"] = self.box_pos
        return snapshot

    def goal_observation(self) -> Observation:
        """Construct a separate solved local state for goal conditioning."""
        desired = CoreGridWorld(
            family=self.family,
            rules=self.rules,
            seed=self.seed,
            max_steps=self.max_steps,
        )
        desired.reset(seed=self.seed)
        if self.family == "door_key":
            door = desired.grid.get(*desired.door_pos)
            if not isinstance(door, Door):
                raise RuntimeError("door fixture is missing")
            door.is_locked = False
            door.is_open = True
            desired.grid.set(*desired.key_pos, None)
            if not desired.rules.consume_key:
                desired.carrying = Key("yellow")
            desired.agent_pos = np.array(desired.goal_pos)
            desired.agent_dir = 0
        else:
            box = desired.grid.get(*desired.box_pos)
            if not isinstance(box, Box):
                raise RuntimeError("box fixture is missing")
            desired.grid.set(*desired.box_pos, None)
            desired.grid.set(*desired.goal_pos, box)
            desired.box_pos = desired.goal_pos
            desired.agent_pos = np.array(
                [
                    desired.goal_pos[0] - desired.rules.push_distance - 1,
                    desired.goal_pos[1],
                ]
            )
            desired.agent_dir = 0
        return _project_grid_observation(desired, step=0)

    def _door_is_locked(self) -> bool:
        if self.family != "door_key":
            return False
        door = self.grid.get(*self.door_pos)
        return bool(isinstance(door, Door) and door.is_locked)

    def _push_box(self) -> None:
        direction = self.dir_vec
        front = (
            int(self.agent_pos[0] + direction[0]),
            int(self.agent_pos[1] + direction[1]),
        )
        box = self.grid.get(*front)
        if not isinstance(box, Box):
            return

        destination = (
            int(front[0] + direction[0] * self.rules.push_distance),
            int(front[1] + direction[1] * self.rules.push_distance),
        )
        for offset in range(1, self.rules.push_distance + 1):
            position = (
                int(front[0] + direction[0] * offset),
                int(front[1] + direction[1] * offset),
            )
            occupant = self.grid.get(*position)
            if occupant is not None and not (
                position == destination and isinstance(occupant, Goal)
            ):
                return

        self.grid.set(*front, None)
        self.grid.set(*destination, box)
        self.box_pos = destination
        self._success = destination == self.goal_pos

    def _pickup_key(self) -> bool:
        direction = self.dir_vec
        front = (
            int(self.agent_pos[0] + direction[0]),
            int(self.agent_pos[1] + direction[1]),
        )
        key = self.grid.get(*front)
        if not isinstance(key, Key) or self.carrying is not None:
            return False
        self.grid.set(*front, None)
        self.carrying = key
        return True


class GridCoreAdapter:
    """Project ``CoreGridWorld`` into the common core interface."""

    def __init__(self, world: CoreGridWorld) -> None:
        self.world = world
        self.actions = ActionSpec("grid-v1", GRID_ACTIONS)
        self.reset_transitions = 0
        self._last_observation: Observation | None = None

    def reset(self, seed: int | None = None) -> Observation:
        self.world.reset(seed=seed)
        self.reset_transitions = 0
        self._last_observation = _project_grid_observation(self.world, step=0)
        return self._last_observation

    def step(self, action: int) -> Transition:
        if self._last_observation is None:
            raise RuntimeError("reset must be called before step")
        before = self._last_observation
        _, _, terminated, truncated, _ = self.world.step(action)
        after = _project_grid_observation(self.world, step=self.world.step_count)
        self._last_observation = after
        return Transition(before, int(action), after, bool(terminated), bool(truncated))

    def diagnostic_snapshot(self) -> dict[str, Any]:
        return self.world.diagnostic_snapshot()

    def goal_observation(self) -> Observation:
        return self.world.goal_observation()

    def close(self) -> None:
        self.world.close()
        self._last_observation = None


def _project_grid_observation(world: CoreGridWorld, step: int) -> Observation:
    frame = np.asarray(
        world.get_frame(tile_size=8, agent_pov=True),
        dtype=np.uint8,
    )
    if frame.shape[:2] != (64, 64):
        row_index = (np.arange(64) * frame.shape[0] / 64).astype(int)
        column_index = (np.arange(64) * frame.shape[1] / 64).astype(int)
        frame = frame[np.ix_(row_index, column_index)]
    carrying = float(world.carrying is not None)
    return Observation(
        rgb=frame[:, :, :3].transpose(2, 0, 1),
        sensors=np.array([carrying], dtype=np.float32),
        sensor_mask=np.array([True]),
        schema="grid-v1",
        step=step,
    )
