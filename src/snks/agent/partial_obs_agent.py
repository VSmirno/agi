"""Stage 54: PartialObsAgent — DoorKey navigation with 7x7 partial observation.

Uses SpatialMap to accumulate observations, FrontierExplorer for exploration,
and BFS pathfinding + subgoal logic for goal-directed navigation.
"""

from __future__ import annotations

import numpy as np

from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    OBJ_DOOR,
    OBJ_EMPTY,
    OBJ_GOAL,
    OBJ_KEY,
    OBJ_WALL,
)

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5

# Direction deltas: dir → (d_row, d_col)
DIR_DR = {0: 0, 1: 1, 2: 0, 3: -1}
DIR_DC = {0: 1, 1: 0, 2: -1, 3: 0}


class PartialObsAgent:
    """Navigate DoorKey with 7x7 partial observation.

    Strategy:
    1. Update spatial map with each observation
    2. If carrying key and facing locked door → toggle
    3. If at key position → pickup
    4. If all objects known → BFS to current target (key → door → goal)
    5. If not all known → frontier exploration
    """

    def __init__(self, grid_width: int, grid_height: int, epsilon: float = 0.05):
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.epsilon = epsilon
        self._has_key = False
        self._door_open = False

    def reset(self) -> None:
        self.spatial_map.reset()
        self._has_key = False
        self._door_open = False

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int) -> int:
        """Main action selection."""
        # Update map
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Epsilon exploration
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return int(np.random.randint(0, 3))  # left/right/forward

        # Check immediate interactions from partial obs
        action = self._check_immediate_action(obs_7x7, agent_dir)
        if action is not None:
            return action

        # Find known objects
        objs = self.spatial_map.find_objects()
        key_pos = objs["key_pos"]
        door_pos = objs["door_pos"]
        goal_pos = objs["goal_pos"]

        # If adjacent to target object, turn to face it
        action = self._turn_to_face_target(key_pos, door_pos,
                                           agent_row, agent_col, agent_dir)
        if action is not None:
            return action

        # Determine current target
        target_rc = self._current_target(key_pos, door_pos, goal_pos, agent_row, agent_col)

        if target_rc is not None:
            return self._navigate_to(target_rc[0], target_rc[1],
                                     agent_row, agent_col, agent_dir)

        # Explore: BFS to nearest frontier
        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    def _turn_to_face_target(self, key_pos: tuple | None, door_pos: tuple | None,
                             agent_row: int, agent_col: int,
                             agent_dir: int) -> int | None:
        """If adjacent to key/door, turn to face it for pickup/toggle."""
        # Which object should we face?
        target_pos = None
        if not self._has_key and key_pos is not None:
            target_pos = key_pos
        elif self._has_key and not self._door_open and door_pos is not None:
            target_pos = door_pos

        if target_pos is None:
            return None

        tr, tc = target_pos
        dr = tr - agent_row
        dc = tc - agent_col

        # Must be adjacent (Manhattan distance 1)
        if abs(dr) + abs(dc) != 1:
            return None

        # What direction do we need to face?
        need_dir = self._dir_from_delta(dr, dc)
        if need_dir is None or need_dir == agent_dir:
            return None  # already facing or can't determine

        # Turn toward target
        diff = (need_dir - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    @staticmethod
    def _dir_from_delta(dr: int, dc: int) -> int | None:
        """Convert (dr, dc) to MiniGrid direction."""
        if dc > 0:
            return 0  # right
        if dr > 0:
            return 1  # down
        if dc < 0:
            return 2  # left
        if dr < 0:
            return 3  # up
        return None

    def _check_immediate_action(self, obs_7x7: np.ndarray,
                                agent_dir: int) -> int | None:
        """Check if agent should interact with adjacent cell."""
        # MiniGrid obs encoding: img[i,j] = view(col=i, row=j)
        # Agent at view(col=3, row=6) = obs[3,6]
        # Front cell = view(col=3, row=5) = obs[3,5]
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])

        # If facing a key and don't have one → pickup
        if front_obj == OBJ_KEY and not self._has_key:
            return ACT_PICKUP

        # If facing a locked door and have key → toggle
        if front_obj == OBJ_DOOR and front_state == 2 and self._has_key:
            return ACT_TOGGLE

        # If facing a closed door → toggle
        if front_obj == OBJ_DOOR and front_state == 1:
            return ACT_TOGGLE

        return None

    def _current_target(self, key_pos: tuple | None, door_pos: tuple | None,
                        goal_pos: tuple | None,
                        agent_row: int, agent_col: int) -> tuple[int, int] | None:
        """Determine current navigation target (row, col).

        For key and door: navigate to adjacent cell (can't walk onto them).
        For goal: navigate directly (Goal is overlap-able in MiniGrid).
        """
        if not self._has_key:
            if key_pos is not None:
                return self._adjacent_cell(key_pos, agent_row, agent_col)
            return None  # need to explore to find key

        if not self._door_open:
            if door_pos is not None:
                return self._adjacent_cell(door_pos, agent_row, agent_col)
            return None  # need to explore to find door

        if goal_pos is not None:
            return goal_pos
        return None  # need to explore to find goal

    def _adjacent_cell(self, target_pos: tuple[int, int],
                       agent_row: int, agent_col: int) -> tuple[int, int]:
        """Find best reachable cell adjacent to target (for pickup/toggle)."""
        dr, dc = target_pos
        obs = self._pathfinding_obs()
        best = None
        best_dist = float('inf')

        for ddr, ddc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ar, ac = dr + ddr, dc + ddc
            if not (0 <= ar < self.spatial_map.height and 0 <= ac < self.spatial_map.width):
                continue
            obj = int(obs[ar, ac, 0])
            if obj in (OBJ_WALL, OBJ_DOOR):
                continue
            path = self.pathfinder.find_path(obs, (agent_row, agent_col), (ar, ac),
                                             allow_door=True)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (ar, ac)

        if best is not None:
            return best
        # Fallback: just go to target position
        return target_pos

    def _pathfinding_obs(self) -> np.ndarray:
        """Create observation for BFS where non-passable objects are walls."""
        obs = self.spatial_map.to_obs()
        # Mark keys as walls for pathfinding (can't walk onto them)
        key_mask = obs[:, :, 0] == OBJ_KEY
        obs[key_mask, 0] = OBJ_WALL
        return obs

    def _navigate_to(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """BFS navigate to target position."""
        if agent_row == target_row and agent_col == target_col:
            # At target — turn to face something useful or explore
            return int(np.random.randint(0, 3))

        obs = self._pathfinding_obs()
        allow_door = self._door_open or self._has_key
        path = self.pathfinder.find_path(obs, (agent_row, agent_col),
                                         (target_row, target_col),
                                         allow_door=allow_door)
        if path is None:
            # Try with doors allowed
            path = self.pathfinder.find_path(obs, (agent_row, agent_col),
                                             (target_row, target_col),
                                             allow_door=True)
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return ACT_FORWARD

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       reward: float) -> None:
        """Update state after action execution."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Detect key pickup from observation
        self._detect_inventory(obs_7x7)
        # Detect door state
        self._detect_door_state()

    def _detect_inventory(self, obs_7x7: np.ndarray) -> None:
        """Detect if agent picked up key.

        In MiniGrid partial obs, when carrying an object, the object disappears
        from the grid and the carrying flag is set.
        """
        # If key was visible before but not anymore, agent likely picked it up
        key_pos = self.spatial_map.find_object(OBJ_KEY)
        if self._has_key:
            return
        # Check if key disappeared from known location
        # This is tricky with partial obs — rely on env carrying state
        # For now, track via the agent's obs: if we executed pickup and
        # key is no longer visible, assume we have it
        # Better: check the MiniGrid carrying property through env

    def _detect_door_state(self) -> None:
        """Detect if door was opened from spatial map."""
        door_pos = self.spatial_map.find_object(OBJ_DOOR)
        if door_pos is not None:
            state = int(self.spatial_map.grid[door_pos[0], door_pos[1], 2])
            if state == 0:  # open
                self._door_open = True

    def update_inventory(self, has_key: bool) -> None:
        """Explicitly set inventory state (called by env wrapper)."""
        self._has_key = has_key


class PartialObsDoorKeyEnv:
    """DoorKey with standard 7x7 partial observation (no FullyObsWrapper).

    Wraps MiniGrid DoorKey and provides agent position/direction info
    needed for SpatialMap updates.
    """

    def __init__(self, size: int = 5, max_steps: int = 200, seed: int | None = None):
        from minigrid.envs.doorkey import DoorKeyEnv
        self._env = DoorKeyEnv(size=size, max_steps=max_steps)
        self._seed = seed
        self.size = size

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, int, int]:
        """Reset and return (obs_7x7, agent_col, agent_row, agent_dir)."""
        s = seed if seed is not None else self._seed
        obs, info = self._env.reset(seed=s)
        img = obs["image"]
        pos = self._env.agent_pos
        d = self._env.agent_dir
        return img, int(pos[0]), int(pos[1]), int(d)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, int, int, int]:
        """Step and return (obs_7x7, reward, terminated, truncated, agent_col, agent_row, agent_dir)."""
        obs, reward, term, trunc, info = self._env.step(action)
        img = obs["image"]
        pos = self._env.agent_pos
        d = self._env.agent_dir
        return img, float(reward), term, trunc, int(pos[0]), int(pos[1]), int(d)

    @property
    def carrying(self) -> bool:
        """Whether agent is carrying an object."""
        return self._env.carrying is not None

    @property
    def unwrapped(self):
        return self._env


class MultiRoomPartialObsAgent:
    """Navigate MultiRoom environments with 7x7 partial observation.

    Strategy:
    1. Update spatial map with each observation
    2. If facing closed door → toggle
    3. If goal found → BFS navigate to goal
    4. Else → frontier exploration (BFS to nearest unknown)
    """

    def __init__(self, grid_width: int = 25, grid_height: int = 25,
                 epsilon: float = 0.05):
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.epsilon = epsilon

    def reset(self) -> None:
        self.spatial_map.reset()

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int) -> int:
        """Main action selection."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Epsilon exploration
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return int(np.random.randint(0, 3))

        # If facing closed door → toggle
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        if front_obj == OBJ_DOOR and front_state in (1, 2):
            return ACT_TOGGLE

        # If goal found → navigate to it
        goal_pos = self.spatial_map.find_object(OBJ_GOAL)
        if goal_pos is not None:
            return self._navigate_to(goal_pos[0], goal_pos[1],
                                     agent_row, agent_col, agent_dir)

        # Explore: frontier navigation
        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    def _navigate_to(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """BFS navigate to target, treating doors as passable."""
        if agent_row == target_row and agent_col == target_col:
            return int(np.random.randint(0, 3))

        obs = self.spatial_map.to_obs()
        path = self.pathfinder.find_path(obs, (agent_row, agent_col),
                                         (target_row, target_col),
                                         allow_door=True)
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return ACT_FORWARD

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       reward: float) -> None:
        """Update spatial map after action."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)


class PartialObsMultiRoomEnv:
    """MultiRoom-N3 with standard 7x7 partial observation (no FullyObsWrapper)."""

    def __init__(self, n_rooms: int = 3, max_room_size: int = 6,
                 max_steps: int = 300):
        from minigrid.envs.multiroom import MultiRoomEnv
        self._env = MultiRoomEnv(
            minNumRooms=n_rooms, maxNumRooms=n_rooms,
            maxRoomSize=max_room_size, max_steps=max_steps,
        )
        self.grid_width = 25
        self.grid_height = 25

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, int, int]:
        """Reset and return (obs_7x7, agent_col, agent_row, agent_dir)."""
        obs, info = self._env.reset(seed=seed)
        img = obs["image"]
        pos = self._env.agent_pos
        d = self._env.agent_dir
        self.grid_width = self._env.grid.width
        self.grid_height = self._env.grid.height
        return img, int(pos[0]), int(pos[1]), int(d)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, int, int, int]:
        obs, reward, term, trunc, info = self._env.step(action)
        img = obs["image"]
        pos = self._env.agent_pos
        d = self._env.agent_dir
        return img, float(reward), term, trunc, int(pos[0]), int(pos[1]), int(d)

    @property
    def unwrapped(self):
        return self._env
