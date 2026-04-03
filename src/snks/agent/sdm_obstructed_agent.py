"""Stage 59: SDM Learned Color Matching — ObstructedMaze-2Dl agent.

СНКС pipeline: obs 7×7 → SpatialMap → ColorStateEncoder (VSA) → SDM → planning.
Proves SDM can learn which key color opens which door from experience.

ObstructedMaze-2Dl: 16×16, 2 keys (random colors), 2 locked doors (matching colors),
1 closed door, goal = pickup blue ball behind locked doors.

Key challenge: agent must learn same_color(key, door) → opens.
Heuristic picks random key → 50% chance per door → ~25% for both doors.
SDM trained should pick correct key → near 100%.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

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
from snks.agent.vsa_world_model import SDMMemory, VSACodebook

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5

# MiniGrid color mapping
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}

# MiniGrid object types
OBJ_BALL = 6

# Door states in MiniGrid obs encoding
DOOR_OPEN = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2


class ColorStateEncoder:
    """Encode key-door color pairs as VSA vectors for SDM storage."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode_color(self, color_name: str) -> torch.Tensor:
        return self.cb.filler(f"color_{color_name}")


class ObstructedMazeEnv:
    """Wrapper for MiniGrid ObstructedMaze-2Dl."""

    def __init__(self, max_steps: int = 500):
        import gymnasium as gym
        import minigrid  # noqa: F401
        self._env = gym.make("MiniGrid-ObstructedMaze-2Dl-v0", max_steps=max_steps)
        self.grid_width = 16
        self.grid_height = 16

    def reset(self, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        return self._extract(obs)

    def step(self, action: int):
        obs, reward, term, trunc, info = self._env.step(action)
        img, col, row, d, carrying_color = self._extract(obs)
        return img, float(reward), term, trunc, col, row, d, carrying_color

    def _extract(self, obs):
        uw = self._env.unwrapped
        img = obs["image"]
        pos = uw.agent_pos
        carrying_color = None
        if uw.carrying is not None:
            carrying_color = uw.carrying.color
        return img, int(pos[0]), int(pos[1]), int(uw.agent_dir), carrying_color

    def get_all_doors(self):
        uw = self._env.unwrapped
        result = []
        for i in range(uw.grid.width):
            for j in range(uw.grid.height):
                obj = uw.grid.get(i, j)
                if obj is not None and obj.type == "door":
                    result.append((obj.color, i, j, obj.is_locked, obj.is_open))
        return result

    def get_all_keys(self):
        uw = self._env.unwrapped
        result = []
        for i in range(uw.grid.width):
            for j in range(uw.grid.height):
                obj = uw.grid.get(i, j)
                if obj is not None and obj.type == "key":
                    result.append((obj.color, i, j))
        return result

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped


class SDMObstructedAgent:
    """Learned ObstructedMaze agent using СНКС pipeline.

    Key learning task: which key color opens which door color.
    Exploration: try keys on doors, record (key_color, door_color) → success/fail.
    Planning: SDM recall → pick correct key for each locked door.
    """

    SG_EXPLORE = 0
    SG_GOTO_KEY = 1
    SG_GOTO_DOOR = 2
    SG_GOTO_BALL = 3
    SG_DROP_KEY = 4

    def __init__(self, grid_width: int = 16, grid_height: int = 16,
                 dim: int = 512, n_locations: int = 1000,
                 explore_episodes: int = 50,
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()

        self.codebook = VSACodebook(dim=dim, device=self.device)
        self.encoder = ColorStateEncoder(self.codebook)
        self.sdm = SDMMemory(n_locations=n_locations, dim=dim, device=self.device)

        self.explore_episodes = explore_episodes
        self._episode_count = 0
        self._exploring = True

        # Per-episode state
        self._carrying_color: str | None = None
        self._last_toggle_door_color: str | None = None
        self._needs_drop: bool = False
        self._opened_doors: set[str] = set()  # colors of doors we've opened
        self._target_key_color: str | None = None  # which key to pick up

    def reset_episode(self) -> None:
        self.spatial_map.reset()
        self._carrying_color = None
        self._last_toggle_door_color = None
        self._needs_drop = False
        self._opened_doors = set()
        self._target_key_color = None

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int,
                      carrying_color: str | None) -> int:
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        self._carrying_color = carrying_color

        if self._needs_drop and carrying_color is not None:
            self._needs_drop = False
            return ACT_DROP

        reflex = self._check_reflexes(obs_7x7, carrying_color)
        if reflex is not None:
            return reflex

        subgoal = self._select_subgoal(carrying_color)
        return self._execute_subgoal(subgoal, agent_row, agent_col, agent_dir)

    def _check_reflexes(self, obs_7x7: np.ndarray, carrying_color: str | None) -> int | None:
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # Facing closed unlocked door → toggle to open
        if front_obj == OBJ_DOOR and front_state == DOOR_CLOSED:
            return ACT_TOGGLE

        # Facing locked door with key → toggle
        if front_obj == OBJ_DOOR and front_state == DOOR_LOCKED and carrying_color is not None:
            self._last_toggle_door_color = IDX_TO_COLOR.get(front_color)
            return ACT_TOGGLE

        # Facing ball → pickup (this is the goal)
        if front_obj == OBJ_BALL and carrying_color is None:
            return ACT_PICKUP

        # Facing key and should pick up
        if front_obj == OBJ_KEY and carrying_color is None:
            key_color = IDX_TO_COLOR.get(front_color)
            if self._should_pickup_key(key_color):
                return ACT_PICKUP

        return None

    def _should_pickup_key(self, key_color: str | None) -> bool:
        if key_color is None:
            return True
        if not self._exploring and self._target_key_color is not None:
            return key_color == self._target_key_color
        # During exploration: pick up any key
        return True

    def _select_subgoal(self, carrying_color: str | None) -> int:
        locked_doors = self._find_locked_doors()
        ball_pos = self._find_ball()

        # No more locked doors → go to ball
        if not locked_doors:
            if ball_pos is not None:
                return self.SG_GOTO_BALL
            return self.SG_EXPLORE

        if carrying_color is not None:
            # Have a key — go to a locked door
            # During planning: SDM picks which door. During exploration: nearest.
            return self.SG_GOTO_DOOR

        # Not carrying — need to pick up a key
        # During planning: SDM decides which key based on next locked door
        if not self._exploring:
            next_door = locked_doors[0]
            best_key = self._sdm_select_key(next_door[2])  # door color
            if best_key is not None:
                self._target_key_color = best_key
        key_pos = self._find_nearest_key()
        if key_pos is not None:
            return self.SG_GOTO_KEY

        return self.SG_EXPLORE

    def _sdm_select_key(self, door_color: str) -> str | None:
        """SDM: which key color works for this door color?"""
        best_color = None
        best_reward = -float("inf")
        action_vsa = self.encoder.encode_color(door_color)
        for color_name in COLOR_TO_IDX:
            state_vsa = self.encoder.encode_color(color_name)
            reward = self.sdm.read_reward(state_vsa, action_vsa)
            if reward > best_reward:
                best_reward = reward
                best_color = color_name
        if best_reward > 0:
            return best_color
        return None

    def _find_locked_doors(self) -> list[tuple[int, int, str]]:
        """Find all locked doors in spatial map. Returns [(row, col, color_name)]."""
        result = []
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                if int(self.spatial_map.grid[r, c, 0]) == OBJ_DOOR:
                    if int(self.spatial_map.grid[r, c, 2]) == DOOR_LOCKED:
                        color_id = int(self.spatial_map.grid[r, c, 1])
                        result.append((r, c, IDX_TO_COLOR.get(color_id, "unknown")))
        return result

    def _find_ball(self) -> tuple[int, int] | None:
        """Find blue ball position."""
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                if int(self.spatial_map.grid[r, c, 0]) == OBJ_BALL:
                    return (r, c)
        return None

    def _find_nearest_key(self) -> tuple[int, int] | None:
        """Find key position, preferring target color if set."""
        if self._target_key_color is not None:
            color_id = COLOR_TO_IDX.get(self._target_key_color)
            if color_id is not None:
                pos = self.spatial_map.find_object_by_type_color(OBJ_KEY, color_id)
                if pos is not None:
                    return pos
        return self.spatial_map.find_object(OBJ_KEY)

    def _execute_subgoal(self, subgoal: int,
                         agent_row: int, agent_col: int, agent_dir: int) -> int:
        target_pos = None

        if subgoal == self.SG_GOTO_KEY:
            target_pos = self._find_nearest_key()
        elif subgoal == self.SG_GOTO_DOOR:
            doors = self._find_locked_doors()
            if doors:
                target_pos = (doors[0][0], doors[0][1])
        elif subgoal == self.SG_GOTO_BALL:
            target_pos = self._find_ball()

        if target_pos is None:
            return self._explore_action(agent_row, agent_col, agent_dir)

        adj = self._find_adjacent_walkable(target_pos, agent_row, agent_col)
        if adj is None:
            return self._explore_action(agent_row, agent_col, agent_dir)

        dr = abs(agent_row - target_pos[0])
        dc = abs(agent_col - target_pos[1])
        if dr + dc == 1:
            return self._turn_toward(target_pos[0], target_pos[1],
                                     agent_row, agent_col, agent_dir)

        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _explore_action(self, agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Frontier exploration that skips frontiers behind locked doors."""
        obs = self.spatial_map.to_obs()
        h, w = self.spatial_map.height, self.spatial_map.width
        frontier_set = set(self.spatial_map.frontiers())
        if not frontier_set:
            return int(np.random.randint(0, 3))

        queue = deque([(agent_row, agent_col)])
        visited = {(agent_row, agent_col)}
        target = None
        while queue:
            r, c = queue.popleft()
            if (r, c) in frontier_set and (r, c) != (agent_row, agent_col):
                target = (r, c)
                break
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    obj = int(obs[nr, nc, 0])
                    if obj == OBJ_WALL:
                        continue
                    if obj == OBJ_DOOR and int(obs[nr, nc, 2]) == DOOR_LOCKED:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        if target is None:
            return int(np.random.randint(0, 3))

        actual_target = target
        if int(obs[target[0], target[1], 0]) == OBJ_DOOR:
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ar, ac = target[0] + dr, target[1] + dc
                if 0 <= ar < h and 0 <= ac < w:
                    if int(obs[ar, ac, 0]) not in (OBJ_WALL, OBJ_DOOR):
                        actual_target = (ar, ac)
                        break

        if actual_target == (agent_row, agent_col):
            return int(np.random.randint(0, 2))

        path = self.pathfinder.find_path(
            obs, (agent_row, agent_col), actual_target, allow_door=False
        )
        if path is None or len(path) <= 1:
            return int(np.random.randint(0, 3))
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        return actions[0] if actions else ACT_FORWARD

    def _find_adjacent_walkable(self, pos, agent_row, agent_col):
        obs = self.spatial_map.to_obs()
        for t in (OBJ_KEY, OBJ_BALL, 7):  # key, ball, box
            mask = obs[:, :, 0] == t
            obs[mask, 0] = OBJ_WALL
        best = None
        best_dist = float("inf")
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.spatial_map.height and 0 <= nc < self.spatial_map.width):
                continue
            if int(obs[nr, nc, 0]) == OBJ_WALL:
                continue
            path = self.pathfinder.find_path(obs, (agent_row, agent_col), (nr, nc), allow_door=True)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (nr, nc)
        return best

    def _navigate_to(self, tr, tc, agent_row, agent_col, agent_dir):
        if agent_row == tr and agent_col == tc:
            return int(np.random.randint(0, 3))
        obs = self.spatial_map.to_obs()
        path = self.pathfinder.find_path(obs, (agent_row, agent_col), (tr, tc), allow_door=True)
        if path is None or len(path) <= 1:
            return self._explore_action(agent_row, agent_col, agent_dir)
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        return actions[0] if actions else ACT_FORWARD

    def _turn_toward(self, tr, tc, ar, ac, agent_dir):
        dr, dc = tr - ar, tc - ac
        if dc > 0: need = 0
        elif dr > 0: need = 1
        elif dc < 0: need = 2
        else: need = 3
        if need == agent_dir:
            return ACT_FORWARD
        diff = (need - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       carrying_color: str | None, reward: float) -> None:
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        if self._last_toggle_door_color is not None and self._carrying_color is not None:
            door_color = self._last_toggle_door_color
            key_color = self._carrying_color

            # Check if door we tried is still locked
            still_locked = False
            for ld in self._find_locked_doors():
                if ld[2] == door_color:
                    still_locked = True
                    break

            if still_locked:
                self._record_color_transition(key_color, door_color, success=False)
                self._needs_drop = True
            else:
                self._record_color_transition(key_color, door_color, success=True)
                self._opened_doors.add(door_color)

            self._last_toggle_door_color = None

        self._carrying_color = carrying_color

    def _record_color_transition(self, key_color: str, door_color: str, success: bool) -> None:
        state_vsa = self.encoder.encode_color(key_color)
        action_vsa = self.encoder.encode_color(door_color)
        reward = 1.0 if success else -1.0
        n_writes = 10 if success else 5
        for _ in range(n_writes):
            self.sdm.write(state_vsa, action_vsa, state_vsa, reward)

    def episode_done(self, success: bool) -> None:
        self._episode_count += 1
        self._exploring = self._episode_count < self.explore_episodes
        self.reset_episode()

    @property
    def is_exploring(self) -> bool:
        return self._exploring
