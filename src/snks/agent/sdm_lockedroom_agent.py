"""Stage 59: SDM Learned Color Matching — LockedRoom agent.

СНКС pipeline: obs 7×7 → SpatialMap → ColorStateEncoder (VSA) → SDM → planning.
Proves SDM can learn which key color opens which door from experience.

Two-phase proof:
- Phase B: mission text provides key color hint, SDM records transitions
- Phase A: no mission text, SDM must learn same_color(key, door) → success

LockedRoom: 19×19, 6 rooms, 6 colored doors (1 locked), 1 key.
"""

from __future__ import annotations

import re

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

# Door states in MiniGrid obs encoding
DOOR_OPEN = 0
DOOR_CLOSED = 1
DOOR_LOCKED = 2


class MissionParser:
    """Parse LockedRoom mission text to extract key/door colors."""

    _PATTERN = re.compile(
        r"get the (\w+) key from the (\w+) room, unlock the (\w+) door"
    )

    def parse(self, mission: str) -> dict[str, str] | None:
        m = self._PATTERN.search(mission)
        if not m:
            return None
        return {
            "key_color": m.group(1),
            "room_color": m.group(2),
            "door_color": m.group(3),
        }


class ColorStateEncoder:
    """Encode key-door color pairs as VSA vectors for SDM storage."""

    def __init__(self, codebook: VSACodebook):
        self.cb = codebook

    def encode_color(self, color_name: str) -> torch.Tensor:
        """Encode a single color name as VSA vector."""
        return self.cb.filler(f"color_{color_name}")

    def encode_color_pair(self, key_color: str, door_color: str) -> torch.Tensor:
        """Encode (key_color, door_color) pair — used as SDM address."""
        kv = self.cb.bind(self.cb.role("key_color"), self.cb.filler(f"color_{key_color}"))
        dv = self.cb.bind(self.cb.role("door_color"), self.cb.filler(f"color_{door_color}"))
        return self.cb.bundle([kv, dv])


class LockedRoomEnv:
    """Wrapper for MiniGrid LockedRoom with partial observation."""

    def __init__(self, max_steps: int = 1000):
        import gymnasium as gym
        import minigrid  # noqa: F401 — needed for env registration
        self._env = gym.make("MiniGrid-LockedRoom-v0", max_steps=max_steps)
        self.grid_width = 19
        self.grid_height = 19

    def reset(self, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        return self._extract(obs)

    def step(self, action: int):
        obs, reward, term, trunc, info = self._env.step(action)
        img, col, row, d, carrying_color, mission = self._extract(obs)
        return img, float(reward), term, trunc, col, row, d, carrying_color, mission

    def _extract(self, obs):
        uw = self._env.unwrapped
        img = obs["image"]
        mission = obs.get("mission", "")
        pos = uw.agent_pos
        carrying_color = None
        if uw.carrying is not None:
            carrying_color = uw.carrying.color
        return img, int(pos[0]), int(pos[1]), int(uw.agent_dir), carrying_color, mission

    def get_all_doors(self) -> list[tuple[str, int, int, bool, bool]]:
        """Debug: get all doors from full grid. Returns (color, col, row, is_locked, is_open)."""
        uw = self._env.unwrapped
        result = []
        for i in range(uw.grid.width):
            for j in range(uw.grid.height):
                obj = uw.grid.get(i, j)
                if obj is not None and obj.type == "door":
                    result.append((obj.color, i, j, obj.is_locked, obj.is_open))
        return result

    def get_all_keys(self) -> list[tuple[str, int, int]]:
        """Debug: get all keys from full grid. Returns (color, col, row)."""
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


class SDMLockedRoomAgent:
    """Learned LockedRoom agent using СНКС pipeline.

    Exploration phase: FrontierExplorer navigates, SDM records color transitions.
    Planning phase: SDM reward lookup selects which door to try for held key.
    """

    # Subgoal IDs
    SG_EXPLORE = 0
    SG_GOTO_KEY = 1
    SG_GOTO_DOOR = 2
    SG_GOTO_GOAL = 3
    SG_DROP_KEY = 4

    def __init__(self, grid_width: int = 19, grid_height: int = 19,
                 dim: int = 512, n_locations: int = 1000,
                 explore_episodes: int = 50,
                 use_mission: bool = True,
                 device: torch.device | str | None = None):
        self.device = torch.device(device) if device else torch.device("cpu")
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()

        self.codebook = VSACodebook(dim=dim, device=self.device)
        self.encoder = ColorStateEncoder(self.codebook)
        self.sdm = SDMMemory(n_locations=n_locations, dim=dim, device=self.device)

        self.mission_parser = MissionParser()
        self.use_mission = use_mission
        self.explore_episodes = explore_episodes

        self._episode_count = 0
        self._exploring = True

        # Per-episode state
        self._carrying_color: str | None = None
        self._target_door_color: str | None = None  # which door we're heading to
        self._tried_doors: set[str] = set()  # doors tried with current key
        self._parsed_mission: dict[str, str] | None = None
        self._last_toggle_door_color: str | None = None
        self._needs_drop: bool = False

    def reset_episode(self) -> None:
        self.spatial_map.reset()
        self._carrying_color = None
        self._target_door_color = None
        self._tried_doors = set()
        self._parsed_mission = None
        self._last_toggle_door_color = None
        self._needs_drop = False

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int,
                      carrying_color: str | None, mission: str) -> int:
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        self._carrying_color = carrying_color

        # Parse mission once per episode
        if self._parsed_mission is None and mission:
            self._parsed_mission = self.mission_parser.parse(mission)

        # Drop key if we need to (wrong key tried on locked door)
        if self._needs_drop and carrying_color is not None:
            self._needs_drop = False
            return ACT_DROP

        # Reflexes: pickup key when facing one (if not carrying)
        reflex = self._check_reflexes(obs_7x7, carrying_color)
        if reflex is not None:
            return reflex

        # Select and execute subgoal
        subgoal = self._select_subgoal(carrying_color)
        return self._execute_subgoal(subgoal, agent_row, agent_col, agent_dir)

    def _check_reflexes(self, obs_7x7: np.ndarray, carrying_color: str | None) -> int | None:
        front_obj = int(obs_7x7[3, 5, 0])
        front_state = int(obs_7x7[3, 5, 2])
        front_color = int(obs_7x7[3, 5, 1])

        # Facing closed unlocked door → toggle to open
        if front_obj == OBJ_DOOR and front_state == DOOR_CLOSED:
            return ACT_TOGGLE

        # Facing locked door with key → toggle (try to unlock)
        if front_obj == OBJ_DOOR and front_state == DOOR_LOCKED and carrying_color is not None:
            self._last_toggle_door_color = IDX_TO_COLOR.get(front_color)
            return ACT_TOGGLE

        # Facing key and should pick up
        if front_obj == OBJ_KEY and carrying_color is None:
            if self._should_pickup_key(IDX_TO_COLOR.get(front_color)):
                return ACT_PICKUP

        return None

    def _should_pickup_key(self, key_color: str | None) -> bool:
        if key_color is None:
            return True
        # Phase B with mission: only pick up the correct color
        if self.use_mission and self._parsed_mission:
            return key_color == self._parsed_mission["key_color"]
        # Phase A or no mission parsed: pick up any key
        return True

    def _select_subgoal(self, carrying_color: str | None) -> int:
        if carrying_color is not None:
            # We have a key — find which door to try
            target = self._select_target_door(carrying_color)
            if target is not None:
                self._target_door_color = target
                return self.SG_GOTO_DOOR
            # No target door found — explore more
            return self.SG_EXPLORE

        # Not carrying key
        locked_door = self._find_locked_door()
        if locked_door is not None:
            # We know there's a locked door — find the key
            key_target = self._select_target_key()
            if key_target is not None:
                return self.SG_GOTO_KEY

        # Check if goal is accessible (door already opened)
        goal_pos = self._find_goal()
        if goal_pos is not None:
            # Check if path to goal exists (no locked doors in way)
            return self.SG_GOTO_GOAL

        return self.SG_EXPLORE

    def _select_target_door(self, key_color: str) -> str | None:
        """Select which locked door to try with current key.

        Planning mode: query SDM for best color match.
        Exploration mode: use mission hint or try untried doors.
        """
        locked = self._find_locked_door()
        if locked is None:
            return None
        locked_color = locked[2]

        if not self._exploring:
            # SDM planning: query reward for (key_color, locked_door_color)
            best_color = self._sdm_select_door(key_color)
            if best_color is not None:
                return best_color

        # Exploration / fallback
        if self.use_mission and self._parsed_mission:
            return self._parsed_mission["door_color"]

        # No mission: just go to the locked door
        return locked_color

    def _sdm_select_door(self, key_color: str) -> str | None:
        """Query SDM: which door color gives best reward for this key color?"""
        best_color = None
        best_reward = -float("inf")

        state_vsa = self.encoder.encode_color(key_color)

        for color_name in COLOR_TO_IDX:
            action_vsa = self.encoder.encode_color(color_name)
            reward = self.sdm.read_reward(state_vsa, action_vsa)
            if reward > best_reward:
                best_reward = reward
                best_color = color_name

        # Only return if we have a clear positive signal
        if best_reward > 0:
            return best_color
        return None

    def _select_target_key(self) -> tuple[int, int] | None:
        """Find position of the key we should pick up."""
        if self.use_mission and self._parsed_mission:
            target_color = self._parsed_mission["key_color"]
            color_id = COLOR_TO_IDX.get(target_color)
            if color_id is not None:
                pos = self.spatial_map.find_object_by_type_color(OBJ_KEY, color_id)
                if pos is not None:
                    return pos
        # Fallback: any key
        return self.spatial_map.find_object(OBJ_KEY)

    def _find_locked_door(self) -> tuple[int, int, str] | None:
        """Find locked door position and color from spatial map."""
        for r in range(self.spatial_map.height):
            for c in range(self.spatial_map.width):
                if not self.spatial_map.explored[r, c]:
                    continue
                if int(self.spatial_map.grid[r, c, 0]) == OBJ_DOOR:
                    state = int(self.spatial_map.grid[r, c, 2])
                    if state == DOOR_LOCKED:
                        color_id = int(self.spatial_map.grid[r, c, 1])
                        color_name = IDX_TO_COLOR.get(color_id, "unknown")
                        return (r, c, color_name)
        return None

    def _find_goal(self) -> tuple[int, int] | None:
        return self.spatial_map.find_object(OBJ_GOAL)

    def _execute_subgoal(self, subgoal: int,
                         agent_row: int, agent_col: int, agent_dir: int) -> int:
        target_pos = None

        if subgoal == self.SG_GOTO_KEY:
            key_pos = self._select_target_key()
            if key_pos is not None:
                target_pos = key_pos

        elif subgoal == self.SG_GOTO_DOOR:
            locked = self._find_locked_door()
            if locked is not None:
                target_pos = (locked[0], locked[1])

        elif subgoal == self.SG_GOTO_GOAL:
            target_pos = self._find_goal()

        if target_pos is None:
            return self._explore_action(agent_row, agent_col, agent_dir)

        # Navigate to adjacent cell
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
        # BFS from agent — only through non-locked-door cells
        from collections import deque
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
                    # Locked doors are impassable without key
                    if obj == OBJ_DOOR and int(obs[nr, nc, 2]) == DOOR_LOCKED:
                        continue
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        if target is None:
            return int(np.random.randint(0, 3))

        # Navigate to target (or adjacent cell if it's a door)
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

    def _find_adjacent_walkable(self, pos: tuple[int, int],
                                agent_row: int, agent_col: int) -> tuple[int, int] | None:
        obs = self.spatial_map.to_obs()
        for t in (OBJ_KEY, 6, 7):  # key, ball, box
            mask = obs[:, :, 0] == t
            obs[mask, 0] = OBJ_WALL
        best = None
        best_dist = float("inf")
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.spatial_map.height and 0 <= nc < self.spatial_map.width):
                continue
            if int(obs[nr, nc, 0]) in (OBJ_WALL,):
                continue
            path = self.pathfinder.find_path(obs, (agent_row, agent_col), (nr, nc), allow_door=True)
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (nr, nc)
        return best

    def _navigate_to(self, tr: int, tc: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        if agent_row == tr and agent_col == tc:
            return int(np.random.randint(0, 3))
        obs = self.spatial_map.to_obs()
        path = self.pathfinder.find_path(obs, (agent_row, agent_col), (tr, tc), allow_door=True)
        if path is None or len(path) <= 1:
            return self._explore_action(agent_row, agent_col, agent_dir)
        actions = self.pathfinder.path_to_actions(path, agent_dir)
        return actions[0] if actions else ACT_FORWARD

    def _turn_toward(self, tr: int, tc: int,
                     ar: int, ac: int, agent_dir: int) -> int:
        dr, dc = tr - ar, tc - ac
        if dc > 0:
            need = 0
        elif dr > 0:
            need = 1
        elif dc < 0:
            need = 2
        else:
            need = 3
        if need == agent_dir:
            return ACT_FORWARD
        diff = (need - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       carrying_color: str | None, reward: float) -> None:
        """Called after each step to update state and record SDM transitions."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)

        # Detect toggle result: if we tried to toggle a locked door
        if self._last_toggle_door_color is not None and self._carrying_color is not None:
            door_color = self._last_toggle_door_color
            key_color = self._carrying_color

            # Check if door is still locked (toggle failed = wrong key)
            locked = self._find_locked_door()
            if locked is not None and locked[2] == door_color:
                # Toggle failed — wrong key
                self._record_color_transition(key_color, door_color, success=False)
                self._tried_doors.add(door_color)
                self._needs_drop = True
            else:
                # Toggle succeeded — right key!
                self._record_color_transition(key_color, door_color, success=True)

            self._last_toggle_door_color = None

        # Record success on episode reward
        if reward > 0 and self._carrying_color is not None:
            locked = self._find_locked_door()
            if locked is None:  # door was opened
                # We don't know which door it was, but we got reward
                pass

        self._carrying_color = carrying_color

    def _record_color_transition(self, key_color: str, door_color: str,
                                 success: bool) -> None:
        """Record (key_color, door_color) → reward in SDM."""
        state_vsa = self.encoder.encode_color(key_color)
        action_vsa = self.encoder.encode_color(door_color)
        reward = 1.0 if success else -1.0
        # Amplify signal
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
