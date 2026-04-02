"""Stage 56: PutNextAgent — BabyAI PutNext task with multi-object handling.

Parses mission text, tracks multiple objects by (type, color) in SpatialMap,
and executes pickup → navigate → drop sequence using BFS pathfinding.
"""

from __future__ import annotations

import re

import numpy as np

from snks.agent.pathfinding import GridPathfinder
from snks.agent.spatial_map import (
    FrontierExplorer,
    SpatialMap,
    OBJ_EMPTY,
    OBJ_UNSEEN,
    OBJ_WALL,
)

# MiniGrid actions
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2
ACT_PICKUP = 3
ACT_DROP = 4
ACT_TOGGLE = 5

# MiniGrid object type IDs
OBJ_KEY = 5
OBJ_BALL = 6
OBJ_BOX = 7

TYPE_NAMES = {'key': OBJ_KEY, 'ball': OBJ_BALL, 'box': OBJ_BOX}
COLOR_NAMES = {'red': 0, 'green': 1, 'blue': 2, 'purple': 3, 'yellow': 4, 'grey': 5}

# Direction deltas: dir → (d_row, d_col)
DIR_DR = {0: 0, 1: 1, 2: 0, 3: -1}
DIR_DC = {0: 1, 1: 0, 2: -1, 3: 0}

_MISSION_RE = re.compile(
    r'put the (\w+) (\w+) next to the (\w+) (\w+)'
)


class MissionParser:
    """Parse BabyAI PutNext mission strings."""

    def parse(self, mission: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse mission → ((source_type, source_color), (target_type, target_color)).

        Returns MiniGrid integer IDs.
        """
        m = _MISSION_RE.match(mission)
        if not m:
            raise ValueError(f"Cannot parse mission: {mission!r}")

        src_color, src_type, tgt_color, tgt_type = m.groups()
        source = (TYPE_NAMES[src_type], COLOR_NAMES[src_color])
        target = (TYPE_NAMES[tgt_type], COLOR_NAMES[tgt_color])
        return source, target


class PutNextAgent:
    """Agent for BabyAI PutNext tasks.

    State machine phases:
    - EXPLORE: find source and target objects via frontier exploration
    - GOTO_SOURCE: BFS navigate to cell adjacent to source object
    - PICKUP: face source object and pickup
    - GOTO_TARGET: BFS navigate to cell adjacent to target object
    - DROP: face empty cell adjacent to target and drop
    """

    def __init__(self, grid_width: int, grid_height: int,
                 mission: str, epsilon: float = 0.0):
        self.spatial_map = SpatialMap(grid_width, grid_height)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.epsilon = epsilon

        parser = MissionParser()
        self.source, self.target = parser.parse(mission)
        self.phase = "EXPLORE"
        self._carrying = False
        self._carrying_type_color: tuple[int, int] | None = None

    def reset(self, mission: str) -> None:
        self.spatial_map.reset()
        parser = MissionParser()
        self.source, self.target = parser.parse(mission)
        self.phase = "EXPLORE"
        self._carrying = False
        self._carrying_type_color = None

    def select_action(self, obs_7x7: np.ndarray,
                      agent_col: int, agent_row: int, agent_dir: int) -> int:
        """Main action selection."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        # MiniGrid shows carried object at agent's cell — clear it
        if self._carrying:
            self.spatial_map.grid[agent_row, agent_col, 0] = OBJ_EMPTY
            self.spatial_map.grid[agent_row, agent_col, 1] = 0
            self.spatial_map.grid[agent_row, agent_col, 2] = 0
        self._update_phase()

        # Epsilon exploration
        if self.epsilon > 0 and np.random.random() < self.epsilon:
            return int(np.random.randint(0, 3))

        if self.phase == "EXPLORE":
            return self._act_explore(agent_row, agent_col, agent_dir)
        elif self.phase == "GOTO_SOURCE":
            return self._act_goto_object(
                self.source, agent_row, agent_col, agent_dir
            )
        elif self.phase == "PICKUP":
            return self._act_pickup(
                self.source, agent_row, agent_col, agent_dir, obs_7x7
            )
        elif self.phase == "GOTO_TARGET":
            return self._act_goto_drop(
                self.target, agent_row, agent_col, agent_dir
            )
        elif self.phase == "DROP":
            return self._act_drop(
                self.target, agent_row, agent_col, agent_dir, obs_7x7
            )

        return self._act_explore(agent_row, agent_col, agent_dir)

    def _update_phase(self) -> None:
        """Update state machine phase based on current state."""
        src_pos = self.spatial_map.find_object_by_type_color(*self.source)
        tgt_pos = self.spatial_map.find_object_by_type_color(*self.target)

        if self.phase == "EXPLORE":
            if self._carrying and tgt_pos is not None:
                self.phase = "GOTO_TARGET"
            elif self._carrying:
                pass  # keep exploring to find target
            elif src_pos is not None and tgt_pos is not None:
                self.phase = "GOTO_SOURCE"
            elif src_pos is not None and tgt_pos is None:
                # Source found but not target - still go get source first
                self.phase = "GOTO_SOURCE"

        elif self.phase == "GOTO_SOURCE":
            if self._carrying:
                self.phase = "GOTO_TARGET" if tgt_pos is not None else "EXPLORE"

        elif self.phase == "PICKUP":
            if self._carrying:
                self.phase = "GOTO_TARGET" if tgt_pos is not None else "EXPLORE"

        elif self.phase == "GOTO_TARGET":
            pass  # transitions handled in _act_goto_drop

    def _act_explore(self, agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Frontier exploration."""
        return self.explorer.select_action(
            self.spatial_map, agent_row, agent_col, agent_dir
        )

    def _act_goto_object(self, obj_spec: tuple[int, int],
                         agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Navigate to cell adjacent to object."""
        pos = self.spatial_map.find_object_by_type_color(*obj_spec)
        if pos is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Check if already adjacent
        dr = abs(agent_row - pos[0])
        dc = abs(agent_col - pos[1])
        if dr + dc == 1:
            # Adjacent — switch to pickup
            self.phase = "PICKUP"
            return self._act_pickup(obj_spec, agent_row, agent_col, agent_dir, None)

        # Navigate to best adjacent cell
        adj = self._find_adjacent_walkable(pos, agent_row, agent_col)
        if adj is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        return self._navigate_to(adj[0], adj[1], agent_row, agent_col, agent_dir)

    def _act_pickup(self, obj_spec: tuple[int, int],
                    agent_row: int, agent_col: int, agent_dir: int,
                    obs_7x7: np.ndarray | None) -> int:
        """Face source object and pickup."""
        pos = self.spatial_map.find_object_by_type_color(*obj_spec)
        if pos is None:
            self.phase = "EXPLORE"
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Check if facing the object
        facing_row = agent_row + DIR_DR[agent_dir]
        facing_col = agent_col + DIR_DC[agent_dir]
        if (facing_row, facing_col) == pos:
            return ACT_PICKUP

        # Need to turn to face the object
        return self._turn_toward(pos[0], pos[1], agent_row, agent_col, agent_dir)

    def _act_goto_drop(self, obj_spec: tuple[int, int],
                       agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Navigate to position where we can drop next to target.

        Strategy: find empty cells adjacent to target (valid drop spots).
        For each drop spot, find a stand cell adjacent to it (where agent stands,
        facing the drop spot, then drops). Navigate to the best stand cell.
        """
        tgt_pos = self.spatial_map.find_object_by_type_color(*obj_spec)
        if tgt_pos is None:
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Find all valid (stand_cell, drop_cell, face_dir) triples
        plan = self._find_drop_plan(tgt_pos, agent_row, agent_col)
        if plan is None:
            # No valid plan — explore more
            return self._act_explore(agent_row, agent_col, agent_dir)

        stand_cell, drop_cell, face_dir = plan

        # If already at stand cell
        if (agent_row, agent_col) == stand_cell:
            self.phase = "DROP"
            # Face the drop cell
            if agent_dir == face_dir:
                return ACT_DROP
            return self._turn_toward_dir(face_dir, agent_dir)

        return self._navigate_to(stand_cell[0], stand_cell[1],
                                 agent_row, agent_col, agent_dir)

    def _act_drop(self, obj_spec: tuple[int, int],
                  agent_row: int, agent_col: int, agent_dir: int,
                  obs_7x7: np.ndarray | None) -> int:
        """Drop carried object facing an empty cell adjacent to target."""
        tgt_pos = self.spatial_map.find_object_by_type_color(*obj_spec)
        if tgt_pos is None:
            self.phase = "EXPLORE"
            return self._act_explore(agent_row, agent_col, agent_dir)

        # Check if current facing direction drops into a valid cell
        fr = agent_row + DIR_DR[agent_dir]
        fc = agent_col + DIR_DC[agent_dir]
        if (0 <= fr < self.spatial_map.height and
                0 <= fc < self.spatial_map.width):
            cell = int(self.spatial_map.grid[fr, fc, 0])
            is_empty = cell in (OBJ_EMPTY, SpatialMap.UNKNOWN)
            adj_to_target = abs(fr - tgt_pos[0]) + abs(fc - tgt_pos[1]) == 1
            not_target = (fr, fc) != tgt_pos
            if is_empty and adj_to_target and not_target:
                return ACT_DROP

        # Find correct direction to face
        plan = self._find_drop_plan(tgt_pos, agent_row, agent_col)
        if plan is not None:
            stand_cell, drop_cell, face_dir = plan
            if (agent_row, agent_col) == stand_cell:
                if agent_dir == face_dir:
                    return ACT_DROP
                return self._turn_toward_dir(face_dir, agent_dir)
            # Need to move to stand cell
            self.phase = "GOTO_TARGET"
            return self._navigate_to(stand_cell[0], stand_cell[1],
                                     agent_row, agent_col, agent_dir)

        # Fallback: go back to GOTO_TARGET to reposition
        self.phase = "GOTO_TARGET"
        return self._act_explore(agent_row, agent_col, agent_dir)

    def _find_drop_plan(self, target_pos: tuple[int, int],
                        agent_row: int, agent_col: int
                        ) -> tuple[tuple[int, int], tuple[int, int], int] | None:
        """Find (stand_cell, drop_cell, face_dir) for dropping next to target.

        The dropped object lands in the cell the agent is facing (front cell).
        That cell must be:
        - Empty (or agent's current position — agent will move away)
        - Adjacent to target (manhattan dist 1)
        - Not the target itself

        Agent stands at stand_cell, faces face_dir, drops into drop_cell.
        """
        tr, tc = target_pos
        obs = self._pathfinding_obs()
        h, w = self.spatial_map.height, self.spatial_map.width

        # Step 1: find all empty cells adjacent to target (valid drop spots)
        # Include agent's current position (type 10) since agent will move to stand cell
        OBJ_AGENT = 10
        drop_spots: list[tuple[int, int]] = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            er, ec = tr + dr, tc + dc
            if not (0 <= er < h and 0 <= ec < w):
                continue
            cell = int(self.spatial_map.grid[er, ec, 0])
            if cell in (OBJ_EMPTY, SpatialMap.UNKNOWN, OBJ_AGENT):
                drop_spots.append((er, ec))

        if not drop_spots:
            return None

        # Step 2: for each drop spot, find stand cells where agent faces the drop spot
        best: tuple[tuple[int, int], tuple[int, int], int, float] | None = None

        for drop_r, drop_c in drop_spots:
            # Agent must face drop spot. For each direction, compute stand cell.
            for face_dir in range(4):
                # Agent at stand, facing face_dir, front = stand + dir_delta = drop
                # So stand = drop - dir_delta
                sr = drop_r - DIR_DR[face_dir]
                sc = drop_c - DIR_DC[face_dir]
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                # Stand cell must be walkable
                stand_cell_type = int(obs[sr, sc, 0])
                if stand_cell_type in (OBJ_WALL,):
                    continue
                # Stand cell must not be the target or the drop spot
                if (sr, sc) == target_pos:
                    continue

                # BFS distance to stand cell
                path = self.pathfinder.find_path(
                    obs, (agent_row, agent_col), (sr, sc), allow_door=True
                )
                dist = len(path) if path is not None else float('inf')
                if dist == float('inf'):
                    continue

                if best is None or dist < best[3]:
                    best = ((sr, sc), (drop_r, drop_c), face_dir, dist)

        if best is None:
            return None
        return best[0], best[1], best[2]

    def _find_adjacent_walkable(self, pos: tuple[int, int],
                                agent_row: int, agent_col: int) -> tuple[int, int] | None:
        """Find walkable cell adjacent to pos, closest to agent."""
        obs = self._pathfinding_obs()
        best = None
        best_dist = float('inf')

        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nr, nc = pos[0] + dr, pos[1] + dc
            if not (0 <= nr < self.spatial_map.height and 0 <= nc < self.spatial_map.width):
                continue
            cell = int(obs[nr, nc, 0])
            if cell in (OBJ_WALL, OBJ_KEY, OBJ_BALL, OBJ_BOX):
                continue
            path = self.pathfinder.find_path(
                obs, (agent_row, agent_col), (nr, nc), allow_door=True
            )
            if path is not None and len(path) < best_dist:
                best_dist = len(path)
                best = (nr, nc)

        return best

    def _navigate_to(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """BFS navigate to target position."""
        if agent_row == target_row and agent_col == target_col:
            return int(np.random.randint(0, 3))

        obs = self._pathfinding_obs()
        path = self.pathfinder.find_path(
            obs, (agent_row, agent_col), (target_row, target_col),
            allow_door=True
        )
        if path is None or len(path) <= 1:
            return self.explorer.select_action(
                self.spatial_map, agent_row, agent_col, agent_dir
            )

        actions = self.pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return ACT_FORWARD

    def _pathfinding_obs(self) -> np.ndarray:
        """Create observation for BFS. Objects are walls for pathfinding."""
        obs = self.spatial_map.to_obs()
        for obj_type in (OBJ_KEY, OBJ_BALL, OBJ_BOX):
            mask = obs[:, :, 0] == obj_type
            obs[mask, 0] = OBJ_WALL
        # Agent cell (type 10) should be walkable
        agent_mask = obs[:, :, 0] == 10
        obs[agent_mask, 0] = OBJ_EMPTY
        return obs

    def _turn_toward(self, target_row: int, target_col: int,
                     agent_row: int, agent_col: int, agent_dir: int) -> int:
        """Return action to turn agent toward target position."""
        dr = target_row - agent_row
        dc = target_col - agent_col
        need_dir = self._dir_from_delta(dr, dc)
        if need_dir is None or need_dir == agent_dir:
            return ACT_FORWARD  # already facing or can't determine
        return self._turn_toward_dir(need_dir, agent_dir)

    @staticmethod
    def _turn_toward_dir(need_dir: int, agent_dir: int) -> int:
        """Return turn action to rotate from agent_dir toward need_dir."""
        diff = (need_dir - agent_dir) % 4
        return ACT_RIGHT if diff <= 2 else ACT_LEFT

    @staticmethod
    def _dir_from_delta(dr: int, dc: int) -> int | None:
        if dc > 0:
            return 0  # right
        if dr > 0:
            return 1  # down
        if dc < 0:
            return 2  # left
        if dr < 0:
            return 3  # up
        return None

    def observe_result(self, obs_7x7: np.ndarray,
                       agent_col: int, agent_row: int, agent_dir: int,
                       reward: float) -> None:
        """Update spatial map after action."""
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        # MiniGrid shows carried object at agent's cell — clear it
        if self._carrying:
            self.spatial_map.grid[agent_row, agent_col, 0] = OBJ_EMPTY
            self.spatial_map.grid[agent_row, agent_col, 1] = 0
            self.spatial_map.grid[agent_row, agent_col, 2] = 0

    def update_carrying(self, carrying_type_color: tuple[int, int] | None) -> None:
        """Update carrying state from env. Call before select_action."""
        was_carrying = self._carrying
        self._carrying = carrying_type_color is not None
        self._carrying_type_color = carrying_type_color

        # If just picked up source, remove it from map
        if self._carrying and not was_carrying:
            src_pos = self.spatial_map.find_object_by_type_color(*self.source)
            if src_pos is not None:
                self.spatial_map.grid[src_pos[0], src_pos[1], 0] = OBJ_EMPTY
                self.spatial_map.grid[src_pos[0], src_pos[1], 1] = 0
                self.spatial_map.grid[src_pos[0], src_pos[1], 2] = 0


class PutNextEnv:
    """Wrapper for BabyAI PutNext environments.

    Provides agent position, carrying state, and mission text.
    """

    def __init__(self, env_name: str = 'BabyAI-PutNextS6N3-v0'):
        import minigrid
        minigrid.register_minigrid_envs()
        import gymnasium as gym
        self._env = gym.make(env_name)
        self._mission: str = ""
        self.grid_width: int = 0
        self.grid_height: int = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, int, int, int, str]:
        """Reset → (obs_7x7, agent_col, agent_row, agent_dir, mission)."""
        obs, info = self._env.reset(seed=seed)
        uw = self._env.unwrapped
        self._mission = obs["mission"]
        self.grid_width = uw.grid.width
        self.grid_height = uw.grid.height
        pos = uw.agent_pos
        return (obs["image"], int(pos[0]), int(pos[1]), int(uw.agent_dir),
                self._mission)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, int, int, int]:
        """Step → (obs_7x7, reward, term, trunc, agent_col, agent_row, agent_dir)."""
        obs, reward, term, trunc, info = self._env.step(action)
        uw = self._env.unwrapped
        pos = uw.agent_pos
        return (obs["image"], float(reward), term, trunc,
                int(pos[0]), int(pos[1]), int(uw.agent_dir))

    @property
    def carrying_type_color(self) -> tuple[int, int] | None:
        """Return (type_id, color_id) of carried object, or None."""
        uw = self._env.unwrapped
        obj = uw.carrying
        if obj is None:
            return None
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        return (OBJECT_TO_IDX[obj.type], COLOR_TO_IDX[obj.color])

    @property
    def carrying(self):
        return self._env.unwrapped.carrying

    def close(self):
        self._env.close()

    @property
    def unwrapped(self):
        return self._env.unwrapped
