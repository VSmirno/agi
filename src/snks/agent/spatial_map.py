"""Stage 54: SpatialMap + FrontierExplorer for partial observability.

SpatialMap accumulates 7x7 partial observations into a full grid map.
FrontierExplorer navigates toward unexplored cells using BFS.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from snks.agent.pathfinding import GridPathfinder


# MiniGrid object types
OBJ_UNSEEN = 0
OBJ_EMPTY = 1
OBJ_WALL = 2
OBJ_DOOR = 4
OBJ_KEY = 5
OBJ_GOAL = 8


def view_to_world(obs_r: int, obs_c: int,
                  agent_col: int, agent_row: int,
                  agent_dir: int) -> tuple[int, int]:
    """Convert 7x7 partial obs coordinates to world (col, row).

    MiniGrid partial obs: 7x7 grid, agent at obs[3, 6] (bottom-center),
    always facing "up" in view frame. Real direction rotates the mapping.

    Returns (world_col, world_row).
    """
    if agent_dir == 0:  # right
        wc = agent_col + (6 - obs_c)
        wr = agent_row + (obs_r - 3)
    elif agent_dir == 1:  # down
        wc = agent_col - (obs_r - 3)
        wr = agent_row + (6 - obs_c)
    elif agent_dir == 2:  # left
        wc = agent_col - (6 - obs_c)
        wr = agent_row - (obs_r - 3)
    else:  # dir == 3, up
        wc = agent_col + (obs_r - 3)
        wr = agent_row - (6 - obs_c)
    return wc, wr


class SpatialMap:
    """2D grid map accumulated from partial 7x7 observations.

    Stores object type, color, and state for each cell.
    Unknown cells marked as -1.
    """

    UNKNOWN = -1

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # 3-channel grid: obj_type, color, state
        self.grid = np.full((height, width, 3), self.UNKNOWN, dtype=np.int8)
        self.explored = np.zeros((height, width), dtype=bool)

    def update(self, obs_7x7: np.ndarray,
               agent_col: int, agent_row: int, agent_dir: int) -> None:
        """Project 7x7 egocentric observation onto the spatial map."""
        for r in range(7):
            for c in range(7):
                obj_type = int(obs_7x7[r, c, 0])
                if obj_type == OBJ_UNSEEN:
                    continue

                wc, wr = view_to_world(r, c, agent_col, agent_row, agent_dir)
                if 0 <= wr < self.height and 0 <= wc < self.width:
                    self.grid[wr, wc, 0] = int(obs_7x7[r, c, 0])
                    self.grid[wr, wc, 1] = int(obs_7x7[r, c, 1])
                    self.grid[wr, wc, 2] = int(obs_7x7[r, c, 2])
                    self.explored[wr, wc] = True

    def to_obs(self) -> np.ndarray:
        """Convert to full-grid observation for BFS pathfinding.

        Unknown cells treated as empty (optimistic).
        Returns (height, width, 3) array with standard MiniGrid encoding.
        """
        obs = self.grid.copy()
        unknown_mask = obs[:, :, 0] == self.UNKNOWN
        obs[unknown_mask, 0] = OBJ_EMPTY
        obs[unknown_mask, 1] = 0
        obs[unknown_mask, 2] = 0
        return obs

    def find_object(self, obj_type: int) -> tuple[int, int] | None:
        """Find first known position of object type. Returns (row, col) or None."""
        mask = self.grid[:, :, 0] == obj_type
        positions = np.argwhere(mask)
        if len(positions) > 0:
            return int(positions[0, 0]), int(positions[0, 1])
        return None

    def find_objects(self) -> dict:
        """Find known positions of key, door, goal.

        Returns dict with key_pos, door_pos, goal_pos as (row, col) or None.
        """
        return {
            "key_pos": self.find_object(OBJ_KEY),
            "door_pos": self.find_object(OBJ_DOOR),
            "goal_pos": self.find_object(OBJ_GOAL),
        }

    def frontiers(self) -> list[tuple[int, int]]:
        """Find frontier cells: explored empty/floor cells adjacent to unexplored.

        Returns list of (row, col) frontier positions.
        """
        result: list[tuple[int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if not self.explored[r, c]:
                    continue
                obj = int(self.grid[r, c, 0])
                if obj in (OBJ_WALL, OBJ_DOOR):
                    continue
                # Check if adjacent to unexplored
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if not self.explored[nr, nc]:
                            result.append((r, c))
                            break
        return result

    def reset(self) -> None:
        """Clear map for new episode."""
        self.grid[:] = self.UNKNOWN
        self.explored[:] = False


class FrontierExplorer:
    """Navigate to nearest frontier cell using BFS."""

    def __init__(self):
        self._pathfinder = GridPathfinder()

    def nearest_frontier(self, spatial_map: SpatialMap,
                         agent_row: int, agent_col: int) -> tuple[int, int] | None:
        """BFS from agent to nearest frontier cell."""
        fronts = spatial_map.frontiers()
        if not fronts:
            return None

        obs = spatial_map.to_obs()
        best_path = None
        best_target = None

        for fr, fc in fronts:
            path = self._pathfinder.find_path(
                obs, (agent_row, agent_col), (fr, fc), allow_door=True
            )
            if path is not None:
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_target = (fr, fc)

        return best_target

    def select_action(self, spatial_map: SpatialMap,
                      agent_row: int, agent_col: int, agent_dir: int) -> int:
        """BFS to nearest frontier, return first action."""
        target = self.nearest_frontier(spatial_map, agent_row, agent_col)
        if target is None:
            return 2  # forward (fallback)

        obs = spatial_map.to_obs()
        path = self._pathfinder.find_path(
            obs, (agent_row, agent_col), target, allow_door=True
        )
        if path is None or len(path) <= 1:
            return 2  # forward

        actions = self._pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return 2  # forward
