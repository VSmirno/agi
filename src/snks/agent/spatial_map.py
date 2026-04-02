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

    def find_object_by_type_color(self, type_id: int, color_id: int) -> tuple[int, int] | None:
        """Find position of object matching both type and color.

        Returns (row, col) or None.
        """
        type_match = self.grid[:, :, 0] == type_id
        color_match = self.grid[:, :, 1] == color_id
        mask = type_match & color_match
        positions = np.argwhere(mask)
        if len(positions) > 0:
            return int(positions[0, 0]), int(positions[0, 1])
        return None

    def find_all_objects(self) -> list[tuple[int, int, int, int]]:
        """Find all non-wall, non-empty objects.

        Returns list of (type_id, color_id, row, col).
        """
        result: list[tuple[int, int, int, int]] = []
        for r in range(self.height):
            for c in range(self.width):
                if not self.explored[r, c]:
                    continue
                t = int(self.grid[r, c, 0])
                if t in (OBJ_UNSEEN, 1, OBJ_WALL, self.UNKNOWN):  # 1=empty
                    continue
                color = int(self.grid[r, c, 1])
                result.append((t, color, r, c))
        return result

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
                if obj == OBJ_WALL:
                    continue
                # Doors are valid frontiers — they lead to unexplored rooms
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
    """Navigate to nearest frontier cell using BFS.

    Uses a single BFS from agent position, expanding outward until
    hitting a frontier cell. O(n) instead of O(frontiers * n).
    """

    def __init__(self):
        self._pathfinder = GridPathfinder()

    def nearest_frontier(self, spatial_map: SpatialMap,
                         agent_row: int, agent_col: int) -> tuple[int, int] | None:
        """Single BFS from agent position to nearest frontier."""
        frontier_set = set(spatial_map.frontiers())
        if not frontier_set:
            return None

        obs = spatial_map.to_obs()
        walls = self._pathfinder.extract_walls(obs, allow_door=True)
        h, w = spatial_map.height, spatial_map.width

        queue = deque([(agent_row, agent_col)])
        visited = {(agent_row, agent_col)}

        while queue:
            r, c = queue.popleft()
            if (r, c) in frontier_set and (r, c) != (agent_row, agent_col):
                return (r, c)
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                    if (nr, nc) not in walls:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        return None

    def select_action(self, spatial_map: SpatialMap,
                      agent_row: int, agent_col: int, agent_dir: int) -> int:
        """BFS to nearest frontier, return first action.

        If frontier is a door, navigates to adjacent cell (can't step on door).
        """
        target = self.nearest_frontier(spatial_map, agent_row, agent_col)
        if target is None:
            # No frontiers — random turn to reveal new areas
            return int(np.random.randint(0, 3))

        # If target is a door, navigate to an adjacent non-wall cell
        obs = spatial_map.to_obs()
        actual_target = target
        if int(obs[target[0], target[1], 0]) == OBJ_DOOR:
            # Find adjacent walkable cell closest to agent
            best_adj = None
            best_dist = float('inf')
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ar, ac = target[0] + dr, target[1] + dc
                if 0 <= ar < spatial_map.height and 0 <= ac < spatial_map.width:
                    obj = int(obs[ar, ac, 0])
                    if obj not in (OBJ_WALL, OBJ_DOOR):
                        d = abs(ar - agent_row) + abs(ac - agent_col)
                        if d < best_dist:
                            best_dist = d
                            best_adj = (ar, ac)
            if best_adj is not None:
                actual_target = best_adj

        if actual_target == (agent_row, agent_col):
            # Already at target — turn randomly to trigger door toggle
            return int(np.random.randint(0, 2))  # left or right

        path = self._pathfinder.find_path(
            obs, (agent_row, agent_col), actual_target, allow_door=True
        )
        if path is None or len(path) <= 1:
            return int(np.random.randint(0, 3))

        actions = self._pathfinder.path_to_actions(path, agent_dir)
        if actions:
            return actions[0]
        return 2  # forward
