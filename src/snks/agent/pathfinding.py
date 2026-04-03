"""Stage 47: BFS pathfinding on MiniGrid observation grids.

Extracts wall positions from 7x7x3 observations and finds shortest paths
using breadth-first search. Converts paths to MiniGrid action sequences.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class GridPathfinder:
    """BFS pathfinding on MiniGrid observation grid."""

    OBJ_WALL = 2
    OBJ_DOOR = 4
    OBJ_KEY = 5
    OBJ_BALL = 6
    OBJ_BOX = 7

    # Object types that block movement in MiniGrid
    SOLID_OBJECTS = {OBJ_KEY, OBJ_BALL, OBJ_BOX}

    def extract_walls(self, obs: np.ndarray,
                      allow_door: bool = False,
                      allow_objects: bool = False) -> set[tuple[int, int]]:
        """Extract impassable cell positions from observation.

        Args:
            obs: 7x7x3 MiniGrid observation (channel 0 = obj type, channel 2 = state)
            allow_door: if True, locked doors are treated as passable
            allow_objects: if True, keys/balls/boxes are treated as passable
                           (useful for navigating TO an object for pickup)
        """
        walls: set[tuple[int, int]] = set()
        h, w = obs.shape[0], obs.shape[1]
        for r in range(h):
            for c in range(w):
                obj_type = int(obs[r, c, 0])
                if obj_type == self.OBJ_WALL:
                    walls.add((r, c))
                elif obj_type == self.OBJ_DOOR:
                    door_state = int(obs[r, c, 2])
                    if door_state == 2 and not allow_door:
                        walls.add((r, c))
                elif obj_type in self.SOLID_OBJECTS and not allow_objects:
                    walls.add((r, c))
        return walls

    def find_path(self, obs: np.ndarray,
                  start: tuple[int, int], goal: tuple[int, int],
                  allow_door: bool = False,
                  allow_objects: bool = False) -> list[tuple[int, int]] | None:
        """BFS shortest path from start to goal, avoiding walls.

        Returns list of (row, col) positions including start and goal,
        or None if no path exists.
        """
        if start == goal:
            return [start]

        walls = self.extract_walls(obs, allow_door=allow_door,
                                   allow_objects=allow_objects)
        h, w = obs.shape[0], obs.shape[1]

        queue: deque[tuple[int, int]] = deque([start])
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                # Reconstruct path
                path: list[tuple[int, int]] = []
                node: tuple[int, int] | None = goal
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in walls and (nr, nc) not in came_from:
                    came_from[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        return None  # no path

    def path_to_actions(self, path: list[tuple[int, int]],
                        current_dir: int) -> list[int]:
        """Convert a position path to MiniGrid actions.

        Direction mapping: 0=right(+col), 1=down(+row), 2=left(-col), 3=up(-row)
        Actions: 0=turn_left, 1=turn_right, 2=forward

        Returns list of actions to follow the path.
        """
        if len(path) <= 1:
            return []

        actions: list[int] = []
        direction = current_dir

        for i in range(len(path) - 1):
            r, c = path[i]
            nr, nc = path[i + 1]
            dr, dc = nr - r, nc - c

            # Determine needed direction
            if dc > 0:
                need_dir = 0  # right
            elif dr > 0:
                need_dir = 1  # down
            elif dc < 0:
                need_dir = 2  # left
            else:
                need_dir = 3  # up

            # Turn to face needed direction
            while direction != need_dir:
                diff = (need_dir - direction) % 4
                if diff <= 2:
                    actions.append(1)  # turn_right
                    direction = (direction + 1) % 4
                else:
                    actions.append(0)  # turn_left
                    direction = (direction - 1) % 4

            actions.append(2)  # forward

        return actions
