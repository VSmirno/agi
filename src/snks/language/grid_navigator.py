"""GridNavigator: BFS pathfinding in MiniGrid (Stage 24c).

Scaffold navigation — reads full grid state (bypasses partial observability).
Converts BFS path to MiniGrid action sequences (turn + forward).
"""

from __future__ import annotations

from collections import deque

from minigrid.core.constants import OBJECT_TO_IDX


# Direction vectors: 0=right, 1=down, 2=left, 3=up.
DIR_VEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

# MiniGrid actions.
ACT_LEFT = 0
ACT_RIGHT = 1
ACT_FORWARD = 2


def _is_passable(grid, x: int, y: int) -> bool:
    """Check if cell (x, y) is passable (empty, floor, goal, or open door)."""
    if x < 0 or y < 0 or x >= grid.width or y >= grid.height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True  # empty cell
    if cell.type in ("empty", "floor", "goal"):
        return True
    if cell.type == "door" and cell.is_open:
        return True
    return False


def _bfs(grid, start: tuple[int, int], target: tuple[int, int]) -> list[tuple[int, int]] | None:
    """BFS shortest path from start to target, returns list of positions."""
    if start == target:
        return [start]

    queue: deque[tuple[int, int]] = deque([start])
    visited: set[tuple[int, int]] = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        cx, cy = queue.popleft()
        for dx, dy in DIR_VEC.values():
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            # Target cell: always reachable (we want to be adjacent or on it).
            if (nx, ny) == target or _is_passable(grid, nx, ny):
                visited.add((nx, ny))
                parent[(nx, ny)] = (cx, cy)
                if (nx, ny) == target:
                    # Reconstruct path.
                    path = [(nx, ny)]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    return list(reversed(path))
                queue.append((nx, ny))

    return None  # no path found


def _direction_to(src: tuple[int, int], dst: tuple[int, int]) -> int:
    """Get direction from src to dst (adjacent cells)."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    for d, (vx, vy) in DIR_VEC.items():
        if (vx, vy) == (dx, dy):
            return d
    return 0  # fallback


def _turn_actions(current_dir: int, target_dir: int) -> list[int]:
    """Minimal turn actions to go from current_dir to target_dir."""
    if current_dir == target_dir:
        return []

    # Right turn = +1 mod 4, Left turn = -1 mod 4.
    right_turns = (target_dir - current_dir) % 4
    left_turns = (current_dir - target_dir) % 4

    if right_turns <= left_turns:
        return [ACT_RIGHT] * right_turns
    else:
        return [ACT_LEFT] * left_turns


class GridNavigator:
    """BFS shortest-path navigation in MiniGrid grid.

    Produces a sequence of MiniGrid actions (left/right/forward) to reach
    a target position from the agent's current position and direction.
    """

    def plan_path(
        self,
        grid,
        agent_pos: tuple[int, int],
        agent_dir: int,
        target_pos: tuple[int, int],
        stop_adjacent: bool = False,
    ) -> list[int]:
        """Plan action sequence to reach target_pos.

        Args:
            grid: MiniGrid Grid object.
            agent_pos: (x, y) current agent position.
            agent_dir: current agent direction.
            target_pos: (x, y) target position.
            stop_adjacent: if True, stop when adjacent to target (for pickup/toggle).

        Returns:
            List of MiniGrid action IDs. Empty if no path or already there.
        """
        if agent_pos == target_pos:
            return []

        path = _bfs(grid, agent_pos, target_pos)
        if path is None:
            return []

        # If stop_adjacent, don't step onto the target cell — stop one before.
        if stop_adjacent and len(path) >= 2:
            path = path[:-1]
            if len(path) == 1:
                # Already adjacent, just face the target.
                target_dir = _direction_to(agent_pos, target_pos)
                return _turn_actions(agent_dir, target_dir)

        # Convert path to actions.
        actions: list[int] = []
        current_dir = agent_dir

        for i in range(len(path) - 1):
            needed_dir = _direction_to(path[i], path[i + 1])
            actions.extend(_turn_actions(current_dir, needed_dir))
            actions.append(ACT_FORWARD)
            current_dir = needed_dir

        # If stop_adjacent, face the target after arriving.
        if stop_adjacent and path:
            final_pos = path[-1]
            target_dir = _direction_to(final_pos, target_pos)
            actions.extend(_turn_actions(current_dir, target_dir))

        return actions
