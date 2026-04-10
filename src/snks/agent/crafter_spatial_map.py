"""Stage 68: CrafterSpatialMap — cognitive map from NearDetector observations.

Accumulates (player_pos → near_str) as agent explores.
Replaces info["semantic"] for object-finding navigation.

Analogy: hippocampal place cells — each visited position stores
what was perceived (via NearDetector) when the agent was there.

Proprioception kept (not replaced):
- info["player_pos"]  — agent knows where its body is
- info["inventory"]   — agent knows what it carries
"""

from __future__ import annotations

import numpy as np
import torch

MOVE_ACTIONS = ["move_left", "move_right", "move_up", "move_down"]


class CrafterSpatialMap:
    """Cognitive map: visited (y, x) → near_str observed by NearDetector.

    Built incrementally as agent explores. Unknown cells = never visited.

    Stage 77a: tracks `blocked` tiles — positions that failed to accept
    agent movement. Updated from observation (agent attempts move_X at
    position P, position doesn't change → the target tile P+dir is
    blocked). This is observation-based world modeling, not a hardcoded
    stuck-avoidance rule in policy code. See IDEOLOGY Stage 73.
    """

    def __init__(self, world_size: int = 64) -> None:
        self.world_size = world_size
        # (y, x) → near_str
        self._map: dict[tuple[int, int], str] = {}
        self._visited: set[tuple[int, int]] = set()
        # Tiles observed to reject movement (walls, water edges, trees).
        self._blocked: set[tuple[int, int]] = set()

    def mark_blocked(self, pos: tuple[int, int]) -> None:
        """Record that `pos` is impassable (learned from failed movement)."""
        self._blocked.add((int(pos[0]), int(pos[1])))

    def is_blocked(self, pos: tuple[int, int]) -> bool:
        return (int(pos[0]), int(pos[1])) in self._blocked

    def update(self, player_pos: tuple[int, int], near_str: str) -> None:
        """Record NearDetector output at current position.

        Args:
            player_pos: (y, x) from info["player_pos"].
            near_str: output of NearDetector.detect(pixels).
        """
        y, x = int(player_pos[0]), int(player_pos[1])
        self._map[(y, x)] = near_str
        self._visited.add((y, x))

    def find_nearest(
        self, target: str, player_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Find nearest known position where target was observed.

        Returns (y, x) of nearest known position, or None if not in map.
        """
        py, px = int(player_pos[0]), int(player_pos[1])
        best_pos: tuple[int, int] | None = None
        best_dist = float("inf")
        for (y, x), near in self._map.items():
            if near == target:
                d = abs(y - py) + abs(x - px)
                if d < best_dist:
                    best_dist = d
                    best_pos = (y, x)
        return best_pos

    def unvisited_neighbors(
        self, player_pos: tuple[int, int], radius: int = 5
    ) -> list[tuple[int, int]]:
        """Find unvisited non-blocked positions within radius for exploration.

        Stage 77a: filters out `self._blocked` tiles so exploration doesn't
        repeatedly pick the same impassable cell (observation-based stuck
        avoidance, not a hardcoded if-else).
        """
        py, px = int(player_pos[0]), int(player_pos[1])
        result = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                    if (ny, nx) in self._visited:
                        continue
                    if (ny, nx) in self._blocked:
                        continue
                    result.append((ny, nx))
        return result

    def reset(self) -> None:
        """Clear for new episode."""
        self._map.clear()
        self._visited.clear()
        self._blocked.clear()

    @property
    def n_visited(self) -> int:
        return len(self._visited)

    @property
    def known_objects(self) -> dict[str, int]:
        """Count of known positions per near_str (excluding empty)."""
        counts: dict[str, int] = {}
        for near in self._map.values():
            if near != "empty":
                counts[near] = counts.get(near, 0) + 1
        return counts


def _step_toward(
    current: tuple[int, int],
    target: tuple[int, int],
    rng: np.random.RandomState,
) -> str:
    """One greedy step toward target, random tie-break.

    Crafter coordinate convention:
      pos[0] = horizontal axis (X, move_left/right)
      pos[1] = vertical axis   (Y, move_up/down)
    Verified: move_right → pos[0]+=1, move_down → pos[1]+=1.
    """
    cx, cy = int(current[0]), int(current[1])
    tx, ty = int(target[0]), int(target[1])
    dx, dy = tx - cx, ty - cy

    moves = []
    if dx > 0:
        moves.append("move_right")
    elif dx < 0:
        moves.append("move_left")
    if dy > 0:
        moves.append("move_down")
    elif dy < 0:
        moves.append("move_up")

    if not moves:
        return str(rng.choice(MOVE_ACTIONS))
    return str(rng.choice(moves))


def find_target_with_map(
    env: object,
    detector: object,
    spatial_map: CrafterSpatialMap,
    target: str,
    max_steps: int = 300,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, dict, bool]:
    """Navigate to target object using spatial map + NearDetector.

    Does NOT use info["semantic"]. Uses only:
    - pixels → NearDetector (perception)
    - info["player_pos"] (proprioception)
    - CrafterSpatialMap (memory)

    Args:
        env: CrafterPixelEnv instance (already reset).
        detector: NearDetector instance.
        spatial_map: CrafterSpatialMap (reset per episode externally or here on done).
        target: near_str to find (e.g. "tree").
        max_steps: max steps before giving up.
        rng: random state; created fresh if None.

    Returns:
        (pixels_tensor, info, found)
    """
    if rng is None:
        rng = np.random.RandomState()

    # type: ignore[attr-defined]
    pixels, info = env.observe()  # type: ignore[union-attr]

    for _ in range(max_steps):
        pix_tensor = torch.from_numpy(pixels)
        near_str = detector.detect(pix_tensor)  # type: ignore[union-attr]
        player_pos = info["player_pos"]
        spatial_map.update(player_pos, near_str)

        if near_str == target:
            return pix_tensor, info, True

        # Navigate: known position first, then explore
        known_pos = spatial_map.find_nearest(target, player_pos)
        if known_pos is not None:
            action = _step_toward(player_pos, known_pos, rng)
        else:
            unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
            if unvisited:
                goal = unvisited[int(rng.randint(len(unvisited)))]
                action = _step_toward(player_pos, goal, rng)
            else:
                action = str(rng.choice(MOVE_ACTIONS))

        pixels, _, done, info = env.step(action)  # type: ignore[union-attr]
        if done:
            pixels, info = env.reset()  # type: ignore[union-attr]
            spatial_map.reset()

    return torch.from_numpy(pixels), info, False
