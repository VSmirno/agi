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
        # (y, x) → (near_str, confidence, observation_count)
        self._map: dict[tuple[int, int], tuple[str, float, int]] = {}
        self._visited: set[tuple[int, int]] = set()
        # Tiles observed to reject movement (walls, water edges, trees).
        self._blocked: set[tuple[int, int]] = set()

    def mark_blocked(self, pos: tuple[int, int]) -> None:
        """Record that `pos` is impassable (learned from failed movement)."""
        self._blocked.add((int(pos[0]), int(pos[1])))

    def is_blocked(self, pos: tuple[int, int]) -> bool:
        return (int(pos[0]), int(pos[1])) in self._blocked

    def concept_at(self, pos: tuple[int, int]) -> str | None:
        """Return the concept observed at `pos`, or None if unvisited."""
        entry = self._map.get((int(pos[0]), int(pos[1])))
        return entry[0] if entry else None

    def update(
        self, player_pos: tuple[int, int], near_str: str, confidence: float = 1.0
    ) -> None:
        """Record observation at position with confidence tracking.

        When the same position is observed multiple times, we keep the
        label with the highest confidence. If the same label is observed
        again, confidence is updated via EMA (0.7 * old + 0.3 * new)
        and observation count increments. If a *different* label arrives
        with higher confidence, it replaces the old one (count resets to 1).

        This prevents the 82% segmenter's misclassifications from
        permanently polluting the map — a low-confidence wrong label
        gets overwritten by the next high-confidence correct observation.

        Args:
            player_pos: (y, x) from info["player_pos"].
            near_str: detected concept label.
            confidence: classifier confidence for this observation (0..1).
        """
        y, x = int(player_pos[0]), int(player_pos[1])
        key = (y, x)
        existing = self._map.get(key)
        if existing is not None:
            old_label, old_conf, old_count = existing
            if old_label == near_str:
                # Same label — reinforce via EMA + increment count
                new_conf = 0.7 * old_conf + 0.3 * confidence
                self._map[key] = (near_str, new_conf, old_count + 1)
            elif near_str == "empty" and confidence > 0.5:
                # Bug B fix: "empty" always wins over stale resource labels.
                # Resources (tree, stone, coal…) are consumable — once gone
                # they don't come back within an episode. An empty observation
                # with reasonable confidence should clear the old entry even
                # if the initial resource observation had higher confidence
                # (e.g. written via near_concept conf≈1.0).
                self._map[key] = (near_str, confidence, 1)
            elif near_str in {"table", "furnace"} and confidence >= 0.5:
                # Placed objects only appear through the agent's own action,
                # so an observation of one is authoritative — it must
                # overwrite the prior "empty" (or any other) label even when
                # confidences are equal. Without this the table the agent
                # just placed stays invisible to the planner because the
                # cell was previously written as "empty" at conf=1.0 and the
                # `confidence > old_conf` rule below tied with conf=1.0
                # from the new table observation, ignoring the new label.
                self._map[key] = (near_str, confidence, 1)
            elif confidence > old_conf:
                # Different label, higher confidence — replace
                self._map[key] = (near_str, confidence, 1)
            # else: different label, lower confidence — ignore (keep old)
        else:
            self._map[key] = (near_str, confidence, 1)
        self._visited.add(key)

    def find_nearest(
        self, target: str, player_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Find nearest known position where target was observed.

        Returns (y, x) of nearest known position, or None if not in map.

        Stage 80 Bug 5 fix: positions equal to the player's current
        position are SKIPPED. Crafter blocks walking onto impassable
        resources (tree/stone/water/coal/iron/diamond/cow), so the
        player can never actually be on a resource tile in env. But
        the Stage 75 tile segmenter sometimes mis-classifies the
        player's own sprite (or its underlayer) as a resource and
        the perception loop writes that classification to
        spatial_map at the player's position. Without this guard,
        find_nearest returns the player's own position (manhattan 0)
        and the planner enters an infinite loop in sim while env
        does nothing useful with the resulting "do" primitive.
        """
        py, px = int(player_pos[0]), int(player_pos[1])
        best_pos: tuple[int, int] | None = None
        best_dist = float("inf")
        for (y, x), (near, _conf, _count) in self._map.items():
            if near == target:
                if (y, x) == (py, px):
                    continue  # stale entry at player's tile — env blocks this
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

    def copy(self) -> "CrafterSpatialMap":
        """Stage 81 (Bug 7): independent copy for sim rollouts.

        Without this, simulate_forward shares the planner's
        spatial_map by reference. When sim's _apply_tick fires a
        "do" rule that gathers a resource, the fired tile is NOT
        marked as gone in sim — so the next rollout tick still sees
        the resource at that position and the planner oscillates.
        Real env's clean-up via Bug 6 only runs in run_mpc_episode
        after env.step, not in sim rollouts.

        Copy the underlying dicts but share world_size (immutable).
        """
        new = CrafterSpatialMap(world_size=self.world_size)
        new._map = dict(self._map)  # tuples are immutable — shallow copy is safe
        new._visited = set(self._visited)
        new._blocked = set(self._blocked)
        return new

    @property
    def n_visited(self) -> int:
        return len(self._visited)

    @property
    def known_objects(self) -> dict[str, int]:
        """Count of known positions per near_str (excluding empty)."""
        counts: dict[str, int] = {}
        for near, _conf, _count in self._map.values():
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
