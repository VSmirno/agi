"""Stage 66: Pixel Crafter environment wrapper.

Wraps real crafter.Env to provide:
- Pixel observations (3, 64, 64) float32 [0, 1]
- Symbolic observations (for decode head supervision only)
- Semantic map for ground truth "what's nearby"
- Inventory from info dict
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import crafter
    import crafter.constants as crafter_const
    HAS_CRAFTER = True
except ImportError:
    HAS_CRAFTER = False

# Semantic map IDs → material names (0-indexed in constants, 1-indexed in semantic map)
# Plus creature IDs observed: 13=player, 14=cow, 15=zombie, 16=skeleton
SEMANTIC_NAMES = {
    0: "unknown",
    1: "water", 2: "grass", 3: "stone", 4: "path", 5: "sand",
    6: "tree", 7: "lava", 8: "coal", 9: "iron", 10: "diamond",
    11: "table", 12: "furnace",
    13: "player", 14: "cow", 15: "zombie", 16: "skeleton",
}

# Objects relevant for "near" detection (subset of SEMANTIC_NAMES)
NEAR_OBJECTS = [
    "water", "tree", "stone", "coal", "iron", "diamond",
    "table", "furnace", "cow", "zombie", "skeleton",
]

# Inventory items tracked by decode head
INVENTORY_ITEMS = [
    "wood", "stone", "coal", "iron", "diamond", "sapling",
    "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
    "wood_sword", "stone_sword", "iron_sword",
]

ACTION_NAMES = [
    "noop", "move_left", "move_right", "move_up", "move_down",
    "do", "sleep", "place_stone", "place_table", "place_furnace",
    "place_plant", "make_wood_pickaxe", "make_stone_pickaxe",
    "make_iron_pickaxe", "make_wood_sword", "make_stone_sword",
    "make_iron_sword",
]

ACTION_TO_IDX = {name: i for i, name in enumerate(ACTION_NAMES)}


class CrafterPixelEnv:
    """Pixel-based Crafter environment wrapper.

    Returns both pixel observations and symbolic ground truth
    (for decode head supervision during training only).
    """

    def __init__(self, seed: int | None = None):
        if not HAS_CRAFTER:
            raise ImportError("crafter package not installed: pip install crafter")
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        self._env = crafter.Env(**kwargs)
        self._last_info: dict = {}
        self._last_obs: np.ndarray | None = None

    @property
    def n_actions(self) -> int:
        return 17

    @property
    def action_names(self) -> list[str]:
        return ACTION_NAMES

    def reset(self) -> tuple[np.ndarray, dict[str, str]]:
        """Reset env. Returns (pixels, symbolic_obs)."""
        obs = self._env.reset()
        self._last_obs = obs
        # Take a noop step to get info dict (reset doesn't return info)
        obs, _, _, info = self._env.step(0)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), self._to_symbolic(info)

    def step(self, action: int | str) -> tuple[np.ndarray, dict[str, str], float, bool]:
        """Step env. Returns (pixels, symbolic_obs, reward, done)."""
        if isinstance(action, str):
            action = ACTION_TO_IDX[action]
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), self._to_symbolic(info), float(reward), bool(done)

    def observe(self) -> tuple[np.ndarray, dict[str, str]]:
        """Current observation without stepping."""
        if self._last_obs is None:
            return self.reset()
        return self._to_pixels(self._last_obs), self._to_symbolic(self._last_info)

    def _to_pixels(self, obs: np.ndarray) -> np.ndarray:
        """(64, 64, 3) uint8 → (3, 64, 64) float32 [0, 1]."""
        return obs.astype(np.float32).transpose(2, 0, 1) / 255.0

    def _to_symbolic(self, info: dict) -> dict[str, str]:
        """Extract symbolic situation from Crafter info dict.

        Returns dict compatible with CrafterSymbolicEnv.observe() format.
        """
        situation: dict[str, str] = {"domain": "crafter"}

        # What's nearby — find dominant non-grass object near player
        semantic = info.get("semantic")
        player_pos = info.get("player_pos")
        if semantic is not None and player_pos is not None:
            near = self._detect_nearby(semantic, player_pos)
            situation["near"] = near
        else:
            situation["near"] = "empty"

        # Inventory
        inventory = info.get("inventory", {})
        for item in INVENTORY_ITEMS:
            count = inventory.get(item, 0)
            if count > 0:
                situation[f"has_{item}"] = str(count)

        return situation

    def _detect_nearby(self, semantic: np.ndarray, player_pos: np.ndarray) -> str:
        """Detect the most relevant object adjacent to the player.

        Checks a 5×5 region around player position in semantic map.
        Returns the nearest non-ground object type.
        """
        py, px = int(player_pos[0]), int(player_pos[1])
        h, w = semantic.shape

        # Search 5×5 neighborhood, prefer closer objects
        best_obj = "empty"
        best_dist = float("inf")

        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    sid = int(semantic[ny, nx])
                    name = SEMANTIC_NAMES.get(sid, "unknown")
                    if name in NEAR_OBJECTS:
                        dist = abs(dy) + abs(dx)
                        if dist < best_dist:
                            best_dist = dist
                            best_obj = name

        return best_obj

    def get_ground_truth(self) -> dict:
        """Full ground truth for current state (for debugging/testing)."""
        return dict(self._last_info)
