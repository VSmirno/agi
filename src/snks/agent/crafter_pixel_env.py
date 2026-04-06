"""Stage 68: Pixel Crafter environment wrapper.

Wraps real crafter.Env to provide:
- Pixel observations (3, 64, 64) float32 [0, 1]
- Native Crafter info dict (inventory, semantic map, player_pos, etc.)

Stage 67: near detection moved to NearDetector (CNN-based).
Stage 68: object-finding navigation moved to CrafterSpatialMap (cognitive map).

Proprioception kept (not replaced by CNN):
- info["player_pos"]  — agent knows where its body is
- info["inventory"]   — agent knows what it carries
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import crafter
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

# Inventory items (also used in decode_head.py)
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

    Returns pixel observations and native Crafter info dict.
    Near object detection is handled externally by NearDetector (Stage 67+).
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

    def reset(self) -> tuple[np.ndarray, dict]:
        """Reset env. Returns (pixels (3,64,64) float32 [0,1], native Crafter info dict).

        Noop step preserved — Crafter does not return info on reset(),
        so one noop step is needed to populate info["inventory"].
        """
        obs = self._env.reset()
        self._last_obs = obs
        # Take a noop step to get info dict (reset doesn't return info)
        obs, _, _, info = self._env.step(0)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), dict(info)

    def step(self, action: int | str) -> tuple[np.ndarray, float, bool, dict]:
        """Step env. Returns (pixels, reward, done, info).

        Stage 67: near detection moved to NearDetector (CNN-based).
        Stage 68: object-finding navigation moved to CrafterSpatialMap.

        Args:
            action: int index or action name string.

        Returns:
            pixels: (3, 64, 64) float32 [0, 1]
            reward: float
            done: bool
            info: native Crafter info dict; info["inventory"] is always present.
        """
        if isinstance(action, str):
            action = ACTION_TO_IDX[action]
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), float(reward), bool(done), dict(info)

    def observe(self) -> tuple[np.ndarray, dict]:
        """Current observation without stepping. Returns (pixels, info)."""
        if self._last_obs is None:
            return self.reset()
        return self._to_pixels(self._last_obs), dict(self._last_info)

    def _to_pixels(self, obs: np.ndarray) -> np.ndarray:
        """(64, 64, 3) uint8 → (3, 64, 64) float32 [0, 1]."""
        return obs.astype(np.float32).transpose(2, 0, 1) / 255.0

    def get_ground_truth(self) -> dict:
        """Full ground truth for current state (for debugging/testing)."""
        return dict(self._last_info)


class CrafterControlledEnv(CrafterPixelEnv):
    """Crafter env with programmatic world editing for controlled data collection.

    Bypasses navigation by placing specific materials adjacent to the player
    and setting inventory directly. Useful for collecting labeled frames of
    rare objects (coal, iron) that are hard to reach via natural chains.
    """

    def reset_near(
        self,
        target: str,
        inventory: dict[str, int] | None = None,
        no_enemies: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Reset and place target material at all 4 cardinal neighbors.

        Args:
            target: material name to place (e.g. "coal", "iron", "stone").
            inventory: items to add to player inventory (e.g. {"wood_pickaxe": 1}).
                       Health/food/drink/energy are preserved.
            no_enemies: if True, patches _balance_chunk to disable zombie/skeleton spawning.

        Returns:
            (pixels, info) with target material adjacent and inventory set.
        """
        self.reset()
        inner = self._env
        py, px = int(inner._player.pos[0]), int(inner._player.pos[1])

        # Place target at all 4 cardinal adjacent cells (overwrites existing material)
        for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            inner._world[py + dy, px + dx] = target

        # Set inventory items (preserve health/food/drink/energy)
        if inventory:
            for item, count in inventory.items():
                inner._player.inventory[item] = count

        # Disable enemy spawning for this episode
        if no_enemies:
            inner._balance_chunk = lambda *args, **kwargs: None  # type: ignore

        # Noop step to refresh semantic map with the newly placed materials
        obs, _, _, info = inner.step(0)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), dict(info)

    def reset_with_items(
        self,
        inventory: dict[str, int],
        no_enemies: bool = True,
    ) -> tuple[np.ndarray, dict]:
        """Reset env and set inventory without placing any materials.

        Useful for collecting empty/table frames: player starts in natural
        grassland context (same distribution as random walk) with enough
        items to perform craft/place actions.

        Args:
            inventory: items to set (e.g. {"wood": 15}).
            no_enemies: if True, disables enemy spawning.

        Returns:
            (pixels, info) after inventory update.
        """
        self.reset()
        inner = self._env

        for item, count in inventory.items():
            inner._player.inventory[item] = count

        if no_enemies:
            inner._balance_chunk = lambda *args, **kwargs: None  # type: ignore

        obs, _, _, info = inner.step(0)
        self._last_obs = obs
        self._last_info = info
        return self._to_pixels(obs), dict(info)
