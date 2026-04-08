"""Patch-based perception: 7×7 pixel templates from Crafter tiles.

Sprites are IDENTICAL for all instances of the same object.
Template matching = 100% accurate (diff=0.0000 confirmed).

Replaces CNN cosine matching for object recognition.
CNN stays as V1 fallback for environments with sprite variation.

Algorithm:
  1. Every move: check if position changed (collision detection)
  2. If blocked: extract 7×7 patch from facing direction
  3. "do" → inventory change → label the patch as that object
  4. Store labeled patches as templates
  5. Match new patches against templates: pixel comparison

No CNN features, no cosine matching, no projection heads.
147 numbers (7×7×3 RGB) compared directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


# Crafter layout: 9×7 game tiles, each 7×7 pixels, player at center (col=4, row=3)
# Facing direction → pixel region of tile ahead
FACING_TO_PATCH: dict[str, tuple[int, int, int, int]] = {
    "move_right": (21, 35, 28, 42),  # (y0, x0, y1, x1)
    "move_left":  (21, 21, 28, 28),
    "move_down":  (28, 28, 35, 35),
    "move_up":    (14, 28, 21, 35),
}

# Also extract patches for ALL visible tiles (9×7 = 63 tiles)
# For spatial perception: what's at each tile position
TILE_SIZE = 7
VIEW_COLS = 9
VIEW_ROWS = 7  # game area only (not inventory)


@dataclass
class PatchTemplate:
    """A learned 7×7 pixel template for an object."""
    label: str
    patch: np.ndarray  # (3, 7, 7) float32
    count: int = 1     # how many times confirmed


@dataclass
class PatchStore:
    """Collection of learned templates from experience."""
    templates: dict[str, PatchTemplate] = field(default_factory=dict)

    def add(self, label: str, patch: np.ndarray) -> None:
        """Add or update template. First observation = store. Later = average."""
        if label not in self.templates:
            self.templates[label] = PatchTemplate(label=label, patch=patch.copy())
        else:
            t = self.templates[label]
            # Running average (all sprites identical, so this converges fast)
            t.patch = (t.patch * t.count + patch) / (t.count + 1)
            t.count += 1

    def match(self, patch: np.ndarray, threshold: float = 0.02) -> str | None:
        """Match patch against templates. Returns label or None."""
        best_label = None
        best_diff = float("inf")
        for label, template in self.templates.items():
            diff = np.abs(patch - template.patch).mean()
            if diff < best_diff:
                best_diff = diff
                best_label = label
        if best_diff < threshold:
            return best_label
        return None

    def match_all_visible(
        self, pixels: np.ndarray, threshold: float = 0.02,
    ) -> list[tuple[str, int, int, float]]:
        """Match ALL visible tiles against templates.

        Returns list of (label, tile_row, tile_col, diff) for matches.
        """
        detections = []
        for row in range(VIEW_ROWS):
            for col in range(VIEW_COLS):
                y0 = row * TILE_SIZE
                x0 = col * TILE_SIZE
                patch = pixels[:, y0:y0+TILE_SIZE, x0:x0+TILE_SIZE]
                if patch.shape != (3, TILE_SIZE, TILE_SIZE):
                    continue
                label = self.match(patch, threshold)
                if label is not None:
                    diff = np.abs(patch - self.templates[label].patch).mean()
                    detections.append((label, row, col, diff))
        return detections


def extract_facing_patch(pixels: np.ndarray, direction: str) -> np.ndarray | None:
    """Extract 7×7 patch from the tile the agent is facing.

    Args:
        pixels: (3, 64, 64) float32 frame
        direction: last move action ("move_right", etc.)

    Returns:
        (3, 7, 7) patch or None if direction unknown
    """
    coords = FACING_TO_PATCH.get(direction)
    if coords is None:
        return None
    y0, x0, y1, x1 = coords
    return pixels[:, y0:y1, x0:x1].copy()


def detect_collision(
    pos_before: np.ndarray | tuple,
    pos_after: np.ndarray | tuple,
) -> bool:
    """Check if agent was blocked (position didn't change)."""
    return (int(pos_before[0]) == int(pos_after[0]) and
            int(pos_before[1]) == int(pos_after[1]))
