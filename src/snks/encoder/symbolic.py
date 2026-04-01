"""Symbolic encoder for MiniGrid observations (Stage 42 diagnostic).

Encodes MiniGrid's symbolic 7×7×3 observation directly to SDR,
bypassing pixel rendering entirely. Provides perfect object+position
information to test whether DAF/STDP can solve DoorKey at all.

Input: (7, 7, 3) int tensor — object_type, color, state per cell
Output: (sdr_size,) binary SDR
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from snks.daf.types import ZoneConfig


# MiniGrid object types
OBJ_EMPTY = 0
OBJ_WALL = 1
OBJ_FLOOR = 2
OBJ_DOOR = 4
OBJ_KEY = 5
OBJ_BALL = 6
OBJ_BOX = 7
OBJ_GOAL = 8
OBJ_LAVA = 9
OBJ_AGENT = 10
N_OBJ_TYPES = 11
N_COLORS = 6
N_STATES = 3
GRID_SIZE = 7  # MiniGrid partial observation is 7×7


class SymbolicEncoder:
    """Encode MiniGrid symbolic observation to SDR.

    Each cell (i, j) gets a block of bits in the SDR:
      - bits [0:N_OBJ_TYPES): one-hot object type
      - bits [N_OBJ_TYPES:N_OBJ_TYPES+N_COLORS): one-hot color
      - bits [N_OBJ_TYPES+N_COLORS:BITS_PER_CELL): one-hot state

    Total active bits ≈ 49 cells × 2-3 bits/cell ≈ ~100-150 active.
    """

    BITS_PER_CELL = N_OBJ_TYPES + N_COLORS + N_STATES  # 20

    def __init__(self, sdr_size: int = 4096, current_strength: float = 1.0) -> None:
        self.sdr_size = sdr_size
        self.current_strength = current_strength
        # Ensure SDR can fit all cells
        self._min_sdr = GRID_SIZE * GRID_SIZE * self.BITS_PER_CELL  # 980
        if sdr_size < self._min_sdr:
            raise ValueError(f"sdr_size {sdr_size} too small, need >= {self._min_sdr}")

    def encode(self, obs: Tensor) -> Tensor:
        """Encode symbolic observation to SDR.

        Args:
            obs: (7, 7, 3) int tensor or (H, W, 3) — object_type, color, state.
                 Also accepts (H, W) grayscale (returns zero SDR — fallback).

        Returns:
            (sdr_size,) binary float32 SDR.
        """
        device = obs.device if isinstance(obs, Tensor) else torch.device("cpu")
        sdr = torch.zeros(self.sdr_size, device=device)

        if obs.dim() == 2:
            # Grayscale image passed — cannot extract symbolic info
            return sdr

        obs_int = obs.long() if isinstance(obs, Tensor) else torch.tensor(obs).long()
        h, w = obs_int.shape[0], obs_int.shape[1]

        for i in range(min(h, GRID_SIZE)):
            for j in range(min(w, GRID_SIZE)):
                obj_type = int(obs_int[i, j, 0].item())
                color = int(obs_int[i, j, 1].item())
                state = int(obs_int[i, j, 2].item())

                if obj_type == OBJ_EMPTY:
                    continue

                base = (i * GRID_SIZE + j) * self.BITS_PER_CELL
                if base + self.BITS_PER_CELL > self.sdr_size:
                    continue

                # One-hot object type
                if 0 <= obj_type < N_OBJ_TYPES:
                    sdr[base + obj_type] = 1.0
                # One-hot color
                if 0 <= color < N_COLORS:
                    sdr[base + N_OBJ_TYPES + color] = 1.0
                # One-hot state
                if 0 <= state < N_STATES:
                    sdr[base + N_OBJ_TYPES + N_COLORS + state] = 1.0

        return sdr

    def sdr_to_currents(
        self, sdr: Tensor, n_nodes: int, zone: "ZoneConfig | None" = None,
    ) -> Tensor:
        """Map SDR to DAF external currents (same API as VisualEncoder)."""
        PRIME = 2654435761
        sz = zone.size if zone is not None else n_nodes
        node_sdr_idx = (torch.arange(sz, device=sdr.device) * PRIME) % self.sdr_size

        currents = torch.zeros(sz, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.current_strength
        return currents
