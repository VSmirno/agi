"""ObsAdapter: convert MiniGrid/Gymnasium observations to grayscale images.

Handles both dict observations (with 'image' key) and raw numpy arrays.
Converts to grayscale float32 tensor of target_size × target_size.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ObsAdapter:
    """Convert environment observations to normalized tensors.

    Modes:
        "grayscale" (default): (H, W) float32 grayscale
        "rgb": (3, H, W) float32 RGB in [0, 1]
    """

    def __init__(self, target_size: int = 64, mode: str = "grayscale") -> None:
        self.target_size = target_size
        self.mode = mode

    def convert(self, obs: dict | np.ndarray | Tensor) -> Tensor:
        """Convert observation to tensor.

        Returns:
            mode="grayscale": (target_size, target_size) float32
            mode="rgb": (3, target_size, target_size) float32 in [0, 1]
        """
        if isinstance(obs, dict):
            obs = obs.get("image", obs.get("observation", obs))

        if isinstance(obs, Tensor):
            arr = obs.detach().cpu().numpy()
        elif isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32)
        else:
            arr = np.array(obs, dtype=np.float32)

        if self.mode == "rgb":
            return self._convert_rgb(arr)

        # Grayscale mode (original behavior)
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        arr_max = arr.max()
        if arr_max > 1.0:
            arr = arr / arr_max

        h, w = arr.shape[:2]
        if h != self.target_size or w != self.target_size:
            arr = self._resize(arr, self.target_size, self.target_size)

        return torch.from_numpy(arr).float()

    def _convert_rgb(self, arr: np.ndarray) -> Tensor:
        """Convert to (3, H, W) RGB float32 in [0, 1]."""
        if arr.ndim == 2:
            # Grayscale → fake RGB
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # Normalize to [0, 1]
        arr_max = arr.max()
        if arr_max > 1.0:
            arr = arr / 255.0 if arr_max > 1.0 else arr

        # Resize
        h, w = arr.shape[:2]
        if h != self.target_size or w != self.target_size:
            arr = self._resize_rgb(arr, self.target_size, self.target_size)

        # (H, W, 3) → (3, H, W)
        return torch.from_numpy(arr.transpose(2, 0, 1)).float()

    @staticmethod
    def _resize_rgb(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Nearest-neighbor resize for RGB (H, W, 3)."""
        h, w = arr.shape[:2]
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return arr[np.ix_(row_idx, col_idx)]

    @staticmethod
    def _resize(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Nearest-neighbor resize without external dependencies."""
        h, w = arr.shape[:2]
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return arr[np.ix_(row_idx, col_idx)]
