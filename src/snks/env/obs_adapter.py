"""ObsAdapter: convert MiniGrid/Gymnasium observations to grayscale images.

Handles both dict observations (with 'image' key) and raw numpy arrays.
Converts to grayscale float32 tensor of target_size × target_size.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


class ObsAdapter:
    """Convert environment observations to normalized grayscale tensors."""

    def __init__(self, target_size: int = 64) -> None:
        self.target_size = target_size

    def convert(self, obs: dict | np.ndarray | Tensor) -> Tensor:
        """Convert observation to (target_size, target_size) float32 tensor.

        Accepts:
        - dict with 'image' key (MiniGrid style)
        - np.ndarray (H, W, C) or (H, W)
        - torch.Tensor
        """
        if isinstance(obs, dict):
            obs = obs.get("image", obs.get("observation", obs))

        if isinstance(obs, Tensor):
            arr = obs.detach().cpu().numpy()
        elif isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32)
        else:
            arr = np.array(obs, dtype=np.float32)

        # To grayscale if needed.
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        elif arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]

        # Normalize to [0, 1].
        arr_max = arr.max()
        if arr_max > 1.0:
            arr = arr / arr_max

        # Resize to target_size via simple nearest-neighbor.
        h, w = arr.shape[:2]
        if h != self.target_size or w != self.target_size:
            arr = self._resize(arr, self.target_size, self.target_size)

        return torch.from_numpy(arr).float()

    @staticmethod
    def _resize(arr: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Nearest-neighbor resize without external dependencies."""
        h, w = arr.shape[:2]
        row_idx = (np.arange(th) * h / th).astype(int)
        col_idx = (np.arange(tw) * w / tw).astype(int)
        return arr[np.ix_(row_idx, col_idx)]
