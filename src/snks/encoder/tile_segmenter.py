"""Stage 75 / 76: TileSegmenter — no-stride FCN for per-tile classification.

Extracted from exp135 for reuse in Stage 76 (continuous learning needs to
load the checkpoint without depending on the experiment file).

Architecture:
  Conv3x3(3→32) → BN → ReLU
  Conv3x3(32→64) → BN → ReLU
  Conv3x3(64→64) → BN → ReLU
  AdaptiveAvgPool2d → (VIEWPORT_ROWS, VIEWPORT_COLS)   # 7×9
  Conv1x1(64→n_classes)

~57K parameters. No stride in feature extraction — each output cell
still sees the full image; downsampling happens only at the pool step.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from snks.agent.decode_head import NEAR_CLASSES
from snks.encoder.tile_head_trainer import VIEWPORT_ROWS, VIEWPORT_COLS


class TileSegmenter(nn.Module):
    """No-stride FCN producing a VIEWPORT_ROWS × VIEWPORT_COLS tile map."""

    def __init__(self, n_classes: int = len(NEAR_CLASSES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((VIEWPORT_ROWS, VIEWPORT_COLS))
        self.head = nn.Conv2d(64, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.head(self.pool(self.features(x)))

    def classify_tiles(
        self, pixels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CNNEncoder-compatible API for downstream code.

        Automatically moves inputs to the module's device. Returns CPU
        tensors so callers can safely call .item() without device transfers.

        Returns:
            (class_ids, confidences) — both shape (VIEWPORT_ROWS, VIEWPORT_COLS)
            for a single frame, or (B, VIEWPORT_ROWS, VIEWPORT_COLS) for batches.
        """
        module_device = next(self.parameters()).device
        if pixels.device != module_device:
            pixels = pixels.to(module_device)
        with torch.no_grad():
            logits = self.forward(pixels)
            if logits.shape[0] == 1:
                logits = logits.squeeze(0)
                probs = torch.softmax(logits, dim=0)
                class_ids = probs.argmax(dim=0)
                confidences = probs.max(dim=0).values
            else:
                probs = torch.softmax(logits, dim=1)
                class_ids = probs.argmax(dim=1)
                confidences = probs.max(dim=1).values
        return class_ids.cpu(), confidences.cpu()


def pick_device() -> torch.device:
    """Return best available torch device (cuda > cpu).

    ROCm builds expose AMD GPUs as `cuda`. HSA_OVERRIDE_GFX_VERSION must
    already be set in the environment before torch is imported for ROCm.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tile_segmenter(
    checkpoint_path: str,
    device: torch.device | str | None = None,
) -> TileSegmenter:
    """Load a TileSegmenter from a Stage 75 state_dict checkpoint.

    Args:
        checkpoint_path: path to state_dict .pt file.
        device: torch device or string. If None, picks cuda when available.
    """
    if device is None:
        device = pick_device()
    device = torch.device(device)
    segmenter = TileSegmenter()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    segmenter.load_state_dict(state)
    segmenter.to(device)
    segmenter.eval()
    return segmenter
