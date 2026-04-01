"""RGB convolutional encoder for visual observations (Stage 42).

Replaces Gabor+grayscale pipeline with a 3-layer CNN on RGB input.
Preserves color and spatial information that Gabor filters destroy.

Weights are frozen (Xavier init, no backprop) — this is a random
projection that still outperforms Gabor on color-coded environments
because it preserves RGB channel separation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from snks.daf.types import EncoderConfig
from snks.encoder.sdr import kwta

if TYPE_CHECKING:
    from snks.daf.types import ZoneConfig


class RGBConvEncoder(nn.Module):
    """RGB CNN encoder: (3, H, W) → SDR.

    Architecture:
        Conv2d(3→32, 3×3, stride=2)  → 64→32
        Conv2d(32→64, 3×3, stride=2) → 32→16
        Conv2d(64→128, 3×3, stride=2) → 16→8
        Flatten → Linear(8192, sdr_size) → kWTA → SDR
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        # For 64×64 input: 128 × 8 × 8 = 8192
        self.proj = nn.Linear(128 * 8 * 8, config.sdr_size)
        self.k = round(config.sdr_size * config.sdr_sparsity)

        # Freeze all weights — no backprop
        for p in self.parameters():
            p.requires_grad_(False)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode RGB image(s) to SDR.

        Args:
            images: (3, H, W) or (B, 3, H, W) RGB float32 in [0, 1].

        Returns:
            (sdr_size,) or (B, sdr_size) binary SDR.
        """
        single = images.dim() == 3
        if single:
            images = images.unsqueeze(0)

        x = self.conv(images)  # (B, 128, 8, 8)
        x = x.flatten(1)       # (B, 8192)
        x = self.proj(x)       # (B, sdr_size)
        sdr = kwta(x, self.k)

        if single:
            return sdr.squeeze(0)
        return sdr

    def sdr_to_currents(
        self, sdr: torch.Tensor, n_nodes: int, zone: "ZoneConfig | None" = None,
    ) -> torch.Tensor:
        """Map SDR to DAF external currents (same API as VisualEncoder)."""
        PRIME = 2654435761
        sz = zone.size if zone is not None else n_nodes
        node_sdr_idx = (torch.arange(sz, device=sdr.device) * PRIME) % self.config.sdr_size

        currents = torch.zeros(sz, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents
