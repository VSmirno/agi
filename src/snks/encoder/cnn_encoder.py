"""Stage 66: CNN encoder — pixels to z_real + z_vsa + near classification.

Replaces VQ Patch Codebook encoder. Single CNN for all three outputs:
- z_real (2048 float): for JEPA predictor + full scene representation
- z_vsa (2048 binary): for SDM hippocampus (binarized z_real)
- near_logits: classification head for nearest object detection

Standard convolutions (depthwise causes ROCm segfault with groups).
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class CNNEncoderOutput(NamedTuple):
    z_real: torch.Tensor     # (B, 2048) float — for predictor
    z_vsa: torch.Tensor      # (B, 2048) binary {0,1} — for SDM
    near_logits: torch.Tensor  # (B, n_classes) — near object classification


class SimpleConv(nn.Module):
    """Standard conv block — avoids ROCm segfault with groups/depthwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CNNEncoder(nn.Module):
    """CNN encoder: (3, 64, 64) → z_real + z_vsa + near_logits.

    Architecture:
        Conv(3→32, 3×3, stride=2)  → (32, 32, 32)
        Conv(32→64, 3×3, stride=2) → (64, 16, 16)
        Conv(64→128, 3×3, stride=2) → (128, 8, 8)
        Conv(128→256, 3×3, stride=2) → (256, 4, 4)
        Flatten → Linear(4096, 2048) → z_real

    Near detection from center features:
        Central 2×2 of (256, 4, 4) = (256, 2, 2) → flatten → Linear → near_logits
        These features correspond to the agent's neighborhood.
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        n_near_classes: int = 12,
        vsa_threshold: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vsa_threshold = vsa_threshold

        self.conv = nn.Sequential(
            SimpleConv(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SimpleConv(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SimpleConv(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SimpleConv(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # 64→32→16→8→4: output (256, 4, 4) = 4096
        self.proj = nn.Linear(256 * 4 * 4, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

        # Near classification from central features
        # Center 2×2 of 4×4 feature map = agent's immediate neighborhood
        self.near_head = nn.Sequential(
            nn.Linear(256 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, n_near_classes),
        )

    def forward(self, pixels: torch.Tensor) -> CNNEncoderOutput:
        """Encode pixel observations.

        Args:
            pixels: (B, 3, 64, 64) or (3, 64, 64) RGB float32 in [0, 1].

        Returns:
            CNNEncoderOutput(z_real, z_vsa, near_logits).
        """
        single = pixels.dim() == 3
        if single:
            pixels = pixels.unsqueeze(0)

        # Feature extraction
        features = self.conv(pixels)  # (B, 256, 4, 4)

        # z_real from full feature map
        flat = features.flatten(1)      # (B, 4096)
        z_real = self.ln(self.proj(flat))  # (B, 2048)

        # z_vsa: binarize z_real (median threshold for ~50% sparsity)
        z_vsa = (z_real > self.vsa_threshold).float()

        # Near classification from central 2×2
        center = features[:, :, 1:3, 1:3].flatten(1)  # (B, 256*2*2=1024)
        near_logits = self.near_head(center)  # (B, n_classes)

        if single:
            return CNNEncoderOutput(
                z_real.squeeze(0), z_vsa.squeeze(0), near_logits.squeeze(0),
            )
        return CNNEncoderOutput(z_real, z_vsa, near_logits)

    @property
    def codebook_utilization(self) -> float:
        """API compatibility with VQ encoder."""
        return 1.0
