"""Stage 66: CNN encoder — pixels to z_real + z_vsa + near classification.

Replaces VQ Patch Codebook encoder. Single CNN for all three outputs:
- z_real (2048 float): for JEPA predictor + full scene representation
- z_vsa (2048 binary): for SDM hippocampus (binarized z_real)
- near_logits: classification head for nearest object detection

ROCm note: MIOpen (ROCm conv backend) segfaults on this GPU.
Workaround: torch.backends.cudnn.enabled = False routes convs through
the fallback kernel (slow path but correct). Call disable_rocm_conv()
once at startup when running on ROCm GPU.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


def disable_rocm_conv() -> None:
    """Disable MIOpen backend so Conv2d uses fallback kernel on AMD ROCm.

    MIOpen segfaults on gfx1151 (Radeon 890M / evo-x2). Disabling cudnn
    routes convolutions through a slower but stable path. Call once at
    process startup when device is ROCm GPU.
    """
    torch.backends.cudnn.enabled = False


class CNNEncoderOutput(NamedTuple):
    z_real: torch.Tensor     # (B, 2048) float — for predictor
    z_vsa: torch.Tensor      # (B, 2048) binary {0,1} — for SDM
    near_logits: torch.Tensor  # (B, n_classes) — near object classification
    feature_map: torch.Tensor  # (B, 256, 4, 4) — spatial features (retinotopic)


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

    Architecture (grid_size=8, 3 stride-2 layers → ~1 tile per cell):
        Conv(3→64, 3×3, stride=2)    → (64, 32, 32)
        Conv(64→128, 3×3, stride=2)  → (128, 16, 16)
        Conv(128→256, 3×3, stride=2) → (256, 8, 8)
        Flatten → Linear(16384, 2048) → z_real

    Architecture (grid_size=4, 4 stride-2 layers → ~4 tiles per cell, legacy):
        Conv(3→32, 3×3, stride=2)  → (32, 32, 32)
        Conv(32→64, 3×3, stride=2) → (64, 16, 16)
        Conv(64→128, 3×3, stride=2) → (128, 8, 8)
        Conv(128→C, 3×3, stride=2)  → (C, 4, 4)
        Flatten → Linear(C*16, 2048) → z_real

    Near detection from center features:
        Central 2×2 of feature map → flatten → Linear → near_logits
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        n_near_classes: int = 12,
        vsa_threshold: float = 0.0,
        feature_channels: int = 512,
        grid_size: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vsa_threshold = vsa_threshold
        self.feature_channels = feature_channels
        self.grid_size = grid_size

        if grid_size == 8:
            # 3 layers: 64→32→16→8. ~1 tile per cell.
            self.conv = nn.Sequential(
                SimpleConv(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                SimpleConv(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                SimpleConv(128, feature_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(),
            )
        else:
            # 4 layers: 64→32→16→8→4. Legacy.
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
                SimpleConv(128, feature_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(),
            )

        self.proj = nn.Linear(feature_channels * grid_size * grid_size, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

        # Near classification from central 2×2
        self.near_head = nn.Sequential(
            nn.Linear(feature_channels * 2 * 2, 128),
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
        g = self.grid_size
        c0 = g // 2 - 1  # center start: 1 for grid=4, 3 for grid=8
        center = features[:, :, c0:c0+2, c0:c0+2].flatten(1)
        near_logits = self.near_head(center)  # (B, n_classes)

        if single:
            return CNNEncoderOutput(
                z_real.squeeze(0), z_vsa.squeeze(0), near_logits.squeeze(0),
                features.squeeze(0),
            )
        return CNNEncoderOutput(z_real, z_vsa, near_logits, features)

    @property
    def codebook_utilization(self) -> float:
        """API compatibility with VQ encoder."""
        return 1.0
