"""Spatial RGB representation without supervised labels or output normalization."""

from torch import Tensor, nn


class CoreEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        if z_dim <= 0:
            raise ValueError("z_dim must be positive")
        self.z_dim = z_dim
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, z_dim),
        )

    def forward(self, rgb: Tensor) -> Tensor:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 64, 64):
            raise ValueError("encoder expects B,3,64,64 RGB")
        return self.layers(rgb)
