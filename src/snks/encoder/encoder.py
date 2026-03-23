"""Visual encoder: image → SDR → DAF external currents."""

import math

import torch
import torch.nn as nn

from snks.daf.types import EncoderConfig
from snks.encoder.gabor import GaborBank
from snks.encoder.sdr import kwta


class VisualEncoder(nn.Module):
    """Encodes grayscale images into sparse distributed representations.

    Pipeline: image → GaborBank → AdaptiveAvgPool2d → flatten → k-WTA → SDR.
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gabor = GaborBank(config)
        self.pool = nn.AdaptiveAvgPool2d((config.pool_h, config.pool_w))
        self.k = round(config.sdr_size * config.sdr_sparsity)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode image(s) to SDR(s).

        Args:
            images: (H, W) or (B, H, W) grayscale float32 images in [0, 1].

        Returns:
            (sdr_size,) or (B, sdr_size) binary SDR.
        """
        single = images.dim() == 2
        if single:
            images = images.unsqueeze(0)

        # (B, H, W) → (B, 1, H, W)
        x = images.unsqueeze(1)
        # Gabor features: (B, 128, H, W)
        x = self.gabor(x)
        # Pool: (B, 128, pool_h, pool_w)
        x = self.pool(x)
        # Flatten: (B, sdr_size)
        x = x.flatten(1)
        # k-WTA: (B, sdr_size) binary
        sdr = kwta(x, self.k)

        if single:
            return sdr.squeeze(0)
        return sdr

    def sdr_to_currents(self, sdr: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Map SDR to external currents for DAF engine.

        Uses modular mapping: node i → SDR bit (i % sdr_size).
        This ensures all SDR bits are covered regardless of n_nodes.
        Active bits inject current_strength into channel 0.

        Args:
            sdr: (sdr_size,) binary SDR.
            n_nodes: number of DAF nodes.

        Returns:
            (n_nodes, 8) external currents tensor.
        """
        # Multiplicative hash distributes nodes uniformly across all SDR bits
        PRIME = 2654435761
        node_sdr_idx = (torch.arange(n_nodes, device=sdr.device) * PRIME) % self.config.sdr_size

        currents = torch.zeros(n_nodes, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents
