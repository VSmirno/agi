"""Visual encoder: image → SDR → DAF external currents."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from snks.daf.types import EncoderConfig
from snks.encoder.gabor import GaborBank
from snks.encoder.sdr import kwta

if TYPE_CHECKING:
    from snks.daf.types import ZoneConfig


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

    def sdr_to_currents(
        self, sdr: torch.Tensor, n_nodes: int, zone: ZoneConfig | None = None,
    ) -> torch.Tensor:
        """Map SDR to external currents for DAF engine.

        Uses modular mapping: node i → SDR bit (i * PRIME % sdr_size).
        Active bits inject current_strength into channel 0.

        Args:
            sdr: (sdr_size,) binary SDR.
            n_nodes: total number of DAF nodes (ignored when zone is set).
            zone: if provided, hash only within zone.size nodes and return
                  (zone.size, 8) tensor for zone-based injection.

        Returns:
            (n_nodes, 8) or (zone.size, 8) external currents tensor.
        """
        PRIME = 2654435761
        sz = zone.size if zone is not None else n_nodes
        node_sdr_idx = (torch.arange(sz, device=sdr.device) * PRIME) % self.config.sdr_size

        currents = torch.zeros(sz, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents

    def firing_to_spatial_map(
        self,
        firing_rates: torch.Tensor,
        zone_size: int,
        image_size: int = 32,
    ) -> torch.Tensor:
        """Reconstruct spatial activation map from visual zone firing rates.

        Maps firing rates back through the modular hash to SDR space,
        then reshapes SDR activations into (pool_h, pool_w) spatial grid
        summed across Gabor filters, and upscales to image_size.

        Args:
            firing_rates: (zone_size,) float firing rates per visual node.
            zone_size: number of nodes in visual zone.
            image_size: target output size (square).

        Returns:
            (image_size, image_size) float spatial activation map in [0, 1].
        """
        PRIME = 2654435761
        device = firing_rates.device

        # Node → SDR bit (same hash as sdr_to_currents)
        node_sdr_idx = (torch.arange(zone_size, device=device) * PRIME) % self.config.sdr_size

        # Accumulate firing rates per SDR bit
        sdr_activation = torch.zeros(self.config.sdr_size, device=device)
        sdr_activation.scatter_add_(0, node_sdr_idx, firing_rates)

        # SDR layout: (n_filters, pool_h, pool_w) flattened
        n_filters = self.config.n_orientations * self.config.n_scales * self.config.n_phases
        ph, pw = self.config.pool_h, self.config.pool_w
        spatial = sdr_activation.view(n_filters, ph, pw)

        # Sum across filters → (pool_h, pool_w) spatial map
        spatial_map = spatial.sum(dim=0)

        # Normalize to [0, 1]
        if spatial_map.max() > 0:
            spatial_map = spatial_map / spatial_map.max()

        # Upscale to image_size × image_size
        spatial_map = spatial_map.unsqueeze(0).unsqueeze(0)  # (1, 1, ph, pw)
        spatial_map = torch.nn.functional.interpolate(
            spatial_map, size=(image_size, image_size), mode="bilinear", align_corners=False,
        )
        return spatial_map.squeeze()
