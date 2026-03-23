"""Gabor filter bank for V1-like feature extraction."""

import math

import numpy as np
import torch
import torch.nn as nn
from skimage.filters import gabor_kernel

from snks.daf.types import EncoderConfig


class GaborBank(nn.Module):
    """Bank of 128 Gabor filters as a frozen Conv2d.

    128 = n_scales(4) × n_orientations(8) × n_phases(4).
    Activation: torch.abs() (complex-cell V1 model).
    """

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.n_filters = config.n_scales * config.n_orientations * config.n_phases
        ks = config.gabor_kernel_size

        # Build kernels
        sigmas = [1.0, 2.0, 3.0, 4.0]
        lambdas = [4.0, 8.0, 12.0, 16.0]
        thetas = [i * math.pi / config.n_orientations for i in range(config.n_orientations)]
        phases = [i * 2 * math.pi / config.n_phases for i in range(config.n_phases)]

        kernels = []
        for s_idx in range(config.n_scales):
            for theta in thetas:
                for phase in phases:
                    kern = gabor_kernel(
                        frequency=1.0 / lambdas[s_idx],
                        theta=theta,
                        sigma_x=sigmas[s_idx],
                        sigma_y=sigmas[s_idx],
                        offset=phase,
                    )
                    # Take real part
                    k_real = np.real(kern).astype(np.float32)
                    kh, kw = k_real.shape
                    # Crop center to ks×ks if kernel is larger
                    if kh > ks or kw > ks:
                        ch = (kh - ks) // 2
                        cw = (kw - ks) // 2
                        k_real = k_real[ch : ch + ks, cw : cw + ks]
                        kh, kw = ks, ks
                    # Pad to ks×ks if kernel is smaller
                    padded = np.zeros((ks, ks), dtype=np.float32)
                    oh = (ks - kh) // 2
                    ow = (ks - kw) // 2
                    padded[oh : oh + kh, ow : ow + kw] = k_real
                    # Zero-mean
                    padded -= padded.mean()
                    # L2 normalize
                    norm = np.linalg.norm(padded)
                    if norm > 1e-8:
                        padded /= norm
                    kernels.append(padded)

        weight = torch.tensor(np.stack(kernels)).unsqueeze(1)  # (128, 1, 19, 19)

        self.conv = nn.Conv2d(
            1, self.n_filters, kernel_size=ks, padding=ks // 2, bias=False,
        )
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gabor filters with abs() activation.

        Args:
            x: (B, 1, H, W) images.

        Returns:
            (B, 128, H, W) feature maps, all non-negative.
        """
        return torch.abs(self.conv(x))
