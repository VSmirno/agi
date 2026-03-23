"""Visual stimulus generator for encoder testing.

Generates oriented sinusoidal gratings — the canonical stimulus for
Gabor-based V1 feature extraction. 10 orientations × N variations
(spatial frequency, phase, noise).
"""

import math
from typing import Tuple

import torch
import numpy as np


class GratingGenerator:
    """Generates oriented gratings as visual stimuli.

    10 classes = 10 evenly spaced orientations (0°, 18°, 36°, ..., 162°).
    Variations: spatial frequency, phase shift, Gaussian noise.
    """

    NUM_CLASSES = 10
    CLASS_NAMES = [f"grating_{i * 18}deg" for i in range(10)]

    def __init__(self, image_size: int = 64, seed: int = 42) -> None:
        self.image_size = image_size
        self.seed = seed
        self.num_classes = self.NUM_CLASSES
        self.class_names = list(self.CLASS_NAMES)

    def generate(self, class_idx: int, n_variations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate variations of a single grating orientation.

        Args:
            class_idx: orientation index (0-9).
            n_variations: number of variations to generate.

        Returns:
            images: (n_variations, H, W) float32 in [0, 1].
            labels: (n_variations,) int64.
        """
        rng = np.random.RandomState(self.seed + class_idx * 1000)
        theta = class_idx * math.pi / self.NUM_CLASSES

        freqs = [0.04, 0.06, 0.08, 0.10]
        noise_levels = [0.0, 0.05, 0.1]

        yy, xx = np.meshgrid(
            np.arange(self.image_size, dtype=np.float32),
            np.arange(self.image_size, dtype=np.float32),
            indexing="ij",
        )

        images = []
        for i in range(n_variations):
            freq = freqs[i % len(freqs)]
            phase = rng.uniform(0, 2 * math.pi)
            noise_sigma = noise_levels[i % len(noise_levels)]

            proj = xx * math.cos(theta) + yy * math.sin(theta)
            grating = 0.5 + 0.5 * np.sin(2 * math.pi * freq * proj + phase)
            img = torch.tensor(grating, dtype=torch.float32)

            if noise_sigma > 0:
                torch.manual_seed(self.seed + class_idx * 1000 + i)
                img = img + noise_sigma * torch.randn_like(img)
                img = img.clamp(0.0, 1.0)

            images.append(img)

        return (
            torch.stack(images),
            torch.full((n_variations,), class_idx, dtype=torch.int64),
        )

    def generate_all(self, n_variations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all 10 orientations × n_variations.

        Returns:
            images: (10*n_variations, H, W) float32.
            labels: (10*n_variations,) int64.
        """
        all_images = []
        all_labels = []
        for c in range(self.num_classes):
            imgs, lbls = self.generate(c, n_variations)
            all_images.append(imgs)
            all_labels.append(lbls)
        return torch.cat(all_images), torch.cat(all_labels)
