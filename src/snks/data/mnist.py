"""MNIST dataset loader for SNKS experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import datasets


class MnistLoader:
    """Load and preprocess MNIST for the SNKS pipeline.

    Resizes 28x28 images to target_size (default 64) to match
    the Gabor filter bank's receptive field.

    preprocess modes:
        "raw"     — resize only (default)
        "contour" — binarize + Sobel edge detection (retina-like preprocessing)
    """

    def __init__(
        self,
        data_root: str = "data/",
        target_size: int = 64,
        seed: int = 42,
        preprocess: str = "raw",
    ) -> None:
        self.data_root = data_root
        self.target_size = target_size
        self.seed = seed
        self.preprocess = preprocess

    def load(
        self,
        split: str = "train",
        n_per_class: int | None = None,
        classes: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load MNIST images and labels.

        Args:
            split: "train" (60K) or "test" (10K).
            n_per_class: If set, take this many images per class (stratified).
            classes: If set, only include these digit classes.

        Returns:
            images: (N, target_size, target_size) float32 in [0, 1].
            labels: (N,) int64.
        """
        train = split == "train"
        ds = datasets.MNIST(root=self.data_root, train=train, download=True)

        # Raw data: (N, 28, 28) uint8, labels (N,)
        all_images = ds.data.float() / 255.0  # (N, 28, 28) float32
        all_labels = ds.targets  # (N,) int64

        # Filter by classes
        if classes is not None:
            mask = torch.zeros(len(all_labels), dtype=torch.bool)
            for c in classes:
                mask |= all_labels == c
            all_images = all_images[mask]
            all_labels = all_labels[mask]

        # Stratified subsample
        if n_per_class is not None:
            rng = torch.Generator().manual_seed(self.seed)
            unique_classes = all_labels.unique().tolist()
            selected_idx: list[int] = []
            for c in unique_classes:
                class_idx = (all_labels == c).nonzero(as_tuple=False).flatten()
                perm = torch.randperm(len(class_idx), generator=rng)
                take = min(n_per_class, len(class_idx))
                selected_idx.extend(class_idx[perm[:take]].tolist())
            idx = torch.tensor(selected_idx, dtype=torch.long)
            all_images = all_images[idx]
            all_labels = all_labels[idx]

        # Resize 28x28 -> target_size x target_size
        if self.target_size != 28:
            # F.interpolate expects (B, C, H, W)
            imgs_4d = all_images.unsqueeze(1)  # (N, 1, 28, 28)
            imgs_4d = F.interpolate(
                imgs_4d,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )
            all_images = imgs_4d.squeeze(1)  # (N, target_size, target_size)

        # Apply preprocessing
        if self.preprocess == "contour":
            all_images = self._contour(all_images)

        return all_images, all_labels

    @staticmethod
    def _contour(imgs: torch.Tensor, threshold: float = 0.3) -> torch.Tensor:
        """Binarize + Sobel edge detection (retina-like contour extraction)."""
        binary = (imgs > threshold).float()
        kx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32,
        ).view(1, 1, 3, 3)
        ky = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32,
        ).view(1, 1, 3, 3)
        x = binary.unsqueeze(1)  # (N, 1, H, W)
        gx = F.conv2d(x, kx, padding=1)
        gy = F.conv2d(x, ky, padding=1)
        mag = torch.sqrt(gx**2 + gy**2).squeeze(1)  # (N, H, W)
        # Per-image normalize to [0, 1]
        maxvals = mag.flatten(1).max(dim=1).values.view(-1, 1, 1).clamp(min=1e-8)
        mag = mag / maxvals
        return mag
