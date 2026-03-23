"""Shape dataset generator for visual encoder testing."""

import math
from typing import List, Tuple

import torch
import numpy as np
from PIL import Image, ImageDraw


class ShapeGenerator:
    """Generates 10 classes of geometric shapes with variations.

    Classes: circle, square, triangle, ellipse, rectangle,
             pentagon, star, cross, diamond, arrow.

    Variations: 3 sizes × random position × random rotation × 3 noise levels.
    """

    CLASS_NAMES = [
        "circle", "square", "triangle", "ellipse", "rectangle",
        "pentagon", "star", "cross", "diamond", "arrow",
    ]

    def __init__(self, image_size: int = 64, seed: int = 42) -> None:
        self.image_size = image_size
        self.seed = seed
        self.num_classes = 10
        self.class_names = list(self.CLASS_NAMES)

    def generate(self, class_idx: int, n_variations: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate variations of a single shape class.

        Returns:
            images: (n_variations, H, W) float32, labels: (n_variations,) int64.
        """
        rng = np.random.RandomState(self.seed + class_idx * 1000)
        sizes = [16, 24, 32]
        noise_levels = [0.0, 0.05, 0.1]

        images = []
        for i in range(n_variations):
            size = sizes[i % len(sizes)]
            noise_sigma = noise_levels[i % len(noise_levels)]
            angle = rng.uniform(0, 360)
            margin = size // 2 + 1
            cx = rng.randint(margin, self.image_size - margin)
            cy = rng.randint(margin, self.image_size - margin)

            img = self._draw_shape(class_idx, cx, cy, size, angle)
            img_tensor = torch.tensor(np.array(img, dtype=np.float32) / 255.0)

            if noise_sigma > 0:
                torch.manual_seed(self.seed + class_idx * 1000 + i)
                img_tensor = img_tensor + noise_sigma * torch.randn_like(img_tensor)

            images.append(img_tensor)

        return (
            torch.stack(images),
            torch.full((n_variations,), class_idx, dtype=torch.int64),
        )

    def generate_all(self, n_variations: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate all 10 classes × n_variations.

        Returns:
            images: (10*n_variations, H, W) float32, labels: (10*n_variations,) int64.
        """
        all_images = []
        all_labels = []
        for c in range(self.num_classes):
            imgs, lbls = self.generate(c, n_variations)
            all_images.append(imgs)
            all_labels.append(lbls)
        return torch.cat(all_images), torch.cat(all_labels)

    def _draw_shape(self, class_idx: int, cx: int, cy: int, size: int, angle: float) -> Image.Image:
        """Draw a shape on a blank image."""
        img = Image.new("L", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(img)
        name = self.CLASS_NAMES[class_idx]

        if name == "circle":
            r = size // 2
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)

        elif name == "square":
            pts = _regular_polygon(cx, cy, size // 2, 4, angle + 45)
            draw.polygon(pts, fill=255)

        elif name == "triangle":
            pts = _regular_polygon(cx, cy, size // 2, 3, angle)
            draw.polygon(pts, fill=255)

        elif name == "ellipse":
            # Rotate by drawing a rotated polygon approximation
            pts = _ellipse_points(cx, cy, size // 2, size // 3, angle, n=36)
            draw.polygon(pts, fill=255)

        elif name == "rectangle":
            hw, hh = size // 2, size // 3
            pts = _rotate_rect(cx, cy, hw, hh, angle)
            draw.polygon(pts, fill=255)

        elif name == "pentagon":
            pts = _regular_polygon(cx, cy, size // 2, 5, angle)
            draw.polygon(pts, fill=255)

        elif name == "star":
            pts = _star_points(cx, cy, size // 2, size // 4, 5, angle)
            draw.polygon(pts, fill=255)

        elif name == "cross":
            arm_w = size // 4
            arm_l = size // 2
            pts = _cross_points(cx, cy, arm_w, arm_l, angle)
            draw.polygon(pts, fill=255)

        elif name == "diamond":
            pts = _regular_polygon(cx, cy, size // 2, 4, angle)
            draw.polygon(pts, fill=255)

        elif name == "arrow":
            pts = _arrow_points(cx, cy, size, angle)
            draw.polygon(pts, fill=255)

        return img


def _rotate_point(x: float, y: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    """Rotate (x, y) around (cx, cy) by angle_deg degrees."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    dx, dy = x - cx, y - cy
    return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)


def _rotate_points(pts: List[Tuple[float, float]], cx: float, cy: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Rotate list of points around center."""
    return [_rotate_point(x, y, cx, cy, angle_deg) for x, y in pts]


def _regular_polygon(cx: float, cy: float, r: float, n: int, angle_deg: float = 0) -> List[Tuple[float, float]]:
    """Generate regular n-gon vertices."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _rotate_points(pts, cx, cy, angle_deg)


def _star_points(cx: float, cy: float, r_outer: float, r_inner: float, n: int, angle_deg: float) -> List[Tuple[float, float]]:
    """Generate star vertices (alternating outer/inner)."""
    pts = []
    for i in range(2 * n):
        a = math.pi * i / n - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return _rotate_points(pts, cx, cy, angle_deg)


def _ellipse_points(cx: float, cy: float, rx: float, ry: float, angle_deg: float, n: int = 36) -> List[Tuple[float, float]]:
    """Generate ellipse as polygon points with rotation."""
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return _rotate_points(pts, cx, cy, angle_deg)


def _rotate_rect(cx: float, cy: float, hw: float, hh: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Rectangle corners rotated around center."""
    pts = [(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + hw, cy + hh), (cx - hw, cy + hh)]
    return _rotate_points(pts, cx, cy, angle_deg)


def _cross_points(cx: float, cy: float, arm_w: float, arm_l: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Cross (plus sign) as 12-vertex polygon."""
    w, l = arm_w, arm_l
    pts = [
        (cx - w, cy - l), (cx + w, cy - l), (cx + w, cy - w),
        (cx + l, cy - w), (cx + l, cy + w), (cx + w, cy + w),
        (cx + w, cy + l), (cx - w, cy + l), (cx - w, cy + w),
        (cx - l, cy + w), (cx - l, cy - w), (cx - w, cy - w),
    ]
    return _rotate_points(pts, cx, cy, angle_deg)


def _arrow_points(cx: float, cy: float, size: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Arrow pointing up (before rotation)."""
    hs = size / 2
    hw = size / 4
    shaft_w = size / 8
    pts = [
        (cx, cy - hs),                  # tip
        (cx + hw, cy - hs + hw),         # right head
        (cx + shaft_w, cy - hs + hw),    # right shaft top
        (cx + shaft_w, cy + hs),         # right shaft bottom
        (cx - shaft_w, cy + hs),         # left shaft bottom
        (cx - shaft_w, cy - hs + hw),    # left shaft top
        (cx - hw, cy - hs + hw),         # left head
    ]
    return _rotate_points(pts, cx, cy, angle_deg)
