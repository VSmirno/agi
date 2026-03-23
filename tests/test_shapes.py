"""Tests for ShapeGenerator (Этап 2)."""

import torch
import pytest

from snks.data.shapes import ShapeGenerator


@pytest.fixture
def generator() -> ShapeGenerator:
    return ShapeGenerator(image_size=64, seed=42)


class TestShapeGeneratorBasic:
    """Basic ShapeGenerator functionality."""

    def test_num_classes(self, generator: ShapeGenerator) -> None:
        """10 shape classes."""
        assert generator.num_classes == 10

    def test_class_names(self, generator: ShapeGenerator) -> None:
        """All 10 expected classes present."""
        expected = {
            "circle", "square", "triangle", "ellipse", "rectangle",
            "pentagon", "star", "cross", "diamond", "arrow",
        }
        assert set(generator.class_names) == expected

    def test_generate_single_class(self, generator: ShapeGenerator) -> None:
        """Generate images for a single class."""
        images, labels = generator.generate(class_idx=0, n_variations=5)
        assert images.shape == (5, 64, 64)
        assert labels.shape == (5,)
        assert (labels == 0).all()

    def test_generate_all_classes(self, generator: ShapeGenerator) -> None:
        """Generate for all 10 classes × 20 variations = 200 images."""
        images, labels = generator.generate_all(n_variations=20)
        assert images.shape == (200, 64, 64)
        assert labels.shape == (200,)
        # Each class has exactly 20 samples
        for c in range(10):
            assert (labels == c).sum().item() == 20


class TestShapeGeneratorOutput:
    """Output properties."""

    def test_image_dtype(self, generator: ShapeGenerator) -> None:
        """Images are float32."""
        images, _ = generator.generate(class_idx=0, n_variations=1)
        assert images.dtype == torch.float32

    def test_image_range(self, generator: ShapeGenerator) -> None:
        """Pixel values in [0, 1] (before noise might push slightly out)."""
        images, _ = generator.generate(class_idx=0, n_variations=10)
        assert images.min() >= -0.5  # noise tolerance
        assert images.max() <= 1.5

    def test_labels_dtype(self, generator: ShapeGenerator) -> None:
        """Labels are int64."""
        _, labels = generator.generate(class_idx=0, n_variations=5)
        assert labels.dtype == torch.int64

    def test_images_on_cpu(self, generator: ShapeGenerator) -> None:
        """Output tensors on CPU."""
        images, labels = generator.generate(class_idx=0, n_variations=1)
        assert images.device == torch.device("cpu")
        assert labels.device == torch.device("cpu")


class TestShapeVariations:
    """Variation generation."""

    def test_variations_differ(self, generator: ShapeGenerator) -> None:
        """Different variations of same class should differ."""
        images, _ = generator.generate(class_idx=0, n_variations=5)
        # Not all images identical
        assert not torch.allclose(images[0], images[1])

    def test_different_classes_differ(self, generator: ShapeGenerator) -> None:
        """Different classes should produce visually different images."""
        img_a, _ = generator.generate(class_idx=0, n_variations=1)
        img_b, _ = generator.generate(class_idx=2, n_variations=1)
        assert not torch.allclose(img_a[0], img_b[0])

    def test_reproducibility_with_seed(self) -> None:
        """Same seed → same images."""
        gen1 = ShapeGenerator(image_size=64, seed=123)
        gen2 = ShapeGenerator(image_size=64, seed=123)
        img1, _ = gen1.generate(class_idx=0, n_variations=3)
        img2, _ = gen2.generate(class_idx=0, n_variations=3)
        assert torch.allclose(img1, img2)

    def test_noisy_images_nonzero_noise(self, generator: ShapeGenerator) -> None:
        """Some generated images should have noise (σ>0 variations)."""
        images, _ = generator.generate(class_idx=0, n_variations=20)
        # At least some images should have non-integer-like pixel values
        # (noise adds continuous values to binary shape)
        has_fractional = False
        for i in range(20):
            unique_vals = images[i].unique()
            if len(unique_vals) > 10:  # noisy image has many unique values
                has_fractional = True
                break
        assert has_fractional
