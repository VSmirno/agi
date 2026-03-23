"""Tests for GratingGenerator (Этап 2)."""

import torch
import pytest

from snks.data.stimuli import GratingGenerator


@pytest.fixture
def generator() -> GratingGenerator:
    return GratingGenerator(image_size=64, seed=42)


class TestGratingGeneratorBasic:
    """Basic GratingGenerator functionality."""

    def test_num_classes(self, generator: GratingGenerator) -> None:
        """10 orientation classes."""
        assert generator.num_classes == 10

    def test_class_names(self, generator: GratingGenerator) -> None:
        """All 10 expected classes present."""
        assert len(generator.class_names) == 10
        assert generator.class_names[0] == "grating_0deg"
        assert generator.class_names[5] == "grating_90deg"

    def test_generate_single_class(self, generator: GratingGenerator) -> None:
        """Generate images for a single orientation."""
        images, labels = generator.generate(class_idx=0, n_variations=5)
        assert images.shape == (5, 64, 64)
        assert labels.shape == (5,)
        assert (labels == 0).all()

    def test_generate_all_classes(self, generator: GratingGenerator) -> None:
        """Generate for all 10 classes × 20 variations = 200 images."""
        images, labels = generator.generate_all(n_variations=20)
        assert images.shape == (200, 64, 64)
        assert labels.shape == (200,)
        for c in range(10):
            assert (labels == c).sum().item() == 20


class TestGratingGeneratorOutput:
    """Output properties."""

    def test_image_dtype(self, generator: GratingGenerator) -> None:
        """Images are float32."""
        images, _ = generator.generate(class_idx=0, n_variations=1)
        assert images.dtype == torch.float32

    def test_image_range(self, generator: GratingGenerator) -> None:
        """Pixel values in [0, 1]."""
        images, _ = generator.generate(class_idx=0, n_variations=10)
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_labels_dtype(self, generator: GratingGenerator) -> None:
        """Labels are int64."""
        _, labels = generator.generate(class_idx=0, n_variations=5)
        assert labels.dtype == torch.int64

    def test_images_on_cpu(self, generator: GratingGenerator) -> None:
        """Output tensors on CPU."""
        images, labels = generator.generate(class_idx=0, n_variations=1)
        assert images.device == torch.device("cpu")
        assert labels.device == torch.device("cpu")


class TestGratingVariations:
    """Variation generation."""

    def test_variations_differ(self, generator: GratingGenerator) -> None:
        """Different variations of same orientation should differ."""
        images, _ = generator.generate(class_idx=0, n_variations=5)
        assert not torch.allclose(images[0], images[1])

    def test_different_classes_differ(self, generator: GratingGenerator) -> None:
        """Different orientations produce different images."""
        img_a, _ = generator.generate(class_idx=0, n_variations=1)
        img_b, _ = generator.generate(class_idx=5, n_variations=1)
        assert not torch.allclose(img_a[0], img_b[0])

    def test_reproducibility_with_seed(self) -> None:
        """Same seed → same images."""
        gen1 = GratingGenerator(image_size=64, seed=123)
        gen2 = GratingGenerator(image_size=64, seed=123)
        img1, _ = gen1.generate(class_idx=0, n_variations=3)
        img2, _ = gen2.generate(class_idx=0, n_variations=3)
        assert torch.allclose(img1, img2)

    def test_grating_has_continuous_values(self, generator: GratingGenerator) -> None:
        """Gratings have smooth sinusoidal values (many unique pixel values)."""
        images, _ = generator.generate(class_idx=0, n_variations=1)
        assert len(images[0].unique()) > 50
