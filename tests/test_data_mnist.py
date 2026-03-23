"""Tests for MnistLoader."""

import pytest
import torch

from snks.data.mnist import MnistLoader


@pytest.fixture
def loader() -> MnistLoader:
    return MnistLoader(data_root="data/", target_size=64, seed=42)


class TestMnistLoader:
    """MnistLoader.load() tests."""

    @pytest.mark.slow
    def test_output_shape(self, loader: MnistLoader) -> None:
        images, labels = loader.load("train", n_per_class=5)
        assert images.dim() == 3
        assert images.shape[1] == 64
        assert images.shape[2] == 64
        assert images.shape[0] == 50  # 10 classes * 5

    @pytest.mark.slow
    def test_dtype_and_range(self, loader: MnistLoader) -> None:
        images, labels = loader.load("train", n_per_class=5)
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    @pytest.mark.slow
    def test_n_per_class(self, loader: MnistLoader) -> None:
        images, labels = loader.load("train", n_per_class=10)
        assert len(images) == 100
        for c in range(10):
            assert (labels == c).sum().item() == 10

    @pytest.mark.slow
    def test_classes_filter(self, loader: MnistLoader) -> None:
        images, labels = loader.load("train", n_per_class=10, classes=[0, 1, 2])
        assert len(images) == 30
        unique = labels.unique().tolist()
        assert sorted(unique) == [0, 1, 2]

    @pytest.mark.slow
    def test_test_split(self, loader: MnistLoader) -> None:
        images, labels = loader.load("test", n_per_class=5)
        assert len(images) == 50

    @pytest.mark.slow
    def test_deterministic(self, loader: MnistLoader) -> None:
        imgs1, lab1 = loader.load("train", n_per_class=5)
        loader2 = MnistLoader(data_root="data/", target_size=64, seed=42)
        imgs2, lab2 = loader2.load("train", n_per_class=5)
        assert torch.equal(imgs1, imgs2)
        assert torch.equal(lab1, lab2)
