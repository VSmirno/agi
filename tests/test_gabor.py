"""Tests for GaborBank (Этап 2)."""

import torch
import pytest
import numpy as np

from snks.daf.types import EncoderConfig
from snks.encoder.gabor import GaborBank


@pytest.fixture
def config() -> EncoderConfig:
    return EncoderConfig()


@pytest.fixture
def bank(config: EncoderConfig) -> GaborBank:
    return GaborBank(config)


class TestGaborBankInit:
    """GaborBank initialization and structure."""

    def test_n_filters(self, bank: GaborBank) -> None:
        """128 filters = 4 scales × 8 orientations × 4 phases."""
        assert bank.n_filters == 128

    def test_conv_weight_shape(self, bank: GaborBank) -> None:
        """Conv2d weight shape: (128, 1, 19, 19)."""
        assert bank.conv.weight.shape == (128, 1, 19, 19)

    def test_conv_no_grad(self, bank: GaborBank) -> None:
        """Gabor filters are frozen (requires_grad=False)."""
        assert not bank.conv.weight.requires_grad

    def test_conv_no_bias(self, bank: GaborBank) -> None:
        """Conv2d has no bias."""
        assert bank.conv.bias is None

    def test_conv_padding(self, bank: GaborBank) -> None:
        """Padding = 9 to preserve spatial dimensions with kernel_size=19."""
        assert bank.conv.padding == (9, 9)


class TestGaborBankForward:
    """GaborBank forward pass."""

    def test_output_shape(self, bank: GaborBank) -> None:
        """(B,1,64,64) → (B,128,64,64)."""
        x = torch.rand(2, 1, 64, 64)
        y = bank(x)
        assert y.shape == (2, 128, 64, 64)

    def test_output_nonnegative(self, bank: GaborBank) -> None:
        """abs() activation → all outputs >= 0."""
        x = torch.rand(1, 1, 64, 64)
        y = bank(x)
        assert (y >= 0).all()

    def test_output_dtype(self, bank: GaborBank) -> None:
        """Output is float32."""
        x = torch.rand(1, 1, 64, 64)
        y = bank(x)
        assert y.dtype == torch.float32

    def test_different_inputs_different_outputs(self, bank: GaborBank) -> None:
        """Different images produce different feature maps."""
        x1 = torch.zeros(1, 1, 64, 64)
        x2 = torch.ones(1, 1, 64, 64)
        y1 = bank(x1)
        y2 = bank(x2)
        assert not torch.allclose(y1, y2)

    def test_zero_input_zero_output(self, bank: GaborBank) -> None:
        """Zero-mean kernels + zero input → zero output."""
        x = torch.zeros(1, 1, 64, 64)
        y = bank(x)
        assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


class TestGaborKernelProperties:
    """Properties of individual Gabor kernels."""

    def test_kernels_zero_mean(self, bank: GaborBank) -> None:
        """Each filter kernel should be approximately zero-mean."""
        w = bank.conv.weight.data  # (128, 1, 19, 19)
        means = w.mean(dim=(1, 2, 3))
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_kernels_l2_normalized(self, bank: GaborBank) -> None:
        """Each kernel is L2-normalized."""
        w = bank.conv.weight.data  # (128, 1, 19, 19)
        norms = w.reshape(128, -1).norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_different_orientations(self, bank: GaborBank) -> None:
        """Filters with different orientations should differ."""
        w = bank.conv.weight.data
        # First two filters differ in orientation (scale=0, phase=0, θ=0 vs θ=1)
        assert not torch.allclose(w[0], w[1])
