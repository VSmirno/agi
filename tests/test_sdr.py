"""Tests for SDR functions (Этап 2)."""

import torch
import pytest

from snks.encoder.sdr import kwta, sdr_overlap, batch_overlap_matrix


class TestKwta:
    """k-Winner-Take-All sparse coding."""

    def test_output_shape(self) -> None:
        """Output shape matches input."""
        x = torch.randn(4096)
        sdr = kwta(x, k=164)
        assert sdr.shape == (4096,)

    def test_output_binary(self) -> None:
        """Output contains only 0s and 1s."""
        x = torch.randn(4096)
        sdr = kwta(x, k=164)
        assert set(sdr.unique().tolist()).issubset({0.0, 1.0})

    def test_exactly_k_active(self) -> None:
        """Exactly k bits are active."""
        x = torch.randn(4096)
        sdr = kwta(x, k=164)
        assert sdr.sum().item() == 164

    def test_top_k_selected(self) -> None:
        """Active bits correspond to top-k values."""
        x = torch.randn(4096)
        k = 164
        sdr = kwta(x, k=k)
        topk_indices = torch.topk(x, k).indices
        assert sdr[topk_indices].sum().item() == k

    def test_batched(self) -> None:
        """Works with batch dimension."""
        x = torch.randn(5, 4096)
        sdr = kwta(x, k=164)
        assert sdr.shape == (5, 4096)
        for i in range(5):
            assert sdr[i].sum().item() == 164

    def test_dtype_float32(self) -> None:
        """Output is float32."""
        x = torch.randn(4096)
        sdr = kwta(x, k=164)
        assert sdr.dtype == torch.float32


class TestSdrOverlap:
    """SDR overlap metric."""

    def test_identical_overlap_one(self) -> None:
        """Identical SDRs → overlap = 1.0."""
        sdr = torch.zeros(4096)
        sdr[:164] = 1.0
        assert sdr_overlap(sdr, sdr, k=164) == pytest.approx(1.0)

    def test_disjoint_overlap_zero(self) -> None:
        """Disjoint SDRs → overlap = 0.0."""
        a = torch.zeros(4096)
        b = torch.zeros(4096)
        a[:164] = 1.0
        b[164:328] = 1.0
        assert sdr_overlap(a, b, k=164) == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        """50% shared bits → overlap = 0.5."""
        a = torch.zeros(4096)
        b = torch.zeros(4096)
        a[:164] = 1.0
        b[82:246] = 1.0  # 82 shared bits out of 164
        assert sdr_overlap(a, b, k=164) == pytest.approx(82 / 164, abs=1e-5)

    def test_symmetric(self) -> None:
        """overlap(a, b) == overlap(b, a)."""
        a = torch.zeros(4096)
        b = torch.zeros(4096)
        a[:164] = 1.0
        b[50:214] = 1.0
        assert sdr_overlap(a, b, k=164) == pytest.approx(sdr_overlap(b, a, k=164))


class TestBatchOverlapMatrix:
    """Batch overlap matrix computation."""

    def test_shape(self) -> None:
        """(N, D) → (N, N) matrix."""
        sdrs = torch.zeros(10, 4096)
        for i in range(10):
            sdrs[i, i * 100 : i * 100 + 164] = 1.0
        mat = batch_overlap_matrix(sdrs, k=164)
        assert mat.shape == (10, 10)

    def test_diagonal_ones(self) -> None:
        """Diagonal elements = 1.0."""
        sdrs = torch.zeros(5, 4096)
        for i in range(5):
            sdrs[i, i * 164 : (i + 1) * 164] = 1.0
        mat = batch_overlap_matrix(sdrs, k=164)
        assert torch.allclose(mat.diag(), torch.ones(5))

    def test_symmetric_matrix(self) -> None:
        """Matrix is symmetric."""
        sdrs = torch.zeros(5, 4096)
        for i in range(5):
            sdrs[i, i * 50 : i * 50 + 164] = 1.0
        mat = batch_overlap_matrix(sdrs, k=164)
        assert torch.allclose(mat, mat.T)

    def test_values_in_range(self) -> None:
        """All values in [0, 1]."""
        sdrs = torch.zeros(5, 4096)
        for i in range(5):
            sdrs[i, i * 100 : i * 100 + 164] = 1.0
        mat = batch_overlap_matrix(sdrs, k=164)
        assert (mat >= 0).all() and (mat <= 1).all()
