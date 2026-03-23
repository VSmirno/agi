"""Tests for VisualEncoder (Этап 2)."""

import torch
import pytest

from snks.daf.types import EncoderConfig
from snks.encoder.encoder import VisualEncoder


@pytest.fixture
def config() -> EncoderConfig:
    return EncoderConfig()


@pytest.fixture
def encoder(config: EncoderConfig) -> VisualEncoder:
    return VisualEncoder(config)


class TestVisualEncoderEncode:
    """VisualEncoder.encode() — image to SDR."""

    def test_output_shape(self, encoder: VisualEncoder) -> None:
        """(64,64) → (4096,) SDR."""
        img = torch.rand(64, 64)
        sdr = encoder.encode(img)
        assert sdr.shape == (4096,)

    def test_output_binary(self, encoder: VisualEncoder) -> None:
        """SDR is binary."""
        img = torch.rand(64, 64)
        sdr = encoder.encode(img)
        assert set(sdr.unique().tolist()).issubset({0.0, 1.0})

    def test_output_sparsity(self, encoder: VisualEncoder) -> None:
        """SDR has exactly k=164 active bits."""
        img = torch.rand(64, 64)
        sdr = encoder.encode(img)
        assert sdr.sum().item() == 164

    def test_batched_encode(self, encoder: VisualEncoder) -> None:
        """Batch of images → batch of SDRs."""
        imgs = torch.rand(5, 64, 64)
        sdrs = encoder.encode(imgs)
        assert sdrs.shape == (5, 4096)
        for i in range(5):
            assert sdrs[i].sum().item() == 164

    def test_deterministic(self, encoder: VisualEncoder) -> None:
        """Same input → same SDR (no randomness in encode)."""
        img = torch.rand(64, 64)
        sdr1 = encoder.encode(img)
        sdr2 = encoder.encode(img)
        assert torch.equal(sdr1, sdr2)

    def test_different_images_different_sdrs(self, encoder: VisualEncoder) -> None:
        """Different inputs → different SDRs."""
        img1 = torch.zeros(64, 64)
        img1[10:30, 10:30] = 1.0  # square
        img2 = torch.zeros(64, 64)
        img2[30:50, 30:50] = 1.0  # different position
        sdr1 = encoder.encode(img1)
        sdr2 = encoder.encode(img2)
        assert not torch.equal(sdr1, sdr2)


class TestVisualEncoderSdrToCurrents:
    """VisualEncoder.sdr_to_currents() — SDR to DAF currents."""

    def test_output_shape(self, encoder: VisualEncoder) -> None:
        """SDR → (N, 8) currents."""
        sdr = torch.zeros(4096)
        sdr[:164] = 1.0
        currents = encoder.sdr_to_currents(sdr, n_nodes=10000)
        assert currents.shape == (10000, 8)

    def test_only_channel_0(self, encoder: VisualEncoder) -> None:
        """Currents only in channel 0, rest zeros."""
        sdr = torch.zeros(4096)
        sdr[:164] = 1.0
        currents = encoder.sdr_to_currents(sdr, n_nodes=10000)
        assert (currents[:, 1:] == 0).all()

    def test_current_strength(self, encoder: VisualEncoder) -> None:
        """Active nodes get current_strength=1.0."""
        sdr = torch.ones(4096)  # all active
        currents = encoder.sdr_to_currents(sdr, n_nodes=10000)
        assert (currents[:, 0] == 1.0).all()

    def test_zero_sdr_zero_currents(self, encoder: VisualEncoder) -> None:
        """Zero SDR → zero currents."""
        sdr = torch.zeros(4096)
        currents = encoder.sdr_to_currents(sdr, n_nodes=10000)
        assert (currents == 0).all()

    def test_mapping_coverage(self, encoder: VisualEncoder) -> None:
        """All nodes should be mapped to some SDR bit."""
        sdr = torch.ones(4096)  # all bits active
        currents = encoder.sdr_to_currents(sdr, n_nodes=10000)
        # All nodes should receive current
        assert (currents[:, 0] > 0).all()

    def test_sdr_to_currents_coverage(self, encoder: VisualEncoder) -> None:
        """Hash-based mapping covers all SDR bits for any N."""
        n_nodes = 10000
        sdr = torch.zeros(4096)
        sdr[0] = 1.0  # only first bit active
        currents = encoder.sdr_to_currents(sdr, n_nodes=n_nodes)
        # Some nodes should have current (hash distributes bit 0 to ~N/sdr_size nodes)
        n_active = (currents[:, 0] > 0).sum().item()
        assert n_active > 0
        assert n_active <= n_nodes
        # With 164 active bits, ~4% of nodes should be stimulated
        sdr_full = torch.zeros(4096)
        sdr_full[:164] = 1.0
        currents_full = encoder.sdr_to_currents(sdr_full, n_nodes=n_nodes)
        n_stim = (currents_full[:, 0] > 0).sum().item()
        assert n_stim > n_nodes * 0.02  # at least 2% coverage


class TestVisualEncoderGate:
    """Gate test: SDR quality on oriented gratings."""

    @pytest.mark.slow
    def test_within_class_overlap_high(self) -> None:
        """Mean within-class SDR overlap > 0.3."""
        from snks.data.stimuli import GratingGenerator
        from snks.encoder.sdr import batch_overlap_matrix

        config = EncoderConfig()
        encoder = VisualEncoder(config)
        gen = GratingGenerator(image_size=64, seed=42)

        images, labels = gen.generate_all(n_variations=20)
        sdrs = encoder.encode(images)

        k = round(config.sdr_size * config.sdr_sparsity)
        mat = batch_overlap_matrix(sdrs, k=k)

        within_overlaps = []
        for c in range(10):
            mask = labels == c
            class_mat = mat[mask][:, mask]
            n = class_mat.shape[0]
            off_diag = class_mat[~torch.eye(n, dtype=torch.bool)]
            within_overlaps.append(off_diag.mean().item())

        mean_within = sum(within_overlaps) / len(within_overlaps)
        assert mean_within > 0.3, f"Within-class overlap {mean_within:.3f} <= 0.3"

    @pytest.mark.slow
    def test_between_class_overlap_low(self) -> None:
        """Mean between-class SDR overlap < 0.1."""
        from snks.data.stimuli import GratingGenerator
        from snks.encoder.sdr import batch_overlap_matrix

        config = EncoderConfig()
        encoder = VisualEncoder(config)
        gen = GratingGenerator(image_size=64, seed=42)

        images, labels = gen.generate_all(n_variations=20)
        sdrs = encoder.encode(images)

        k = round(config.sdr_size * config.sdr_sparsity)
        mat = batch_overlap_matrix(sdrs, k=k)

        between_overlaps = []
        for c1 in range(10):
            for c2 in range(c1 + 1, 10):
                mask1 = labels == c1
                mask2 = labels == c2
                cross_mat = mat[mask1][:, mask2]
                between_overlaps.append(cross_mat.mean().item())

        mean_between = sum(between_overlaps) / len(between_overlaps)
        assert mean_between < 0.1, f"Between-class overlap {mean_between:.3f} >= 0.1"
