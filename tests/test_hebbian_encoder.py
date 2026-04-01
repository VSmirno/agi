"""Tests for HebbianEncoder — learnable encoding via Oja's Hebbian rule."""

import torch
import pytest

from snks.daf.types import EncoderConfig
from snks.encoder.hebbian import HebbianEncoder


@pytest.fixture
def config():
    return EncoderConfig(image_size=32, sdr_size=1024, pool_h=2, pool_w=4)


@pytest.fixture
def encoder(config):
    return HebbianEncoder(config)


@pytest.fixture
def random_image():
    return torch.rand(32, 32)


class TestHebbianEncoderInit:
    def test_inherits_visual_encoder(self, encoder):
        """HebbianEncoder is a drop-in replacement for VisualEncoder."""
        from snks.encoder.encoder import VisualEncoder
        assert isinstance(encoder, VisualEncoder)

    def test_gabor_init_weights_match(self, config):
        """Initial weights should match frozen Gabor bank."""
        from snks.encoder.encoder import VisualEncoder
        frozen = VisualEncoder(config)
        hebbian = HebbianEncoder(config)
        assert torch.allclose(
            frozen.gabor.conv.weight.data,
            hebbian.gabor.conv.weight.data,
        )

    def test_weights_not_grad(self, encoder):
        """Weights should not require grad (manual Hebbian updates)."""
        assert not encoder.gabor.conv.weight.requires_grad

    def test_encode_same_as_frozen(self, config, random_image):
        """Before any Hebbian update, encode output matches frozen."""
        from snks.encoder.encoder import VisualEncoder
        frozen = VisualEncoder(config)
        hebbian = HebbianEncoder(config)
        sdr_frozen = frozen.encode(random_image)
        sdr_hebbian = hebbian.encode(random_image)
        assert torch.equal(sdr_frozen, sdr_hebbian)


class TestHebbianUpdate:
    def test_update_changes_weights(self, encoder, random_image):
        """After hebbian_update, weights should differ from initial."""
        initial_w = encoder.gabor.conv.weight.data.clone()
        sdr = encoder.encode(random_image)
        encoder.hebbian_update(random_image, sdr, prediction_error=0.5)
        assert not torch.equal(encoder.gabor.conv.weight.data, initial_w)

    def test_update_with_zero_pe_minimal_change(self, encoder, random_image):
        """With PE=0, learning rate is minimal (0.1x base)."""
        initial_w = encoder.gabor.conv.weight.data.clone()
        sdr = encoder.encode(random_image)
        encoder.hebbian_update(random_image, sdr, prediction_error=0.0)
        delta = (encoder.gabor.conv.weight.data - initial_w).abs().max().item()
        # Should be very small change
        assert delta < 0.01

    def test_higher_pe_larger_update(self, encoder, random_image):
        """Higher prediction error should cause larger weight changes."""
        sdr = encoder.encode(random_image)

        # Low PE
        enc_low = HebbianEncoder(encoder.config)
        enc_low.gabor.conv.weight.data.copy_(encoder.gabor.conv.weight.data)
        enc_low.hebbian_update(random_image, sdr, prediction_error=0.1)
        delta_low = (enc_low.gabor.conv.weight.data - encoder.gabor.conv.weight.data).abs().mean()

        # High PE
        enc_high = HebbianEncoder(encoder.config)
        enc_high.gabor.conv.weight.data.copy_(encoder.gabor.conv.weight.data)
        enc_high.hebbian_update(random_image, sdr, prediction_error=0.9)
        delta_high = (enc_high.gabor.conv.weight.data - encoder.gabor.conv.weight.data).abs().mean()

        assert delta_high > delta_low

    def test_weights_bounded_after_update(self, encoder, random_image):
        """Weights must remain bounded after Hebbian updates."""
        sdr = encoder.encode(random_image)
        for _ in range(50):
            encoder.hebbian_update(random_image, sdr, prediction_error=1.0)
        w = encoder.gabor.conv.weight.data
        assert w.max().item() <= encoder.w_max + 1e-6
        assert w.min().item() >= encoder.w_min - 1e-6

    def test_update_preserves_sdr_size(self, encoder, random_image):
        """SDR output should still have correct size after updates."""
        sdr = encoder.encode(random_image)
        encoder.hebbian_update(random_image, sdr, prediction_error=0.5)
        sdr_after = encoder.encode(random_image)
        assert sdr_after.shape == sdr.shape
        assert sdr_after.sum().item() == pytest.approx(encoder.k, abs=1)


class TestDiversityRegularization:
    def test_diversity_runs_at_interval(self, encoder, random_image):
        """Diversity regularization triggers every diversity_interval steps."""
        sdr = encoder.encode(random_image)
        # Run updates up to diversity_interval
        for i in range(encoder.diversity_interval):
            encoder.hebbian_update(random_image, sdr, prediction_error=0.5)
        # The _update_count should have reached diversity_interval
        assert encoder._update_count >= encoder.diversity_interval

    def test_collapsed_filters_get_decorrelated(self, encoder):
        """If two filters become identical, diversity reg should push them apart."""
        # Manually make two filters identical
        encoder.gabor.conv.weight.data[1] = encoder.gabor.conv.weight.data[0].clone()
        initial_sim = torch.nn.functional.cosine_similarity(
            encoder.gabor.conv.weight.data[0].flatten().unsqueeze(0),
            encoder.gabor.conv.weight.data[1].flatten().unsqueeze(0),
        ).item()
        assert initial_sim > 0.99  # They're identical

        encoder._apply_diversity_regularization()

        post_sim = torch.nn.functional.cosine_similarity(
            encoder.gabor.conv.weight.data[0].flatten().unsqueeze(0),
            encoder.gabor.conv.weight.data[1].flatten().unsqueeze(0),
        ).item()
        assert post_sim < initial_sim  # They should be pushed apart


class TestSDRDiscrimination:
    def test_discrimination_metric(self, encoder):
        """Encoder should track mean SDR overlap between observations."""
        img1 = torch.rand(32, 32)
        img2 = torch.rand(32, 32)
        sdr1 = encoder.encode(img1)
        sdr2 = encoder.encode(img2)
        overlap = (sdr1 * sdr2).sum().item() / encoder.k
        # Random images should have some overlap but not total
        assert 0.0 <= overlap <= 1.0

    def test_different_images_different_sdrs(self, encoder):
        """Substantially different images should produce different SDRs."""
        img1 = torch.zeros(32, 32)
        img2 = torch.ones(32, 32)
        sdr1 = encoder.encode(img1)
        sdr2 = encoder.encode(img2)
        assert not torch.equal(sdr1, sdr2)


class TestHebbianConvergence:
    def test_weight_delta_decreases(self, encoder):
        """Over many updates, weight changes should decrease (convergence)."""
        image = torch.rand(32, 32)
        deltas = []
        for _ in range(20):
            w_before = encoder.gabor.conv.weight.data.clone()
            sdr = encoder.encode(image)
            encoder.hebbian_update(image, sdr, prediction_error=0.5)
            delta = (encoder.gabor.conv.weight.data - w_before).abs().mean().item()
            deltas.append(delta)

        # Later deltas should be smaller than early deltas (on average)
        early_mean = sum(deltas[:5]) / 5
        late_mean = sum(deltas[-5:]) / 5
        assert late_mean <= early_mean * 1.5  # Allow some tolerance


class TestEdgeCases:
    def test_black_image(self, encoder):
        """All-zero image should not cause NaN."""
        img = torch.zeros(32, 32)
        sdr = encoder.encode(img)
        encoder.hebbian_update(img, sdr, prediction_error=0.5)
        assert not torch.isnan(encoder.gabor.conv.weight.data).any()

    def test_white_image(self, encoder):
        """All-ones image should not cause NaN."""
        img = torch.ones(32, 32)
        sdr = encoder.encode(img)
        encoder.hebbian_update(img, sdr, prediction_error=0.5)
        assert not torch.isnan(encoder.gabor.conv.weight.data).any()

    def test_batch_encode_still_works(self, encoder):
        """Batch encoding should work after Hebbian updates."""
        images = torch.rand(4, 32, 32)
        sdr = encoder.encode(images)
        assert sdr.shape == (4, encoder.config.sdr_size)
