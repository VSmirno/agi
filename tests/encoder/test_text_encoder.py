"""Tests for TextEncoder."""

import pytest
import torch

from snks.daf.types import EncoderConfig
from snks.encoder.text_encoder import TextEncoder


@pytest.fixture(scope="module")
def encoder():
    config = EncoderConfig()  # sdr_size=4096, sdr_sparsity=0.04
    return TextEncoder(config)


def test_encode_returns_binary_sdr(encoder):
    sdr = encoder.encode("hello world")
    assert sdr.shape == (4096,)
    assert sdr.dtype == torch.float32
    assert sdr.min().item() >= 0.0
    assert sdr.max().item() <= 1.0
    n_active = int(sdr.sum().item())
    assert abs(n_active - 164) <= 2, f"Expected ~164 active bits, got {n_active}"


def test_similar_texts_higher_overlap(encoder):
    # Two very similar sentences vs one from a completely different domain
    sdr_a = encoder.encode("The cat is resting on the sofa")
    sdr_b = encoder.encode("A cat sleeps on the couch")
    sdr_c = encoder.encode("The CPU executes machine learning code on the GPU")

    def jaccard(a, b):
        intersection = (a * b).sum().item()
        union = ((a + b) > 0).float().sum().item()
        return intersection / union if union > 0 else 0.0

    overlap_similar = jaccard(sdr_a, sdr_b)
    overlap_different = jaccard(sdr_a, sdr_c)
    assert overlap_similar > overlap_different, (
        f"Similar texts overlap ({overlap_similar:.4f}) should exceed "
        f"different topics overlap ({overlap_different:.4f})"
    )


def test_sdr_to_currents_shape(encoder):
    sdr = encoder.encode("test sentence")
    currents = encoder.sdr_to_currents(sdr, n_nodes=1000)
    assert currents.shape == (1000, 8)
    assert currents.dtype == torch.float32


def test_sdr_to_currents_values(encoder):
    sdr = encoder.encode("test sentence")
    currents = encoder.sdr_to_currents(sdr, n_nodes=4096)
    # Channel 0 should have non-zero values where SDR is active
    assert currents[:, 0].sum().item() > 0
    # Channels 1-7 should be zero
    assert currents[:, 1:].sum().item() == 0.0


def test_deterministic(encoder):
    sdr1 = encoder.encode("test")
    sdr2 = encoder.encode("test")
    assert torch.equal(sdr1, sdr2)


def test_different_texts_different_sdrs(encoder):
    sdr1 = encoder.encode("the sky is blue")
    sdr2 = encoder.encode("machine learning algorithms")
    assert not torch.equal(sdr1, sdr2)
