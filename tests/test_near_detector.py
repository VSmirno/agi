"""Unit tests for NearDetector (Stage 67)."""

import pytest
import torch

from snks.encoder.cnn_encoder import CNNEncoder
from snks.encoder.near_detector import NearDetector
from snks.agent.decode_head import NEAR_CLASSES


@pytest.fixture
def encoder() -> CNNEncoder:
    """Untrained CNNEncoder with correct n_near_classes."""
    return CNNEncoder(n_near_classes=len(NEAR_CLASSES))


@pytest.fixture
def detector(encoder: CNNEncoder) -> NearDetector:
    return NearDetector(encoder)


def test_detect_returns_string_from_near_classes(detector: NearDetector) -> None:
    """detect() must return a string from NEAR_CLASSES."""
    pixels = torch.rand(3, 64, 64)
    result = detector.detect(pixels)
    assert isinstance(result, str)
    assert result in NEAR_CLASSES, f"'{result}' not in NEAR_CLASSES"


def test_detect_random_noise_no_crash(detector: NearDetector) -> None:
    """detect() must not raise on random noise input."""
    for _ in range(5):
        pixels = torch.rand(3, 64, 64)
        result = detector.detect(pixels)
        assert result in NEAR_CLASSES


def test_detect_zeros_returns_valid(detector: NearDetector) -> None:
    """detect() must handle all-zeros pixels gracefully."""
    pixels = torch.zeros(3, 64, 64)
    result = detector.detect(pixels)
    assert result in NEAR_CLASSES


def test_detect_ones_returns_valid(detector: NearDetector) -> None:
    """detect() must handle all-ones pixels gracefully."""
    pixels = torch.ones(3, 64, 64)
    result = detector.detect(pixels)
    assert result in NEAR_CLASSES


def test_near_detector_assert_wrong_n_classes() -> None:
    """NearDetector must raise if encoder n_near_classes != len(NEAR_CLASSES)."""
    bad_encoder = CNNEncoder(n_near_classes=len(NEAR_CLASSES) + 1)
    with pytest.raises(AssertionError, match="n_near_classes"):
        NearDetector(bad_encoder)


def test_near_classes_includes_empty() -> None:
    """NEAR_CLASSES must include 'empty' as a valid fallback."""
    assert "empty" in NEAR_CLASSES


def test_detect_output_is_deterministic(detector: NearDetector) -> None:
    """detect() is deterministic for same input (no dropout in eval mode)."""
    pixels = torch.rand(3, 64, 64)
    result1 = detector.detect(pixels)
    result2 = detector.detect(pixels)
    assert result1 == result2
