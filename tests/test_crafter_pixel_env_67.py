"""Integration tests for CrafterPixelEnv Stage 67 API.

Verifies:
- New step/reset/observe signatures (Stage 67)
- _to_symbolic and _detect_nearby are removed
- info["inventory"] is present and correct type
"""

import pytest
import numpy as np

pytest.importorskip("crafter", reason="crafter package not installed")

from snks.agent.crafter_pixel_env import CrafterPixelEnv


@pytest.fixture
def env() -> CrafterPixelEnv:
    return CrafterPixelEnv(seed=42)


def test_reset_returns_pixels_and_info(env: CrafterPixelEnv) -> None:
    """reset() must return (pixels, info) — NOT (pixels, symbolic_obs)."""
    result = env.reset()
    assert len(result) == 2
    pixels, info = result
    assert isinstance(pixels, np.ndarray)
    assert isinstance(info, dict)


def test_reset_pixels_shape(env: CrafterPixelEnv) -> None:
    pixels, _ = env.reset()
    assert pixels.shape == (3, 64, 64)
    assert pixels.dtype == np.float32
    assert pixels.min() >= 0.0
    assert pixels.max() <= 1.0


def test_reset_info_has_inventory(env: CrafterPixelEnv) -> None:
    _, info = env.reset()
    assert "inventory" in info
    assert isinstance(info["inventory"], dict)


def test_step_returns_four_tuple(env: CrafterPixelEnv) -> None:
    """step() must return (pixels, reward, done, info)."""
    env.reset()
    result = env.step(0)
    assert len(result) == 4
    pixels, reward, done, info = result
    assert isinstance(pixels, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_pixels_shape(env: CrafterPixelEnv) -> None:
    env.reset()
    pixels, _, _, _ = env.step(0)
    assert pixels.shape == (3, 64, 64)
    assert pixels.dtype == np.float32


def test_step_info_has_inventory(env: CrafterPixelEnv) -> None:
    env.reset()
    _, _, _, info = env.step(0)
    assert "inventory" in info
    assert isinstance(info["inventory"], dict)


def test_step_action_string(env: CrafterPixelEnv) -> None:
    """step() must accept action name strings."""
    env.reset()
    pixels, reward, done, info = env.step("noop")
    assert pixels.shape == (3, 64, 64)


def test_observe_returns_pixels_and_info(env: CrafterPixelEnv) -> None:
    env.reset()
    result = env.observe()
    assert len(result) == 2
    pixels, info = result
    assert pixels.shape == (3, 64, 64)
    assert isinstance(info, dict)


def test_no_to_symbolic_method(env: CrafterPixelEnv) -> None:
    """_to_symbolic must not exist (Stage 67 removed it)."""
    assert not hasattr(env, "_to_symbolic"), (
        "_to_symbolic still exists — Stage 67 migration incomplete"
    )


def test_no_detect_nearby_method(env: CrafterPixelEnv) -> None:
    """_detect_nearby must not exist (Stage 67 removed it)."""
    assert not hasattr(env, "_detect_nearby"), (
        "_detect_nearby still exists — Stage 67 migration incomplete"
    )
