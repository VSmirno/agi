"""Shared pytest fixtures for SNKS tests."""

import pytest
import torch

from snks.device import get_device
from snks.daf.types import DafConfig


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the best available device for testing."""
    return get_device("auto")


@pytest.fixture
def cpu_device() -> torch.device:
    """Force CPU device (for deterministic tests)."""
    return torch.device("cpu")


@pytest.fixture
def small_config() -> DafConfig:
    """Small DAF config for fast tests."""
    return DafConfig(
        num_nodes=1000,
        state_dim=8,
        avg_degree=20,
        dt=0.0001,
        noise_sigma=0.01,
        oscillator_model="kuramoto",
        device="cpu",
    )


@pytest.fixture
def tiny_config() -> DafConfig:
    """Tiny DAF config for unit tests."""
    return DafConfig(
        num_nodes=100,
        state_dim=8,
        avg_degree=10,
        dt=0.0001,
        noise_sigma=0.005,
        oscillator_model="kuramoto",
        device="cpu",
    )
