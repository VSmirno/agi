"""Tests for oscillator dynamics."""

import math
import torch

from snks.daf.oscillator import kuramoto_derivatives, fhn_derivatives, init_states
from snks.daf.types import DafConfig


class TestKuramoto:
    def test_derivatives_shape(self, device, tiny_config):
        N = tiny_config.num_nodes
        states = init_states(N, 8, "kuramoto", device)
        d = kuramoto_derivatives(states, tiny_config)
        assert d.shape == (N, 8)

    def test_dtheta_equals_omega(self, device, tiny_config):
        N = tiny_config.num_nodes
        states = init_states(N, 8, "kuramoto", device)
        d = kuramoto_derivatives(states, tiny_config)
        assert torch.allclose(d[:, 0], states[:, 2])

    def test_other_channels_zero(self, device, tiny_config):
        N = tiny_config.num_nodes
        states = init_states(N, 8, "kuramoto", device)
        d = kuramoto_derivatives(states, tiny_config)
        assert d[:, 1:].abs().max() == 0.0


class TestFHN:
    def test_derivatives_shape(self, device):
        cfg = DafConfig(num_nodes=100, oscillator_model="fhn")
        states = init_states(100, 8, "fhn", device)
        d = fhn_derivatives(states, cfg)
        assert d.shape == (100, 8)

    def test_no_explosion(self, device):
        """FHN should oscillate, not explode."""
        cfg = DafConfig(num_nodes=50, oscillator_model="fhn")
        states = init_states(50, 8, "fhn", device)
        dt = 0.001
        for _ in range(2000):
            d = fhn_derivatives(states, cfg)
            states = states + dt * d
        assert states[:, 0].abs().max() < 10.0

    def test_dv_and_dw_nonzero(self, device):
        cfg = DafConfig(num_nodes=50, oscillator_model="fhn")
        states = init_states(50, 8, "fhn", device)
        states[:, 0] = 0.5
        d = fhn_derivatives(states, cfg)
        assert d[:, 0].abs().sum() > 0
        assert d[:, 4].abs().sum() > 0


class TestInitStates:
    def test_kuramoto_phases_in_range(self, device):
        s = init_states(1000, 8, "kuramoto", device)
        assert s[:, 0].min() >= 0.0
        assert s[:, 0].max() <= 2.0 * math.pi

    def test_kuramoto_frequencies_clamped(self, device):
        s = init_states(10000, 8, "kuramoto", device)
        assert s[:, 2].min() >= -5.0
        assert s[:, 2].max() <= 5.0

    def test_fhn_v_near_zero(self, device):
        s = init_states(1000, 8, "fhn", device)
        assert s[:, 0].abs().mean() < 0.5

    def test_thresholds_set(self, device):
        s = init_states(100, 8, "kuramoto", device)
        assert (s[:, 3] == 0.5).all()

    def test_unknown_model_raises(self):
        import pytest
        with pytest.raises(ValueError):
            init_states(10, 8, "unknown", torch.device("cpu"))
