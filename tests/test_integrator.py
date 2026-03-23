"""Tests for Euler-Maruyama integrator."""

import torch

from snks.daf.integrator import euler_maruyama_step, integrate_n_steps


class TestEulerMaruyamaStep:
    def test_shape(self, device):
        N = 100
        states = torch.zeros(N, 8, device=device)
        result = euler_maruyama_step(
            states, lambda s: torch.zeros_like(s), dt=0.001, noise_sigma=0.0
        )
        assert result.shape == (N, 8)

    def test_zero_noise_deterministic(self, device):
        N = 50
        s1 = torch.ones(N, 8, device=device) * 0.5
        s2 = s1.clone()
        fn = lambda s: torch.ones_like(s)
        euler_maruyama_step(s1, fn, 0.01, 0.0)
        euler_maruyama_step(s2, fn, 0.01, 0.0)
        assert torch.allclose(s1, s2)

    def test_drift_applied(self, device):
        N = 10
        states = torch.zeros(N, 8, device=device)
        # unit drift, dt=1.0 → states should become 1.0
        euler_maruyama_step(states, lambda s: torch.ones_like(s), dt=1.0, noise_sigma=0.0)
        assert torch.allclose(states, torch.ones(N, 8, device=device))

    def test_noise_adds_variance(self, device):
        N = 10000
        states = torch.zeros(N, 8, device=device)
        euler_maruyama_step(states, lambda s: torch.zeros_like(s), dt=0.01, noise_sigma=1.0)
        # With noise, std should be ~sqrt(dt)*sigma = 0.1
        actual_std = states[:, 0].std().item()
        assert 0.05 < actual_std < 0.2

    def test_in_place_modification(self, device):
        states = torch.zeros(5, 8, device=device)
        original_ptr = states.data_ptr()
        euler_maruyama_step(states, lambda s: torch.ones_like(s), dt=0.1, noise_sigma=0.0)
        assert states.data_ptr() == original_ptr  # same tensor
        assert states.abs().sum() > 0  # modified

    def test_noise_buf_reused(self, device):
        N = 50
        states = torch.zeros(N, 8, device=device)
        noise_buf = torch.empty(N, 8, device=device)
        euler_maruyama_step(
            states, lambda s: torch.zeros_like(s), dt=0.01, noise_sigma=1.0, noise_buf=noise_buf
        )
        # States should be nonzero due to noise
        assert states.abs().sum() > 0


class TestIntegrateNSteps:
    def test_fired_history_shape(self, device):
        N, T = 100, 50
        states = torch.randn(N, 8, device=device)
        _, hist = integrate_n_steps(
            states, lambda s: torch.zeros_like(s), T, 0.001, 0.01
        )
        assert hist.shape == (T, N)
        assert hist.dtype == torch.bool

    def test_states_evolve(self, device):
        N = 100
        states = torch.zeros(N, 8, device=device)
        states_before = states.clone()
        states, _ = integrate_n_steps(
            states, lambda s: torch.ones_like(s), 10, 0.001, 0.0
        )
        # After 10 steps with unit drift: states ≈ 10 * 0.001 = 0.01
        assert torch.allclose(states, torch.ones(N, 8, device=device) * 0.01, atol=1e-5)

    def test_spike_detection(self, device):
        N = 10
        # Start above threshold → should always fire
        states = torch.ones(N, 8, device=device) * 2.0
        _, hist = integrate_n_steps(
            states, lambda s: torch.zeros_like(s), 5, 0.001, 0.0,
            spike_threshold_col=0, spike_threshold_val=0.5,
        )
        assert hist.all()

    def test_no_spikes_below_threshold(self, device):
        N = 10
        states = torch.zeros(N, 8, device=device)  # all zeros < 0.5
        _, hist = integrate_n_steps(
            states, lambda s: torch.zeros_like(s), 5, 0.001, 0.0,
            spike_threshold_col=0, spike_threshold_val=0.5,
        )
        assert not hist.any()
