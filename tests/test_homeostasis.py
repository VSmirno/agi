"""Tests for homeostatic plasticity."""

import torch

from snks.daf.homeostasis import Homeostasis
from snks.daf.types import DafConfig


class TestHomeostasis:
    def test_raises_threshold_for_overactive(self, device):
        N, T = 100, 50
        cfg = DafConfig(num_nodes=N, homeostasis_target=0.1)
        h = Homeostasis(cfg, N, device)

        states = torch.zeros(N, 8, device=device)
        states[:, 3] = 0.5

        # First 50 nodes always fire (rate=1.0 >> target=0.1)
        fired = torch.zeros(T, N, dtype=torch.bool, device=device)
        fired[:, :50] = True

        threshold_before = states[:50, 3].clone()
        h.update(fired, states)
        assert (states[:50, 3] > threshold_before).all()

    def test_lowers_threshold_for_silent(self, device):
        N, T = 100, 50
        cfg = DafConfig(num_nodes=N, homeostasis_target=0.1)
        h = Homeostasis(cfg, N, device)

        states = torch.zeros(N, 8, device=device)
        states[:, 3] = 1.0

        # Nodes 50..99 never fire (rate=0 < target)
        fired = torch.zeros(T, N, dtype=torch.bool, device=device)
        threshold_before = states[50:, 3].clone()
        h.update(fired, states)
        assert (states[50:, 3] < threshold_before).all()

    def test_thresholds_stay_bounded(self, device):
        N = 50
        cfg = DafConfig(num_nodes=N, homeostasis_target=0.1)
        h = Homeostasis(cfg, N, device)

        states = torch.zeros(N, 8, device=device)
        states[:, 3] = 0.5

        # Extreme firing patterns for many iterations
        for _ in range(100):
            fired = torch.ones(20, N, dtype=torch.bool, device=device)
            h.update(fired, states)

        assert states[:, 3].min() >= 0.01 - 1e-6
        assert states[:, 3].max() <= 5.0 + 1e-6

    def test_get_firing_rates(self, device):
        N = 50
        cfg = DafConfig(num_nodes=N, homeostasis_target=0.1)
        h = Homeostasis(cfg, N, device)
        rates = h.get_firing_rates()
        assert rates.shape == (N,)
        # Initially set to target rate
        assert torch.allclose(rates, torch.full((N,), 0.1, device=device))

    def test_at_target_no_change(self, device):
        """If firing rate == target, threshold should barely change."""
        N = 50
        cfg = DafConfig(num_nodes=N, homeostasis_target=0.5)
        h = Homeostasis(cfg, N, device)

        states = torch.zeros(N, 8, device=device)
        states[:, 3] = 1.0

        # Exactly 50% firing rate
        fired = torch.zeros(20, N, dtype=torch.bool, device=device)
        fired[:10, :] = True

        threshold_before = states[:, 3].clone()
        h.update(fired, states)
        # Change should be very small
        delta = (states[:, 3] - threshold_before).abs().max().item()
        assert delta < 0.01
