"""Gate test: Kuramoto synchronization.

1000 oscillators should synchronize (order parameter r > 0.7).
"""

import torch
import pytest

from snks.daf.engine import DafEngine
from snks.daf.types import DafConfig


def compute_order_parameter(states: torch.Tensor) -> float:
    """Kuramoto order parameter r ∈ [0, 1].

    r = |1/N * Σ exp(i*θ_j)| = sqrt(mean(cos)^2 + mean(sin)^2)
    0 = full desynchronization, 1 = full synchronization.
    """
    theta = states[:, 0]
    r = torch.sqrt(theta.cos().mean() ** 2 + theta.sin().mean() ** 2)
    return float(r)


class TestKuramotoSync:
    def test_initial_desynchronized(self, device):
        """Random initial phases should give low r."""
        cfg = DafConfig(
            num_nodes=1000, avg_degree=20, oscillator_model="kuramoto",
            coupling_strength=1.0, noise_sigma=0.005, dt=0.001,
            device=str(device),
        )
        engine = DafEngine(cfg)
        r = compute_order_parameter(engine.states)
        assert r < 0.3, f"Initial r should be low, got {r:.3f}"

    @pytest.mark.slow
    def test_gate_synchronization(self, device):
        """GATE: 1K Kuramoto oscillators synchronize to r > 0.7.

        K_c for Normal(0, σ) frequencies = 2σ/(π·g(0)) = 2σ·√(2π)/π ≈ 1.6σ.
        We use σ=0.3 in init_states (via clamp of Normal(0,1)*0.3),
        so K_c ≈ 0.48. With K=3.0 we're well above critical.
        """
        cfg = DafConfig(
            num_nodes=1000, avg_degree=20, oscillator_model="kuramoto",
            coupling_strength=3.0, noise_sigma=0.001, dt=0.01,
            device=str(device),
        )
        engine = DafEngine(cfg)
        # Narrow frequency spread for reliable sync
        engine.states[:, 2] = engine.states[:, 2] * 0.3

        # Integrate: 50 × 100 = 5000 steps = 50 sec model time
        for _ in range(50):
            engine.step(100)

        r = compute_order_parameter(engine.states)
        assert r > 0.7, f"Synchronization failed: r = {r:.3f} (need > 0.7)"

    def test_stronger_coupling_better_sync(self, device):
        """Stronger coupling → higher order parameter."""
        results = {}
        for K in [0.5, 3.0]:
            cfg = DafConfig(
                num_nodes=500, avg_degree=20, oscillator_model="kuramoto",
                coupling_strength=K, noise_sigma=0.001, dt=0.01,
                device=str(device),
            )
            engine = DafEngine(cfg, enable_learning=False)
            # Narrow frequency spread
            engine.states[:, 2] *= 0.3
            for _ in range(30):
                engine.step(100)
            results[K] = compute_order_parameter(engine.states)

        assert results[3.0] > results[0.5]
