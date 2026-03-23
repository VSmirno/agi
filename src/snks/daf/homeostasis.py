"""Homeostatic plasticity — adapts firing thresholds to maintain target activity."""

import torch

from snks.daf.types import DafConfig


class Homeostasis:
    """Regulates oscillator thresholds based on exponential moving average of firing rates."""

    def __init__(self, config: DafConfig, num_nodes: int, device: torch.device) -> None:
        self.target_rate = config.homeostasis_target
        self.lambda_reg = config.homeostasis_lambda
        self.dt = config.dt
        self.tau = config.homeostasis_tau

        # EMA of firing rate per node
        self._firing_rate_ema = torch.full(
            (num_nodes,), self.target_rate, device=device, dtype=torch.float32
        )

    def update(
        self,
        fired_history: torch.Tensor,
        states: torch.Tensor,
    ) -> None:
        """Update thresholds (states[:, 3]) based on firing activity.

        Args:
            fired_history: (T, N) bool — spike history
            states: (N, 8) — states[:, 3] = threshold, modified in-place
        """
        T = fired_history.shape[0]

        # Current firing rate: fraction of steps with spike
        current_rate = fired_history.float().mean(dim=0)  # (N,)

        # EMA update: alpha = T * dt / tau (batch of T steps)
        alpha = min(T * self.dt / self.tau, 1.0)
        self._firing_rate_ema.mul_(1.0 - alpha).add_(current_rate, alpha=alpha)

        # Threshold adaptation:
        # Over-active → raise threshold, under-active → lower threshold
        error = self._firing_rate_ema - self.target_rate  # (N,)
        states[:, 3].add_(error, alpha=self.lambda_reg)
        states[:, 3].clamp_(0.01, 5.0)

    def get_firing_rates(self) -> torch.Tensor:
        """Return (N,) EMA firing rates."""
        return self._firing_rate_ema
