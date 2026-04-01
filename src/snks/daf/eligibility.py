"""Eligibility trace for reward-modulated STDP (Stage 41).

Accumulates STDP weight changes over time, allowing reward signals
to reach decisions made many steps ago.

    e(t) = λ × e(t-1) + dw(t)      # accumulate STDP changes
    Δw   = η × reward × e(t)        # apply reward-modulated credit
"""

from __future__ import annotations

import torch
from torch import Tensor

from snks.daf.graph import SparseDafGraph


class EligibilityTrace:
    """Edge-level eligibility trace for temporal credit assignment.

    Each edge in the DAF graph maintains a decaying trace of its
    recent STDP weight changes. When reward arrives, all edges
    receive credit proportional to their trace magnitude.
    """

    def __init__(
        self,
        decay: float = 0.92,
        reward_lr: float = 0.5,
    ) -> None:
        self.decay = decay
        self.reward_lr = reward_lr
        self._trace: Tensor | None = None
        self._steps_accumulated: int = 0
        self._total_reward_applied: float = 0.0
        self._total_edges_modulated: int = 0

    def accumulate(self, dw: Tensor) -> None:
        """Add STDP weight changes to the decaying trace.

        Args:
            dw: (E,) per-edge weight change from STDP.apply()
        """
        if self._trace is None or self._trace.shape != dw.shape:
            self._trace = dw.clone()
        else:
            self._trace.mul_(self.decay).add_(dw)
        self._steps_accumulated += 1

    def apply_reward(
        self,
        reward: float,
        graph: SparseDafGraph,
        w_min: float,
        w_max: float,
    ) -> int:
        """Apply reward-modulated credit to all traced edges.

        Args:
            reward: reward signal (positive or negative)
            graph: DAF graph whose edge weights will be modified
            w_min: minimum allowed weight
            w_max: maximum allowed weight

        Returns:
            Number of edges that received non-trivial modulation.
        """
        if self._trace is None or abs(reward) < 1e-8:
            return 0

        # Modulate weights: Δw = η × reward × trace
        delta = self.reward_lr * reward * self._trace
        w = graph.get_strength()

        # Safety: match edge count (structural pruning may have changed it)
        if w.shape[0] != delta.shape[0]:
            self.reset()
            return 0

        w_new = (w + delta).clamp_(w_min, w_max)
        graph.set_strength(w_new)

        modulated = int((delta.abs() > 1e-8).sum().item())
        self._total_reward_applied += abs(reward)
        self._total_edges_modulated += modulated
        return modulated

    def reset(self) -> None:
        """Reset trace (e.g., at episode start)."""
        if self._trace is not None:
            self._trace.zero_()
        self._steps_accumulated = 0

    @property
    def effective_window(self) -> int:
        """Number of steps where trace retains >= 5% of original signal."""
        if self.decay <= 0 or self.decay >= 1:
            return 0
        import math
        return int(math.log(0.05) / math.log(self.decay))

    @property
    def trace_magnitude(self) -> float:
        """Current L1 norm of trace (for monitoring)."""
        if self._trace is None:
            return 0.0
        return float(self._trace.abs().sum().item())

    @property
    def stats(self) -> dict:
        """Return trace statistics."""
        return {
            "steps_accumulated": self._steps_accumulated,
            "trace_magnitude": self.trace_magnitude,
            "effective_window": self.effective_window,
            "total_reward_applied": self._total_reward_applied,
            "total_edges_modulated": self._total_edges_modulated,
        }
