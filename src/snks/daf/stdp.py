"""STDP (Spike-Timing-Dependent Plasticity) learning rule.

Vectorized over all edges — no Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from snks.daf.graph import SparseDafGraph
from snks.daf.types import DafConfig


@dataclass
class STDPResult:
    """Result of one STDP application."""

    edges_potentiated: int
    edges_depressed: int
    mean_weight_change: float


class STDP:
    """Spike-Timing-Dependent Plasticity with homeostatic regularization."""

    def __init__(self, config: DafConfig) -> None:
        self.mode = config.stdp_mode  # "timing" or "rate"
        self.a_plus = config.stdp_a_plus
        self.a_minus = config.stdp_a_minus
        self.tau_plus = config.stdp_tau_plus
        self.tau_minus = config.stdp_tau_minus
        self.w_min = config.stdp_w_min
        self.w_max = config.stdp_w_max
        self.lambda_reg = config.homeostasis_lambda
        self.w_target = config.stdp_w_target
        self.dt = config.dt

    def apply(
        self,
        graph: SparseDafGraph,
        fired_history: torch.Tensor,
        lr_modulation: torch.Tensor | None = None,
    ) -> STDPResult:
        """Apply STDP to graph weights based on spike history.

        Args:
            graph: SparseDafGraph — edge_attr[:, 0] (strength) is modified in-place.
            fired_history: (T, N) bool — spike history from integrator.
            lr_modulation: (N,) optional per-node learning rate multiplier.
                If provided, dw is scaled by mean(lr[src], lr[dst]) per edge.

        Returns:
            STDPResult with statistics.
        """
        if self.mode == "rate":
            return self._apply_rate(graph, fired_history, lr_modulation)
        return self._apply_timing(graph, fired_history, lr_modulation)

    def _apply_rate(
        self,
        graph: SparseDafGraph,
        fired_history: torch.Tensor,
        lr_modulation: torch.Tensor | None = None,
    ) -> STDPResult:
        """Rate-based Hebbian: dw ∝ rate_src × rate_dst - baseline.

        Strengthens connections between nodes with correlated firing rates.
        Works for Kuramoto where spike timing is less meaningful.
        """
        T, N = fired_history.shape

        # Firing rate per node
        rates = fired_history.float().sum(dim=0) / T  # (N,)
        baseline = rates.mean() ** 2

        src_idx = graph.edge_index[0]
        dst_idx = graph.edge_index[1]

        rate_product = rates[src_idx] * rates[dst_idx]  # (E,)
        dw = self.a_plus * (rate_product - baseline)

        # Prediction-error-driven learning rate modulation
        if lr_modulation is not None:
            lr_src = lr_modulation[src_idx]
            lr_dst = lr_modulation[dst_idx]
            dw *= (lr_src + lr_dst) / 2.0

        # Homeostatic regularization: λ * (w_target - w)
        w = graph.get_strength()
        dw += self.lambda_reg * (self.w_target - w)

        w_new = (w + dw).clamp_(self.w_min, self.w_max)
        graph.set_strength(w_new)

        return STDPResult(
            edges_potentiated=int((dw > 0).sum()),
            edges_depressed=int((dw < 0).sum()),
            mean_weight_change=float(dw.abs().mean()),
        )

    def _apply_timing(
        self,
        graph: SparseDafGraph,
        fired_history: torch.Tensor,
        lr_modulation: torch.Tensor | None = None,
    ) -> STDPResult:
        """Classic spike-timing STDP for FHN and other spiking models."""
        T, N = fired_history.shape
        device = fired_history.device

        # --- Compute last spike time per node ---
        time_idx = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
        fired_float = fired_history.float()  # (T, N)
        weighted = fired_float * time_idx  # (T, N) — 0 where not fired
        last_spike = weighted.max(dim=0).values  # (N,)

        any_fired = fired_history.any(dim=0)  # (N,) bool
        last_spike = torch.where(
            any_fired, last_spike, torch.full_like(last_spike, -1e6)
        )

        # --- Per-edge timing ---
        src_idx = graph.edge_index[0]  # (E,)
        dst_idx = graph.edge_index[1]  # (E,)

        t_pre = last_spike[src_idx]  # (E,)
        t_post = last_spike[dst_idx]  # (E,)
        delta_t = t_post - t_pre  # (E,) positive = pre before post

        both_fired = any_fired[src_idx] & any_fired[dst_idx]  # (E,)

        dt_abs = delta_t.abs().clamp_max_(10.0 * max(self.tau_plus, self.tau_minus) / self.dt)
        dt_abs_sec = dt_abs * self.dt

        # --- STDP weight changes ---
        potentiation_mask = both_fired & (delta_t > 0)
        depression_mask = both_fired & (delta_t < 0)

        dw = torch.zeros_like(graph.edge_attr[:, 0])
        dw[potentiation_mask] = self.a_plus * torch.exp(
            -dt_abs_sec[potentiation_mask] / self.tau_plus
        )
        dw[depression_mask] = -self.a_minus * torch.exp(
            -dt_abs_sec[depression_mask] / self.tau_minus
        )

        # Prediction-error-driven learning rate modulation
        if lr_modulation is not None:
            lr_src = lr_modulation[src_idx]
            lr_dst = lr_modulation[dst_idx]
            dw *= (lr_src + lr_dst) / 2.0

        # Homeostatic regularization: λ * (w_target - w)
        w = graph.get_strength()
        dw += self.lambda_reg * (self.w_target - w)

        w_new = (w + dw).clamp_(self.w_min, self.w_max)
        graph.set_strength(w_new)

        return STDPResult(
            edges_potentiated=int(potentiation_mask.sum()),
            edges_depressed=int(depression_mask.sum()),
            mean_weight_change=float(dw.abs().mean()),
        )
