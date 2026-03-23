"""Euler-Maruyama stochastic ODE integrator."""

from __future__ import annotations

from typing import Callable

import torch


# derivative_fn: (states (N,8)) → dstates (N,8)
DerivativeFn = Callable[[torch.Tensor], torch.Tensor]


def euler_maruyama_step(
    states: torch.Tensor,
    derivative_fn: DerivativeFn,
    dt: float,
    noise_sigma: float,
    noise_buf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single Euler-Maruyama step (in-place).

    x(t+dt) = x(t) + dt * f(x) + sqrt(dt) * sigma * N(0,1)

    Args:
        states: (N, 8) — modified in-place
        derivative_fn: computes drift from states
        dt: time step
        noise_sigma: noise intensity
        noise_buf: pre-allocated (N, 8) buffer for noise (optional)

    Returns:
        states — same tensor, modified in-place
    """
    drift = derivative_fn(states)  # (N, 8)
    states.add_(drift, alpha=dt)

    if noise_sigma > 0.0:
        if noise_buf is not None:
            noise_buf.normal_()
        else:
            noise_buf = torch.randn_like(states)
        states.add_(noise_buf, alpha=(dt ** 0.5) * noise_sigma)

    return states


def integrate_n_steps(
    states: torch.Tensor,
    derivative_fn: DerivativeFn,
    n_steps: int,
    dt: float,
    noise_sigma: float,
    spike_threshold_col: int = 0,
    spike_threshold_val: float = 0.5,
    spike_mode: str = "threshold",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate n Euler-Maruyama steps, record spike history.

    Args:
        states: (N, 8) — modified in-place
        derivative_fn: drift function
        n_steps: number of integration steps
        dt: time step
        noise_sigma: noise intensity
        spike_threshold_col: state column for spike detection (0=v/theta)
        spike_threshold_val: threshold for spike detection
        spike_mode: "threshold" (FHN: v > val) or "phase_crossing"
            (Kuramoto: sin(θ) crosses zero upward → 1 spike per cycle)

    Returns:
        states: (N, 8) final states
        fired_history: (n_steps, N) bool
    """
    N = states.shape[0]
    device = states.device

    fired_history = torch.empty(n_steps, N, dtype=torch.bool, device=device)
    noise_buf = torch.empty_like(states) if noise_sigma > 0.0 else None

    if spike_mode == "phase_crossing":
        prev_sin = torch.sin(states[:, 0])
        for t in range(n_steps):
            states = euler_maruyama_step(states, derivative_fn, dt, noise_sigma, noise_buf)
            curr_sin = torch.sin(states[:, 0])
            fired_history[t] = (curr_sin > 0) & (prev_sin <= 0)
            prev_sin = curr_sin
    else:
        for t in range(n_steps):
            states = euler_maruyama_step(states, derivative_fn, dt, noise_sigma, noise_buf)
            fired_history[t] = states[:, spike_threshold_col] > spike_threshold_val

    return states, fired_history
