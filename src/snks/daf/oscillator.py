"""Oscillator dynamics: Kuramoto (phase) and FitzHugh-Nagumo (spiking)."""

import torch

from snks.daf.types import DafConfig


def kuramoto_derivatives(
    states: torch.Tensor,
    config: DafConfig,
) -> torch.Tensor:
    """Intrinsic Kuramoto dynamics: dθ/dt = ω_i.

    Coupling is added externally via coupling.py.

    Args:
        states: (N, 8) — states[:, 0] = θ (phase), states[:, 2] = ω (frequency)
        config: DafConfig

    Returns:
        dstates: (N, 8) — only dstates[:, 0] is non-zero.
    """
    dstates = torch.zeros_like(states)
    dstates[:, 0] = states[:, 2]  # dθ/dt = ω
    return dstates


def fhn_derivatives(
    states: torch.Tensor,
    config: DafConfig,
) -> torch.Tensor:
    """FitzHugh-Nagumo dynamics.

    dv/dt = v - v³/3 - w + I_base
    dw/dt = (v + a - b*w) / τ

    Args:
        states: (N, 8) — [:, 0]=v, [:, 4]=w_recovery
        config: DafConfig

    Returns:
        dstates: (N, 8) — dstates[:, 0]=dv, dstates[:, 4]=dw
    """
    v = states[:, 0]
    w = states[:, 4]

    dv = v - v * v * v / 3.0 - w + config.fhn_I_base
    dw = (v + config.fhn_a - config.fhn_b * w) / config.fhn_tau

    dstates = torch.zeros_like(states)
    dstates[:, 0] = dv
    dstates[:, 4] = dw
    return dstates


def init_states(
    num_nodes: int,
    state_dim: int,
    model: str,
    device: torch.device,
    omega_std: float = 1.0,
) -> torch.Tensor:
    """Initialize oscillator states.

    Args:
        num_nodes: N
        state_dim: columns per node (8)
        model: "kuramoto" or "fhn"
        device: target device
        omega_std: natural frequency spread for Kuramoto (std of Normal distribution)

    Returns:
        states: (N, state_dim) float32 on device
    """
    states = torch.zeros(num_nodes, state_dim, device=device)

    if model == "kuramoto":
        # Random initial phases [0, 2π]
        states[:, 0] = torch.rand(num_nodes, device=device) * 2.0 * torch.pi
        # Natural frequencies — Normal(0, omega_std), clamped
        omega = torch.randn(num_nodes, device=device) * omega_std
        states[:, 2] = omega.clamp_(-5.0 * omega_std, 5.0 * omega_std)
    elif model == "fhn":
        # Small random v near zero
        states[:, 0] = torch.randn(num_nodes, device=device) * 0.1
        # w_recovery = 0
        states[:, 4] = 0.0
    else:
        raise ValueError(f"Unknown oscillator model: {model}")

    # Default thresholds
    states[:, 3] = 0.5

    return states
