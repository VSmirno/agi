"""Torch-compilable FHN integration step.

Fuses FHN dynamics + scatter_add coupling + noise + spike detection
into a single compiled kernel, reducing HIP/CUDA launch overhead
from ~10 kernels/step to ~1-2.
"""

from __future__ import annotations

import torch


def _fhn_step_inner(
    v: torch.Tensor,           # (N,)
    w: torch.Tensor,           # (N,)
    ext_v: torch.Tensor,       # (N,) external current channel 0
    src_idx: torch.Tensor,     # (E,) int64
    dst_idx: torch.Tensor,     # (E,) int64
    edge_weight: torch.Tensor, # (E,) float32 = sign * strength
    K: float,
    I_base: float,
    a: float,
    b: float,
    inv_tau: float,
    dt: float,
    sqrt_dt_sigma: float,
    spike_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single FHN Euler-Maruyama step with coupling via scatter_add.

    Returns (v_new, w_new, fired).
    """
    # FHN derivatives
    dv = v - v * v * v / 3.0 - w + I_base

    # Coupling: coupling[i] = K * sum_{j->i} weight_ij * (v_j - v_i)
    v_src = v[src_idx]
    v_dst = v[dst_idx]
    contrib = edge_weight * (v_src - v_dst)
    coupling = torch.zeros_like(v)
    coupling.scatter_add_(0, dst_idx, contrib)
    coupling = coupling * K

    dv = dv + coupling + ext_v

    dw = (v + a - b * w) * inv_tau

    # Euler-Maruyama update
    v_new = v + dt * dv + sqrt_dt_sigma * torch.randn_like(v)
    w_new = w + dt * dw + sqrt_dt_sigma * torch.randn_like(w)

    fired = v_new > spike_thresh

    return v_new, w_new, fired


_compiled_cache: dict[str, object] = {}


def make_compiled_integrate(backend: str = "inductor"):
    """Create a compiled integration step function.

    Tries torch.compile with warmup. Falls back to raw function on failure.

    Args:
        backend: torch.compile backend ("inductor", "aot_eager", etc.)

    Returns:
        compiled or raw step function
    """
    if "fn" in _compiled_cache:
        return _compiled_cache["fn"]

    try:
        is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
        # dynamic=True: compile once for symbolic shapes — no re-tracing when N/E changes
        compile_opts: dict = {"backend": backend, "dynamic": True}
        if is_hip:
            # PyTorch 2.6+: mode and options are mutually exclusive — use options only.
            # triton.cudagraphs=False: HIP doesn't support CUDA Graphs reliably.
            compile_opts["options"] = {"triton.cudagraphs": False}
        else:
            compile_opts["mode"] = "max-autotune"
        compiled = torch.compile(_fhn_step_inner, **compile_opts)
        # Warmup: run once to trigger compilation and catch errors early
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N, E = 64, 128
        _v = torch.randn(N, device=device)
        _w = torch.randn(N, device=device)
        _ext = torch.zeros(N, device=device)
        _src = torch.randint(0, N, (E,), device=device)
        _dst = torch.randint(0, N, (E,), device=device)
        _ew = torch.randn(E, device=device)
        compiled(_v, _w, _ext, _src, _dst, _ew, 0.05, 0.0, 0.7, 0.8, 0.08, 0.01, 0.0005, 0.5)
        _compiled_cache["fn"] = compiled
        return compiled
    except Exception:
        _compiled_cache["fn"] = _fhn_step_inner
        return _fhn_step_inner


def integrate_fhn_compiled(
    states: torch.Tensor,          # (N, 8)
    ext_currents: torch.Tensor,    # (N, 8)
    src_idx: torch.Tensor,         # (E,)
    dst_idx: torch.Tensor,         # (E,)
    edge_weight: torch.Tensor,     # (E,) sign * strength
    n_steps: int,
    K: float,
    I_base: float,
    a: float,
    b: float,
    tau: float,
    dt: float,
    noise_sigma: float,
    spike_thresh: float = 0.5,
    step_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run n FHN integration steps using a (potentially compiled) step function.

    Args:
        states: (N, 8) — modified in-place
        ext_currents: (N, 8) — external input
        src_idx, dst_idx: edge indices
        edge_weight: (E,) = sign * strength for each edge
        n_steps: integration steps
        K, I_base, a, b, tau: FHN params
        dt, noise_sigma: integration params
        spike_thresh: spike detection threshold
        step_fn: compiled or raw step function

    Returns:
        (states, fired_history) where fired_history is (n_steps, N) bool
    """
    if step_fn is None:
        step_fn = _fhn_step_inner

    N = states.shape[0]
    v = states[:, 0].contiguous()
    w = states[:, 4].contiguous()
    ext_v = ext_currents[:, 0].contiguous()

    inv_tau = 1.0 / tau
    sqrt_dt_sigma = (dt ** 0.5) * noise_sigma

    fired_history = torch.empty(n_steps, N, dtype=torch.bool, device=states.device)

    for t in range(n_steps):
        v, w, fired = step_fn(
            v, w, ext_v, src_idx, dst_idx, edge_weight,
            K, I_base, a, b, inv_tau, dt, sqrt_dt_sigma, spike_thresh,
        )
        fired_history[t] = fired

    states[:, 0] = v
    states[:, 4] = w
    return states, fired_history
