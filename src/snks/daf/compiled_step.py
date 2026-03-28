"""Torch-compilable FHN integration step.

Fuses FHN dynamics + scatter_add coupling + noise + spike detection
into a compiled kernel, reducing HIP/CUDA launch overhead.

Strategy: compile a CHUNK_SIZE-step kernel (default 10), then call it
n_steps // CHUNK_SIZE times per perception cycle. This gives:
  - Fast compile: 10-step graph << 100-step graph (~10x less complex)
  - Low dispatch overhead: 10 HIP launches vs 100 (10x reduction)
  - Predictable performance: each 10-step call = 1 HIP dispatch + GPU compute
"""

from __future__ import annotations

import torch

# Number of FHN steps to compile into a single kernel.
# Trade-off: larger chunk = fewer dispatches, but slower to compile.
# 10 steps compiles in ~1-2s; 100 steps takes 10+ minutes on AMD ROCm inductor.
_COMPILE_CHUNK: int = 10


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


def _make_fhn_chunk(chunk_size: int):
    """Return a function that runs exactly chunk_size FHN steps.

    The loop is inlined so torch.compile generates one GPU kernel per
    chunk call, reducing HIP dispatch overhead from chunk_size to 1.
    chunk_size=10 compiles quickly (~1-2s) while still giving 10x
    dispatch reduction vs single-step compilation.
    """
    def _fhn_chunk(
        v: torch.Tensor,
        w: torch.Tensor,
        ext_v: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        edge_weight: torch.Tensor,
        K: float,
        I_base: float,
        a: float,
        b: float,
        inv_tau: float,
        dt: float,
        sqrt_dt_sigma: float,
        spike_thresh: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run chunk_size FHN steps as one compiled kernel. Returns (v, w, fired_chunk)."""
        N = v.shape[0]
        fired_chunk = torch.zeros(chunk_size, N, dtype=torch.bool, device=v.device)
        for t in range(chunk_size):
            dv = v - v * v * v / 3.0 - w + I_base
            v_src = v[src_idx]
            v_dst = v[dst_idx]
            contrib = edge_weight * (v_src - v_dst)
            coupling = torch.zeros_like(v)
            coupling.scatter_add_(0, dst_idx, contrib)
            coupling = coupling * K
            dv = dv + coupling + ext_v
            dw = (v + a - b * w) * inv_tau
            v = v + dt * dv + sqrt_dt_sigma * torch.randn_like(v)
            w = w + dt * dw + sqrt_dt_sigma * torch.randn_like(w)
            fired_chunk[t] = v > spike_thresh
        return v, w, fired_chunk

    return _fhn_chunk


_compiled_cache: dict[str, object] = {}


def make_compiled_integrate(
    backend: str = "inductor",
    chunk_size: int = _COMPILE_CHUNK,
    hint_N: int = 0,
    hint_E: int = 0,
):
    """Create a compiled chunk-step FHN integration function.

    Compiles a function that runs chunk_size steps per call.
    Call it n_steps // chunk_size times to cover a full perception cycle.

    Args:
        backend: torch.compile backend ("inductor", "aot_eager", etc.)
        chunk_size: steps per compiled call (default 10 — fast to compile)
        hint_N: expected number of nodes for warmup (0 = use small default).
            Pass the actual N to pre-compile for that size and avoid retrace
            on the first real call (important for AMD ROCm where retrace is slow).
        hint_E: expected number of edges for warmup (0 = use small default).

    Returns:
        compiled fn(v,w,ext,src,dst,ew,K,I,a,b,inv_tau,dt,sig,thresh) → (v,w,fired_chunk)
        or None if compilation failed.
    """
    cache_key = f"fn_chunk{chunk_size}_N{hint_N}_E{hint_E}"
    if cache_key in _compiled_cache:
        return _compiled_cache[cache_key]

    try:
        is_hip = hasattr(torch.version, "hip") and torch.version.hip is not None
        # dynamic=True: symbolic N/E shapes — no re-tracing when node/edge count changes
        compile_opts: dict = {"backend": backend, "dynamic": True}
        if is_hip:
            # PyTorch 2.6+: mode and options are mutually exclusive — use options only.
            # triton.cudagraphs=False: HIP doesn't support CUDA Graphs reliably.
            compile_opts["options"] = {"triton.cudagraphs": False}
        else:
            compile_opts["mode"] = "max-autotune"

        chunk_fn = _make_fhn_chunk(chunk_size)
        compiled = torch.compile(chunk_fn, **compile_opts)

        # Warmup: trigger compilation for the expected shapes so no retrace on first real call.
        # Using hint_N/hint_E (actual graph size) avoids the slow N=50K→large re-trace
        # on AMD ROCm where inductor retrace for new shapes takes 3+ minutes.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        N = hint_N if hint_N > 0 else 64
        E = hint_E if hint_E > 0 else 128
        _v = torch.randn(N, device=device)
        _w = torch.randn(N, device=device)
        _ext = torch.zeros(N, device=device)
        _src = torch.randint(0, N, (E,), device=device)
        _dst = torch.randint(0, N, (E,), device=device)
        _ew = torch.randn(E, device=device)
        compiled(_v, _w, _ext, _src, _dst, _ew, 0.05, 0.0, 0.7, 0.8, 0.08, 0.01, 0.0005, 0.5)
        _compiled_cache[cache_key] = compiled
        return compiled
    except Exception:
        _compiled_cache[cache_key] = None
        return None


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
    chunk_size: int = _COMPILE_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run n FHN integration steps, using compiled chunk calls when available.

    When step_fn is provided, calls it in chunks of chunk_size (e.g. 10 steps each).
    This reduces HIP dispatch overhead from n_steps launches to n_steps/chunk_size.
    Falls back to Python single-step loop when step_fn is None.

    Args:
        states: (N, 8) — modified in-place
        ext_currents: (N, 8) — external input
        src_idx, dst_idx: edge indices
        edge_weight: (E,) = sign * strength for each edge
        n_steps: total integration steps
        K, I_base, a, b, tau: FHN params
        dt, noise_sigma: integration params
        spike_thresh: spike detection threshold
        step_fn: compiled chunk function → (v, w, fired_chunk), or None
        chunk_size: steps per compiled call (must match what step_fn was compiled for)

    Returns:
        (states, fired_history) where fired_history is (n_steps, N) bool
    """
    N = states.shape[0]
    v = states[:, 0].contiguous()
    w = states[:, 4].contiguous()
    ext_v = ext_currents[:, 0].contiguous()

    inv_tau = 1.0 / tau
    sqrt_dt_sigma = (dt ** 0.5) * noise_sigma

    if step_fn is not None:
        # Compiled chunk path: n_steps/chunk_size dispatches instead of n_steps
        n_chunks = n_steps // chunk_size
        remainder = n_steps % chunk_size

        chunks: list[torch.Tensor] = []
        for _ in range(n_chunks):
            v, w, fired_chunk = step_fn(
                v, w, ext_v, src_idx, dst_idx, edge_weight,
                K, I_base, a, b, inv_tau, dt, sqrt_dt_sigma, spike_thresh,
            )
            chunks.append(fired_chunk)

        # Handle remainder steps with Python fallback
        if remainder > 0:
            rem_history = torch.empty(remainder, N, dtype=torch.bool, device=states.device)
            for t in range(remainder):
                v, w, fired = _fhn_step_inner(
                    v, w, ext_v, src_idx, dst_idx, edge_weight,
                    K, I_base, a, b, inv_tau, dt, sqrt_dt_sigma, spike_thresh,
                )
                rem_history[t] = fired
            chunks.append(rem_history)

        fired_history = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    else:
        # Fallback: Python for loop (slow on AMD ROCm due to HIP dispatch overhead)
        fired_history = torch.empty(n_steps, N, dtype=torch.bool, device=states.device)
        for t in range(n_steps):
            v, w, fired = _fhn_step_inner(
                v, w, ext_v, src_idx, dst_idx, edge_weight,
                K, I_base, a, b, inv_tau, dt, sqrt_dt_sigma, spike_thresh,
            )
            fired_history[t] = fired

    states[:, 0] = v
    states[:, 4] = w
    return states, fired_history
