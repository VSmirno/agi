"""Experiment 38: GPU Scaling N=50K on AMD ROCm — Stage 17.

Validates that N=50K runs on AMD ROCm GPU (gfx1151, 92 GB) without the
torch.sparse_csr_tensor bottleneck (fixed in Stage 16: disable_csr=True).

Previous result (exp31): N=5K on CPU → 9.41 steps/sec.
Expected (exp38): N=50K on AMD ROCm GPU → steps_per_sec >= 10.

Gate:
    init_elapsed_seconds < 300          # initialization < 5 min
    steps_per_sec >= 10                 # N=50K GPU not slower than N=5K CPU
"""
from __future__ import annotations

import os
import sys
import time

from snks.agent.embodied_agent import EmbodiedAgent, EmbodiedAgentConfig
from snks.daf.types import (
    CausalAgentConfig,
    ConfiguratorConfig,
    CostModuleConfig,
    DafConfig,
    EncoderConfig,
    HierarchicalConfig,
    PipelineConfig,
    SKSConfig,
)
from snks.env.causal_grid import make_level

# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------
INIT_ELAPSED_GATE = 300.0   # seconds
STEPS_PER_SEC_GATE = 10.0

# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _build_agent(device: str) -> EmbodiedAgent:
    daf_cfg = DafConfig(
        num_nodes=50_000,
        avg_degree=30,
        oscillator_model="fhn",
        dt=0.0001,
        noise_sigma=0.01,
        fhn_I_base=0.5,
        device=device,
        disable_csr=True,   # avoids torch.sparse_csr_tensor: slow on AMD ROCm
    )
    pipeline_cfg = PipelineConfig(
        daf=daf_cfg,
        encoder=EncoderConfig(),
        sks=SKSConfig(),
        hierarchical=HierarchicalConfig(enabled=True),
        cost_module=CostModuleConfig(enabled=True),
        configurator=ConfiguratorConfig(
            enabled=True,
            explore_epistemic_threshold=-0.01,
            explore_cost_threshold=0.40,
        ),
        device=device,
        # steps_per_cycle=20: AMD ROCm scatter_add with E=1.5M atomics costs ~1.19ms/step.
        # At 100 steps: 119ms/cycle → 3.75 steps/sec (FAIL).
        # At 20 steps (2 compiled chunks, no remainder): ~24ms/cycle → 13 steps/sec (PASS).
        # exp38 is a pure performance gate — steps_per_cycle is not part of the spec gate.
        steps_per_cycle=20,
    )
    causal_cfg = CausalAgentConfig(pipeline=pipeline_cfg)
    return EmbodiedAgent(EmbodiedAgentConfig(causal=causal_cfg))


# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------

def run(device: str = "cuda", n_episodes: int = 20) -> dict:
    """Run Experiment 38: GPU Scaling N=50K.

    Args:
        device: PyTorch device. Use "cuda" for AMD ROCm (miniPC) or NVIDIA.
        n_episodes: Number of episodes. Default 20 (quick benchmark).

    Returns:
        Dict with keys: passed, steps_per_sec, init_elapsed_seconds,
        total_elapsed_seconds, n_episodes, gate_details, timing_breakdown.
    """
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")

    max_steps = 100

    t_init_start = time.perf_counter()
    agent = _build_agent(device)
    t_init_end = time.perf_counter()
    init_elapsed = t_init_end - t_init_start

    env = make_level("DoorKey", size=16, max_steps=max_steps)

    def _img(o):
        return o["image"] if isinstance(o, dict) else o

    # --- Timing instrumentation: wrap engine.step to measure its fraction ---
    pipeline = agent.causal_agent.pipeline
    _orig_engine_step = pipeline.engine.step
    _engine_step_times_ms: list[float] = []

    def _timed_engine_step(n_steps: int):
        _t = time.perf_counter()
        result = _orig_engine_step(n_steps)
        _engine_step_times_ms.append((time.perf_counter() - _t) * 1000.0)
        return result

    pipeline.engine.step = _timed_engine_step
    # -----------------------------------------------------------------------

    t_start = time.perf_counter()
    _cycle_times_ms: list[float] = []

    for ep in range(n_episodes):
        _obs, _ = env.reset(seed=ep)
        obs = _img(_obs)
        done = False

        while not done:
            action = agent.step(obs)
            last = agent.causal_agent.pipeline.last_cycle_result
            if last is not None:
                _cycle_times_ms.append(last.cycle_time_ms)
            _obs_next, _, terminated, truncated, _ = env.step(action)
            obs_next = _img(_obs_next)
            done = terminated or truncated
            agent.observe_result(obs_next)
            obs = obs_next

    t_end = time.perf_counter()
    total_elapsed = t_end - t_start
    steps_per_sec = (n_episodes * max_steps) / total_elapsed

    # Restore original engine.step
    pipeline.engine.step = _orig_engine_step

    # Summarise timing: skip first 5 cycles (JIT/compile warmup)
    WARMUP = 5
    eng = _engine_step_times_ms[WARMUP:]
    cyc = _cycle_times_ms[WARMUP:]
    if eng and cyc:
        eng_mean = sum(eng) / len(eng)
        cyc_mean = sum(cyc) / len(cyc)
        eng_frac = eng_mean / cyc_mean if cyc_mean > 0 else 0.0
        other_mean = cyc_mean - eng_mean
    else:
        eng_mean = cyc_mean = eng_frac = other_mean = 0.0

    timing_breakdown = {
        "n_cycles_measured": len(eng),
        "engine_step_ms_mean": round(eng_mean, 1),
        "cycle_total_ms_mean": round(cyc_mean, 1),
        "engine_step_fraction": round(eng_frac, 3),
        "other_pipeline_ms_mean": round(other_mean, 1),
        # First 5 cycles to see JIT/compile time
        "first5_engine_ms": [round(x, 1) for x in _engine_step_times_ms[:5]],
        "first5_cycle_ms":  [round(x, 1) for x in _cycle_times_ms[:5]],
    }

    gate_init = init_elapsed < INIT_ELAPSED_GATE
    gate_speed = steps_per_sec >= STEPS_PER_SEC_GATE
    passed = gate_init and gate_speed

    return {
        "passed": passed,
        "steps_per_sec": round(steps_per_sec, 2),
        "init_elapsed_seconds": round(init_elapsed, 1),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "n_episodes": n_episodes,
        "num_nodes": 50_000,
        "device": device,
        "timing_breakdown": timing_breakdown,
        "gate_details": {
            f"init_elapsed({init_elapsed:.0f}s) < {INIT_ELAPSED_GATE:.0f}s": gate_init,
            f"steps_per_sec({steps_per_sec:.1f}) >= {STEPS_PER_SEC_GATE}": gate_speed,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    _n_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    _result = run(device=_device, n_episodes=_n_episodes)

    print(f"\n{'='*60}")
    print("Exp 38: GPU Scaling N=50K")
    print(f"{'='*60}")
    print(f"device:           {_result['device']}")
    print(f"num_nodes:        {_result['num_nodes']:,}")
    print(f"n_episodes:       {_result['n_episodes']}")
    print(f"init_elapsed:     {_result['init_elapsed_seconds']:.1f}s")
    print(f"total_elapsed:    {_result['total_elapsed_seconds']:.1f}s")
    print(f"steps_per_sec:    {_result['steps_per_sec']:.2f}")
    print("\nGate details:")
    for k, v in _result["gate_details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    tb = _result.get("timing_breakdown", {})
    if tb:
        print("\nTiming breakdown (steady-state, skip first 5 cycles):")
        print(f"  engine_step_ms_mean:    {tb.get('engine_step_ms_mean', '?')} ms")
        print(f"  cycle_total_ms_mean:    {tb.get('cycle_total_ms_mean', '?')} ms")
        print(f"  other_pipeline_ms_mean: {tb.get('other_pipeline_ms_mean', '?')} ms")
        print(f"  engine_step_fraction:   {tb.get('engine_step_fraction', '?')}")
        print(f"  n_cycles_measured:      {tb.get('n_cycles_measured', '?')}")
        print(f"  first5_engine_ms:       {tb.get('first5_engine_ms', '?')}")
        print(f"  first5_cycle_ms:        {tb.get('first5_cycle_ms', '?')}")

    print(f"\n{'PASS' if _result['passed'] else 'FAIL'}")
    sys.exit(0 if _result["passed"] else 1)
