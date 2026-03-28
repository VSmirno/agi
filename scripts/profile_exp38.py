"""Quick profiling script for exp38 bottleneck identification.

Runs 5 full perception_cycles with per-component timing and exits.
Prints breakdown: engine.step vs rest of pipeline.

Usage:
    python scripts/profile_exp38.py [device] [n_steps]
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
N_PROFILE = int(sys.argv[2]) if len(sys.argv) > 2 else 10   # cycles to profile

from snks.experiments.exp38_scaling_gpu import _build_agent
from snks.env.causal_grid import make_level

print(f"Building agent (device={device})...", flush=True)
t0 = time.perf_counter()
agent = _build_agent(device)
print(f"  build_agent: {time.perf_counter()-t0:.2f}s", flush=True)

env = make_level("DoorKey", size=16, max_steps=100)
obs, _ = env.reset(seed=0)
if isinstance(obs, dict):
    obs = obs["image"]

pipeline = agent.causal_agent.pipeline

# ----- instrument engine.step -----
_orig = pipeline.engine.step
engine_times = []

def _timed(n):
    t = time.perf_counter()
    r = _orig(n)
    engine_times.append((time.perf_counter() - t) * 1000)
    return r

pipeline.engine.step = _timed

# ----- instrument STDP.apply (with sync to separate FHN async from STDP) -----
_orig_stdp = pipeline.engine.stdp.apply
stdp_times = []
fhn_sync_times = []  # time to sync GPU before STDP (= async FHN GPU work)

import torch as _torch

def _timed_stdp(graph, fh):
    # Sync GPU first: this measures pending async FHN work
    t_sync = time.perf_counter()
    _torch.cuda.synchronize()
    fhn_sync_times.append((time.perf_counter() - t_sync) * 1000)
    # Now measure STDP itself
    t = time.perf_counter()
    r = _orig_stdp(graph, fh)
    _torch.cuda.synchronize()  # ensure STDP GPU ops complete
    stdp_times.append((time.perf_counter() - t) * 1000)
    return r

pipeline.engine.stdp.apply = _timed_stdp

# ----- instrument encoder -----
_orig_enc = pipeline.encoder.encode
enc_times = []

def _timed_enc(img):
    t = time.perf_counter()
    r = _orig_enc(img)
    enc_times.append((time.perf_counter() - t) * 1000)
    return r

pipeline.encoder.encode = _timed_enc

# ----- run N_PROFILE cycles -----
print(f"\nProfiling {N_PROFILE} cycles...", flush=True)
cycle_times = []
for i in range(N_PROFILE):
    t = time.perf_counter()
    action = agent.step(obs)
    ct = (time.perf_counter() - t) * 1000
    cycle_times.append(ct)

    obs_next, _, term, trunc, _ = env.step(action)
    if isinstance(obs_next, dict):
        obs_next = obs_next["image"]
    agent.observe_result(obs_next)
    if term or trunc:
        obs, _ = env.reset(seed=i+1)
        if isinstance(obs, dict):
            obs = obs["image"]
    else:
        obs = obs_next

    fhn_s = fhn_sync_times[-1] if fhn_sync_times else 0
    stdp_s = stdp_times[-1] if stdp_times else 0
    print(f"  cycle {i+1:2d}: total={ct:.0f}ms  engine={engine_times[-1]:.0f}ms  fhn_gpu_async={fhn_s:.0f}ms  stdp={stdp_s:.0f}ms  enc={enc_times[-1]:.0f}ms", flush=True)

# restore
pipeline.engine.step = _orig
pipeline.engine.stdp.apply = _orig_stdp
pipeline.encoder.encode = _orig_enc

def _avg(lst):
    return round(sum(lst) / len(lst), 1) if lst else 0.0

print("\n" + "="*55)
print("TIMING SUMMARY (ms per cycle)")
print("="*55)
print(f"  total cycle:   {_avg(cycle_times):6.1f} ms  ({1000/_avg(cycle_times):.2f} steps/s)")
print(f"  engine.step:   {_avg(engine_times):6.1f} ms  ({_avg(engine_times)/_avg(cycle_times)*100:.1f}%)")
print(f"  fhn_gpu_async: {_avg(fhn_sync_times):6.1f} ms  ← FHN GPU compute waiting (async)")
print(f"  stdp_cpu+gpu:  {_avg(stdp_times):6.1f} ms  ← STDP after GPU sync")
print(f"  encoder:       {_avg(enc_times):6.1f} ms  ({_avg(enc_times)/_avg(cycle_times)*100:.1f}%)")
other = _avg(cycle_times) - _avg(engine_times) - _avg(enc_times)
print(f"  other:         {other:6.1f} ms  ({other/_avg(cycle_times)*100:.1f}%)")
print(f"\n  compile/status:")
print(f"    _compiled_step_fn: {pipeline.engine._compiled_step_fn}")
print(f"    _compiled_chunk_size: {pipeline.engine._compiled_chunk_size}")
print(f"\n  engine details:")
print(f"    first5 engine_ms: {[round(x,0) for x in engine_times[:5]]}")
print(f"    last5  engine_ms: {[round(x,0) for x in engine_times[-5:]]}")
print("="*55)
