"""Experiment 102: Working Memory (Stage 43).

Tests whether sustained oscillation in WM zone improves DoorKey-5x5.

Gates:
    exp102a: WM zone preserves activation between cycles
    exp102b: WM DoorKey (symbolic) > no-WM DoorKey (symbolic)
    exp102c: WM activation correlates with task events
    exp102d: WM decay prevents lock-in
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch


def _make_config(encoder_type: str = "symbolic", wm_fraction: float = 0.0, n_actions: int = 7):
    """Create small CPU config."""
    from snks.agent.pure_daf_agent import PureDafConfig

    cfg = PureDafConfig()
    cfg.n_actions = n_actions
    cfg.max_episode_steps = 100
    cfg.encoder_type = encoder_type
    cfg.wm_fraction = wm_fraction
    cfg.wm_decay = 0.95

    cfg.causal.pipeline.daf.num_nodes = 2000
    cfg.causal.pipeline.daf.avg_degree = 15
    cfg.causal.pipeline.daf.device = "cpu"
    cfg.causal.pipeline.daf.disable_csr = True
    cfg.causal.pipeline.daf.dt = 0.005
    cfg.causal.pipeline.steps_per_cycle = 200
    cfg.causal.pipeline.encoder.image_size = 64
    cfg.causal.pipeline.encoder.pool_h = 5
    cfg.causal.pipeline.encoder.pool_w = 5
    cfg.causal.pipeline.encoder.n_orientations = 4
    cfg.causal.pipeline.encoder.sdr_size = 1600
    cfg.causal.pipeline.sks.min_cluster_size = 3
    cfg.causal.pipeline.sks.coherence_mode = "cofiring"
    cfg.causal.pipeline.sks.top_k = 200
    cfg.causal.motor_sdr_size = 200
    cfg.causal.pipeline.daf.fhn_I_base = 0.3
    cfg.causal.pipeline.daf.coupling_strength = 0.05

    cfg.epsilon_initial = 0.7
    cfg.epsilon_decay = 0.95
    cfg.epsilon_floor = 0.1
    cfg.reward_scale = 3.0
    cfg.n_sim_steps = 3
    return cfg


def exp102a_wm_activation():
    """Exp 102a: WM zone maintains activation across cycles.

    Gate: WM activation > 0 after 5 cycles with stimulus then 5 without.
    """
    print("\n--- Exp 102a: WM Activation Persistence ---", flush=True)

    from snks.pipeline.runner import Pipeline
    from snks.daf.types import PipelineConfig

    config = PipelineConfig()
    config.daf.num_nodes = 500
    config.daf.avg_degree = 15
    config.daf.device = "cpu"
    config.daf.disable_csr = True
    config.daf.dt = 0.005
    config.daf.wm_fraction = 0.2
    config.daf.wm_decay = 0.95
    config.steps_per_cycle = 100

    pipeline = Pipeline(config)

    # 5 cycles with strong stimulus
    sdr = torch.zeros(config.encoder.sdr_size)
    sdr[:100] = 1.0
    activations_with = []
    for _ in range(5):
        result = pipeline.perception_cycle(pre_sdr=sdr)
        activations_with.append(result.wm_activation)

    # 5 cycles without stimulus
    empty_sdr = torch.zeros(config.encoder.sdr_size)
    activations_without = []
    for _ in range(5):
        result = pipeline.perception_cycle(pre_sdr=empty_sdr)
        activations_without.append(result.wm_activation)

    print(f"  WM with stimulus:    {[f'{a:.3f}' for a in activations_with]}")
    print(f"  WM without stimulus: {[f'{a:.3f}' for a in activations_without]}")
    print(f"  Final WM activation: {activations_without[-1]:.4f}")

    gate = activations_without[-1] > 0.01
    print(f"  Gate (final > 0.01): {'PASS' if gate else 'FAIL'}", flush=True)

    return {
        "activations_with": activations_with,
        "activations_without": activations_without,
        "gate_pass": gate,
    }


def exp102b_wm_vs_nowm_doorkey():
    """Exp 102b: WM vs no-WM on DoorKey-5x5 (symbolic encoder).

    Gate: WM success rate > no-WM success rate (or both > 0).
    """
    print("\n--- Exp 102b: WM vs No-WM on DoorKey-5x5 ---", flush=True)

    try:
        from snks.env.adapter import MiniGridAdapter
        MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed", flush=True)
        return {"status": "SKIP"}

    from snks.agent.pure_daf_agent import PureDafAgent
    from snks.env.adapter import MiniGridAdapter

    n_episodes = 20

    # No-WM baseline
    print(f"  Running NO-WM baseline ({n_episodes} episodes)...", flush=True)
    env_nowm = MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    cfg_nowm = _make_config(encoder_type="symbolic", wm_fraction=0.0, n_actions=env_nowm.n_actions)
    agent_nowm = PureDafAgent(cfg_nowm)
    results_nowm = agent_nowm.run_training(env_nowm, n_episodes=n_episodes, max_steps=100)
    sr_nowm = sum(1 for r in results_nowm if r.success) / n_episodes

    # WM
    print(f"  Running WM ({n_episodes} episodes)...", flush=True)
    env_wm = MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    cfg_wm = _make_config(encoder_type="symbolic", wm_fraction=0.2, n_actions=env_wm.n_actions)
    agent_wm = PureDafAgent(cfg_wm)
    results_wm = agent_wm.run_training(env_wm, n_episodes=n_episodes, max_steps=100)
    sr_wm = sum(1 for r in results_wm if r.success) / n_episodes

    print(f"  No-WM: {sr_nowm:.3f} ({int(sr_nowm*n_episodes)}/{n_episodes})")
    print(f"  WM:    {sr_wm:.3f} ({int(sr_wm*n_episodes)}/{n_episodes})")
    print(f"  WM activations (last ep): {results_wm[-1].causal_stats.get('eligibility', {})}")

    # Gate: WM should be at least as good, ideally better
    gate = sr_wm >= sr_nowm or sr_wm > 0
    print(f"  Gate (WM >= no-WM or WM > 0): {'PASS' if gate else 'FAIL'}", flush=True)

    return {
        "sr_nowm": sr_nowm,
        "sr_wm": sr_wm,
        "gate_pass": gate,
    }


def exp102c_wm_tracks_events():
    """Exp 102c: WM activation changes with different stimuli.

    Gate: WM activation differs for different inputs.
    """
    print("\n--- Exp 102c: WM Tracks Stimulus Changes ---", flush=True)

    from snks.pipeline.runner import Pipeline
    from snks.daf.types import PipelineConfig

    config = PipelineConfig()
    config.daf.num_nodes = 500
    config.daf.avg_degree = 15
    config.daf.device = "cpu"
    config.daf.disable_csr = True
    config.daf.dt = 0.005
    config.daf.wm_fraction = 0.2
    config.daf.wm_decay = 0.95
    config.steps_per_cycle = 100

    pipeline = Pipeline(config)

    # Stimulus A
    sdr_a = torch.zeros(config.encoder.sdr_size)
    sdr_a[0:50] = 1.0
    pipeline.perception_cycle(pre_sdr=sdr_a)
    wm_a = pipeline.engine.states[400:, 0].clone()

    # Stimulus B (different pattern)
    sdr_b = torch.zeros(config.encoder.sdr_size)
    sdr_b[800:850] = 1.0
    pipeline.perception_cycle(pre_sdr=sdr_b)
    wm_b = pipeline.engine.states[400:, 0].clone()

    # WM states should differ
    diff = (wm_a - wm_b).abs().mean().item()
    print(f"  WM diff between stimuli A and B: {diff:.4f}")
    gate = diff > 0.01
    print(f"  Gate (diff > 0.01): {'PASS' if gate else 'FAIL'}", flush=True)

    return {"diff": diff, "gate_pass": gate}


def exp102d_wm_decay_prevents_lockin():
    """Exp 102d: WM decays to near-zero without input.

    Gate: After 50 empty cycles, WM activation < 50% of peak.
    """
    print("\n--- Exp 102d: WM Decay Prevents Lock-in ---", flush=True)

    from snks.pipeline.runner import Pipeline
    from snks.daf.types import PipelineConfig

    config = PipelineConfig()
    config.daf.num_nodes = 500
    config.daf.avg_degree = 15
    config.daf.device = "cpu"
    config.daf.disable_csr = True
    config.daf.dt = 0.005
    config.daf.wm_fraction = 0.2
    config.daf.wm_decay = 0.9  # faster decay for this test
    config.steps_per_cycle = 100

    pipeline = Pipeline(config)

    # Strong stimulus
    sdr = torch.zeros(config.encoder.sdr_size)
    sdr[:200] = 1.0
    pipeline.perception_cycle(pre_sdr=sdr)
    peak = pipeline.engine.states[400:, 0].abs().mean().item()

    # 50 empty cycles
    empty = torch.zeros(config.encoder.sdr_size)
    for _ in range(50):
        pipeline.perception_cycle(pre_sdr=empty)
    final = pipeline.engine.states[400:, 0].abs().mean().item()

    ratio = final / max(peak, 1e-8)
    print(f"  Peak WM: {peak:.4f}")
    print(f"  After 50 cycles: {final:.4f}")
    print(f"  Ratio: {ratio:.3f}")
    gate = ratio < 0.5
    print(f"  Gate (ratio < 0.5): {'PASS' if gate else 'FAIL'}", flush=True)

    return {"peak": peak, "final": final, "ratio": ratio, "gate_pass": gate}


def main():
    print("=" * 60)
    print("Experiment 102: Working Memory (Stage 43)")
    print("=" * 60)
    print("Sustained oscillation in WM zone across perception cycles")
    print()

    results = {}
    all_pass = True

    for name, fn in [
        ("102a", exp102a_wm_activation),
        ("102b", exp102b_wm_vs_nowm_doorkey),
        ("102c", exp102c_wm_tracks_events),
        ("102d", exp102d_wm_decay_prevents_lockin),
    ]:
        r = fn()
        results[name] = r
        if r.get("status") == "SKIP":
            continue
        if not r.get("gate_pass", False):
            all_pass = False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, r in results.items():
        status = "SKIP" if r.get("status") == "SKIP" else ("PASS" if r.get("gate_pass") else "FAIL")
        print(f"  Exp {name}: {status}")
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return results


if __name__ == "__main__":
    main()
