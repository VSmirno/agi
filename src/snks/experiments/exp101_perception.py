"""Experiment 101: Perception Diagnostic (Stage 42).

Compares 3 encoder modes on DoorKey-5x5:
  - Gabor (baseline, expected ~0%)
  - Symbolic (perfect information, diagnostic)
  - CNN (RGB, practical replacement)

Gates:
    exp101a: symbolic SDR discriminates key/door/goal (>3 unique)
    exp101b: symbolic DoorKey-5x5 success >= 0.15 (CPU, 2K nodes, 20 eps)
    exp101c: CNN SDR color discrimination (overlap < 0.5)
    exp101d: CNN DoorKey-5x5 success >= 0.05 (CPU, 2K nodes, 20 eps)
    exp101e: Gabor baseline reference (measured, expect ~0%)
"""

from __future__ import annotations

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch


def _make_config(encoder_type: str = "gabor", n_actions: int = 7):
    """Create small CPU config for experiments."""
    from snks.agent.pure_daf_agent import PureDafConfig

    cfg = PureDafConfig()
    cfg.n_actions = n_actions
    cfg.max_episode_steps = 100
    cfg.encoder_type = encoder_type

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


def exp101a_symbolic_discrimination():
    """Exp 101a: Symbolic encoder discriminates MiniGrid objects.

    Gate: >3 distinct SDR patterns for key/door/goal/wall/empty.
    """
    print("\n--- Exp 101a: Symbolic SDR Discrimination ---", flush=True)

    from snks.encoder.symbolic import SymbolicEncoder

    enc = SymbolicEncoder(sdr_size=1600)
    sdrs = {}

    # Key (type=5, yellow=4)
    obs = torch.zeros(7, 7, 3)
    obs[3, 3] = torch.tensor([5, 4, 0])
    sdrs["key"] = enc.encode(obs)

    # Door closed (type=4, yellow=4, state=1)
    obs = torch.zeros(7, 7, 3)
    obs[3, 3] = torch.tensor([4, 4, 1])
    sdrs["door"] = enc.encode(obs)

    # Goal (type=8, green=1)
    obs = torch.zeros(7, 7, 3)
    obs[3, 3] = torch.tensor([8, 1, 0])
    sdrs["goal"] = enc.encode(obs)

    # Wall (type=1, grey=2)
    obs = torch.zeros(7, 7, 3)
    obs[3, 3] = torch.tensor([1, 2, 0])
    sdrs["wall"] = enc.encode(obs)

    # Empty
    obs = torch.zeros(7, 7, 3)
    sdrs["empty"] = enc.encode(obs)

    # Check uniqueness
    n_unique = 0
    names = list(sdrs.keys())
    for i in range(len(names)):
        unique = True
        for j in range(len(names)):
            if i != j and torch.equal(sdrs[names[i]], sdrs[names[j]]):
                unique = False
        if unique:
            n_unique += 1

    # Compute overlaps
    print(f"  Unique SDRs: {n_unique}/{len(sdrs)}")
    for name, sdr in sdrs.items():
        print(f"    {name}: {int(sdr.sum())} active bits")

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = float((sdrs[names[i]] * sdrs[names[j]]).sum())
            total = max(float(sdrs[names[i]].sum()), 1)
            print(f"    overlap({names[i]}, {names[j]}): {overlap/total:.2f}")

    print(f"  Gate (>3 unique): {'PASS' if n_unique > 3 else 'FAIL'}", flush=True)
    return {"n_unique": n_unique, "gate_pass": n_unique > 3}


def _run_doorkey_test(encoder_type: str, n_episodes: int = 20, max_steps: int = 100):
    """Run DoorKey-5x5 with given encoder type."""
    from snks.agent.pure_daf_agent import PureDafAgent
    from snks.env.adapter import MiniGridAdapter

    env = MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    cfg = _make_config(encoder_type=encoder_type, n_actions=env.n_actions)
    cfg.max_episode_steps = max_steps
    agent = PureDafAgent(cfg)

    results = agent.run_training(env, n_episodes=n_episodes, max_steps=max_steps)

    successes = sum(1 for r in results if r.success)
    success_rate = successes / n_episodes
    mean_steps = np.mean([r.steps for r in results])
    mean_reward = np.mean([r.reward for r in results])

    return {
        "success_rate": success_rate,
        "successes": successes,
        "n_episodes": n_episodes,
        "mean_steps": float(mean_steps),
        "mean_reward": float(mean_reward),
    }


def exp101b_symbolic_doorkey():
    """Exp 101b: Symbolic encoder on DoorKey-5x5.

    DIAGNOSTIC: Can DAF/STDP solve DoorKey with perfect perception?
    Gate: success_rate >= 0.15 (CPU, 2K nodes, 20 episodes)
    """
    print("\n--- Exp 101b: Symbolic Encoder on DoorKey-5x5 ---", flush=True)
    print("  DIAGNOSTIC: perfect information → DAF/STDP learning test", flush=True)

    try:
        from snks.env.adapter import MiniGridAdapter
        MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed", flush=True)
        return {"status": "SKIP"}

    r = _run_doorkey_test("symbolic", n_episodes=20, max_steps=100)
    print(f"  Success rate: {r['success_rate']:.3f} ({r['successes']}/{r['n_episodes']})")
    print(f"  Mean steps: {r['mean_steps']:.1f}")
    print(f"  Mean reward: {r['mean_reward']:.3f}")
    print(f"  Gate (>=0.15): {'PASS' if r['success_rate'] >= 0.15 else 'FAIL'}", flush=True)
    r["gate_pass"] = r["success_rate"] >= 0.15
    return r


def exp101c_cnn_discrimination():
    """Exp 101c: CNN encoder discriminates colors.

    Gate: overlap(red_sdr, green_sdr) < 0.5
    """
    print("\n--- Exp 101c: CNN SDR Color Discrimination ---", flush=True)

    from snks.encoder.rgb_conv import RGBConvEncoder
    from snks.daf.types import EncoderConfig

    config = EncoderConfig(image_size=64, sdr_size=1600, sdr_sparsity=0.04)
    enc = RGBConvEncoder(config)

    # Red, green, blue, yellow images
    colors = {
        "red": torch.tensor([1, 0, 0], dtype=torch.float32),
        "green": torch.tensor([0, 1, 0], dtype=torch.float32),
        "blue": torch.tensor([0, 0, 1], dtype=torch.float32),
        "yellow": torch.tensor([1, 1, 0], dtype=torch.float32),
    }

    sdrs = {}
    for name, color in colors.items():
        img = color.view(3, 1, 1).expand(3, 64, 64)
        sdrs[name] = enc.encode(img)

    # Compute overlaps
    names = list(sdrs.keys())
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = float((sdrs[names[i]] * sdrs[names[j]]).sum())
            total = max(float(sdrs[names[i]].sum()), 1)
            ratio = overlap / total
            overlaps.append(ratio)
            print(f"    overlap({names[i]}, {names[j]}): {ratio:.3f}")

    mean_overlap = np.mean(overlaps)
    print(f"  Mean overlap: {mean_overlap:.3f}")
    print(f"  Gate (mean < 0.5): {'PASS' if mean_overlap < 0.5 else 'FAIL'}", flush=True)
    return {"mean_overlap": mean_overlap, "gate_pass": mean_overlap < 0.5}


def exp101d_cnn_doorkey():
    """Exp 101d: CNN encoder on DoorKey-5x5.

    Gate: success_rate >= 0.05 (CPU, 2K nodes, 20 episodes)
    """
    print("\n--- Exp 101d: CNN Encoder on DoorKey-5x5 ---", flush=True)

    try:
        from snks.env.adapter import MiniGridAdapter
        MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed", flush=True)
        return {"status": "SKIP"}

    r = _run_doorkey_test("cnn", n_episodes=20, max_steps=100)
    print(f"  Success rate: {r['success_rate']:.3f} ({r['successes']}/{r['n_episodes']})")
    print(f"  Mean steps: {r['mean_steps']:.1f}")
    print(f"  Mean reward: {r['mean_reward']:.3f}")
    print(f"  Gate (>=0.05): {'PASS' if r['success_rate'] >= 0.05 else 'FAIL'}", flush=True)
    r["gate_pass"] = r["success_rate"] >= 0.05
    return r


def exp101e_gabor_baseline():
    """Exp 101e: Gabor baseline (reference measurement).

    No gate — just record for comparison.
    """
    print("\n--- Exp 101e: Gabor Baseline (reference) ---", flush=True)

    try:
        from snks.env.adapter import MiniGridAdapter
        MiniGridAdapter("MiniGrid-DoorKey-5x5-v0")
    except ImportError:
        print("  SKIP: MiniGrid not installed", flush=True)
        return {"status": "SKIP"}

    r = _run_doorkey_test("gabor", n_episodes=10, max_steps=100)
    print(f"  Success rate: {r['success_rate']:.3f} ({r['successes']}/{r['n_episodes']})")
    print(f"  Mean steps: {r['mean_steps']:.1f}")
    print(f"  (No gate — reference only)", flush=True)
    r["gate_pass"] = True  # reference, always pass
    return r


def main():
    print("=" * 60)
    print("Experiment 101: Perception Diagnostic (Stage 42)")
    print("=" * 60)
    print("Comparing: Gabor (baseline) vs Symbolic vs CNN")
    print()

    results = {}
    all_pass = True

    for name, fn in [
        ("101a", exp101a_symbolic_discrimination),
        ("101b", exp101b_symbolic_doorkey),
        ("101c", exp101c_cnn_discrimination),
        ("101d", exp101d_cnn_doorkey),
        ("101e", exp101e_gabor_baseline),
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
        detail = ""
        if "success_rate" in r:
            detail = f" (success={r['success_rate']:.3f})"
        elif "n_unique" in r:
            detail = f" (unique={r['n_unique']})"
        elif "mean_overlap" in r:
            detail = f" (overlap={r['mean_overlap']:.3f})"
        print(f"  Exp {name}: {status}{detail}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    return results


if __name__ == "__main__":
    main()
