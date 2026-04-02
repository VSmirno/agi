"""Exp 108: Partial Observability — Stage 54 gate test.

Sub-experiments:
  108a: SpatialMap coverage — % of grid explored in N steps
  108b: PartialObsAgent on 200 random DoorKey-5x5 with 7x7 view (PRIMARY GATE)
  108c: Ablation — exploration-only vs full agent
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from snks.agent.partial_obs_agent import PartialObsAgent, PartialObsDoorKeyEnv


def run_episode(seed: int, epsilon: float = 0.05,
                max_steps: int = 200) -> dict:
    """Run single episode, return metrics."""
    env = PartialObsDoorKeyEnv(size=5, seed=seed)
    agent = PartialObsAgent(5, 5, epsilon=epsilon)
    agent.reset()

    obs, col, row, d = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        action = agent.select_action(obs, col, row, d)
        agent.update_inventory(env.carrying)
        obs, reward, term, trunc, col, row, d = env.step(action)
        agent.observe_result(obs, col, row, d, reward)
        total_reward += reward

        if term or trunc:
            break

    explored = int(np.sum(agent.spatial_map.explored))
    return {
        "seed": seed,
        "success": total_reward > 0,
        "steps": step + 1,
        "reward": total_reward,
        "explored_cells": explored,
        "total_cells": 25,
    }


def exp108a_coverage(n_seeds: int = 50) -> dict:
    """SpatialMap coverage: how much of 5x5 grid gets explored."""
    print("=== Exp 108a: SpatialMap Coverage ===")
    results = []
    for seed in range(n_seeds):
        r = run_episode(seed, epsilon=0.05)
        results.append(r)
        if (seed + 1) % 10 == 0:
            avg_explored = np.mean([x["explored_cells"] for x in results])
            print(f"  seeds 0-{seed}: avg explored {avg_explored:.1f}/25")

    explored_vals = [r["explored_cells"] for r in results]
    return {
        "mean_explored": float(np.mean(explored_vals)),
        "min_explored": int(np.min(explored_vals)),
        "max_explored": int(np.max(explored_vals)),
        "coverage_pct": float(np.mean(explored_vals)) / 25 * 100,
    }


def exp108b_gate(n_seeds: int = 200) -> dict:
    """PRIMARY GATE: ≥80% success on 200 random DoorKey-5x5 with 7x7 view."""
    print("=== Exp 108b: Gate Test (200 seeds) ===")
    t0 = time.time()
    results = []

    for seed in range(n_seeds):
        r = run_episode(seed, epsilon=0.05)
        results.append(r)
        if (seed + 1) % 50 == 0:
            rate = np.mean([x["success"] for x in results])
            elapsed = time.time() - t0
            eta = elapsed / (seed + 1) * (n_seeds - seed - 1)
            print(f"  [{seed+1}/{n_seeds}] success={rate:.1%}, "
                  f"elapsed={elapsed:.1f}s, ETA={eta:.1f}s")

    successes = sum(1 for r in results if r["success"])
    rate = successes / n_seeds
    steps = [r["steps"] for r in results if r["success"]]
    mean_steps = float(np.mean(steps)) if steps else 0

    print(f"\n  RESULT: {successes}/{n_seeds} = {rate:.1%}")
    print(f"  Mean steps (successes): {mean_steps:.1f}")
    print(f"  Gate ≥80%: {'PASS' if rate >= 0.8 else 'FAIL'}")

    return {
        "successes": successes,
        "total": n_seeds,
        "success_rate": rate,
        "mean_steps": mean_steps,
        "gate_pass": rate >= 0.8,
    }


def exp108c_ablation(n_seeds: int = 50) -> dict:
    """Ablation: compare full agent vs explore-only (no subgoal planning)."""
    print("=== Exp 108c: Ablation ===")

    # Full agent
    full_results = []
    for seed in range(n_seeds):
        r = run_episode(seed, epsilon=0.05)
        full_results.append(r)
    full_rate = np.mean([x["success"] for x in full_results])

    # Random-only (epsilon=1.0)
    random_results = []
    for seed in range(n_seeds):
        r = run_episode(seed, epsilon=1.0)
        random_results.append(r)
    random_rate = np.mean([x["success"] for x in random_results])

    print(f"  Full agent: {full_rate:.1%}")
    print(f"  Random-only: {random_rate:.1%}")
    print(f"  Delta: {full_rate - random_rate:+.1%}")

    return {
        "full_agent_rate": float(full_rate),
        "random_rate": float(random_rate),
        "delta": float(full_rate - random_rate),
    }


def main():
    results = {}

    results["108a"] = exp108a_coverage()
    print()
    results["108b"] = exp108b_gate()
    print()
    results["108c"] = exp108c_ablation()

    # Save results
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "_docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp108_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
