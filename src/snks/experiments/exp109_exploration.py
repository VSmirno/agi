"""Exp 109: Exploration Strategy — Stage 55 gate test.

Sub-experiments:
  109a: Exploration efficiency — cells explored per step
  109b: MultiRoom-N3 with 7x7 partial obs (PRIMARY GATE)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from snks.agent.partial_obs_agent import (
    MultiRoomPartialObsAgent,
    PartialObsMultiRoomEnv,
)


def run_episode(seed: int, epsilon: float = 0.05,
                max_steps: int = 300) -> dict:
    """Run single MultiRoom episode."""
    env = PartialObsMultiRoomEnv()
    obs, col, row, d = env.reset(seed=seed)
    agent = MultiRoomPartialObsAgent(
        env.grid_width, env.grid_height, epsilon=epsilon
    )
    agent.reset()

    total_reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(obs, col, row, d)
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
    }


def exp109a_efficiency(n_seeds: int = 50) -> dict:
    """Exploration efficiency: cells per step."""
    print("=== Exp 109a: Exploration Efficiency ===")
    results = []
    for seed in range(n_seeds):
        r = run_episode(seed)
        results.append(r)

    explored = [r["explored_cells"] for r in results]
    steps = [r["steps"] for r in results]
    rate = np.mean([e / s for e, s in zip(explored, steps)])
    print(f"  Mean explored: {np.mean(explored):.1f}")
    print(f"  Mean steps: {np.mean(steps):.1f}")
    print(f"  Cells/step: {rate:.2f}")
    return {
        "mean_explored": float(np.mean(explored)),
        "mean_steps": float(np.mean(steps)),
        "cells_per_step": float(rate),
    }


def exp109b_gate(n_seeds: int = 200) -> dict:
    """PRIMARY GATE: ≥60% success on 200 MultiRoom-N3 with 7x7 view."""
    print("=== Exp 109b: Gate Test (200 seeds) ===")
    t0 = time.time()
    results = []

    for seed in range(n_seeds):
        r = run_episode(seed)
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
    print(f"  Gate ≥60%: {'PASS' if rate >= 0.6 else 'FAIL'}")

    return {
        "successes": successes,
        "total": n_seeds,
        "success_rate": rate,
        "mean_steps": mean_steps,
        "gate_pass": rate >= 0.6,
    }


def main():
    results = {}
    results["109a"] = exp109a_efficiency()
    print()
    results["109b"] = exp109b_gate()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "_docs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp109_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
