"""Exp 116: Demo-Guided Agent — Gate Experiments.

Three phases testing the demo-guided agent in MiniGrid environments:
  116a: DoorKey-8x8, 20 seeds, gate >=95% success rate
  116b: LockedRoom, 20 seeds, gate >=80% success rate
  116c: Ablation — with vs without causal model, delta >=40%

IMPORTANT: Run ONLY on minipc, NEVER locally.
Deploy: git push → ssh minipc "cd /opt/agi && git pull" → tmux launch
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from snks.agent.demo_guided_agent import (
    COLOR_NAMES,
    DemoGuidedAgent,
)

TRAIN_COLORS = ["red", "blue", "yellow"]
N_SEEDS = 20
MAX_STEPS = 300


def run_doorkey_episode(agent: DemoGuidedAgent, size: int = 8,
                        seed: int = 0, max_steps: int = MAX_STEPS
                        ) -> dict:
    """Run one DoorKey episode. Returns episode stats."""
    from minigrid.envs.doorkey import DoorKeyEnv

    env = DoorKeyEnv(size=size, max_steps=max_steps)
    obs_dict, info = env.reset(seed=seed)
    obs = obs_dict["image"]
    pos = env.agent_pos
    d = env.agent_dir

    agent.reset()
    total_reward = 0.0

    for step_i in range(max_steps):
        action = agent.select_action(obs, int(pos[0]), int(pos[1]), int(d))

        obs_dict, reward, term, trunc, info = env.step(action)
        obs = obs_dict["image"]
        pos = env.agent_pos
        d = env.agent_dir
        total_reward += float(reward)

        carrying = env.carrying is not None
        agent.observe_result(obs, int(pos[0]), int(pos[1]), int(d),
                             float(reward), carrying)

        if term or trunc:
            break

    success = total_reward > 0
    stats = agent.get_stats()
    return {
        "seed": seed,
        "success": success,
        "steps": step_i + 1,
        "reward": total_reward,
        **stats,
    }


def run_lockedroom_episode(agent: DemoGuidedAgent, seed: int = 0,
                           max_steps: int = MAX_STEPS) -> dict:
    """Run one LockedRoom episode. Returns episode stats."""
    from minigrid.envs.lockedroom import LockedRoom

    env = LockedRoom(max_steps=max_steps)
    obs_dict, info = env.reset(seed=seed)
    obs = obs_dict["image"]
    pos = env.agent_pos
    d = env.agent_dir

    # LockedRoom has larger grid
    grid_w = env.grid.width
    grid_h = env.grid.height
    agent.spatial_map = __import__(
        "snks.agent.spatial_map", fromlist=["SpatialMap"]
    ).SpatialMap(grid_w, grid_h)
    agent.executor.spatial_map = agent.spatial_map

    agent.reset()
    total_reward = 0.0

    for step_i in range(max_steps):
        action = agent.select_action(obs, int(pos[0]), int(pos[1]), int(d))

        obs_dict, reward, term, trunc, info = env.step(action)
        obs = obs_dict["image"]
        pos = env.agent_pos
        d = env.agent_dir
        total_reward += float(reward)

        carrying = env.carrying is not None
        agent.observe_result(obs, int(pos[0]), int(pos[1]), int(d),
                             float(reward), carrying)

        if term or trunc:
            break

    success = total_reward > 0
    stats = agent.get_stats()
    return {
        "seed": seed,
        "success": success,
        "steps": step_i + 1,
        "reward": total_reward,
        "grid_size": f"{grid_w}x{grid_h}",
        **stats,
    }


def phase_a(n_seeds: int = N_SEEDS) -> dict:
    """116a: DoorKey-8x8, gate >=95%."""
    print("=== Exp 116a: DoorKey-8x8 ===")
    t0 = time.time()

    agent = DemoGuidedAgent(grid_width=8, grid_height=8)
    agent.learn_from_demos(TRAIN_COLORS)

    results = []
    successes = 0
    for seed in range(n_seeds):
        ep = run_doorkey_episode(agent, size=8, seed=seed)
        results.append(ep)
        if ep["success"]:
            successes += 1
        status = "OK" if ep["success"] else "FAIL"
        print(f"  [{status}] seed={seed:2d}  steps={ep['steps']:3d}  "
              f"explore={ep['explore_steps']:3d}  execute={ep['execute_steps']:3d}")

    success_rate = successes / n_seeds
    gate = success_rate >= 0.95
    elapsed = time.time() - t0

    mean_steps = np.mean([r["steps"] for r in results if r["success"]]) if successes else 0
    mean_explore = np.mean([r["explore_steps"] for r in results if r["success"]]) if successes else 0

    print(f"  Success rate: {success_rate:.1%} ({successes}/{n_seeds})")
    print(f"  Mean steps (success): {mean_steps:.1f}")
    print(f"  Mean explore steps: {mean_explore:.1f}")
    print(f"  Gate (>=95%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "116a",
        "env": "DoorKey-8x8",
        "success_rate": round(success_rate, 4),
        "successes": successes,
        "n_seeds": n_seeds,
        "mean_steps": round(float(mean_steps), 1),
        "mean_explore_steps": round(float(mean_explore), 1),
        "gate": gate,
        "gate_threshold": 0.95,
        "episodes": results,
        "elapsed_s": round(elapsed, 2),
    }


def phase_b(n_seeds: int = N_SEEDS) -> dict:
    """116b: LockedRoom, gate >=80%."""
    print("\n=== Exp 116b: LockedRoom ===")
    t0 = time.time()

    agent = DemoGuidedAgent(grid_width=19, grid_height=19)
    agent.learn_from_demos(TRAIN_COLORS)

    results = []
    successes = 0
    for seed in range(n_seeds):
        ep = run_lockedroom_episode(agent, seed=seed)
        results.append(ep)
        if ep["success"]:
            successes += 1
        status = "OK" if ep["success"] else "FAIL"
        print(f"  [{status}] seed={seed:2d}  steps={ep['steps']:3d}  "
              f"explore={ep['explore_steps']:3d}  execute={ep['execute_steps']:3d}")

    success_rate = successes / n_seeds
    gate = success_rate >= 0.80
    elapsed = time.time() - t0

    mean_steps = np.mean([r["steps"] for r in results if r["success"]]) if successes else 0

    print(f"  Success rate: {success_rate:.1%} ({successes}/{n_seeds})")
    print(f"  Mean steps (success): {mean_steps:.1f}")
    print(f"  Gate (>=80%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "116b",
        "env": "LockedRoom",
        "success_rate": round(success_rate, 4),
        "successes": successes,
        "n_seeds": n_seeds,
        "mean_steps": round(float(mean_steps), 1),
        "gate": gate,
        "gate_threshold": 0.80,
        "episodes": results,
        "elapsed_s": round(elapsed, 2),
    }


def phase_c(n_seeds: int = N_SEEDS) -> dict:
    """116c: Ablation — with vs without causal model, delta >=40%."""
    print("\n=== Exp 116c: Ablation ===")
    t0 = time.time()

    # With causal model (trained)
    agent_trained = DemoGuidedAgent(grid_width=19, grid_height=19)
    agent_trained.learn_from_demos(TRAIN_COLORS)

    # Without causal model (untrained — random color decisions)
    agent_random = DemoGuidedAgent(grid_width=19, grid_height=19)
    # Don't call learn_from_demos — model is untrained

    trained_successes = 0
    random_successes = 0

    print("  --- Trained agent ---")
    for seed in range(n_seeds):
        ep = run_lockedroom_episode(agent_trained, seed=seed)
        if ep["success"]:
            trained_successes += 1

    print("  --- Random agent (no causal model) ---")
    for seed in range(n_seeds):
        ep = run_lockedroom_episode(agent_random, seed=seed)
        if ep["success"]:
            random_successes += 1

    trained_rate = trained_successes / n_seeds
    random_rate = random_successes / n_seeds
    delta = trained_rate - random_rate
    gate = delta >= 0.40

    elapsed = time.time() - t0

    print(f"  Trained success rate:  {trained_rate:.1%}")
    print(f"  Random success rate:   {random_rate:.1%}")
    print(f"  Delta: {delta:.1%}")
    print(f"  Gate (delta >=40%): {'PASS' if gate else 'FAIL'}")
    print(f"  Time: {elapsed:.2f}s")

    return {
        "phase": "116c",
        "trained_success_rate": round(trained_rate, 4),
        "random_success_rate": round(random_rate, 4),
        "delta": round(delta, 4),
        "gate": gate,
        "gate_threshold": 0.40,
        "elapsed_s": round(elapsed, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exp 116: Demo-Guided Agent Gates")
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--output", type=str, default="_docs/exp116_results.json")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["a", "b", "c", "all"])
    args = parser.parse_args()

    results = {}

    if args.phase in ("a", "all"):
        results["a"] = phase_a(args.n_seeds)
    if args.phase in ("b", "all"):
        results["b"] = phase_b(args.n_seeds)
    if args.phase in ("c", "all"):
        results["c"] = phase_c(args.n_seeds)

    if args.phase == "all":
        all_gates = all(r["gate"] for r in results.values())
        print(f"\n{'=' * 50}")
        print(f"ALL GATES: {'PASS' if all_gates else 'FAIL'}")
        for key, r in results.items():
            rate_key = "success_rate" if "success_rate" in r else "delta"
            rate = r[rate_key]
            thresh = r["gate_threshold"]
            print(f"  116{key}: {rate:.1%} (gate >={thresh:.0%}) "
                  f"{'PASS' if r['gate'] else 'FAIL'}")
    else:
        all_gates = results[args.phase]["gate"]

    full_results = {
        "experiment": "exp116_demo_guided_agent",
        "n_seeds": args.n_seeds,
        "phases": results,
        "all_gates": all_gates,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
