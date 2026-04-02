"""Stage 56: PutNext gate experiments.

exp110a: PutNextLocalS5N3 — 200 seeds, gate ≥50%
exp110b: PutNextS6N3 — 200 seeds, gate ≥50%, 5+ unique object types
exp110c: Object type coverage — count distinct (type, color) pairs
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')
import minigrid
minigrid.register_minigrid_envs()

from snks.agent.putnext_agent import PutNextAgent, PutNextEnv


def run_experiment(env_name: str, n_seeds: int = 200,
                   max_steps: int = 288) -> dict:
    """Run PutNext agent on n_seeds and return results."""
    env = PutNextEnv(env_name=env_name)
    successes = 0
    total_steps_success = 0
    object_types_seen: set[tuple[int, int]] = set()
    failures: list[int] = []

    t0 = time.time()
    for seed in range(n_seeds):
        obs, col, row, d, mission = env.reset(seed=seed)
        agent = PutNextAgent(
            env.grid_width, env.grid_height, mission, epsilon=0.0
        )

        # Track objects
        object_types_seen.add(agent.source)
        object_types_seen.add(agent.target)

        for step in range(max_steps):
            carrying_tc = env.carrying_type_color
            agent.update_carrying(carrying_tc)
            action = agent.select_action(obs, col, row, d)
            obs, reward, term, trunc, col, row, d = env.step(action)
            agent.observe_result(obs, col, row, d, reward)

            if term:
                if reward > 0:
                    successes += 1
                    total_steps_success += step + 1
                else:
                    failures.append(seed)
                break
            if trunc:
                failures.append(seed)
                break

        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = successes / (seed + 1)
            eta = elapsed / (seed + 1) * (n_seeds - seed - 1)
            print(f"  [{seed+1}/{n_seeds}] rate={rate:.1%} "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    elapsed = time.time() - t0
    env.close()

    rate = successes / n_seeds
    mean_steps = total_steps_success / successes if successes > 0 else float('inf')

    return {
        "env": env_name,
        "n_seeds": n_seeds,
        "successes": successes,
        "success_rate": rate,
        "mean_steps_success": round(mean_steps, 1),
        "n_unique_object_types": len(object_types_seen),
        "object_types": sorted([list(t) for t in object_types_seen]),
        "elapsed_seconds": round(elapsed, 1),
        "first_failures": failures[:10],
    }


def main():
    results = {}

    # exp110a: PutNextLocalS5N3 (small, 5x5, 3 objects)
    print("=" * 60)
    print("exp110a: BabyAI-PutNextLocalS5N3-v0 (200 seeds)")
    print("=" * 60)
    r = run_experiment('BabyAI-PutNextLocalS5N3-v0', n_seeds=200)
    results["110a"] = r
    gate_a = r["success_rate"] >= 0.50
    print(f"  Result: {r['successes']}/{r['n_seeds']} = {r['success_rate']:.1%} "
          f"(gate ≥50%: {'PASS' if gate_a else 'FAIL'})")
    print(f"  Mean steps: {r['mean_steps_success']}")
    print(f"  Unique object types: {r['n_unique_object_types']}")
    print()

    # exp110b: PutNextS6N3 (larger, 11x6, 6 objects)
    print("=" * 60)
    print("exp110b: BabyAI-PutNextS6N3-v0 (200 seeds)")
    print("=" * 60)
    r = run_experiment('BabyAI-PutNextS6N3-v0', n_seeds=200)
    results["110b"] = r
    gate_b = r["success_rate"] >= 0.50
    print(f"  Result: {r['successes']}/{r['n_seeds']} = {r['success_rate']:.1%} "
          f"(gate ≥50%: {'PASS' if gate_b else 'FAIL'})")
    print(f"  Mean steps: {r['mean_steps_success']}")
    print(f"  Unique object types: {r['n_unique_object_types']}")
    print()

    # exp110c: Object type diversity
    all_types = set()
    for key in ("110a", "110b"):
        for t in results[key]["object_types"]:
            all_types.add(tuple(t))
    results["110c"] = {
        "total_unique_object_types": len(all_types),
        "types": sorted([list(t) for t in all_types]),
        "gate_5plus": len(all_types) >= 5,
    }
    print(f"exp110c: Total unique object types across experiments: {len(all_types)}")
    print(f"  Gate ≥5: {'PASS' if len(all_types) >= 5 else 'FAIL'}")

    # Save results
    out_path = Path("_docs/exp110_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
