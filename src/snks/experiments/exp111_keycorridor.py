"""Exp 111: Stage 57 — Long Subgoal Chains on KeyCorridor environments.

Gate criteria:
- KeyCorridorS4R3: ≥40% success on 200 random seeds, subgoal chain ≥5
- KeyCorridorS3R3: ≥50% success on 200 random seeds
- Mean steps (successful) ≤ 300 for S4R3

Run on minipc: python src/snks/experiments/exp111_keycorridor.py
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from snks.agent.keycorridor_agent import KeyCorridorAgent, KeyCorridorEnv


def run_episode(env_name: str, seed: int, max_steps: int = 480) -> dict:
    """Run one episode. Returns result dict."""
    env = KeyCorridorEnv(env_name)
    obs, col, row, d, mission = env.reset(seed=seed)
    agent = KeyCorridorAgent(env.grid_width, env.grid_height, mission)

    t0 = time.time()
    for step in range(max_steps):
        ct = env.carrying_type_color
        if ct is not None:
            agent.update_carrying(ct[0], ct[1])
        else:
            agent.clear_carrying()

        action = agent.select_action(obs, col, row, d)
        obs, reward, term, trunc, col, row, d = env.step(action)
        agent.observe_result(obs, col, row, d, reward)

        if term or trunc:
            break

    elapsed = time.time() - t0
    env.close()

    return {
        "seed": seed,
        "success": reward > 0,
        "steps": step + 1,
        "subgoals_completed": agent.subgoals_completed,
        "completed_names": agent._completed_names,
        "reward": reward,
        "elapsed_s": round(elapsed, 3),
    }


def run_experiment(env_name: str, n_seeds: int = 200, max_steps: int = 480) -> dict:
    """Run experiment on n_seeds. Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"Exp 111: {env_name} — {n_seeds} seeds, max_steps={max_steps}")
    print(f"{'='*60}")

    results = []
    successes = 0
    total_steps = 0
    total_subgoals = 0
    t0 = time.time()

    for i in range(n_seeds):
        r = run_episode(env_name, seed=i, max_steps=max_steps)
        results.append(r)
        if r["success"]:
            successes += 1
            total_steps += r["steps"]
        total_subgoals += r["subgoals_completed"]

        if (i + 1) % 20 == 0 or i == 0:
            rate = successes / (i + 1)
            mean_subs = total_subgoals / (i + 1)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_seeds - i - 1)
            print(f"  [{i+1:3d}/{n_seeds}] success={rate:.1%} "
                  f"mean_subgoals={mean_subs:.1f} "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    total_elapsed = time.time() - t0
    success_rate = successes / n_seeds
    mean_steps_success = total_steps / max(successes, 1)
    mean_subgoals = total_subgoals / n_seeds

    # Find min subgoal count across successful episodes
    successful_results = [r for r in results if r["success"]]
    min_subgoals = min((r["subgoals_completed"] for r in successful_results), default=0)
    max_subgoals = max((r["subgoals_completed"] for r in successful_results), default=0)

    summary = {
        "env": env_name,
        "n_seeds": n_seeds,
        "max_steps": max_steps,
        "success_rate": round(success_rate, 4),
        "successes": successes,
        "mean_steps_success": round(mean_steps_success, 1),
        "mean_subgoals": round(mean_subgoals, 1),
        "min_subgoals_success": min_subgoals,
        "max_subgoals_success": max_subgoals,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    print(f"\n  RESULT: {success_rate:.1%} ({successes}/{n_seeds})")
    print(f"  Mean steps (success): {mean_steps_success:.1f}")
    print(f"  Mean subgoals: {mean_subgoals:.1f}")
    print(f"  Subgoals range (success): {min_subgoals}-{max_subgoals}")
    print(f"  Total time: {total_elapsed:.1f}s")

    return {"summary": summary, "episodes": results}


def main():
    all_results = {}

    # Exp 111a: KeyCorridorS4R3 (primary gate)
    r = run_experiment("BabyAI-KeyCorridorS4R3-v0", n_seeds=200, max_steps=480)
    all_results["exp111a_s4r3"] = r

    # Exp 111b: KeyCorridorS3R3
    r = run_experiment("BabyAI-KeyCorridorS3R3-v0", n_seeds=200, max_steps=270)
    all_results["exp111b_s3r3"] = r

    # Exp 111c: KeyCorridorS5R3 (stretch)
    r = run_experiment("BabyAI-KeyCorridorS5R3-v0", n_seeds=200, max_steps=750)
    all_results["exp111c_s5r3"] = r

    # Save results
    out_path = Path("_docs/exp111_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Gate check
    print("\n" + "=" * 60)
    print("GATE CHECK")
    print("=" * 60)
    s4 = all_results["exp111a_s4r3"]["summary"]
    s3 = all_results["exp111b_s3r3"]["summary"]
    print(f"  S4R3: {s4['success_rate']:.1%} (gate ≥40%) — {'PASS' if s4['success_rate'] >= 0.40 else 'FAIL'}")
    print(f"  S3R3: {s3['success_rate']:.1%} (gate ≥50%) — {'PASS' if s3['success_rate'] >= 0.50 else 'FAIL'}")
    print(f"  S4R3 mean steps: {s4['mean_steps_success']:.0f} (gate ≤300) — "
          f"{'PASS' if s4['mean_steps_success'] <= 300 else 'FAIL'}")
    print(f"  Min subgoals (S4R3): {s4['min_subgoals_success']} (gate ≥5) — "
          f"{'PASS' if s4['min_subgoals_success'] >= 5 else 'FAIL'}")


if __name__ == "__main__":
    main()
