"""Exp 112: Stage 58 — SDM Learned Agent on DoorKey-5x5 with Partial Obs.

Phase 1 (Learning Budget): exploration episodes to fill SDM
Phase 2 (Evaluation): 200 seeds, compare with random baseline

Gate: ≥30% success on 200 seeds + ≥1000 SDM transitions

Run on minipc: python src/snks/experiments/exp112_sdm_doorkey.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from snks.agent.sdm_doorkey_agent import SDMDoorKeyAgent, SDMDoorKeyEnv


def run_episode(agent: SDMDoorKeyAgent, seed: int,
                max_steps: int = 200) -> dict:
    env = SDMDoorKeyEnv(size=5, max_steps=max_steps)
    obs, col, row, d, has_key, door_state = env.reset(seed=seed)
    agent.reset_episode()

    t0 = time.time()
    reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(obs, col, row, d, has_key, door_state)
        obs, reward, term, trunc, col, row, d, has_key, door_state = env.step(action)
        agent.observe_result(obs, col, row, d, has_key, door_state, reward)

        if term or trunc:
            break

    elapsed = time.time() - t0
    success = reward > 0
    agent._episode_done(success=success)
    env.close()

    return {
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "reward": round(reward, 4),
        "elapsed_s": round(elapsed, 3),
        "exploring": agent._exploring,
        "sdm_writes": agent.sdm.n_writes,
    }


def run_experiment(explore_episodes: int = 50, eval_seeds: int = 200,
                   max_steps: int = 200) -> dict:
    print(f"\n{'='*60}")
    print(f"Exp 112: SDM DoorKey-5x5 Partial Obs")
    print(f"  Exploration: {explore_episodes} episodes")
    print(f"  Evaluation: {eval_seeds} seeds")
    print(f"{'='*60}")

    agent = SDMDoorKeyAgent(
        grid_width=5, grid_height=5,
        dim=512, n_locations=5000,
        explore_episodes=explore_episodes,
        epsilon=0.15,
    )

    # Phase 1: Learning Budget — exploration
    print(f"\n--- Phase 1: Exploration ({explore_episodes} episodes) ---")
    explore_successes = 0
    t0 = time.time()
    for i in range(explore_episodes):
        # Use different seeds for exploration diversity
        r = run_episode(agent, seed=i + 1000, max_steps=max_steps)
        if r["success"]:
            explore_successes += 1
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (explore_episodes - i - 1)
            print(f"  [{i+1:3d}/{explore_episodes}] "
                  f"explore_success={explore_successes}/{i+1} "
                  f"sdm_writes={agent.sdm.n_writes} "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    explore_time = time.time() - t0
    print(f"\n  Exploration done: {explore_successes}/{explore_episodes} successes")
    print(f"  SDM writes: {agent.sdm.n_writes}")
    print(f"  Time: {explore_time:.1f}s")

    # Phase 2: Evaluation — 200 seeds
    print(f"\n--- Phase 2: Evaluation ({eval_seeds} seeds) ---")
    eval_results = []
    eval_successes = 0
    t0 = time.time()
    for i in range(eval_seeds):
        r = run_episode(agent, seed=i, max_steps=max_steps)
        eval_results.append(r)
        if r["success"]:
            eval_successes += 1
        if (i + 1) % 20 == 0:
            rate = eval_successes / (i + 1)
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (eval_seeds - i - 1)
            print(f"  [{i+1:3d}/{eval_seeds}] success={rate:.1%} "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    eval_time = time.time() - t0
    success_rate = eval_successes / eval_seeds

    successful = [r for r in eval_results if r["success"]]
    mean_steps = sum(r["steps"] for r in successful) / max(len(successful), 1)

    # Phase 3: Random baseline (0 exploration episodes)
    print(f"\n--- Phase 3: Random Baseline (0 exploration) ---")
    random_agent = SDMDoorKeyAgent(
        grid_width=5, grid_height=5, explore_episodes=0, epsilon=1.0,
    )
    random_agent._exploring = False  # force planning mode (but SDM empty → random)
    random_successes = 0
    for i in range(eval_seeds):
        r = run_episode(random_agent, seed=i, max_steps=max_steps)
        if r["success"]:
            random_successes += 1
    random_rate = random_successes / eval_seeds
    print(f"  Random baseline: {random_rate:.1%} ({random_successes}/{eval_seeds})")

    summary = {
        "explore_episodes": explore_episodes,
        "explore_successes": explore_successes,
        "sdm_writes": agent.sdm.n_writes,
        "eval_seeds": eval_seeds,
        "eval_success_rate": round(success_rate, 4),
        "eval_successes": eval_successes,
        "eval_mean_steps_success": round(mean_steps, 1),
        "random_baseline_rate": round(random_rate, 4),
        "explore_time_s": round(explore_time, 1),
        "eval_time_s": round(eval_time, 1),
    }

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  SDM writes: {agent.sdm.n_writes} (gate ≥1000) — "
          f"{'PASS' if agent.sdm.n_writes >= 1000 else 'FAIL'}")
    print(f"  Learned agent: {success_rate:.1%} ({eval_successes}/{eval_seeds}) "
          f"(gate ≥30%) — {'PASS' if success_rate >= 0.30 else 'FAIL'}")
    print(f"  Random baseline: {random_rate:.1%}")
    print(f"  Improvement over random: {success_rate - random_rate:+.1%}")
    print(f"  Mean steps (success): {mean_steps:.1f}")
    print(f"  Symbolic baseline (Stage 54): 100%")

    return {"summary": summary, "episodes": eval_results}


def main():
    # Try different exploration budgets
    results = {}

    # Main experiment: 50 exploration episodes
    r = run_experiment(explore_episodes=50, eval_seeds=200)
    results["exp112a_50ep"] = r

    # Ablation: 100 exploration episodes
    r = run_experiment(explore_episodes=100, eval_seeds=200)
    results["exp112b_100ep"] = r

    # Save results
    out_path = Path("_docs/exp112_results.json")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
