"""Exp 113: Stage 59 — SDM Learned Color Matching on ObstructedMaze-2Dl.

2 keys, 2 locked doors, agent must learn same_color(key, door) → opens.
Heuristic: random key choice → ~25% (50% per door × 2 doors).
SDM trained: correct key first → should be significantly higher.

Run on minipc:
  python src/snks/experiments/exp113_sdm_obstructed.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from snks.agent.sdm_obstructed_agent import SDMObstructedAgent, ObstructedMazeEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_episode(agent: SDMObstructedAgent, env: ObstructedMazeEnv,
                seed: int, max_steps: int = 500) -> dict:
    agent.reset_episode()
    img, col, row, d, carrying = env.reset(seed=seed)

    t0 = time.time()
    reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(img, col, row, d, carrying)
        img, reward, term, trunc, col, row, d, carrying = env.step(action)
        agent.observe_result(img, col, row, d, carrying, reward)

        if term or trunc:
            break

    elapsed = time.time() - t0
    success = reward > 0
    agent.episode_done(success=success)

    return {
        "seed": seed,
        "success": success,
        "steps": step + 1,
        "reward": round(reward, 4),
        "elapsed_s": round(elapsed, 3),
        "sdm_writes": agent.sdm.n_writes,
    }


def run_experiment(n_train: int = 100, n_eval: int = 200,
                   max_steps: int = 500) -> dict:
    print(f"=== Exp 113: SDM ObstructedMaze-2Dl ===")
    print(f"Train: {n_train} seeds, Eval: {n_eval} seeds, max_steps: {max_steps}")
    print(f"Device: {DEVICE}")

    env = ObstructedMazeEnv(max_steps=max_steps)

    # --- Training phase ---
    agent = SDMObstructedAgent(
        explore_episodes=n_train,
        device=DEVICE,
    )
    print(f"\n--- Training ({n_train} episodes) ---")
    t_train_start = time.time()
    train_results = []
    for i, seed in enumerate(range(n_train)):
        result = run_episode(agent, env, seed=seed, max_steps=max_steps)
        train_results.append(result)
        if (i + 1) % 10 == 0 or i == 0:
            rate = sum(1 for r in train_results if r["success"]) / len(train_results)
            eta = (time.time() - t_train_start) / (i + 1) * (n_train - i - 1)
            print(f"  Train {i+1}/{n_train}: rate={rate:.1%}, SDM writes={agent.sdm.n_writes}, ETA={eta:.0f}s")

    train_rate = sum(1 for r in train_results if r["success"]) / n_train
    train_time = time.time() - t_train_start
    print(f"Training done: {train_rate:.1%} success, {agent.sdm.n_writes} SDM writes, {train_time:.1f}s")

    # --- Eval: SDM trained ---
    print(f"\n--- Eval SDM trained ({n_eval} episodes) ---")
    eval_results = []
    for i, seed in enumerate(range(1000, 1000 + n_eval)):
        result = run_episode(agent, env, seed=seed, max_steps=max_steps)
        eval_results.append(result)
        if (i + 1) % 50 == 0:
            rate = sum(1 for r in eval_results if r["success"]) / len(eval_results)
            print(f"  Eval {i+1}/{n_eval}: rate={rate:.1%}")

    sdm_rate = sum(1 for r in eval_results if r["success"]) / n_eval
    print(f"SDM trained: {sdm_rate:.1%} ({sum(1 for r in eval_results if r['success'])}/{n_eval})")

    # --- Ablation: heuristic (exploration mode forever) ---
    print(f"\n--- Ablation: heuristic only ---")
    h_agent = SDMObstructedAgent(explore_episodes=999999, device=DEVICE)
    h_results = []
    for seed in range(1000, 1000 + n_eval):
        result = run_episode(h_agent, env, seed=seed, max_steps=max_steps)
        h_results.append(result)
    heuristic_rate = sum(1 for r in h_results if r["success"]) / n_eval
    print(f"Heuristic: {heuristic_rate:.1%} ({sum(1 for r in h_results if r['success'])}/{n_eval})")

    # --- Ablation: SDM untrained ---
    print(f"\n--- Ablation: SDM untrained ---")
    u_agent = SDMObstructedAgent(explore_episodes=0, device=DEVICE)
    u_results = []
    for seed in range(1000, 1000 + n_eval):
        result = run_episode(u_agent, env, seed=seed, max_steps=max_steps)
        u_results.append(result)
    untrained_rate = sum(1 for r in u_results if r["success"]) / n_eval
    print(f"Untrained: {untrained_rate:.1%} ({sum(1 for r in u_results if r['success'])}/{n_eval})")

    env.close()

    print(f"\n=== RESULTS ===")
    print(f"  SDM trained:  {sdm_rate:.1%}")
    print(f"  Heuristic:    {heuristic_rate:.1%}")
    print(f"  Untrained:    {untrained_rate:.1%}")
    print(f"  SDM writes:   {agent.sdm.n_writes}")
    print(f"  Delta:        {sdm_rate - heuristic_rate:+.1%}")
    gate = sdm_rate >= 0.50 and sdm_rate > heuristic_rate
    print(f"  Gate (≥50% AND > heuristic): {'PASS' if gate else 'FAIL'}")

    return {
        "env": "ObstructedMaze-2Dl",
        "n_train": n_train,
        "n_eval": n_eval,
        "max_steps": max_steps,
        "sdm_writes": agent.sdm.n_writes,
        "train_success_rate": round(train_rate, 4),
        "train_time_s": round(train_time, 1),
        "results": {
            "sdm_trained": {"success_rate": round(sdm_rate, 4), "n_success": sum(1 for r in eval_results if r["success"])},
            "heuristic": {"success_rate": round(heuristic_rate, 4), "n_success": sum(1 for r in h_results if r["success"])},
            "untrained": {"success_rate": round(untrained_rate, 4), "n_success": sum(1 for r in u_results if r["success"])},
        },
        "gate_pass": gate,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = run_experiment(
        n_train=args.n_train,
        n_eval=args.n_eval,
        max_steps=args.max_steps,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
