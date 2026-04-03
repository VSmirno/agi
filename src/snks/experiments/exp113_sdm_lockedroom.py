"""Exp 113: Stage 59 — SDM Learned Color Matching on LockedRoom.

Phase B: with mission text (SDM + parsed hint)
Phase A: without mission text (pure SDM learning from experience)

Gate B: ≥70% on 200 unseen seeds
Gate A: ≥50% on 200 unseen seeds
Both: SDM trained > heuristic, p < 0.05

Run on minipc:
  python src/snks/experiments/exp113_sdm_lockedroom.py --phase B
  python src/snks/experiments/exp113_sdm_lockedroom.py --phase A
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from snks.agent.sdm_lockedroom_agent import SDMLockedRoomAgent, LockedRoomEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_episode(agent: SDMLockedRoomAgent, env: LockedRoomEnv,
                seed: int, max_steps: int = 1000) -> dict:
    agent.reset_episode()
    img, col, row, d, carrying, mission = env.reset(seed=seed)

    t0 = time.time()
    reward = 0.0
    for step in range(max_steps):
        action = agent.select_action(img, col, row, d, carrying, mission)
        img, reward, term, trunc, col, row, d, carrying, mission = env.step(action)
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


def run_experiment(phase: str, n_train: int, n_eval: int,
                   max_steps: int = 1000) -> dict:
    use_mission = (phase == "B")
    print(f"=== Exp 113 Phase {phase} ({'mission' if use_mission else 'no mission'}) ===")
    print(f"Train: {n_train} seeds, Eval: {n_eval} seeds, max_steps: {max_steps}")
    print(f"Device: {DEVICE}")

    env = LockedRoomEnv(max_steps=max_steps)

    # --- Training phase ---
    agent = SDMLockedRoomAgent(
        explore_episodes=n_train,
        use_mission=use_mission,
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

    # --- Eval phase: SDM trained ---
    print(f"\n--- Eval SDM trained ({n_eval} episodes) ---")
    t_eval_start = time.time()
    eval_results = []
    for i, seed in enumerate(range(1000, 1000 + n_eval)):
        result = run_episode(agent, env, seed=seed, max_steps=max_steps)
        eval_results.append(result)
        if (i + 1) % 50 == 0:
            rate = sum(1 for r in eval_results if r["success"]) / len(eval_results)
            print(f"  Eval {i+1}/{n_eval}: rate={rate:.1%}")

    sdm_rate = sum(1 for r in eval_results if r["success"]) / n_eval
    eval_time = time.time() - t_eval_start
    print(f"SDM trained: {sdm_rate:.1%} ({sum(1 for r in eval_results if r['success'])}/{n_eval}), {eval_time:.1f}s")

    # --- Ablation: heuristic only (no SDM, random door choice) ---
    print(f"\n--- Ablation: heuristic only ---")
    heuristic_agent = SDMLockedRoomAgent(
        explore_episodes=999999,  # always exploring = heuristic mode
        use_mission=use_mission,
        device=DEVICE,
    )
    heuristic_results = []
    for seed in range(1000, 1000 + n_eval):
        result = run_episode(heuristic_agent, env, seed=seed, max_steps=max_steps)
        heuristic_results.append(result)

    heuristic_rate = sum(1 for r in heuristic_results if r["success"]) / n_eval
    print(f"Heuristic: {heuristic_rate:.1%} ({sum(1 for r in heuristic_results if r['success'])}/{n_eval})")

    # --- Ablation: SDM untrained ---
    print(f"\n--- Ablation: SDM untrained ---")
    untrained_agent = SDMLockedRoomAgent(
        explore_episodes=0,  # immediately planning with empty SDM
        use_mission=use_mission,
        device=DEVICE,
    )
    untrained_results = []
    for seed in range(1000, 1000 + n_eval):
        result = run_episode(untrained_agent, env, seed=seed, max_steps=max_steps)
        untrained_results.append(result)

    untrained_rate = sum(1 for r in untrained_results if r["success"]) / n_eval
    print(f"Untrained: {untrained_rate:.1%} ({sum(1 for r in untrained_results if r['success'])}/{n_eval})")

    env.close()

    # --- Summary ---
    print(f"\n=== RESULTS Phase {phase} ===")
    print(f"  SDM trained:  {sdm_rate:.1%}")
    print(f"  Heuristic:    {heuristic_rate:.1%}")
    print(f"  Untrained:    {untrained_rate:.1%}")
    print(f"  SDM writes:   {agent.sdm.n_writes}")
    gate = 0.70 if phase == "B" else 0.50
    passed = sdm_rate >= gate
    print(f"  Gate (≥{gate:.0%}): {'PASS' if passed else 'FAIL'}")

    return {
        "phase": phase,
        "use_mission": use_mission,
        "n_train": n_train,
        "n_eval": n_eval,
        "max_steps": max_steps,
        "sdm_writes": agent.sdm.n_writes,
        "train_success_rate": round(train_rate, 4),
        "train_time_s": round(train_time, 1),
        "results": {
            "sdm_trained": {"success_rate": round(sdm_rate, 4), "n_success": sum(1 for r in eval_results if r["success"])},
            "heuristic": {"success_rate": round(heuristic_rate, 4), "n_success": sum(1 for r in heuristic_results if r["success"])},
            "untrained": {"success_rate": round(untrained_rate, 4), "n_success": sum(1 for r in untrained_results if r["success"])},
        },
        "gate": gate,
        "gate_pass": passed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B"], required=True)
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_train = args.n_train or (50 if args.phase == "B" else 100)
    result = run_experiment(
        phase=args.phase,
        n_train=n_train,
        n_eval=args.n_eval,
        max_steps=args.max_steps,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
