#!/usr/bin/env python3
"""Exp117: BossLevel integration test (Stage 62).

117a: Full BossLevel, 50 seeds, gate ≥50%
117b: Ablation — trained vs untrained MissionModel
117c: Per-mission-type breakdown (diagnostic)
"""

import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np

from snks.agent.boss_level_agent import BossLevelAgent

DEMO_PATH = Path("_docs/demo_episodes_bosslevel.json")


def run_episode(agent: BossLevelAgent, seed: int) -> dict:
    """Run one BossLevel episode, return result dict."""
    env = gym.make("BabyAI-BossLevel-v0")
    obs, _ = env.reset(seed=seed)
    uw = env.unwrapped
    mission = obs["mission"]

    agent.spatial_map = type(agent.spatial_map)(uw.grid.width, uw.grid.height)
    agent.reset(mission)

    done = False
    step = 0
    t0 = time.time()

    while not done and step < uw.max_steps:
        action = agent.select_action(
            obs["image"],
            int(uw.agent_pos[0]),
            int(uw.agent_pos[1]),
            int(uw.agent_dir),
        )
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        carrying = uw.carrying
        agent.observe_result(
            obs["image"],
            int(uw.agent_pos[0]),
            int(uw.agent_pos[1]),
            int(uw.agent_dir),
            float(reward),
            carrying.type if carrying else None,
            carrying.color if carrying else None,
        )
        step += 1

    elapsed = time.time() - t0
    stats = agent.get_stats()
    env.close()

    return {
        "seed": seed,
        "mission": mission,
        "success": reward > 0,
        "steps": step,
        "max_steps": uw.max_steps,
        "elapsed_s": round(elapsed, 2),
        "subgoals_completed": stats["subgoals_completed"],
        "total_subgoals": stats["total_subgoals"],
        "explore_steps": stats["explore_steps"],
        "execute_steps": stats["execute_steps"],
    }


def classify_mission(mission: str) -> str:
    """Classify mission complexity."""
    connectives = sum(1 for w in ["and", "then", "after"]
                      if f" {w} " in mission)
    if connectives == 0:
        return "simple"
    elif connectives == 1:
        return "compound"
    else:
        return "complex"


def exp117a(agent: BossLevelAgent, n_seeds: int = 50) -> dict:
    """Main gate: full BossLevel."""
    print(f"\n{'='*60}")
    print(f"EXP117a: BossLevel Full ({n_seeds} seeds)")
    print(f"{'='*60}")

    results = []
    for seed in range(n_seeds):
        r = run_episode(agent, seed)
        results.append(r)
        status = "OK" if r["success"] else "FAIL"
        print(f"  {seed:3d}: {status} ({r['steps']:4d}/{r['max_steps']}) "
              f"{r['subgoals_completed']}/{r['total_subgoals']} "
              f"({r['elapsed_s']:.1f}s) — {r['mission'][:50]}")

    successes = sum(1 for r in results if r["success"])
    rate = successes / n_seeds * 100
    avg_steps = np.mean([r["steps"] for r in results if r["success"]]) if successes else 0
    sg_done = sum(r["subgoals_completed"] for r in results)
    sg_total = sum(r["total_subgoals"] for r in results)

    print(f"\nGATE: {successes}/{n_seeds} = {rate:.0f}% (need ≥50%)")
    print(f"{'PASS' if rate >= 50 else 'FAIL'}")
    print(f"Mean steps (success): {avg_steps:.0f}")
    print(f"Subgoal completion: {sg_done}/{sg_total} = {sg_done*100/sg_total:.0f}%")

    return {
        "phase": "117a",
        "n_seeds": n_seeds,
        "successes": successes,
        "rate_pct": round(rate, 1),
        "gate_pass": rate >= 50,
        "avg_steps_success": round(avg_steps, 1),
        "subgoal_rate_pct": round(sg_done * 100 / sg_total, 1) if sg_total else 0,
        "results": results,
    }


def exp117b(demos: list[dict], n_seeds: int = 20) -> dict:
    """Ablation: trained vs untrained MissionModel."""
    print(f"\n{'='*60}")
    print(f"EXP117b: Ablation ({n_seeds} seeds)")
    print(f"{'='*60}")

    # Trained agent
    trained = BossLevelAgent(grid_width=22, grid_height=22)
    trained.train(demos)
    trained_results = [run_episode(trained, seed) for seed in range(n_seeds)]
    trained_rate = sum(1 for r in trained_results if r["success"]) / n_seeds * 100

    # Untrained agent (no demos, just causal rules)
    untrained = BossLevelAgent(grid_width=22, grid_height=22)
    untrained.causal_model.learn_all_rules(
        ["red", "green", "blue", "purple", "yellow", "grey"]
    )
    untrained._trained = True
    untrained_results = [run_episode(untrained, seed) for seed in range(n_seeds)]
    untrained_rate = sum(1 for r in untrained_results if r["success"]) / n_seeds * 100

    delta = trained_rate - untrained_rate

    print(f"\nTrained: {trained_rate:.0f}%")
    print(f"Untrained: {untrained_rate:.0f}%")
    print(f"Delta: {delta:.0f}% (need ≥30%)")
    print(f"{'PASS' if delta >= 30 else 'FAIL'}")

    return {
        "phase": "117b",
        "trained_rate_pct": round(trained_rate, 1),
        "untrained_rate_pct": round(untrained_rate, 1),
        "delta_pct": round(delta, 1),
        "gate_pass": delta >= 30,
    }


def exp117c(results_117a: list[dict]) -> dict:
    """Per-mission-type breakdown (diagnostic)."""
    print(f"\n{'='*60}")
    print("EXP117c: Per-Mission-Type Breakdown")
    print(f"{'='*60}")

    by_type = {}
    for r in results_117a:
        mt = classify_mission(r["mission"])
        if mt not in by_type:
            by_type[mt] = {"total": 0, "success": 0}
        by_type[mt]["total"] += 1
        if r["success"]:
            by_type[mt]["success"] += 1

    for mt, stats in sorted(by_type.items()):
        rate = stats["success"] / stats["total"] * 100
        print(f"  {mt:10s}: {stats['success']}/{stats['total']} = {rate:.0f}%")

    return {"phase": "117c", "breakdown": by_type}


def main():
    print("Loading demos...")
    with open(DEMO_PATH) as f:
        demos = json.load(f)

    agent = BossLevelAgent(grid_width=22, grid_height=22)
    stats = agent.train(demos)
    print(f"Trained: {stats}")

    # Phase A
    result_a = exp117a(agent)

    # Phase B
    result_b = exp117b(demos)

    # Phase C
    result_c = exp117c(result_a["results"])

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"117a: {result_a['rate_pct']}% ({'PASS' if result_a['gate_pass'] else 'FAIL'})")
    print(f"117b: delta={result_b['delta_pct']}% ({'PASS' if result_b['gate_pass'] else 'FAIL'})")
    print(f"117c: diagnostic only")

    all_pass = result_a["gate_pass"] and result_b["gate_pass"]
    print(f"\nALL GATES: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
