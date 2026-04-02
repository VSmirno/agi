"""Exp 108: Multi-Room Navigation — Stage 49 gate experiments.

108a: BFS pathfinding correctness on 50 random MultiRoom-N3 layouts
108b: MultiRoomNavigator success rate on 200 random MultiRoom-N3 layouts
108c: Average steps analysis
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from snks.agent.multi_room_nav import (
    MultiRoomEnvWrapper,
    MultiRoomNavigator,
    find_objects,
)
from snks.agent.pathfinding import GridPathfinder


def exp108a_bfs_pathfinding(n_layouts: int = 50) -> dict:
    """Test BFS pathfinding on random MultiRoom-N3 layouts.

    Gate: 100% path found from agent to goal.
    """
    print(f"\n=== Exp 108a: BFS pathfinding ({n_layouts} layouts) ===")
    pf = GridPathfinder()
    paths_found = 0
    path_lengths = []

    for seed in range(n_layouts):
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6)
        obs = env.reset(seed=seed)
        objs = find_objects(obs)

        if objs["agent_pos"] is None or objs["goal_pos"] is None:
            print(f"  Seed {seed}: SKIP (no agent or goal)")
            continue

        path = pf.find_path(obs, objs["agent_pos"], objs["goal_pos"],
                            allow_door=True)
        if path is not None:
            paths_found += 1
            path_lengths.append(len(path))
        else:
            print(f"  Seed {seed}: NO PATH")

    rate = paths_found / n_layouts
    mean_len = np.mean(path_lengths) if path_lengths else 0
    gate_pass = rate == 1.0

    print(f"  Paths found: {paths_found}/{n_layouts} ({rate:.0%})")
    print(f"  Mean path length: {mean_len:.1f}")
    print(f"  Gate (100%): {'PASS' if gate_pass else 'FAIL'}")

    return {
        "name": "exp108a_bfs",
        "paths_found": paths_found,
        "total": n_layouts,
        "rate": rate,
        "mean_path_length": float(mean_len),
        "gate": "PASS" if gate_pass else "FAIL",
    }


def exp108b_navigation(n_episodes: int = 200, max_steps: int = 500) -> dict:
    """Test MultiRoomNavigator on random MultiRoom-N3 layouts.

    Gate: ≥60% success rate.
    """
    print(f"\n=== Exp 108b: Navigation ({n_episodes} episodes) ===")
    successes = 0
    steps_list = []
    rewards = []
    failures = []
    t0 = time.time()

    for seed in range(n_episodes):
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6, max_steps=max_steps)
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = env.reset(seed=seed)
        success, steps, reward = nav.run_episode(env, obs, max_steps=max_steps)

        if success:
            successes += 1
            steps_list.append(steps)
            rewards.append(reward)
        else:
            failures.append(seed)

        if (seed + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = successes / (seed + 1)
            eta = elapsed / (seed + 1) * (n_episodes - seed - 1)
            print(f"  [{seed+1}/{n_episodes}] success={rate:.1%} "
                  f"elapsed={elapsed:.1f}s ETA={eta:.1f}s")

    elapsed = time.time() - t0
    rate = successes / n_episodes
    mean_steps = np.mean(steps_list) if steps_list else 0
    mean_reward = np.mean(rewards) if rewards else 0
    gate_pass = rate >= 0.60

    print(f"\n  Success: {successes}/{n_episodes} ({rate:.1%})")
    print(f"  Mean steps (success): {mean_steps:.1f}")
    print(f"  Mean reward (success): {mean_reward:.3f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/n_episodes:.2f}s/ep)")
    if failures and len(failures) <= 20:
        print(f"  Failed seeds: {failures}")
    print(f"  Gate (≥60%): {'PASS' if gate_pass else 'FAIL'}")

    return {
        "name": "exp108b_navigation",
        "successes": successes,
        "total": n_episodes,
        "rate": rate,
        "mean_steps": float(mean_steps),
        "mean_reward": float(mean_reward),
        "time_s": elapsed,
        "failed_seeds": failures[:20],
        "gate": "PASS" if gate_pass else "FAIL",
    }


def exp108c_steps_analysis(n_episodes: int = 200, max_steps: int = 500) -> dict:
    """Analyze step distribution for successful episodes.

    Gate: mean steps ≤ 150.
    """
    print(f"\n=== Exp 108c: Steps analysis ({n_episodes} episodes) ===")
    steps_list = []

    for seed in range(n_episodes):
        env = MultiRoomEnvWrapper(n_rooms=3, max_room_size=6, max_steps=max_steps)
        nav = MultiRoomNavigator(epsilon=0.0)
        obs = env.reset(seed=seed)
        success, steps, _ = nav.run_episode(env, obs, max_steps=max_steps)
        if success:
            steps_list.append(steps)

    if not steps_list:
        print("  No successful episodes!")
        return {"name": "exp108c_steps", "gate": "FAIL"}

    mean_steps = np.mean(steps_list)
    max_steps_val = max(steps_list)
    min_steps = min(steps_list)
    p95 = np.percentile(steps_list, 95)
    gate_pass = mean_steps <= 150

    print(f"  Successful episodes: {len(steps_list)}/{n_episodes}")
    print(f"  Steps: mean={mean_steps:.1f}, min={min_steps}, max={max_steps_val}, p95={p95:.0f}")
    print(f"  Gate (mean ≤ 150): {'PASS' if gate_pass else 'FAIL'}")

    return {
        "name": "exp108c_steps",
        "n_success": len(steps_list),
        "mean_steps": float(mean_steps),
        "min_steps": int(min_steps),
        "max_steps": int(max_steps_val),
        "p95_steps": float(p95),
        "gate": "PASS" if gate_pass else "FAIL",
    }


if __name__ == "__main__":
    results = []

    r108a = exp108a_bfs_pathfinding(n_layouts=50)
    results.append(r108a)

    r108b = exp108b_navigation(n_episodes=200)
    results.append(r108b)

    r108c = exp108c_steps_analysis(n_episodes=200)
    results.append(r108c)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']}: {r['gate']}")

    all_pass = all(r["gate"] == "PASS" for r in results)
    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAIL'}")

    # Save results
    out_path = Path("_docs/exp108_results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {out_path}")
