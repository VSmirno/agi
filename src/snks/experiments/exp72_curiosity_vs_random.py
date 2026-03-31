"""Experiment 72: Curiosity vs Random Exploration (Stage 29).

Direct comparison of _explore() behavior:
- CuriosityAgent._explore(): curiosity-guided navigation (prefers novel cells)
- Random baseline: random navigation actions for same budget

Both agents use the SAME step budget (EXPLORE_STEPS) in Empty-8x8.
We compare distinct (x,y) positions visited.

Gates:
    curious_positions / random_positions >= 1.3 (more unique cells visited)
    curious_coverage >= 0.25 (25% of passable cells)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import random
import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.curiosity_module import CuriosityModule


N_TRIALS = 5
EXPLORE_STEPS = 100  # direct exploration budget per trial


def count_passable_cells(env) -> int:
    uw = env.unwrapped
    count = 0
    for j in range(uw.grid.height):
        for i in range(uw.grid.width):
            cell = uw.grid.get(i, j)
            if cell is None or cell.type in ("empty", "goal"):
                count += 1
    return count


def _extract_positions(curiosity: CuriosityModule) -> set[tuple[int, int]]:
    """Extract (x,y) positions from curiosity state keys (token >= 10000)."""
    positions: set[tuple[int, int]] = set()
    for key in curiosity._counts:
        for tok in key:
            if 10000 <= tok < 20000:
                x = (tok - 10000) // 100
                y = (tok - 10000) % 100
                positions.add((x, y))
    return positions


def run_random_explore(env, steps: int) -> set[tuple[int, int]]:
    """Random navigation for given steps. Returns distinct positions visited."""
    visited: set[tuple[int, int]] = set()
    uw = env.unwrapped
    for _ in range(steps):
        visited.add(tuple(uw.agent_pos))
        action = random.randint(0, 2)  # turn left, turn right, forward
        obs, reward, terminated, truncated, _ = env.step(action)
        # Don't reset on termination — keep counting.
    return visited


def main():
    print("=" * 60)
    print("Experiment 72: Curiosity vs Random Exploration (Stage 29)")
    print("=" * 60)

    # Count passable cells.
    env_tmp = gym.make("MiniGrid-Empty-8x8-v0")
    env_tmp.reset(seed=0)
    passable = count_passable_cells(env_tmp)
    env_tmp.close()
    print(f"  Passable cells in Empty-8x8: {passable}")
    print(f"  Exploration budget: {EXPLORE_STEPS} steps per trial\n")

    curious_coverages = []
    ratios = []

    for trial in range(N_TRIALS):
        seed = 42 + trial

        # --- Curious agent: call _explore() directly ---
        env_c = gym.make("MiniGrid-Empty-8x8-v0")
        env_c.reset(seed=seed)
        agent = CuriosityAgent(env_c, curiosity_across_episodes=False)
        # Perceive initial state to populate _objects and _perception.
        uw_c = env_c.unwrapped
        agent._perception.perceive(
            uw_c.grid, tuple(uw_c.agent_pos), int(uw_c.agent_dir),
        )
        agent._explore(EXPLORE_STEPS)
        curious_positions = _extract_positions(agent.curiosity)
        n_curious = len(curious_positions)
        env_c.close()

        # --- Random agent: random nav actions ---
        env_r = gym.make("MiniGrid-Empty-8x8-v0")
        env_r.reset(seed=seed)
        random_positions = run_random_explore(env_r, EXPLORE_STEPS)
        n_random = len(random_positions)
        env_r.close()

        ratio = n_curious / max(n_random, 1)
        coverage = n_curious / max(passable, 1)
        curious_coverages.append(coverage)
        ratios.append(ratio)

        print(f"  trial={trial}: curious={n_curious} random={n_random} "
              f"ratio={ratio:.2f} coverage={coverage:.2f}")

    avg_ratio = sum(ratios) / N_TRIALS
    avg_coverage = sum(curious_coverages) / N_TRIALS

    print(f"\n--- Results ---")
    print(f"  Avg curious/random ratio: {avg_ratio:.2f}")
    print(f"  Avg curious coverage: {avg_coverage:.3f}")

    gate_ratio = avg_ratio >= 1.3
    gate_coverage = avg_coverage >= 0.25

    print(f"\n{'=' * 60}")
    print(f"GATE: distinct_ratio {'PASS' if gate_ratio else 'FAIL'} ({avg_ratio:.2f} >= 1.30)")
    print(f"GATE: coverage       {'PASS' if gate_coverage else 'FAIL'} ({avg_coverage:.3f} >= 0.250)")
    print(f"{'=' * 60}")

    if gate_ratio and gate_coverage:
        print(">>> Experiment 72: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 72: GATE FAIL <<<")


if __name__ == "__main__":
    main()
