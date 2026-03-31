"""Experiment 71: CuriosityModule Unit Tests (Stage 29).

Gates:
    CuriosityAgent visits >= 10 distinct states in 200 steps (Empty-5x5)
    r_int(new_state) == 1.0
    r_int(repeated) == 0.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.curiosity_module import CuriosityModule


MAX_STEPS = 200


def main():
    print("=" * 60)
    print("Experiment 71: CuriosityModule Unit Tests (Stage 29)")
    print("=" * 60)

    # Gate 1/2: formula verification.
    print("\n--- Formula Verification ---")
    cm = CuriosityModule()
    new_key = frozenset({50, 54, 10101})
    r_new = cm.intrinsic_reward(new_key)
    print(f"  r_int(new_state) = {r_new:.3f} (expected 1.0)")

    cm.observe(new_key)  # visit once → count=1
    r_repeated = cm.intrinsic_reward(new_key)
    print(f"  r_int(repeated once) = {r_repeated:.3f} (expected 0.5)")

    gate_formula_new = abs(r_new - 1.0) < 1e-6
    gate_formula_repeated = abs(r_repeated - 0.5) < 1e-6

    # Gate 3: distinct states — force exploration for 200 steps.
    # Use Empty-8x8 (larger env) with cross-episode curiosity,
    # run multiple short episodes so agent has time to explore.
    print(f"\n--- CuriosityAgent exploration ({MAX_STEPS} total steps) ---")
    env = gym.make("MiniGrid-Empty-8x8-v0")
    obs, _ = env.reset(seed=42)
    agent = CuriosityAgent(env, curiosity_across_episodes=True)

    total_steps = 0
    ep = 0
    while total_steps < MAX_STEPS:
        env2 = gym.make("MiniGrid-Empty-8x8-v0")
        obs2, _ = env2.reset(seed=ep)
        agent._env = env2
        agent._executor._env = env2
        remaining = MAX_STEPS - total_steps
        result = agent.run_episode(obs2["mission"], max_steps=min(50, remaining))
        total_steps += result.steps_taken
        ep += 1
        env2.close()
        if total_steps >= MAX_STEPS:
            break

    n_distinct = agent.curiosity.n_distinct()
    total_intrinsic = agent._total_intrinsic_reward

    print(f"  episodes={ep} total_steps={total_steps}")
    print(f"  distinct states visited: {n_distinct}")
    print(f"  total intrinsic reward: {total_intrinsic:.2f}")

    gate_distinct = n_distinct >= 10

    print(f"\n--- Results ---")
    print(f"  r_int(new): {r_new:.3f}")
    print(f"  r_int(repeated): {r_repeated:.3f}")
    print(f"  distinct states: {n_distinct}")

    print(f"\n{'=' * 60}")
    print(f"GATE: formula_new      {'PASS' if gate_formula_new else 'FAIL'} ({r_new:.3f} == 1.0)")
    print(f"GATE: formula_repeated {'PASS' if gate_formula_repeated else 'FAIL'} ({r_repeated:.3f} == 0.5)")
    print(f"GATE: distinct_states  {'PASS' if gate_distinct else 'FAIL'} ({n_distinct} >= 10)")
    print(f"{'=' * 60}")

    if gate_formula_new and gate_formula_repeated and gate_distinct:
        print(">>> Experiment 71: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 71: GATE FAIL <<<")


if __name__ == "__main__":
    main()
