"""Experiment 58: Autonomous Goal Decomposition (Stage 25).

GoalAgent correctly decomposes DoorKey mission into sub-goals
after warmup on 5 episodes.

Gate: decomposition_accuracy >= 0.9
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap


N_WARMUP = 5
N_TEST = 20
MAX_STEPS = 200
EXPECTED_SUBGOALS = {"pickup", "toggle"}


def main():
    print("=" * 60)
    print("Experiment 58: Autonomous Goal Decomposition (Stage 25)")
    print("=" * 60)

    gmap = GroundingMap()

    # Warmup phase: learn causal links.
    print(f"\n--- Warmup ({N_WARMUP} episodes) ---")
    warmup_agent = None
    for seed in range(N_WARMUP):
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=seed)

        if warmup_agent is None:
            warmup_agent = GoalAgent(env, grounding_map=gmap)
        else:
            warmup_agent._env = env
            warmup_agent._executor._env = env

        result = warmup_agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        n_links = warmup_agent.causal_model.n_links
        print(f"  seed={seed}: success={result.success} explored={result.explored} links={n_links}")
        env.close()

    print(f"  Causal links after warmup: {warmup_agent.causal_model.n_links}")

    # Test phase: check decomposition accuracy.
    print(f"\n--- Test ({N_TEST} episodes) ---")
    correct = 0
    for seed in range(100, 100 + N_TEST):
        env = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env.reset(seed=seed)

        warmup_agent._env = env
        warmup_agent._executor._env = env

        result = warmup_agent.run_episode(obs["mission"], max_steps=MAX_STEPS)

        # Check if subgoals contain pickup and toggle.
        subgoal_text = " ".join(result.subgoals_identified).lower()
        has_expected = all(s in subgoal_text for s in EXPECTED_SUBGOALS)

        if has_expected:
            correct += 1
            status = "OK"
        else:
            status = f"MISS (got: {result.subgoals_identified})"

        print(f"  seed={seed}: {status} success={result.success}")
        env.close()

    accuracy = correct / N_TEST
    print(f"\n--- Results ---")
    print(f"  Decomposition accuracy: {accuracy:.3f} ({correct}/{N_TEST})")

    gate_pass = accuracy >= 0.9
    print(f"\n{'=' * 60}")
    print(f"GATE: decomposition {'PASS' if gate_pass else 'FAIL'} ({accuracy:.3f} >= 0.900)")
    print(f"{'=' * 60}")

    if gate_pass:
        print(">>> Experiment 58: GATE PASS <<<")
    else:
        print(">>> Experiment 58: GATE FAIL <<<")


if __name__ == "__main__":
    main()
