"""Experiment 73: Curiosity + Goal Completion (Stage 29).

CuriosityAgent solves DoorKey using pre-trained causal model.
Curiosity helps exploration when external reward signal is delayed.

Gates:
    DoorKey success rate >= 0.9 (with curiosity-enhanced exploration)
    CuriosityAgent uses >= 1 skill per episode on average
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.grounding_map import GroundingMap


N_TRAIN = 5
N_TEST = 10
MAX_STEPS_TRAIN = 200
MAX_STEPS_TEST = 200


def main():
    print("=" * 60)
    print("Experiment 73: Curiosity + Goal Completion (Stage 29)")
    print("=" * 60)

    # Phase 1: Train CuriosityAgent on DoorKey.
    print(f"\n--- Phase 1: Train ({N_TRAIN} episodes) ---")
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    agent = CuriosityAgent(env, curiosity_across_episodes=True)

    for ep in range(N_TRAIN):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            agent._env = env_new
            agent._executor._env = env_new

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TRAIN)
        print(f"  Ep {ep}: success={result.success} steps={result.steps_taken} "
              f"distinct={agent.curiosity.n_distinct()}")

    lib = agent.library
    cm = agent.causal_model
    print(f"  Skills: {len(lib.skills)}, Causal links: {cm.n_links}")
    print(f"  Total distinct states across train: {agent.curiosity.n_distinct()}")

    # Phase 2: Test CuriosityAgent.
    print(f"\n--- Phase 2: Test ({N_TEST} episodes) ---")
    successes = 0
    total_skills_used = 0

    for seed in range(50, 50 + N_TEST):
        env_test = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env_test.reset(seed=seed)

        test_agent = CuriosityAgent(
            env_test,
            skill_library=lib,
            grounding_map=GroundingMap(),
            causal_model=cm,
        )

        result = test_agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TEST)

        if result.success:
            successes += 1
        total_skills_used += len(result.skills_used)

        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills={result.skills_used} distinct={test_agent.curiosity.n_distinct()}")
        env_test.close()

    rate = successes / N_TEST
    avg_skills = total_skills_used / N_TEST

    print(f"\n--- Results ---")
    print(f"  Success rate: {rate:.3f} ({successes}/{N_TEST})")
    print(f"  Avg skills used: {avg_skills:.1f}")

    gate_success = rate >= 0.9
    gate_skills = avg_skills >= 1.0

    print(f"\n{'=' * 60}")
    print(f"GATE: success_rate   {'PASS' if gate_success else 'FAIL'} ({rate:.3f} >= 0.900)")
    print(f"GATE: skills_per_ep  {'PASS' if gate_skills else 'FAIL'} ({avg_skills:.1f} >= 1.0)")
    print(f"{'=' * 60}")

    if gate_success and gate_skills:
        print(">>> Experiment 73: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 73: GATE FAIL <<<")


if __name__ == "__main__":
    main()
