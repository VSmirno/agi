"""Experiment 69: Analogy-Driven Solving (Stage 28).

Gates:
    success rate on CardGateWorld >= 0.8
    avg analogies used per episode >= 1.0
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.env.card_gate_world import CardGateWorld
from snks.language.grounding_map import GroundingMap
from snks.language.skill_agent import SkillAgent


N_TRAIN = 5
N_TEST = 10
MAX_STEPS_TRAIN = 200
MAX_STEPS_TEST = 200


def main():
    print("=" * 60)
    print("Experiment 69: Analogy-Driven Solving (Stage 28)")
    print("=" * 60)

    # Phase 1: Train on DoorKey-5x5, extract skills.
    print(f"\n--- Phase 1: Train on DoorKey-5x5 ({N_TRAIN} episodes) ---")
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    agent = SkillAgent(env)

    for ep in range(N_TRAIN):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            agent._env = env_new
            agent._executor._env = env_new

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS_TRAIN)
        print(f"  Ep {ep}: success={result.success} steps={result.steps_taken}")

    lib = agent.library
    cm = agent.causal_model
    print(f"  Skills: {len(lib.skills)}, Causal links: {cm.n_links}")

    # Phase 2: Solve CardGateWorld using analogy.
    print(f"\n--- Phase 2: CardGateWorld ({N_TEST} episodes, analogy transfer) ---")
    successes = 0
    total_analogies_used = 0

    for seed in range(50, 50 + N_TEST):
        cg_env = CardGateWorld(size=5)
        cg_env.reset(seed=seed)

        # New grounding map but reuse skill library + causal model.
        transfer_agent = SkillAgent(
            cg_env,
            skill_library=lib,
            grounding_map=GroundingMap(),
            causal_model=cm,
        )

        result = transfer_agent.run_episode(
            "use the card to open the gate and get to the goal",
            max_steps=MAX_STEPS_TEST,
        )

        if result.success:
            successes += 1

        # Count analogies used (skills with "adapted_" prefix).
        analogies_used = sum(1 for s in result.skills_used if s.startswith("adapted_"))
        total_analogies_used += analogies_used

        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills={result.skills_used} analogies={analogies_used}")
        cg_env.close()

    rate = successes / N_TEST
    avg_analogies = total_analogies_used / N_TEST

    print(f"\n--- Results ---")
    print(f"  Success rate: {rate:.3f} ({successes}/{N_TEST})")
    print(f"  Avg analogies used per episode: {avg_analogies:.1f}")

    gate_success = rate >= 0.8
    gate_analogies = avg_analogies >= 1.0

    print(f"\n{'=' * 60}")
    print(f"GATE: success        {'PASS' if gate_success else 'FAIL'} ({rate:.3f} >= 0.800)")
    print(f"GATE: analogies_used {'PASS' if gate_analogies else 'FAIL'} ({avg_analogies:.1f} >= 1.0)")
    print(f"{'=' * 60}")

    if gate_success and gate_analogies:
        print(">>> Experiment 69: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 69: GATE FAIL <<<")


if __name__ == "__main__":
    main()
