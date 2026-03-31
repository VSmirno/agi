"""Experiment 70: Regression Test (Stage 28).

Gates:
    DoorKey-5x5 success rate >= 0.9 (no regression from Stage 27 changes)
    MultiRoomDoorKey success rate >= 0.8 (no regression)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.env.multi_room import MultiRoomDoorKey
from snks.language.grounding_map import GroundingMap
from snks.language.skill_agent import SkillAgent


N_TRAIN = 5
N_TEST = 10
MAX_STEPS = 200
MAX_STEPS_MR = 300


def main():
    print("=" * 60)
    print("Experiment 70: Regression Test (Stage 28)")
    print("=" * 60)

    # Phase 1: Train.
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
        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)

    lib = agent.library
    cm = agent.causal_model
    print(f"  Skills: {len(lib.skills)}, Causal links: {cm.n_links}")

    # Phase 2: DoorKey regression.
    print(f"\n--- Phase 2: DoorKey-5x5 regression ({N_TEST} episodes) ---")
    dk_successes = 0
    for seed in range(50, 50 + N_TEST):
        env_test = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env_test.reset(seed=seed)
        agent._env = env_test
        agent._executor._env = env_test
        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        if result.success:
            dk_successes += 1
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} skills={result.skills_used}")
        env_test.close()

    dk_rate = dk_successes / N_TEST

    # Phase 3: MultiRoom regression.
    print(f"\n--- Phase 3: MultiRoomDoorKey regression ({N_TEST} episodes) ---")
    mr_successes = 0
    for seed in range(50, 50 + N_TEST):
        env_mr = MultiRoomDoorKey(size=10, seed=seed)
        obs, _ = env_mr.reset()
        transfer_agent = SkillAgent(
            env_mr,
            skill_library=lib,
            grounding_map=GroundingMap(),
            causal_model=cm,
        )
        result = transfer_agent.run_episode(
            "use the keys to open the doors and get to the goal",
            max_steps=MAX_STEPS_MR,
        )
        if result.success:
            mr_successes += 1
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} skills={result.skills_used}")
        env_mr.close()

    mr_rate = mr_successes / N_TEST

    print(f"\n--- Results ---")
    print(f"  DoorKey success rate:    {dk_rate:.3f} ({dk_successes}/{N_TEST})")
    print(f"  MultiRoom success rate:  {mr_rate:.3f} ({mr_successes}/{N_TEST})")

    gate_dk = dk_rate >= 0.9
    gate_mr = mr_rate >= 0.8

    print(f"\n{'=' * 60}")
    print(f"GATE: doorkey_rate   {'PASS' if gate_dk else 'FAIL'} ({dk_rate:.3f} >= 0.900)")
    print(f"GATE: multiroom_rate {'PASS' if gate_mr else 'FAIL'} ({mr_rate:.3f} >= 0.800)")
    print(f"{'=' * 60}")

    if gate_dk and gate_mr:
        print(">>> Experiment 70: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 70: GATE FAIL <<<")


if __name__ == "__main__":
    main()
