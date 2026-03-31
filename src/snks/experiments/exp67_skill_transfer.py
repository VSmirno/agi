"""Experiment 67: Skill Transfer to MultiRoom (Stage 27).

Gates:
    success rate >= 0.9
    >= 1 skill reused per episode
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
MAX_STEPS_TRAIN = 200
MAX_STEPS_TEST = 300


def main():
    print("=" * 60)
    print("Experiment 67: Skill Transfer to MultiRoom (Stage 27)")
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
    for s in lib.skills:
        print(f"    {s.name}: composite={s.is_composite} rate={s.success_rate:.2f}")

    # Phase 2: Transfer to MultiRoom.
    print(f"\n--- Phase 2: Transfer to MultiRoomDoorKey ({N_TEST} episodes) ---")
    successes = 0
    total_skills_used = 0

    for seed in range(50, 50 + N_TEST):
        env_mr = MultiRoomDoorKey(size=10, seed=seed)
        obs, _ = env_mr.reset()

        # New GroundingMap but reuse skill library + causal model.
        transfer_agent = SkillAgent(
            env_mr,
            skill_library=lib,
            grounding_map=GroundingMap(),
            causal_model=cm,
        )

        result = transfer_agent.run_episode(
            "use the keys to open the doors and get to the goal",
            max_steps=MAX_STEPS_TEST,
        )

        if result.success:
            successes += 1
        total_skills_used += len(result.skills_used)
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills={result.skills_used}")
        env_mr.close()

    rate = successes / N_TEST
    avg_skills = total_skills_used / N_TEST

    print(f"\n--- Results ---")
    print(f"  Success rate: {rate:.3f} ({successes}/{N_TEST})")
    print(f"  Avg skills per episode: {avg_skills:.1f}")

    gate_success = rate >= 0.9
    gate_skills = avg_skills >= 1.0

    print(f"\n{'=' * 60}")
    print(f"GATE: success       {'PASS' if gate_success else 'FAIL'} ({rate:.3f} >= 0.900)")
    print(f"GATE: skill_reuse   {'PASS' if gate_skills else 'FAIL'} ({avg_skills:.1f} >= 1.0)")
    print(f"{'=' * 60}")

    if gate_success and gate_skills:
        print(">>> Experiment 67: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 67: GATE FAIL <<<")


if __name__ == "__main__":
    main()
