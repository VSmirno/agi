"""Experiment 65: Skill Extraction from DoorKey (Stage 27).

Gates:
    >= 2 primitive skills extracted
    >= 1 composite skill formed
    skills success_rate >= 0.8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.skill_agent import SkillAgent


N_EPISODES = 10
MAX_STEPS = 200


def main():
    print("=" * 60)
    print("Experiment 65: Skill Extraction from DoorKey (Stage 27)")
    print("=" * 60)

    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    agent = SkillAgent(env)

    for ep in range(N_EPISODES):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            agent._env = env_new
            agent._executor._env = env_new

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        lib = agent.library
        primitives = [s for s in lib.skills if not s.is_composite]
        composites = [s for s in lib.skills if s.is_composite]
        print(f"  Ep {ep}: success={result.success} steps={result.steps_taken} "
              f"skills_used={result.skills_used} "
              f"primitives={len(primitives)} composites={len(composites)}")

    # Final state.
    lib = agent.library
    primitives = [s for s in lib.skills if not s.is_composite]
    composites = [s for s in lib.skills if s.is_composite]

    print(f"\n--- Extracted Skills ---")
    for s in lib.skills:
        print(f"  {s.name}: pre={sorted(s.preconditions)} eff={sorted(s.effects)} "
              f"rate={s.success_rate:.2f} ({s.success_count}/{s.attempt_count}) "
              f"composite={s.is_composite}")

    # Gates.
    gate_primitives = len(primitives) >= 2
    gate_composites = len(composites) >= 1
    min_rate = min((s.success_rate for s in primitives), default=0.0)
    gate_rate = min_rate >= 0.8

    print(f"\n{'=' * 60}")
    print(f"GATE: primitives     {'PASS' if gate_primitives else 'FAIL'} ({len(primitives)} >= 2)")
    print(f"GATE: composites     {'PASS' if gate_composites else 'FAIL'} ({len(composites)} >= 1)")
    print(f"GATE: success_rate   {'PASS' if gate_rate else 'FAIL'} (min={min_rate:.2f} >= 0.80)")
    print(f"{'=' * 60}")

    if gate_primitives and gate_composites and gate_rate:
        print(">>> Experiment 65: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 65: GATE FAIL <<<")


if __name__ == "__main__":
    main()
