"""Experiment 66: Skill Reuse Speedup (Stage 27).

Gates:
    skill agent mean_steps <= 0.67 * control mean_steps (>= 1.5x speedup)
    skill agent exploration_episodes <= 1
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.goal_agent import GoalAgent
from snks.language.grounding_map import GroundingMap
from snks.language.skill_agent import SkillAgent


N_WARMUP = 5
N_TEST = 10
MAX_STEPS = 200


def main():
    print("=" * 60)
    print("Experiment 66: Skill Reuse Speedup (Stage 27)")
    print("=" * 60)

    # Phase A: Warmup — SkillAgent learns + extracts skills.
    print(f"\n--- Phase A: Warmup ({N_WARMUP} episodes) ---")
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    agent = SkillAgent(env)

    for ep in range(N_WARMUP):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            agent._env = env_new
            agent._executor._env = env_new

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        print(f"  Warmup ep={ep}: success={result.success} steps={result.steps_taken}")

    lib = agent.library
    print(f"  Skills extracted: {len(lib.skills)}")
    for s in lib.skills:
        print(f"    {s.name}: composite={s.is_composite}")

    # Phase B: Test — SkillAgent with pre-extracted skills.
    print(f"\n--- Phase B: SkillAgent test ({N_TEST} episodes) ---")
    skill_steps = []
    skill_exploration = 0

    for seed in range(50, 50 + N_TEST):
        env_test = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env_test.reset(seed=seed)
        agent._env = env_test
        agent._executor._env = env_test

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        skill_steps.append(result.steps_taken)
        if result.explored:
            skill_exploration += 1
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills={result.skills_used} explored={result.explored}")
        env_test.close()

    # Phase C: Control — fresh GoalAgent per episode (no shared knowledge).
    print(f"\n--- Phase C: GoalAgent control — fresh per episode ({N_TEST} episodes) ---")
    control_steps = []

    for seed in range(50, 50 + N_TEST):
        env_ctrl = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env_ctrl.reset(seed=seed)

        # Fresh agent each episode — must explore from scratch.
        control_agent = GoalAgent(env_ctrl)
        result = control_agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        control_steps.append(result.steps_taken)
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} explored={result.explored}")
        env_ctrl.close()

    # Results.
    skill_mean = sum(skill_steps) / len(skill_steps)
    control_mean = sum(control_steps) / len(control_steps)
    speedup = control_mean / max(skill_mean, 0.1)

    print(f"\n--- Results ---")
    print(f"  SkillAgent: mean_steps={skill_mean:.1f} exploration_episodes={skill_exploration}")
    print(f"  GoalAgent:  mean_steps={control_mean:.1f}")
    print(f"  Speedup: {speedup:.2f}x")

    gate_speedup = skill_mean <= 0.67 * control_mean
    gate_exploration = skill_exploration <= 1

    print(f"\n{'=' * 60}")
    print(f"GATE: speedup       {'PASS' if gate_speedup else 'FAIL'} "
          f"({skill_mean:.1f} <= {0.67 * control_mean:.1f} = 0.67 * {control_mean:.1f})")
    print(f"GATE: exploration   {'PASS' if gate_exploration else 'FAIL'} ({skill_exploration} <= 1)")
    print(f"{'=' * 60}")

    if gate_speedup and gate_exploration:
        print(">>> Experiment 66: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 66: GATE FAIL <<<")


if __name__ == "__main__":
    main()
