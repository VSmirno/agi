"""Experiment 75: Few-Shot Goal Completion (Stage 30).

Record 1 and 3 demos from expert. FewShotAgent learns from demos,
then solves unseen DoorKey layouts.

Gates:
    1_demo_success >= 0.5
    3_demo_success >= 0.8
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.demonstration import DemoStep, Demonstration
from snks.language.few_shot_agent import FewShotAgent
from snks.language.grid_perception import GridPerception
from snks.language.grounding_map import GroundingMap


N_TRAIN_EXPERT = 5
N_TEST = 10
MAX_STEPS = 200


def _record_demo(expert, seed: int) -> Demonstration:
    """Record a demo from expert on a fresh DoorKey env."""
    perception = GridPerception(GroundingMap())
    env_demo = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env_demo.reset(seed=seed)

    expert._env = env_demo
    expert._executor._env = env_demo

    uw = env_demo.unwrapped
    steps: list[DemoStep] = []
    orig_step = env_demo.step

    def recording_step(action):
        carrying = getattr(uw, "carrying", None)
        sks_before = frozenset(perception.perceive(uw.grid, uw.agent_pos, uw.agent_dir, carrying))
        result = orig_step(action)
        carrying = getattr(uw, "carrying", None)
        sks_after = frozenset(perception.perceive(uw.grid, uw.agent_pos, uw.agent_dir, carrying))
        steps.append(DemoStep(sks_before=sks_before, action=action, sks_after=sks_after))
        return result

    env_demo.step = recording_step
    result = expert.run_episode(obs["mission"], max_steps=MAX_STEPS)
    env_demo.step = orig_step

    demo = Demonstration(steps=steps, goal_instruction=obs["mission"], success=result.success)
    env_demo.close()
    return demo


def _test_few_shot(demos: list[Demonstration], n_demos_label: str) -> float:
    """Test FewShotAgent with given demos on N_TEST layouts."""
    print(f"\n--- Test: {n_demos_label} ({N_TEST} episodes) ---")
    successes = 0

    for seed in range(100, 100 + N_TEST):
        env_test = gym.make("MiniGrid-DoorKey-5x5-v0")
        obs, _ = env_test.reset(seed=seed)

        agent = FewShotAgent(env_test)
        new_skills = agent.learn_from_demos(demos)

        result = agent.run_episode(obs["mission"], max_steps=MAX_STEPS)
        if result.success:
            successes += 1
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills_learned={new_skills}")
        env_test.close()

    rate = successes / N_TEST
    print(f"  Success rate: {rate:.3f} ({successes}/{N_TEST})")
    return rate


def main():
    print("=" * 60)
    print("Experiment 75: Few-Shot Goal Completion (Stage 30)")
    print("=" * 60)

    # Phase 1: Train expert.
    print(f"\n--- Phase 1: Train expert ({N_TRAIN_EXPERT} episodes) ---")
    env = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env.reset(seed=0)
    expert = CuriosityAgent(env, curiosity_across_episodes=True)

    for ep in range(N_TRAIN_EXPERT):
        if ep > 0:
            env_new = gym.make("MiniGrid-DoorKey-5x5-v0")
            obs, _ = env_new.reset(seed=ep)
            expert._env = env_new
            expert._executor._env = env_new
        result = expert.run_episode(obs["mission"], max_steps=MAX_STEPS)
        print(f"  Ep {ep}: success={result.success}")

    # Phase 2: Record demos.
    print("\n--- Phase 2: Record demonstrations ---")
    demos: list[Demonstration] = []
    for i, seed in enumerate([42, 43, 44]):
        demo = _record_demo(expert, seed)
        demos.append(demo)
        print(f"  Demo {i}: {demo.n_steps} steps, success={demo.success}")

    # Phase 3: Test with 1 demo.
    successful_demos = [d for d in demos if d.success]
    if len(successful_demos) == 0:
        print("ERROR: No successful demos recorded!")
        return

    rate_1 = _test_few_shot(successful_demos[:1], "1 demo")

    # Phase 4: Test with 3 demos (or all successful).
    rate_3 = _test_few_shot(successful_demos[:3], f"{min(3, len(successful_demos))} demos")

    # Gates.
    gate_1 = rate_1 >= 0.5
    gate_3 = rate_3 >= 0.8

    print(f"\n{'=' * 60}")
    print(f"GATE: 1_demo_success  {'PASS' if gate_1 else 'FAIL'} ({rate_1:.3f} >= 0.500)")
    print(f"GATE: 3_demo_success  {'PASS' if gate_3 else 'FAIL'} ({rate_3:.3f} >= 0.800)")
    print(f"{'=' * 60}")

    if gate_1 and gate_3:
        print(">>> Experiment 75: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 75: GATE FAIL <<<")


if __name__ == "__main__":
    main()
