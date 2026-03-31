"""Experiment 76: Few-Shot Cross-Environment Transfer (Stage 30).

Record 3 demos from expert on DoorKey (key/door).
FewShotAgent + AnalogicalReasoner solves CardGate (card/gate)
without any CardGate demos.

Gates:
    cross_env_success >= 0.7
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
from snks.env.card_gate_world import CardGateWorld


N_TRAIN_EXPERT = 5
N_DEMO = 3
N_TEST = 10
MAX_STEPS = 200


def _record_demo(expert, seed: int) -> Demonstration:
    """Record a demo from expert on DoorKey."""
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


def main():
    print("=" * 60)
    print("Experiment 76: Few-Shot Cross-Env Transfer (Stage 30)")
    print("=" * 60)

    # Phase 1: Train expert on DoorKey.
    print(f"\n--- Phase 1: Train expert ({N_TRAIN_EXPERT} eps) ---")
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

    # Phase 2: Record DoorKey demos.
    print(f"\n--- Phase 2: Record {N_DEMO} demos on DoorKey ---")
    demos: list[Demonstration] = []
    for i, seed in enumerate([42, 43, 44]):
        demo = _record_demo(expert, seed)
        demos.append(demo)
        print(f"  Demo {i}: {demo.n_steps} steps, success={demo.success}")

    successful_demos = [d for d in demos if d.success]
    print(f"  Successful demos: {len(successful_demos)}/{len(demos)}")

    if not successful_demos:
        print("ERROR: No successful demos!")
        return

    # Phase 3: Test on CardGate with FewShotAgent + analogical reasoning.
    print(f"\n--- Phase 3: Test on CardGate ({N_TEST} eps) ---")
    successes = 0

    for seed in range(200, 200 + N_TEST):
        env_test = CardGateWorld(size=5)
        obs, _ = env_test.reset(seed=seed)

        agent = FewShotAgent(env_test, analogy_threshold=0.5)
        new_skills = agent.learn_from_demos(successful_demos)

        instruction = obs.get("mission", "pick up the card and open the gate")
        result = agent.run_episode(instruction, max_steps=MAX_STEPS)

        if result.success:
            successes += 1
        print(f"  seed={seed}: success={result.success} steps={result.steps_taken} "
              f"skills_learned={new_skills}")
        env_test.close()

    rate = successes / N_TEST
    gate_pass = rate >= 0.7

    print(f"\n{'=' * 60}")
    print(f"GATE: cross_env_success {'PASS' if gate_pass else 'FAIL'} ({rate:.3f} >= 0.700)")
    print(f"{'=' * 60}")

    if gate_pass:
        print(">>> Experiment 76: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 76: GATE FAIL <<<")


if __name__ == "__main__":
    main()
