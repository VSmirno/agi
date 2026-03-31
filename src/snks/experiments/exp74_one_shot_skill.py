"""Experiment 74: One-Shot Skill Extraction (Stage 30).

Expert agent solves DoorKey once. Record demonstration.
FewShotLearner extracts skills from single demo.

Gates:
    skill_extraction_accuracy >= 0.9 (fraction of expected skills found)
    composite_skill_found == True (pickup_key+toggle_door)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid  # noqa: F401

from snks.language.curiosity_agent import CuriosityAgent
from snks.language.demonstration import DemoStep, Demonstration
from snks.language.few_shot_learner import FewShotLearner
from snks.language.grid_perception import GridPerception, SKS_KEY_PRESENT, SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_DOOR_OPEN, SKS_GOAL_PRESENT
from snks.language.grounding_map import GroundingMap


N_TRAIN_EXPERT = 5   # train expert first
N_DEMO = 1           # record 1 demo


def _record_expert_demo(expert_agent, env, seed: int) -> Demonstration:
    """Record a single successful demonstration from expert."""
    perception = GridPerception(GroundingMap())

    env_demo = gym.make("MiniGrid-DoorKey-5x5-v0")
    obs, _ = env_demo.reset(seed=seed)

    # Set up expert on this env
    expert_agent._env = env_demo
    expert_agent._executor._env = env_demo

    # Manual recording: capture sks before/after each step
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
    result = expert_agent.run_episode(obs["mission"], max_steps=200)
    env_demo.step = orig_step

    demo = Demonstration(
        steps=steps,
        goal_instruction=obs["mission"],
        success=result.success,
    )
    env_demo.close()
    return demo


def main():
    print("=" * 60)
    print("Experiment 74: One-Shot Skill Extraction (Stage 30)")
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
        result = expert.run_episode(obs["mission"], max_steps=200)
        print(f"  Train ep {ep}: success={result.success}")

    # Phase 2: Record 1 demo.
    print(f"\n--- Phase 2: Record {N_DEMO} demo ---")
    demo = _record_expert_demo(expert, env, seed=42)
    print(f"  Demo: {demo.n_steps} steps, success={demo.success}")
    print(f"  Actions: {[s.action for s in demo.steps]}")
    print(f"  Unique SKS: {demo.unique_sks()}")

    # Phase 3: Extract skills from single demo.
    print("\n--- Phase 3: One-shot skill extraction ---")
    learner = FewShotLearner(min_observations=1)
    model, library = learner.learn_from_demonstrations([demo])

    skill_names = {s.name for s in library.skills}
    print(f"  Skills extracted: {skill_names}")
    print(f"  Causal links: {model.n_links}")

    # Evaluate.
    expected_primitives = {"pickup_key", "toggle_door"}
    found_primitives = expected_primitives & skill_names
    composites = [s for s in library.skills if s.is_composite]
    composite_found = any("pickup_key" in s.name and "toggle_door" in s.name for s in composites)

    accuracy = len(found_primitives) / len(expected_primitives)

    print(f"\n--- Results ---")
    print(f"  Primitives found: {found_primitives} ({accuracy:.2f})")
    print(f"  Composites: {[s.name for s in composites]}")
    print(f"  Composite found: {composite_found}")

    gate_accuracy = accuracy >= 0.9
    gate_composite = composite_found

    print(f"\n{'=' * 60}")
    print(f"GATE: skill_accuracy   {'PASS' if gate_accuracy else 'FAIL'} ({accuracy:.2f} >= 0.90)")
    print(f"GATE: composite_found  {'PASS' if gate_composite else 'FAIL'} ({composite_found})")
    print(f"{'=' * 60}")

    if gate_accuracy and gate_composite:
        print(">>> Experiment 74: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 74: GATE FAIL <<<")


if __name__ == "__main__":
    main()
