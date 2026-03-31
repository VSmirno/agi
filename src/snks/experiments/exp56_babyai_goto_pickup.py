"""Experiment 56: BabyAI GoTo + Pickup success rate (Stage 24c).

Full e2e: text instruction → parse → navigate → execute in MiniGrid.

Gate:
    goto_success_rate   >= 0.6
    pickup_success_rate >= 0.5
    overall_success_rate >= 0.55
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.babyai_executor import BabyAIExecutor
from snks.language.grid_perception import GridPerception
from snks.language.grounding_map import GroundingMap


N_EPISODES = 50
MAX_STEPS = 100


def run_goto(n: int = N_EPISODES) -> dict:
    """Run GoTo experiments on BabyAI-GoToObj-v0."""
    successes = 0
    total_steps = 0
    parse_ok = 0

    for seed in range(n):
        env = gym.make("BabyAI-GoToObj-v0")
        obs, _ = env.reset(seed=seed)
        mission = obs["mission"]

        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute(mission, max_steps=MAX_STEPS)

        if result.parsed_action:
            parse_ok += 1
        if result.success:
            successes += 1
            total_steps += result.steps_taken

        env.close()

    return {
        "type": "goto",
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "parse_accuracy": parse_ok / n,
        "avg_steps": total_steps / max(successes, 1),
    }


def run_pickup(n: int = N_EPISODES) -> dict:
    """Run Pickup experiments on BabyAI-PickupLoc-v0."""
    successes = 0
    total_steps = 0
    parse_ok = 0

    for seed in range(n):
        env = gym.make("BabyAI-PickupLoc-v0")
        obs, _ = env.reset(seed=seed)
        mission = obs["mission"]

        gmap = GroundingMap()
        perc = GridPerception(gmap)
        executor = BabyAIExecutor(env, perc)

        result = executor.execute(mission, max_steps=MAX_STEPS)

        if result.parsed_action:
            parse_ok += 1
        if result.success:
            successes += 1
            total_steps += result.steps_taken

        env.close()

    return {
        "type": "pickup",
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "parse_accuracy": parse_ok / n,
        "avg_steps": total_steps / max(successes, 1),
    }


def main():
    print("=" * 60)
    print("Experiment 56: BabyAI GoTo + Pickup (Stage 24c)")
    print("=" * 60)

    goto_results = run_goto()
    print(f"\n--- GoTo Results ---")
    print(f"  Success rate: {goto_results['success_rate']:.3f} (gate >= 0.600)")
    print(f"  Parse accuracy: {goto_results['parse_accuracy']:.3f}")
    print(f"  Avg steps (success): {goto_results['avg_steps']:.1f}")

    pickup_results = run_pickup()
    print(f"\n--- Pickup Results ---")
    print(f"  Success rate: {pickup_results['success_rate']:.3f} (gate >= 0.500)")
    print(f"  Parse accuracy: {pickup_results['parse_accuracy']:.3f}")
    print(f"  Avg steps (success): {pickup_results['avg_steps']:.1f}")

    total_success = goto_results["successes"] + pickup_results["successes"]
    total_n = goto_results["n"] + pickup_results["n"]
    overall = total_success / total_n

    print(f"\n--- Overall ---")
    print(f"  Overall success rate: {overall:.3f} (gate >= 0.550)")

    # Gate check
    goto_pass = goto_results["success_rate"] >= 0.6
    pickup_pass = pickup_results["success_rate"] >= 0.5
    overall_pass = overall >= 0.55

    print(f"\n{'=' * 60}")
    print(f"GATE: goto   {'PASS' if goto_pass else 'FAIL'} ({goto_results['success_rate']:.3f} >= 0.600)")
    print(f"GATE: pickup {'PASS' if pickup_pass else 'FAIL'} ({pickup_results['success_rate']:.3f} >= 0.500)")
    print(f"GATE: overall {'PASS' if overall_pass else 'FAIL'} ({overall:.3f} >= 0.550)")
    print(f"{'=' * 60}")

    if goto_pass and pickup_pass and overall_pass:
        print(">>> Experiment 56: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 56: GATE FAIL <<<")


if __name__ == "__main__":
    main()
