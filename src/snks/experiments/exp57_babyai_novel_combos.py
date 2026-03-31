"""Experiment 57: Novel combination generalization (Stage 24c).

Tests compositional generalization: agent has seen "red key" and "blue ball"
separately, can it handle "blue key" (novel color+object combo)?

Uses BabyAI-GoToLocalS6N4-v0 (6x6 grid, 4 objects, single room).

Gate:
    novel_success_rate >= 0.3
    known_success_rate >= 0.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gymnasium as gym
import minigrid

from snks.language.babyai_executor import BabyAIExecutor
from snks.language.grid_perception import GridPerception
from snks.language.grounding_map import GroundingMap


# Known training combinations — agent "learns" grounding for these.
KNOWN_COMBOS = {
    ("red", "key"), ("blue", "ball"), ("green", "box"),
    ("yellow", "key"), ("grey", "ball"), ("purple", "box"),
}

# Novel test combinations — never seen color+object together.
NOVEL_COMBOS = {
    ("blue", "key"), ("red", "ball"), ("green", "key"),
    ("yellow", "ball"), ("grey", "box"), ("purple", "key"),
}

MAX_STEPS = 100


def _is_novel(color: str, obj_type: str) -> bool:
    """Check if (color, obj_type) is a novel combination."""
    return (color, obj_type) in NOVEL_COMBOS


def _is_known(color: str, obj_type: str) -> bool:
    """Check if (color, obj_type) is a known combination."""
    return (color, obj_type) in KNOWN_COMBOS


def _parse_mission_target(mission: str) -> tuple[str, str] | None:
    """Extract (color, obj_type) from mission string like 'go to the red key'."""
    # Remove articles.
    words = mission.lower().replace("go to", "").replace("pick up", "").strip()
    words = words.replace("the ", "").replace("a ", "").strip().split()
    if len(words) >= 2:
        return (words[0], words[1])
    return None


def run_experiment(n_seeds: int = 200) -> dict:
    """Run novel combo experiment over many seeds.

    Scans BabyAI-GoToLocalS6N4-v0 episodes to find ones with novel
    and known targets, then tests execution on both.
    """
    known_success = 0
    known_total = 0
    novel_success = 0
    novel_total = 0
    novel_grounding_ok = 0

    for seed in range(n_seeds):
        env = gym.make("BabyAI-GoToLocalS6N4-v0")
        obs, _ = env.reset(seed=seed)
        mission = obs["mission"]

        target = _parse_mission_target(mission)
        if target is None:
            env.close()
            continue

        color, obj_type = target
        is_novel = _is_novel(color, obj_type)
        is_known = _is_known(color, obj_type)

        if not is_novel and not is_known:
            env.close()
            continue

        # Create fresh perception + executor for each episode.
        # For known combos, register them. For novel combos,
        # register the color and obj_type separately (compositional).
        gmap = GroundingMap()
        perc = GridPerception(gmap)

        # "Training phase" — register known combos.
        for kc, ko in KNOWN_COMBOS:
            perc.register_object(ko, kc)

        executor = BabyAIExecutor(env, perc)
        result = executor.execute(mission, max_steps=MAX_STEPS)

        if is_known:
            known_total += 1
            if result.success:
                known_success += 1
        elif is_novel:
            novel_total += 1
            # Check if grounding resolved correctly.
            if not result.error or "not found" not in result.error:
                novel_grounding_ok += 1
            if result.success:
                novel_success += 1

        env.close()

    return {
        "known_total": known_total,
        "known_success": known_success,
        "known_success_rate": known_success / max(known_total, 1),
        "novel_total": novel_total,
        "novel_success": novel_success,
        "novel_success_rate": novel_success / max(novel_total, 1),
        "novel_grounding_accuracy": novel_grounding_ok / max(novel_total, 1),
    }


def main():
    print("=" * 60)
    print("Experiment 57: Novel Combination Generalization (Stage 24c)")
    print("=" * 60)

    results = run_experiment(n_seeds=200)

    print(f"\n--- Known Combos ---")
    print(f"  Total episodes: {results['known_total']}")
    print(f"  Success rate: {results['known_success_rate']:.3f} (gate >= 0.500)")

    print(f"\n--- Novel Combos ---")
    print(f"  Total episodes: {results['novel_total']}")
    print(f"  Success rate: {results['novel_success_rate']:.3f} (gate >= 0.300)")
    print(f"  Grounding accuracy: {results['novel_grounding_accuracy']:.3f}")

    # Gate check
    known_pass = results["known_success_rate"] >= 0.5
    novel_pass = results["novel_success_rate"] >= 0.3

    print(f"\n{'=' * 60}")
    print(f"GATE: known  {'PASS' if known_pass else 'FAIL'} ({results['known_success_rate']:.3f} >= 0.500)")
    print(f"GATE: novel  {'PASS' if novel_pass else 'FAIL'} ({results['novel_success_rate']:.3f} >= 0.300)")
    print(f"{'=' * 60}")

    if known_pass and novel_pass:
        print(">>> Experiment 57: ALL GATES PASS <<<")
    else:
        print(">>> Experiment 57: GATE FAIL <<<")


if __name__ == "__main__":
    main()
