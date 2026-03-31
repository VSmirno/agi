"""Experiment 82: Multi-Task Strategy Selection (Stage 32).

Tests MetaLearner on 3+ distinct task types to verify correct
strategy selection across diverse scenarios.

Gates:
    multi_task_accuracy >= 0.8 (correct strategy for each task type)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.language.meta_learner import (
    EpisodeResult,
    MetaLearner,
    TaskProfile,
)


def _task_empty_room() -> list[tuple[TaskProfile, str]]:
    """Empty room: needs pure exploration. Expected: curiosity."""
    return [
        (TaskProfile(state_coverage=0.0, known_skills=0, causal_links=0), "curiosity"),
        (TaskProfile(state_coverage=0.1, known_skills=0, causal_links=1), "curiosity"),
        (TaskProfile(state_coverage=0.2, known_skills=0, causal_links=2), "curiosity"),
    ]


def _task_doorkey_with_demo() -> list[tuple[TaskProfile, str]]:
    """DoorKey with demonstration available. Expected: few_shot → skill."""
    return [
        (TaskProfile(has_demos=True, n_demos=1, known_skills=0, causal_links=0), "few_shot"),
        # After few-shot learning, skills become available
        (TaskProfile(has_demos=True, n_demos=1, known_skills=3, causal_links=8,
                     state_coverage=0.4), "skill"),
        (TaskProfile(known_skills=4, causal_links=12, state_coverage=0.7,
                     mean_prediction_error=0.2), "skill"),
    ]


def _task_new_environment() -> list[tuple[TaskProfile, str]]:
    """Completely new environment, no prior knowledge. Expected: curiosity → explore → skill."""
    return [
        (TaskProfile(state_coverage=0.0), "curiosity"),
        (TaskProfile(state_coverage=0.4, known_skills=1, causal_links=3,
                     mean_prediction_error=0.6), "curiosity"),
        (TaskProfile(state_coverage=0.6, known_skills=2, causal_links=6,
                     mean_prediction_error=0.3), "skill"),
    ]


def _task_transfer_learning() -> list[tuple[TaskProfile, str]]:
    """Transfer to analogous environment with existing skills."""
    return [
        (TaskProfile(known_skills=5, causal_links=15, state_coverage=0.3,
                     mean_prediction_error=0.4), "skill"),
        (TaskProfile(known_skills=6, causal_links=20, state_coverage=0.5,
                     mean_prediction_error=0.2), "skill"),
    ]


def _task_high_uncertainty() -> list[tuple[TaskProfile, str]]:
    """Environment with high prediction error."""
    return [
        (TaskProfile(state_coverage=0.5, known_skills=1, causal_links=4,
                     mean_prediction_error=0.9), "curiosity"),
        (TaskProfile(state_coverage=0.3, known_skills=0, causal_links=1,
                     mean_prediction_error=0.8), "curiosity"),
    ]


TASK_TYPES = [
    ("Empty Room (exploration)", _task_empty_room),
    ("DoorKey + Demo (few-shot)", _task_doorkey_with_demo),
    ("New Environment (progressive)", _task_new_environment),
    ("Transfer Learning (analogical)", _task_transfer_learning),
    ("High Uncertainty (noisy)", _task_high_uncertainty),
]


def main() -> None:
    print("=" * 60)
    print("Experiment 82: Multi-Task Strategy Selection")
    print("=" * 60)

    total_correct = 0
    total_scenarios = 0
    task_results: list[tuple[str, int, int]] = []

    for task_name, task_fn in TASK_TYPES:
        print(f"\n--- {task_name} ---")
        learner = MetaLearner()
        scenarios = task_fn()
        correct = 0

        for i, (profile, expected) in enumerate(scenarios):
            config = learner.select_strategy(profile)
            match = config.strategy == expected
            status = "PASS" if match else "FAIL"
            print(f"  Phase {i}: expected={expected}, got={config.strategy} [{status}]")
            if match:
                correct += 1

        task_results.append((task_name, correct, len(scenarios)))
        total_correct += correct
        total_scenarios += len(scenarios)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Task Results:")
    for name, c, t in task_results:
        print(f"  {name}: {c}/{t} ({'PASS' if c == t else 'PARTIAL'})")

    accuracy = total_correct / total_scenarios
    print(f"\nmulti_task_accuracy = {accuracy:.3f} (gate >= 0.8)")

    # Check all task types have at least one correct
    all_tasks_represented = all(c > 0 for _, c, _ in task_results)
    print(f"All {len(TASK_TYPES)} task types represented: {all_tasks_represented}")

    g1 = accuracy >= 0.8
    print(f"\nGate multi_task_accuracy >= 0.8: {'PASS' if g1 else 'FAIL'}")

    if g1:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
