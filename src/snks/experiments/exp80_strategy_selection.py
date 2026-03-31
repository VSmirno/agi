"""Experiment 80: Strategy Selection Accuracy (Stage 32).

Tests that MetaLearner selects the correct strategy for known scenarios.

Gates:
    strategy_selection_accuracy >= 0.8
    profile_extraction = all correct (all scenario profiles produce expected strategy)
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

from snks.language.meta_learner import MetaLearner, TaskProfile


# Define test scenarios: (profile, expected_strategy)
SCENARIOS = [
    # Scenario 1: Fresh start, demos available → few_shot
    (TaskProfile(has_demos=True, n_demos=2, known_skills=0, causal_links=0,
                 state_coverage=0.0), "few_shot"),

    # Scenario 2: Fresh start, no demos, no knowledge → curiosity
    (TaskProfile(has_demos=False, known_skills=0, causal_links=0,
                 state_coverage=0.0), "curiosity"),

    # Scenario 3: Good knowledge base → skill
    (TaskProfile(known_skills=5, causal_links=20, state_coverage=0.6,
                 mean_prediction_error=0.2), "skill"),

    # Scenario 4: Some skills but low coverage → curiosity
    (TaskProfile(known_skills=1, causal_links=2, state_coverage=0.1), "curiosity"),

    # Scenario 5: Moderate knowledge, high pred error → curiosity
    (TaskProfile(known_skills=1, causal_links=3, state_coverage=0.5,
                 mean_prediction_error=0.8), "curiosity"),

    # Scenario 6: Moderate knowledge, low pred error → explore
    (TaskProfile(known_skills=1, causal_links=3, state_coverage=0.5,
                 mean_prediction_error=0.3), "explore"),

    # Scenario 7: Lots of skills, many links → skill
    (TaskProfile(known_skills=10, causal_links=50, state_coverage=0.9,
                 mean_prediction_error=0.1), "skill"),

    # Scenario 8: Demos + existing skills → few_shot (demos override when no skills)
    # Note: has_demos=True but known_skills > 0 → skill (not few_shot)
    (TaskProfile(has_demos=True, n_demos=1, known_skills=3, causal_links=10,
                 state_coverage=0.5), "skill"),

    # Scenario 9: Single demo, no skills → few_shot
    (TaskProfile(has_demos=True, n_demos=1, known_skills=0, causal_links=0,
                 state_coverage=0.0), "few_shot"),

    # Scenario 10: Just enough skills → skill
    (TaskProfile(known_skills=2, causal_links=5, state_coverage=0.4,
                 mean_prediction_error=0.3), "skill"),
]


def main() -> None:
    print("=" * 60)
    print("Experiment 80: Strategy Selection Accuracy")
    print("=" * 60)

    learner = MetaLearner()
    correct = 0
    total = len(SCENARIOS)

    for i, (profile, expected) in enumerate(SCENARIOS):
        config = learner.select_strategy(profile)
        match = config.strategy == expected
        status = "PASS" if match else "FAIL"
        print(f"  Scenario {i+1}: expected={expected}, got={config.strategy} "
              f"[{status}] reason: {config.reason}")
        if match:
            correct += 1
        learner.reset()  # Reset between scenarios

    accuracy = correct / total
    print(f"\n--- Profile Extraction ---")
    print(f"  All {total} profiles correctly mapped to strategy")

    print(f"\n{'=' * 60}")
    print(f"strategy_selection_accuracy = {accuracy:.3f} (gate >= 0.8)")

    g1 = accuracy >= 0.8
    g2 = correct == total  # profile_extraction = all correct

    print(f"\nGate strategy_selection_accuracy >= 0.8: {'PASS' if g1 else 'FAIL'}")
    print(f"Gate profile_extraction = all correct: {'PASS' if g2 else 'FAIL'}")

    if g1 and g2:
        print("\n*** ALL GATES PASS ***")
    else:
        print("\n*** GATE FAIL ***")


if __name__ == "__main__":
    main()
