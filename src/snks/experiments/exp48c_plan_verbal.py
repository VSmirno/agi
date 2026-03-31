"""Experiment 48c: Plan verbalization correctness (Stage 21).

Tests that Verbalizer.verbalize_plan correctly converts action sequences
into human-readable plan descriptions.

Gate: accuracy > 0.7
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.grounding_map import GroundingMap
from snks.language.verbalizer import Verbalizer


MINIGRID_ACTIONS = {
    0: "go left", 1: "go right", 2: "go forward",
    3: "pick up", 4: "drop", 5: "toggle", 6: "done",
}

# Synthetic world
OBJECTS = {
    "key": 10,
    "key held": 11,
    "door": 20,
    "door open": 21,
    "ball": 30,
    "ball held": 31,
}

# Causal chain: key → pickup → key_held → toggle → door_open
TRANSITIONS = [
    ({10}, 3, {10, 11}),                     # pickup key → key held
    ({10, 11, 20}, 5, {10, 11, 20, 21}),     # toggle door → door open
    ({30}, 3, {30, 31}),                     # pickup ball → ball held
]

# Test plans: (action_ids, initial_sks, expected_parts)
# expected_parts: list of (action_substr, object_substr) that must appear in output
# NOTE: initial_sks must match the training contexts exactly because
# CausalWorldModel uses context hashing for lookup.
PLANS = [
    # Two-step plan: pick up key, then toggle door
    # Step 1 context = {10}, step 2 context = {10, 11, 20} (after adding door)
    # We need door (20) in state for step 2 to match training context.
    # So use intermediate state construction matching the training data.
    (
        [3],
        {10},
        [("pick up", "key")],
    ),
    # Toggle door (context must be {10, 11, 20} to match training)
    (
        [5],
        {10, 11, 20},
        [("toggle", "door")],
    ),
    # Single-step plan: pick up ball
    (
        [3],
        {30},
        [("pick up", "ball")],
    ),
]


def run(device: str = "cpu") -> dict:
    """Run experiment 48c: plan verbalization."""
    print("\n" + "=" * 60)
    print("Exp 48c: Plan verbalization correctness")
    print("=" * 60)

    # Build grounding map
    gmap = GroundingMap()
    for word, sks_id in OBJECTS.items():
        gmap.register(word, sks_id, torch.zeros(64))

    # Build causal model
    cfg = CausalAgentConfig(causal_min_observations=2, causal_context_bins=64)
    model = CausalWorldModel(cfg)
    for _ in range(10):
        for pre, action, post in TRANSITIONS:
            model.observe_transition(pre, action, post)

    v = Verbalizer(gmap, MINIGRID_ACTIONS)

    correct = 0
    total = len(PLANS)

    for action_ids, initial_sks, expected_parts in PLANS:
        result = v.verbalize_plan(action_ids, initial_sks, model)
        print(f"\n  Plan: actions={action_ids}, initial_sks={initial_sks}")
        print(f"  Output: '{result}'")

        # Check: output starts with "I need to" and contains expected parts
        has_prefix = result.startswith("I need to")
        parts_ok = all(
            action in result and obj in result
            for action, obj in expected_parts
        )

        if has_prefix and parts_ok:
            correct += 1
            print(f"  PASS")
        else:
            print(f"  FAIL: prefix={has_prefix}, parts_ok={parts_ok}")

    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 40}")
    print(f"Accuracy: {correct}/{total} = {accuracy:.3f}")
    print(f"Gate (> 0.7): {'PASS' if accuracy > 0.7 else 'FAIL'}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "pass": accuracy > 0.7,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
