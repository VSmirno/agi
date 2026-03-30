"""Experiment 48b: Causal verbalization correctness (Stage 21).

Tests that Verbalizer.explain_causal produces correct causal phrases
for known synthetic transitions.

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

# Synthetic world: objects and their SKS IDs
OBJECTS = {
    "key": 10,
    "key held": 11,
    "door": 20,
    "door open": 21,
    "ball": 30,
    "box": 40,
}

# Known causal transitions: (pre_sks, action_id, post_sks)
TRANSITIONS = [
    ({10}, 3, {10, 11}),           # see key, pick up → key held
    ({10, 11, 20}, 5, {10, 11, 20, 21}),  # key held + door, toggle → door open
    ({30}, 3, {30, 40}),           # see ball, pick up → ball in box (simplified)
]

# Expected: for each query sks_id, what action and effect words should appear
QUERIES = [
    # (sks_id, expected_action_substr, expected_words_in_output)
    (10, "pick up", ["key"]),          # key is in context of pickup transition
    (20, "toggle", ["door"]),          # door is in context/effect of toggle
    (30, "pick up", ["ball"]),         # ball pickup
]


def run(device: str = "cpu") -> dict:
    """Run experiment 48b: causal verbalization."""
    print("\n" + "=" * 60)
    print("Exp 48b: Causal verbalization correctness")
    print("=" * 60)

    # Build grounding map
    gmap = GroundingMap()
    for word, sks_id in OBJECTS.items():
        gmap.register(word, sks_id, torch.zeros(64))

    # Build causal model with repeated observations
    cfg = CausalAgentConfig(causal_min_observations=2, causal_context_bins=64)
    model = CausalWorldModel(cfg)
    for _ in range(10):
        for pre, action, post in TRANSITIONS:
            model.observe_transition(pre, action, post)

    v = Verbalizer(gmap, MINIGRID_ACTIONS)

    correct = 0
    total = len(QUERIES)

    for sks_id, expected_action, expected_words in QUERIES:
        result = v.explain_causal(sks_id, model)
        print(f"\n  Query SKS: {sks_id} ({gmap.sks_to_word(sks_id)})")
        print(f"  Output: '{result}'")

        # Check: output is non-empty and contains expected elements
        action_ok = expected_action in result if result else False
        words_ok = all(w in result for w in expected_words) if result else False

        if action_ok and words_ok:
            correct += 1
            print(f"  PASS (action='{expected_action}', words={expected_words})")
        else:
            print(f"  FAIL: action_ok={action_ok}, words_ok={words_ok}")

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
