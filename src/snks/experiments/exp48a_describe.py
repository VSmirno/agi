"""Experiment 48a: Describe state recall & precision (Stage 21).

Tests that Verbalizer.describe_state correctly mentions all grounded
objects (recall) and doesn't mention false objects (precision).

Synthetic data: no DAF/pipeline needed.

Gate: recall > 0.7, precision = 1.0
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.language.grounding_map import GroundingMap
from snks.language.verbalizer import Verbalizer


MINIGRID_ACTIONS = {
    0: "go left", 1: "go right", 2: "go forward",
    3: "pick up", 4: "drop", 5: "toggle", 6: "done",
}

# Vocabulary: (word, sks_id)
VOCAB = [
    ("key", 10),
    ("door", 20),
    ("ball", 30),
    ("box", 40),
    ("wall", 50),
    ("goal", 60),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


# Test scenarios: (active_sks, expected_words)
SCENARIOS = [
    # All grounded
    ([10, 20, 30], {"key", "door", "ball"}),
    # Mix of grounded and ungrounded
    ([10, 99, 20, 100], {"key", "door"}),
    # Single object
    ([40], {"box"}),
    # All grounded, full vocab
    ([10, 20, 30, 40, 50, 60], {"key", "door", "ball", "box", "wall", "goal"}),
    # Only ungrounded — expect empty
    ([99, 100, 101], set()),
    # Overlap + noise
    ([10, 30, 50, 77, 88], {"key", "ball", "wall"}),
    # Two objects
    ([20, 60], {"door", "goal"}),
    # Repeated grounded IDs
    ([10, 10, 20], {"key", "door"}),
]


def run(device: str = "cpu") -> dict:
    """Run experiment 48a: describe state recall & precision."""
    print("\n" + "=" * 60)
    print("Exp 48a: Describe state — recall & precision")
    print("=" * 60)

    gmap = _make_gmap()
    v = Verbalizer(gmap, MINIGRID_ACTIONS)

    total_recall_sum = 0.0
    total_precision_sum = 0.0
    n_scenarios = 0

    for active_sks, expected_words in SCENARIOS:
        result = v.describe_state(active_sks)
        print(f"\n  Active SKS: {active_sks}")
        print(f"  Output: '{result}'")

        if not expected_words:
            # Expect empty output
            if result == "":
                recall = 1.0
                precision = 1.0
            else:
                recall = 0.0
                precision = 0.0
        else:
            # Check recall: how many expected words are in the output?
            found = sum(1 for w in expected_words if w in result)
            recall = found / len(expected_words)

            # Check precision: are there any words from VOCAB that shouldn't be there?
            all_vocab_words = {w for w, _ in VOCAB}
            mentioned = {w for w in all_vocab_words if w in result}
            false_positives = mentioned - expected_words
            if mentioned:
                precision = 1.0 - len(false_positives) / len(mentioned)
            else:
                precision = 0.0  # nothing mentioned but should have been

        print(f"  Recall: {recall:.2f}, Precision: {precision:.2f}")
        total_recall_sum += recall
        total_precision_sum += precision
        n_scenarios += 1

    avg_recall = total_recall_sum / n_scenarios
    avg_precision = total_precision_sum / n_scenarios

    print(f"\n{'=' * 40}")
    print(f"Average recall:    {avg_recall:.3f}")
    print(f"Average precision: {avg_precision:.3f}")
    print(f"Gate (recall > 0.7): {'PASS' if avg_recall > 0.7 else 'FAIL'}")
    print(f"Gate (precision = 1.0): {'PASS' if avg_precision >= 1.0 else 'FAIL'}")

    passed = avg_recall > 0.7 and avg_precision >= 1.0
    print(f"Overall: {'PASS' if passed else 'FAIL'}")

    return {
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "n_scenarios": n_scenarios,
        "pass": passed,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
