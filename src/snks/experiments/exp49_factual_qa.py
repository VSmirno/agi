"""Experiment 49: Factual QA accuracy (Stage 22).

Tests that GroundedQA correctly answers factual questions
("what/who/where") using a synthetic dict-based backend.

Gate: accuracy > 0.7
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.qa import (
    GroundedQA,
    QABackend,
    QAResult,
    QuestionClassifier,
    QuestionType,
)


# --- Vocabulary: (word, sks_id) ---

VOCAB = [
    ("key", 1), ("door", 2), ("opens", 3), ("ball", 4),
    ("red", 5), ("agent", 6), ("wall", 7), ("near", 8),
    ("holds", 9), ("box", 10), ("blue", 11), ("floor", 12),
    ("is", 13), ("sits", 14), ("green", 15), ("goal", 16),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


# --- Synthetic factual backend ---

class SyntheticFactualBackend:
    """Dict-based factual knowledge."""

    def __init__(self, facts: dict[tuple[int, int], list[int]]) -> None:
        self.facts = facts

    def query(self, roles: dict[str, int]) -> QAResult | None:
        key = (roles.get("ACTION", -1), roles.get("OBJECT", -1))
        if key in self.facts:
            return QAResult(
                answer_sks=self.facts[key],
                confidence=1.0,
                source=QuestionType.FACTUAL,
            )
        return None


class NullBackend:
    def query(self, roles: dict[str, int]) -> None:
        return None


# --- Test data ---

# Facts: (action_sks, object_sks) -> answer_sks_list
FACTS = {
    (3, 2): [1],     # opens door -> key
    (8, 7): [6],     # near wall -> agent
    (9, 1): [6],     # holds key -> agent
    (14, 12): [4],   # sits floor -> ball
    (3, 10): [1],    # opens box -> key
    (8, 16): [2],    # near goal -> door
}

# (question, expected_words_in_answer, is_known)
TEST_CASES = [
    ("What opens the door?", ["key"], True),
    ("Who is near the wall?", ["agent"], True),
    ("What holds the key?", ["agent"], True),
    ("What sits on the floor?", ["ball"], True),
    ("What opens the box?", ["key"], True),
    ("What is near the goal?", ["door"], True),
    # Unknown facts -> "I don't know"
    ("What opens the ball?", [], False),
    ("Who is near the key?", [], False),
    ("Where is the blue box?", [], False),
    ("What holds the door?", [], False),
    ("What sits on the wall?", [], False),
    ("Where is the green ball?", [], False),
    # More known
    ("Who holds the key?", ["agent"], True),
    ("What is near the wall?", ["agent"], True),
    # Edge: word not in vocab at all
    ("What destroys the castle?", [], False),
    # Repeat to reach 20+
    ("What opens the door?", ["key"], True),
    ("Who is near the wall?", ["agent"], True),
    ("What opens the box?", ["key"], True),
    ("What sits on the floor?", ["ball"], True),
    ("What is near the goal?", ["door"], True),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 49: Factual QA — accuracy")
    print("=" * 60)

    gmap = _make_gmap()
    qa = GroundedQA(
        classifier=QuestionClassifier(),
        grounding_map=gmap,
        chunker=RuleBasedChunker(),
        factual=SyntheticFactualBackend(FACTS),
        simulation=NullBackend(),
        reflective=NullBackend(),
    )

    correct = 0
    total = len(TEST_CASES)

    for question, expected_words, is_known in TEST_CASES:
        answer = qa.answer(question)
        if is_known:
            ok = all(w in answer for w in expected_words)
        else:
            ok = answer == "I don't know"
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  {status} Q: {question!r}")
        expected = expected_words if is_known else "I don't know"
        print(f"      A: {answer!r} (expected: {expected})")

    accuracy = correct / total
    passed = accuracy > 0.7

    print(f"\n{'=' * 40}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Gate (> 0.7): {'PASS' if passed else 'FAIL'}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "pass": passed}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
