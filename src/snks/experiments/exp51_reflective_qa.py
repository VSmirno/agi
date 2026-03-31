"""Experiment 51: Reflective QA accuracy (Stage 22).

Tests that GroundedQA correctly answers "why did you" questions
using a synthetic dict-based metacog backend.

Gate: accuracy > 0.6
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.qa import (
    GroundedQA,
    QAResult,
    QuestionClassifier,
    QuestionType,
)


VOCAB = [
    ("go", 1), ("left", 2), ("right", 3), ("forward", 4),
    ("pick up", 5), ("open", 6), ("drop", 7), ("stop", 8),
    ("turn", 9), ("explore", 10),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


class SyntheticReflectiveBackend:
    """Dict-based metacog log: action_sks -> reason."""

    def __init__(self, log: dict[int, dict]) -> None:
        self.log = log

    def query(self, roles: dict[str, int]) -> QAResult | None:
        action_sks = roles.get("ACTION", -1)
        if action_sks in self.log:
            entry = self.log[action_sks]
            return QAResult(
                answer_sks=[],
                confidence=1.0,
                source=QuestionType.REFLECTIVE,
                metadata={"reason": entry["reason"]},
            )
        return None


class NullBackend:
    def query(self, roles: dict[str, int]) -> None:
        return None


# Metacog log: action_sks -> reason
METACOG_LOG = {
    1: {"reason": "Prediction error was high. I explored left."},       # go
    5: {"reason": "It was my goal to get the key."},                     # pick up
    6: {"reason": "The door was blocking the path to the goal."},        # open
    7: {"reason": "I no longer needed the object."},                     # drop
    8: {"reason": "All goals were achieved."},                           # stop
    9: {"reason": "I needed to change direction to explore."},           # turn
}

# (question, expected_substring, is_known)
TEST_CASES = [
    ("Why did you go left?", "Prediction error", True),
    ("Why did you pick up the key?", "goal", True),
    ("Why did you open the door?", "blocking", True),
    ("Why did you drop the ball?", "no longer needed", True),
    ("Why did you stop?", "goals were achieved", True),
    ("Why did you turn right?", "change direction", True),
    # Unknown actions -> "I don't know"
    ("Why did you explore the room?", "", False),
    ("Why are you going forward?", "", False),
    # Repeats
    ("Why did you go right?", "Prediction error", True),
    ("Why did you pick up the box?", "goal", True),
    ("Why did you open the box?", "blocking", True),
    ("Why did you drop the key?", "no longer needed", True),
    ("Why did you stop?", "goals were achieved", True),
    ("Why did you turn left?", "change direction", True),
    # More unknown
    ("Why are you exploring?", "", False),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 51: Reflective QA — accuracy")
    print("=" * 60)

    gmap = _make_gmap()
    qa = GroundedQA(
        classifier=QuestionClassifier(),
        grounding_map=gmap,
        chunker=RuleBasedChunker(),
        factual=NullBackend(),
        simulation=NullBackend(),
        reflective=SyntheticReflectiveBackend(METACOG_LOG),
    )

    correct = 0
    total = len(TEST_CASES)

    for question, expected_sub, is_known in TEST_CASES:
        answer = qa.answer(question)
        if is_known:
            ok = expected_sub in answer
        else:
            ok = answer == "I don't know"
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  {status} Q: {question!r}")
        print(f"      A: {answer!r}")

    accuracy = correct / total
    passed = accuracy > 0.6

    print(f"\n{'=' * 40}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Gate (> 0.6): {'PASS' if passed else 'FAIL'}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "pass": passed}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
