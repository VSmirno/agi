"""Experiment 50: Simulation QA accuracy (Stage 22).

Tests that GroundedQA correctly answers "what happens if" questions
using a synthetic dict-based simulation backend.

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
    QABackend,
    QAResult,
    QuestionClassifier,
    QuestionType,
)


VOCAB = [
    ("key", 1), ("door", 2), ("ball", 4), ("box", 10),
    ("pick up", 9), ("open", 3), ("drop", 5), ("toggle", 7),
    ("go to", 8), ("wall", 11), ("goal", 12),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


class SyntheticSimulationBackend:
    """Dict-based simulation: (action_sks, object_sks) -> effects."""

    def __init__(self, effects: dict[tuple[int, int], dict]) -> None:
        self.effects = effects

    def query(self, roles: dict[str, int]) -> QAResult | None:
        key = (roles.get("ACTION", -1), roles.get("OBJECT", -1))
        if key in self.effects:
            entry = self.effects[key]
            return QAResult(
                answer_sks=entry["answer_sks"],
                confidence=1.0,
                source=QuestionType.SIMULATION,
                metadata={"action": entry.get("action", "")},
            )
        return None


class NullBackend:
    def query(self, roles: dict[str, int]) -> None:
        return None


# Simulation effects
EFFECTS = {
    (9, 1): {"answer_sks": [1], "action": "pick up"},     # pick up key -> key held
    (3, 2): {"answer_sks": [2], "action": "open"},         # open door -> door open
    (9, 4): {"answer_sks": [4], "action": "pick up"},     # pick up ball -> ball held
    (5, 1): {"answer_sks": [], "action": "drop"},          # drop key -> nothing (empty)
    (7, 2): {"answer_sks": [2], "action": "toggle"},       # toggle door -> door state
    (9, 10): {"answer_sks": [10], "action": "pick up"},   # pick up box -> box held
    (3, 10): {"answer_sks": [10], "action": "open"},       # open box -> box open
}

# (question, check_fn_name, is_known)
# check_fn_name: "contains_word" or "nothing" or "dont_know"
TEST_CASES = [
    ("What happens if I pick up the key?", ["key"], True),
    ("What happens if I open the door?", ["door"], True),
    ("What happens if I pick up the ball?", ["ball"], True),
    ("What would happen if I drop the key?", [], True),      # empty effects -> "nothing happens"
    ("What happens if I toggle the door?", ["door"], True),
    ("What happens if I pick up the box?", ["box"], True),
    ("What happens if I open the box?", ["box"], True),
    # Unknown effects -> "I don't know"
    ("What happens if I open the wall?", [], False),
    ("What happens if I pick up the goal?", [], False),
    ("What happens if I drop the ball?", [], False),
    ("What would happen if I toggle the wall?", [], False),
    # Repeats for volume
    ("What happens if I pick up the key?", ["key"], True),
    ("What happens if I open the door?", ["door"], True),
    ("What happens if I pick up the ball?", ["ball"], True),
    ("What would happen if I drop the key?", [], True),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 50: Simulation QA — accuracy")
    print("=" * 60)

    gmap = _make_gmap()
    qa = GroundedQA(
        classifier=QuestionClassifier(),
        grounding_map=gmap,
        chunker=RuleBasedChunker(),
        factual=NullBackend(),
        simulation=SyntheticSimulationBackend(EFFECTS),
        reflective=NullBackend(),
    )

    correct = 0
    total = len(TEST_CASES)

    for question, expected_words, is_known in TEST_CASES:
        answer = qa.answer(question)
        if is_known:
            if expected_words:
                ok = all(w in answer for w in expected_words)
            else:
                # Empty effects -> "nothing happens"
                ok = answer == "nothing happens"
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
