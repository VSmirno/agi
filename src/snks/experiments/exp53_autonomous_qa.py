"""Experiment 53: Autonomous QA (Stage 23).

Verifies that QA pipeline works with GroundedTokenizer-based
GroundingMap (no sentence-transformers needed at inference time).

Repeats Exp 49 factual QA setup. Gate: accuracy > 0.8 × Exp49 accuracy.
Since QA uses GroundingMap.word_to_sks() (not TextEncoder), accuracy = Exp49.

Gate: accuracy > 0.6
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.daf.types import EncoderConfig
from snks.encoder.grounded_tokenizer import GroundedTokenizer
from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.qa import (
    GroundedQA,
    QAResult,
    QuestionClassifier,
    QuestionType,
)


SDR_SIZE = 4096
K = 164

VOCAB = [
    ("key", 1), ("door", 2), ("opens", 3), ("ball", 4),
    ("holds", 9), ("box", 10), ("sits", 14), ("floor", 12),
    ("near", 8), ("wall", 7), ("agent", 6), ("goal", 16),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    g = torch.Generator().manual_seed(42)
    for word, sks_id in VOCAB:
        sdr = torch.zeros(SDR_SIZE)
        indices = torch.randperm(SDR_SIZE, generator=g)[:K]
        sdr[indices] = 1.0
        gmap.register(word, sks_id, sdr)
    return gmap


class SyntheticFactualBackend:
    def __init__(self, facts):
        self.facts = facts

    def query(self, roles):
        key = (roles.get("ACTION", -1), roles.get("OBJECT", -1))
        if key in self.facts:
            return QAResult(
                answer_sks=self.facts[key],
                confidence=1.0,
                source=QuestionType.FACTUAL,
            )
        return None


class NullBackend:
    def query(self, roles):
        return None


FACTS = {
    (3, 2): [1],     # opens door -> key
    (9, 1): [6],     # holds key -> agent
    (14, 12): [4],   # sits floor -> ball
    (3, 10): [1],    # opens box -> key
    (8, 16): [2],    # near goal -> door
}

TEST_CASES = [
    ("What opens the door?", ["key"], True),
    ("What holds the key?", ["agent"], True),
    ("What sits on the floor?", ["ball"], True),
    ("What opens the box?", ["key"], True),
    # Unknown
    ("What opens the ball?", [], False),
    ("What holds the door?", [], False),
    ("What sits on the wall?", [], False),
    ("Where is the green ball?", [], False),
    # More known
    ("What opens the door?", ["key"], True),
    ("What holds the key?", ["agent"], True),
    ("What sits on the floor?", ["ball"], True),
    ("What opens the box?", ["key"], True),
    # More unknown
    ("What destroys the castle?", [], False),
    ("Where is the blue box?", [], False),
    ("What opens the goal?", [], False),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 53: Autonomous QA (GroundedTokenizer)")
    print("=" * 60)

    gmap = _make_gmap()
    config = EncoderConfig(sdr_size=SDR_SIZE, sdr_sparsity=0.04)

    # Verify GroundedTokenizer works for vocab
    tokenizer = GroundedTokenizer(gmap, config)
    print(f"\n  GroundedTokenizer vocab: {len(tokenizer.vocab)} words")
    for word, sks_id in VOCAB:
        sdr = tokenizer.encode(word)
        active = int(sdr.sum().item())
        print(f"    {word!r} -> {active} active bits (expected {K})")

    # Run QA pipeline (uses GroundingMap, not TextEncoder)
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

    print("\n  QA results:")
    for question, expected_words, is_known in TEST_CASES:
        answer = qa.answer(question)
        if is_known:
            ok = all(w in answer for w in expected_words)
        else:
            ok = answer == "I don't know"
        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"    {status} Q: {question!r} -> {answer!r}")

    accuracy = correct / total
    # Gate: > 0.8 × Exp49 accuracy (0.750) = 0.600
    passed = accuracy > 0.6

    print(f"\n{'=' * 40}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Gate (> 0.6): {'PASS' if passed else 'FAIL'}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "pass": passed}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
