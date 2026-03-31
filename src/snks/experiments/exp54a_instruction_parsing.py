"""Experiment 54a: Instruction parsing accuracy (Stage 24a).

Tests RuleBasedChunker on all 5 BabyAI instruction types:
GoTo, Pickup, Open, Sequential, Spatial.

Gate: accuracy > 0.9
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap


VOCAB = [
    ("key", 1), ("door", 2), ("ball", 3), ("box", 4), ("goal", 5), ("wall", 6),
    ("red", 10), ("blue", 11), ("green", 12), ("yellow", 13), ("purple", 14), ("grey", 15),
    ("pick up", 20), ("go to", 21), ("open", 22), ("put", 23), ("drop", 24), ("toggle", 25),
]


def _make_gmap() -> GroundingMap:
    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


# (instruction, expected_roles: list of (text, role))
# Partial order: we check that all expected roles appear in output.
TEST_INSTRUCTIONS = [
    # --- GoTo (6) ---
    ("go to the red ball",
     [("go to", "ACTION"), ("red", "ATTR"), ("ball", "OBJECT")]),
    ("go to the door",
     [("go to", "ACTION"), ("door", "OBJECT")]),
    ("go to the blue key",
     [("go to", "ACTION"), ("blue", "ATTR"), ("key", "OBJECT")]),
    ("go to the green box",
     [("go to", "ACTION"), ("green", "ATTR"), ("box", "OBJECT")]),
    ("go to the goal",
     [("go to", "ACTION"), ("goal", "OBJECT")]),
    ("go to the grey wall",
     [("go to", "ACTION"), ("grey", "ATTR"), ("wall", "OBJECT")]),

    # --- Pickup (6) ---
    ("pick up the blue key",
     [("pick up", "ACTION"), ("blue", "ATTR"), ("key", "OBJECT")]),
    ("pick up the ball",
     [("pick up", "ACTION"), ("ball", "OBJECT")]),
    ("pick up the red box",
     [("pick up", "ACTION"), ("red", "ATTR"), ("box", "OBJECT")]),
    ("pick up the yellow key",
     [("pick up", "ACTION"), ("yellow", "ATTR"), ("key", "OBJECT")]),
    ("pick up the green ball",
     [("pick up", "ACTION"), ("green", "ATTR"), ("ball", "OBJECT")]),
    ("pick up the key",
     [("pick up", "ACTION"), ("key", "OBJECT")]),

    # --- Open (4) ---
    ("open the yellow door",
     [("open", "ACTION"), ("yellow", "ATTR"), ("door", "OBJECT")]),
    ("open the door",
     [("open", "ACTION"), ("door", "OBJECT")]),
    ("open the purple box",
     [("open", "ACTION"), ("purple", "ATTR"), ("box", "OBJECT")]),
    ("open the red door",
     [("open", "ACTION"), ("red", "ATTR"), ("door", "OBJECT")]),

    # --- Sequential (8) ---
    ("pick up the key then open the door",
     [("pick up", "ACTION"), ("key", "OBJECT"), ("", "SEQ_BREAK"),
      ("open", "ACTION"), ("door", "OBJECT")]),
    ("go to the ball and then pick up the key",
     [("go to", "ACTION"), ("ball", "OBJECT"), ("", "SEQ_BREAK"),
      ("pick up", "ACTION"), ("key", "OBJECT")]),
    ("pick up the red key then open the yellow door",
     [("pick up", "ACTION"), ("red", "ATTR"), ("key", "OBJECT"), ("", "SEQ_BREAK"),
      ("open", "ACTION"), ("yellow", "ATTR"), ("door", "OBJECT")]),
    ("open the door then go to the goal",
     [("open", "ACTION"), ("door", "OBJECT"), ("", "SEQ_BREAK"),
      ("go to", "ACTION"), ("goal", "OBJECT")]),
    ("pick up the blue ball then drop the blue ball",
     [("pick up", "ACTION"), ("blue", "ATTR"), ("ball", "OBJECT"), ("", "SEQ_BREAK"),
      ("drop", "ACTION"), ("blue", "ATTR"), ("ball", "OBJECT")]),
    ("go to the red key then pick up the red key",
     [("go to", "ACTION"), ("red", "ATTR"), ("key", "OBJECT"), ("", "SEQ_BREAK"),
      ("pick up", "ACTION"), ("red", "ATTR"), ("key", "OBJECT")]),
    ("open the door then go to the goal then pick up the key",
     [("open", "ACTION"), ("door", "OBJECT"), ("", "SEQ_BREAK"),
      ("go to", "ACTION"), ("goal", "OBJECT"), ("", "SEQ_BREAK"),
      ("pick up", "ACTION"), ("key", "OBJECT")]),
    ("pick up the key then open the door then go to the goal",
     [("pick up", "ACTION"), ("key", "OBJECT"), ("", "SEQ_BREAK"),
      ("open", "ACTION"), ("door", "OBJECT"), ("", "SEQ_BREAK"),
      ("go to", "ACTION"), ("goal", "OBJECT")]),

    # --- Spatial (6) ---
    ("put the ball next to the box",
     [("put", "ACTION"), ("ball", "OBJECT"), ("box", "LOCATION")]),
    ("put the red ball next to the blue box",
     [("put", "ACTION"), ("red", "ATTR"), ("ball", "OBJECT"),
      ("blue", "ATTR"), ("box", "LOCATION")]),
    ("put the key next to the door",
     [("put", "ACTION"), ("key", "OBJECT"), ("door", "LOCATION")]),
    ("put the green ball next to the red box",
     [("put", "ACTION"), ("green", "ATTR"), ("ball", "OBJECT"),
      ("red", "ATTR"), ("box", "LOCATION")]),
    ("put the yellow key next to the blue ball",
     [("put", "ACTION"), ("yellow", "ATTR"), ("key", "OBJECT"),
      ("blue", "ATTR"), ("ball", "LOCATION")]),
    ("put the box next to the wall",
     [("put", "ACTION"), ("box", "OBJECT"), ("wall", "LOCATION")]),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 54a: Instruction Parsing — accuracy")
    print("=" * 60)

    chunker = RuleBasedChunker()
    gmap = _make_gmap()

    correct = 0
    total = len(TEST_INSTRUCTIONS)
    grounding_hits = 0
    grounding_total = 0

    for instruction, expected_roles in TEST_INSTRUCTIONS:
        chunks = chunker.chunk(instruction)
        actual = [(c.text, c.role) for c in chunks]

        # Check: all expected roles present in actual
        matched = sum(1 for exp in expected_roles if exp in actual)
        score = matched / len(expected_roles) if expected_roles else 1.0
        ok = score >= 0.8  # partial match threshold

        if ok:
            correct += 1

        # Grounding resolve check
        for c in chunks:
            if c.role in ("ACTION", "OBJECT", "ATTR", "LOCATION") and c.text:
                grounding_total += 1
                if gmap.word_to_sks(c.text) is not None:
                    grounding_hits += 1

        status = "✓" if ok else "✗"
        print(f"  {status} [{score:.0%}] {instruction!r}")
        if not ok:
            print(f"      expected: {expected_roles}")
            print(f"      actual:   {actual}")

    accuracy = correct / total
    grounding_ratio = grounding_hits / grounding_total if grounding_total else 0
    passed = accuracy > 0.9

    print(f"\n{'=' * 40}")
    print(f"Parsing accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Grounding resolve: {grounding_ratio:.3f} ({grounding_hits}/{grounding_total})")
    print(f"Gate (> 0.9): {'PASS' if passed else 'FAIL'}")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "grounding_ratio": grounding_ratio,
        "pass": passed,
    }


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
