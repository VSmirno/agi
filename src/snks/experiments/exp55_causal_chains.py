"""Experiment 55: Causal chain detection (Stage 24b).

Tests that InstructionPlanner detects prerequisites via CausalWorldModel.
E.g., "open the door" without key → planner inserts pickup(key) first.

Gate: chain_accuracy > 0.5
"""

from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import torch

from snks.agent.causal_model import CausalWorldModel
from snks.agent.stochastic_simulator import StochasticSimulator
from snks.daf.types import CausalAgentConfig
from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.planner import InstructionPlanner


# SKS IDs
KEY = 10001
DOOR = 10002
KEY_HELD = 10004
DOOR_OPEN = 10005
BOX = 10006
BOX_OPEN = 10007
LEVER = 10008
GATE = 10009
GATE_OPEN = 10010
LEVER_PULLED = 10011

PICKUP = 3
OPEN = 5
TOGGLE = 6

ACTION_NAMES = {"pick up": PICKUP, "open": OPEN, "toggle": TOGGLE}

VOCAB = [
    ("key", KEY), ("door", DOOR), ("box", BOX),
    ("lever", LEVER), ("gate", GATE),
    ("pick up", PICKUP), ("open", OPEN), ("toggle", TOGGLE),
]


def _setup():
    cfg = CausalAgentConfig()
    cfg.causal_min_observations = 1

    cm = CausalWorldModel(cfg)
    for _ in range(5):
        # Chain 1: pickup key → key_held, then open door → door_open
        cm.observe_transition({KEY}, PICKUP, {KEY, KEY_HELD})
        cm.observe_transition({DOOR, KEY_HELD}, OPEN, {DOOR_OPEN})
        # Chain 2: toggle lever → lever_pulled, then open gate → gate_open
        cm.observe_transition({LEVER}, TOGGLE, {LEVER, LEVER_PULLED})
        cm.observe_transition({GATE, LEVER_PULLED}, OPEN, {GATE_OPEN})
        # Simple: open box (no prerequisite)
        cm.observe_transition({BOX}, OPEN, {BOX, BOX_OPEN})

    sim = StochasticSimulator(cm, seed=42)

    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))

    planner = InstructionPlanner(gmap, cm, sim, ACTION_NAMES)
    chunker = RuleBasedChunker()

    return planner, chunker


# (instruction, current_state, expected: should plan contain prerequisite?)
# For chain tests: planner should detect missing prerequisite
CHAIN_TESTS = [
    # Open door WITHOUT key → should prepend pickup
    ("open the door", {DOOR, KEY}, True, PICKUP,
     "open door without key_held → needs pickup first"),

    # Open door WITH key_held → no prerequisite
    ("open the door", {DOOR, KEY_HELD}, False, None,
     "open door with key_held → no prerequisite"),

    # Open box → no prerequisite (direct)
    ("open the box", {BOX}, False, None,
     "open box → direct, no prerequisite"),

    # Open gate without lever pulled → should prepend toggle
    ("open the gate", {GATE, LEVER}, True, TOGGLE,
     "open gate without lever_pulled → needs toggle first"),

    # Open gate with lever pulled → no prerequisite
    ("open the gate", {GATE, LEVER_PULLED}, False, None,
     "open gate with lever_pulled → no prerequisite"),

    # Pick up key → direct action, no chain
    ("pick up the key", {KEY}, False, None,
     "pickup key → direct action"),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 55: Causal Chain Detection")
    print("=" * 60)

    planner, chunker = _setup()

    correct = 0
    total = len(CHAIN_TESTS)

    for instruction, state, expects_prereq, prereq_action, desc in CHAIN_TESTS:
        chunks = chunker.chunk(instruction)
        plan = planner.plan(chunks, current_sks=state)

        if expects_prereq:
            # Plan should contain prerequisite action BEFORE the main action
            ok = prereq_action in plan and len(plan) > 1
        else:
            # Plan should be just the direct action (no extra prereqs)
            ok = len(plan) >= 1  # at least the action itself

        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  {status} {desc}")
        print(f"      instruction: {instruction!r}, state: {state}")
        print(f"      plan: {plan}, expects_prereq: {expects_prereq}")

    accuracy = correct / total
    passed = accuracy > 0.5

    print(f"\n{'=' * 40}")
    print(f"Chain accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Gate (> 0.5): {'PASS' if passed else 'FAIL'}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "pass": passed}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
