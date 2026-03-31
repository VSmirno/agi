"""Experiment 54b: Plan correctness (Stage 24b).

Tests InstructionPlanner produces correct action sequences
from parsed BabyAI instructions using synthetic CausalWorldModel.

Gate: accuracy > 0.7
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


# SKS IDs (perceptual hash range to avoid coarsening)
KEY = 10001
DOOR = 10002
BALL = 10003
BOX = 10004
GOAL = 10005
KEY_HELD = 10006
DOOR_OPEN = 10007
NEAR_BALL = 10008
NEAR_GOAL = 10009
BOX_OPEN = 10010

# Action IDs
PICKUP = 3
OPEN = 5
GOTO = 8
DROP = 4
TOGGLE = 6

ACTION_NAMES = {
    "pick up": PICKUP, "open": OPEN, "go to": GOTO,
    "drop": DROP, "toggle": TOGGLE,
}

VOCAB = [
    ("key", KEY), ("door", DOOR), ("ball", BALL), ("box", BOX), ("goal", GOAL),
    ("pick up", PICKUP), ("open", OPEN), ("go to", GOTO), ("drop", DROP), ("toggle", TOGGLE),
    ("red", 10020), ("blue", 10021), ("green", 10022), ("yellow", 10023),
]


def _setup():
    cfg = CausalAgentConfig()
    cfg.causal_min_observations = 1

    cm = CausalWorldModel(cfg)
    # Train causal transitions
    for _ in range(5):
        cm.observe_transition({KEY}, PICKUP, {KEY, KEY_HELD})
        cm.observe_transition({DOOR, KEY_HELD}, OPEN, {DOOR_OPEN})
        cm.observe_transition({BALL}, GOTO, {BALL, NEAR_BALL})
        cm.observe_transition({GOAL}, GOTO, {GOAL, NEAR_GOAL})
        cm.observe_transition({BOX}, OPEN, {BOX, BOX_OPEN})
        cm.observe_transition({KEY_HELD}, DROP, {KEY})

    sim = StochasticSimulator(cm, seed=42)

    gmap = GroundingMap()
    for word, sks_id in VOCAB:
        gmap.register(word, sks_id, torch.zeros(64))

    planner = InstructionPlanner(gmap, cm, sim, ACTION_NAMES)
    chunker = RuleBasedChunker()

    return planner, chunker


# (instruction, current_state, expected_actions_contain)
TEST_CASES = [
    # Simple GoTo
    ("go to the ball", {BALL}, [GOTO]),
    ("go to the goal", {GOAL}, [GOTO]),
    # Simple Pickup
    ("pick up the key", {KEY}, [PICKUP]),
    # Simple Open
    ("open the box", {BOX}, [OPEN]),
    # Sequential
    ("pick up the key then go to the goal", {KEY, GOAL}, [PICKUP, GOTO]),
    ("go to the ball then pick up the key", {BALL, KEY}, [GOTO, PICKUP]),
    # Open door (with key already held)
    ("open the door", {DOOR, KEY_HELD}, [OPEN]),
    # Drop
    ("drop the key", {KEY_HELD}, [DROP]),
    # Triple sequential
    ("pick up the key then open the door then go to the goal",
     {KEY, DOOR, GOAL}, [PICKUP, OPEN, GOTO]),
    # GoTo with color (color doesn't affect action)
    ("go to the red ball", {BALL}, [GOTO]),
    ("pick up the blue key", {KEY}, [PICKUP]),
    ("open the yellow door", {DOOR, KEY_HELD}, [OPEN]),
]


def run(device: str = "cpu") -> dict:
    print("\n" + "=" * 60)
    print("Exp 54b: Plan Correctness")
    print("=" * 60)

    planner, chunker = _setup()

    correct = 0
    total = len(TEST_CASES)

    for instruction, state, expected_actions in TEST_CASES:
        chunks = chunker.chunk(instruction)
        plan = planner.plan(chunks, current_sks=state)

        # Check: all expected actions present in plan, in order
        ok = True
        last_idx = -1
        for exp_a in expected_actions:
            if exp_a not in plan:
                ok = False
                break
            idx = plan.index(exp_a)
            if idx <= last_idx:
                ok = False
                break
            last_idx = idx

        if ok:
            correct += 1
        status = "✓" if ok else "✗"
        print(f"  {status} {instruction!r}")
        print(f"      plan={plan}, expected={expected_actions}")

    accuracy = correct / total
    passed = accuracy > 0.7

    print(f"\n{'=' * 40}")
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")
    print(f"Gate (> 0.7): {'PASS' if passed else 'FAIL'}")

    return {"accuracy": accuracy, "correct": correct, "total": total, "pass": passed}


if __name__ == "__main__":
    result = run()
    sys.exit(0 if result["pass"] else 1)
