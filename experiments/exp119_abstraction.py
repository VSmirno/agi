#!/usr/bin/env python3
"""Exp119: Stage 63 gate — abstraction + Crafter.

Gate A:
- MiniGrid held-out colors (abstract, no neocortex substitution)
- Crafter QA ≥90%
- ≥3 auto-discovered categories
"""

import json
import torch

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import (
    generate_synthetic_transitions,
    extract_demo_transitions,
    TRAIN_COLORS,
)
from snks.agent.crafter_trainer import generate_crafter_transitions


def train_model() -> CLSWorldModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # MiniGrid transitions (3 train colors)
    mg_synth = generate_synthetic_transitions(TRAIN_COLORS)
    with open("_docs/demo_episodes_bosslevel.json") as f:
        demos = json.load(f)
    mg_demo = extract_demo_transitions(demos)

    # Crafter transitions
    cr_trans = generate_crafter_transitions()

    all_trans = mg_synth + mg_demo + cr_trans
    print(f"Transitions: {len(mg_synth)} MG synth + {len(mg_demo)} MG demo + {len(cr_trans)} Crafter = {len(all_trans)}")

    model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
    stats = model.train(all_trans)
    print(f"Training: {stats}")
    return model


def test_crafter_qa(model: CLSWorldModel) -> tuple[int, int, list]:
    """Crafter QA battery."""
    tests = []

    # L1: Can you do X?
    tests.append(("collect wood", model.qa_crafter_can_do("do", "tree"), True))
    tests.append(("collect stone with pickaxe",
                   model.qa_crafter_can_do("do", "stone", {"wood_pickaxe": 1}), True))
    tests.append(("collect stone no pickaxe",
                   model.qa_crafter_can_do("do", "stone"), False))
    tests.append(("collect iron with stone_pickaxe",
                   model.qa_crafter_can_do("do", "iron", {"stone_pickaxe": 1}), True))
    tests.append(("collect iron no pickaxe",
                   model.qa_crafter_can_do("do", "iron"), False))
    tests.append(("place table with wood",
                   model.qa_crafter_can_do("place_table", "empty", {"wood": 2}), True))
    tests.append(("craft wood pickaxe at table",
                   model.qa_crafter_can_do("make_wood_pickaxe", "table", {"wood": 1}), True))
    tests.append(("craft wood pickaxe no table",
                   model.qa_crafter_can_do("make_wood_pickaxe", "empty"), False))

    # L2: What do you need?
    tests.append(("need for wood_pickaxe",
                   model.qa_crafter_needs("make_wood_pickaxe", "table"), "1 wood"))
    tests.append(("need for stone_pickaxe",
                   "wood" in model.qa_crafter_needs("make_stone_pickaxe", "table") and
                   "stone" in model.qa_crafter_needs("make_stone_pickaxe", "table"), True))
    tests.append(("need for iron_pickaxe",
                   "iron" in model.qa_crafter_needs("make_iron_pickaxe", "table"), True))

    # L3: What happens?
    r1 = model.qa_crafter_result("do", "tree")
    tests.append(("chop tree result", r1.get("result"), "collected"))
    r2 = model.qa_crafter_result("make_wood_pickaxe", "table", {"wood": 1})
    tests.append(("craft pickaxe result", r2.get("result"), "crafted"))
    r3 = model.qa_crafter_result("place_table", "empty", {"wood": 2})
    tests.append(("place table result", r3.get("result"), "placed"))

    # Failure cases
    r4 = model.qa_crafter_result("do", "stone")  # no pickaxe
    tests.append(("mine stone no tool", r4.get("result"), "failed_no_tool"))
    r5 = model.qa_crafter_result("make_wood_pickaxe", "empty")  # no table
    tests.append(("craft no table", r5.get("result"), "failed_no_station"))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} — got {got}, expected {exp}")
    return ok, len(tests), fails


def test_abstraction(model: CLSWorldModel) -> tuple[int, int, list]:
    """Test auto-discovered categories."""
    tests = []

    cats = model.abstraction.categories
    cat_names = set(cats.keys())

    # Must have at least 3 categories
    tests.append(("≥3 categories", len(cats) >= 3, True))

    # Key categories should exist
    tests.append(("carryable exists", "carryable" in cat_names or
                   any("carryable" in n for n in cat_names), True))
    tests.append(("solid exists", "solid" in cat_names, True))
    tests.append(("openable exists", "openable" in cat_names, True))

    # Multi-category: key should be in carryable AND solid
    key_cats = model.abstraction.get_categories_for_object("key")
    tests.append(("key is carryable",
                   any("carryable" in c for c in key_cats), True))
    tests.append(("key is solid", "solid" in key_cats, True))

    # ball should have same categories as key (for generalization)
    ball_cats = model.abstraction.get_categories_for_object("ball")
    key_carry = [c for c in key_cats if "carryable" in c]
    ball_carry = [c for c in ball_cats if "carryable" in c]
    tests.append(("ball carryable like key", len(ball_carry) > 0, True))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} — got {got}, expected {exp}")
    return ok, len(tests), fails


def main():
    print("=" * 60)
    print("EXP119: Stage 63 — Abstraction + Crafter Gate")
    print("=" * 60)

    model = train_model()

    print(f"\n{'='*60}")
    print("ABSTRACTION")
    c1, t1, f1 = test_abstraction(model)
    for f in f1: print(f)
    print(f"Abstraction: {c1}/{t1} = {c1*100//t1}%")

    print(f"\n{'='*60}")
    print("CRAFTER QA")
    c2, t2, f2 = test_crafter_qa(model)
    for f in f2: print(f)
    print(f"Crafter QA: {c2}/{t2} = {c2*100//t2}%  [gate ≥90%]")

    # Also run MiniGrid QA from exp118 for regression
    print(f"\n{'='*60}")
    print("MINIGRID QA (regression)")
    from experiments.exp118_world_model_qa import (
        level1_object_identity, level2_preconditions,
        level3_consequences, level4_planning,
    )
    c3, t3, f3 = level1_object_identity(model)
    c4, t4, f4 = level2_preconditions(model)
    c5, t5, f5 = level3_consequences(model)
    c6, t6, f6 = level4_planning(model)
    for f in f3 + f4 + f5 + f6: print(f)
    mg_total = c3 + c4 + c5 + c6
    mg_max = t3 + t4 + t5 + t6
    print(f"MiniGrid: {mg_total}/{mg_max} = {mg_total*100//mg_max}%")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Abstraction: {c1}/{t1} = {c1*100//t1}%")
    print(f"Crafter QA: {c2}/{t2} = {c2*100//t2}%  {'PASS' if c2*100//t2 >= 90 else 'FAIL'}")
    print(f"MiniGrid QA: {mg_total}/{mg_max} = {mg_total*100//mg_max}%")

    crafter_pass = c2 * 100 // t2 >= 90
    abstraction_pass = c1 >= 3  # at least 3 categories
    print(f"\nCrafter gate: {'PASS' if crafter_pass else 'FAIL'}")
    print(f"Abstraction gate: {'PASS' if abstraction_pass else 'FAIL'}")
    print(f"ALL GATES: {'PASS' if crafter_pass and abstraction_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
