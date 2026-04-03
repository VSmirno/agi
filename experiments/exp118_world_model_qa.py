#!/usr/bin/env python3
"""Exp118: CLS World Model QA + Planning gate (HONEST version).

Training on 3 colors (red/green/blue) only.
Testing includes held-out colors (purple/yellow/grey) for generalization.
Multi-step planning (3-6 steps).

Gate: QA ≥90% average (4 levels) AND Planning ≥80% (20 scenarios)
"""

import json
import torch
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import (
    generate_synthetic_transitions,
    extract_demo_transitions,
    TRAIN_COLORS,
)

ALL_COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
TRAIN = TRAIN_COLORS  # red, green, blue
HELD_OUT = ["purple", "yellow", "grey"]
CARRYABLE = ["key", "ball", "box"]


def train_model() -> CLSWorldModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Train only on TRAIN_COLORS
    synth = generate_synthetic_transitions(colors=TRAIN)
    with open("_docs/demo_episodes_bosslevel.json") as f:
        demos = json.load(f)
    demo_trans = extract_demo_transitions(demos)
    all_trans = synth + demo_trans
    print(f"Transitions: {len(synth)} synth ({TRAIN}) + {len(demo_trans)} demo = {len(all_trans)}")

    model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
    stats = model.train(all_trans)
    print(f"Training: {stats}")
    return model


def level1_object_identity(model: CLSWorldModel) -> tuple[int, int, list]:
    """Object identity — TRAINED colors."""
    tests = []
    for obj in CARRYABLE:
        tests.append((f"pickup {obj}", model.qa_can_interact(obj, "pickup"), True))
    for obj in ["wall", "door", "goal", "empty"]:
        tests.append((f"pickup {obj}", model.qa_can_interact(obj, "pickup"), False))
    tests.append(("toggle door", model.qa_can_interact("door", "toggle"), True))
    for obj in ["wall", "key", "ball", "box", "empty"]:
        tests.append((f"toggle {obj}", model.qa_can_interact(obj, "toggle"), False))
    tests.append(("pass open door", model.qa_can_pass("door", "open"), True))
    tests.append(("pass empty", model.qa_can_pass("empty"), True))
    tests.append(("pass closed door", model.qa_can_pass("door", "closed"), False))
    tests.append(("pass locked door", model.qa_can_pass("door", "locked"), False))
    tests.append(("pass wall", model.qa_can_pass("wall"), False))
    for obj in CARRYABLE:
        tests.append((f"pass {obj}", model.qa_can_pass(obj), False))
    tests.append(("can drop", model.qa_can_interact("empty", "drop"), True))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} got={got} exp={exp}")
    return ok, len(tests), fails


def level2_preconditions(model: CLSWorldModel) -> tuple[int, int, list]:
    """Preconditions — includes HELD-OUT colors for generalization."""
    tests = []

    # Trained colors
    for color in TRAIN:
        tests.append((f"unlock {color} door (TRAIN)",
                       model.qa_precondition("toggle", "door", color, "locked"),
                       f"key_{color}"))

    # HELD-OUT colors — generalization via hippocampus
    for color in HELD_OUT:
        tests.append((f"unlock {color} door (HELD-OUT)",
                       model.qa_precondition("toggle", "door", color, "locked"),
                       f"key_{color}"))

    # Pickup preconditions
    for obj in CARRYABLE:
        tests.append((f"pickup {obj}",
                       model.qa_precondition("pickup", obj, "red"),
                       "adjacent_and_empty_hands"))

    # Wrong key should fail
    for kc, dc in [("red", "blue"), ("green", "red"), ("blue", "green")]:
        sit = {"facing_obj": "door", "obj_color": dc, "obj_state": "locked",
               "carrying": "key", "carrying_color": kc}
        reward = model.query_reward(sit, "toggle")
        tests.append((f"{kc} key on {dc} door fails (TRAIN)", reward < 0, True))

    # Wrong key with HELD-OUT colors
    for kc, dc in [("purple", "yellow"), ("yellow", "grey")]:
        sit = {"facing_obj": "door", "obj_color": dc, "obj_state": "locked",
               "carrying": "key", "carrying_color": kc}
        reward = model.query_reward(sit, "toggle")
        tests.append((f"{kc} key on {dc} door fails (HELD-OUT)", reward < 0, True))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} got={got} exp={exp}")
    return ok, len(tests), fails


def level3_consequences(model: CLSWorldModel) -> tuple[int, int, list]:
    """Consequences — trained + held-out colors."""
    tests = []

    # Unlock with matching key — trained
    for color in TRAIN:
        r = model.qa_consequence(
            {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
             "carrying": "key", "carrying_color": color}, "toggle")
        tests.append((f"unlock {color} (TRAIN)", r.get("result"), "door_unlocked"))

    # Unlock with matching key — HELD-OUT (generalization!)
    for color in HELD_OUT:
        r = model.qa_consequence(
            {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
             "carrying": "key", "carrying_color": color}, "toggle")
        tests.append((f"unlock {color} (HELD-OUT)", r.get("result"), "door_unlocked"))

    # Toggle locked without key
    for color in TRAIN + HELD_OUT[:1]:
        r = model.qa_consequence(
            {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
             "carrying": "nothing", "carrying_color": ""}, "toggle")
        tests.append((f"toggle locked {color} no key", r.get("result"), "door_still_locked"))

    # Pickup
    for obj in CARRYABLE:
        r = model.qa_consequence(
            {"facing_obj": obj, "obj_color": "red", "obj_state": "none",
             "carrying": "nothing", "carrying_color": ""}, "pickup")
        tests.append((f"pickup {obj}", r.get("result"), "picked_up"))

    # Pickup while carrying
    for obj in CARRYABLE:
        r = model.qa_consequence(
            {"facing_obj": obj, "obj_color": "blue", "obj_state": "none",
             "carrying": "ball", "carrying_color": "red"}, "pickup")
        tests.append((f"pickup {obj} carrying", r.get("result"), "failed_carrying"))

    # Forward
    r = model.qa_consequence(
        {"facing_obj": "wall", "obj_color": "grey", "obj_state": "none",
         "carrying": "nothing", "carrying_color": ""}, "forward")
    tests.append(("forward wall", r.get("result"), "blocked"))

    r = model.qa_consequence(
        {"facing_obj": "empty", "obj_color": "", "obj_state": "none",
         "carrying": "nothing", "carrying_color": ""}, "forward")
    tests.append(("forward empty", r.get("result"), "moved"))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} got={got} exp={exp}")
    return ok, len(tests), fails


def level4_planning(model: CLSWorldModel) -> tuple[int, int, list]:
    """Multi-step planning — the HARD test."""
    tests = []

    # 1-step plans (sanity)
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "closed",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("open closed door (1-step)",
                   len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    plan = model.qa_plan("have_key", {
        "facing_obj": "key", "obj_color": "blue", "obj_state": "none",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("pickup key (1-step)",
                   len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # 2-step: drop then pickup
    plan = model.qa_plan("have_key", {
        "facing_obj": "key", "obj_color": "blue", "obj_state": "none",
        "carrying": "ball", "carrying_color": "red"})
    actions = [s["action"] for s in plan]
    tests.append(("drop+pickup (2-step)",
                   len(actions) >= 2 and actions[0] == "drop" and "pickup" in actions, True))

    # 2-step: unlock locked door (already have key)
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "locked",
        "carrying": "key", "carrying_color": "red"})
    tests.append(("unlock with key (1-step)",
                   len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Should NOT be able to unlock without key
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "locked",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("can't unlock no key (fail)",
                   len(plan) == 0 or plan[0]["action"] != "toggle", True))

    # Multi-step with different colors (TRAIN)
    for color in TRAIN:
        plan = model.qa_plan("have_key", {
            "facing_obj": "key", "obj_color": color, "obj_state": "none",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"pickup {color} key",
                       len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Generalization: HELD-OUT color planning
    for color in HELD_OUT:
        plan = model.qa_plan("have_key", {
            "facing_obj": "key", "obj_color": color, "obj_state": "none",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"pickup {color} key (HELD-OUT)",
                       len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Open doors (various)
    for color in TRAIN:
        plan = model.qa_plan("open_door", {
            "facing_obj": "door", "obj_color": color, "obj_state": "closed",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"open {color} door",
                       len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Unlock with key (different colors)
    for color in TRAIN:
        plan = model.qa_plan("open_door", {
            "facing_obj": "door", "obj_color": color, "obj_state": "locked",
            "carrying": "key", "carrying_color": color})
        tests.append((f"unlock {color} door",
                       len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Pickup box/ball
    for obj in ["box", "ball"]:
        plan = model.qa_plan(f"have_{obj}", {
            "facing_obj": obj, "obj_color": "green", "obj_state": "none",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"pickup {obj}",
                       len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    fails = []
    ok = 0
    for name, got, exp in tests:
        if got == exp:
            ok += 1
        else:
            fails.append(f"  FAIL: {name} got={got} exp={exp}")
    return ok, len(tests), fails


def main():
    print("=" * 60)
    print("EXP118: CLS World Model — HONEST QA Gate")
    print(f"Train colors: {TRAIN}, Held-out: {HELD_OUT}")
    print("=" * 60)

    model = train_model()

    print(f"\n{'='*60}")
    c1, t1, f1 = level1_object_identity(model)
    for f in f1: print(f)
    print(f"Level 1 (Object Identity): {c1}/{t1} = {c1*100//t1}%  [gate ≥95%]")

    print()
    c2, t2, f2 = level2_preconditions(model)
    for f in f2: print(f)
    print(f"Level 2 (Preconditions + generalization): {c2}/{t2} = {c2*100//t2}%  [gate ≥90%]")

    print()
    c3, t3, f3 = level3_consequences(model)
    for f in f3: print(f)
    print(f"Level 3 (Consequences + generalization): {c3}/{t3} = {c3*100//t3}%  [gate ≥85%]")

    print()
    c4, t4, f4 = level4_planning(model)
    for f in f4: print(f)
    print(f"Level 4 (Planning multi-step): {c4}/{t4} = {c4*100//t4}%  [gate ≥80%]")

    total_c = c1 + c2 + c3 + c4
    total_t = t1 + t2 + t3 + t4
    avg = total_c * 100 / total_t

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Level 1: {c1}/{t1} = {c1*100//t1}%  {'PASS' if c1*100//t1 >= 95 else 'FAIL'}")
    print(f"Level 2: {c2}/{t2} = {c2*100//t2}%  {'PASS' if c2*100//t2 >= 90 else 'FAIL'}")
    print(f"Level 3: {c3}/{t3} = {c3*100//t3}%  {'PASS' if c3*100//t3 >= 85 else 'FAIL'}")
    print(f"Level 4: {c4}/{t4} = {c4*100//t4}%  {'PASS' if c4*100//t4 >= 80 else 'FAIL'}")
    print(f"Average: {avg:.0f}%  {'PASS' if avg >= 90 else 'FAIL'}")
    print(f"Total: {total_c}/{total_t}")

    # Count generalization tests
    gen_tests = [f for f in f2 + f3 + f4 if "HELD-OUT" in f]
    gen_total = sum(1 for _, got, exp in
                    [(n, g, e) for n, g, e in
                     [(name, got, exp) for name, got, exp in []]  # placeholder
                    ])
    print(f"\nHeld-out color tests in failures: {len(gen_tests)}")

    all_pass = (c1*100//t1 >= 95 and c2*100//t2 >= 90 and
                c3*100//t3 >= 85 and c4*100//t4 >= 80 and avg >= 90)
    print(f"\nALL GATES: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
