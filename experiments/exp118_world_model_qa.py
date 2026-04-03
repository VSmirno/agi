#!/usr/bin/env python3
"""Exp118: CLS World Model QA + Planning gate.

Gate: QA ≥90% average (4 levels) AND Planning ≥80% (20 scenarios)
"""

import json
import torch
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import (
    generate_synthetic_transitions,
    extract_demo_transitions,
)

COLORS = ["red", "green", "blue", "purple", "yellow", "grey"]
CARRYABLE = ["key", "ball", "box"]


def train_model() -> CLSWorldModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    synth = generate_synthetic_transitions()
    with open("_docs/demo_episodes_bosslevel.json") as f:
        demos = json.load(f)
    demo_trans = extract_demo_transitions(demos)
    all_trans = synth + demo_trans
    print(f"Transitions: {len(all_trans)}")

    model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
    stats = model.train(all_trans)
    print(f"Training: {stats}")
    return model


def level1_object_identity(model: CLSWorldModel) -> tuple[int, int]:
    """25 questions: can you interact with X?"""
    tests = []

    # Pickup tests
    for obj in CARRYABLE:
        tests.append((f"pickup {obj}", model.qa_can_interact(obj, "pickup"), True))
    for obj in ["wall", "door", "goal", "empty"]:
        tests.append((f"pickup {obj}", model.qa_can_interact(obj, "pickup"), False))

    # Toggle tests
    tests.append(("toggle door", model.qa_can_interact("door", "toggle"), True))
    for obj in ["wall", "key", "ball", "box", "empty"]:
        tests.append((f"toggle {obj}", model.qa_can_interact(obj, "toggle"), False))

    # Pass-through tests
    tests.append(("pass open door", model.qa_can_pass("door", "open"), True))
    tests.append(("pass empty", model.qa_can_pass("empty"), True))
    tests.append(("pass closed door", model.qa_can_pass("door", "closed"), False))
    tests.append(("pass locked door", model.qa_can_pass("door", "locked"), False))
    tests.append(("pass wall", model.qa_can_pass("wall"), False))
    for obj in CARRYABLE:
        tests.append((f"pass {obj}", model.qa_can_pass(obj), False))

    # Drop tests
    tests.append(("can drop (carrying)", model.qa_can_interact("empty", "drop"), True))

    correct = 0
    for name, got, expected in tests:
        if got == expected:
            correct += 1
        else:
            print(f"  FAIL L1: {name} — got {got}, expected {expected}")

    return correct, len(tests)


def level2_preconditions(model: CLSWorldModel) -> tuple[int, int]:
    """25 questions: what do you need to do X?"""
    tests = []

    # Locked door preconditions (6 colors)
    for color in COLORS:
        tests.append((
            f"unlock {color} door",
            model.qa_precondition("toggle", "door", color, "locked"),
            f"key_{color}",
        ))

    # Pickup preconditions (3 object types × 3 colors)
    for obj in CARRYABLE:
        for color in ["red", "blue", "green"]:
            tests.append((
                f"pickup {color} {obj}",
                model.qa_precondition("pickup", obj, color),
                "adjacent_and_empty_hands",
            ))

    # Drop precondition
    tests.append(("drop", model.qa_precondition("drop", "", ""), "must_be_carrying"))

    # Cross-color: wrong key should NOT work
    for kc, dc in [("red", "blue"), ("green", "purple"), ("yellow", "grey")]:
        sit = {"facing_obj": "door", "obj_color": dc, "obj_state": "locked",
               "carrying": "key", "carrying_color": kc}
        reward = model.query_reward(sit, "toggle")
        tests.append((f"{kc} key on {dc} door fails", reward < 0, True))

    correct = 0
    for name, got, expected in tests:
        if got == expected:
            correct += 1
        else:
            print(f"  FAIL L2: {name} — got {got}, expected {expected}")

    return correct, len(tests)


def level3_consequences(model: CLSWorldModel) -> tuple[int, int]:
    """25 questions: what happens if you do X?"""
    tests = []

    # Unlock door with matching key (6 colors)
    for color in COLORS:
        r = model.qa_consequence(
            {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
             "carrying": "key", "carrying_color": color}, "toggle")
        tests.append((f"unlock {color}", r.get("result"), "door_unlocked"))

    # Toggle locked door without key
    for color in ["red", "blue", "green"]:
        r = model.qa_consequence(
            {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
             "carrying": "nothing", "carrying_color": ""}, "toggle")
        tests.append((f"toggle locked {color} no key", r.get("result"), "door_still_locked"))

    # Pickup empty hands
    for obj in CARRYABLE:
        r = model.qa_consequence(
            {"facing_obj": obj, "obj_color": "red", "obj_state": "none",
             "carrying": "nothing", "carrying_color": ""}, "pickup")
        tests.append((f"pickup {obj} empty hands", r.get("result"), "picked_up"))

    # Pickup while carrying
    for obj in CARRYABLE:
        r = model.qa_consequence(
            {"facing_obj": obj, "obj_color": "blue", "obj_state": "none",
             "carrying": "ball", "carrying_color": "red"}, "pickup")
        tests.append((f"pickup {obj} while carrying", r.get("result"), "failed_carrying"))

    # Forward into wall
    r = model.qa_consequence(
        {"facing_obj": "wall", "obj_color": "grey", "obj_state": "none",
         "carrying": "nothing", "carrying_color": ""}, "forward")
    tests.append(("forward wall", r.get("result"), "blocked"))

    # Forward into empty
    r = model.qa_consequence(
        {"facing_obj": "empty", "obj_color": "", "obj_state": "none",
         "carrying": "nothing", "carrying_color": ""}, "forward")
    tests.append(("forward empty", r.get("result"), "moved"))

    # Toggle closed door
    r = model.qa_consequence(
        {"facing_obj": "door", "obj_color": "green", "obj_state": "closed",
         "carrying": "nothing", "carrying_color": ""}, "toggle")
    tests.append(("toggle closed door", r.get("result"), "door_opened"))

    correct = 0
    for name, got, expected in tests:
        if got == expected:
            correct += 1
        else:
            print(f"  FAIL L3: {name} — got {got}, expected {expected}")

    return correct, len(tests)


def level4_reasoning(model: CLSWorldModel) -> tuple[int, int]:
    """25 questions: multi-step reasoning."""
    tests = []

    # Open closed door (1 step)
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "closed",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("open closed door", len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Pick up key (1 step)
    plan = model.qa_plan("have_key", {
        "facing_obj": "key", "obj_color": "blue", "obj_state": "none",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("pickup key", len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Pick up ball (1 step)
    plan = model.qa_plan("have_ball", {
        "facing_obj": "ball", "obj_color": "red", "obj_state": "none",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("pickup ball", len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Drop carrying (1 step)
    plan = model.qa_plan("drop", {
        "facing_obj": "empty", "obj_color": "", "obj_state": "none",
        "carrying": "key", "carrying_color": "red"})
    tests.append(("drop key", len(plan) >= 1 and plan[0]["action"] == "drop", True))

    # Unlock door with key (1 step — already have key, facing locked door)
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "locked",
        "carrying": "key", "carrying_color": "red"})
    tests.append(("unlock with key", len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Can't open locked door without key (should fail/empty plan)
    plan = model.qa_plan("open_door", {
        "facing_obj": "door", "obj_color": "red", "obj_state": "locked",
        "carrying": "nothing", "carrying_color": ""})
    tests.append(("can't unlock no key", len(plan) == 0, True))

    # Drop then pickup (2 steps)
    plan = model.qa_plan("have_key", {
        "facing_obj": "key", "obj_color": "blue", "obj_state": "none",
        "carrying": "ball", "carrying_color": "red"})
    actions = [s["action"] for s in plan]
    tests.append(("drop then pickup", "drop" in actions and "pickup" in actions, True))

    # Various 1-step plans for each color
    for color in COLORS[:4]:
        plan = model.qa_plan("have_key", {
            "facing_obj": "key", "obj_color": color, "obj_state": "none",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"pickup {color} key",
                       len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Pick up box
    for color in COLORS[:3]:
        plan = model.qa_plan("have_box", {
            "facing_obj": "box", "obj_color": color, "obj_state": "none",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"pickup {color} box",
                       len(plan) >= 1 and plan[0]["action"] == "pickup", True))

    # Open doors of different colors
    for color in COLORS[:3]:
        plan = model.qa_plan("open_door", {
            "facing_obj": "door", "obj_color": color, "obj_state": "closed",
            "carrying": "nothing", "carrying_color": ""})
        tests.append((f"open {color} closed door",
                       len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    # Unlock with matching key (different colors)
    for color in COLORS[:3]:
        plan = model.qa_plan("open_door", {
            "facing_obj": "door", "obj_color": color, "obj_state": "locked",
            "carrying": "key", "carrying_color": color})
        tests.append((f"unlock {color} door",
                       len(plan) >= 1 and plan[0]["action"] == "toggle", True))

    correct = 0
    for name, got, expected in tests:
        if got == expected:
            correct += 1
        else:
            print(f"  FAIL L4: {name} — got {got}, expected {expected}")

    return correct, len(tests)


def main():
    print("=" * 60)
    print("EXP118: CLS World Model QA + Planning Gate")
    print("=" * 60)

    model = train_model()

    print("\n" + "=" * 60)
    c1, t1 = level1_object_identity(model)
    print(f"Level 1 (Object Identity): {c1}/{t1} = {c1*100//t1}%  [gate ≥95%]")

    print()
    c2, t2 = level2_preconditions(model)
    print(f"Level 2 (Preconditions): {c2}/{t2} = {c2*100//t2}%  [gate ≥90%]")

    print()
    c3, t3 = level3_consequences(model)
    print(f"Level 3 (Consequences): {c3}/{t3} = {c3*100//t3}%  [gate ≥85%]")

    print()
    c4, t4 = level4_reasoning(model)
    print(f"Level 4 (Reasoning/Planning): {c4}/{t4} = {c4*100//t4}%  [gate ≥80%]")

    total_c = c1 + c2 + c3 + c4
    total_t = t1 + t2 + t3 + t4
    avg = total_c * 100 / total_t

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Level 1: {c1}/{t1} = {c1*100//t1}%  {'PASS' if c1*100//t1 >= 95 else 'FAIL'}")
    print(f"Level 2: {c2}/{t2} = {c2*100//t2}%  {'PASS' if c2*100//t2 >= 90 else 'FAIL'}")
    print(f"Level 3: {c3}/{t3} = {c3*100//t3}%  {'PASS' if c3*100//t3 >= 85 else 'FAIL'}")
    print(f"Level 4: {c4}/{t4} = {c4*100//t4}%  {'PASS' if c4*100//t4 >= 80 else 'FAIL'}")
    print(f"Average: {avg:.0f}%  {'PASS' if avg >= 90 else 'FAIL'}")
    print(f"Total: {total_c}/{total_t}")

    all_pass = (c1*100//t1 >= 95 and c2*100//t2 >= 90 and
                c3*100//t3 >= 85 and c4*100//t4 >= 80 and avg >= 90)
    print(f"\nALL GATES: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
