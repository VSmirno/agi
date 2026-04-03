#!/usr/bin/env python3
"""Quick test of UnifiedWorldModel on GPU."""
import json
from snks.agent.unified_world_model import UnifiedWorldModel
from snks.agent.world_model_trainer import (
    generate_synthetic_transitions,
    extract_demo_transitions,
)

synth = generate_synthetic_transitions()
with open("_docs/demo_episodes_bosslevel.json") as f:
    demos = json.load(f)
demo_trans = extract_demo_transitions(demos)
all_trans = synth + demo_trans
print(f"Transitions: {len(synth)} synth + {len(demo_trans)} demo = {len(all_trans)}")

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = UnifiedWorldModel(dim=1024, n_locations=10000, device=device)
print("Training...")
model.train(all_trans)
print(f"Trained: {model.n_trained}, SDM writes: {model.sdm.n_writes}")

print("\n=== Level 1: Object Identity ===")
tests_l1 = [
    ("pickup key", model.qa_can_interact("key", "pickup"), True),
    ("pickup ball", model.qa_can_interact("ball", "pickup"), True),
    ("pickup box", model.qa_can_interact("box", "pickup"), True),
    ("pickup wall", model.qa_can_interact("wall", "pickup"), False),
    ("pickup door", model.qa_can_interact("door", "pickup"), False),
    ("toggle door", model.qa_can_interact("door", "toggle"), True),
    ("toggle wall", model.qa_can_interact("wall", "toggle"), False),
    ("toggle key", model.qa_can_interact("key", "toggle"), False),
    ("pass open door", model.qa_can_pass("door", "open"), True),
    ("pass locked door", model.qa_can_pass("door", "locked"), False),
    ("pass closed door", model.qa_can_pass("door", "closed"), False),
    ("pass wall", model.qa_can_pass("wall"), False),
    ("pass empty", model.qa_can_pass("empty"), True),
]
correct_l1 = 0
for name, got, expected in tests_l1:
    ok = got == expected
    if ok:
        correct_l1 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 1: {correct_l1}/{len(tests_l1)} = {correct_l1*100//len(tests_l1)}%")

print("\n=== Level 2: Preconditions ===")
tests_l2 = [
    ("open locked red door", model.qa_precondition("toggle", "door", "red", "locked"), "key_red"),
    ("open locked blue door", model.qa_precondition("toggle", "door", "blue", "locked"), "key_blue"),
    ("open locked green door", model.qa_precondition("toggle", "door", "green", "locked"), "key_green"),
    ("pickup ball", model.qa_precondition("pickup", "ball", "red"), "adjacent_and_empty_hands"),
    ("pickup key", model.qa_precondition("pickup", "key", "blue"), "adjacent_and_empty_hands"),
]
correct_l2 = 0
for name, got, expected in tests_l2:
    ok = got == expected
    if ok:
        correct_l2 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 2: {correct_l2}/{len(tests_l2)} = {correct_l2*100//len(tests_l2)}%")

print("\n=== Level 3: Consequences ===")
r1 = model.qa_consequence(
    {"facing_obj": "door", "obj_color": "red", "obj_state": "locked",
     "carrying": "key", "carrying_color": "red"}, "toggle")
r2 = model.qa_consequence(
    {"facing_obj": "door", "obj_color": "red", "obj_state": "locked",
     "carrying": "nothing", "carrying_color": ""}, "toggle")
r3 = model.qa_consequence(
    {"facing_obj": "key", "obj_color": "blue", "obj_state": "none",
     "carrying": "nothing", "carrying_color": ""}, "pickup")
r4 = model.qa_consequence(
    {"facing_obj": "key", "obj_color": "blue", "obj_state": "none",
     "carrying": "ball", "carrying_color": "red"}, "pickup")
r5 = model.qa_consequence(
    {"facing_obj": "wall", "obj_color": "grey", "obj_state": "none",
     "carrying": "nothing", "carrying_color": ""}, "forward")

tests_l3 = [
    ("unlock red door with red key", r1.get("result"), "door_unlocked"),
    ("toggle locked door no key", r2.get("result"), "door_still_locked"),
    ("pickup key empty hands", r3.get("result"), "picked_up"),
    ("pickup key while carrying", r4.get("result"), "failed_carrying"),
    ("forward into wall", r5.get("result"), "blocked"),
]
correct_l3 = 0
for name, got, expected in tests_l3:
    ok = got == expected
    if ok:
        correct_l3 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 3: {correct_l3}/{len(tests_l3)} = {correct_l3*100//len(tests_l3)}%")

print(f"\n=== SUMMARY ===")
print(f"L1: {correct_l1}/{len(tests_l1)}")
print(f"L2: {correct_l2}/{len(tests_l2)}")
print(f"L3: {correct_l3}/{len(tests_l3)}")
