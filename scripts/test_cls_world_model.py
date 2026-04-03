#!/usr/bin/env python3
"""Test CLSWorldModel on GPU."""
import json
import torch
from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import (
    generate_synthetic_transitions,
    extract_demo_transitions,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

synth = generate_synthetic_transitions()
with open("_docs/demo_episodes_bosslevel.json") as f:
    demos = json.load(f)
demo_trans = extract_demo_transitions(demos)
all_trans = synth + demo_trans
print(f"Transitions: {len(synth)} synth + {len(demo_trans)} demo = {len(all_trans)}")

model = CLSWorldModel(dim=2048, n_locations=5000, device=device)
print("Training...")
stats = model.train(all_trans)
print(f"Stats: {stats}")

print("\n=== Level 1: Object Identity ===")
tests = [
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
ok1 = 0
for name, got, expected in tests:
    match = got == expected
    if match:
        ok1 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 1: {ok1}/{len(tests)} = {ok1*100//len(tests)}%")

print("\n=== Level 2: Preconditions ===")
tests2 = [
    ("locked red door", model.qa_precondition("toggle", "door", "red", "locked"), "key_red"),
    ("locked blue door", model.qa_precondition("toggle", "door", "blue", "locked"), "key_blue"),
    ("locked green door", model.qa_precondition("toggle", "door", "green", "locked"), "key_green"),
    ("locked purple door", model.qa_precondition("toggle", "door", "purple", "locked"), "key_purple"),
    ("pickup ball", model.qa_precondition("pickup", "ball", "red"), "adjacent_and_empty_hands"),
    ("pickup key", model.qa_precondition("pickup", "key", "blue"), "adjacent_and_empty_hands"),
    ("pickup box", model.qa_precondition("pickup", "box", "green"), "adjacent_and_empty_hands"),
]
ok2 = 0
for name, got, expected in tests2:
    match = got == expected
    if match:
        ok2 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 2: {ok2}/{len(tests2)} = {ok2*100//len(tests2)}%")

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
r6 = model.qa_consequence(
    {"facing_obj": "empty", "obj_color": "", "obj_state": "none",
     "carrying": "nothing", "carrying_color": ""}, "forward")
r7 = model.qa_consequence(
    {"facing_obj": "door", "obj_color": "green", "obj_state": "closed",
     "carrying": "nothing", "carrying_color": ""}, "toggle")

tests3 = [
    ("unlock red door with red key", r1.get("result"), "door_unlocked"),
    ("toggle locked door no key", r2.get("result"), "door_still_locked"),
    ("pickup key empty hands", r3.get("result"), "picked_up"),
    ("pickup key while carrying", r4.get("result"), "failed_carrying"),
    ("forward into wall", r5.get("result"), "blocked"),
    ("forward into empty", r6.get("result"), "moved"),
    ("toggle closed door", r7.get("result"), "door_opened"),
]
ok3 = 0
for name, got, expected in tests3:
    match = got == expected
    if match:
        ok3 += 1
    else:
        print(f"  FAIL: {name} — got {got}, expected {expected}")
print(f"Level 3: {ok3}/{len(tests3)} = {ok3*100//len(tests3)}%")

# Level 4: Planning
print("\n=== Level 4: Planning ===")
plan1 = model.qa_plan("open_door", {
    "facing_obj": "door", "obj_color": "red", "obj_state": "closed",
    "carrying": "nothing", "carrying_color": "",
})
print(f"Plan: open closed door = {[s['action'] for s in plan1]}")

plan2 = model.qa_plan("have_key", {
    "facing_obj": "key", "obj_color": "blue", "obj_state": "none",
    "carrying": "nothing", "carrying_color": "",
})
print(f"Plan: pick up key = {[s['action'] for s in plan2]}")

# Generalization: unseen color
print("\n=== Generalization (unseen colors) ===")
# Train only had red/green/blue/purple/yellow/grey
# Test with exact same colors but via hippocampus path
for color in ["red", "blue", "purple"]:
    outcome, conf, source = model.query(
        {"facing_obj": "door", "obj_color": color, "obj_state": "locked",
         "carrying": "key", "carrying_color": color}, "toggle")
    print(f"  {color} key + {color} door: {outcome.get('result')} (source={source}, conf={conf:.3f})")

print(f"\n=== SUMMARY ===")
print(f"L1: {ok1}/{len(tests)} = {ok1*100//len(tests)}%")
print(f"L2: {ok2}/{len(tests2)} = {ok2*100//len(tests2)}%")
print(f"L3: {ok3}/{len(tests3)} = {ok3*100//len(tests3)}%")
total = ok1 + ok2 + ok3
total_max = len(tests) + len(tests2) + len(tests3)
print(f"Overall: {total}/{total_max} = {total*100//total_max}%")
