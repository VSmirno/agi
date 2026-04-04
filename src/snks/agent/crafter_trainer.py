"""Stage 63: Crafter transition generator.

Generates state transitions from Crafter tech tree for world model training.
Crafter has a crafting dependency graph — we encode it as transitions.

Crafter tech tree:
- collect_wood: do action near tree → +wood
- place_table: requires wood ≥ 1
- make_wood_pickaxe: requires wood ≥ 1, near table
- collect_stone: do action near stone, requires wood_pickaxe
- place_furnace: requires stone ≥ 1
- make_stone_pickaxe: requires wood ≥ 1, stone ≥ 1, near table
- collect_coal: do near coal, requires wood_pickaxe
- collect_iron: do near iron, requires stone_pickaxe
- make_iron_pickaxe: requires wood ≥ 1, iron ≥ 1, near table
"""

from __future__ import annotations

from snks.agent.world_model_trainer import Transition


# Crafter tech tree as explicit rules
CRAFTER_RULES = [
    # Collecting resources
    {"action": "do", "near": "tree", "requires": {}, "gives": "wood",
     "result": "collected"},
    {"action": "do", "near": "stone", "requires": {"wood_pickaxe": 1}, "gives": "stone",
     "result": "collected"},
    {"action": "do", "near": "coal", "requires": {"wood_pickaxe": 1}, "gives": "coal",
     "result": "collected"},
    {"action": "do", "near": "iron", "requires": {"stone_pickaxe": 1}, "gives": "iron",
     "result": "collected"},
    {"action": "do", "near": "diamond", "requires": {"iron_pickaxe": 1}, "gives": "diamond",
     "result": "collected"},
    {"action": "do", "near": "water", "requires": {}, "gives": "drink",
     "result": "collected"},
    {"action": "do", "near": "cow", "requires": {"wood_sword": 1}, "gives": "food",
     "result": "collected"},

    # Placing structures
    {"action": "place_table", "near": "empty", "requires": {"wood": 2},
     "gives": "table", "result": "placed"},
    {"action": "place_furnace", "near": "empty", "requires": {"stone": 4},
     "gives": "furnace", "result": "placed"},
    {"action": "place_plant", "near": "empty", "requires": {"sapling": 1},
     "gives": "plant", "result": "placed"},
    {"action": "place_stone", "near": "empty", "requires": {"stone": 1},
     "gives": "stone_wall", "result": "placed"},

    # Crafting tools (requires near table)
    {"action": "make_wood_pickaxe", "near": "table", "requires": {"wood": 1},
     "gives": "wood_pickaxe", "result": "crafted"},
    {"action": "make_stone_pickaxe", "near": "table",
     "requires": {"wood": 1, "stone": 1}, "gives": "stone_pickaxe",
     "result": "crafted"},
    {"action": "make_iron_pickaxe", "near": "table",
     "requires": {"wood": 1, "iron": 1}, "gives": "iron_pickaxe",
     "result": "crafted"},
    {"action": "make_wood_sword", "near": "table", "requires": {"wood": 1},
     "gives": "wood_sword", "result": "crafted"},
    {"action": "make_stone_sword", "near": "table",
     "requires": {"wood": 1, "stone": 1}, "gives": "stone_sword",
     "result": "crafted"},
    {"action": "make_iron_sword", "near": "table",
     "requires": {"wood": 1, "iron": 1}, "gives": "iron_sword",
     "result": "crafted"},
]

# Failed attempts
CRAFTER_FAILURES = [
    # Can't collect without tool
    {"action": "do", "near": "stone", "requires_missing": "wood_pickaxe",
     "result": "failed_no_tool"},
    {"action": "do", "near": "iron", "requires_missing": "stone_pickaxe",
     "result": "failed_no_tool"},
    {"action": "do", "near": "coal", "requires_missing": "wood_pickaxe",
     "result": "failed_no_tool"},
    {"action": "do", "near": "diamond", "requires_missing": "iron_pickaxe",
     "result": "failed_no_tool"},

    # Can't craft without resources
    {"action": "make_wood_pickaxe", "near": "table",
     "requires_missing": "wood", "result": "failed_no_resource"},
    {"action": "make_stone_pickaxe", "near": "table",
     "requires_missing": "stone", "result": "failed_no_resource"},
    {"action": "make_iron_pickaxe", "near": "table",
     "requires_missing": "iron", "result": "failed_no_resource"},

    # Can't craft without table
    {"action": "make_wood_pickaxe", "near": "empty",
     "requires_missing": "table", "result": "failed_no_station"},
    {"action": "make_stone_pickaxe", "near": "empty",
     "requires_missing": "table", "result": "failed_no_station"},
]


def generate_crafter_transitions() -> list[Transition]:
    """Generate transitions from Crafter tech tree rules."""
    transitions = []

    # Successful actions
    for rule in CRAFTER_RULES:
        # Build situation
        situation = {
            "domain": "crafter",
            "near": rule["near"],
            "action": rule["action"],
        }
        # Add required items to situation
        for item, count in rule.get("requires", {}).items():
            situation[f"has_{item}"] = str(count)

        outcome = {
            "result": rule["result"],
            "gives": rule["gives"],
        }

        transitions.append(Transition(
            situation=situation,
            action=rule["action"],
            outcome=outcome,
            reward=1.0,
        ))

    # Failed actions
    for fail in CRAFTER_FAILURES:
        situation = {
            "domain": "crafter",
            "near": fail["near"],
            "action": fail["action"],
            "missing": fail["requires_missing"],
        }
        outcome = {"result": fail["result"]}

        transitions.append(Transition(
            situation=situation,
            action=fail["action"],
            outcome=outcome,
            reward=-1.0,
        ))

    return transitions
