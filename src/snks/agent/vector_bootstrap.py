"""Stage 83: Bootstrap VectorWorldModel from crafter_textbook.yaml.

Parses the existing YAML textbook and writes seed associations into
the VectorWorldModel's SDM. Each action-triggered rule becomes one
SDM write. Passive rules (body_rate, spatial damage) also seeded.

This is the "teacher" handing rough priors to the agent before it
enters the environment. After bootstrap, the agent refines through
experience.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from snks.agent.vector_world_model import VectorWorldModel


def load_from_textbook(model: "VectorWorldModel", yaml_path: str | Path) -> dict:
    """Parse crafter_textbook.yaml and seed VectorWorldModel.

    Creates concept, action, and role vectors for everything declared
    in the textbook. Writes seed associations into SDM.

    Returns summary dict with counts for diagnostics.
    """
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    stats = {
        "concepts": 0,
        "actions": 0,
        "action_rules": 0,
        "passive_rules": 0,
    }

    # --- Vocabulary: create concept vectors ---
    for entry in data.get("vocabulary", []):
        cid = entry["id"]
        model._ensure_concept(cid)
        stats["concepts"] += 1

    # --- Body variables: create role vectors ---
    for var_entry in data.get("body", {}).get("variables", []):
        model._ensure_role(var_entry["name"])

    # --- Primitives: create action vectors ---
    for prim_name in data.get("primitives", {}):
        model._ensure_action(prim_name)
        stats["actions"] += 1

    # Standard actions
    for action_name in ["do", "make", "place", "sleep"]:
        model._ensure_action(action_name)
    model._ensure_action("proximity")  # for spatial damage associations
    stats["actions"] += 5

    # --- Action-triggered rules: seed SDM ---
    for rule in data.get("rules", []):
        if "action" not in rule:
            continue

        action = rule["action"]
        target = rule.get("target") or rule.get("result") or rule.get("item", "")
        if not target:
            continue

        # Build effect dict from rule
        effect: dict[str, int] = {}
        rule_effect = rule.get("effect", {})

        # Inventory deltas
        for item, delta in rule_effect.get("inventory", {}).items():
            effect[item] = int(delta)

        # Body deltas
        for var, delta in rule_effect.get("body", {}).items():
            effect[var] = int(delta)

        if not effect:
            continue

        # Write seed association multiple times for confidence
        model._ensure_concept(target)
        model._ensure_action(action)
        for role_name in effect:
            model._ensure_role(role_name)

        for _ in range(5):
            model.learn(target, action, effect)

        stats["action_rules"] += 1

    # --- Passive spatial rules: seed as proximity associations ---
    for rule in data.get("rules", []):
        if rule.get("passive") != "spatial":
            continue

        entity = rule.get("entity", "")
        rule_effect = rule.get("effect", {})
        body_deltas = rule_effect.get("body", {})

        effect = {}
        for var, delta in body_deltas.items():
            # Scale rough prior to integer-ish range for thermometer encoding
            # -0.5 per tick is "significant damage"
            effect[var] = int(delta * 10)  # -0.5 → -5

        if effect and entity:
            model._ensure_concept(entity)
            for _ in range(5):
                model.learn(entity, "proximity", effect)
            stats["passive_rules"] += 1

    # --- Passive body_rate: seed as background rates ---
    # These are per-tick rates — small values. We encode them as
    # associations with a special "tick" action.
    model._ensure_action("tick")
    for rule in data.get("rules", []):
        if rule.get("passive") != "body_rate":
            continue

        variable = rule.get("variable", "")
        rate = rule.get("rate", 0)

        if variable and rate != 0:
            # Scale: -0.02 → -1 (very rough, thermometer granularity)
            scaled = max(-10, min(10, int(rate * 50)))
            if scaled != 0:
                effect = {variable: scaled}
                model._ensure_role(variable)
                for _ in range(3):
                    model.learn("background", "tick", effect)
                stats["passive_rules"] += 1

    return stats
