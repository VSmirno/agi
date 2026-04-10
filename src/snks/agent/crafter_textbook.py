"""Crafter textbook loader: structured YAML + legacy regex fallback.

Stage 77a rewrite: each rule in YAML is a dict (structured) rather than a
string (parsed by regex). Legacy string rules still work for backward
compatibility with Stage 71-76 textbooks; they're parsed by the regex
codepath and converted to equivalent structured CausalLinks.

New format (preferred):
    rules:
      - action: do
        target: tree
        effect: { inventory: { wood: +1 } }
      - passive: movement
        entity: zombie
        behavior: chase_player

Legacy format (supported, deprecated):
    rules:
      - "do tree gives wood"
      - "zombie nearby without wood_sword means flee"

Design: docs/superpowers/specs/2026-04-10-stage77a-conceptstore-forward-sim-design.md
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from snks.agent.concept_store import CausalLink, ConceptStore
from snks.agent.forward_sim_types import RuleEffect, StatefulCondition


# ---------------------------------------------------------------------------
# New structured dict parser (Stage 77a primary path)
# ---------------------------------------------------------------------------


def _parse_rule_dict(entry: dict) -> tuple[str, CausalLink] | None:
    """Parse a structured YAML rule dict into (concept_id, CausalLink).

    Returns None if entry is malformed (caller skips it).
    """
    if "action" in entry:
        return _parse_action_rule(entry)
    if "passive" in entry:
        return _parse_passive_rule(entry)
    return None


def _parse_action_rule(entry: dict) -> tuple[str, CausalLink] | None:
    """Parse an action-triggered rule (do / make / place / sleep).

    Returns (concept_id, link) where concept_id is the concept on which
    the action operates:
      - do X   → X
      - place X near Y → Y (the surface)
      - make X near Y  → Y (the crafting station)
      - sleep          → "_self"
    """
    action = entry["action"]

    if action == "do":
        concept_id = entry.get("target")
    elif action == "place":
        # For `place`: `near` is the concept you're standing next to (surface)
        concept_id = entry.get("near")
    elif action == "make":
        concept_id = entry.get("near")
    elif action == "sleep":
        concept_id = "_self"
    else:
        return None

    if not concept_id:
        return None

    effect_dict = entry.get("effect", {}) or {}
    effect = _build_action_effect(action, entry, effect_dict)
    if effect is None:
        return None

    requires = dict(entry.get("requires", {}) or {})
    legacy_result = _derive_legacy_result(action, entry, effect)

    link = CausalLink(
        action=action,
        result=legacy_result,
        requires=requires,
        confidence=0.5,
        kind="action_triggered",
        effect=effect,
        concept=concept_id,
    )
    return (concept_id, link)


def _build_action_effect(
    action: str, entry: dict, effect_dict: dict
) -> RuleEffect | None:
    """Build a RuleEffect for an action-triggered rule.

    The effect.kind is determined by the content of effect_dict plus the
    action type, choosing the most specific fitting category.
    """
    inventory_delta = {k: int(v) for k, v in effect_dict.get("inventory", {}).items()}
    body_delta = {k: float(v) for k, v in effect_dict.get("body", {}).items()}
    scene_remove = effect_dict.get("remove_entity")
    world_place_raw = effect_dict.get("world_place")

    # Normalize world_place to (item, where) tuple
    world_place: tuple[str, str] | None = None
    if world_place_raw:
        if isinstance(world_place_raw, dict):
            world_place = (world_place_raw["item"], world_place_raw["where"])
        elif isinstance(world_place_raw, (list, tuple)) and len(world_place_raw) == 2:
            world_place = (str(world_place_raw[0]), str(world_place_raw[1]))

    # Determine effect kind — pick most specific
    if scene_remove:
        kind = "remove"
    elif world_place:
        kind = "place"
    elif action == "make":
        kind = "craft"
    elif action == "sleep":
        kind = "self"
    elif action == "do" and body_delta and not inventory_delta:
        kind = "consume"
    elif action == "do" and inventory_delta:
        kind = "gather"
    else:
        return None  # empty / unparseable effect

    return RuleEffect(
        kind=kind,
        inventory_delta=inventory_delta,
        body_delta=body_delta,
        scene_remove=scene_remove,
        world_place=world_place,
    )


def _derive_legacy_result(
    action: str, entry: dict, effect: RuleEffect
) -> str:
    """Derive the legacy `result: str` from the new effect structure.

    Stage 71-76 code reads `link.result` for plan lookup, verification, and
    tests. This keeps those callers working during the staged refactor until
    Commit 8 removes the `result` field entirely.
    """
    if effect.kind == "gather":
        positives = [k for k, v in effect.inventory_delta.items() if v > 0]
        return positives[0] if positives else ""
    if effect.kind == "craft":
        # Prefer explicit `result` field in YAML; fall back to largest positive delta
        explicit = entry.get("result")
        if explicit:
            return str(explicit)
        positives = [k for k, v in effect.inventory_delta.items() if v > 0]
        return positives[0] if positives else ""
    if effect.kind == "place":
        return str(entry.get("item", effect.world_place[0] if effect.world_place else ""))
    if effect.kind in ("consume", "self"):
        positives = [k for k, v in effect.body_delta.items() if v > 0]
        return f"restore_{positives[0]}" if positives else ""
    if effect.kind == "remove":
        return f"kill_{effect.scene_remove}" if effect.scene_remove else ""
    return ""


def _parse_passive_rule(entry: dict) -> tuple[str, CausalLink] | None:
    """Parse a passive rule (body_rate / movement / spatial / stateful).

    Returns (concept_id, link) where concept_id is:
      - body_rate: "_passive" (global)
      - movement:  the entity name
      - spatial:   the entity name
      - stateful:  "_passive" (global, condition on body var)
    """
    passive_type = entry.get("passive")

    if passive_type == "body_rate":
        var = entry["variable"]
        rate = float(entry["rate"])
        effect = RuleEffect(
            kind="body_rate",
            body_rate=rate,
            body_rate_variable=var,
        )
        link = CausalLink(
            action="_passive",
            result="",
            kind="passive_body_rate",
            effect=effect,
            confidence=1.0,  # innate, fully trusted
            concept=None,
        )
        return ("_passive", link)

    if passive_type == "movement":
        entity = entry["entity"]
        behavior = entry["behavior"]
        effect = RuleEffect(
            kind="movement",
            movement_behavior=behavior,
        )
        link = CausalLink(
            action="_passive",
            result="",
            kind="passive_movement",
            effect=effect,
            confidence=0.5,
            concept=entity,
        )
        return (entity, link)

    if passive_type == "spatial":
        entity = entry["entity"]
        range_ = int(entry.get("range", 1))
        effect_inner = entry.get("effect", {}) or {}
        body_delta = {k: float(v) for k, v in effect_inner.get("body", {}).items()}
        effect = RuleEffect(
            kind="spatial",
            body_delta=body_delta,
            spatial_range=range_,
        )
        link = CausalLink(
            action="_passive",
            result="",
            kind="passive_spatial",
            effect=effect,
            confidence=0.5,
            concept=entity,
        )
        return (entity, link)

    if passive_type == "stateful":
        when = entry.get("when", {}) or {}
        cond = StatefulCondition(
            var=when["var"],
            op=when["op"],
            threshold=float(when["value"]),
        )
        effect_inner = entry.get("effect", {}) or {}
        body_delta = {k: float(v) for k, v in effect_inner.get("body", {}).items()}
        effect = RuleEffect(
            kind="stateful",
            body_delta=body_delta,
            stateful_condition=cond,
        )
        link = CausalLink(
            action="_passive",
            result="",
            kind="passive_stateful",
            effect=effect,
            confidence=1.0,
            concept=None,
        )
        return ("_passive", link)

    return None


# ---------------------------------------------------------------------------
# Legacy regex parser (Stage 71-76 compatibility)
# ---------------------------------------------------------------------------


def _parse_rule_legacy(text: str) -> dict[str, Any] | None:
    """Parse a legacy string rule into a dict form (old format).

    Supported formats:
        "do OBJECT gives RESULT"
        "do OBJECT gives RESULT requires ITEM"
        "do OBJECT gives RESULT requires ITEM1 and ITEM2"
        "place RESULT on OBJECT requires ITEM"
        "make RESULT near OBJECT requires ITEM"
        "make RESULT near OBJECT requires ITEM1 and ITEM2"
        "do OBJECT with ITEM kills OBJECT"  (combat)
        "OBJECT nearby without ITEM means flee" (survival)
        "do OBJECT restores STAT"
        "sleep restores STAT"

    Returns dict with keys: concept, action, result, requires, type.
    This is the Stage 71 format; caller builds a CausalLink from it.
    """
    text = text.strip().lower()

    # Combat: "do zombie with wood_sword kills zombie"
    m = re.match(r"do (\w+) with (\w+) kills (\w+)", text)
    if m:
        return {
            "concept": m.group(1),
            "action": "do",
            "result": f"kill_{m.group(3)}",
            "requires": {m.group(2): 1},
            "type": "combat",
        }

    # Survival: "zombie nearby without wood_sword means flee"
    m = re.match(r"(\w+) nearby without (\w+) means flee", text)
    if m:
        return {
            "concept": m.group(1),
            "action": "nearby",
            "result": "flee",
            "requires": {},
            "without": m.group(2),
            "type": "survival",
        }

    # Restore: "do OBJECT restores STAT" or "sleep restores STAT"
    m = re.match(r"do (\w+) restores (\w+)", text)
    if m:
        return {
            "concept": m.group(1),
            "action": "do",
            "result": f"restore_{m.group(2)}",
            "requires": {},
            "type": "need",
            "restores": m.group(2),
        }
    m = re.match(r"sleep restores (\w+)", text)
    if m:
        return {
            "concept": "_self",
            "action": "sleep",
            "result": f"restore_{m.group(1)}",
            "requires": {},
            "type": "need",
            "restores": m.group(1),
        }

    # Gather: "do OBJECT gives RESULT [requires ...]"
    m = re.match(r"do (\w+) gives (\w+)(?:\s+requires\s+(.+))?$", text)
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(1),
            "action": "do",
            "result": m.group(2),
            "requires": requires,
            "type": "gather",
        }

    # Place: "place RESULT on OBJECT requires ..."
    m = re.match(r"place (\w+) on (\w+)(?:\s+requires\s+(.+))?$", text)
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(2),  # "empty"
            "action": "place",
            "result": m.group(1),  # "table"
            "requires": requires,
            "type": "craft",
        }

    # Make: "make RESULT near OBJECT requires ..."
    m = re.match(r"make (\w+) near (\w+)(?:\s+requires\s+(.+))?$", text)
    if m:
        requires = _parse_requires(m.group(3)) if m.group(3) else {}
        return {
            "concept": m.group(2),  # "table"
            "action": "make",
            "result": m.group(1),  # "wood_pickaxe"
            "requires": requires,
            "type": "craft",
        }

    return None


def _parse_requires(text: str) -> dict[str, int]:
    """Parse 'wood and stone_item' → {'wood': 1, 'stone_item': 1}"""
    items = re.split(r"\s+and\s+", text.strip())
    result: dict[str, int] = {}
    for item in items:
        item = item.strip()
        if item:
            result[item] = result.get(item, 0) + 1
    return result


# Backward-compat dispatcher for Stage 71-76 tests that import the old name.
# Polymorphic on input type: string → legacy regex parser, dict → new
# structured parser (converted back to legacy dict shape so test assertions
# still work). Removed in Commit 8 along with other legacy shims.
def _parse_rule(rule: Any) -> dict[str, Any] | None:
    if isinstance(rule, str):
        return _parse_rule_legacy(rule)
    if isinstance(rule, dict):
        result = _parse_rule_dict(rule)
        if result is None:
            return None
        _, link = result
        return {
            "concept": link.concept or "",
            "action": link.action,
            "result": link.result,
            "requires": link.requires,
            "type": link.effect.kind if link.effect else "",
        }
    return None


# ---------------------------------------------------------------------------
# CrafterTextbook — unified loader
# ---------------------------------------------------------------------------


class CrafterTextbook:
    """Loads a YAML textbook and registers concepts + rules into ConceptStore.

    Supports both new structured dict format (Stage 77a) and legacy string
    format (Stage 71-76) in the same file. Detection is per-rule by type:
    strings → legacy regex parser, dicts → structured parser.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path) as f:
            self.data = yaml.safe_load(f)

    @property
    def vocabulary(self) -> list[dict[str, Any]]:
        return self.data.get("vocabulary", [])

    @property
    def rules(self) -> list:
        """Raw rules list — may contain strings (legacy) or dicts (new)."""
        return self.data.get("rules", [])

    @property
    def domain(self) -> str:
        return self.data.get("domain", "unknown")

    @property
    def body_block(self) -> dict:
        """New-format body block (dict with `variables`, `prior_strength`, ...).

        Returns empty dict for legacy format (where `body` is a list).
        Used by HomeostaticTracker.init_from_textbook (Stage 77a Commit 3).
        """
        body = self.data.get("body", {})
        if isinstance(body, dict):
            return body
        return {}

    @property
    def body_rules(self) -> list[dict[str, Any]]:
        """Legacy format: list of {concept, variable, rate} entries.

        For new format, synthesizes the same shape from `passive: body_rate`
        rules in the `rules` block, plus `passive: spatial` rules which
        legacy callers used to represent conditional rates. This keeps
        tracker.init_from_body_rules working during the staged refactor.
        """
        body = self.data.get("body", [])

        # Legacy format: body is already a list of {concept, variable, rate}
        if isinstance(body, list):
            return body

        # New format: synthesize from rules
        result: list[dict[str, Any]] = []
        for rule_entry in self.rules:
            if not isinstance(rule_entry, dict):
                continue
            if rule_entry.get("passive") == "body_rate":
                result.append({
                    "concept": "_background",
                    "variable": rule_entry["variable"],
                    "rate": rule_entry["rate"],
                })
            elif rule_entry.get("passive") == "spatial":
                # Legacy conditional rate from spatial damage rule
                body_delta = (rule_entry.get("effect", {}) or {}).get("body", {}) or {}
                for var, rate in body_delta.items():
                    result.append({
                        "concept": rule_entry["entity"],
                        "variable": var,
                        "rate": rate,
                    })
        return result

    def load_into(self, store: ConceptStore) -> int:
        """Parse vocabulary and rules, register into ConceptStore.

        Returns number of rules successfully loaded.
        """
        # Register vocabulary
        for entry in self.vocabulary:
            attrs = {k: v for k, v in entry.items() if k != "id"}
            store.register(entry["id"], attrs)

        links_added = 0
        for rule_entry in self.rules:
            if isinstance(rule_entry, str):
                # Legacy string format → regex parser → manual CausalLink build
                parsed = _parse_rule_legacy(rule_entry)
                if parsed is None:
                    continue
                concept_id = parsed["concept"]
                store.register(concept_id)
                if parsed["result"] not in ("flee",):
                    store.register(parsed["result"])
                link = CausalLink(
                    action=parsed["action"],
                    result=parsed["result"],
                    requires=parsed.get("requires", {}),
                    confidence=0.5,
                    kind="action_triggered",
                    concept=concept_id,
                )
                if parsed["type"] == "survival":
                    link.condition = f"without_{parsed.get('without', '')}"
                store.add_causal(concept_id, link)
                links_added += 1

            elif isinstance(rule_entry, dict):
                # New structured format
                result = _parse_rule_dict(rule_entry)
                if result is None:
                    continue
                concept_id, link = result
                if link.kind == "action_triggered":
                    store.register(concept_id)
                    # Also register any result item so plan() can find it
                    if link.result and link.result not in ("flee",):
                        store.register(link.result)
                    store.add_causal(concept_id, link)
                else:
                    # Passive rule — goes into flat list
                    store.add_passive_rule(link)
                links_added += 1

        return links_added
