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

    link = CausalLink(
        action=action,
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


def _parse_stateful_condition(when: dict) -> StatefulCondition:
    """Parse a `when:` clause into a StatefulCondition.

    Stage 82 grammar (ideology-audit 1.1): the clause is either

      atomic:   { var: food, op: ">", value: 0 }
      any_of:   { any_of: [ {var:...,op:...,value:...}, ... ] }
      all_of:   { all_of: [ {var:...,op:...,value:...}, ... ] }

    Any form may additionally carry `action_filter: <primitive>` to
    restrict the rule to ticks where the current action is that
    primitive (e.g., sleep-specific starvation damage).

    This is how the teacher writes conjunctive rules directly instead
    of forcing the surprise nursery to rediscover them (Stage 78a/c/79
    motivation is removed).
    """
    action_filter = when.get("action_filter")
    if "any_of" in when or "all_of" in when:
        mode = "any_of" if "any_of" in when else "all_of"
        raw_children = when.get("any_of") or when.get("all_of") or []
        children = [_parse_stateful_condition(child) for child in raw_children]
        return StatefulCondition(
            mode=mode,
            children=children,
            action_filter=action_filter,
        )
    return StatefulCondition(
        var=when["var"],
        op=when["op"],
        threshold=float(when["value"]),
        mode="atomic",
        action_filter=action_filter,
    )


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

            kind="passive_spatial",
            effect=effect,
            confidence=0.5,
            concept=entity,
        )
        return (entity, link)

    if passive_type == "stateful":
        when = entry.get("when", {}) or {}
        cond = _parse_stateful_condition(when)
        effect_inner = entry.get("effect", {}) or {}
        body_delta = {k: float(v) for k, v in effect_inner.get("body", {}).items()}
        effect = RuleEffect(
            kind="stateful",
            body_delta=body_delta,
            stateful_condition=cond,
        )
        link = CausalLink(
            action="_passive",

            kind="passive_stateful",
            effect=effect,
            confidence=1.0,
            concept=None,
        )
        return ("_passive", link)

    return None


# ---------------------------------------------------------------------------
# CrafterTextbook — unified loader
# ---------------------------------------------------------------------------


class CrafterTextbook:
    """Loads a structured YAML textbook and registers concepts + rules into
    ConceptStore. Stage 77a structured-only — legacy string format removed
    in Commit 8.
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
        """Raw rules list — list of dicts in structured YAML format."""
        return self.data.get("rules", [])

    @property
    def domain(self) -> str:
        return self.data.get("domain", "unknown")

    @property
    def body_block(self) -> dict:
        """Body block: dict with `variables`, `prior_strength`, etc.
        Used by HomeostaticTracker.init_from_textbook.
        """
        body = self.data.get("body", {})
        if isinstance(body, dict):
            return body
        return {}

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
            if isinstance(rule_entry, dict):
                result = _parse_rule_dict(rule_entry)
                if result is None:
                    continue
                concept_id, link = result
                if link.kind == "action_triggered":
                    store.register(concept_id)
                    store.add_causal(concept_id, link)
                else:
                    # Passive rule — flat list on store
                    store.add_passive_rule(link)
                links_added += 1

        return links_added
