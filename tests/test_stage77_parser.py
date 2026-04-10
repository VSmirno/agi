"""Stage 77a Commit 2: Tests for structured YAML textbook parser.

Verifies that each rule type (action_triggered × 4 + passive × 4) parses
into a correct CausalLink with structured RuleEffect. Legacy string parser
and `result` backward-compat were removed in Commit 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from snks.agent.concept_store import CausalLink, ConceptStore
from snks.agent.crafter_textbook import (
    CrafterTextbook,
    _parse_rule_dict,
)
from snks.agent.forward_sim_types import RuleEffect, StatefulCondition


# ---------------------------------------------------------------------------
# Action-triggered rule parsing
# ---------------------------------------------------------------------------


class TestParseActionRules:
    def test_do_gather(self):
        entry = {"action": "do", "target": "tree", "effect": {"inventory": {"wood": 1}}}
        result = _parse_rule_dict(entry)
        assert result is not None
        concept_id, link = result
        assert concept_id == "tree"
        assert link.action == "do"
        assert link.kind == "action_triggered"
        assert link.effect.kind == "gather"
        assert link.effect.inventory_delta == {"wood": 1}
        assert link.confidence == 0.5

    def test_do_gather_with_requires(self):
        entry = {
            "action": "do",
            "target": "stone",
            "requires": {"wood_pickaxe": 1},
            "effect": {"inventory": {"stone_item": 1}},
        }
        _, link = _parse_rule_dict(entry)
        assert link.requires == {"wood_pickaxe": 1}
        assert link.effect.kind == "gather"
        assert link.effect.inventory_delta == {"stone_item": 1}

    def test_do_consume(self):
        """`do cow` updates body, not inventory — kind=consume."""
        entry = {"action": "do", "target": "cow", "effect": {"body": {"food": 5}}}
        _, link = _parse_rule_dict(entry)
        assert link.effect.kind == "consume"
        assert link.effect.body_delta == {"food": 5.0}

    def test_do_remove(self):
        """Combat: do zombie with sword removes it from scene."""
        entry = {
            "action": "do",
            "target": "zombie",
            "requires": {"wood_sword": 1},
            "effect": {"remove_entity": "zombie"},
        }
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "zombie"
        assert link.effect.kind == "remove"
        assert link.effect.scene_remove == "zombie"
        assert link.requires == {"wood_sword": 1}

    def test_make_craft(self):
        entry = {
            "action": "make",
            "near": "table",
            "requires": {"wood": 1},
            "effect": {"inventory": {"wood_sword": 1, "wood": -1}},
        }
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "table"  # concept_id = "near", where action fires
        assert link.action == "make"
        assert link.effect.kind == "craft"
        assert link.effect.inventory_delta == {"wood_sword": 1, "wood": -1}

    def test_place(self):
        entry = {
            "action": "place",
            "item": "table",
            "near": "empty",
            "requires": {"wood": 2},
            "effect": {
                "world_place": {"item": "table", "where": "adjacent_empty"},
                "inventory": {"wood": -2},
            },
        }
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "empty"
        assert link.effect.kind == "place"
        assert link.effect.world_place == ("table", "adjacent_empty")
        assert link.effect.inventory_delta == {"wood": -2}

    def test_sleep(self):
        entry = {"action": "sleep", "effect": {"body": {"energy": 5}}}
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "_self"
        assert link.action == "sleep"
        assert link.effect.kind == "self"
        assert link.effect.body_delta == {"energy": 5.0}


# ---------------------------------------------------------------------------
# Passive rule parsing
# ---------------------------------------------------------------------------


class TestParsePassiveRules:
    def test_body_rate(self):
        entry = {"passive": "body_rate", "variable": "food", "rate": -0.04}
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "_passive"
        assert link.kind == "passive_body_rate"
        assert link.effect.kind == "body_rate"
        assert link.effect.body_rate == -0.04
        assert link.effect.body_rate_variable == "food"
        assert link.confidence == 1.0  # innate

    def test_movement(self):
        entry = {"passive": "movement", "entity": "zombie", "behavior": "chase_player"}
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "zombie"
        assert link.kind == "passive_movement"
        assert link.effect.kind == "movement"
        assert link.effect.movement_behavior == "chase_player"
        assert link.concept == "zombie"

    def test_movement_random(self):
        entry = {"passive": "movement", "entity": "cow", "behavior": "random_walk"}
        _, link = _parse_rule_dict(entry)
        assert link.effect.movement_behavior == "random_walk"

    def test_spatial(self):
        entry = {
            "passive": "spatial",
            "entity": "zombie",
            "range": 1,
            "effect": {"body": {"health": -2}},
        }
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "zombie"
        assert link.kind == "passive_spatial"
        assert link.effect.kind == "spatial"
        assert link.effect.spatial_range == 1
        assert link.effect.body_delta == {"health": -2.0}

    def test_spatial_default_range(self):
        entry = {
            "passive": "spatial",
            "entity": "zombie",
            "effect": {"body": {"health": -1}},
        }
        _, link = _parse_rule_dict(entry)
        assert link.effect.spatial_range == 1  # default

    def test_stateful_positive(self):
        entry = {
            "passive": "stateful",
            "when": {"var": "food", "op": ">", "value": 0},
            "effect": {"body": {"health": 0.1}},
        }
        concept_id, link = _parse_rule_dict(entry)
        assert concept_id == "_passive"
        assert link.kind == "passive_stateful"
        assert link.effect.kind == "stateful"
        assert link.effect.body_delta == {"health": 0.1}
        assert link.effect.stateful_condition.var == "food"
        assert link.effect.stateful_condition.op == ">"
        assert link.effect.stateful_condition.threshold == 0.0

    def test_stateful_negative(self):
        entry = {
            "passive": "stateful",
            "when": {"var": "food", "op": "==", "value": 0},
            "effect": {"body": {"health": -0.5}},
        }
        _, link = _parse_rule_dict(entry)
        assert link.effect.body_delta == {"health": -0.5}
        assert link.effect.stateful_condition.op == "=="


# ---------------------------------------------------------------------------
# Full textbook loading
# ---------------------------------------------------------------------------


class TestLoadIntoStore:
    def test_new_yaml_loads(self):
        """The new structured crafter_textbook.yaml loads without error."""
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        store = ConceptStore()
        n = tb.load_into(store)
        assert n >= 20  # At least 20 rules in the new textbook
        assert len(store.concepts) >= 15
        assert len(store.passive_rules) >= 10

    def test_new_yaml_has_action_and_passive_rules(self):
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        store = ConceptStore()
        tb.load_into(store)

        # Action-triggered: tree should have a gather rule
        tree = store.concepts.get("tree")
        assert tree is not None
        assert any(l.effect and l.effect.kind == "gather" for l in tree.causal_links)

        # Passive: movement rule for zombie
        zombie_movement = store.movement_rule_for("zombie")
        assert zombie_movement is not None
        assert zombie_movement.effect.movement_behavior == "chase_player"

        # Passive: spatial damage for zombie — rough directional prior
        zombie_spatial = store.spatial_rules_for("zombie")
        assert len(zombie_spatial) >= 1
        assert zombie_spatial[0].effect.body_delta == {"health": -0.5}

        # Passive: body_rate rules
        body_rates = store.body_rate_rules()
        assert len(body_rates) >= 3  # food, drink, energy

        # Passive: stateful rules
        stateful = store.stateful_rules()
        assert len(stateful) >= 4  # food>0, drink>0, food==0, drink==0

    def test_body_block_new_format(self):
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        body = tb.body_block
        assert body.get("prior_strength") == 20
        variables = body.get("variables", [])
        assert len(variables) == 4  # health, food, drink, energy
        health_var = next(v for v in variables if v["name"] == "health")
        assert health_var["reference_min"] == 0
        assert health_var["reference_max"] == 9

# ---------------------------------------------------------------------------
# ConceptStore passive rule helpers
# ---------------------------------------------------------------------------


class TestConceptStorePassiveHelpers:
    def _make_loaded_store(self):
        tb = CrafterTextbook("configs/crafter_textbook.yaml")
        store = ConceptStore()
        tb.load_into(store)
        return store

    def test_body_rate_rules(self):
        store = self._make_loaded_store()
        rates = store.body_rate_rules()
        variables = {r.effect.body_rate_variable for r in rates}
        assert "food" in variables
        assert "drink" in variables
        assert "energy" in variables

    def test_stateful_rules(self):
        store = self._make_loaded_store()
        stateful = store.stateful_rules()
        # At least one positive and one negative
        positive = [r for r in stateful if r.effect.stateful_condition.op == ">"]
        negative = [r for r in stateful if r.effect.stateful_condition.op == "=="]
        assert len(positive) >= 2
        assert len(negative) >= 2

    def test_movement_rule_for_known_entity(self):
        store = self._make_loaded_store()
        rule = store.movement_rule_for("zombie")
        assert rule is not None
        assert rule.effect.movement_behavior == "chase_player"

    def test_movement_rule_for_unknown_entity(self):
        store = self._make_loaded_store()
        assert store.movement_rule_for("nonexistent") is None

    def test_spatial_rules_for_known_entity(self):
        store = self._make_loaded_store()
        rules = store.spatial_rules_for("zombie")
        assert len(rules) == 1
        # Rough directional prior — exact value not precise
        assert rules[0].effect.body_delta == {"health": -0.5}

    def test_spatial_rules_for_unknown_entity(self):
        store = self._make_loaded_store()
        assert store.spatial_rules_for("nonexistent") == []


# ---------------------------------------------------------------------------
# Malformed input handling
# ---------------------------------------------------------------------------


class TestParserRobustness:
    def test_empty_dict(self):
        assert _parse_rule_dict({}) is None

    def test_action_without_target(self):
        assert _parse_rule_dict({"action": "do"}) is None

    def test_make_without_near(self):
        assert _parse_rule_dict({"action": "make", "result": "sword"}) is None

    def test_passive_unknown_type(self):
        assert _parse_rule_dict({"passive": "nonsense"}) is None

    def test_action_unknown_type(self):
        assert _parse_rule_dict({"action": "frobnicate", "target": "x"}) is None
