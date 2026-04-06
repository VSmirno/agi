"""Tests for Stage 69: OutcomeLabeler."""

from __future__ import annotations

import pytest
from snks.agent.outcome_labeler import OutcomeLabeler, inv_diff


class TestInvDiff:
    def test_gain(self):
        gains, losses = inv_diff({"wood": 0}, {"wood": 3})
        assert gains == {"wood": 3}
        assert losses == {}

    def test_loss(self):
        gains, losses = inv_diff({"wood": 5}, {"wood": 3})
        assert gains == {}
        assert losses == {"wood": 2}

    def test_mixed(self):
        gains, losses = inv_diff({"wood": 5, "stone": 0}, {"wood": 3, "stone": 1})
        assert gains == {"stone": 1}
        assert losses == {"wood": 2}

    def test_empty(self):
        gains, losses = inv_diff({}, {})
        assert gains == {}
        assert losses == {}

    def test_new_item(self):
        gains, losses = inv_diff({}, {"wood_pickaxe": 1})
        assert gains == {"wood_pickaxe": 1}
        assert losses == {}


class TestOutcomeLabeler:
    def setup_method(self):
        self.labeler = OutcomeLabeler()

    # --- do action ---
    def test_do_tree(self):
        assert self.labeler.label("do", {"wood": 0}, {"wood": 1}) == "tree"

    def test_do_stone(self):
        assert self.labeler.label("do", {"stone": 0}, {"stone": 1}) == "stone"

    def test_do_coal(self):
        assert self.labeler.label("do", {"coal": 0}, {"coal": 1}) == "coal"

    def test_do_iron(self):
        assert self.labeler.label("do", {"iron": 0}, {"iron": 1}) == "iron"

    def test_do_diamond(self):
        assert self.labeler.label("do", {"diamond": 0}, {"diamond": 1}) == "diamond"

    def test_do_nothing_gained(self):
        assert self.labeler.label("do", {"wood": 1}, {"wood": 1}) is None

    def test_do_no_inventory_change(self):
        assert self.labeler.label("do", {}, {}) is None

    # --- make_* actions ---
    def test_make_wood_pickaxe(self):
        inv_b = {"wood": 2}
        inv_a = {"wood": 1, "wood_pickaxe": 1}
        assert self.labeler.label("make_wood_pickaxe", inv_b, inv_a) == "table"

    def test_make_stone_pickaxe(self):
        inv_b = {"wood": 1, "stone": 1}
        inv_a = {"stone_pickaxe": 1}
        assert self.labeler.label("make_stone_pickaxe", inv_b, inv_a) == "table"

    def test_make_iron_sword(self):
        inv_b = {"wood": 1, "iron": 1}
        inv_a = {"iron_sword": 1}
        assert self.labeler.label("make_iron_sword", inv_b, inv_a) == "table"

    def test_make_failed_no_item_gained(self):
        # make action but item not gained (wrong requirements)
        inv_b = {"wood": 0}
        inv_a = {"wood": 0}
        assert self.labeler.label("make_wood_pickaxe", inv_b, inv_a) is None

    # --- place_* actions ---
    def test_place_table_success(self):
        inv_b = {"wood": 3}
        inv_a = {"wood": 1}
        assert self.labeler.label("place_table", inv_b, inv_a) == "empty"

    def test_place_furnace_success(self):
        inv_b = {"stone": 5}
        inv_a = {"stone": 1}
        assert self.labeler.label("place_furnace", inv_b, inv_a) == "empty"

    def test_place_stone_success(self):
        inv_b = {"stone": 2}
        inv_a = {"stone": 1}
        assert self.labeler.label("place_stone", inv_b, inv_a) == "empty"

    def test_place_plant_success(self):
        inv_b = {"sapling": 1}
        inv_a = {"sapling": 0}
        assert self.labeler.label("place_plant", inv_b, inv_a) == "empty"

    def test_place_table_failed(self):
        # Not enough wood — action didn't execute
        inv_b = {"wood": 1}
        inv_a = {"wood": 1}
        assert self.labeler.label("place_table", inv_b, inv_a) is None

    # --- unknown/movement actions ---
    def test_movement_returns_none(self):
        assert self.labeler.label("move_left", {}, {}) is None

    def test_noop_returns_none(self):
        assert self.labeler.label("noop", {"wood": 1}, {"wood": 1}) is None

    def test_sleep_returns_none(self):
        assert self.labeler.label("sleep", {}, {}) is None
