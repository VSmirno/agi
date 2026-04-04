"""Stage 63: Crafter QA unit tests."""

import pytest

from snks.agent.cls_world_model import CLSWorldModel
from snks.agent.world_model_trainer import generate_synthetic_transitions, TRAIN_COLORS
from snks.agent.crafter_trainer import generate_crafter_transitions


@pytest.fixture(scope="module")
def model():
    mg = generate_synthetic_transitions(TRAIN_COLORS)
    cr = generate_crafter_transitions()
    # Small SDM for fast CPU tests — logic correctness, not capacity
    m = CLSWorldModel(dim=512, n_locations=500)
    m.train(mg + cr)
    return m


class TestCrafterL1:
    def test_collect_wood(self, model):
        assert model.qa_crafter_can_do("do", "tree") is True

    def test_collect_stone_with_pickaxe(self, model):
        assert model.qa_crafter_can_do("do", "stone", {"wood_pickaxe": 1}) is True

    def test_collect_stone_no_pickaxe(self, model):
        assert model.qa_crafter_can_do("do", "stone") is False

    def test_collect_iron_with_stone_pickaxe(self, model):
        assert model.qa_crafter_can_do("do", "iron", {"stone_pickaxe": 1}) is True

    def test_collect_iron_no_pickaxe(self, model):
        assert model.qa_crafter_can_do("do", "iron") is False

    def test_place_table(self, model):
        assert model.qa_crafter_can_do("place_table", "empty", {"wood": 2}) is True

    def test_craft_pickaxe_at_table(self, model):
        assert model.qa_crafter_can_do("make_wood_pickaxe", "table", {"wood": 1}) is True

    def test_craft_pickaxe_no_table(self, model):
        assert model.qa_crafter_can_do("make_wood_pickaxe", "empty") is False


class TestCrafterL2:
    def test_need_for_wood_pickaxe(self, model):
        assert model.qa_crafter_needs("make_wood_pickaxe", "table") == "1 wood"

    def test_need_for_stone_pickaxe(self, model):
        needs = model.qa_crafter_needs("make_stone_pickaxe", "table")
        assert "wood" in needs and "stone" in needs

    def test_need_for_iron_pickaxe(self, model):
        assert "iron" in model.qa_crafter_needs("make_iron_pickaxe", "table")


class TestCrafterL3:
    def test_chop_tree(self, model):
        r = model.qa_crafter_result("do", "tree")
        assert r["result"] == "collected"

    def test_craft_pickaxe(self, model):
        r = model.qa_crafter_result("make_wood_pickaxe", "table", {"wood": 1})
        assert r["result"] == "crafted"

    def test_place_table(self, model):
        r = model.qa_crafter_result("place_table", "empty", {"wood": 2})
        assert r["result"] == "placed"


class TestCrafterFailures:
    def test_mine_stone_no_tool(self, model):
        r = model.qa_crafter_result("do", "stone")
        assert r["result"] == "failed_no_tool"

    def test_craft_no_table(self, model):
        r = model.qa_crafter_result("make_wood_pickaxe", "empty")
        assert r["result"] == "failed_no_station"


class TestAbstraction:
    def test_min_categories(self, model):
        assert len(model.abstraction.categories) >= 3

    def test_carryable_exists(self, model):
        assert any("carryable" in n for n in model.abstraction.categories)

    def test_solid_exists(self, model):
        assert "solid" in model.abstraction.categories

    def test_key_is_carryable(self, model):
        cats = model.abstraction.get_categories_for_object("key")
        assert any("carryable" in c for c in cats)

    def test_key_is_solid(self, model):
        cats = model.abstraction.get_categories_for_object("key")
        assert "solid" in cats
