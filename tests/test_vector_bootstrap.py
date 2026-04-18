"""Tests for Stage 83: vector_bootstrap — textbook YAML → seed associations."""

from __future__ import annotations

import pytest
from pathlib import Path

from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.vector_bootstrap import load_from_textbook

TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


@pytest.fixture
def seeded_model():
    model = VectorWorldModel(dim=8192, n_locations=5000, seed=42)
    stats = load_from_textbook(model, TEXTBOOK_PATH)
    return model, stats


class TestBootstrap:
    def test_textbook_loads_without_error(self, seeded_model):
        model, stats = seeded_model
        assert stats["concepts"] > 0
        assert stats["action_rules"] > 0

    def test_concepts_created(self, seeded_model):
        model, _ = seeded_model
        assert "tree" in model.concepts
        assert "stone" in model.concepts
        assert "zombie" in model.concepts
        assert "cow" in model.concepts
        assert "arrow" in model.concepts

    def test_actions_created(self, seeded_model):
        model, _ = seeded_model
        assert "do" in model.actions
        assert "make" in model.actions
        assert "place" in model.actions

    def test_roles_created(self, seeded_model):
        model, _ = seeded_model
        assert "wood" in model.roles
        assert "health" in model.roles
        assert "food" in model.roles

    def test_predict_do_tree_gives_wood(self, seeded_model):
        model, _ = seeded_model
        effect_vec, conf = model.predict("tree", "do")
        assert conf > 0.0, "Seed should provide confidence for do(tree)"
        decoded = model.decode_effect(effect_vec)
        assert decoded.get("wood", 0) > 0, f"Expected wood > 0, got {decoded}"

    def test_predict_do_cow_gives_food(self, seeded_model):
        model, _ = seeded_model
        effect_vec, conf = model.predict("cow", "do")
        assert conf > 0.0
        decoded = model.decode_effect(effect_vec)
        assert decoded.get("food", 0) > 0, f"Expected food > 0, got {decoded}"

    def test_predict_zombie_proximity_damages_health(self, seeded_model):
        model, _ = seeded_model
        effect_vec, conf = model.predict("zombie", "proximity")
        assert conf > 0.0
        decoded = model.decode_effect(effect_vec)
        assert decoded.get("health", 0) < 0, f"Expected health < 0, got {decoded}"

    def test_predict_arrow_proximity_damages_health(self, seeded_model):
        model, _ = seeded_model
        effect_vec, conf = model.predict("arrow", "proximity")
        assert conf > 0.0
        decoded = model.decode_effect(effect_vec)
        assert decoded.get("health", 0) < 0, f"Expected health < 0, got {decoded}"

    def test_unknown_concept_differs_from_known(self, seeded_model):
        model, _ = seeded_model
        # Unicorn was never seeded — its prediction should differ
        # from tree's prediction (SDM noise may give nonzero conf
        # but the content should be different from any seeded rule)
        tree_effect, tree_conf = model.predict("tree", "do")
        unicorn_effect, _ = model.predict("unicorn", "do")
        from snks.agent.vector_world_model import hamming_similarity
        sim = hamming_similarity(tree_effect, unicorn_effect)
        # Should not be highly similar (random noise ≈ 0.5)
        assert sim < 0.8 or tree_conf < 0.01

    def test_action_rules_count(self, seeded_model):
        _, stats = seeded_model
        # Textbook has: do tree, do stone, do coal, do iron, do cow,
        # do water, do zombie, do skeleton, make pickaxe, make stone_pickaxe,
        # make wood_sword, place table, sleep = 13 action rules
        assert stats["action_rules"] >= 10

    def test_passive_rules_count(self, seeded_model):
        _, stats = seeded_model
        # Spatial: zombie, skeleton = 2
        # Body rate: food, drink, energy = 3
        assert stats["passive_rules"] >= 4
