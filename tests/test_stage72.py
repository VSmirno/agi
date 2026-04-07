"""Stage 72: Gate tests — Perception Pivot.

Tests self-organized perception, experiential grounding, drive-based planning,
and spatial map navigation without GT.

All tests run locally without GPU.

Design: docs/superpowers/specs/2026-04-07-stage72-perception-pivot-design.md
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snks.agent.concept_store import ConceptStore, PlannedStep
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.perception import (
    perceive,
    on_action_outcome,
    select_goal,
    get_drive_strengths,
    explore_action,
    babble_probability,
    MIN_SIMILARITY,
    EMA_ALPHA,
    BABBLE_BASE_PROB,
    BABBLE_MIN_PROB,
)
from snks.agent.reactive_check import ReactiveCheck


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> ConceptStore:
    store = ConceptStore()
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    tb.load_into(store)
    return store


_EncoderOutput = namedtuple("_EncoderOutput", ["z_real", "z_vsa", "near_logits"])


class _MockEncoder:
    """Returns a fixed z_real vector."""

    def __init__(self, z_real: torch.Tensor | None = None, dim: int = 2048):
        self.dim = dim
        self._z = z_real if z_real is not None else torch.randn(dim)

    def __call__(self, pixels: torch.Tensor) -> _EncoderOutput:
        z = self._z.unsqueeze(0) if self._z.dim() == 1 else self._z
        return _EncoderOutput(
            z_real=z,
            z_vsa=(z > 0).float(),
            near_logits=torch.zeros(1, 12),
        )


# ---------------------------------------------------------------------------
# Gate 1: query_visual_scored returns score
# ---------------------------------------------------------------------------


class TestGate1QueryVisualScored:
    def test_returns_score(self):
        store = _make_store()
        z_tree = torch.randn(2048)
        store.ground_visual("tree", z_tree)

        concept, score = store.query_visual_scored(z_tree)
        assert concept is not None
        assert concept.id == "tree"
        assert score > 0.99  # same vector → cosine ~1.0

    def test_no_visuals_returns_none(self):
        store = _make_store()
        z = torch.randn(2048)
        concept, score = store.query_visual_scored(z)
        assert concept is None
        assert score == -1.0

    def test_dissimilar_low_score(self):
        store = _make_store()
        z_tree = torch.randn(2048)
        store.ground_visual("tree", z_tree)

        z_other = -z_tree  # opposite direction
        concept, score = store.query_visual_scored(z_other)
        assert score < 0.0

    def test_backward_compat_query_visual(self):
        """query_visual() still works, returns concept without score."""
        store = _make_store()
        z_tree = torch.randn(2048)
        store.ground_visual("tree", z_tree)

        concept = store.query_visual(z_tree)
        assert concept is not None
        assert concept.id == "tree"


# ---------------------------------------------------------------------------
# Gate 2: perceive() — concept matching with threshold
# ---------------------------------------------------------------------------


class TestGate2Perceive:
    def test_perceive_known_concept(self):
        store = _make_store()
        z_tree = torch.randn(2048)
        z_norm = F.normalize(z_tree.unsqueeze(0), dim=1).squeeze(0)
        store.ground_visual("tree", z_norm)

        encoder = _MockEncoder(z_real=z_norm)
        pixels = torch.randn(3, 64, 64)

        concept, z_real = perceive(pixels, encoder, store)
        assert concept is not None
        assert concept.id == "tree"

    def test_perceive_unknown_below_threshold(self):
        store = _make_store()
        z_tree = torch.randn(2048)
        store.ground_visual("tree", z_tree)

        # Very different z_real
        z_other = torch.randn(2048) * 100
        encoder = _MockEncoder(z_real=z_other)
        pixels = torch.randn(3, 64, 64)

        concept, z_real = perceive(pixels, encoder, store, min_similarity=0.99)
        # With min_similarity=0.99 and random vectors, should return None
        assert concept is None

    def test_perceive_no_visuals(self):
        store = _make_store()
        encoder = _MockEncoder()
        pixels = torch.randn(3, 64, 64)

        concept, z_real = perceive(pixels, encoder, store)
        assert concept is None


# ---------------------------------------------------------------------------
# Gate 3: on_action_outcome — experiential grounding
# ---------------------------------------------------------------------------


class TestGate3ExperientialGrounding:
    def test_one_shot_grounding(self):
        store = _make_store()
        labeler = OutcomeLabeler()
        z = torch.randn(2048)

        inv_before = {"wood": 0}
        inv_after = {"wood": 1}

        result = on_action_outcome("do", inv_before, inv_after, z, store, labeler)
        assert result == "tree"

        # Check visual was grounded
        concept = store.query_text("tree")
        assert concept.visual is not None

    def test_ema_refinement(self):
        store = _make_store()
        labeler = OutcomeLabeler()

        # First grounding
        z1 = F.normalize(torch.randn(2048).unsqueeze(0), dim=1).squeeze(0)
        on_action_outcome("do", {"wood": 0}, {"wood": 1}, z1, store, labeler)
        visual_after_first = store.query_text("tree").visual.clone()

        # Second grounding — should EMA update
        z2 = F.normalize(torch.randn(2048).unsqueeze(0), dim=1).squeeze(0)
        on_action_outcome("do", {"wood": 1}, {"wood": 2}, z2, store, labeler)
        visual_after_second = store.query_text("tree").visual

        # Should have changed (EMA update)
        assert not torch.allclose(visual_after_first, visual_after_second)

        # Should be closer to z1 than z2 (EMA with alpha=0.1)
        sim_to_z1 = F.cosine_similarity(visual_after_second.unsqueeze(0), z1.unsqueeze(0)).item()
        sim_to_z2 = F.cosine_similarity(visual_after_second.unsqueeze(0), z2.unsqueeze(0)).item()
        assert sim_to_z1 > sim_to_z2  # weighted toward old

    def test_no_effect_returns_none(self):
        store = _make_store()
        labeler = OutcomeLabeler()
        z = torch.randn(2048)

        result = on_action_outcome("do", {"wood": 0}, {"wood": 0}, z, store, labeler)
        assert result is None

    def test_craft_grounding(self):
        store = _make_store()
        labeler = OutcomeLabeler()
        z = torch.randn(2048)

        inv_before = {"wood": 2}
        inv_after = {"wood": 1, "wood_pickaxe": 1}

        result = on_action_outcome("make_wood_pickaxe", inv_before, inv_after, z, store, labeler)
        assert result == "table"


# ---------------------------------------------------------------------------
# Gate 4: select_goal — drive competition
# ---------------------------------------------------------------------------


class TestGate4DriveGoalSelection:
    def test_low_food_selects_food(self):
        store = _make_store()
        inv = {"food": 1, "drink": 9, "energy": 9}
        goal, plan = select_goal(inv, store)
        assert goal == "restore_food"

    def test_low_drink_selects_drink(self):
        store = _make_store()
        inv = {"food": 9, "drink": 1, "energy": 9}
        goal, plan = select_goal(inv, store)
        assert goal == "restore_drink"

    def test_low_energy_selects_energy(self):
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 1}
        goal, plan = select_goal(inv, store)
        assert goal == "restore_energy"

    def test_all_ok_selects_wood(self):
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9}
        goal, plan = select_goal(inv, store)
        assert goal == "wood"

    def test_drive_strengths(self):
        strengths = get_drive_strengths({"food": 2, "drink": 9, "energy": 9})
        assert strengths["restore_food"] == 6.0  # (5-2)*2
        assert strengths["restore_drink"] == 0.0
        assert strengths["wood"] > 0

    def test_wood_plan_has_steps(self):
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9}
        goal, plan = select_goal(inv, store)
        assert goal == "wood"
        assert len(plan) >= 1
        assert plan[0].action == "do"
        assert plan[0].target == "tree"

    def test_progression_wood_to_sword(self):
        """After getting wood, first priority is sword."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 3}
        goal, plan = select_goal(inv, store)
        assert goal == "wood_sword"

    def test_progression_sword_to_pickaxe(self):
        """After sword, drive shifts to pickaxe."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 5, "wood_sword": 1}
        goal, plan = select_goal(inv, store)
        assert goal == "wood_pickaxe"

    def test_progression_pickaxe_to_stone(self):
        """After getting pickaxe, drive shifts to stone."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 5, "wood_pickaxe": 1, "wood_sword": 1}
        goal, plan = select_goal(inv, store)
        assert goal == "stone_item"


# ---------------------------------------------------------------------------
# Gate 4b: Motor babbling / curiosity-driven exploration
# ---------------------------------------------------------------------------


class TestGate4bMotorBabbling:
    def test_babble_probability_high_when_no_prototypes(self):
        store = _make_store()
        prob = babble_probability(store)
        assert prob == BABBLE_BASE_PROB  # no grounded concepts

    def test_babble_probability_decays_with_grounding(self):
        store = _make_store()
        # Ground 5 concepts
        for cid in ["tree", "stone", "coal", "iron", "table"]:
            store.ground_visual(cid, torch.randn(2048))
        prob = babble_probability(store)
        assert prob < BABBLE_BASE_PROB
        assert prob >= BABBLE_MIN_PROB

    def test_babble_probability_floor(self):
        store = _make_store()
        # Ground many concepts
        for cid in store.concepts:
            store.ground_visual(cid, torch.randn(2048))
        prob = babble_probability(store)
        assert prob >= BABBLE_MIN_PROB

    def test_explore_action_returns_valid(self):
        store = _make_store()
        rng = np.random.RandomState(42)
        actions = set()
        for _ in range(200):
            a = explore_action(rng, store)
            actions.add(a)
        # Should include both moves and babble_do
        assert "babble_do" in actions
        moves = actions - {"babble_do"}
        assert len(moves) > 0  # has move actions too

    def test_explore_mostly_moves_when_grounded(self):
        store = _make_store()
        for cid in store.concepts:
            store.ground_visual(cid, torch.randn(2048))
        rng = np.random.RandomState(42)
        babble_count = sum(
            1 for _ in range(1000) if explore_action(rng, store) == "babble_do"
        )
        # Should be rare when fully grounded
        assert babble_count < 100  # < 10%


# ---------------------------------------------------------------------------
# Gate 5: Spatial map navigation without GT
# ---------------------------------------------------------------------------


class TestGate5SpatialNavigation:
    def test_map_update_and_find(self):
        sm = CrafterSpatialMap()
        sm.update((10, 10), "tree")
        sm.update((10, 11), "empty")
        sm.update((15, 15), "stone")

        pos = sm.find_nearest("tree", (12, 12))
        assert pos == (10, 10)

    def test_map_find_unknown(self):
        sm = CrafterSpatialMap()
        sm.update((10, 10), "tree")
        pos = sm.find_nearest("stone", (10, 10))
        assert pos is None

    def test_unvisited_neighbors(self):
        sm = CrafterSpatialMap()
        sm.update((10, 10), "empty")
        unvisited = sm.unvisited_neighbors((10, 10), radius=1)
        # 3x3 - 1 visited = 8 unvisited
        assert len(unvisited) == 8

    def test_reset_clears_map(self):
        sm = CrafterSpatialMap()
        sm.update((10, 10), "tree")
        assert sm.n_visited == 1
        sm.reset()
        assert sm.n_visited == 0


# ---------------------------------------------------------------------------
# Gate 6: Reactive check — danger only (check_needs deprecated)
# ---------------------------------------------------------------------------


class TestGate6ReactiveCheck:
    def test_danger_with_weapon(self):
        store = _make_store()
        rc = ReactiveCheck(store)
        assert rc.check("zombie", {"wood_sword": 1}) == "do"

    def test_danger_without_weapon(self):
        store = _make_store()
        rc = ReactiveCheck(store)
        assert rc.check("zombie", {}) == "flee"

    def test_non_dangerous_returns_none(self):
        store = _make_store()
        rc = ReactiveCheck(store)
        assert rc.check("tree", {}) is None

    def test_unknown_returns_none(self):
        store = _make_store()
        rc = ReactiveCheck(store)
        assert rc.check("unknown_thing", {}) is None


# ---------------------------------------------------------------------------
# Gate 7: Prediction-verification loop
# ---------------------------------------------------------------------------


class TestGate7PredictionVerification:
    def test_predict_verify_confirm(self):
        store = _make_store()

        pred = store.predict_before_action("tree", "do", {})
        assert pred is not None
        assert pred.expected == "wood"

        initial_conf = pred.link.confidence
        store.verify_after_action(pred, "do", "wood")
        assert pred.link.confidence > initial_conf

    def test_predict_verify_refute(self):
        store = _make_store()

        pred = store.predict_before_action("tree", "do", {})
        initial_conf = pred.link.confidence
        store.verify_after_action(pred, "do", "nothing")
        assert pred.link.confidence < initial_conf

    def test_surprise_on_no_prediction(self):
        store = _make_store()
        store.verify_after_action(None, "do", "diamond", near="unknown")
        assert len(store.surprises) == 1

    def test_confidence_accumulates(self):
        store = _make_store()
        for _ in range(5):
            pred = store.predict_before_action("tree", "do", {})
            store.verify_after_action(pred, "do", "wood")
        assert pred.link.confidence >= 0.8


# ---------------------------------------------------------------------------
# Gate 8: Backward chaining plans still work
# ---------------------------------------------------------------------------


class TestGate8BackwardChaining:
    def test_wood_plan(self):
        store = _make_store()
        plan = store.plan("wood")
        assert len(plan) == 1
        assert plan[0].target == "tree"

    def test_iron_plan(self):
        store = _make_store()
        plan = store.plan("iron_item")
        assert len(plan) >= 5
        assert plan[0].target == "tree"
        assert plan[-1].expected_gain == "iron_item"

    def test_restore_food_plan(self):
        store = _make_store()
        plan = store.plan("restore_food")
        assert len(plan) == 1
        assert plan[0].target == "cow"
        assert plan[0].action == "do"

    def test_restore_drink_plan(self):
        store = _make_store()
        plan = store.plan("restore_drink")
        assert len(plan) == 1
        assert plan[0].target == "water"

    def test_restore_energy_plan(self):
        store = _make_store()
        plan = store.plan("restore_energy")
        assert len(plan) == 1
        assert plan[0].action == "sleep"
