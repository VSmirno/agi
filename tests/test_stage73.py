"""Stage 73: Gate tests — Autonomous Craft + Self-Organized Perception.

Tests empty grounding from first frame, zombie grounding through damage,
craft babbling, universal verification, and drive progression.

All tests run locally without GPU.

Design: docs/superpowers/specs/2026-04-07-stage73-autonomous-craft-design.md
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snks.agent.concept_store import ConceptStore, PlannedStep
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.perception import (
    perceive,
    perceive_field,
    VisualField,
    ground_empty_on_start,
    ground_zombie_on_damage,
    on_action_outcome,
    verify_outcome,
    explore_action,
    select_goal,
    babble_probability,
    MIN_SIMILARITY,
    EMA_ALPHA,
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


_EncoderOutput = namedtuple("_EncoderOutput", ["z_real", "z_vsa", "near_logits", "feature_map"])


class _MockEncoder:
    """Returns a fixed z_real vector + feature map."""

    def __init__(self, z_real: torch.Tensor | None = None, dim: int = 2048):
        self.dim = dim
        self._z = z_real if z_real is not None else torch.randn(dim)

    def __call__(self, pixels: torch.Tensor) -> _EncoderOutput:
        z = self._z.unsqueeze(0) if self._z.dim() == 1 else self._z
        feat = torch.randn(1, 256, 4, 4)
        if self._z.dim() == 1 and self._z.shape[0] >= 256:
            feat[0, :, 1:3, 1:3] = self._z[:256].reshape(256, 1, 1).expand(256, 2, 2)
        return _EncoderOutput(
            z_real=z,
            z_vsa=(z > 0).float(),
            near_logits=torch.zeros(1, 12),
            feature_map=feat,
        )


def _make_vf(z: torch.Tensor) -> VisualField:
    """Create a VisualField with center_feature from z[:256]."""
    vf = VisualField()
    vf.center_feature = z[:256] if z.shape[0] >= 256 else z
    return vf


# ---------------------------------------------------------------------------
# Gate 1: Empty grounding from first frame
# ---------------------------------------------------------------------------


class TestGate1EmptyGrounding:
    def test_grounds_empty_first_time(self):
        """First call with no visual grounding returns True, sets concept.visual."""
        store = _make_store()
        z = torch.randn(2048)
        encoder = _MockEncoder(z_real=z)
        pixels = torch.randn(3, 64, 64)

        result = ground_empty_on_start(pixels, encoder, store)

        assert result is True
        concept = store.query_text("empty")
        assert concept is not None
        assert concept.visual is not None

    def test_skips_if_already_grounded(self):
        """If empty already has a visual, returns False and does not overwrite."""
        store = _make_store()
        z_first = F.normalize(torch.randn(256).unsqueeze(0), dim=1).squeeze(0)
        store.ground_visual("empty", z_first)

        z_second = torch.randn(2048)
        encoder = _MockEncoder(z_real=z_second)
        pixels = torch.randn(3, 64, 64)

        result = ground_empty_on_start(pixels, encoder, store)

        assert result is False
        concept = store.query_text("empty")
        assert torch.allclose(concept.visual, z_first)

    def test_empty_is_perceivable(self):
        """After grounding empty, perceive returns the empty concept."""
        store = _make_store()
        z = torch.randn(2048)
        encoder = _MockEncoder(z_real=z)
        pixels = torch.randn(3, 64, 64)

        ground_empty_on_start(pixels, encoder, store)

        concept, cf = perceive(pixels, encoder, store)
        assert concept is not None
        assert concept.id == "empty"


# ---------------------------------------------------------------------------
# Gate 2: Zombie grounding through damage
# ---------------------------------------------------------------------------


class TestGate2ZombieGrounding:
    def test_grounds_on_health_drop(self):
        """Health drop with food>0 and drink>0 grounds zombie, returns True."""
        store = _make_store()
        z = torch.randn(2048)

        inv_before = {"health": 9, "food": 5, "drink": 5}
        inv_after = {"health": 7, "food": 5, "drink": 5}

        result = ground_zombie_on_damage(inv_before, inv_after, _make_vf(z), store)

        assert result is True
        concept = store.query_text("zombie")
        assert concept is not None
        assert concept.visual is not None

    def test_ignores_starvation(self):
        """Health drop with food=0 is starvation, not zombie damage; returns False."""
        store = _make_store()
        z = torch.randn(2048)

        inv_before = {"health": 9, "food": 0, "drink": 5}
        inv_after = {"health": 8, "food": 0, "drink": 5}

        result = ground_zombie_on_damage(inv_before, inv_after, _make_vf(z), store)

        assert result is False
        concept = store.query_text("zombie")
        assert concept.visual is None

    def test_ignores_dehydration(self):
        """Health drop with drink=0 is dehydration, not zombie damage; returns False."""
        store = _make_store()
        z = torch.randn(2048)

        inv_before = {"health": 9, "food": 5, "drink": 0}
        inv_after = {"health": 8, "food": 5, "drink": 0}

        result = ground_zombie_on_damage(inv_before, inv_after, _make_vf(z), store)

        assert result is False

    def test_ema_refines(self):
        """Second damage encounter updates zombie visual via EMA."""
        store = _make_store()

        z1 = F.normalize(torch.randn(256).unsqueeze(0), dim=1).squeeze(0)
        inv_before = {"health": 9, "food": 5, "drink": 5}
        inv_mid = {"health": 7, "food": 5, "drink": 5}
        ground_zombie_on_damage(inv_before, inv_mid, _make_vf(z1), store)
        visual_after_first = store.query_text("zombie").visual.clone()

        z2 = F.normalize(torch.randn(256).unsqueeze(0), dim=1).squeeze(0)
        inv_after = {"health": 5, "food": 5, "drink": 5}
        ground_zombie_on_damage(inv_mid, inv_after, _make_vf(z2), store)
        visual_after_second = store.query_text("zombie").visual

        # Visual must have changed after second encounter
        assert not torch.allclose(visual_after_first, visual_after_second)

        # EMA with alpha=0.1: new visual should be closer to z1 than z2
        sim_to_z1 = F.cosine_similarity(
            visual_after_second.unsqueeze(0), z1.unsqueeze(0)
        ).item()
        sim_to_z2 = F.cosine_similarity(
            visual_after_second.unsqueeze(0), z2.unsqueeze(0)
        ).item()
        assert sim_to_z1 > sim_to_z2

    def test_no_damage_returns_false(self):
        """No health drop → returns False regardless of other conditions."""
        store = _make_store()
        z = torch.randn(2048)

        inv_before = {"health": 7, "food": 5, "drink": 5}
        inv_after = {"health": 7, "food": 5, "drink": 5}

        result = ground_zombie_on_damage(inv_before, inv_after, _make_vf(z), store)
        assert result is False

    def test_reactive_after_grounding(self):
        """After zombie is grounded as dangerous, ReactiveCheck.check returns flee."""
        store = _make_store()
        z = torch.randn(2048)

        inv_before = {"health": 9, "food": 5, "drink": 5}
        inv_after = {"health": 7, "food": 5, "drink": 5}
        ground_zombie_on_damage(inv_before, inv_after, _make_vf(z), store)

        rc = ReactiveCheck(store)
        # No weapon → should flee
        result = rc.check("zombie", {})
        assert result == "flee"


# ---------------------------------------------------------------------------
# Gate 3: Craft babbling
# ---------------------------------------------------------------------------


class TestGate3CraftBabbling:
    def test_babble_place_table_with_wood(self):
        """With wood>=2, explore_action can return babble_place_table."""
        store = _make_store()
        rng = np.random.RandomState(0)
        inventory = {"wood": 2}

        actions = {explore_action(rng, store, inventory) for _ in range(500)}
        assert "babble_place_table" in actions

    def test_babble_no_craft_without_resources(self):
        """With wood=0, craft babble actions are never returned."""
        store = _make_store()
        rng = np.random.RandomState(0)
        inventory = {"wood": 0}

        actions = [explore_action(rng, store, inventory) for _ in range(300)]
        assert "babble_place_table" not in actions
        assert "babble_make_wood_pickaxe" not in actions

    def test_babble_make_pickaxe_with_wood(self):
        """With wood>=1, explore_action can return babble_make_wood_pickaxe."""
        store = _make_store()
        rng = np.random.RandomState(7)
        inventory = {"wood": 1}

        actions = {explore_action(rng, store, inventory) for _ in range(500)}
        assert "babble_make_wood_pickaxe" in actions

    def test_on_action_outcome_place_table(self):
        """place_table with wood loss labels near as 'empty' via OutcomeLabeler PLACE detection."""
        store = _make_store()
        labeler = OutcomeLabeler()
        z = torch.randn(2048)

        inv_before = {"wood": 2}
        inv_after = {"wood": 0}

        # place_table costs 2 wood → OutcomeLabeler returns "empty"
        result = on_action_outcome("place_table", inv_before, inv_after, z, store, labeler)
        assert result == "empty"

    def test_on_action_outcome_no_craft_without_cost(self):
        """place_table with no wood loss returns None (action failed)."""
        store = _make_store()
        labeler = OutcomeLabeler()
        z = torch.randn(2048)

        inv_before = {"wood": 2}
        inv_after = {"wood": 2}  # no change → action failed

        result = on_action_outcome("place_table", inv_before, inv_after, z, store, labeler)
        assert result is None


# ---------------------------------------------------------------------------
# Gate 4: Universal verification
# ---------------------------------------------------------------------------


class TestGate4UniversalVerify:
    def test_verify_confirms_rule(self):
        """verify_outcome("tree", "do", "wood") increases the causal link confidence."""
        store = _make_store()
        tree_concept = store.query_text("tree")
        assert tree_concept is not None

        link = tree_concept.find_causal("do")
        assert link is not None
        initial_confidence = link.confidence

        verify_outcome("tree", "do", "wood", store)

        assert link.confidence > initial_confidence

    def test_verify_refutes_wrong(self):
        """verify_outcome("tree", "do", "nothing") decreases the causal link confidence."""
        store = _make_store()
        tree_concept = store.query_text("tree")
        link = tree_concept.find_causal("do")
        initial_confidence = link.confidence

        verify_outcome("tree", "do", "nothing", store)

        assert link.confidence < initial_confidence

    def test_multiple_verifications_build_confidence(self):
        """5 confirmations of tree→do→wood drive confidence to >=0.8."""
        store = _make_store()
        for _ in range(5):
            verify_outcome("tree", "do", "wood", store)

        tree_concept = store.query_text("tree")
        link = tree_concept.find_causal("do")
        assert link.confidence >= 0.8

    def test_verify_none_near_is_noop(self):
        """verify_outcome with None near_label is a safe no-op."""
        store = _make_store()
        # Should not raise
        verify_outcome(None, "do", "wood", store)

    def test_verify_none_outcome_is_noop(self):
        """verify_outcome with None actual_outcome is a safe no-op."""
        store = _make_store()
        tree_concept = store.query_text("tree")
        link = tree_concept.find_causal("do")
        initial_confidence = link.confidence

        verify_outcome("tree", "do", None, store)

        assert link.confidence == initial_confidence

    def test_verify_unknown_concept_is_noop(self):
        """verify_outcome with unknown concept_id does not raise."""
        store = _make_store()
        # Should not raise
        verify_outcome("nonexistent_thing", "do", "wood", store)


# ---------------------------------------------------------------------------
# Gate 5: Drive progression
# ---------------------------------------------------------------------------


class TestGate5DriveProgression:
    def test_wood_to_sword(self):
        """With wood=3 and no sword, first priority is sword."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 3}
        goal, plan = select_goal(inv, store)
        assert goal == "wood_sword"

    def test_sword_to_pickaxe(self):
        """After sword, drive shifts to pickaxe."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 5, "wood_sword": 1}
        goal, plan = select_goal(inv, store)
        assert goal == "wood_pickaxe"

    def test_pickaxe_to_stone(self):
        """With sword+pickaxe, drive shifts to stone."""
        store = _make_store()
        inv = {
            "food": 9,
            "drink": 9,
            "energy": 9,
            "wood": 5,
            "wood_pickaxe": 1,
            "wood_sword": 1,
            "stone_item": 0,
        }
        goal, plan = select_goal(inv, store)
        assert goal == "stone_item"

    def test_survival_overrides_progression(self):
        """Low food overrides resource progression: goal is restore_food."""
        store = _make_store()
        inv = {
            "food": 1,
            "drink": 9,
            "energy": 9,
            "wood": 3,
        }
        goal, plan = select_goal(inv, store)
        assert goal == "restore_food"

    def test_low_drink_overrides(self):
        """Low drink overrides resource progression: goal is restore_drink."""
        store = _make_store()
        inv = {
            "food": 9,
            "drink": 1,
            "energy": 9,
            "wood": 3,
        }
        goal, plan = select_goal(inv, store)
        assert goal == "restore_drink"

    def test_all_ok_no_wood_selects_wood(self):
        """All survival stats OK and no resources → goal is wood."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9}
        goal, plan = select_goal(inv, store)
        assert goal == "wood"

    def test_plan_returned_with_steps(self):
        """select_goal always returns a non-empty plan list."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 3, "wood_sword": 1}
        goal, plan = select_goal(inv, store)
        assert isinstance(plan, list)
        assert len(plan) >= 1

    def test_wood_sword_plan_has_steps(self):
        """wood_sword plan contains craft steps."""
        store = _make_store()
        inv = {"food": 9, "drink": 9, "energy": 9, "wood": 3}
        goal, plan = select_goal(inv, store)
        assert goal == "wood_sword"
        assert len(plan) >= 1
