"""Tests for AnalogicalReasoner (Stage 28)."""

from __future__ import annotations

import pytest

from snks.language.analogical_reasoner import AnalogicalReasoner, AnalogyMap
from snks.language.grid_perception import (
    SKS_KEY_PRESENT, SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_DOOR_OPEN,
    SKS_CARD_PRESENT, SKS_CARD_HELD, SKS_GATE_LOCKED, SKS_GATE_OPEN,
)
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


def _make_library() -> SkillLibrary:
    lib = SkillLibrary()
    pickup_key = Skill(
        name="pickup_key",
        preconditions=frozenset({SKS_KEY_PRESENT}),
        effects=frozenset({SKS_KEY_HELD}),
        terminal_action=3,
        target_word="key",
    )
    toggle_door = Skill(
        name="toggle_door",
        preconditions=frozenset({SKS_KEY_HELD, SKS_DOOR_LOCKED}),
        effects=frozenset({SKS_DOOR_OPEN}),
        terminal_action=5,
        target_word="door",
    )
    composite = Skill(
        name="pickup_key+toggle_door",
        preconditions=frozenset({SKS_KEY_PRESENT, SKS_DOOR_LOCKED}),
        effects=frozenset({SKS_DOOR_OPEN}),
        terminal_action=None,
        target_word="door",
        sub_skills=["pickup_key", "toggle_door"],
    )
    lib.register(pickup_key)
    lib.register(toggle_door)
    lib.register(composite)
    return lib


class TestAnalogicalReasoner:

    def test_find_analogy_detects_card_gate_world(self):
        """Should find analogies when target state has card/gate predicates."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.7)
        target_sks = {SKS_CARD_PRESENT, SKS_GATE_LOCKED}

        analogies = reasoner.find_analogy(lib, target_sks)
        assert len(analogies) > 0, "Should find at least one analogy"

    def test_analogy_similarity_above_threshold(self):
        """pickup_key has 2 predicates, both have analogs → similarity=1.0."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.7)
        target_sks = {SKS_CARD_PRESENT}

        analogies = reasoner.find_analogy(lib, target_sks)
        names = [a.source_skill_name for a in analogies]
        assert "pickup_key" in names

        a = next(a for a in analogies if a.source_skill_name == "pickup_key")
        assert a.similarity == pytest.approx(1.0)

    def test_adapted_skill_has_card_target_word(self):
        """pickup_key adapted → adapted_pickup_key with target_word='card'."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.7)
        target_sks = {SKS_CARD_PRESENT}

        analogies = reasoner.find_analogy(lib, target_sks)
        a = next(a for a in analogies if a.source_skill_name == "pickup_key")
        assert a.adapted_skill.target_word == "card"

    def test_adapted_skill_preconditions_mapped(self):
        """pickup_key.preconditions={SKS_KEY_PRESENT} → {SKS_CARD_PRESENT}."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.5)
        target_sks = {SKS_CARD_PRESENT}

        analogies = reasoner.find_analogy(lib, target_sks)
        a = next(a for a in analogies if a.source_skill_name == "pickup_key")
        assert SKS_CARD_PRESENT in a.adapted_skill.preconditions
        assert SKS_KEY_PRESENT not in a.adapted_skill.preconditions

    def test_adapted_skill_effects_mapped(self):
        """pickup_key.effects={SKS_KEY_HELD} → {SKS_CARD_HELD}."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.5)
        target_sks = {SKS_CARD_PRESENT}

        analogies = reasoner.find_analogy(lib, target_sks)
        a = next(a for a in analogies if a.source_skill_name == "pickup_key")
        assert SKS_CARD_HELD in a.adapted_skill.effects
        assert SKS_KEY_HELD not in a.adapted_skill.effects

    def test_no_analogy_for_unrelated_skill(self):
        """Skill with no predicates in ROLE_REGISTRY → similarity=0, no analogy."""
        lib = SkillLibrary()
        weird_skill = Skill(
            name="weird",
            preconditions=frozenset({100, 101}),
            effects=frozenset({102}),
            terminal_action=0,
            target_word="unknown",
        )
        lib.register(weird_skill)
        reasoner = AnalogicalReasoner(threshold=0.7)
        analogies = reasoner.find_analogy(lib, {SKS_CARD_PRESENT})
        assert len(analogies) == 0

    def test_composite_analogy_found(self):
        """pickup_key+toggle_door should be analogically mapped to card/gate."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.7)
        target_sks = {SKS_CARD_PRESENT, SKS_GATE_LOCKED}

        analogies = reasoner.find_analogy(lib, target_sks)
        names = [a.source_skill_name for a in analogies]
        assert "pickup_key+toggle_door" in names

    def test_analogy_map_has_sks_mapping(self):
        """AnalogyMap.sks_mapping should map key→card and door→gate predicates."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.5)
        analogies = reasoner.find_analogy(lib, {SKS_CARD_PRESENT, SKS_GATE_LOCKED})

        # Find toggle_door analogy.
        a = next((x for x in analogies if x.source_skill_name == "toggle_door"), None)
        assert a is not None
        assert a.sks_mapping.get(SKS_KEY_HELD) == SKS_CARD_HELD
        assert a.sks_mapping.get(SKS_DOOR_LOCKED) == SKS_GATE_LOCKED
        assert a.sks_mapping.get(SKS_DOOR_OPEN) == SKS_GATE_OPEN

    def test_composites_returned_first(self):
        """Composites should come before primitives in analogy list."""
        lib = _make_library()
        reasoner = AnalogicalReasoner(threshold=0.5)
        analogies = reasoner.find_analogy(lib, {SKS_CARD_PRESENT, SKS_GATE_LOCKED})

        if len(analogies) >= 2:
            # First result should be composite.
            assert analogies[0].adapted_skill.is_composite, \
                f"Expected composite first, got {analogies[0].source_skill_name}"
