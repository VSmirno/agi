"""Tests for language/skill.py and language/skill_library.py (Stage 27)."""

import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.grid_perception import (
    SKS_DOOR_LOCKED,
    SKS_DOOR_OPEN,
    SKS_GOAL_PRESENT,
    SKS_KEY_HELD,
    SKS_KEY_PRESENT,
)
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


def make_model(**kwargs) -> CausalWorldModel:
    config = CausalAgentConfig(**kwargs)
    return CausalWorldModel(config)


def trained_doorkey_model() -> CausalWorldModel:
    """Model with DoorKey causal links observed multiple times."""
    model = make_model(causal_min_observations=1)
    for _ in range(5):
        # pickup key: key_present → key_held
        model.observe_transition(
            {SKS_KEY_PRESENT, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT},
            action=3,
            post_sks={SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT},
        )
        # toggle door: key_held + door_locked → door_open
        model.observe_transition(
            {SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT},
            action=5,
            post_sks={SKS_KEY_HELD, SKS_DOOR_OPEN, SKS_GOAL_PRESENT},
        )
    return model


class TestSkill:
    def test_primitive_skill(self):
        s = Skill(
            name="pickup_key",
            preconditions=frozenset({SKS_KEY_PRESENT}),
            effects=frozenset({SKS_KEY_HELD}),
            terminal_action=3,
            target_word="key",
        )
        assert not s.is_composite
        assert s.success_rate == 0.0

    def test_composite_skill(self):
        s = Skill(
            name="solve_doorkey",
            preconditions=frozenset({SKS_KEY_PRESENT, SKS_DOOR_LOCKED}),
            effects=frozenset({SKS_DOOR_OPEN}),
            terminal_action=None,
            target_word="goal",
            sub_skills=["pickup_key", "open_door"],
        )
        assert s.is_composite
        assert len(s.sub_skills) == 2

    def test_success_rate(self):
        s = Skill(
            name="test",
            preconditions=frozenset(),
            effects=frozenset(),
            terminal_action=3,
            target_word="key",
            success_count=8,
            attempt_count=10,
        )
        assert abs(s.success_rate - 0.8) < 0.01


class TestSkillLibrary:
    def test_register_and_get(self):
        lib = SkillLibrary()
        s = Skill(
            name="pickup_key",
            preconditions=frozenset({SKS_KEY_PRESENT}),
            effects=frozenset({SKS_KEY_HELD}),
            terminal_action=3,
            target_word="key",
        )
        lib.register(s)
        assert lib.get("pickup_key") is s
        assert lib.get("nonexistent") is None

    def test_extract_from_causal_model(self):
        model = trained_doorkey_model()
        lib = SkillLibrary()
        n = lib.extract_from_causal_model(model, min_confidence=0.5)
        assert n >= 2  # pickup_key and open_door at minimum
        assert lib.get("pickup_key") is not None
        assert lib.get("toggle_door") is not None

    def test_find_applicable(self):
        lib = SkillLibrary()
        pickup = Skill(
            name="pickup_key",
            preconditions=frozenset({SKS_KEY_PRESENT}),
            effects=frozenset({SKS_KEY_HELD}),
            terminal_action=3,
            target_word="key",
        )
        toggle = Skill(
            name="toggle_door",
            preconditions=frozenset({SKS_KEY_HELD, SKS_DOOR_LOCKED}),
            effects=frozenset({SKS_DOOR_OPEN}),
            terminal_action=5,
            target_word="door",
        )
        lib.register(pickup)
        lib.register(toggle)

        # Only pickup applicable when key present but not held
        current = {SKS_KEY_PRESENT, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT}
        result = lib.find_applicable(current, goal_sks=frozenset({SKS_DOOR_OPEN}))
        names = [s.name for s in result]
        assert "pickup_key" in names  # preconditions met, part of path to goal

    def test_compose_skills(self):
        model = trained_doorkey_model()
        lib = SkillLibrary()
        lib.extract_from_causal_model(model)
        n_composites = lib.compose_skills()
        assert n_composites >= 1
        # Should have a composite that chains pickup → toggle
        composites = [s for s in lib.skills if s.is_composite]
        assert len(composites) >= 1

    def test_extract_dedup(self):
        model = trained_doorkey_model()
        lib = SkillLibrary()
        n1 = lib.extract_from_causal_model(model)
        n2 = lib.extract_from_causal_model(model)
        assert n2 == 0  # no new skills on second extraction

    def test_empty_model_no_skills(self):
        model = make_model()
        lib = SkillLibrary()
        n = lib.extract_from_causal_model(model)
        assert n == 0
        assert len(lib.skills) == 0
