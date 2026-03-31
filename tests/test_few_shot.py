"""Tests for Few-Shot Learning (Stage 30).

Tests:
  1. DemoStep / Demonstration data structures
  2. DemonstrationRecorder captures transitions
  3. FewShotLearner extracts causal model + skills from demos
  4. FewShotAgent learns from demos and uses skills
"""

from __future__ import annotations

import pytest

from snks.language.demonstration import DemoStep, Demonstration, DemonstrationRecorder
from snks.language.few_shot_learner import FewShotLearner
from snks.language.few_shot_agent import FewShotAgent
from snks.language.skill_library import SkillLibrary
from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.grid_perception import (
    SKS_KEY_PRESENT, SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_DOOR_OPEN, SKS_GOAL_PRESENT,
)


# ── DemoStep / Demonstration ─────────────────────────────────────────

class TestDemoStep:
    def test_fields(self):
        step = DemoStep(
            sks_before=frozenset({50, 52}),
            action=3,
            sks_after=frozenset({51, 52}),
        )
        assert step.action == 3
        assert 50 in step.sks_before
        assert 51 in step.sks_after


class TestDemonstration:
    def test_empty(self):
        demo = Demonstration()
        assert demo.n_steps == 0
        assert demo.unique_actions() == set()
        assert demo.unique_sks() == set()

    def test_with_steps(self):
        steps = [
            DemoStep(frozenset({50, 52}), 3, frozenset({51, 52})),
            DemoStep(frozenset({51, 52}), 5, frozenset({51, 53})),
        ]
        demo = Demonstration(steps=steps, goal_instruction="open the door", success=True)
        assert demo.n_steps == 2
        assert demo.unique_actions() == {3, 5}
        assert demo.unique_sks() == {50, 51, 52, 53}


# ── FewShotLearner ───────────────────────────────────────────────────

class TestFewShotLearner:
    def _make_doorkey_demo(self) -> Demonstration:
        """Simulate a successful DoorKey trajectory:
        1. Navigate (action 2) — no state change in predicates
        2. Pickup key (action 3) — key_present → key_held
        3. Navigate (action 2) — no change
        4. Toggle door (action 5) — door_locked → door_open
        """
        # Start: key visible, door locked, goal present
        s0 = frozenset({SKS_KEY_PRESENT, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        # After navigation, same state predicates
        s1 = frozenset({SKS_KEY_PRESENT, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        # After pickup key: key_held replaces key_present
        s2 = frozenset({SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        # After navigation, same
        s3 = frozenset({SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        # After toggle door: door_open replaces door_locked
        s4 = frozenset({SKS_KEY_HELD, SKS_DOOR_OPEN, SKS_GOAL_PRESENT})

        steps = [
            DemoStep(s0, 2, s1),  # navigate
            DemoStep(s1, 3, s2),  # pickup key
            DemoStep(s2, 2, s3),  # navigate
            DemoStep(s3, 5, s4),  # toggle door
        ]
        return Demonstration(steps=steps, goal_instruction="open the door", success=True)

    def test_extracts_causal_links(self):
        demo = self._make_doorkey_demo()
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations([demo])
        # Should have causal links for pickup(3) and toggle(5)
        links = model.get_causal_links(min_confidence=0.0)
        actions_with_links = {link.action for link in links}
        # At minimum action 3 and 5 should produce non-trivial effects
        assert 3 in actions_with_links or 5 in actions_with_links

    def test_extracts_skills(self):
        demo = self._make_doorkey_demo()
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations([demo])
        skill_names = {s.name for s in library.skills}
        # Should extract pickup_key and toggle_door
        assert "pickup_key" in skill_names, f"Expected pickup_key, got {skill_names}"
        assert "toggle_door" in skill_names, f"Expected toggle_door, got {skill_names}"

    def test_composes_skills(self):
        demo = self._make_doorkey_demo()
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations([demo])
        composites = [s for s in library.skills if s.is_composite]
        assert len(composites) >= 1, "Should compose pickup_key+toggle_door"

    def test_ignores_failed_demos(self):
        demo = self._make_doorkey_demo()
        demo.success = False
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations([demo])
        assert len(library.skills) == 0, "Failed demos should not produce skills"

    def test_multiple_demos_strengthen(self):
        demo1 = self._make_doorkey_demo()
        demo2 = self._make_doorkey_demo()
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations([demo1, demo2])
        # Skills should have higher counts
        pickup = library.get("pickup_key")
        assert pickup is not None
        assert pickup.attempt_count >= 2

    def test_merge_with_existing(self):
        config = CausalAgentConfig(causal_min_observations=1)
        existing_model = CausalWorldModel(config)
        existing_lib = SkillLibrary()
        demo = self._make_doorkey_demo()
        learner = FewShotLearner(min_observations=1)
        model, library = learner.learn_from_demonstrations(
            [demo], existing_model=existing_model, existing_library=existing_lib
        )
        assert model is existing_model
        assert library is existing_lib


# ── DemonstrationRecorder ────────────────────────────────────────────

class _FakeGrid:
    """Minimal fake grid for perception."""
    def __init__(self):
        self.width = 5
        self.height = 5
        self._grid = [None] * 25

    def get(self, i, j):
        return self._grid[j * self.width + i]

    def set(self, i, j, v):
        self._grid[j * self.width + i] = v


class _FakeObj:
    def __init__(self, type_str, color, **kwargs):
        self.type = type_str
        self.color = color
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeEnv:
    """Minimal fake environment for DemonstrationRecorder tests."""
    def __init__(self):
        self.grid = _FakeGrid()
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        self.carrying = None
        self.unwrapped = self

    def step(self, action):
        return {}, 0.0, False, False, {}

    def reset(self, **kwargs):
        return {"mission": "test"}, {}


class TestDemonstrationRecorder:
    def test_records_steps(self):
        from snks.language.grid_perception import GridPerception
        from snks.language.grounding_map import GroundingMap

        env = _FakeEnv()
        gmap = GroundingMap()
        perception = GridPerception(gmap)

        recorder = DemonstrationRecorder(env, perception)
        recorder.start("test instruction")

        # Do some steps
        env.step(2)  # navigate
        env.step(3)  # pickup

        demo = recorder.stop(success=True)
        assert demo.n_steps == 2
        assert demo.goal_instruction == "test instruction"
        assert demo.success is True
        assert demo.steps[0].action == 2
        assert demo.steps[1].action == 3

    def test_restores_original_step(self):
        env = _FakeEnv()
        from snks.language.grid_perception import GridPerception
        from snks.language.grounding_map import GroundingMap
        gmap = GroundingMap()
        perception = GridPerception(gmap)

        recorder = DemonstrationRecorder(env, perception)
        recorder.start("test")
        # During recording, step is wrapped
        env.step(0)
        demo = recorder.stop()
        # After stop, step should work normally (no recording)
        result = env.step(0)
        assert demo.n_steps == 1
        assert isinstance(result, tuple)


# ── FewShotAgent ─────────────────────────────────────────────────────

class TestFewShotAgent:
    def test_learn_from_demos(self):
        """FewShotAgent learns skills from demonstrations."""
        # Create synthetic demo
        s0 = frozenset({SKS_KEY_PRESENT, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        s1 = frozenset({SKS_KEY_HELD, SKS_DOOR_LOCKED, SKS_GOAL_PRESENT})
        s2 = frozenset({SKS_KEY_HELD, SKS_DOOR_OPEN, SKS_GOAL_PRESENT})
        steps = [
            DemoStep(s0, 3, s1),
            DemoStep(s1, 5, s2),
        ]
        demo = Demonstration(steps=steps, goal_instruction="open the door", success=True)

        env = _FakeEnv()
        agent = FewShotAgent(env)
        new_skills = agent.learn_from_demos([demo])

        assert new_skills >= 2  # pickup_key, toggle_door
        assert agent.n_demos_learned == 1
        skill_names = {s.name for s in agent.library.skills}
        assert "pickup_key" in skill_names
        assert "toggle_door" in skill_names
