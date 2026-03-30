"""Unit tests for Stage 22: Grounded QA."""

from __future__ import annotations

import pytest
import torch

from snks.language.chunker import RuleBasedChunker
from snks.language.grounding_map import GroundingMap
from snks.language.qa import (
    GroundedQA,
    QAResult,
    QuestionClassifier,
    QuestionType,
)


# --- Synthetic backends ---


class DictFactualBackend:
    """Lookup facts by (ACTION_sks, OBJECT_sks) -> answer SKS IDs."""

    def __init__(self, facts: dict[tuple[int, int], list[int]]) -> None:
        self.facts = facts

    def query(self, roles: dict[str, int]) -> QAResult | None:
        key = (roles.get("ACTION", -1), roles.get("OBJECT", -1))
        if key in self.facts:
            return QAResult(
                answer_sks=self.facts[key],
                confidence=1.0,
                source=QuestionType.FACTUAL,
            )
        return None


class DictSimulationBackend:
    """Lookup simulation by (ACTION_sks, OBJECT_sks) -> effects."""

    def __init__(self, effects: dict[tuple[int, int], dict]) -> None:
        self.effects = effects

    def query(self, roles: dict[str, int]) -> QAResult | None:
        key = (roles.get("ACTION", -1), roles.get("OBJECT", -1))
        if key in self.effects:
            entry = self.effects[key]
            return QAResult(
                answer_sks=entry["answer_sks"],
                confidence=1.0,
                source=QuestionType.SIMULATION,
                metadata={"action": entry.get("action", "")},
            )
        return None


class DictReflectiveBackend:
    """Lookup reflections by ACTION_sks -> reason."""

    def __init__(self, log: dict[int, dict]) -> None:
        self.log = log

    def query(self, roles: dict[str, int]) -> QAResult | None:
        action_sks = roles.get("ACTION", -1)
        if action_sks in self.log:
            entry = self.log[action_sks]
            return QAResult(
                answer_sks=[],
                confidence=1.0,
                source=QuestionType.REFLECTIVE,
                metadata={"reason": entry["reason"]},
            )
        return None


# --- Fixtures ---


@pytest.fixture
def grounding_map() -> GroundingMap:
    gmap = GroundingMap()
    dummy = torch.zeros(64)
    gmap.register("key", 1, dummy)
    gmap.register("door", 2, dummy)
    gmap.register("opens", 3, dummy)
    gmap.register("ball", 4, dummy)
    gmap.register("red", 5, dummy)
    gmap.register("agent", 6, dummy)
    gmap.register("wall", 7, dummy)
    gmap.register("near", 8, dummy)
    gmap.register("pick up", 9, dummy)
    gmap.register("left", 10, dummy)
    gmap.register("go", 11, dummy)
    return gmap


@pytest.fixture
def classifier() -> QuestionClassifier:
    return QuestionClassifier()


@pytest.fixture
def chunker() -> RuleBasedChunker:
    return RuleBasedChunker()


@pytest.fixture
def factual_backend() -> DictFactualBackend:
    return DictFactualBackend({
        (3, 2): [1],       # opens door -> key
        (8, 7): [6],       # near wall -> agent
    })


@pytest.fixture
def simulation_backend() -> DictSimulationBackend:
    return DictSimulationBackend({
        (9, 1): {"answer_sks": [1], "action": "pick up"},   # pick up key -> key_held
    })


@pytest.fixture
def reflective_backend() -> DictReflectiveBackend:
    # Key is ACTION sks_id: "go"=11, "pick up"=9
    return DictReflectiveBackend({
        11: {"reason": "Prediction error was high. I explored left."},
        9: {"reason": "It was my goal to get the key."},
    })


@pytest.fixture
def qa(
    classifier,
    grounding_map,
    chunker,
    factual_backend,
    simulation_backend,
    reflective_backend,
) -> GroundedQA:
    return GroundedQA(
        classifier=classifier,
        grounding_map=grounding_map,
        chunker=chunker,
        factual=factual_backend,
        simulation=simulation_backend,
        reflective=reflective_backend,
    )


# === QuestionClassifier tests ===


class TestQuestionClassifier:
    def test_factual_what(self, classifier):
        assert classifier.classify("What opens the door?") == QuestionType.FACTUAL

    def test_factual_who(self, classifier):
        assert classifier.classify("Who is near the wall?") == QuestionType.FACTUAL

    def test_factual_where(self, classifier):
        assert classifier.classify("Where is the key?") == QuestionType.FACTUAL

    def test_factual_which(self, classifier):
        assert classifier.classify("Which object is red?") == QuestionType.FACTUAL

    def test_simulation_happens(self, classifier):
        assert classifier.classify("What happens if I pick up the key?") == QuestionType.SIMULATION

    def test_simulation_would(self, classifier):
        assert classifier.classify("What would happen if I open the door?") == QuestionType.SIMULATION

    def test_reflective_did(self, classifier):
        assert classifier.classify("Why did you go left?") == QuestionType.REFLECTIVE

    def test_reflective_are(self, classifier):
        assert classifier.classify("Why are you exploring?") == QuestionType.REFLECTIVE

    def test_unclassifiable_raises(self, classifier):
        with pytest.raises(ValueError, match="Cannot classify"):
            classifier.classify("Hello world")

    def test_simulation_before_factual(self, classifier):
        """'What happens if' must be SIMULATION, not FACTUAL."""
        assert classifier.classify("What happens if I drop the ball?") == QuestionType.SIMULATION


# === GroundedQA pipeline tests ===


class TestGroundedQA:
    def test_factual_known(self, qa):
        answer = qa.answer("What opens the door?")
        assert "key" in answer

    def test_factual_unknown(self, qa):
        answer = qa.answer("What color is the ball?")
        assert answer == "I don't know"

    def test_simulation_known(self, qa):
        answer = qa.answer("What happens if I pick up the key?")
        assert "key" in answer

    def test_simulation_unknown(self, qa):
        answer = qa.answer("What happens if I open the door?")
        assert answer == "I don't know"

    def test_reflective_known(self, qa):
        answer = qa.answer("Why did you go left?")
        assert "Prediction error" in answer

    def test_reflective_unknown(self, qa):
        answer = qa.answer("Why did you pick up the ball?")
        assert answer == "I don't know"

    def test_i_dont_know_consistency(self, qa):
        """All unknown answers must be exactly 'I don't know'."""
        unknowns = [
            "What color is the ball?",
            "What happens if I open the door?",
        ]
        for q in unknowns:
            assert qa.answer(q) == "I don't know"


# === Template tests ===


class TestTemplates:
    def test_factual_single(self):
        from snks.language.templates import factual_answer_template
        assert factual_answer_template(["key"]) == "the key"

    def test_factual_multiple(self):
        from snks.language.templates import factual_answer_template
        assert factual_answer_template(["key", "ball"]) == "key and ball"

    def test_factual_empty(self):
        from snks.language.templates import factual_answer_template
        assert factual_answer_template([]) == "I don't know"

    def test_simulation_with_effects(self):
        from snks.language.templates import simulation_answer_template
        assert simulation_answer_template("pick up", ["key"]) == "you will have key"

    def test_simulation_no_effects(self):
        from snks.language.templates import simulation_answer_template
        assert simulation_answer_template("open", []) == "nothing happens"

    def test_reflective(self):
        from snks.language.templates import reflective_answer_template
        reason = "Prediction error was high."
        assert reflective_answer_template(reason) == reason
