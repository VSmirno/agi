"""Grounded QA: question answering from world model (Stage 22).

Three question types routed to pluggable backends:
- Factual  → DCAM causal query
- Simulation → StochasticSimulator
- Reflective → Metacog log

Synthetic scope: backends are dict-based for testing.
Real integration deferred to Stage 24.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from snks.language.chunker import BaseChunker
from snks.language.grounding_map import GroundingMap
from snks.language.templates import (
    factual_answer_template,
    reflective_answer_template,
    simulation_answer_template,
)


class QuestionType(Enum):
    FACTUAL = "factual"
    SIMULATION = "simulation"
    REFLECTIVE = "reflective"


@dataclass
class QAResult:
    """Result from a QA backend query."""

    answer_sks: list[int]
    confidence: float
    source: QuestionType
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class QABackend(Protocol):
    """Protocol for QA knowledge backends."""

    def query(self, roles: dict[str, int]) -> QAResult | None:
        """Query the knowledge source.

        Args:
            roles: resolved role -> sks_id mapping.
                Keys: "AGENT", "ACTION", "OBJECT", "LOCATION", etc.
                Values: integer SKS IDs from GroundingMap.

        Returns:
            QAResult if knowledge found, None otherwise.
        """
        ...


class QuestionClassifier:
    """Rule-based question type classifier.

    Order matters: simulation patterns checked before factual
    (both start with "what").
    """

    _SIMULATION = [
        re.compile(r"what (happens|would happen|will happen) if\b", re.I),
        re.compile(r"what would\b", re.I),
    ]
    _REFLECTIVE = [
        re.compile(r"why (did|are|do|were) (you|i)\b", re.I),
    ]
    _FACTUAL = [
        re.compile(r"(what|who|where|which)\b", re.I),
    ]

    def classify(self, question: str) -> QuestionType:
        q = question.strip()
        for pat in self._SIMULATION:
            if pat.match(q):
                return QuestionType.SIMULATION
        for pat in self._REFLECTIVE:
            if pat.match(q):
                return QuestionType.REFLECTIVE
        for pat in self._FACTUAL:
            if pat.match(q):
                return QuestionType.FACTUAL
        raise ValueError(f"Cannot classify question: {question!r}")


class GroundedQA:
    """Orchestrator: classify -> resolve -> query -> verbalize."""

    def __init__(
        self,
        classifier: QuestionClassifier,
        grounding_map: GroundingMap,
        chunker: BaseChunker,
        factual: QABackend,
        simulation: QABackend,
        reflective: QABackend,
    ) -> None:
        self._classifier = classifier
        self._gmap = grounding_map
        self._chunker = chunker
        self._backends = {
            QuestionType.FACTUAL: factual,
            QuestionType.SIMULATION: simulation,
            QuestionType.REFLECTIVE: reflective,
        }

    # Regex patterns to strip question prefixes, leaving declarative content.
    _STRIP_PATTERNS = [
        re.compile(r"what (happens|would happen|will happen) if (i )?\b", re.I),
        re.compile(r"what would (i )?\b", re.I),
        re.compile(r"why (did|are|do|were) (you|i) \b", re.I),
        re.compile(r"(what|who|where|which) \b", re.I),
    ]

    _ARTICLES = frozenset({"the", "a", "an"})

    def answer(self, question: str) -> str:
        """Full QA pipeline: classify -> resolve -> query -> verbalize."""
        # 1. Classify.
        qtype = self._classifier.classify(question)

        # 2. Strip question prefix, resolve roles.
        content = self._extract_content(question)
        roles = self._resolve_roles(content, qtype)

        # 3. Route to backend.
        result = self._backends[qtype].query(roles)

        # 4. Verbalize.
        if result is None:
            return "I don't know"
        return self._verbalize(result, qtype)

    def _resolve_roles(
        self, content: str, qtype: QuestionType,
    ) -> dict[str, int]:
        """Resolve content words to SKS-ID roles.

        For simulation questions, uses chunker (imperative pattern).
        For factual/reflective, does word-level grounding map scan
        since the chunker expects declarative sentences.
        """
        if qtype == QuestionType.SIMULATION:
            chunks = self._chunker.chunk(content)
            roles: dict[str, int] = {}
            for chunk in chunks:
                sks_id = self._gmap.word_to_sks(chunk.text)
                if sks_id is not None:
                    roles[chunk.role] = sks_id
            return roles

        # Factual / Reflective: word-level scan.
        words = [w for w in content.lower().split() if w not in self._ARTICLES]
        grounded = []
        for w in words:
            sks_id = self._gmap.word_to_sks(w)
            if sks_id is not None:
                grounded.append((w, sks_id))

        roles = {}
        if qtype == QuestionType.FACTUAL:
            # "opens the door" → first=ACTION, second=OBJECT
            if len(grounded) >= 1:
                roles["ACTION"] = grounded[0][1]
            if len(grounded) >= 2:
                roles["OBJECT"] = grounded[1][1]
        elif qtype == QuestionType.REFLECTIVE:
            # "go left" → first=ACTION
            if len(grounded) >= 1:
                roles["ACTION"] = grounded[0][1]
        return roles

    def _extract_content(self, question: str) -> str:
        """Strip question prefix to get declarative content for chunker."""
        q = question.strip().rstrip("?").strip()
        for pat in self._STRIP_PATTERNS:
            m = pat.match(q)
            if m:
                q = q[m.end():]
                break
        return q

    def _verbalize(self, result: QAResult, qtype: QuestionType) -> str:
        words: list[str] = []
        for sks_id in result.answer_sks:
            w = self._gmap.sks_to_word(sks_id)
            if w is not None:
                words.append(w)

        if qtype == QuestionType.FACTUAL:
            return factual_answer_template(words)
        elif qtype == QuestionType.SIMULATION:
            action = result.metadata.get("action", "do something")
            return simulation_answer_template(action, words)
        elif qtype == QuestionType.REFLECTIVE:
            reason = result.metadata.get("reason", "unknown")
            return reflective_answer_template(reason)
        return "I don't know"
