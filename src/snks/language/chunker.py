"""Rule-based sentence chunker for compositional understanding (Stage 20).

Splits sentences into (text, role) chunks. Three grammar patterns:
- SVO(L): "cat sits on mat"
- ATTR+SVO(L): "red cat sits on mat"
- MiniGrid imperative: "pick up the red key"

This is a scaffold (like sentence-transformers in Stage 19).
Pipeline depends only on BaseChunker.chunk() — swap implementation later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """A text fragment with an assigned semantic role."""

    text: str
    role: str  # "AGENT", "ACTION", "OBJECT", "LOCATION", "GOAL", "ATTR"


class BaseChunker(ABC):
    """Abstract chunker interface. Pipeline depends only on this."""

    @abstractmethod
    def chunk(self, sentence: str) -> list[Chunk]: ...


# --- Vocabularies for rule-based detection ---

ADJECTIVES = frozenset({
    "red", "green", "blue", "purple", "yellow", "grey", "gray",
    "big", "small", "old", "new",
})

# MiniGrid imperative action phrases (checked as prefix).
ACTION_PHRASES = (
    "pick up", "go to", "open", "toggle", "put", "drop",
)

# Prepositions that introduce LOCATION.
LOCATION_PREPS = frozenset({"on", "in", "at", "near", "next to"})

# Articles to strip.
ARTICLES = frozenset({"the", "a", "an"})


class RuleBasedChunker(BaseChunker):
    """Rule-based chunker covering SVO, ATTR+SVO, and MiniGrid patterns."""

    def chunk(self, sentence: str) -> list[Chunk]:
        pattern = self.detect_pattern(sentence)
        if pattern == "minigrid":
            return self._parse_minigrid(sentence)
        if pattern == "svo_attr":
            return self._parse_svo_attr(sentence)
        return self._parse_svo(sentence)

    def detect_pattern(self, sentence: str) -> str:
        """Detect grammar pattern from first word(s)."""
        lower = sentence.lower().strip()
        for phrase in ACTION_PHRASES:
            if lower.startswith(phrase):
                return "minigrid"
        words = lower.split()
        if words and words[0] in ADJECTIVES:
            return "svo_attr"
        return "svo"

    # --- Pattern parsers ---

    def _parse_svo(self, sentence: str) -> list[Chunk]:
        """SVO(L): 'cat sits on mat'."""
        words = self._strip_articles(sentence.lower().split())
        chunks: list[Chunk] = []
        loc_idx = self._find_location_prep(words)

        if loc_idx is not None:
            main_words = words[:loc_idx]
            loc_words = words[loc_idx:]
            # Location preposition removed, keep the noun(s)
            loc_nouns = [w for w in loc_words if w not in LOCATION_PREPS]
            if loc_nouns:
                chunks.append(Chunk(text=" ".join(loc_nouns), role="LOCATION"))
        else:
            main_words = words

        if len(main_words) >= 1:
            chunks.insert(0, Chunk(text=main_words[0], role="AGENT"))
        if len(main_words) >= 2:
            chunks.insert(1, Chunk(text=main_words[1], role="ACTION"))
        if len(main_words) >= 3:
            obj_words = main_words[2:]
            chunks.insert(2, Chunk(text=" ".join(obj_words), role="OBJECT"))

        return chunks

    def _parse_svo_attr(self, sentence: str) -> list[Chunk]:
        """ATTR+SVO(L): 'red cat sits on mat'."""
        words = self._strip_articles(sentence.lower().split())
        chunks: list[Chunk] = []

        # First word is ATTR (already validated by detect_pattern).
        if words:
            chunks.append(Chunk(text=words[0], role="ATTR"))
            rest = " ".join(words[1:])
            chunks.extend(self._parse_svo(rest))

        return chunks

    def _parse_minigrid(self, sentence: str) -> list[Chunk]:
        """MiniGrid imperative: 'pick up the red key'."""
        lower = sentence.lower().strip()
        chunks: list[Chunk] = []

        # Extract action phrase.
        action_text = ""
        remaining = lower
        for phrase in ACTION_PHRASES:
            if lower.startswith(phrase):
                action_text = phrase
                remaining = lower[len(phrase):].strip()
                break

        if action_text:
            chunks.append(Chunk(text=action_text, role="ACTION"))

        # Parse remaining words for ATTR and OBJECT.
        words = self._strip_articles(remaining.split())
        for w in words:
            if w in ADJECTIVES:
                chunks.append(Chunk(text=w, role="ATTR"))
            elif w not in LOCATION_PREPS:
                chunks.append(Chunk(text=w, role="OBJECT"))

        return chunks

    # --- Helpers ---

    @staticmethod
    def _strip_articles(words: list[str]) -> list[str]:
        return [w for w in words if w not in ARTICLES]

    @staticmethod
    def _find_location_prep(words: list[str]) -> int | None:
        """Return index of first location preposition, or None."""
        for i, w in enumerate(words):
            if w in LOCATION_PREPS:
                return i
        return None
