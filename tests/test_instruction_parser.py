"""Unit tests for Stage 24a: Instruction parsing (sequential, spatial, attributes)."""

from __future__ import annotations

import pytest
import torch

from snks.language.chunker import Chunk, RuleBasedChunker
from snks.language.grounding_map import GroundingMap


@pytest.fixture
def chunker() -> RuleBasedChunker:
    return RuleBasedChunker()


@pytest.fixture
def grounding_map() -> GroundingMap:
    gmap = GroundingMap()
    dummy = torch.zeros(64)
    # Objects
    for word, sks in [("key", 1), ("door", 2), ("ball", 3), ("box", 4), ("goal", 5)]:
        gmap.register(word, sks, dummy)
    # Colors
    for word, sks in [("red", 10), ("blue", 11), ("green", 12), ("yellow", 13), ("purple", 14), ("grey", 15)]:
        gmap.register(word, sks, dummy)
    # Actions
    for word, sks in [("pick up", 20), ("go to", 21), ("open", 22), ("put", 23)]:
        gmap.register(word, sks, dummy)
    return gmap


def _roles(chunks: list[Chunk]) -> list[tuple[str, str]]:
    """Extract (text, role) pairs for easy assertion."""
    return [(c.text, c.role) for c in chunks]


# === Sequential tests ===


class TestSequential:
    def test_two_parts(self, chunker):
        chunks = chunker.chunk("pick up the key then open the door")
        roles = _roles(chunks)
        assert ("", "SEQ_BREAK") in roles
        # Before SEQ_BREAK: pick up + key
        idx = [i for i, (t, r) in enumerate(roles) if r == "SEQ_BREAK"][0]
        before = roles[:idx]
        after = roles[idx + 1:]
        assert ("pick up", "ACTION") in before
        assert ("key", "OBJECT") in before
        assert ("open", "ACTION") in after
        assert ("door", "OBJECT") in after

    def test_and_then(self, chunker):
        chunks = chunker.chunk("go to the ball and then pick up the key")
        roles = _roles(chunks)
        assert ("", "SEQ_BREAK") in roles

    def test_three_parts(self, chunker):
        chunks = chunker.chunk("open the door then go to the goal then pick up the key")
        roles = _roles(chunks)
        breaks = [i for i, (t, r) in enumerate(roles) if r == "SEQ_BREAK"]
        assert len(breaks) == 2


# === Spatial tests ===


class TestSpatial:
    def test_basic_spatial(self, chunker):
        chunks = chunker.chunk("put the ball next to the box")
        roles = _roles(chunks)
        assert ("put", "ACTION") in roles
        assert ("ball", "OBJECT") in roles
        assert ("box", "LOCATION") in roles

    def test_spatial_with_attrs(self, chunker):
        chunks = chunker.chunk("put the red ball next to the blue box")
        roles = _roles(chunks)
        assert ("put", "ACTION") in roles
        assert ("red", "ATTR") in roles
        assert ("ball", "OBJECT") in roles
        assert ("blue", "ATTR") in roles
        assert ("box", "LOCATION") in roles


# === Attribute tests ===


class TestAttributes:
    def test_pick_up_red_key(self, chunker):
        chunks = chunker.chunk("pick up the red key")
        roles = _roles(chunks)
        assert ("pick up", "ACTION") in roles
        assert ("red", "ATTR") in roles
        assert ("key", "OBJECT") in roles

    def test_go_to_blue_ball(self, chunker):
        chunks = chunker.chunk("go to the blue ball")
        roles = _roles(chunks)
        assert ("go to", "ACTION") in roles
        assert ("blue", "ATTR") in roles
        assert ("ball", "OBJECT") in roles

    def test_open_yellow_door(self, chunker):
        chunks = chunker.chunk("open the yellow door")
        roles = _roles(chunks)
        assert ("open", "ACTION") in roles
        assert ("yellow", "ATTR") in roles
        assert ("door", "OBJECT") in roles


# === Combined tests ===


class TestCombined:
    def test_sequential_with_attrs(self, chunker):
        chunks = chunker.chunk("pick up the red key then open the yellow door")
        roles = _roles(chunks)
        assert ("", "SEQ_BREAK") in roles
        assert ("red", "ATTR") in roles
        assert ("yellow", "ATTR") in roles
        assert ("key", "OBJECT") in roles
        assert ("door", "OBJECT") in roles


# === Grounding resolve tests ===


class TestGroundingResolve:
    def test_attr_and_object_resolve(self, chunker, grounding_map):
        chunks = chunker.chunk("pick up the red key")
        for c in chunks:
            if c.role == "ATTR":
                assert grounding_map.word_to_sks(c.text) is not None, f"ATTR {c.text!r} not grounded"
            if c.role == "OBJECT":
                assert grounding_map.word_to_sks(c.text) is not None, f"OBJECT {c.text!r} not grounded"

    def test_unknown_color(self, chunker, grounding_map):
        chunks = chunker.chunk("pick up the orange key")
        # "orange" is not in ADJECTIVES, so it won't be ATTR
        # But even if it were, it's not in grounding_map
        assert grounding_map.word_to_sks("orange") is None

    def test_all_babyai_colors(self, grounding_map):
        for color in ["red", "blue", "green", "yellow", "purple", "grey"]:
            assert grounding_map.word_to_sks(color) is not None, f"{color} not grounded"


# === Edge cases ===


class TestEdgeCases:
    def test_then_at_start(self, chunker):
        """'then open the door' should not produce SEQ_BREAK at start."""
        chunks = chunker.chunk("then open the door")
        roles = _roles(chunks)
        # Should parse as single instruction (no leading SEQ_BREAK)
        if roles and roles[0][1] == "SEQ_BREAK":
            pytest.fail("SEQ_BREAK at start")

    def test_empty_second_part(self, chunker):
        """'pick up the key then' should not crash."""
        chunks = chunker.chunk("pick up the key then ")
        roles = _roles(chunks)
        assert ("pick up", "ACTION") in roles
        assert ("key", "OBJECT") in roles
