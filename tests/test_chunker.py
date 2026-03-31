"""Tests for rule-based sentence chunker (Stage 20)."""

import pytest

from snks.language.chunker import Chunk, RuleBasedChunker


@pytest.fixture
def chunker():
    return RuleBasedChunker()


class TestDetectPattern:

    def test_svo(self, chunker):
        assert chunker.detect_pattern("cat sits on mat") == "svo"

    def test_svo_attr(self, chunker):
        assert chunker.detect_pattern("red cat sits on mat") == "svo_attr"

    def test_minigrid_pick_up(self, chunker):
        assert chunker.detect_pattern("pick up the red key") == "minigrid"

    def test_minigrid_go_to(self, chunker):
        assert chunker.detect_pattern("go to the blue door") == "minigrid"

    def test_minigrid_open(self, chunker):
        assert chunker.detect_pattern("open the yellow door") == "minigrid"


class TestParseSVO:

    def test_basic_svo(self, chunker):
        chunks = chunker.chunk("cat sits on mat")
        roles = {c.role: c.text for c in chunks}
        assert roles["AGENT"] == "cat"
        assert roles["ACTION"] == "sits"
        assert roles["LOCATION"] == "mat"

    def test_svo_no_location(self, chunker):
        chunks = chunker.chunk("dog sees ball")
        roles = {c.role: c.text for c in chunks}
        assert roles["AGENT"] == "dog"
        assert roles["ACTION"] == "sees"
        assert roles["OBJECT"] == "ball"

    def test_svo_with_article(self, chunker):
        chunks = chunker.chunk("the cat sits on the mat")
        roles = {c.role: c.text for c in chunks}
        assert roles["AGENT"] == "cat"
        assert roles["LOCATION"] == "mat"

    def test_svo_in_location(self, chunker):
        chunks = chunker.chunk("cat sits in room")
        roles = {c.role: c.text for c in chunks}
        assert roles["LOCATION"] == "room"


class TestParseSVOAttr:

    def test_basic_attr_svo(self, chunker):
        chunks = chunker.chunk("red cat sits on mat")
        roles = {c.role: c.text for c in chunks}
        assert roles["ATTR"] == "red"
        assert roles["AGENT"] == "cat"
        assert roles["ACTION"] == "sits"
        assert roles["LOCATION"] == "mat"

    def test_blue_dog(self, chunker):
        chunks = chunker.chunk("blue dog runs on floor")
        roles = {c.role: c.text for c in chunks}
        assert roles["ATTR"] == "blue"
        assert roles["AGENT"] == "dog"
        assert roles["ACTION"] == "runs"


class TestParseMiniGrid:

    def test_pick_up(self, chunker):
        chunks = chunker.chunk("pick up the red key")
        roles = {c.role: c.text for c in chunks}
        assert roles["ACTION"] == "pick up"
        assert roles["ATTR"] == "red"
        assert roles["OBJECT"] == "key"

    def test_go_to(self, chunker):
        chunks = chunker.chunk("go to the blue door")
        roles = {c.role: c.text for c in chunks}
        assert roles["ACTION"] == "go to"
        assert roles["OBJECT"] == "door"

    def test_open(self, chunker):
        chunks = chunker.chunk("open the yellow door")
        roles = {c.role: c.text for c in chunks}
        assert roles["ACTION"] == "open"
        assert roles["OBJECT"] == "door"

    def test_toggle_with_color(self, chunker):
        chunks = chunker.chunk("toggle the green box")
        roles = {c.role: c.text for c in chunks}
        assert roles["ACTION"] == "toggle"
        assert roles["ATTR"] == "green"
        assert roles["OBJECT"] == "box"


class TestChunkInterface:

    def test_chunk_returns_list_of_chunks(self, chunker):
        result = chunker.chunk("cat sits on mat")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)

    def test_case_insensitive(self, chunker):
        chunks = chunker.chunk("Cat Sits On Mat")
        roles = {c.role: c.text for c in chunks}
        assert roles["AGENT"] == "cat"
