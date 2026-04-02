"""Tests for Stage 50: LanguageGrounder — text → VSA vector → subgoals."""

from __future__ import annotations

import pytest
import torch

from snks.agent.vsa_world_model import VSACodebook
from snks.language.language_grounder import LanguageGrounder


@pytest.fixture
def codebook():
    return VSACodebook(dim=512, seed=42)


@pytest.fixture
def grounder(codebook):
    return LanguageGrounder(codebook)


# --- Chunker compatibility ---


class TestChunkerIntegration:
    def test_minigrid_instruction(self, grounder):
        chunks = grounder.chunker.chunk("pick up the key")
        roles = [c.role for c in chunks]
        assert "ACTION" in roles
        assert "OBJECT" in roles

    def test_sequential_instruction(self, grounder):
        chunks = grounder.chunker.chunk("pick up the key then open the door")
        roles = [c.role for c in chunks]
        assert roles.count("SEQ_BREAK") >= 1
        assert roles.count("ACTION") >= 2

    def test_attributed_instruction(self, grounder):
        chunks = grounder.chunker.chunk("pick up the red key")
        roles = [c.role for c in chunks]
        assert "ATTR" in roles


# --- Word mapping ---


class TestWordMapping:
    def test_action_words_mapped(self, grounder):
        for action in ["pick up", "open", "go to", "toggle", "drop", "put"]:
            assert grounder._resolve_action(action) is not None, f"Action '{action}' not mapped"

    def test_object_words_mapped(self, grounder):
        for obj in ["key", "door", "goal", "ball", "box"]:
            assert grounder._resolve_object(obj) is not None, f"Object '{obj}' not mapped"

    def test_attr_words_mapped(self, grounder):
        for attr in ["red", "green", "blue", "purple", "yellow", "grey"]:
            assert grounder._resolve_attr(attr) is not None, f"Attr '{attr}' not mapped"

    def test_unknown_action(self, grounder):
        assert grounder._resolve_action("juggle") is None

    def test_unknown_object(self, grounder):
        assert grounder._resolve_object("unicorn") is None


# --- Encode / Decode ---


class TestEncodeDecode:
    def test_pick_up_key_roundtrip(self, grounder):
        vsa = grounder.encode("pick up the key")
        decoded = grounder.decode(vsa)
        assert decoded["action"] == "action_pickup"
        assert decoded["object"] == "object_key"

    def test_open_door_roundtrip(self, grounder):
        vsa = grounder.encode("open the door")
        decoded = grounder.decode(vsa)
        assert decoded["action"] == "action_open"
        assert decoded["object"] == "object_door"

    def test_go_to_goal_roundtrip(self, grounder):
        vsa = grounder.encode("go to the goal")
        decoded = grounder.decode(vsa)
        assert decoded["action"] == "action_goto"
        assert decoded["object"] == "object_goal"

    def test_attributed_instruction(self, grounder):
        vsa = grounder.encode("pick up the red key")
        decoded = grounder.decode(vsa)
        assert decoded["action"] == "action_pickup"
        assert decoded["object"] == "object_key"
        assert decoded.get("attr") == "color_red"

    def test_toggle_door(self, grounder):
        vsa = grounder.encode("toggle the door")
        decoded = grounder.decode(vsa)
        assert decoded["action"] == "action_toggle"
        assert decoded["object"] == "object_door"

    def test_encoded_vector_shape(self, grounder):
        vsa = grounder.encode("pick up the key")
        assert vsa.shape == (512,)
        assert vsa.dtype == torch.float32

    def test_encoded_vector_is_binary(self, grounder):
        vsa = grounder.encode("pick up the key")
        assert torch.all((vsa == 0) | (vsa == 1))

    def test_different_instructions_different_vectors(self, grounder):
        v1 = grounder.encode("pick up the key")
        v2 = grounder.encode("open the door")
        sim = VSACodebook.similarity(v1, v2)
        assert sim < 0.7, f"Vectors too similar: {sim}"


# --- Sequential instructions ---


class TestSequentialInstructions:
    def test_two_step_sequence(self, grounder):
        vectors = grounder.encode_sequence("pick up the key then open the door")
        assert len(vectors) == 2

    def test_three_step_sequence(self, grounder):
        vectors = grounder.encode_sequence(
            "pick up the key then open the door then go to the goal"
        )
        assert len(vectors) == 3

    def test_sequence_decode_first(self, grounder):
        vectors = grounder.encode_sequence("pick up the key then open the door")
        decoded = grounder.decode(vectors[0])
        assert decoded["action"] == "action_pickup"
        assert decoded["object"] == "object_key"

    def test_sequence_decode_second(self, grounder):
        vectors = grounder.encode_sequence("pick up the key then open the door")
        decoded = grounder.decode(vectors[1])
        assert decoded["action"] == "action_open"
        assert decoded["object"] == "object_door"

    def test_single_instruction_returns_one_vector(self, grounder):
        vectors = grounder.encode_sequence("pick up the key")
        assert len(vectors) == 1


# --- Subgoal mapping ---


class TestSubgoalMapping:
    def test_pickup_key(self, grounder):
        subgoals = grounder.to_subgoals("pick up the key")
        assert subgoals == ["pickup_key"]

    def test_open_door(self, grounder):
        subgoals = grounder.to_subgoals("open the door")
        assert subgoals == ["open_door"]

    def test_reach_goal(self, grounder):
        subgoals = grounder.to_subgoals("go to the goal")
        assert subgoals == ["reach_goal"]

    def test_toggle_door_maps_to_open(self, grounder):
        subgoals = grounder.to_subgoals("toggle the door")
        assert subgoals == ["open_door"]

    def test_sequential_subgoals(self, grounder):
        subgoals = grounder.to_subgoals("pick up the key then open the door")
        assert subgoals == ["pickup_key", "open_door"]

    def test_full_doorkey_sequence(self, grounder):
        subgoals = grounder.to_subgoals(
            "pick up the key then open the door then go to the goal"
        )
        assert subgoals == ["pickup_key", "open_door", "reach_goal"]

    def test_unknown_instruction_empty(self, grounder):
        subgoals = grounder.to_subgoals("fly to the moon")
        assert subgoals == []


# --- Gate test: ≥90% accuracy ---


class TestGate:
    INSTRUCTIONS = [
        ("pick up the key", {"action": "action_pickup", "object": "object_key"}),
        ("pick up the red key", {"action": "action_pickup", "object": "object_key", "attr": "color_red"}),
        ("pick up the blue key", {"action": "action_pickup", "object": "object_key", "attr": "color_blue"}),
        ("pick up the green key", {"action": "action_pickup", "object": "object_key", "attr": "color_green"}),
        ("pick up the purple key", {"action": "action_pickup", "object": "object_key", "attr": "color_purple"}),
        ("pick up the yellow key", {"action": "action_pickup", "object": "object_key", "attr": "color_yellow"}),
        ("pick up the grey key", {"action": "action_pickup", "object": "object_key", "attr": "color_grey"}),
        ("open the door", {"action": "action_open", "object": "object_door"}),
        ("toggle the door", {"action": "action_toggle", "object": "object_door"}),
        ("go to the goal", {"action": "action_goto", "object": "object_goal"}),
        ("go to the key", {"action": "action_goto", "object": "object_key"}),
        ("go to the door", {"action": "action_goto", "object": "object_door"}),
        ("drop the key", {"action": "action_drop", "object": "object_key"}),
        ("drop the ball", {"action": "action_drop", "object": "object_ball"}),
        ("put the box", {"action": "action_put", "object": "object_box"}),
        ("pick up the ball", {"action": "action_pickup", "object": "object_ball"}),
        ("pick up the box", {"action": "action_pickup", "object": "object_box"}),
        ("open the red door", {"action": "action_open", "object": "object_door", "attr": "color_red"}),
        ("go to the red key", {"action": "action_goto", "object": "object_key", "attr": "color_red"}),
        ("go to the blue ball", {"action": "action_goto", "object": "object_ball", "attr": "color_blue"}),
    ]

    def test_gate_90_percent(self, grounder):
        correct = 0
        for instruction, expected in self.INSTRUCTIONS:
            vsa = grounder.encode(instruction)
            decoded = grounder.decode(vsa)
            match = True
            for key in ["action", "object"]:
                if decoded.get(key) != expected.get(key):
                    match = False
            # Attr is optional — check only if expected
            if "attr" in expected and decoded.get("attr") != expected["attr"]:
                match = False
            if match:
                correct += 1
        accuracy = correct / len(self.INSTRUCTIONS)
        assert accuracy >= 0.90, f"Gate FAIL: {accuracy:.1%} ({correct}/{len(self.INSTRUCTIONS)})"

    def test_subgoal_gate(self, grounder):
        """All DoorKey-relevant instructions map to correct subgoals."""
        cases = [
            ("pick up the key", ["pickup_key"]),
            ("pick up the red key", ["pickup_key"]),
            ("open the door", ["open_door"]),
            ("toggle the door", ["open_door"]),
            ("go to the goal", ["reach_goal"]),
            ("pick up the key then open the door", ["pickup_key", "open_door"]),
            ("pick up the key then open the door then go to the goal",
             ["pickup_key", "open_door", "reach_goal"]),
        ]
        correct = 0
        for instruction, expected in cases:
            result = grounder.to_subgoals(instruction)
            if result == expected:
                correct += 1
        accuracy = correct / len(cases)
        assert accuracy >= 0.90, f"Subgoal gate FAIL: {accuracy:.1%}"
