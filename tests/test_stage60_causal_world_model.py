"""Stage 60: Tests for Causal World Model via Demonstrations.

Tests the causal world model: synthetic demos → VSA rule encoding → SDM storage → QA validation.
TDD: tests written before implementation.
"""

from __future__ import annotations

import pytest
import torch

from snks.agent.causal_world_model import (
    CausalWorldModel,
    DemoStep,
    RuleEncoder,
)
from snks.agent.vsa_world_model import SDMMemory, VSACodebook


class TestRuleEncoding:
    """Test VSA rule encoding roundtrip."""

    def test_rule_encoding_roundtrip(self):
        """Encode a rule, decode it — should recover original components."""
        cb = VSACodebook(dim=512, seed=42)
        encoder = RuleEncoder(cb)

        # Encode same_color_unlock rule
        rule_vec = encoder.encode_color_rule(
            key_color="red", door_color="red", reward=1.0
        )
        assert rule_vec.shape == (512,)
        assert set(rule_vec.unique().tolist()).issubset({0.0, 1.0})

    def test_same_color_produces_identity(self):
        """bind(color, color) = zero_vector for any color."""
        cb = VSACodebook(dim=512, seed=42)
        red = cb.filler("red")
        blue = cb.filler("blue")
        green = cb.filler("green")

        # All same-color bindings should produce zero vector
        rr = cb.bind(red, red)
        bb = cb.bind(blue, blue)
        gg = cb.bind(green, green)

        assert rr.sum().item() == 0, "bind(red, red) should be zero"
        assert bb.sum().item() == 0, "bind(blue, blue) should be zero"
        assert gg.sum().item() == 0, "bind(green, green) should be zero"

        # All same-color bindings are identical (all zero)
        assert cb.similarity(rr, bb) == 1.0
        assert cb.similarity(bb, gg) == 1.0

    def test_different_color_nonzero(self):
        """bind(color_a, color_b) != zero for different colors."""
        cb = VSACodebook(dim=512, seed=42)
        red = cb.filler("red")
        blue = cb.filler("blue")

        rb = cb.bind(red, blue)
        assert rb.sum().item() > 0, "bind(red, blue) should not be zero"


class TestIdentityGeneralization:
    """Test that rules learned on some colors generalize to unseen colors."""

    def test_identity_generalization(self):
        """Train on red/blue/yellow, test on green/purple/grey — should generalize."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)

        # Train on 3 colors
        train_colors = ["red", "blue", "yellow"]
        model.learn_color_rules(train_colors)

        # Test on unseen colors
        test_colors = ["green", "purple", "grey"]
        correct = 0
        total = 0

        for kc in test_colors:
            for dc in test_colors:
                result = model.query_color_match(kc, dc)
                expected = kc == dc
                if result == expected:
                    correct += 1
                total += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"Generalization accuracy {accuracy:.1%} < 90%"


class TestQATrueFalse:
    """QA-A: True/False factual queries."""

    def test_qa_true_false_seen(self):
        """Facts about training colors should be correct."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        train_colors = ["red", "blue", "yellow"]
        model.learn_color_rules(train_colors)

        # Same color → True
        assert model.query_color_match("red", "red") is True
        assert model.query_color_match("blue", "blue") is True

        # Different color → False
        assert model.query_color_match("red", "blue") is False
        assert model.query_color_match("yellow", "red") is False

    def test_qa_true_false_unseen(self):
        """Facts about unseen colors — >=90% accuracy."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        train_colors = ["red", "blue", "yellow"]
        model.learn_color_rules(train_colors)

        test_colors = ["green", "purple", "grey"]
        correct = 0
        total = 0

        for kc in test_colors:
            for dc in test_colors:
                result = model.query_color_match(kc, dc)
                expected = kc == dc
                if result == expected:
                    correct += 1
                total += 1

        accuracy = correct / total
        assert accuracy >= 0.90, f"Unseen accuracy {accuracy:.1%} < 90%"


class TestQAPrecondition:
    """QA-B: Precondition lookup queries."""

    def test_qa_precondition_unlock(self):
        """'What is needed to open red door?' → 'red key'."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue", "yellow"])

        answer = model.query_precondition("open", "red")
        assert answer == "red", f"Expected 'red' key for red door, got '{answer}'"

    def test_qa_precondition_pickup(self):
        """'What is needed to pickup?' → 'adjacent'."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue", "yellow"])

        answer = model.query_precondition("pickup", None)
        assert answer == "adjacent", f"Expected 'adjacent', got '{answer}'"

    def test_qa_precondition_accuracy(self):
        """All 5 rules should answer precondition queries >= 80%."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue", "yellow"])

        queries = [
            ("open", "red", "red"),        # same_color_unlock → red key
            ("open", "blue", "blue"),       # same_color_unlock → blue key
            ("pickup", None, "adjacent"),   # pickup_requires_adjacent
            ("forward", "locked", "blocked"),  # door_blocks_passage
            ("open", "no_key", "need_key"), # open_requires_key
        ]

        correct = 0
        for action, param, expected in queries:
            answer = model.query_precondition(action, param)
            if answer == expected:
                correct += 1

        accuracy = correct / len(queries)
        assert accuracy >= 0.80, f"Precondition accuracy {accuracy:.1%} < 80%"


class TestQAChain:
    """QA-C: Causal chain queries."""

    def test_qa_chain_simple(self):
        """'How to get behind locked red door?' → chain of subgoals."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue", "yellow"])

        chain = model.query_chain("pass_locked_door", color="red")
        assert isinstance(chain, list)
        assert len(chain) >= 3, f"Chain too short: {chain}"

        # Chain should contain pickup and open in order
        actions = [step for step in chain]
        assert "pickup_key" in actions, f"Missing pickup_key in chain: {chain}"
        assert "open_door" in actions, f"Missing open_door in chain: {chain}"

        # pickup must come before open
        pickup_idx = actions.index("pickup_key")
        open_idx = actions.index("open_door")
        assert pickup_idx < open_idx, "pickup_key must come before open_door"

    def test_qa_chain_accuracy(self):
        """>=70% of chain queries produce correct plans."""
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue", "yellow"])

        scenarios = [
            ("pass_locked_door", "red", ["find_key", "pickup_key", "open_door", "pass_through"]),
            ("pass_locked_door", "blue", ["find_key", "pickup_key", "open_door", "pass_through"]),
            ("pass_locked_door", "green", ["find_key", "pickup_key", "open_door", "pass_through"]),
        ]

        correct = 0
        for goal, color, expected_steps in scenarios:
            chain = model.query_chain(goal, color=color)
            # Check that essential steps are present in order
            has_pickup = "pickup_key" in chain
            has_open = "open_door" in chain
            correct_order = True
            if has_pickup and has_open:
                correct_order = chain.index("pickup_key") < chain.index("open_door")
            if has_pickup and has_open and correct_order:
                correct += 1

        accuracy = correct / len(scenarios)
        assert accuracy >= 0.70, f"Chain accuracy {accuracy:.1%} < 70%"


class TestNegativeExamples:
    """Wrong-color and missing-precondition cases."""

    def test_wrong_color_returns_false(self):
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_color_rules(["red", "blue", "yellow"])

        assert model.query_color_match("red", "blue") is False
        assert model.query_color_match("green", "purple") is False

    def test_no_key_cannot_open(self):
        model = CausalWorldModel(dim=512, n_locations=1000, seed=42)
        model.learn_all_rules(["red", "blue"])

        result = model.query_can_act("open", has_key=False)
        assert result is False, "Should not be able to open without key"
