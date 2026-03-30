"""Unit tests for Verbalizer (Stage 21)."""

from __future__ import annotations

import torch
import pytest

from snks.agent.causal_model import CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.grounding_map import GroundingMap
from snks.language.verbalizer import Verbalizer
from snks.language.templates import describe_template, causal_template, plan_template


# --- Fixtures ---

MINIGRID_ACTIONS = {
    0: "go left",
    1: "go right",
    2: "go forward",
    3: "pick up",
    4: "drop",
    5: "toggle",
    6: "done",
}


def _make_gmap(*pairs: tuple[str, int]) -> GroundingMap:
    """Create a GroundingMap with given (word, sks_id) pairs."""
    gmap = GroundingMap()
    for word, sks_id in pairs:
        gmap.register(word, sks_id, torch.zeros(64))
    return gmap


def _make_causal_model(
    transitions: list[tuple[set[int], int, set[int]]],
    n_repeats: int = 5,
) -> CausalWorldModel:
    """Create CausalWorldModel with known transitions.

    Each transition is (pre_sks, action, post_sks), repeated n_repeats times.
    """
    cfg = CausalAgentConfig(causal_min_observations=2, causal_context_bins=64)
    model = CausalWorldModel(cfg)
    for _ in range(n_repeats):
        for pre, action, post in transitions:
            model.observe_transition(pre, action, post)
    return model


# --- Template tests ---

class TestTemplates:
    def test_describe_empty(self):
        assert describe_template([]) == ""

    def test_describe_single(self):
        assert describe_template(["key"]) == "I see key"

    def test_describe_two(self):
        assert describe_template(["key", "door"]) == "I see key and door"

    def test_describe_multiple(self):
        result = describe_template(["key", "door", "ball"])
        assert result == "I see key, door and ball"

    def test_causal(self):
        result = causal_template("pick up", "key", "key held")
        assert result == "pick up key causes key held"

    def test_plan_empty(self):
        assert plan_template([]) == ""

    def test_plan_single(self):
        result = plan_template([("pick up", "key")])
        assert result == "I need to pick up key"

    def test_plan_multi(self):
        result = plan_template([("pick up", "key"), ("toggle", "door")])
        assert result == "I need to pick up key, then toggle door"


# --- Verbalizer.describe_state tests ---

class TestDescribeState:
    def test_all_grounded(self):
        gmap = _make_gmap(("key", 10), ("door", 20))
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        result = v.describe_state([10, 20])
        assert "key" in result
        assert "door" in result

    def test_partial_grounding(self):
        gmap = _make_gmap(("key", 10))
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        result = v.describe_state([10, 99])  # 99 has no label
        assert "key" in result
        assert "99" not in result

    def test_empty_active(self):
        gmap = _make_gmap(("key", 10))
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        assert v.describe_state([]) == ""

    def test_no_grounded(self):
        gmap = _make_gmap(("key", 10))
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        assert v.describe_state([99, 100]) == ""


# --- Verbalizer.explain_causal tests ---

class TestExplainCausal:
    def test_basic_causal(self):
        gmap = _make_gmap(("key", 10), ("key held", 11))
        # Transition: seeing key (sks 10), pickup → key held (sks 11)
        model = _make_causal_model([({10}, 3, {10, 11})])
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        result = v.explain_causal(10, model)
        # Should mention pick up and key
        assert result != ""

    def test_no_links(self):
        gmap = _make_gmap(("key", 10))
        cfg = CausalAgentConfig(causal_min_observations=2, causal_context_bins=64)
        model = CausalWorldModel(cfg)
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        assert v.explain_causal(10, model) == ""

    def test_unrelated_sks(self):
        gmap = _make_gmap(("key", 10), ("ball", 20))
        model = _make_causal_model([({10}, 3, {10, 11})])
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        # sks 20 not involved in any link
        assert v.explain_causal(20, model) == ""


# --- Verbalizer.verbalize_plan tests ---

class TestVerbalizePlan:
    def test_basic_plan(self):
        gmap = _make_gmap(("key", 10), ("key held", 11), ("door", 20))
        model = _make_causal_model([
            ({10}, 3, {10, 11}),       # pickup key → key held
            ({10, 11}, 5, {10, 11, 20}),  # toggle → door
        ])
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        result = v.verbalize_plan([3, 5], {10}, model)
        assert "I need to" in result
        assert "pick up" in result
        assert "toggle" in result

    def test_empty_plan(self):
        gmap = _make_gmap(("key", 10))
        cfg = CausalAgentConfig(causal_min_observations=2, causal_context_bins=64)
        model = CausalWorldModel(cfg)
        v = Verbalizer(gmap, MINIGRID_ACTIONS)
        assert v.verbalize_plan([], {10}, model) == ""

    def test_unknown_action(self):
        gmap = _make_gmap(("key", 10))
        model = _make_causal_model([({10}, 99, {10, 11})])
        v = Verbalizer(gmap, {99: "zap"})
        result = v.verbalize_plan([99], {10}, model)
        assert "zap" in result
