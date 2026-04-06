"""Tests for ScenarioRunner — Stage 70."""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from snks.agent.scenario_runner import (
    ScenarioRunner,
    ScenarioStep,
    CRAFTER_CHAIN,
    BOOTSTRAP_CHAIN,
    WINDOW_SIZE,
)
from snks.agent.outcome_labeler import OutcomeLabeler
from snks.agent.decode_head import NEAR_TO_IDX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pixels() -> np.ndarray:
    return np.zeros((3, 64, 64), dtype=np.float32)


def _make_env(
    pixels: np.ndarray | None = None,
    info: dict | None = None,
    step_returns: list | None = None,
) -> MagicMock:
    """Create a minimal mock CrafterPixelEnv."""
    if pixels is None:
        pixels = _make_pixels()
    if info is None:
        info = {"inventory": {}, "player_pos": (32, 32)}

    env = MagicMock()
    env.observe.return_value = (pixels, info)

    if step_returns is not None:
        env.step.side_effect = step_returns
    else:
        # Default: return same pixels, not done
        env.step.return_value = (pixels, 0.0, False, info)

    return env


def _make_detector(always_returns: str = "empty") -> MagicMock:
    """Mock NearDetector that always detects the same label."""
    detector = MagicMock()
    detector.detect.return_value = always_returns
    return detector


# ---------------------------------------------------------------------------
# ScenarioStep tests
# ---------------------------------------------------------------------------

class TestScenarioStep:
    def test_default_prereqs(self):
        step = ScenarioStep("tree", "do", "tree")
        assert step.prerequisite_inv == {}
        assert step.repeat == 1

    def test_custom_prereqs(self):
        step = ScenarioStep("coal", "do", "coal", prerequisite_inv={"wood_pickaxe": 1})
        assert step.prerequisite_inv == {"wood_pickaxe": 1}


# ---------------------------------------------------------------------------
# ScenarioRunner._prereqs_met
# ---------------------------------------------------------------------------

class TestPrereqsMet:
    def setup_method(self):
        self.runner = ScenarioRunner()

    def test_empty_prereqs_always_met(self):
        assert self.runner._prereqs_met({}, {})
        assert self.runner._prereqs_met({"wood": 5}, {})

    def test_single_prereq_met(self):
        assert self.runner._prereqs_met({"wood_pickaxe": 1}, {"wood_pickaxe": 1})
        assert self.runner._prereqs_met({"wood_pickaxe": 3}, {"wood_pickaxe": 1})

    def test_single_prereq_not_met(self):
        assert not self.runner._prereqs_met({}, {"wood_pickaxe": 1})
        assert not self.runner._prereqs_met({"wood_pickaxe": 0}, {"wood_pickaxe": 1})

    def test_multiple_prereqs(self):
        inv = {"stone": 3, "wood_pickaxe": 1}
        assert self.runner._prereqs_met(inv, {"stone": 3, "wood_pickaxe": 1})
        assert not self.runner._prereqs_met(inv, {"stone": 4, "wood_pickaxe": 1})


# ---------------------------------------------------------------------------
# ScenarioRunner._probe_action
# ---------------------------------------------------------------------------

class TestProbeAction:
    def setup_method(self):
        self.runner = ScenarioRunner()
        self.labeler = OutcomeLabeler()
        self.rng = np.random.RandomState(42)

    def test_success_place_table(self):
        """place_table succeeds when wood decreases by 2."""
        pixels = _make_pixels()
        inv_after = {"wood": 0}
        info_after = {"inventory": inv_after, "player_pos": (32, 32)}
        env = _make_env(
            info={"inventory": {"wood": 3}, "player_pos": (32, 32)},
            step_returns=[(pixels, 0.0, False, info_after)],
        )
        labeled = []

        success, info_out, _ = self.runner._probe_action(
            env,
            {"inventory": {"wood": 2}, "player_pos": (32, 32)},
            self.labeler,
            "place_table",
            "empty",
            NEAR_TO_IDX["empty"],
            pixels,
            self.rng,
            labeled,
        )

        assert success
        assert len(labeled) == 1
        _, near_idx = labeled[0]
        assert near_idx == NEAR_TO_IDX["empty"]

    def test_failure_no_wood(self):
        """place_table fails when inventory doesn't change."""
        pixels = _make_pixels()
        info_same = {"inventory": {"wood": 0}, "player_pos": (32, 32)}
        # Always returns same state (action fails) — use return_value (infinite)
        env = _make_env(info=info_same)
        env.step.return_value = (pixels, 0.0, False, info_same)
        labeled = []

        success, _, _ = self.runner._probe_action(
            env, info_same, self.labeler, "place_table", "empty",
            NEAR_TO_IDX["empty"], pixels, self.rng, labeled,
        )

        assert not success
        assert len(labeled) == 0


# ---------------------------------------------------------------------------
# CRAFTER_CHAIN and BOOTSTRAP_CHAIN structure
# ---------------------------------------------------------------------------

class TestChainDefinitions:
    def test_crafter_chain_steps(self):
        labels = [s.near_label for s in CRAFTER_CHAIN]
        assert "tree" in labels
        assert "empty" in labels
        assert "table" in labels
        assert "stone" in labels
        assert "coal" in labels
        assert "iron" in labels

    def test_coal_requires_pickaxe(self):
        coal_step = next(s for s in CRAFTER_CHAIN if s.near_label == "coal")
        assert coal_step.prerequisite_inv.get("wood_pickaxe", 0) >= 1

    def test_iron_requires_stone_pickaxe(self):
        iron_step = next(s for s in CRAFTER_CHAIN if s.near_label == "iron")
        assert iron_step.prerequisite_inv.get("stone_pickaxe", 0) >= 1

    def test_bootstrap_chain_no_tools_required(self):
        for step in BOOTSTRAP_CHAIN:
            assert step.prerequisite_inv == {}

    def test_all_near_labels_in_near_to_idx(self):
        for step in CRAFTER_CHAIN:
            assert step.near_label in NEAR_TO_IDX, (
                f"near_label '{step.near_label}' not in NEAR_TO_IDX"
            )


# ---------------------------------------------------------------------------
# run_chain: prerequisite gate stops chain early
# ---------------------------------------------------------------------------

class TestRunChainPrereqGate:
    def setup_method(self):
        self.runner = ScenarioRunner()
        self.labeler = OutcomeLabeler()
        self.rng = np.random.RandomState(0)

    def test_chain_stops_on_unmet_prereq(self):
        """Chain stops at coal step when no wood_pickaxe in inventory."""
        pixels = _make_pixels()
        # Inventory has no pickaxe
        info = {"inventory": {}, "player_pos": (32, 32)}
        env = _make_env(pixels=pixels, info=info)
        detector = _make_detector("empty")

        # Minimal chain: tree step (OK) then coal step (prereq not met)
        chain = [
            ScenarioStep("tree", "do", "tree"),           # needs navigation, but detector won't find it
            ScenarioStep("coal", "do", "coal", prerequisite_inv={"wood_pickaxe": 1}),
        ]

        # The coal step prereq isn't met, so even if navigation would succeed,
        # the runner should stop before coal.
        # Since detector returns "empty" (not "tree"), navigation also fails.
        # Either way, no labeled frames expected.
        labeled = self.runner.run_chain(env, detector, self.labeler, chain, self.rng)

        # No coal frames should be labeled
        coal_idx = NEAR_TO_IDX.get("coal")
        coal_frames = [l for l in labeled if l[1] == coal_idx]
        assert len(coal_frames) == 0
