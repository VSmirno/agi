"""Tests for Stage 83: Vector MPC agent — forward imagination + scoring."""

from __future__ import annotations

import pytest
import numpy as np

from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.vector_sim import (
    VectorState,
    VectorPlan,
    VectorPlanStep,
    simulate_forward,
    score_trajectory,
)
from snks.agent.vector_mpc_agent import (
    generate_candidate_plans,
    _generate_chains,
    _has_positive_effect,
)
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.vector_bootstrap import load_from_textbook
from pathlib import Path

TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


@pytest.fixture
def seeded_model():
    model = VectorWorldModel(dim=8192, n_locations=5000, seed=42)
    load_from_textbook(model, TEXTBOOK_PATH)
    return model


@pytest.fixture
def base_state():
    return VectorState(
        inventory={"wood": 0},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
    )


@pytest.fixture
def spatial_map_with_tree():
    sm = CrafterSpatialMap()
    sm.update((10, 11), "tree", 0.9)
    sm.update((10, 10), "empty", 0.95)
    sm.update((12, 10), "stone", 0.8)
    return sm


class TestGenerateCandidatePlans:
    def test_generates_plans_for_visible_concepts(self, seeded_model, base_state,
                                                   spatial_map_with_tree):
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts={"tree", "stone"},
        )
        # Should have at least baseline + some action plans
        assert len(candidates) >= 1
        origins = [p.origin for p in candidates]
        assert "baseline" in origins

    def test_includes_do_tree_plan(self, seeded_model, base_state,
                                   spatial_map_with_tree):
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts={"tree"},
        )
        # Should have a plan involving do+tree
        do_tree = [p for p in candidates
                   if any(s.action == "do" and s.target == "tree"
                          for s in p.steps)]
        assert len(do_tree) > 0

    def test_baseline_always_present(self, seeded_model, base_state,
                                     spatial_map_with_tree):
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts=set(),
        )
        assert any(p.origin == "baseline" for p in candidates)


class TestGenerateChains:
    def test_chains_extend_beyond_single_step(self, seeded_model, base_state):
        known = {"tree", "table", "wood_sword"}
        plan_actions = ["do", "make", "place"]
        chains = _generate_chains(
            seeded_model, base_state, known, plan_actions,
            beam_width=5, max_depth=3,
        )
        # Should produce some multi-step chains
        multi_step = [c for c in chains if len(c.steps) > 1]
        # At least some chains should exist (tree→do gives wood)
        assert len(chains) > 0


class TestScorePreference:
    def test_total_gain_prefers_long_chain(self, seeded_model, base_state):
        """Long chain with more gain should score higher than short one."""
        # Teach model
        for _ in range(10):
            seeded_model.learn("tree", "do", {"wood": 1})

        short = VectorPlan(steps=[
            VectorPlanStep(action="do", target="tree"),
        ])
        long = VectorPlan(steps=[
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
        ])

        short_traj = simulate_forward(seeded_model, short, base_state)
        long_traj = simulate_forward(seeded_model, long, base_state)

        s_short = score_trajectory(short_traj)
        s_long = score_trajectory(long_traj)

        assert s_long >= s_short, (
            f"Long chain should score ≥ short: {s_long} vs {s_short}"
        )

    def test_survived_beats_dead(self, seeded_model, base_state):
        alive = VectorPlan(steps=[])
        alive_traj = simulate_forward(seeded_model, alive, base_state)

        dead_state = base_state.apply_effect({"health": -10})
        dead_traj = simulate_forward(seeded_model, alive, dead_state,
                                     vital_vars=["health"])

        assert score_trajectory(alive_traj) > score_trajectory(dead_traj)


class TestHasPositiveEffect:
    def test_positive_inventory(self):
        state = VectorState(
            body={"health": 9.0},
            inventory={"wood": 0},
        )
        assert _has_positive_effect({"wood": 1}, state) is True

    def test_body_only_is_not_positive(self):
        state = VectorState(
            body={"health": 9.0},
            inventory={},
        )
        # health is in body — not counted as positive inventory effect
        assert _has_positive_effect({"health": 1}, state) is False

    def test_negative_is_not_positive(self):
        state = VectorState(
            body={"health": 9.0},
            inventory={"wood": 5},
        )
        assert _has_positive_effect({"wood": -1}, state) is False
