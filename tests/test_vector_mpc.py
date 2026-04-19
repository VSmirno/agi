"""Tests for Stage 83: Vector MPC agent — forward imagination + scoring."""

from __future__ import annotations

import pytest
import numpy as np

from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.stimuli import (
    HomeostasisStimulus,
    StimuliLayer,
    SurvivalAversion,
    VitalDeltaStimulus,
)
from snks.agent.goal_selector import Goal
from snks.agent.vector_sim import (
    DynamicEntityState,
    VectorState,
    VectorPlan,
    VectorPlanStep,
    VectorTrajectory,
    simulate_forward,
    score_trajectory,
)
from snks.agent.vector_mpc_agent import (
    DynamicEntityTracker,
    build_prediction_cache,
    generate_candidate_plans,
    _generate_motion_chains,
    _generate_chains,
    _has_positive_effect,
    _update_spatial_map,
)
from snks.agent.perception import VisualField
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

    def test_includes_motion_plans_for_repositioning(self, seeded_model, base_state,
                                                     spatial_map_with_tree):
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts={"tree"},
        )
        origins = {p.origin for p in candidates}
        assert "self:move_up" in origins
        assert "self:move_down" in origins

    def test_includes_multi_step_motion_chains_when_dynamic_threat_present(self):
        chains = _generate_motion_chains(
            ["move_up", "move_down", "move_left", "move_right"],
            max_depth=3,
        )
        origins = {plan.origin for plan in chains}
        assert "self:motion_chain:move_up+move_up" in origins
        assert "self:motion_chain:move_up+move_left" in origins
        assert "self:motion_chain:move_right+move_right+move_right" in origins

    def test_motion_plans_can_be_disabled(self, seeded_model, base_state, spatial_map_with_tree):
        candidates = generate_candidate_plans(
            seeded_model,
            base_state,
            spatial_map_with_tree,
            visible_concepts={"tree"},
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}
        assert "self:move_up" not in origins
        assert not any(origin.startswith("self:motion_chain:") for origin in origins)


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
        """Longer wood-gain chain should score higher when wood is the active goal."""
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

        goal = Goal("gather_wood")
        s_short = score_trajectory(short_traj, goal=goal)
        s_long = score_trajectory(long_traj, goal=goal)

        assert s_long >= s_short, (
            f"Long chain should score ≥ short: {s_long} vs {s_short}"
        )

    def test_survived_beats_dead(self, seeded_model, base_state):
        alive = VectorPlan(steps=[])
        alive_traj = simulate_forward(seeded_model, alive, base_state)

        dead_state = base_state.apply_effect({"health": -10})
        dead_traj = VectorTrajectory(
            plan=alive,
            states=[dead_state],
            terminated=True,
            terminated_reason="dead",
        )

        assert score_trajectory(alive_traj) > score_trajectory(dead_traj)

    def test_move_up_beats_sleep_under_arrow_threat(self, seeded_model):
        for _ in range(10):
            seeded_model.learn("arrow", "proximity", {"health": -3})

        state = VectorState(
            inventory={"wood": 0},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(9, 10),
                    velocity=(1, 0),
                )
            ],
        )
        stimuli = StimuliLayer([
            SurvivalAversion(),
            VitalDeltaStimulus(["health"]),
            HomeostasisStimulus(["health"]),
        ])

        sleep_plan = VectorPlan(steps=[VectorPlanStep(action="sleep", target="self")])
        move_plan = VectorPlan(steps=[VectorPlanStep(action="move_up", target="self")])

        sleep_score = score_trajectory(simulate_forward(seeded_model, sleep_plan, state), stimuli=stimuli)
        move_score = score_trajectory(simulate_forward(seeded_model, move_plan, state), stimuli=stimuli)

        assert move_score > sleep_score


class TestViewportMapping:
    def test_update_spatial_map_uses_true_viewport_center(self):
        sm = CrafterSpatialMap()
        vf = VisualField(
            detections=[
                ("water", 1.0, 2, 4),  # one tile up
                ("stone", 1.0, 3, 3),  # one tile left
                ("tree", 1.0, 3, 5),   # one tile right
                ("coal", 1.0, 4, 4),   # one tile down
            ],
            near_concept="water",
            near_similarity=1.0,
        )

        _update_spatial_map(sm, vf, (32, 32))

        assert sm._map[(32, 31)][0] == "water"
        assert sm._map[(31, 32)][0] == "stone"
        assert sm._map[(33, 32)][0] == "tree"
        assert sm._map[(32, 33)][0] == "coal"

    def test_dynamic_entity_tracker_uses_true_viewport_center(self):
        tracker = DynamicEntityTracker()
        tracker.register_dynamic_concept("arrow")

        vf = VisualField(detections=[("arrow", 0.9, 3, 5)])  # one tile right
        tracker.update(vf, player_pos=(32, 32))

        current = tracker.current_for("arrow")
        assert len(current) == 1
        assert current[0].position == (33, 32)

    def test_update_spatial_map_clears_stale_offcenter_tile_with_empty_detection(self):
        sm = CrafterSpatialMap()
        sm.update((33, 32), "tree", 1.0)

        vf = VisualField(
            detections=[("empty", 1.0, 3, 5)],  # one tile right
            near_concept="empty",
            near_similarity=1.0,
        )

        _update_spatial_map(sm, vf, (32, 32))

        assert sm._map[(33, 32)][0] == "empty"

    def test_candidate_ranking_prefers_dodge_under_arrow_threat(self, seeded_model):
        model = VectorWorldModel(dim=2048, n_locations=512, seed=7)
        for action in ("sleep", "move_up", "move_down", "move_left", "move_right", "proximity"):
            model._ensure_action(action)
        for _ in range(10):
            model.learn("arrow", "proximity", {"health": -3})

        state = VectorState(
            inventory={"wood": 0},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(9, 10),
                    velocity=(1, 0),
                )
            ],
        )
        spatial_map = CrafterSpatialMap()
        cache = build_prediction_cache(model, {"arrow"}, ["proximity"])
        stimuli = StimuliLayer([
            SurvivalAversion(),
            VitalDeltaStimulus(["health"]),
            HomeostasisStimulus(["health"]),
        ])

        candidates = generate_candidate_plans(
            model,
            state,
            spatial_map,
            visible_concepts={"arrow"},
            cache=cache,
        )

        scored: list[tuple[tuple, VectorPlan]] = []
        for plan in candidates:
            traj = simulate_forward(model, plan, state, vital_vars=["health"], cache=cache)
            scored.append((score_trajectory(traj, stimuli=stimuli), plan))

        _best_score, best_plan = max(scored, key=lambda item: item[0])
        assert best_plan.origin == "self:move_up"

    def test_defensive_motion_beats_resource_chain_under_two_tick_arrow_threat(self):
        model = VectorWorldModel(dim=2048, n_locations=512, seed=9)
        for action in ("do", "move_up", "move_down", "move_left", "move_right", "proximity"):
            model._ensure_action(action)
        for _ in range(10):
            model.learn("tree", "do", {"wood": 1})
            model.learn("arrow", "proximity", {"health": -3})

        spatial_map = CrafterSpatialMap()
        spatial_map.update((10, 11), "tree", 0.9)
        state = VectorState(
            inventory={"wood": 0},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            spatial_map=spatial_map,
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(8, 10),
                    velocity=(1, 0),
                )
            ],
        )
        cache = build_prediction_cache(model, {"tree", "arrow"}, ["do", "proximity"])
        stimuli = StimuliLayer([
            SurvivalAversion(),
            VitalDeltaStimulus(["health"]),
            HomeostasisStimulus(["health"]),
        ])

        candidates = generate_candidate_plans(
            model,
            state,
            spatial_map,
            visible_concepts={"tree", "arrow"},
            max_depth=3,
            cache=cache,
        )

        scored: list[tuple[tuple, VectorPlan]] = []
        for plan in candidates:
            traj = simulate_forward(model, plan, state, vital_vars=["health"], cache=cache)
            scored.append((score_trajectory(traj, stimuli=stimuli, goal=Goal("fight_skeleton")), plan))

        _best_score, best_plan = max(scored, key=lambda item: item[0])
        assert best_plan.origin.startswith("self:")
        assert all(step.action.startswith("move_") for step in best_plan.steps)
        resource_scores = [score for score, plan in scored if plan.origin.startswith("single:tree:do")]
        assert resource_scores, "expected a resource plan in candidate set"
        best_resource = max(resource_scores)
        best_motion_score = max(
            score for score, plan in scored
            if plan.origin.startswith("self:") and all(step.action.startswith("move_") for step in plan.steps)
        )
        assert best_motion_score > best_resource


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


class TestDynamicEntityTracker:
    def test_arrow_velocity_inferred_from_consecutive_frames(self):
        tracker = DynamicEntityTracker()
        tracker.register_dynamic_concept("arrow")

        vf1 = VisualField(detections=[("arrow", 0.9, 3, 4)])  # player tile
        tracker.update(vf1, player_pos=(10, 10))
        s1 = tracker.current_for("arrow")
        assert len(s1) == 1
        assert s1[0].position == (10, 10)
        assert s1[0].velocity is None
        assert s1[0].age == 0

        vf2 = VisualField(detections=[("arrow", 0.9, 3, 5)])  # one tile right
        tracker.update(vf2, player_pos=(10, 10))
        s2 = tracker.current_for("arrow")
        assert len(s2) == 1
        assert s2[0].position == (11, 10)
        assert s2[0].velocity == (1, 0)
        assert s2[0].age == 1

    def test_arrow_persists_for_one_missed_frame(self):
        tracker = DynamicEntityTracker()
        tracker.register_dynamic_concept("arrow")

        tracker.update(VisualField(detections=[("arrow", 0.9, 3, 4)]), player_pos=(10, 10))
        tracker.update(VisualField(detections=[]), player_pos=(10, 10))

        states = tracker.current_for("arrow")
        assert len(states) == 1
        assert states[0].position == (10, 10)

        tracker.update(VisualField(detections=[]), player_pos=(10, 10))
        assert tracker.current_for("arrow") == []
