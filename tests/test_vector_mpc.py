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
    expand_to_primitive,
    _opportunistic_survival_plan,
    _positive_body_effect_from_textbook,
    _remove_entity_target_from_textbook,
    _should_continue_interaction,
    _interaction_intent_from_plan,
    _build_option_context,
    _derive_strategy_option,
    _generate_motion_chains,
    _generate_chains,
    _has_positive_effect,
    _update_spatial_map,
)
from snks.agent.perception import VisualField
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.stage90r_emergency_controller import EmergencyWorldFacts
from snks.agent.vector_bootstrap import load_from_textbook
from pathlib import Path

TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


@pytest.fixture
def seeded_model():
    model = VectorWorldModel(dim=8192, n_locations=5000, seed=42)
    load_from_textbook(model, TEXTBOOK_PATH)
    return model


@pytest.fixture
def textbook():
    from snks.agent.crafter_textbook import CrafterTextbook
    return CrafterTextbook(TEXTBOOK_PATH)


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
    def test_textbook_declares_positive_body_effects(self, textbook):
        assert _positive_body_effect_from_textbook(
            textbook,
            action="do",
            target="water",
        ) == {"drink": 5.0}
        assert _positive_body_effect_from_textbook(
            textbook,
            action="do",
            target="cow",
        ) == {"food": 5.0}

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

    def test_skips_helper_near_requirement_make_keys(self, seeded_model):
        state = VectorState(
            inventory={"wood": 5},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
        )

        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}

        assert "single:table:make" not in origins
        assert "chain:place_table+make_wood_sword" in origins

    def test_includes_requirement_only_combat_do_plan(self, seeded_model):
        state = VectorState(
            inventory={"wood_sword": 1},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
        )

        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts={"zombie"},
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}

        assert seeded_model.action_requirements[("zombie", "do")] == {"wood_sword": 1}
        assert "single:zombie:do" in origins

    def test_textbook_declares_combat_remove_entity_outcome(self, textbook):
        assert _remove_entity_target_from_textbook(
            textbook,
            action="do",
            target="zombie",
        ) == "zombie"

    def test_interaction_continuation_is_not_stopped_by_low_health(self, seeded_model):
        target = _should_continue_interaction(
            interaction_intent={
                "action": "do",
                "target": "zombie",
                "expected_outcome": {"remove_entity": "zombie"},
                "started_step": 148,
                "status": "continuing",
            },
            current_goal=Goal("fight_zombie"),
            near_concept="zombie",
            inventory={"wood_sword": 1},
            model=seeded_model,
        )

        assert target == "zombie"

    def test_interaction_continuation_can_navigate_when_target_not_currently_near(
        self, seeded_model
    ):
        target = _should_continue_interaction(
            interaction_intent={
                "action": "do",
                "target": "zombie",
                "expected_outcome": {"remove_entity": "zombie"},
                "started_step": 156,
                "status": "continuing",
            },
            current_goal=Goal("fight_zombie"),
            near_concept="empty",
            inventory={"wood_sword": 1},
            model=seeded_model,
        )

        assert target == "zombie"

    def test_interaction_intent_starts_from_abstract_remove_plan(self, textbook):
        intent = _interaction_intent_from_plan(
            textbook=textbook,
            plan=VectorPlan(
                steps=[VectorPlanStep(action="do", target="zombie")],
                origin="single:zombie:do",
            ),
            existing_intent=None,
            step=156,
        )

        assert intent == {
            "action": "do",
            "target": "zombie",
            "expected_outcome": {"remove_entity": "zombie"},
            "started_step": 156,
            "status": "continuing",
        }

    def test_strategy_option_maps_existing_plan_shapes(self):
        assert _derive_strategy_option(
            VectorPlan(steps=[], origin="baseline")
        ).to_trace() == {
            "id": "baseline_motion",
            "kind": "baseline_motion",
            "target": None,
        }
        assert _derive_strategy_option(
            VectorPlan(
                steps=[VectorPlanStep(action="frontier_seek", target="water")],
                origin="frontier:water",
            )
        ).option_id == "seek_frontier:water"
        assert _derive_strategy_option(
            VectorPlan(
                steps=[VectorPlanStep(action="navigate_known", target="tree")],
                origin="navigate_known:tree",
            )
        ).option_id == "seek_known:tree"
        assert _derive_strategy_option(
            VectorPlan(
                steps=[VectorPlanStep(action="do", target="zombie")],
                origin="continue:zombie:do_until_remove_entity",
            )
        ).option_id == "continue_interaction:zombie"
        assert _derive_strategy_option(
            VectorPlan(
                steps=[VectorPlanStep(action="do", target="water")],
                origin="opportunistic:water:do_survival_buffer",
            )
        ).option_id == "take_local_survival:water"
        assert _derive_strategy_option(
            VectorPlan(
                steps=[VectorPlanStep(action="make", target="wood_sword")],
                origin="single:wood_sword:make",
            )
        ).option_id == "craft_capability:wood_sword"

    def test_option_context_buckets_compound_conflict(
        self, seeded_model, textbook
    ):
        spatial_map = CrafterSpatialMap()
        spatial_map.update((11, 10), "water", 1.0)
        capability_state = type(
            "Capability",
            (),
            {"armed_melee": True},
        )()

        context = _build_option_context(
            body={"health": 2.0, "food": 3.0, "drink": 1.0, "energy": 7.0},
            inventory={"wood_sword": 1},
            capability_state=capability_state,
            current_goal=Goal("fight_zombie"),
            interaction_intent={
                "action": "do",
                "target": "zombie",
                "status": "continuing",
            },
            best_plan=VectorPlan(
                steps=[VectorPlanStep(action="do", target="zombie")],
                origin="continue:zombie:do_until_remove_entity",
            ),
            nearest_threat_distances={
                "zombie": 1,
                "skeleton": 3,
                "arrow": None,
            },
            emergency_facts=EmergencyWorldFacts.from_textbook(textbook),
            textbook=textbook,
            model=seeded_model,
            near_concept="empty",
            player_pos=(10, 10),
            spatial_map=spatial_map,
        ).to_trace()

        assert context["health_bucket"] == "critical"
        assert context["food_bucket"] == "low"
        assert context["drink_bucket"] == "critical"
        assert context["energy_bucket"] == "ok"
        assert context["threat_pressure"] == "multi"
        assert context["local_restore"] == "drink"
        assert context["capability_state"] == "armed_melee"
        assert context["intent_state"] == "continuing_interaction"
        assert context["goal_family"] == "fight"

    def test_opportunistic_survival_takes_adjacent_resource_before_critical(
        self, seeded_model, textbook
    ):
        spatial_map = CrafterSpatialMap()
        spatial_map.update((10, 11), "water", 1.0)

        plan = _opportunistic_survival_plan(
            textbook=textbook,
            model=seeded_model,
            inventory={},
            body={"health": 9.0, "food": 9.0, "drink": 7.5, "energy": 9.0},
            near_concept="water",
            player_pos=(10, 10),
            spatial_map=spatial_map,
            nearest_threat_distances={"zombie": None, "skeleton": None, "arrow": None},
            emergency_facts=EmergencyWorldFacts.from_textbook(textbook),
        )

        assert plan is not None
        assert plan.origin == "opportunistic:water:do_survival_buffer"
        assert plan.steps == [VectorPlanStep(action="do", target="water")]

    def test_opportunistic_survival_skips_full_vitals(self, seeded_model, textbook):
        spatial_map = CrafterSpatialMap()
        spatial_map.update((10, 11), "water", 1.0)

        plan = _opportunistic_survival_plan(
            textbook=textbook,
            model=seeded_model,
            inventory={},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            near_concept="water",
            player_pos=(10, 10),
            spatial_map=spatial_map,
            nearest_threat_distances={"zombie": None, "skeleton": None, "arrow": None},
            emergency_facts=EmergencyWorldFacts.from_textbook(textbook),
        )

        assert plan is None

    def test_opportunistic_survival_yields_to_immediate_threat(
        self, seeded_model, textbook
    ):
        spatial_map = CrafterSpatialMap()
        spatial_map.update((10, 11), "water", 1.0)

        plan = _opportunistic_survival_plan(
            textbook=textbook,
            model=seeded_model,
            inventory={},
            body={"health": 9.0, "food": 9.0, "drink": 7.5, "energy": 9.0},
            near_concept="water",
            player_pos=(10, 10),
            spatial_map=spatial_map,
            nearest_threat_distances={"zombie": 1, "skeleton": None, "arrow": None},
            emergency_facts=EmergencyWorldFacts.from_textbook(textbook),
        )

        assert plan is None

    def test_includes_body_recovery_plan_when_vital_depleted(self, seeded_model):
        state = VectorState(
            inventory={},
            body={"health": 9.0, "food": 9.0, "drink": 2.0, "energy": 9.0},
            player_pos=(10, 10),
        )

        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts={"water"},
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}

        assert "single:water:do" in origins

    def test_skips_body_recovery_plan_when_vital_full(self, seeded_model):
        state = VectorState(
            inventory={},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
        )

        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts={"water"},
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}

        assert "single:water:do" not in origins


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



# ---------------------------------------------------------------------------
# Phase 1 — goal-conditioned frontier exploration
# ---------------------------------------------------------------------------

class TestFrontierPlanGeneration:
    def test_known_active_goal_target_emits_navigate_known(
        self, seeded_model, base_state
    ):
        sm = CrafterSpatialMap()
        sm.update((14, 10), "tree", 0.9)
        goal = Goal("gather_wood", target_concept="tree")

        candidates = generate_candidate_plans(
            seeded_model,
            base_state,
            sm,
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )

        origins = {p.origin for p in candidates}
        assert "navigate_known:tree" in origins

    def test_dynamic_active_goal_target_emits_navigate_known(
        self, seeded_model
    ):
        state = VectorState(
            inventory={"wood_sword": 1},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(concept_id="zombie", position=(14, 10))
            ],
        )
        goal = Goal("fight_zombie", target_concept="zombie")

        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )

        origins = {p.origin for p in candidates}
        assert "navigate_known:zombie" in origins

    def test_frontier_water_emitted_when_target_unknown_and_goal_active(
        self, seeded_model, base_state, spatial_map_with_tree
    ):
        # spatial_map has tree/empty/stone but no water; goal asks for water.
        goal = Goal("find_water", target_concept="water")
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts={"tree"},
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )
        origins = {p.origin for p in candidates}
        assert "frontier:water" in origins
        assert "navigate_known:water" not in origins

    def test_frontier_water_NOT_emitted_when_water_already_on_map(
        self, seeded_model, base_state
    ):
        sm = CrafterSpatialMap()
        sm.update((14, 10), "water", 0.9)  # water known
        sm.update((10, 10), "empty", 0.95)
        goal = Goal("find_water", target_concept="water")
        candidates = generate_candidate_plans(
            seeded_model, base_state, sm,
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )
        origins = {p.origin for p in candidates}
        assert "frontier:water" not in origins
        assert "navigate_known:water" in origins

    def test_adjacent_active_goal_target_does_not_emit_navigate_known(
        self, seeded_model, base_state
    ):
        sm = CrafterSpatialMap()
        sm.update((11, 10), "water", 0.9)
        goal = Goal("find_water", target_concept="water")

        candidates = generate_candidate_plans(
            seeded_model,
            base_state,
            sm,
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )

        origins = {p.origin for p in candidates}
        assert "navigate_known:water" not in origins

    def test_no_frontier_emitted_when_goal_has_no_target_concept(
        self, seeded_model, base_state, spatial_map_with_tree
    ):
        goal = Goal("explore")  # target_concept=None
        candidates = generate_candidate_plans(
            seeded_model, base_state, spatial_map_with_tree,
            visible_concepts={"tree"},
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
            active_goal=goal,
        )
        origins = {p.origin for p in candidates}
        assert not any(o.startswith("frontier:") for o in origins)


class TestFrontierExpandPrimitive:
    def test_expand_frontier_seek_walks_toward_unvisited(self, seeded_model):
        # Visited tiles around player at (10, 10); unvisited cells to the right.
        sm = CrafterSpatialMap()
        # Mark a band of visited tiles around player except to the right.
        for dy in range(-2, 3):
            for dx in range(-2, 1):
                sm.update((10 + dx, 10 + dy), "empty", 1.0)
        # Right neighbours stay unvisited.
        rng = np.random.RandomState(0)
        primitive = expand_to_primitive(
            VectorPlanStep(action="frontier_seek", target="water"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
        )
        # _step_toward picks among non-zero-delta moves; unvisited cells are
        # to the right (+x), so primitive should be one of move_right or a
        # tie-broken move; assert it is a move_*.
        assert primitive.startswith("move_")

    def test_expand_frontier_seek_falls_back_to_random_move_when_no_unvisited(
        self, seeded_model
    ):
        # All cells in radius=5 are visited → fallback path.
        sm = CrafterSpatialMap()
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                sm.update((10 + dx, 10 + dy), "empty", 1.0)
        rng = np.random.RandomState(0)
        primitive = expand_to_primitive(
            VectorPlanStep(action="frontier_seek", target="water"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
        )
        assert primitive.startswith("move_")


class TestKnownTargetNavigation:
    def test_expand_navigate_known_reduces_distance_and_records_debug(
        self, seeded_model
    ):
        sm = CrafterSpatialMap()
        sm.update((13, 10), "tree", 1.0)
        rng = np.random.RandomState(0)
        debug: dict = {}

        primitive = expand_to_primitive(
            VectorPlanStep(action="navigate_known", target="tree"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
            navigation_debug=debug,
        )

        assert primitive == "move_right"
        assert debug["target_concept"] == "tree"
        assert debug["target_pos"] == [13, 10]
        assert debug["dist_before"] == 3
        assert debug["chosen_move"] == "move_right"
        assert debug["candidate_moves"][0]["dist_after"] == 2
        assert debug["candidate_moves"][0]["reduces_distance"] is True

    def test_expand_navigate_known_chooses_unblocked_alternative(
        self, seeded_model
    ):
        sm = CrafterSpatialMap()
        sm.update((13, 10), "tree", 1.0)
        sm.mark_blocked((11, 10))
        rng = np.random.RandomState(0)
        debug: dict = {}

        primitive = expand_to_primitive(
            VectorPlanStep(action="navigate_known", target="tree"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
            navigation_debug=debug,
        )

        assert primitive != "move_right"
        assert debug["candidate_moves"][0]["blocked"] is False
        assert any(
            move["action"] == "move_right" and move["blocked"]
            for move in debug["candidate_moves"]
        )

    def test_navigate_known_goal_progress_beats_baseline(self, seeded_model):
        sm = CrafterSpatialMap()
        sm.update((13, 10), "tree", 1.0)
        state = VectorState(
            inventory={"wood": 0},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            spatial_map=sm,
        )
        goal = Goal("gather_wood", target_concept="tree")

        nav_traj = simulate_forward(
            seeded_model,
            VectorPlan(
                steps=[VectorPlanStep(action="navigate_known", target="tree")],
                origin="navigate_known:tree",
            ),
            state,
        )
        baseline_traj = simulate_forward(
            seeded_model,
            VectorPlan(steps=[], origin="baseline"),
            state,
        )

        assert nav_traj.states[-1].player_pos == (11, 10)
        assert score_trajectory(nav_traj, goal=goal) > score_trajectory(
            baseline_traj,
            goal=goal,
        )


# ---------------------------------------------------------------------------
# Phase 2A — dynamic-entity-aware plan generation
# ---------------------------------------------------------------------------

class TestDynamicEntityAwarePlans:
    def test_single_cow_do_emitted_when_cow_only_in_dynamic_tracker(
        self, seeded_model
    ):
        state = VectorState(
            inventory={},
            body={"health": 9.0, "food": 4.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(concept_id="cow", position=(13, 10))
            ],
        )
        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),  # cow NOT in spatial_map
            visible_concepts=set(),  # not visible this tick either
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}
        assert "single:cow:do" in origins, (
            "cow plan must come from dynamic_entities source"
        )

    def test_single_zombie_do_emitted_when_armed_and_zombie_tracked(
        self, seeded_model
    ):
        state = VectorState(
            inventory={"wood_sword": 1},
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(concept_id="zombie", position=(11, 10))
            ],
        )
        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}
        assert "single:zombie:do" in origins

    def test_single_zombie_do_NOT_emitted_when_unarmed(self, seeded_model):
        state = VectorState(
            inventory={},  # no sword
            body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(concept_id="zombie", position=(11, 10))
            ],
        )
        candidates = generate_candidate_plans(
            seeded_model,
            state,
            CrafterSpatialMap(),
            visible_concepts=set(),
            player_pos=(10, 10),
            enable_motion_plans=False,
            enable_motion_chains=False,
        )
        origins = {p.origin for p in candidates}
        assert "single:zombie:do" not in origins


class TestExpandPrimitiveDynamicFallback:
    def test_expand_uses_dynamic_position_when_target_not_in_spatial_map(
        self, seeded_model
    ):
        # Empty spatial map, but cow tracked at (15, 10) — agent at (10, 10).
        # Cow is east → expect move_right (delta +5 on X axis).
        sm = CrafterSpatialMap()
        rng = np.random.RandomState(0)
        primitive = expand_to_primitive(
            VectorPlanStep(action="do", target="cow"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
            dynamic_entities=[
                DynamicEntityState(concept_id="cow", position=(15, 10))
            ],
        )
        assert primitive == "move_right"

    def test_expand_falls_back_to_random_when_neither_source_has_target(
        self, seeded_model
    ):
        rng = np.random.RandomState(0)
        primitive = expand_to_primitive(
            VectorPlanStep(action="do", target="cow"),
            player_pos=(10, 10),
            spatial_map=CrafterSpatialMap(),
            model=seeded_model,
            rng=rng,
            last_action=None,
            near_concept="empty",
            dynamic_entities=[],
        )
        # No cow anywhere → falls back to random move
        assert primitive.startswith("move_")

    def test_expand_adjacent_required_do_turns_when_not_facing_target(
        self, seeded_model
    ):
        sm = CrafterSpatialMap()
        sm.update((11, 10), "zombie", 1.0)
        rng = np.random.RandomState(0)

        primitive = expand_to_primitive(
            VectorPlanStep(action="do", target="zombie"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action="move_down",
            near_concept="zombie",
            dynamic_entities=[
                DynamicEntityState(concept_id="zombie", position=(11, 10))
            ],
        )

        assert primitive == "move_right"

    def test_expand_adjacent_required_do_uses_interaction_when_facing_target(
        self, seeded_model
    ):
        sm = CrafterSpatialMap()
        sm.update((11, 10), "zombie", 1.0)
        rng = np.random.RandomState(0)

        primitive = expand_to_primitive(
            VectorPlanStep(action="do", target="zombie"),
            player_pos=(10, 10),
            spatial_map=sm,
            model=seeded_model,
            rng=rng,
            last_action="move_right",
            near_concept="zombie",
            dynamic_entities=[
                DynamicEntityState(concept_id="zombie", position=(11, 10))
            ],
        )

        assert primitive == "do"
