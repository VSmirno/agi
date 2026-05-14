from __future__ import annotations

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.perception import VisualField
from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.textbook_promoter import TextbookPromoter
from snks.agent.vector_mpc_agent import (
    _select_mixed_control_rescue_action,
    _build_local_counterfactual_outcomes,
    _load_promoted_entities_into_spatial_map,
    _mixed_control_rescue_trigger,
    _should_record_local_counterfactuals,
    _station_spatial_debug,
)
from snks.agent.vector_sim import DynamicEntityState, VectorState


def test_should_record_local_counterfactuals_salient_only_uses_resource_or_threat_salience():
    assert _should_record_local_counterfactuals(
        "salient_only",
        near_concept="tree",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[],
    )
    assert _should_record_local_counterfactuals(
        "salient_only",
        near_concept="empty",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[
            DynamicEntityState(concept_id="zombie", position=(12, 10)),
        ],
    )
    assert not _should_record_local_counterfactuals(
        "salient_only",
        near_concept="empty",
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        observed_dynamic_entities=[],
    )


def test_mixed_control_rescue_trigger_prefers_low_vitals_or_contact_or_stall():
    assert _mixed_control_rescue_trigger(
        body={"health": 3.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        nearest_threat_distances={"zombie": 3, "skeleton": None, "arrow": None},
        actor_non_progress_streak=0,
        low_vitals_threshold=4.0,
        hostile_distance_threshold=1,
        stall_streak_threshold=2,
    ) == "low_vitals"
    assert _mixed_control_rescue_trigger(
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        nearest_threat_distances={"zombie": 1, "skeleton": None, "arrow": None},
        actor_non_progress_streak=0,
        low_vitals_threshold=4.0,
        hostile_distance_threshold=1,
        stall_streak_threshold=2,
    ) == "hostile_contact"
    assert _mixed_control_rescue_trigger(
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        nearest_threat_distances={"zombie": None, "skeleton": None, "arrow": None},
        actor_non_progress_streak=2,
        low_vitals_threshold=4.0,
        hostile_distance_threshold=1,
        stall_streak_threshold=2,
    ) == "repeated_non_progress"
    assert _mixed_control_rescue_trigger(
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        nearest_threat_distances={"zombie": None, "skeleton": None, "arrow": None},
        actor_non_progress_streak=0,
        low_vitals_threshold=4.0,
        hostile_distance_threshold=1,
        stall_streak_threshold=2,
    ) is None


def test_select_mixed_control_rescue_action_prefers_planner_on_disagreement():
    assert _select_mixed_control_rescue_action(
        actor_action="move_down",
        planner_action="move_up",
        rescue_trigger="low_vitals",
        advisory_ranked=[{"action": "move_left"}],
    ) == ("move_up", "planner_override")


def test_select_mixed_control_rescue_action_uses_advisory_override_on_dangerous_consensus():
    assert _select_mixed_control_rescue_action(
        actor_action="move_up",
        planner_action="move_up",
        rescue_trigger="hostile_contact",
        advisory_ranked=[{"action": "move_right"}],
    ) == ("move_right", "advisory_override")


def test_select_mixed_control_rescue_action_skips_consensus_without_alternative():
    assert _select_mixed_control_rescue_action(
        actor_action="move_up",
        planner_action="move_up",
        rescue_trigger="hostile_contact",
        advisory_ranked=[{"action": "move_up"}],
    ) is None


def test_station_spatial_debug_reports_known_station_entries():
    spatial_map = CrafterSpatialMap()
    spatial_map._map[(28, 34)] = ("table", 1.0, 25)
    spatial_map._map[(27, 31)] = ("furnace", 0.8, 3)
    spatial_map._map[(26, 31)] = ("tree", 1.0, 4)

    debug = _station_spatial_debug(
        spatial_map,
        (26, 31),
        concepts=("table", "furnace"),
    )

    assert debug["nearest"]["table"] == {"pos": [28, 34], "dist": 5}
    assert debug["nearest"]["furnace"] == {"pos": [27, 31], "dist": 1}
    assert debug["entries"] == [
        {
            "concept": "furnace",
            "pos": [27, 31],
            "dist": 1,
            "confidence": 0.8,
            "count": 3,
        },
        {
            "concept": "table",
            "pos": [28, 34],
            "dist": 5,
            "confidence": 1.0,
            "count": 25,
        },
    ]
    assert debug["n_entries"] == 2


def test_promoted_entities_do_not_seed_current_episode_spatial_map(tmp_path):
    promoted_path = tmp_path / "seed17_promoted.yaml"
    promoter = TextbookPromoter()
    nodes = [
        {
            "type": "entity_observation",
            "body": {"concept": "table", "position": [26, 31]},
            "provenance": {
                "source": "spatial_map_compiler",
                "observation_count": 10,
                "observed_in_episodes": 1,
                "first_seen_episode": 0,
                "last_seen_episode": 0,
                "confidence": 1.0,
            },
        }
    ]
    promoter.save_nodes(nodes, promoted_path)

    spatial_map = CrafterSpatialMap()
    loaded = _load_promoted_entities_into_spatial_map(
        promoter=promoter,
        promoted_path=promoted_path,
        spatial_map=spatial_map,
    )

    assert loaded == nodes
    assert spatial_map.find_nearest("table", (26, 31)) is None
    assert _station_spatial_debug(
        spatial_map,
        (26, 31),
        concepts=("table", "furnace"),
    )["n_entries"] == 0


def test_build_local_counterfactual_outcomes_emits_feasibility_labels_for_blocked_move():
    model = VectorWorldModel(dim=1024, n_locations=512, seed=42)
    model.proximity_ranges["zombie"] = 1
    for _ in range(10):
        model.learn("zombie", "proximity", {"health": -3})

    spatial_map = CrafterSpatialMap()
    spatial_map.mark_blocked((10, 9))
    state = VectorState(
        inventory={},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        spatial_map=spatial_map,
        dynamic_entities=[
            DynamicEntityState(concept_id="zombie", position=(11, 10), velocity=None)
        ],
    )
    vf = VisualField(detections=[], near_concept="tree", near_similarity=1.0)

    outcomes = _build_local_counterfactual_outcomes(
        model=model,
        state=state,
        vf=vf,
        cache=None,
        vitals=["health"],
        horizon=1,
        enable_post_plan_passive_rollout=False,
    )

    move_up = next(outcome for outcome in outcomes if outcome["action"] == "move_up")

    assert move_up["label"]["effective_displacement_h"] == 0
    assert move_up["label"]["blocked_h"] is True
    assert move_up["label"]["adjacent_hostile_after_h"] is True


def test_build_local_counterfactual_outcomes_keeps_near_concept_for_do_target():
    model = VectorWorldModel(dim=1024, n_locations=512, seed=42)
    for _ in range(10):
        model.learn("tree", "do", {"wood": 1})

    state = VectorState(
        inventory={"wood": 0},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        last_action="move_right",
        spatial_map=CrafterSpatialMap(),
        dynamic_entities=[],
    )
    vf = VisualField(detections=[], near_concept="tree", near_similarity=1.0)

    outcomes = _build_local_counterfactual_outcomes(
        model=model,
        state=state,
        vf=vf,
        cache=None,
        vitals=["health"],
        horizon=1,
        enable_post_plan_passive_rollout=False,
    )

    do_outcome = next(outcome for outcome in outcomes if outcome["action"] == "do")
    assert do_outcome["target"] == "tree"
