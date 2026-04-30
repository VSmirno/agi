from __future__ import annotations

from snks.agent.vector_mpc_agent import (
    _select_mixed_control_rescue_action,
    _mixed_control_rescue_trigger,
    _should_record_local_counterfactuals,
)
from snks.agent.vector_sim import DynamicEntityState


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
