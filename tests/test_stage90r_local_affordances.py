from __future__ import annotations

from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.stage90r_local_affordances import build_local_affordance_snapshot
from snks.agent.vector_sim import DynamicEntityState


def test_build_local_affordance_snapshot_marks_blocked_and_adjacent_moves():
    spatial_map = CrafterSpatialMap()
    spatial_map.mark_blocked((10, 9))
    snapshot = build_local_affordance_snapshot(
        player_pos=(10, 10),
        spatial_map=spatial_map,
        dynamic_entities=[
            DynamicEntityState(concept_id="zombie", position=(11, 10), velocity=None)
        ],
        last_move="move_right",
    )

    assert snapshot["scene"]["facing_tile"] == [11, 10]
    assert snapshot["scene"]["facing_blocked"] is True
    assert snapshot["scene"]["nearest_hostile_distance"] == 1
    assert snapshot["scene"]["nearest_hostile_direction"] == "right"

    move_up = snapshot["actions"]["move_up"]
    assert move_up["would_move"] is False
    assert move_up["blocked_static"] is True
    assert move_up["blocked_occupied"] is False
    assert move_up["adjacent_hostile_after"] is True
    assert move_up["effective_displacement"] == 0

    move_right = snapshot["actions"]["move_right"]
    assert move_right["would_move"] is False
    assert move_right["blocked_static"] is False
    assert move_right["blocked_occupied"] is True
    assert move_right["contact_after"] is False


def test_build_local_affordance_snapshot_exposes_do_affordance_without_policy():
    spatial_map = CrafterSpatialMap()
    spatial_map.update((11, 10), "tree", 1.0)

    snapshot = build_local_affordance_snapshot(
        player_pos=(10, 10),
        spatial_map=spatial_map,
        dynamic_entities=[
            DynamicEntityState(concept_id="zombie", position=(10, 11), velocity=None)
        ],
        last_move="move_right",
    )

    do_action = snapshot["actions"]["do"]
    assert do_action["do_target_concept"] == "tree"
    assert do_action["do_affordance_present"] is True
    assert do_action["do_under_contact_pressure"] is True
