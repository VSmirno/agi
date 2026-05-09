from __future__ import annotations

from pathlib import Path

from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.stage90r_emergency_controller import (
    EmergencySafetyController,
    EmergencyWorldFacts,
)


ROOT = Path(__file__).parent.parent


def test_emergency_facts_load_hostiles_and_ranges_from_textbook():
    textbook = CrafterTextbook(ROOT / "configs" / "crafter_textbook.yaml")

    facts = EmergencyWorldFacts.from_textbook(textbook)

    assert facts.source == "textbook.emergency_control"
    assert set(facts.hostile_concepts) == {"zombie", "skeleton", "arrow"}
    assert {"tree", "water", "cow"}.issubset(set(facts.resource_concepts))
    assert facts.threat_ranges["zombie"] == 1
    assert facts.threat_ranges["skeleton"] == 5
    assert facts.threat_ranges["arrow"] == 1
    assert facts.movement_behaviors["zombie"] == "chase_player"


def test_emergency_activation_does_not_require_planner_learner_disagreement():
    facts = EmergencyWorldFacts(
        hostile_concepts=("zombie",),
        threat_ranges={"zombie": 1},
        activation_threshold=1.0,
        source="test",
    )
    controller = EmergencySafetyController(facts=facts)

    features = controller.evaluate(
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        nearest_threat_distances={"zombie": 1},
        actor_non_progress_streak=0,
        planner_action="move_up",
        current_action="move_up",
        learner_action="move_up",
        belief_state_signature={},
        candidate_outcomes=[],
    )

    assert features.activated
    assert features.primary_reason == "hostile_contact"
    assert features.values["planner_disagreement_feature"] is False


def test_emergency_selector_prefers_escape_over_progress_under_threat():
    controller = EmergencySafetyController(
        facts=EmergencyWorldFacts(
            hostile_concepts=("zombie",),
            threat_ranges={"zombie": 1},
            source="test",
        )
    )
    candidate_outcomes = [
        {
            "action": "do",
            "label": {
                "survived_h": True,
                "damage_h": 1.0,
                "health_delta_h": -1.0,
                "escape_delta_h": -1,
                "nearest_hostile_now": 1,
                "nearest_hostile_h": 0,
                "resource_gain_h": 1,
            },
        },
        {
            "action": "move_right",
            "label": {
                "survived_h": True,
                "damage_h": 0.0,
                "health_delta_h": 0.0,
                "escape_delta_h": 2,
                "nearest_hostile_now": 1,
                "nearest_hostile_h": 3,
                "resource_gain_h": 0,
            },
        },
    ]

    selection = controller.select_action(
        current_action="do",
        planner_action="do",
        learner_action="do",
        candidate_outcomes=candidate_outcomes,
        advisory_ranked=[{"action": "do"}],
        allowed_actions=["do", "move_right"],
    )

    assert selection.action == "move_right"
    assert selection.override_source == "independent_emergency_choice"
    assert selection.utility_components["damage_h"] == 0.0


def test_emergency_selector_penalizes_blocked_or_still_adjacent_moves():
    controller = EmergencySafetyController(
        facts=EmergencyWorldFacts(
            hostile_concepts=("zombie",),
            threat_ranges={"zombie": 1},
            source="test",
        )
    )
    candidate_outcomes = [
        {
            "action": "move_left",
            "label": {
                "survived_h": True,
                "damage_h": 0.0,
                "health_delta_h": 0.0,
                "escape_delta_h": 1,
                "nearest_hostile_now": 1,
                "nearest_hostile_h": 1,
                "effective_displacement_h": 0,
                "blocked_h": True,
                "adjacent_hostile_after_h": True,
                "resource_gain_h": 0,
            },
        },
        {
            "action": "move_right",
            "label": {
                "survived_h": True,
                "damage_h": 0.0,
                "health_delta_h": 0.0,
                "escape_delta_h": 1,
                "nearest_hostile_now": 1,
                "nearest_hostile_h": 2,
                "effective_displacement_h": 1,
                "blocked_h": False,
                "adjacent_hostile_after_h": False,
                "resource_gain_h": 0,
            },
        },
    ]

    selection = controller.select_action(
        current_action="move_left",
        planner_action="move_left",
        learner_action="move_left",
        candidate_outcomes=candidate_outcomes,
        advisory_ranked=[{"action": "move_left"}, {"action": "move_right"}],
        allowed_actions=["move_left", "move_right"],
    )

    assert selection.action == "move_right"
    assert selection.utility_components["blocked_h"] is False
    assert selection.utility_components["adjacent_hostile_after_h"] is False
