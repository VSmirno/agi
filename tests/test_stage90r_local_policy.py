from __future__ import annotations

from snks.agent.perception import VisualField
from snks.agent.stage90r_local_policy import (
    BeliefStateEncoder,
    build_local_observation_package,
    build_state_signature,
    build_state_centered_training_examples,
    build_local_training_examples,
    dense_viewport_scene,
)


def test_dense_viewport_scene_preserves_geometry_and_best_confidence():
    vf = VisualField(
        detections=[
            ("tree", 0.7, 1, 2),
            ("cow", 0.9, 3, 4),
            ("tree", 0.8, 1, 2),
        ],
        near_concept="cow",
        near_similarity=0.9,
    )

    class_ids, confidences = dense_viewport_scene(vf)

    assert class_ids[1][2] != 0
    assert confidences[1][2] == 0.8
    assert class_ids[3][4] != 0
    assert confidences[0][0] == 0.0


def test_local_observation_package_does_not_depend_on_near_concept():
    vf = VisualField(
        detections=[("zombie", 0.95, 2, 5)],
        near_concept="tree",
        near_similarity=0.6,
    )

    obs = build_local_observation_package(
        vf,
        body={"health": 4.0, "food": 8.0, "drink": 7.0, "energy": 6.0},
        inventory={"wood": 2, "stone": 0},
    )

    assert "near_concept" not in obs
    assert obs["body_vector"] == [4.0, 8.0, 7.0, 6.0]
    assert obs["inventory_vector"][0] == 2
    assert obs["viewport_class_ids"][2][5] != 0


def test_belief_state_context_excludes_direct_action_identity_features():
    tracker = BeliefStateEncoder()

    initial = tracker.build_context(near_concept="empty")
    assert initial["feature_names"] == [
        "belief_affordance_stability_norm",
        "belief_progress_norm",
        "belief_stall_risk_norm",
        "belief_threat_trend_norm",
        "belief_resource_flow_norm",
        "belief_damage_pressure_norm",
    ]
    assert "prev_action" not in initial["signature"]
    assert "action_streak_bucket" not in initial["signature"]
    assert "stationary_streak_bucket" not in initial["signature"]

    tracker.observe_transition(
        near_concept="empty",
        player_pos_before=(10, 10),
        player_pos_after=(10, 11),
        body_before={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        body_after={"health": 8.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        inventory_before={"wood": 0},
        inventory_after={"wood": 1},
        nearest_threat_distance_before=3,
    )
    next_context = tracker.build_context(near_concept="empty")

    assert len(next_context["vector"]) == 6
    assert next_context["signature"]["affordance_stability_bucket"] == "none"
    assert next_context["signature"]["progress_bucket"] == "low"
    assert next_context["signature"]["stall_bucket"] == "none"
    assert next_context["signature"]["threat_trend_bucket"] == "none"
    assert next_context["signature"]["resource_flow_bucket"] == "medium"
    assert next_context["signature"]["damage_pressure_bucket"] == "medium"


def test_build_local_training_examples_computes_horizon_labels():
    local_trace = [
        {
            "step": 0,
            "action": "move_left",
            "action_index": 1,
            "plan_origin": "baseline",
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [5.0, 7.0, 7.0, 7.0],
                "inventory_vector": [0] * 12,
                "body": {"health": 5.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
                "inventory": {"wood": 0},
            },
            "nearest_threat_distances": {"zombie": 1, "skeleton": None, "arrow": None},
            "done_after_step": False,
        },
        {
            "step": 1,
            "action": "do",
            "action_index": 5,
            "plan_origin": "single:tree:do",
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [3.0, 7.0, 7.0, 7.0],
                "inventory_vector": [1] + [0] * 11,
                "body": {"health": 3.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
                "inventory": {"wood": 1},
            },
            "nearest_threat_distances": {"zombie": 3, "skeleton": None, "arrow": None},
            "done_after_step": True,
        },
    ]

    samples = build_local_training_examples(
        local_trace=local_trace,
        final_body={"health": 3.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
        final_inventory={"wood": 1},
        seed=42,
        episode_id=0,
        horizon=1,
    )

    assert len(samples) == 2
    first = samples[0]
    assert first["label"]["damage_h"] == 2.0
    assert first["label"]["resource_gain_h"] == 1
    assert first["label"]["escape_delta_h"] == 2
    assert first["label"]["survived_h"] is False
    assert first["label"]["progress_delta_h"] == 1.0
    assert first["label"]["stall_risk_h"] == 0.0
    assert first["label"]["affordance_persistence_h"] == 0.0
    assert first["label"]["threat_trend_h"] == 2.0
    assert first["primary_regime"] == "hostile_contact"
    assert "state_signature_key" in first


def test_build_state_centered_training_examples_groups_actions_for_same_signature():
    observation = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [5.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
        "body": {"health": 5.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
        "inventory": {"wood": 0},
    }
    samples = [
        {
            "seed": 1,
            "episode_id": 0,
            "step": 0,
            "action": "move_left",
            "action_index": 1,
            "plan_origin": "baseline",
            "observation": observation,
            "nearest_threat_distances": {"zombie": 2, "skeleton": None, "arrow": None},
            "regime_labels": ["hostile_near"],
            "primary_regime": "hostile_near",
            "state_signature": {
                "center_patch_ids": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "adjacent_tiles": {"left": 0, "right": 0, "up": 0, "down": 0},
                "body_buckets": [5, 7, 7, 7],
                "inventory_presence": [],
                "nearest_hostile_bucket": "near",
                "visible_hostiles": [],
                "resource_tiles": [],
                "regime_labels": ["hostile_near"],
                "primary_regime": "hostile_near",
            },
            "state_signature_key": "shared",
            "label": {
                "health_delta_h": -1.0,
                "damage_h": 1.0,
                "resource_gain_h": 0,
                "inventory_delta_h": {},
                "survived_h": 1.0,
                "escape_delta_h": 1,
                "nearest_hostile_now": 2,
                "nearest_hostile_h": 3,
            },
        },
        {
            "seed": 2,
            "episode_id": 1,
            "step": 4,
            "action": "move_right",
            "action_index": 2,
            "plan_origin": "baseline",
            "observation": observation,
            "nearest_threat_distances": {"zombie": 2, "skeleton": None, "arrow": None},
            "regime_labels": ["hostile_near"],
            "primary_regime": "hostile_near",
            "state_signature": {
                "center_patch_ids": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                "adjacent_tiles": {"left": 0, "right": 0, "up": 0, "down": 0},
                "body_buckets": [5, 7, 7, 7],
                "inventory_presence": [],
                "nearest_hostile_bucket": "near",
                "visible_hostiles": [],
                "resource_tiles": [],
                "regime_labels": ["hostile_near"],
                "primary_regime": "hostile_near",
            },
            "state_signature_key": "shared",
            "label": {
                "health_delta_h": -2.0,
                "damage_h": 2.0,
                "resource_gain_h": 0,
                "inventory_delta_h": {},
                "survived_h": 0.0,
                "escape_delta_h": -1,
                "nearest_hostile_now": 2,
                "nearest_hostile_h": 1,
            },
        },
    ]

    grouped = build_state_centered_training_examples(samples)

    assert len(grouped) == 1
    state_sample = grouped[0]
    assert state_sample["comparison_coverage"]["n_candidate_actions"] == 2
    assert {candidate["action"] for candidate in state_sample["candidate_actions"]} == {
        "move_left",
        "move_right",
    }


def test_build_state_centered_training_examples_prefers_counterfactual_candidate_labels():
    observation = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [5.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
        "body": {"health": 5.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
        "inventory": {"wood": 0},
    }
    grouped = build_state_centered_training_examples(
        [
            {
                "seed": 1,
                "episode_id": 0,
                "step": 0,
                "action": "move_left",
                "action_index": 1,
                "plan_origin": "baseline",
                "observation": observation,
                "nearest_threat_distances": {"zombie": 2, "skeleton": None, "arrow": None},
                "regime_labels": ["hostile_near"],
                "primary_regime": "hostile_near",
                "state_signature": {"shape": "shared"},
                "state_signature_key": "shared",
                "label": {
                    "health_delta_h": -1.0,
                    "damage_h": 1.0,
                    "resource_gain_h": 0,
                    "inventory_delta_h": {},
                    "survived_h": 1.0,
                    "escape_delta_h": 0,
                    "nearest_hostile_now": 2,
                    "nearest_hostile_h": 2,
                },
                "counterfactual_outcomes": [
                    {
                        "action": "move_left",
                        "mean_confidence": 0.8,
                        "label": {
                            "health_delta_h": 0.0,
                            "damage_h": 0.0,
                            "resource_gain_h": 0,
                            "inventory_delta_h": {},
                            "survived_h": 1.0,
                            "escape_delta_h": 2,
                            "nearest_hostile_now": 2,
                            "nearest_hostile_h": 4,
                        },
                    }
                ],
            }
        ]
    )

    candidate = grouped[0]["candidate_actions"][0]
    assert candidate["comparison_priority"] == "counterfactual"
    assert candidate["source"] == "counterfactual_local_rollout"
    assert candidate["label"]["damage_h"] == 0.0
    assert grouped[0]["comparison_coverage"]["n_counterfactual_actions"] == 1


def test_build_state_centered_training_examples_ignores_zero_support_counterfactual_without_observed_fallback():
    observation = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [5.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
        "body": {"health": 5.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
        "inventory": {"wood": 0},
    }

    grouped = build_state_centered_training_examples(
        [
            {
                "seed": 1,
                "episode_id": 0,
                "step": 0,
                "action": "move_left",
                "action_index": 1,
                "plan_origin": "baseline",
                "observation": observation,
                "nearest_threat_distances": {"zombie": None, "skeleton": None, "arrow": None},
                "regime_labels": ["neutral"],
                "primary_regime": "neutral",
                "state_signature": {"shape": "shared"},
                "state_signature_key": "shared",
                "label": {
                    "health_delta_h": 0.0,
                    "damage_h": 0.0,
                    "resource_gain_h": 0,
                    "inventory_delta_h": {},
                    "survived_h": 1.0,
                    "escape_delta_h": None,
                    "nearest_hostile_now": None,
                    "nearest_hostile_h": None,
                },
                "counterfactual_outcomes": [
                    {
                        "action": "do",
                        "mean_confidence": 0.0,
                        "label": {
                            "health_delta_h": 0.0,
                            "damage_h": 0.0,
                            "resource_gain_h": 1,
                            "inventory_delta_h": {"wood": 1},
                            "survived_h": 1.0,
                            "escape_delta_h": None,
                            "nearest_hostile_now": None,
                            "nearest_hostile_h": None,
                        },
                    }
                ],
            }
        ]
    )

    actions = {candidate["action"] for candidate in grouped[0]["candidate_actions"]}
    assert "do" not in actions
    assert grouped[0]["comparison_coverage"]["n_counterfactual_actions"] == 0


def test_build_state_centered_training_examples_falls_back_to_observed_when_counterfactual_is_unsupported():
    observation = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [5.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
        "body": {"health": 5.0, "food": 7.0, "drink": 7.0, "energy": 7.0},
        "inventory": {"wood": 0},
    }

    grouped = build_state_centered_training_examples(
        [
            {
                "seed": 1,
                "episode_id": 0,
                "step": 0,
                "action": "do",
                "action_index": 5,
                "plan_origin": "single:tree:do",
                "observation": observation,
                "nearest_threat_distances": {"zombie": None, "skeleton": None, "arrow": None},
                "regime_labels": ["local_resource_facing"],
                "primary_regime": "local_resource_facing",
                "state_signature": {"shape": "shared"},
                "state_signature_key": "shared",
                "label": {
                    "health_delta_h": 0.0,
                    "damage_h": 0.0,
                    "resource_gain_h": 0,
                    "inventory_delta_h": {},
                    "survived_h": 1.0,
                    "escape_delta_h": None,
                    "nearest_hostile_now": None,
                    "nearest_hostile_h": None,
                },
                "counterfactual_outcomes": [
                    {
                        "action": "do",
                        "mean_confidence": 0.0,
                        "label": {
                            "health_delta_h": 0.0,
                            "damage_h": 0.0,
                            "resource_gain_h": 1,
                            "inventory_delta_h": {"wood": 1},
                            "survived_h": 1.0,
                            "escape_delta_h": None,
                            "nearest_hostile_now": None,
                            "nearest_hostile_h": None,
                        },
                    }
                ],
            }
        ]
    )

    do_candidate = next(
        candidate
        for candidate in grouped[0]["candidate_actions"]
        if candidate["action"] == "do"
    )
    assert do_candidate["comparison_priority"] == "observed"
    assert do_candidate["source"] == "observed_singleton"
    assert do_candidate["label"]["resource_gain_h"] == 0
    assert grouped[0]["comparison_coverage"]["n_counterfactual_actions"] == 0


def test_build_state_signature_distinguishes_relative_geometry():
    left_hostile_obs = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [6.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
    }
    right_hostile_obs = {
        "viewport_class_ids": [[0] * 9 for _ in range(7)],
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [6.0, 7.0, 7.0, 7.0],
        "inventory_vector": [0] * 12,
    }
    left_hostile_obs["viewport_class_ids"][3][1] = 10
    right_hostile_obs["viewport_class_ids"][3][7] = 10

    left_signature = build_state_signature(
        left_hostile_obs,
        {"zombie": 2, "skeleton": None, "arrow": None},
    )
    right_signature = build_state_signature(
        right_hostile_obs,
        {"zombie": 2, "skeleton": None, "arrow": None},
    )

    assert left_signature["visible_hostiles"] == right_signature["visible_hostiles"]
    assert left_signature["hostile_geometry"] != right_signature["hostile_geometry"]
