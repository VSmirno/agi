from __future__ import annotations

from snks.agent.perception import VisualField
from snks.agent.stage90r_local_policy import (
    build_local_observation_package,
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
