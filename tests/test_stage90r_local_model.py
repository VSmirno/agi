from __future__ import annotations

import torch

from snks.agent.stage90r_local_model import (
    LocalActionEvaluator,
    LocalEvaluatorConfig,
    build_local_advisory_entry,
    compare_stage90r_target_labels,
    collate_local_samples,
    flatten_state_samples,
    stage90r_target_order_key,
    stage90r_target_utility,
    stage90r_action_utility,
    split_samples_by_episode,
)


def test_split_samples_by_episode_keeps_episode_boundaries():
    samples = [
        {"seed": 1, "episode_id": 0},
        {"seed": 1, "episode_id": 0},
        {"seed": 2, "episode_id": 1},
        {"seed": 2, "episode_id": 1},
    ]

    train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    train_keys = {(sample["seed"], sample["episode_id"]) for sample in train}
    valid_keys = {(sample["seed"], sample["episode_id"]) for sample in valid}
    assert train_keys
    assert valid_keys
    assert train_keys.isdisjoint(valid_keys)


def test_collate_local_samples_builds_expected_tensors():
    batch = [
        {
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [1.0, 2.0, 3.0, 4.0],
                "inventory_vector": [0] * 12,
            },
            "action_index": 3,
            "label": {
                "damage_h": 1.0,
                "resource_gain_h": 0,
                "survived_h": True,
                "escape_delta_h": None,
            },
        }
    ]

    out = collate_local_samples(batch)

    assert out["class_ids"].shape == (1, 7, 9)
    assert out["confidences"].shape == (1, 7, 9)
    assert out["body"].shape == (1, 4)
    assert out["inventory"].shape == (1, 12)
    assert out["action"].tolist() == [3]
    assert out["escape_mask"].tolist() == [0.0]


def test_local_action_evaluator_forward_shapes():
    model = LocalActionEvaluator(LocalEvaluatorConfig())
    class_ids = torch.zeros((2, 7, 9), dtype=torch.long)
    confidences = torch.zeros((2, 7, 9), dtype=torch.float32)
    body = torch.zeros((2, 4), dtype=torch.float32)
    inventory = torch.zeros((2, 12), dtype=torch.float32)
    action = torch.tensor([1, 4], dtype=torch.long)

    out = model(class_ids, confidences, body, inventory, action)

    assert out["pred_damage"].shape == (2,)
    assert out["pred_resource_gain"].shape == (2,)
    assert out["pred_survival_logit"].shape == (2,)
    assert out["pred_escape_delta"].shape == (2,)


def test_stage90r_action_utility_prefers_safer_action():
    utility_safe = stage90r_action_utility(
        pred_damage=torch.tensor([0.1]),
        pred_resource_gain=torch.tensor([0.0]),
        pred_survival_logit=torch.tensor([3.0]),
        pred_escape_delta=torch.tensor([1.0]),
    )
    utility_risky = stage90r_action_utility(
        pred_damage=torch.tensor([2.0]),
        pred_resource_gain=torch.tensor([1.0]),
        pred_survival_logit=torch.tensor([-1.0]),
        pred_escape_delta=torch.tensor([-1.0]),
    )
    assert utility_safe.item() > utility_risky.item()


def test_flatten_state_samples_emits_candidate_training_rows():
    state_samples = [
        {
            "state_id": 7,
            "primary_regime": "neutral",
            "regime_labels": ["neutral"],
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [5.0, 6.0, 7.0, 8.0],
                "inventory_vector": [0] * 12,
            },
            "representative_ref": {"seed": 11, "episode_id": 3, "step": 9},
            "candidate_actions": [
                {
                    "action": "move_left",
                    "action_index": 1,
                    "label": {
                        "damage_h": 0.0,
                        "resource_gain_h": 0.0,
                        "survived_h": 1.0,
                        "escape_delta_h": 1.0,
                    },
                    "support_refs": [{"seed": 11, "episode_id": 3, "step": 9}],
                }
            ],
        }
    ]

    rows = flatten_state_samples(state_samples)

    assert len(rows) == 1
    assert rows[0]["seed"] == 11
    assert rows[0]["episode_id"] == 3
    assert rows[0]["action"] == "move_left"
    assert rows[0]["state_id"] == 7


def test_stage90r_target_utility_prefers_safer_label():
    safe = stage90r_target_utility(
        {
            "damage_h": 0.0,
            "resource_gain_h": 0.0,
            "survived_h": 1.0,
            "escape_delta_h": 1.0,
        }
    )
    risky = stage90r_target_utility(
        {
            "damage_h": 2.0,
            "resource_gain_h": 1.0,
            "survived_h": 0.0,
            "escape_delta_h": -1.0,
        }
    )

    assert safe > risky


def test_stage90r_target_order_prefers_survival_then_damage_then_escape():
    safer = {
        "damage_h": 1.0,
        "resource_gain_h": 0.0,
        "survived_h": 1.0,
        "escape_delta_h": 0.0,
        "health_delta_h": -1.0,
    }
    dead_but_rich = {
        "damage_h": 0.0,
        "resource_gain_h": 3.0,
        "survived_h": 0.0,
        "escape_delta_h": 2.0,
        "health_delta_h": -3.0,
    }
    same_survival_lower_damage = {
        "damage_h": 0.0,
        "resource_gain_h": 0.0,
        "survived_h": 1.0,
        "escape_delta_h": -1.0,
        "health_delta_h": 0.0,
    }

    assert stage90r_target_order_key(safer) > stage90r_target_order_key(dead_but_rich)
    assert compare_stage90r_target_labels(safer, dead_but_rich) == 1
    assert compare_stage90r_target_labels(same_survival_lower_damage, safer) == 1


def test_build_local_advisory_entry_reports_planner_rank_and_gap():
    entry = build_local_advisory_entry(
        planner_action="move_right",
        planner_plan_origin="baseline",
        ranked_candidates=[
            {"action": "move_left", "score": 1.4},
            {"action": "move_right", "score": 0.9},
            {"action": "sleep", "score": 0.2},
        ],
        top_k=2,
    )

    assert entry["planner_action"] == "move_right"
    assert entry["planner_rank_by_local_predictor"] == 2
    assert entry["advisory_best_action"] == "move_left"
    assert entry["advisory_agrees_with_planner"] is False
    assert entry["score_gap_to_advisory_best"] == 0.5
    assert len(entry["top_candidates"]) == 2
