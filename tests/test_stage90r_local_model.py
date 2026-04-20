from __future__ import annotations

import torch

from snks.agent.stage90r_local_model import (
    LocalActionEvaluator,
    LocalEvaluatorConfig,
    collate_local_samples,
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
