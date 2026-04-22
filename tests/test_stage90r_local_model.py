from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.stage90r_train_local_evaluator import _evaluate_ranking
from snks.agent.stage90r_local_policy import (
    TemporalBeliefTracker,
    build_local_observation_package,
)
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
    rank_local_action_candidates,
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
                "temporal_vector": [0.5, 0.0],
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
    assert out["temporal"].shape == (1, 2)
    assert out["action"].tolist() == [3]
    assert out["escape_mask"].tolist() == [0.0]


def test_local_action_evaluator_forward_shapes():
    model = LocalActionEvaluator(LocalEvaluatorConfig(temporal_dim=3))
    class_ids = torch.zeros((2, 7, 9), dtype=torch.long)
    confidences = torch.zeros((2, 7, 9), dtype=torch.float32)
    body = torch.zeros((2, 4), dtype=torch.float32)
    inventory = torch.zeros((2, 12), dtype=torch.float32)
    action = torch.tensor([1, 4], dtype=torch.long)
    temporal = torch.zeros((2, 3), dtype=torch.float32)

    out = model(class_ids, confidences, body, inventory, action, temporal)

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


class _FixedActionEvaluator:
    def __call__(self, class_ids, confidences, body, inventory, action):
        action_id = int(action.item())
        damage = {
            1: 1.2,
            2: 1.1,
            5: 0.1,
            6: 1.3,
        }.get(action_id, 1.0)
        resource = {
            1: 0.0,
            2: 0.0,
            5: 1.0,
            6: 0.0,
        }.get(action_id, 0.0)
        survival_logit = 6.0
        escape = {
            1: 0.2,
            2: 0.1,
            5: 0.0,
            6: 0.1,
        }.get(action_id, 0.0)
        return {
            "pred_damage": torch.tensor([damage], dtype=torch.float32),
            "pred_resource_gain": torch.tensor([resource], dtype=torch.float32),
            "pred_survival_logit": torch.tensor([survival_logit], dtype=torch.float32),
            "pred_escape_delta": torch.tensor([escape], dtype=torch.float32),
        }


def _test_observation(*, adjacent_right: int) -> dict:
    class_ids = [[0] * 9 for _ in range(7)]
    class_ids[3][5] = adjacent_right
    return {
        "viewport_class_ids": class_ids,
        "viewport_confidences": [[0.0] * 9 for _ in range(7)],
        "body_vector": [9.0, 9.0, 9.0, 9.0],
        "inventory_vector": [0] * 12,
    }


def test_rank_local_action_candidates_blocks_do_without_adjacent_object():
    ranked = rank_local_action_candidates(
        evaluator=_FixedActionEvaluator(),
        observation=_test_observation(adjacent_right=0),
        allowed_actions=["move_left", "move_right", "do", "sleep"],
        action_to_idx={"move_left": 1, "move_right": 2, "do": 5, "sleep": 6},
        device="cpu",
    )

    assert [candidate["action"] for candidate in ranked] == [
        "move_right",
        "move_left",
        "sleep",
    ]


def test_rank_local_action_candidates_keeps_do_with_adjacent_object():
    ranked = rank_local_action_candidates(
        evaluator=_FixedActionEvaluator(),
        observation=_test_observation(adjacent_right=2),
        allowed_actions=["move_left", "move_right", "do", "sleep"],
        action_to_idx={"move_left": 1, "move_right": 2, "do": 5, "sleep": 6},
        device="cpu",
    )

    assert ranked[0]["action"] == "do"
    assert "do" in [candidate["action"] for candidate in ranked]


def test_rank_local_action_candidates_blocks_do_for_non_actionable_adjacent_tile():
    ranked = rank_local_action_candidates(
        evaluator=_FixedActionEvaluator(),
        observation=_test_observation(adjacent_right=1),
        allowed_actions=["move_left", "move_right", "do", "sleep"],
        action_to_idx={"move_left": 1, "move_right": 2, "do": 5, "sleep": 6},
        device="cpu",
    )

    assert [candidate["action"] for candidate in ranked] == [
        "move_right",
        "move_left",
        "sleep",
    ]


def test_temporal_belief_tracker_emits_nonzero_context_after_repeated_stall():
    tracker = TemporalBeliefTracker()
    tracker.observe_transition(
        action="do",
        near_concept="tree",
        player_pos_before=(10, 10),
        player_pos_after=(10, 10),
        body_before={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        body_after={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        inventory_before={"wood": 0},
        inventory_after={"wood": 1},
    )
    tracker.observe_transition(
        action="do",
        near_concept="tree",
        player_pos_before=(10, 10),
        player_pos_after=(10, 10),
        body_before={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        body_after={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        inventory_before={"wood": 1},
        inventory_after={"wood": 2},
    )

    context = tracker.build_context(near_concept="tree")

    assert len(context["vector"]) > 10
    assert context["signature"]["prev_action"] == "do"
    assert context["signature"]["near_concept"] == "tree"
    assert context["signature"]["action_streak_bucket"] in {"short", "medium", "long"}


def test_build_local_observation_package_attaches_temporal_context():
    class _VF:
        detections = []

    obs = build_local_observation_package(
        _VF(),
        {"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        {},
        temporal_context={
            "vector": [0.1, 0.2, 0.3],
            "feature_names": ["a", "b", "c"],
            "signature": {"prev_action": "move_up"},
        },
    )

    assert obs["temporal_vector"] == [0.1, 0.2, 0.3]
    assert obs["temporal_signature"] == {"prev_action": "move_up"}


class _TieAwareRankingEvaluator:
    def __call__(self, class_ids, confidences, body, inventory, action):
        batch_size = int(action.shape[0])
        pred_damage = torch.tensor(
            [
                {
                    1: 1.8,
                    2: 1.7,
                    3: 1.3,
                    4: 0.9,
                    6: 1.4,
                }.get(int(action[idx].item()), 1.0)
                for idx in range(batch_size)
            ],
            dtype=torch.float32,
        )
        return {
            "pred_damage": pred_damage,
            "pred_resource_gain": torch.zeros(batch_size, dtype=torch.float32),
            "pred_survival_logit": torch.full((batch_size,), 6.0, dtype=torch.float32),
            "pred_escape_delta": torch.zeros(batch_size, dtype=torch.float32),
        }


def test_evaluate_ranking_counts_tied_targets_as_top1_hits():
    state_samples = [
        {
            "state_id": 1,
            "primary_regime": "neutral",
            "regime_labels": ["neutral"],
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [9.0, 9.0, 9.0, 9.0],
                "inventory_vector": [0] * 12,
            },
            "comparison_coverage": {"n_counterfactual_actions": 2},
            "candidate_actions": [
                {
                    "action": "move_left",
                    "action_index": 1,
                    "label": {"damage_h": 2.0, "resource_gain_h": 0.0, "survived_h": 1.0, "escape_delta_h": None, "health_delta_h": -2.0},
                },
                {
                    "action": "move_right",
                    "action_index": 2,
                    "label": {"damage_h": 2.0, "resource_gain_h": 0.0, "survived_h": 1.0, "escape_delta_h": None, "health_delta_h": -2.0},
                },
                {
                    "action": "move_up",
                    "action_index": 3,
                    "label": {"damage_h": 2.0, "resource_gain_h": 0.0, "survived_h": 1.0, "escape_delta_h": None, "health_delta_h": -2.0},
                },
                {
                    "action": "move_down",
                    "action_index": 4,
                    "label": {"damage_h": 2.0, "resource_gain_h": 0.0, "survived_h": 1.0, "escape_delta_h": None, "health_delta_h": -2.0},
                },
                {
                    "action": "sleep",
                    "action_index": 6,
                    "label": {"damage_h": 2.0, "resource_gain_h": 0.0, "survived_h": 1.0, "escape_delta_h": None, "health_delta_h": -2.0},
                },
            ],
        }
    ]

    report = _evaluate_ranking(_TieAwareRankingEvaluator(), state_samples, device=torch.device("cpu"))

    assert report["overall"]["top1_agreement"] == 1.0
    assert report["overall"]["exact_top1_agreement"] == 0.0
    assert report["overall"]["target_top1_tie_fraction"] == 1.0
    assert report["explanatory_examples"][0]["predicted_best_action"] == "move_down"
    assert report["explanatory_examples"][0]["tied_target_best_actions"] == [
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "sleep",
    ]
