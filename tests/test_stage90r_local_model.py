from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.stage90r_train_local_evaluator import (
    _aggregate_teacher_targets_by_signature,
    _apply_episode_split,
    _anti_collapse_gate,
    _checkpoint_priority,
    _evaluate_ranking,
    _gate_policy,
    _set_random_seed,
)
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
    dataset_training_interface,
    flatten_state_samples,
    rank_local_actor_candidates,
    training_rows_from_payload,
    transition_records_to_action_rows,
    stage90r_target_order_key,
    stage90r_target_utility,
    stage90r_action_utility,
    split_samples_by_episode,
    rank_local_action_candidates,
)


def _split_test_sample(
    *,
    seed: int,
    episode_id: int,
    step: int,
    state_key: str,
    regime: str,
    action: str,
    damage: float,
    resource_gain: float,
    survived: float = 1.0,
    escape_delta: float | None = 0.0,
) -> dict[str, object]:
    action_to_index = {
        "move_left": 0,
        "move_right": 1,
        "move_up": 2,
        "move_down": 3,
        "do": 4,
        "sleep": 5,
    }
    return {
        "seed": seed,
        "episode_id": episode_id,
        "step": step,
        "action": action,
        "action_index": action_to_index[action],
        "state_signature_key": state_key,
        "state_signature": {"state": state_key},
        "observation": {
            "viewport_class_ids": [[0] * 9 for _ in range(7)],
            "viewport_confidences": [[0.0] * 9 for _ in range(7)],
            "body_vector": [1.0, 1.0, 1.0, 1.0],
            "inventory_vector": [0] * 12,
            "belief_state_vector": [0.0] * 6,
        },
        "nearest_threat_distances": {},
        "regime_labels": [regime],
        "primary_regime": regime,
        "counterfactual_outcomes": [],
        "label": {
            "health_delta_h": 0.0,
            "damage_h": damage,
            "resource_gain_h": resource_gain,
            "progress_delta_h": resource_gain,
            "stall_risk_h": 0.0,
            "affordance_persistence_h": 1.0 if regime == "local_resource_facing" else 0.0,
            "threat_trend_h": -1.0 if regime.startswith("hostile") else 0.0,
            "inventory_delta_h": {},
            "survived_h": survived,
            "escape_delta_h": escape_delta,
            "escape_valid_fraction": 1.0 if escape_delta is not None else 0.0,
            "nearest_hostile_now": None,
            "nearest_hostile_h": None,
        },
    }


def test_split_samples_by_episode_keeps_episode_boundaries_outside_tiny_fallback():
    samples = [
        {"seed": 1, "episode_id": 0},
        {"seed": 1, "episode_id": 0},
        {"seed": 2, "episode_id": 1},
        {"seed": 2, "episode_id": 1},
        {"seed": 3, "episode_id": 2},
        {"seed": 3, "episode_id": 2},
        {"seed": 4, "episode_id": 3},
        {"seed": 4, "episode_id": 3},
    ]

    train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    train_keys = {(sample["seed"], sample["episode_id"]) for sample in train}
    valid_keys = {(sample["seed"], sample["episode_id"]) for sample in valid}
    assert train_keys
    assert valid_keys
    assert train_keys.isdisjoint(valid_keys)


def test_split_samples_by_episode_uses_state_level_fallback_for_two_episode_threat_holdout():
    samples: list[dict[str, object]] = []

    def add_state(seed: int, episode_id: int, state_name: str, regime: str, winner: str) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
            "move_down": (0.0, 1.0 if winner == "move_down" else 0.0),
        }
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(len(samples) + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )

    add_state(1000, 0, "neutral0", "neutral", "move_right")
    add_state(1000, 0, "neutral1", "neutral", "move_right")
    add_state(1001, 1, "threat0", "hostile_near", "move_down")
    add_state(1001, 1, "threat1", "hostile_near", "move_down")
    add_state(1001, 1, "threat2", "hostile_contact", "move_left")
    add_state(1001, 1, "resource0", "local_resource_facing", "move_up")

    train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    train_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in train}
    valid_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in valid}
    assert (1001, 1) in train_keys
    assert (1001, 1) in valid_keys
    assert any(str(sample["primary_regime"]).startswith("hostile") for sample in train)
    assert any(str(sample["primary_regime"]).startswith("hostile") for sample in valid)


def test_split_samples_by_episode_counts_secondary_resource_labels_in_state_level_fallback():
    samples: list[dict[str, object]] = []

    def add_state(
        seed: int,
        episode_id: int,
        state_name: str,
        regime: str,
        winner: str,
        *,
        extra_regime_labels: tuple[str, ...] = (),
    ) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
            "move_down": (0.0, 1.0 if winner == "move_down" else 0.0),
        }
        start = len(samples)
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(start + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )
        if extra_regime_labels:
            for sample in samples[start:]:
                sample["regime_labels"] = [regime, *extra_regime_labels]

    add_state(1100, 0, "neutral0", "neutral", "move_right")
    add_state(1100, 0, "resource0", "local_resource_facing", "move_up")
    add_state(1101, 1, "threat0", "hostile_near", "move_down")
    add_state(
        1101,
        1,
        "threat_resource0",
        "hostile_contact",
        "move_left",
        extra_regime_labels=("local_resource_facing",),
    )

    train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    assert any("local_resource_facing" in sample["regime_labels"] for sample in train)
    assert any("local_resource_facing" in sample["regime_labels"] for sample in valid)


def test_split_samples_by_episode_uses_stable_non_tail_partition():
    samples = [
        {"seed": seed, "episode_id": seed - 1}
        for seed in range(1, 11)
    ]

    train, valid = split_samples_by_episode(samples, train_ratio=0.8)

    train_keys = {(sample["seed"], sample["episode_id"]) for sample in train}
    valid_keys = {(sample["seed"], sample["episode_id"]) for sample in valid}

    assert valid_keys == {(7, 6), (10, 9)}
    assert train_keys.isdisjoint(valid_keys)
    assert valid_keys != {(9, 8), (10, 9)}


def test_split_samples_by_episode_prefers_validation_with_threat_and_target_diversity():
    samples: list[dict[str, object]] = []

    def add_state(seed: int, episode_id: int, state_name: str, regime: str, winner: str) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
        }
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=step,
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )

    add_state(900, 0, "n0", "neutral", "move_right")
    add_state(901, 1, "t0", "hostile_near", "move_left")
    add_state(902, 2, "n1", "neutral", "move_right")
    add_state(903, 3, "r0", "local_resource_facing", "move_up")
    add_state(904, 4, "n2", "neutral", "move_right")
    add_state(905, 5, "r1", "local_resource_facing", "move_up")

    train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    valid_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in valid}
    assert (901, 1) in valid_keys
    valid_actions = {str(sample["action"]) for sample in valid}
    assert "move_left" in valid_actions
    assert "move_up" in valid_actions


def test_split_samples_by_episode_prefers_richer_threat_support_when_diversity_is_available():
    samples: list[dict[str, object]] = []

    def add_state(seed: int, episode_id: int, state_name: str, regime: str, winner: str) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
        }
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(len(samples) + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )

    add_state(900, 0, "n0", "neutral", "move_right")
    add_state(901, 1, "t_small", "hostile_near", "move_left")
    add_state(902, 2, "n1", "neutral", "move_right")
    add_state(903, 3, "r0", "local_resource_facing", "move_up")
    add_state(904, 4, "t_big0", "hostile_near", "move_left")
    add_state(904, 4, "t_big1", "hostile_near", "move_left")
    add_state(904, 4, "t_big2", "hostile_near", "move_left")
    add_state(905, 5, "r1", "local_resource_facing", "move_up")

    _train, valid = split_samples_by_episode(samples, train_ratio=0.5)

    valid_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in valid}
    assert (904, 4) in valid_keys


def test_split_samples_by_episode_avoids_single_winner_resource_validation_slice_when_diverse_alternative_exists():
    samples: list[dict[str, object]] = []

    def add_state(seed: int, episode_id: int, state_name: str, regime: str, winner: str) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
            "move_down": (0.0, 1.0 if winner == "move_down" else 0.0),
        }
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(len(samples) + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )

    for idx in range(5):
        add_state(910, 0, f"resource{idx}", "local_resource_facing", "move_right")
    add_state(911, 1, "diverse0", "local_resource_facing", "move_right")
    add_state(911, 1, "diverse1", "neutral", "move_left")
    add_state(911, 1, "diverse2", "neutral", "move_up")
    add_state(911, 1, "diverse3", "local_resource_facing", "move_down")
    add_state(912, 2, "filler0", "neutral", "move_left")
    add_state(913, 3, "filler1", "neutral", "move_right")

    _train, valid = split_samples_by_episode(samples, train_ratio=0.75)

    valid_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in valid}
    assert valid_keys == {(911, 1)}


def test_split_samples_by_episode_prefers_supported_diversity_over_tiny_perfect_tie_slice():
    samples: list[dict[str, object]] = []

    def add_state(seed: int, episode_id: int, state_name: str, regime: str, winner: str) -> None:
        action_specs = {
            "move_right": (0.0, 1.0 if winner == "move_right" else 0.0),
            "move_left": (0.0, 1.0 if winner == "move_left" else 0.0),
            "move_up": (0.0, 1.0 if winner == "move_up" else 0.0),
            "move_down": (0.0, 1.0 if winner == "move_down" else 0.0),
        }
        for step, (action, (damage, resource_gain)) in enumerate(action_specs.items()):
            if action != winner:
                damage = 1.0
                resource_gain = 0.0
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(len(samples) + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=damage,
                    resource_gain=resource_gain,
                )
            )

    def add_tied_state(seed: int, episode_id: int, state_name: str, regime: str) -> None:
        for step, action in enumerate(("move_right", "move_left", "move_up", "move_down")):
            samples.append(
                _split_test_sample(
                    seed=seed,
                    episode_id=episode_id,
                    step=(len(samples) + step),
                    state_key=f"{seed}:{state_name}",
                    regime=regime,
                    action=action,
                    damage=0.0,
                    resource_gain=0.0,
                )
            )

    for idx in range(5):
        add_state(920, 0, f"resource{idx}", "local_resource_facing", "move_right")
    add_tied_state(921, 1, "tie0", "neutral")
    add_state(922, 2, "diverse0", "local_resource_facing", "move_right")
    add_state(922, 2, "diverse1", "neutral", "move_left")
    add_state(922, 2, "diverse2", "neutral", "move_up")
    add_state(922, 2, "diverse3", "local_resource_facing", "move_down")
    add_state(923, 3, "filler0", "neutral", "move_left")

    _train, valid = split_samples_by_episode(samples, train_ratio=0.75)

    valid_keys = {(int(sample["seed"]), int(sample["episode_id"])) for sample in valid}
    assert valid_keys == {(922, 2)}


def test_apply_episode_split_uses_state_keys_when_episode_keys_overlap():
    records = [
        {
            "seed": 1,
            "episode_id": 0,
            "step": 0,
            "state_signature_key": "neutral",
        },
        {
            "seed": 1,
            "episode_id": 0,
            "step": 1,
            "state_signature_key": "threat",
        },
    ]

    train_records, valid_records = _apply_episode_split(
        records,
        train_keys={(1, 0)},
        valid_keys={(1, 0)},
        train_state_keys={(1, 0, "neutral", 0)},
        valid_state_keys={(1, 0, "threat", 0)},
    )

    assert [record["state_signature_key"] for record in train_records] == ["neutral"]
    assert [record["state_signature_key"] for record in valid_records] == ["threat"]


def test_aggregate_teacher_targets_by_signature_builds_soft_targets_and_consistency_weights():
    records = [
        {"seed": 1, "episode_id": 0, "step": 0, "state_signature_key": "sigA", "planner_action_index": 2},
        {"seed": 1, "episode_id": 0, "step": 1, "state_signature_key": "sigA", "planner_action_index": 2},
        {"seed": 1, "episode_id": 0, "step": 2, "state_signature_key": "sigA", "planner_action_index": 1},
        {"seed": 1, "episode_id": 0, "step": 3, "state_signature_key": "sigB", "planner_action_index": 0},
    ]

    aggregated = _aggregate_teacher_targets_by_signature(records, n_actions=4)

    sig_a = [record for record in aggregated if record["state_signature_key"] == "sigA"]
    assert len(sig_a) == 3
    for record in sig_a:
        assert record["teacher_action_distribution"] == [0.0, 1 / 3, 2 / 3, 0.0]
        assert record["teacher_target_weight"] == 0.6667

    sig_b = next(record for record in aggregated if record["state_signature_key"] == "sigB")
    assert sig_b["teacher_action_distribution"] == [1.0, 0.0, 0.0, 0.0]
    assert sig_b["teacher_target_weight"] == 1.0


def test_collate_local_samples_preserves_soft_teacher_targets():
    batch = [
        {
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [1.0, 1.0, 1.0, 1.0],
                "inventory_vector": [0] * 12,
                "belief_state_vector": [0.0, 0.0],
            },
            "next_belief_state": {"vector": [0.0, 0.0]},
            "planner_action_index": 2,
            "teacher_policy": "planner",
            "teacher_action_distribution": [0.0, 0.25, 0.75, 0.0],
            "teacher_target_weight": 0.75,
            "action_index": 2,
            "label": {
                "damage_h": 0.0,
                "resource_gain_h": 0.0,
                "survived_h": 1.0,
                "escape_delta_h": None,
            },
        }
    ]

    out = collate_local_samples(batch)

    assert out["teacher_action_distribution"].tolist() == [[0.0, 0.25, 0.75, 0.0]]
    assert out["teacher_target_weight"].tolist() == [0.75]


def test_anti_collapse_gate_marks_bootstrap_threat_gap_as_unsupported_not_fail():
    ranking = {
        "overall": {
            "n_states": 4,
            "dominant_action_share": 0.5,
            "predicted_top1_normalized_entropy": 0.58,
        },
        "regime_metrics": {
            "hostile_contact_or_near": {"n_states": 0, "unique_top1_actions": 0},
            "local_resource_facing": {"n_states": 2, "unique_top1_actions": 1},
        },
    }

    gate = _anti_collapse_gate(ranking, gate_mode="planner_bootstrap")

    assert gate["status"] == "pass_with_unsupported_checks"
    assert gate["passed"] is True
    assert gate["enforced"] is False
    assert gate["blocks_checkpoint_promotion"] is False
    threat_check = next(check for check in gate["checks"] if check["name"] == "threat_slice_diversity")
    assert threat_check["status"] == "unsupported"
    assert threat_check["supported"] is False


def test_anti_collapse_gate_keeps_mixed_control_threat_gap_as_hard_fail():
    ranking = {
        "overall": {
            "n_states": 4,
            "dominant_action_share": 0.5,
            "predicted_top1_normalized_entropy": 0.58,
        },
        "regime_metrics": {
            "hostile_contact_or_near": {"n_states": 0, "unique_top1_actions": 0},
            "local_resource_facing": {"n_states": 2, "unique_top1_actions": 1},
        },
    }

    gate = _anti_collapse_gate(ranking, gate_mode="mixed_control")

    assert gate["status"] == "fail"
    assert gate["passed"] is False
    assert gate["enforced"] is True
    assert gate["blocks_checkpoint_promotion"] is True
    threat_check = next(check for check in gate["checks"] if check["name"] == "threat_slice_diversity")
    assert threat_check["status"] == "fail"
    assert threat_check["supported"] is True


def test_gate_policy_treats_one_epoch_mixed_control_as_smoke_only():
    policy = _gate_policy("mixed_control", epochs=1)

    assert policy["enforced"] is False
    assert policy["saved_outcome"] == "smoke_checkpoint_saved"
    assert policy["policy"] == "mixed_control_smoke_only"


def test_set_random_seed_makes_python_numpy_torch_streams_repeatable():
    _set_random_seed(7)
    first = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
    )

    _set_random_seed(7)
    second = (
        random.random(),
        float(np.random.rand()),
        torch.rand(3).tolist(),
    )

    assert first == second


def test_anti_collapse_gate_keeps_smoke_only_mixed_control_advisory():
    ranking = {
        "overall": {
            "n_states": 6,
            "dominant_action_share": 0.6667,
            "predicted_top1_normalized_entropy": 0.3552,
        },
        "regime_metrics": {
            "hostile_contact_or_near": {"n_states": 6, "unique_top1_actions": 2},
            "local_resource_facing": {"n_states": 1, "unique_top1_actions": 1},
        },
    }

    gate = _anti_collapse_gate(
        ranking,
        gate_mode="mixed_control",
        gate_enforced=False,
    )

    assert gate["status"] == "advisory_fail"
    assert gate["passed"] is False
    assert gate["enforced"] is False
    assert gate["blocks_checkpoint_promotion"] is False


def test_collate_local_samples_builds_expected_tensors():
    batch = [
        {
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [1.0, 2.0, 3.0, 4.0],
                "inventory_vector": [0] * 12,
                "belief_state_vector": [0.5, 0.0],
            },
            "next_belief_state": {"vector": [0.7, 0.2]},
            "action_index": 3,
            "primary_regime": "hostile_near",
            "regime_labels": ["hostile_near", "local_resource_facing"],
            "horizon_outcome": {
                "damage_h": 1.0,
                "resource_gain_h": 0,
                "survived_h": True,
                "escape_delta_h": None,
                "affordance_persistence_h": 1.0,
                "threat_trend_h": -1.0,
            },
        }
    ]

    out = collate_local_samples(batch)

    assert out["class_ids"].shape == (1, 7, 9)
    assert out["confidences"].shape == (1, 7, 9)
    assert out["body"].shape == (1, 4)
    assert out["inventory"].shape == (1, 12)
    assert out["temporal"].shape == (1, 2)
    assert out["next_belief"].shape == (1, 2)
    assert out["next_belief"].tolist() == [[0.699999988079071, 0.20000000298023224]]
    assert out["action"].tolist() == [3]
    assert out["escape_mask"].tolist() == [0.0]
    assert out["danger_pressure"].tolist() == [0.75]
    assert out["resource_opportunity"].tolist() == [0.75]


def test_local_action_evaluator_forward_shapes():
    model = LocalActionEvaluator(LocalEvaluatorConfig(belief_state_dim=3))
    class_ids = torch.zeros((2, 7, 9), dtype=torch.long)
    confidences = torch.zeros((2, 7, 9), dtype=torch.float32)
    body = torch.zeros((2, 4), dtype=torch.float32)
    inventory = torch.zeros((2, 12), dtype=torch.float32)
    action = torch.tensor([1, 4], dtype=torch.long)
    belief_state = torch.zeros((2, 3), dtype=torch.float32)

    out = model(class_ids, confidences, body, inventory, action, belief_state)

    assert out["pred_damage"].shape == (2,)
    assert out["pred_resource_gain"].shape == (2,)
    assert out["pred_survival_logit"].shape == (2,)
    assert out["pred_escape_delta"].shape == (2,)
    assert out["pred_progress_delta"].shape == (2,)
    assert out["pred_stall_risk_logit"].shape == (2,)
    assert out["pred_affordance_persistence_logit"].shape == (2,)
    assert out["pred_threat_trend"].shape == (2,)
    assert out["pred_danger_pressure_logit"].shape == (2,)
    assert out["pred_resource_opportunity_logit"].shape == (2,)
    assert out["pred_next_belief_state"].shape == (2, 3)
    assert out["pred_actor_logits"].shape == (2, model.config.n_actions)


def test_training_rows_from_payload_prefers_primary_transition_records():
    payload = {
        "learner_transition_records": [{"seed": 1, "episode_id": 0, "step": 0}],
        "samples": [{"seed": 2, "episode_id": 1, "step": 1}],
    }

    rows = training_rows_from_payload(payload)

    assert rows == [{"seed": 1, "episode_id": 0, "step": 0}]


def test_training_rows_from_payload_prefers_auxiliary_state_samples_before_legacy_rows():
    payload = {
        "auxiliary_state_samples": [
            {
                "state_id": 4,
                "primary_regime": "neutral",
                "regime_labels": ["neutral"],
                "observation": {
                    "viewport_class_ids": [[0] * 9 for _ in range(7)],
                    "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                    "body_vector": [1.0, 1.0, 1.0, 1.0],
                    "inventory_vector": [0] * 12,
                },
                "candidate_actions": [
                    {
                        "action": "move_right",
                        "action_index": 1,
                        "label": {"damage_h": 0.0, "resource_gain_h": 1.0, "survived_h": 1.0},
                        "support_refs": [{"seed": 7, "episode_id": 3, "step": 2}],
                    }
                ],
            }
        ],
        "samples": [{"seed": 99, "episode_id": 8, "step": 5}],
    }

    rows = training_rows_from_payload(payload)

    assert rows[0]["seed"] == 7
    assert rows[0]["episode_id"] == 3
    assert rows[0]["action"] == "move_right"


def test_dataset_training_interface_reports_auxiliary_and_legacy_fallbacks():
    assert dataset_training_interface(
        {"auxiliary_action_samples": [{"seed": 1, "episode_id": 0}]}
    ) == "auxiliary_action_samples"
    assert dataset_training_interface(
        {"state_samples": [{"state_id": 1}]}
    ) == "legacy_state_samples"


def test_transition_records_to_action_rows_projects_horizon_labels():
    rows = transition_records_to_action_rows(
        [
            {
                "seed": 5,
                "episode_id": 6,
                "step": 7,
                "horizon": 3,
                "action": "move_right",
                "action_index": 2,
                "plan_origin": "baseline",
                "observation": {
                    "viewport_class_ids": [[0] * 9 for _ in range(7)],
                    "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                    "body_vector": [1.0, 1.0, 1.0, 1.0],
                    "inventory_vector": [0] * 12,
                },
                "nearest_threat_distances": {"zombie": None, "skeleton": None, "arrow": None},
                "regime_labels": ["neutral"],
                "primary_regime": "neutral",
                "state_signature": {"shape": "s"},
                "state_signature_key": "s",
                "horizon_outcome": {"damage_h": 0.0, "resource_gain_h": 1.0, "survived_h": 1.0, "escape_delta_h": None},
                "auxiliary_counterfactual_probes": [{"action": "do"}],
            }
        ]
    )

    assert rows[0]["label_source"] == "learner_transition_record"
    assert rows[0]["label"]["resource_gain_h"] == 1.0
    assert rows[0]["counterfactual_outcomes"] == [{"action": "do"}]


def test_collate_local_samples_marks_planner_teacher_records():
    batch = [
        {
            "observation": {
                "viewport_class_ids": [[0] * 9 for _ in range(7)],
                "viewport_confidences": [[0.0] * 9 for _ in range(7)],
                "body_vector": [1.0, 2.0, 3.0, 4.0],
                "inventory_vector": [0] * 12,
                "belief_state_vector": [0.1, 0.2],
            },
            "planner_action": "move_left",
            "planner_action_index": 1,
            "teacher_policy": "planner",
            "resulting_outcome": {
                "damage_h": 0.0,
                "resource_gain_h": 0.0,
                "survived_h": 1.0,
                "escape_delta_h": 1.0,
            },
        }
    ]

    out = collate_local_samples(batch)

    assert out["teacher_action"].tolist() == [1]
    assert out["teacher_mask"].tolist() == [1.0]


def test_stage90r_action_utility_prefers_safer_action():
    utility_safe = stage90r_action_utility(
        pred_damage=torch.tensor([0.1]),
        pred_resource_gain=torch.tensor([0.0]),
        pred_survival_logit=torch.tensor([3.0]),
        pred_escape_delta=torch.tensor([1.0]),
        pred_progress_delta=torch.tensor([0.8]),
        pred_stall_risk_logit=torch.tensor([-2.0]),
        pred_affordance_persistence_logit=torch.tensor([1.0]),
        pred_threat_trend=torch.tensor([0.5]),
    )
    utility_risky = stage90r_action_utility(
        pred_damage=torch.tensor([2.0]),
        pred_resource_gain=torch.tensor([1.0]),
        pred_survival_logit=torch.tensor([-1.0]),
        pred_escape_delta=torch.tensor([-1.0]),
        pred_progress_delta=torch.tensor([0.0]),
        pred_stall_risk_logit=torch.tensor([2.0]),
        pred_affordance_persistence_logit=torch.tensor([-1.0]),
        pred_threat_trend=torch.tensor([-1.0]),
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
            "pred_progress_delta": torch.tensor([0.5 if action_id == 5 else 0.1], dtype=torch.float32),
            "pred_stall_risk_logit": torch.tensor([0.8 if action_id == 6 else -0.5], dtype=torch.float32),
            "pred_affordance_persistence_logit": torch.tensor([1.2 if action_id == 5 else -0.2], dtype=torch.float32),
            "pred_threat_trend": torch.tensor([0.0 if action_id == 5 else 0.1], dtype=torch.float32),
            "pred_next_belief_state": torch.zeros((1, 0), dtype=torch.float32),
            "pred_actor_logits": torch.tensor([[0.0, 1.2, 0.5, -0.2, 0.0, 2.0, -1.0]], dtype=torch.float32),
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


def test_rank_local_actor_candidates_filters_to_allowed_primitives():
    ranked = rank_local_actor_candidates(
        evaluator=_FixedActionEvaluator(),
        observation=_test_observation(adjacent_right=2),
        allowed_actions=["move_left", "move_right", "do", "sleep"],
        action_to_idx={"move_left": 1, "move_right": 2, "do": 5, "sleep": 6},
        action_names=["noop", "move_left", "move_right", "move_down", "move_up", "do", "sleep"],
        device="cpu",
    )

    assert [candidate["action"] for candidate in ranked] == ["do", "move_left", "move_right", "sleep"]


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
        near_concept="tree",
        player_pos_before=(10, 10),
        player_pos_after=(10, 10),
        body_before={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        body_after={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        inventory_before={"wood": 0},
        inventory_after={"wood": 1},
    )
    tracker.observe_transition(
        near_concept="tree",
        player_pos_before=(10, 10),
        player_pos_after=(10, 10),
        body_before={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        body_after={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        inventory_before={"wood": 1},
        inventory_after={"wood": 2},
    )

    context = tracker.build_context(near_concept="tree")

    assert len(context["vector"]) == 6
    assert context["signature"]["affordance_stability_bucket"] in {"low", "medium", "high"}
    assert context["signature"]["resource_flow_bucket"] in {"low", "medium", "high"}


def test_build_local_observation_package_attaches_belief_context():
    class _VF:
        detections = []

    obs = build_local_observation_package(
        _VF(),
        {"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        {},
        belief_context={
            "vector": [0.1, 0.2, 0.3],
            "feature_names": ["a", "b", "c"],
            "signature": {"progress_bucket": "medium"},
        },
    )

    assert obs["belief_state_vector"] == [0.1, 0.2, 0.3]
    assert obs["belief_state_signature"] == {"progress_bucket": "medium"}


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
            "pred_progress_delta": torch.zeros(batch_size, dtype=torch.float32),
            "pred_stall_risk_logit": torch.zeros(batch_size, dtype=torch.float32),
            "pred_affordance_persistence_logit": torch.zeros(batch_size, dtype=torch.float32),
            "pred_threat_trend": torch.zeros(batch_size, dtype=torch.float32),
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


def test_checkpoint_priority_prefers_lower_valid_loss_before_ranking_score():
    assert _checkpoint_priority(valid_loss=3.2, selection_score=0.70) > _checkpoint_priority(
        valid_loss=4.0,
        selection_score=0.80,
    )
