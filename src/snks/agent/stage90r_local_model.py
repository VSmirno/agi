"""Stage 90R local action evaluator."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from snks.agent.decode_head import NEAR_CLASSES


@dataclass(frozen=True)
class LocalEvaluatorConfig:
    viewport_rows: int = 7
    viewport_cols: int = 9
    n_classes: int = 13
    n_body: int = 4
    n_inventory: int = 12
    n_actions: int = 17
    tile_embed_dim: int = 12
    action_embed_dim: int = 8
    belief_state_dim: int = 0
    belief_state_hidden_dim: int = 32
    hidden_dim: int = 256


class Stage90RLocalDataset(Dataset):
    """JSON-backed dataset of viewport-first local decision samples."""

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.samples[index]


def load_local_dataset(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if (
        "learner_transition_records" not in payload
        and "auxiliary_action_samples" not in payload
        and "auxiliary_state_samples" not in payload
        and "samples" not in payload
        and "state_samples" not in payload
    ):
        raise ValueError(
            "Dataset missing learner_transition_records / auxiliary_action_samples / "
            f"auxiliary_state_samples: {path}"
        )
    return payload


def _first_payload_list(
    payload: dict[str, Any],
    *keys: str,
) -> tuple[list[dict[str, Any]], str | None]:
    for key in keys:
        if key not in payload:
            continue
        rows = list(payload.get(key, []))
        if rows:
            return rows, key
    for key in keys:
        if key in payload:
            return [], key
    return [], None


def dataset_training_interface(payload: dict[str, Any]) -> str:
    learner_transition_records, learner_key = _first_payload_list(
        payload,
        "learner_transition_records",
    )
    if learner_transition_records:
        return str(learner_key)

    state_samples, state_key = _first_payload_list(
        payload,
        "auxiliary_state_samples",
        "state_samples",
    )
    if state_samples:
        return "legacy_state_samples" if state_key == "state_samples" else str(state_key)

    action_samples, action_key = _first_payload_list(
        payload,
        "auxiliary_action_samples",
        "samples",
    )
    if action_samples:
        return "legacy_action_samples" if action_key == "samples" else str(action_key)

    if state_key == "state_samples":
        return "legacy_state_samples"
    if action_key == "samples":
        return "legacy_action_samples"
    return str(state_key or action_key or learner_key or "unknown")


def flatten_state_samples(state_samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert state-centered comparison buckets back into action-conditioned rows."""
    rows: list[dict[str, Any]] = []
    for state in state_samples:
        observation = state["observation"]
        for candidate in state.get("candidate_actions", []):
            support_refs = list(candidate.get("support_refs", []))
            representative = support_refs[0] if support_refs else state.get("representative_ref", {})
            rows.append(
                {
                    "seed": int(representative.get("seed", 0)),
                    "episode_id": int(representative.get("episode_id", 0)),
                    "step": int(representative.get("step", 0)),
                    "state_id": int(state.get("state_id", 0)),
                    "primary_regime": state.get("primary_regime", "neutral"),
                    "regime_labels": list(state.get("regime_labels", [])),
                    "observation": observation,
                    "action": candidate["action"],
                    "action_index": int(candidate["action_index"]),
                    "label": candidate["label"],
                    "label_source": candidate.get("source", "state_centered"),
                    "support_episode_keys": sorted(
                        {
                            (int(ref.get("seed", 0)), int(ref.get("episode_id", 0)))
                            for ref in support_refs
                        }
                    ),
                }
            )
    return rows


def training_rows_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    learner_transition_records, _learner_key = _first_payload_list(
        payload,
        "learner_transition_records",
    )
    if learner_transition_records:
        return learner_transition_records
    state_samples, _state_key = _first_payload_list(
        payload,
        "auxiliary_state_samples",
        "state_samples",
    )
    if state_samples:
        return flatten_state_samples(state_samples)
    action_samples, _action_key = _first_payload_list(
        payload,
        "auxiliary_action_samples",
        "samples",
    )
    return action_samples


def transition_records_to_action_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "seed": int(record["seed"]),
                "episode_id": int(record["episode_id"]),
                "step": int(record["step"]),
                "horizon": int(record.get("horizon", 1)),
                "action": record["action"],
                "action_index": int(record["action_index"]),
                "plan_origin": record.get("plan_origin"),
                "observation": record["observation"],
                "nearest_threat_distances": dict(record.get("nearest_threat_distances", {})),
                "regime_labels": list(record.get("regime_labels", [])),
                "primary_regime": record.get("primary_regime", "neutral"),
                "state_signature": dict(record.get("state_signature", {})),
                "state_signature_key": str(record.get("state_signature_key", "")),
                "label_source": "learner_transition_record",
                "label": dict(record.get("horizon_outcome", {})),
                "counterfactual_outcomes": list(record.get("auxiliary_counterfactual_probes", [])),
            }
        )
    return rows


def split_samples_by_episode(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def fallback_split() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        episode_keys = sorted(
            {(int(sample["seed"]), int(sample["episode_id"])) for sample in samples},
            key=split_order,
        )
        if not episode_keys:
            return [], []
        cut = max(1, int(round(len(episode_keys) * train_ratio)))
        cut = min(cut, len(episode_keys) - 1) if len(episode_keys) > 1 else len(episode_keys)
        train_keys = set(episode_keys[:cut])
        train, valid = [], []
        for sample in samples:
            key = (int(sample["seed"]), int(sample["episode_id"]))
            if key in train_keys:
                train.append(sample)
            else:
                valid.append(sample)
        if not valid:
            return train, train[-max(1, len(train) // 5):]
        return train, valid

    def split_order(key: tuple[int, int]) -> tuple[str, tuple[int, int]]:
        digest = hashlib.sha256(f"{key[0]}:{key[1]}".encode("utf-8")).hexdigest()
        return digest, key

    def state_split_order(
        key: tuple[int, int, str, int],
    ) -> tuple[str, tuple[int, int, str, int]]:
        digest = hashlib.sha256(f"{key[0]}:{key[1]}:{key[2]}:{key[3]}".encode("utf-8")).hexdigest()
        return digest, key

    def state_group_key(sample: dict[str, Any]) -> tuple[int, int, str, int]:
        state_signature = sample.get("state_signature_key")
        if state_signature not in (None, ""):
            state_token = str(state_signature)
            state_step = 0
        else:
            state_token = str(sample.get("state_id", ""))
            state_step = int(sample.get("step", 0))
        return (
            int(sample["seed"]),
            int(sample["episode_id"]),
            state_token,
            state_step,
        )

    def sample_regimes(sample: dict[str, Any]) -> set[str]:
        regimes = {str(regime) for regime in sample.get("regime_labels", []) if regime}
        primary_regime = str(sample.get("primary_regime", "neutral"))
        if primary_regime:
            regimes.add(primary_regime)
        return regimes or {"neutral"}

    def threat_support_from_counts(regime_counts: dict[str, int]) -> int:
        return (
            regime_counts.get("hostile_contact", 0)
            + regime_counts.get("hostile_near", 0)
            + regime_counts.get("low_vitals", 0)
        )

    def state_level_support_split() -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
        indexed_pairs = list(zip(samples, scoring_samples, strict=False))
        if not indexed_pairs:
            return None

        state_groups: dict[tuple[int, int, str, int], dict[str, Any]] = {}
        for original_sample, scoring_sample in indexed_pairs:
            key = state_group_key(scoring_sample)
            bucket = state_groups.setdefault(
                key,
                {
                    "rows": [],
                    "n_samples": 0,
                    "regime_counts": None,
                },
            )
            bucket["rows"].append(original_sample)
            bucket["n_samples"] += 1
            if bucket["regime_counts"] is None:
                bucket["regime_counts"] = {regime: 1 for regime in sample_regimes(scoring_sample)}

        ordered_keys = sorted(state_groups, key=state_split_order)
        if len(ordered_keys) <= 1:
            return None

        total_threat = sum(
            threat_support_from_counts(state_groups[key]["regime_counts"]) for key in ordered_keys
        )
        if total_threat <= 1:
            return None

        target_valid_samples = len(samples) - int(round(len(samples) * train_ratio))
        if len(ordered_keys) > 16:
            valid_key_set: set[tuple[int, int, str, int]] = set()
            valid_samples = 0
            for key in ordered_keys:
                if valid_samples >= target_valid_samples and valid_key_set:
                    break
                valid_key_set.add(key)
                valid_samples += int(state_groups[key]["n_samples"])
            train, valid = [], []
            for key in ordered_keys:
                rows = state_groups[key]["rows"]
                if key in valid_key_set:
                    valid.extend(rows)
                else:
                    train.extend(rows)
            if train and valid:
                return train, valid
            return None

        best_valid_keys: tuple[tuple[int, int, str, int], ...] | None = None
        best_score: tuple[Any, ...] | None = None
        best_digest: str | None = None
        n_groups = len(ordered_keys)
        for mask in range(1, (1 << n_groups) - 1):
            valid_keys = tuple(ordered_keys[idx] for idx in range(n_groups) if mask & (1 << idx))
            valid_key_set = set(valid_keys)
            valid_samples = sum(int(state_groups[key]["n_samples"]) for key in valid_keys)
            valid_threat = sum(
                threat_support_from_counts(state_groups[key]["regime_counts"]) for key in valid_keys
            )
            train_threat = total_threat - valid_threat
            if valid_threat <= 0 or train_threat <= 0:
                continue

            valid_resource = sum(
                int(state_groups[key]["regime_counts"].get("local_resource_facing", 0) > 0)
                for key in valid_keys
            )
            train_resource = sum(
                int(state_groups[key]["regime_counts"].get("local_resource_facing", 0) > 0)
                for key in ordered_keys
                if key not in valid_key_set
            )
            valid_neutral = sum(
                int(state_groups[key]["regime_counts"].get("neutral", 0) > 0)
                for key in valid_keys
            )
            train_neutral = sum(
                int(state_groups[key]["regime_counts"].get("neutral", 0) > 0)
                for key in ordered_keys
                if key not in valid_key_set
            )

            score = (
                1,
                min(valid_threat, train_threat),
                int(valid_resource > 0 and train_resource > 0),
                min(valid_resource, train_resource),
                int(valid_neutral > 0 and train_neutral > 0),
                -abs(valid_samples - target_valid_samples),
                -abs(len(valid_keys) - max(1, round(n_groups * (1.0 - train_ratio)))),
            )
            digest = hashlib.sha256(repr(valid_keys).encode("utf-8")).hexdigest()
            if (
                best_score is None
                or score > best_score
                or (score == best_score and digest < str(best_digest))
            ):
                best_score = score
                best_valid_keys = valid_keys
                best_digest = digest

        if not best_valid_keys:
            return None

        valid_key_set = set(best_valid_keys)
        train, valid = [], []
        for key in ordered_keys:
            rows = state_groups[key]["rows"]
            if key in valid_key_set:
                valid.extend(rows)
            else:
                train.extend(rows)
        if not train or not valid:
            return None
        return train, valid

    if not samples:
        return [], []

    scoring_samples = (
        transition_records_to_action_rows(samples)
        if "horizon_outcome" in samples[0] and "label" not in samples[0]
        else samples
    )

    required_keys = {"action", "state_signature_key", "state_signature", "observation"}
    if not required_keys.issubset(scoring_samples[0]):
        return fallback_split()

    episode_keys = sorted(
        {(int(sample["seed"]), int(sample["episode_id"])) for sample in scoring_samples},
        key=split_order,
    )
    if len(episode_keys) <= 1:
        return fallback_split()
    if len(episode_keys) == 2:
        return state_level_support_split() or fallback_split()
    if len(episode_keys) > 12:
        return fallback_split()

    from snks.agent.stage90r_local_policy import build_state_centered_training_examples

    cut = max(1, int(round(len(episode_keys) * train_ratio)))
    cut = min(cut, len(episode_keys) - 1)
    valid_count = len(episode_keys) - cut
    episode_samples: dict[tuple[int, int], list[dict[str, Any]]] = {key: [] for key in episode_keys}
    for sample in scoring_samples:
        episode_samples[(int(sample["seed"]), int(sample["episode_id"]))].append(sample)

    critical_regimes = (
        "hostile_contact",
        "hostile_near",
        "low_vitals",
        "local_resource_facing",
        "neutral",
    )
    episode_profiles: dict[tuple[int, int], dict[str, Any]] = {}
    for key, episode_rows in episode_samples.items():
        state_samples = build_state_centered_training_examples(episode_rows)
        regime_counts: dict[str, int] = {}
        target_winner_counts: dict[str, float] = {}
        comparison_state_count = 0
        for state in state_samples:
            for regime in sample_regimes(state):
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            candidates = list(state.get("candidate_actions", []))
            if len(candidates) < 2:
                continue
            comparison_state_count += 1
            best_key = max(stage90r_target_order_key(candidate["label"]) for candidate in candidates)
            winners = [
                str(candidate["action"])
                for candidate in candidates
                if stage90r_target_order_key(candidate["label"]) == best_key
            ]
            if not winners:
                continue
            increment = 1.0 / len(winners)
            for action in winners:
                target_winner_counts[action] = target_winner_counts.get(action, 0.0) + increment
        episode_profiles[key] = {
            "n_samples": len(episode_rows),
            "state_regime_counts": regime_counts,
            "target_winner_counts": target_winner_counts,
            "comparison_state_count": comparison_state_count,
        }

    target_valid_samples = len(samples) - int(round(len(samples) * train_ratio))
    best_valid_keys: tuple[tuple[int, int], ...] | None = None
    best_score: tuple[Any, ...] | None = None
    best_digest: str | None = None
    for valid_keys in itertools.combinations(episode_keys, valid_count):
        aggregate_regimes: dict[str, int] = {}
        aggregate_winners: dict[str, float] = {}
        n_valid_samples = 0
        comparison_state_support = 0
        for key in valid_keys:
            profile = episode_profiles[key]
            n_valid_samples += int(profile["n_samples"])
            comparison_state_support += int(profile["comparison_state_count"])
            for regime, count in profile["state_regime_counts"].items():
                aggregate_regimes[regime] = aggregate_regimes.get(regime, 0) + int(count)
            for action, count in profile["target_winner_counts"].items():
                aggregate_winners[action] = aggregate_winners.get(action, 0.0) + float(count)
        winner_total = sum(aggregate_winners.values())
        winner_entropy = 0.0
        if winner_total > 0.0:
            for count in aggregate_winners.values():
                probability = count / winner_total
                if probability > 0.0:
                    winner_entropy -= probability * math.log(probability)
        max_entropy = math.log(max(len(aggregate_winners), 1)) if len(aggregate_winners) > 1 else 1.0
        normalized_winner_entropy = winner_entropy / max(max_entropy, 1e-6)
        dominant_winner_share = (
            max(aggregate_winners.values()) / winner_total if winner_total > 0.0 else 1.0
        )
        threat_support = (
            aggregate_regimes.get("hostile_contact", 0)
            + aggregate_regimes.get("hostile_near", 0)
            + aggregate_regimes.get("low_vitals", 0)
        )
        resource_support = aggregate_regimes.get("local_resource_facing", 0)
        neutral_support = aggregate_regimes.get("neutral", 0)
        score = (
            threat_support,
            int(len(aggregate_winners) >= 2),
            len(aggregate_winners),
            comparison_state_support,
            resource_support,
            round(normalized_winner_entropy, 4),
            round(1.0 - dominant_winner_share, 4),
            neutral_support,
            sum(1 for regime in critical_regimes if aggregate_regimes.get(regime, 0) > 0),
            -abs(n_valid_samples - target_valid_samples),
        )
        digest = hashlib.sha256(repr(valid_keys).encode("utf-8")).hexdigest()
        if (
            best_score is None
            or score > best_score
            or (score == best_score and digest < str(best_digest))
        ):
            best_score = score
            best_valid_keys = valid_keys
            best_digest = digest

    if not best_valid_keys:
        return fallback_split()

    train_threat_support = sum(
        threat_support_from_counts(episode_profiles[key]["state_regime_counts"])
        for key in episode_keys
        if key not in best_valid_keys
    )
    valid_threat_support = sum(
        threat_support_from_counts(episode_profiles[key]["state_regime_counts"])
        for key in best_valid_keys
    )
    if train_threat_support <= 0 < valid_threat_support:
        state_level_split = state_level_support_split()
        if state_level_split is not None:
            return state_level_split

    valid_key_set = set(best_valid_keys)
    train, valid = [], []
    for sample in samples:
        key = (int(sample["seed"]), int(sample["episode_id"]))
        if key in valid_key_set:
            valid.append(sample)
        else:
            train.append(sample)
    if not valid:
        return fallback_split()
    return train, valid


def collate_local_samples(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    def sample_label(sample: dict[str, Any]) -> dict[str, Any]:
        label = (
            sample.get("label")
            or sample.get("horizon_outcome")
            or sample.get("resulting_outcome")
            or {}
        )
        return {
            "damage_h": float(label.get("damage_h", 0.0)),
            "resource_gain_h": float(label.get("resource_gain_h", 0.0)),
            "survived_h": float(label.get("survived_h", 1.0)),
            "escape_delta_h": label.get("escape_delta_h"),
            "progress_delta_h": float(label.get("progress_delta_h", 0.0)),
            "stall_risk_h": float(label.get("stall_risk_h", 0.0)),
            "affordance_persistence_h": float(label.get("affordance_persistence_h", 0.0)),
            "threat_trend_h": float(label.get("threat_trend_h", 0.0)),
        }

    class_ids = torch.tensor(
        [sample["observation"]["viewport_class_ids"] for sample in batch],
        dtype=torch.long,
    )
    confidences = torch.tensor(
        [sample["observation"]["viewport_confidences"] for sample in batch],
        dtype=torch.float32,
    )
    body = torch.tensor(
        [sample["observation"]["body_vector"] for sample in batch],
        dtype=torch.float32,
    )
    inventory = torch.tensor(
        [sample["observation"]["inventory_vector"] for sample in batch],
        dtype=torch.float32,
    )
    action = torch.tensor(
        [
            sample.get("action_index", sample.get("planner_action_index", 0))
            for sample in batch
        ],
        dtype=torch.long,
    )
    temporal = torch.tensor(
        [
            sample["observation"].get(
                "belief_state_vector",
                sample["observation"].get("temporal_vector", []),
            )
            for sample in batch
        ],
        dtype=torch.float32,
    )
    next_belief = torch.tensor(
        [
            sample.get("next_belief_state", {}).get(
                "vector",
                sample["observation"].get(
                    "belief_state_vector",
                    sample["observation"].get("temporal_vector", []),
                ),
            )
            for sample in batch
        ],
        dtype=torch.float32,
    )
    damage = torch.tensor(
        [sample_label(sample)["damage_h"] for sample in batch],
        dtype=torch.float32,
    )
    resource_gain = torch.tensor(
        [sample_label(sample)["resource_gain_h"] for sample in batch],
        dtype=torch.float32,
    )
    survived = torch.tensor(
        [float(sample_label(sample)["survived_h"]) for sample in batch],
        dtype=torch.float32,
    )
    escape_mask = torch.tensor(
        [0.0 if sample_label(sample)["escape_delta_h"] is None else 1.0 for sample in batch],
        dtype=torch.float32,
    )
    escape_delta = torch.tensor(
        [
            0.0 if sample_label(sample)["escape_delta_h"] is None
            else float(sample_label(sample)["escape_delta_h"])
            for sample in batch
        ],
        dtype=torch.float32,
    )
    progress_delta = torch.tensor(
        [float(sample_label(sample).get("progress_delta_h", 0.0)) for sample in batch],
        dtype=torch.float32,
    )
    stall_risk = torch.tensor(
        [float(sample_label(sample).get("stall_risk_h", 0.0)) for sample in batch],
        dtype=torch.float32,
    )
    affordance_persistence = torch.tensor(
        [float(sample_label(sample).get("affordance_persistence_h", 0.0)) for sample in batch],
        dtype=torch.float32,
    )
    threat_trend = torch.tensor(
        [float(sample_label(sample).get("threat_trend_h", 0.0)) for sample in batch],
        dtype=torch.float32,
    )
    teacher_action = torch.tensor(
        [
            int(sample.get("planner_action_index", sample.get("action_index", 0)))
            for sample in batch
        ],
        dtype=torch.long,
    )
    teacher_mask = torch.tensor(
        [
            1.0
            if "planner_action_index" in sample or sample.get("teacher_policy") == "planner"
            else 0.0
            for sample in batch
        ],
        dtype=torch.float32,
    )
    return {
        "class_ids": class_ids,
        "confidences": confidences,
        "body": body,
        "inventory": inventory,
        "action": action,
        "temporal": temporal,
        "next_belief": next_belief,
        "damage": damage,
        "resource_gain": resource_gain,
        "survived": survived,
        "escape_delta": escape_delta,
        "escape_mask": escape_mask,
        "progress_delta": progress_delta,
        "stall_risk": stall_risk,
        "affordance_persistence": affordance_persistence,
        "threat_trend": threat_trend,
        "teacher_action": teacher_action,
        "teacher_mask": teacher_mask,
    }


class LocalActionEvaluator(nn.Module):
    """Action-conditioned short-horizon evaluator for Stage 90R."""

    def __init__(self, config: LocalEvaluatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalEvaluatorConfig()
        flattened_tiles = self.config.viewport_rows * self.config.viewport_cols
        tile_width = flattened_tiles * (self.config.tile_embed_dim + 1)
        action_width = self.config.action_embed_dim
        belief_state_width = (
            self.config.belief_state_hidden_dim if self.config.belief_state_dim > 0 else 0
        )
        input_dim = tile_width + self.config.n_body + self.config.n_inventory + action_width + belief_state_width
        policy_input_dim = tile_width + self.config.n_body + self.config.n_inventory + belief_state_width

        self.tile_embedding = nn.Embedding(self.config.n_classes, self.config.tile_embed_dim)
        self.action_embedding = nn.Embedding(self.config.n_actions, self.config.action_embed_dim)
        self.belief_state_encoder = (
            nn.Sequential(
                nn.Linear(self.config.belief_state_dim, self.config.belief_state_hidden_dim),
                nn.ReLU(),
            )
            if self.config.belief_state_dim > 0
            else None
        )
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )
        self.policy_backbone = nn.Sequential(
            nn.Linear(policy_input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )
        self.damage_head = nn.Linear(self.config.hidden_dim, 1)
        self.resource_head = nn.Linear(self.config.hidden_dim, 1)
        self.survival_head = nn.Linear(self.config.hidden_dim, 1)
        self.escape_head = nn.Linear(self.config.hidden_dim, 1)
        self.progress_head = nn.Linear(self.config.hidden_dim, 1)
        self.stall_head = nn.Linear(self.config.hidden_dim, 1)
        self.affordance_head = nn.Linear(self.config.hidden_dim, 1)
        self.threat_trend_head = nn.Linear(self.config.hidden_dim, 1)
        self.next_belief_head = (
            nn.Linear(self.config.hidden_dim, self.config.belief_state_dim)
            if self.config.belief_state_dim > 0
            else None
        )
        self.actor_head = nn.Linear(self.config.hidden_dim, self.config.n_actions)

    def encode(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        body: torch.Tensor,
        inventory: torch.Tensor,
        action: torch.Tensor,
        belief_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tile_embed = self.tile_embedding(class_ids)  # (B, 7, 9, E)
        tile_features = torch.cat([tile_embed, confidences.unsqueeze(-1)], dim=-1)
        tile_features = tile_features.reshape(tile_features.shape[0], -1)
        action_features = self.action_embedding(action)
        features = [tile_features, body, inventory, action_features]
        if self.belief_state_encoder is not None:
            if belief_state is None:
                belief_state = torch.zeros(
                    (class_ids.shape[0], self.config.belief_state_dim),
                    dtype=body.dtype,
                    device=body.device,
                )
            features.append(self.belief_state_encoder(belief_state))
        return torch.cat(features, dim=1)

    def encode_policy_context(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        body: torch.Tensor,
        inventory: torch.Tensor,
        belief_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tile_embed = self.tile_embedding(class_ids)
        tile_features = torch.cat([tile_embed, confidences.unsqueeze(-1)], dim=-1)
        tile_features = tile_features.reshape(tile_features.shape[0], -1)
        features = [tile_features, body, inventory]
        if self.belief_state_encoder is not None:
            if belief_state is None:
                belief_state = torch.zeros(
                    (class_ids.shape[0], self.config.belief_state_dim),
                    dtype=body.dtype,
                    device=body.device,
                )
            features.append(self.belief_state_encoder(belief_state))
        return torch.cat(features, dim=1)

    def forward(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        body: torch.Tensor,
        inventory: torch.Tensor,
        action: torch.Tensor,
        belief_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.encode(class_ids, confidences, body, inventory, action, belief_state)
        h = self.backbone(x)
        policy_context = self.encode_policy_context(class_ids, confidences, body, inventory, belief_state)
        policy_h = self.policy_backbone(policy_context)
        return {
            "pred_damage": self.damage_head(h).squeeze(-1),
            "pred_resource_gain": self.resource_head(h).squeeze(-1),
            "pred_survival_logit": self.survival_head(h).squeeze(-1),
            "pred_escape_delta": self.escape_head(h).squeeze(-1),
            "pred_progress_delta": self.progress_head(h).squeeze(-1),
            "pred_stall_risk_logit": self.stall_head(h).squeeze(-1),
            "pred_affordance_persistence_logit": self.affordance_head(h).squeeze(-1),
            "pred_threat_trend": self.threat_trend_head(h).squeeze(-1),
            "pred_next_belief_state": (
                self.next_belief_head(h)
                if self.next_belief_head is not None
                else torch.zeros((h.shape[0], 0), dtype=h.dtype, device=h.device)
            ),
            "pred_actor_logits": self.actor_head(policy_h),
        }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.numel() == 0 or target.numel() == 0:
        return torch.zeros((), device=prediction.device if prediction.numel() else target.device)
    if mask is None:
        return torch.mean((prediction - target) ** 2)
    active = mask > 0.0
    if not torch.any(active):
        return torch.zeros((), device=prediction.device)
    return torch.mean((prediction[active] - target[active]) ** 2)


def stage90r_action_utility(
    *,
    pred_damage: torch.Tensor,
    pred_resource_gain: torch.Tensor,
    pred_survival_logit: torch.Tensor,
    pred_escape_delta: torch.Tensor,
    pred_progress_delta: torch.Tensor,
    pred_stall_risk_logit: torch.Tensor,
    pred_affordance_persistence_logit: torch.Tensor,
    pred_threat_trend: torch.Tensor,
    **_ignored: torch.Tensor,
) -> torch.Tensor:
    """Scalar action utility for local-only canary evaluation.

    Higher survival probability and larger escape distance are better.
    Higher predicted damage is worse.
    """
    survival_prob = torch.sigmoid(pred_survival_logit)
    stall_risk = torch.sigmoid(pred_stall_risk_logit)
    affordance_persistence = torch.sigmoid(pred_affordance_persistence_logit)
    return (
        2.0 * survival_prob
        - pred_damage
        + 0.20 * pred_resource_gain
        + 0.10 * pred_escape_delta
        + 0.25 * pred_progress_delta
        - 0.35 * stall_risk
        + 0.10 * affordance_persistence
        + 0.15 * pred_threat_trend
    )


def stage90r_target_utility(label: dict[str, Any]) -> float:
    """Legacy helper kept for tests and compact reporting.

    Offline ranking should prefer the explicit tuple/comparator helpers below.
    """
    order_key = stage90r_target_order_key(label)
    return round(
        (1000.0 * float(order_key[0]))
        + (100.0 * float(order_key[1]))
        + (10.0 * float(order_key[2]))
        + float(order_key[3])
        + (0.1 * float(order_key[4]))
        + (0.01 * float(order_key[5])),
        4,
    )


def stage90r_target_order_key(label: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    """Explicit offline target preference order.

    Ranking priority:
    1. survive the horizon
    2. take less damage
    3. increase hostile separation when escape signal exists
    4. gain resources
    5. preserve health delta
    """
    survived = 1.0 if float(label.get("survived_h", 0.0)) >= 0.5 else 0.0
    damage = -round(float(label.get("damage_h", 0.0)), 4)
    escape_delta_raw = label.get("escape_delta_h")
    escape_valid = 1.0 if escape_delta_raw is not None else 0.0
    escape_delta = round(float(escape_delta_raw), 4) if escape_delta_raw is not None else 0.0
    resource_gain = round(float(label.get("resource_gain_h", 0.0)), 4)
    health_delta = round(float(label.get("health_delta_h", 0.0)), 4)
    return (
        survived,
        damage,
        escape_valid,
        escape_delta,
        resource_gain,
        health_delta,
    )


def compare_stage90r_target_labels(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_key = stage90r_target_order_key(left)
    right_key = stage90r_target_order_key(right)
    if left_key > right_key:
        return 1
    if left_key < right_key:
        return -1
    return 0


def _observation_supports_do(observation: dict[str, Any]) -> bool:
    class_ids = observation.get("viewport_class_ids") or []
    if not class_ids or not class_ids[0]:
        return False
    center_y = len(class_ids) // 2
    center_x = len(class_ids[0]) // 2
    adjacent = (
        (center_y, center_x - 1),
        (center_y, center_x + 1),
        (center_y - 1, center_x),
        (center_y + 1, center_x),
    )
    for gy, gx in adjacent:
        if gy < 0 or gx < 0:
            continue
        if gy >= len(class_ids) or gx >= len(class_ids[gy]):
            continue
        class_id = int(class_ids[gy][gx])
        if class_id <= 0 or class_id >= len(NEAR_CLASSES):
            continue
        if str(NEAR_CLASSES[class_id]) in _DO_ACTIONABLE_CONCEPTS:
            return True
    return False


def rank_local_action_candidates(
    *,
    evaluator: LocalActionEvaluator,
    observation: dict[str, Any],
    allowed_actions: list[str],
    action_to_idx: dict[str, int],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    class_ids = torch.tensor([observation["viewport_class_ids"]], dtype=torch.long, device=device)
    confidences = torch.tensor([observation["viewport_confidences"]], dtype=torch.float32, device=device)
    body_vec = torch.tensor([observation["body_vector"]], dtype=torch.float32, device=device)
    inv_vec = torch.tensor([observation["inventory_vector"]], dtype=torch.float32, device=device)
    belief_state_vector = observation.get("belief_state_vector", observation.get("temporal_vector", []))
    belief_state_vec = (
        torch.tensor([belief_state_vector], dtype=torch.float32, device=device)
        if belief_state_vector
        else None
    )

    ranked: list[dict[str, Any]] = []
    for primitive in allowed_actions:
        if primitive == "do" and not _observation_supports_do(observation):
            continue
        action_idx = torch.tensor([action_to_idx[primitive]], dtype=torch.long, device=device)
        try:
            preds = evaluator(class_ids, confidences, body_vec, inv_vec, action_idx, belief_state_vec)
        except TypeError:
            preds = evaluator(class_ids, confidences, body_vec, inv_vec, action_idx)
        utility = float(stage90r_action_utility(**preds).item())
        ranked.append(
            {
                "action": primitive,
                "score": round(utility, 4),
                "pred_damage": round(float(preds["pred_damage"].item()), 4),
                "pred_resource_gain": round(float(preds["pred_resource_gain"].item()), 4),
                "pred_survival_prob": round(float(torch.sigmoid(preds["pred_survival_logit"]).item()), 4),
                "pred_escape_delta": round(float(preds["pred_escape_delta"].item()), 4),
                "pred_progress_delta": round(float(preds["pred_progress_delta"].item()), 4),
                "pred_stall_risk": round(float(torch.sigmoid(preds["pred_stall_risk_logit"]).item()), 4),
                "pred_affordance_persistence": round(
                    float(torch.sigmoid(preds["pred_affordance_persistence_logit"]).item()),
                    4,
                ),
                "pred_threat_trend": round(float(preds["pred_threat_trend"].item()), 4),
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def rank_local_actor_candidates(
    *,
    evaluator: LocalActionEvaluator,
    observation: dict[str, Any],
    allowed_actions: list[str],
    action_to_idx: dict[str, int],
    device: torch.device | str,
    action_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    class_ids = torch.tensor([observation["viewport_class_ids"]], dtype=torch.long, device=device)
    confidences = torch.tensor([observation["viewport_confidences"]], dtype=torch.float32, device=device)
    body_vec = torch.tensor([observation["body_vector"]], dtype=torch.float32, device=device)
    inv_vec = torch.tensor([observation["inventory_vector"]], dtype=torch.float32, device=device)
    belief_state_vector = observation.get("belief_state_vector", observation.get("temporal_vector", []))
    belief_state_vec = (
        torch.tensor([belief_state_vector], dtype=torch.float32, device=device)
        if belief_state_vector
        else None
    )
    noop_idx = next(
        (
            idx
            for idx, name in enumerate(action_names or [])
            if str(name) == "noop"
        ),
        0,
    )
    action_idx = torch.tensor([noop_idx], dtype=torch.long, device=device)
    try:
        preds = evaluator(class_ids, confidences, body_vec, inv_vec, action_idx, belief_state_vec)
    except TypeError:
        preds = evaluator(class_ids, confidences, body_vec, inv_vec, action_idx)
    logits = preds["pred_actor_logits"][0]
    probabilities = torch.softmax(logits, dim=0)
    ranked: list[dict[str, Any]] = []
    for primitive in allowed_actions:
        if primitive == "do" and not _observation_supports_do(observation):
            continue
        primitive_idx = int(action_to_idx[primitive])
        ranked.append(
            {
                "action": primitive,
                "action_index": primitive_idx,
                "logit": round(float(logits[primitive_idx].item()), 4),
                "probability": round(float(probabilities[primitive_idx].item()), 4),
            }
        )
    ranked.sort(key=lambda item: (item["probability"], item["logit"]), reverse=True)
    return ranked


def build_local_advisory_entry(
    *,
    planner_action: str,
    planner_plan_origin: str | None,
    ranked_candidates: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    planner_candidate = next(
        (candidate for candidate in ranked_candidates if candidate["action"] == planner_action),
        None,
    )
    planner_rank = next(
        (
            index + 1
            for index, candidate in enumerate(ranked_candidates)
            if candidate["action"] == planner_action
        ),
        None,
    )
    advisory_best = ranked_candidates[0] if ranked_candidates else None
    planner_score = (
        round(float(planner_candidate["score"]), 4)
        if planner_candidate is not None
        else None
    )
    advisory_best_score = (
        round(float(advisory_best["score"]), 4)
        if advisory_best is not None
        else None
    )
    return {
        "planner_action": planner_action,
        "planner_plan_origin": planner_plan_origin,
        "planner_rank_by_local_predictor": planner_rank,
        "planner_predicted_score": planner_score,
        "advisory_best_action": advisory_best["action"] if advisory_best is not None else None,
        "advisory_best_score": advisory_best_score,
        "advisory_agrees_with_planner": bool(
            advisory_best is not None and advisory_best["action"] == planner_action
        ),
        "score_gap_to_advisory_best": (
            round(float(advisory_best_score) - float(planner_score), 4)
            if advisory_best is not None and planner_score is not None
            else None
        ),
        "top_candidates": ranked_candidates[: max(1, top_k)],
    }


def load_local_evaluator_artifact(
    path: str | Path,
    device: torch.device | str | None = None,
) -> tuple[LocalActionEvaluator, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = dict(ckpt["config"])
    if "temporal_dim" in raw_config and "belief_state_dim" not in raw_config:
        raw_config["belief_state_dim"] = raw_config.pop("temporal_dim")
    if "temporal_hidden_dim" in raw_config and "belief_state_hidden_dim" not in raw_config:
        raw_config["belief_state_hidden_dim"] = raw_config.pop("temporal_hidden_dim")
    config = LocalEvaluatorConfig(**raw_config)
    model = LocalActionEvaluator(config)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    return model, ckpt


def load_local_evaluator_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> LocalActionEvaluator:
    model, _artifact = load_local_evaluator_artifact(path, device=device)
    return model
_DO_ACTIONABLE_CONCEPTS = {"tree", "cow"}
