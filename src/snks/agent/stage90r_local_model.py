"""Stage 90R local action evaluator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset


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
    if "samples" not in payload and "state_samples" not in payload:
        raise ValueError(f"Dataset missing 'samples' or 'state_samples': {path}")
    return payload


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
    state_samples = list(payload.get("state_samples", []))
    if state_samples:
        return flatten_state_samples(state_samples)
    return list(payload.get("samples", []))


def split_samples_by_episode(
    samples: list[dict[str, Any]],
    train_ratio: float = 0.8,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_keys = sorted({(int(sample["seed"]), int(sample["episode_id"])) for sample in samples})
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


def collate_local_samples(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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
        [sample["action_index"] for sample in batch],
        dtype=torch.long,
    )
    damage = torch.tensor(
        [sample["label"]["damage_h"] for sample in batch],
        dtype=torch.float32,
    )
    resource_gain = torch.tensor(
        [sample["label"]["resource_gain_h"] for sample in batch],
        dtype=torch.float32,
    )
    survived = torch.tensor(
        [float(sample["label"]["survived_h"]) for sample in batch],
        dtype=torch.float32,
    )
    escape_mask = torch.tensor(
        [0.0 if sample["label"]["escape_delta_h"] is None else 1.0 for sample in batch],
        dtype=torch.float32,
    )
    escape_delta = torch.tensor(
        [
            0.0 if sample["label"]["escape_delta_h"] is None
            else float(sample["label"]["escape_delta_h"])
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
        "damage": damage,
        "resource_gain": resource_gain,
        "survived": survived,
        "escape_delta": escape_delta,
        "escape_mask": escape_mask,
    }


class LocalActionEvaluator(nn.Module):
    """Action-conditioned short-horizon evaluator for Stage 90R."""

    def __init__(self, config: LocalEvaluatorConfig | None = None) -> None:
        super().__init__()
        self.config = config or LocalEvaluatorConfig()
        flattened_tiles = self.config.viewport_rows * self.config.viewport_cols
        tile_width = flattened_tiles * (self.config.tile_embed_dim + 1)
        action_width = self.config.action_embed_dim
        input_dim = tile_width + self.config.n_body + self.config.n_inventory + action_width

        self.tile_embedding = nn.Embedding(self.config.n_classes, self.config.tile_embed_dim)
        self.action_embedding = nn.Embedding(self.config.n_actions, self.config.action_embed_dim)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
        )
        self.damage_head = nn.Linear(self.config.hidden_dim, 1)
        self.resource_head = nn.Linear(self.config.hidden_dim, 1)
        self.survival_head = nn.Linear(self.config.hidden_dim, 1)
        self.escape_head = nn.Linear(self.config.hidden_dim, 1)

    def encode(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        body: torch.Tensor,
        inventory: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        tile_embed = self.tile_embedding(class_ids)  # (B, 7, 9, E)
        tile_features = torch.cat([tile_embed, confidences.unsqueeze(-1)], dim=-1)
        tile_features = tile_features.reshape(tile_features.shape[0], -1)
        action_features = self.action_embedding(action)
        return torch.cat([tile_features, body, inventory, action_features], dim=1)

    def forward(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        body: torch.Tensor,
        inventory: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = self.encode(class_ids, confidences, body, inventory, action)
        h = self.backbone(x)
        return {
            "pred_damage": self.damage_head(h).squeeze(-1),
            "pred_resource_gain": self.resource_head(h).squeeze(-1),
            "pred_survival_logit": self.survival_head(h).squeeze(-1),
            "pred_escape_delta": self.escape_head(h).squeeze(-1),
        }


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
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
) -> torch.Tensor:
    """Scalar action utility for local-only canary evaluation.

    Higher survival probability and larger escape distance are better.
    Higher predicted damage is worse.
    """
    survival_prob = torch.sigmoid(pred_survival_logit)
    return (
        2.0 * survival_prob
        - pred_damage
        + 0.25 * pred_resource_gain
        + 0.10 * pred_escape_delta
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
        if int(class_ids[gy][gx]) != 0:
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

    ranked: list[dict[str, Any]] = []
    for primitive in allowed_actions:
        if primitive == "do" and not _observation_supports_do(observation):
            continue
        action_idx = torch.tensor([action_to_idx[primitive]], dtype=torch.long, device=device)
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
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
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
    config = LocalEvaluatorConfig(**ckpt["config"])
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
