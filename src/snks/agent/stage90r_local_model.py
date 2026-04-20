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
    if "samples" not in payload:
        raise ValueError(f"Dataset missing 'samples': {path}")
    return payload


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
        [1.0 if sample["label"]["survived_h"] else 0.0 for sample in batch],
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


def load_local_evaluator_checkpoint(
    path: str | Path,
    device: torch.device | str | None = None,
) -> LocalActionEvaluator:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = LocalEvaluatorConfig(**ckpt["config"])
    model = LocalActionEvaluator(config)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(torch.device(device))
    model.eval()
    return model
