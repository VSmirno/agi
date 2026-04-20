"""Stage 90R helpers for viewport-first local policy data."""

from __future__ import annotations

from typing import Any

from snks.agent.crafter_pixel_env import ACTION_NAMES, ACTION_TO_IDX, INVENTORY_ITEMS
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.perception import VisualField
from snks.encoder.tile_head_trainer import VIEWPORT_COLS, VIEWPORT_ROWS

LOCAL_BODY_KEYS = ("health", "food", "drink", "energy")
LOCAL_HOSTILE_KEYS = ("zombie", "skeleton", "arrow")


def dense_viewport_scene(vf: VisualField) -> tuple[list[list[int]], list[list[float]]]:
    """Convert sparse VisualField detections into dense 7x9 ids + confidences."""
    class_ids = [
        [0 for _ in range(VIEWPORT_COLS)]
        for _ in range(VIEWPORT_ROWS)
    ]
    confidences = [
        [0.0 for _ in range(VIEWPORT_COLS)]
        for _ in range(VIEWPORT_ROWS)
    ]

    for concept_id, confidence, gy, gx in vf.detections:
        if not (0 <= gy < VIEWPORT_ROWS and 0 <= gx < VIEWPORT_COLS):
            continue
        if confidence < confidences[gy][gx]:
            continue
        class_ids[gy][gx] = int(NEAR_TO_IDX.get(concept_id, 0))
        confidences[gy][gx] = round(float(confidence), 3)

    return class_ids, confidences


def encode_body_vector(body: dict[str, float]) -> list[float]:
    return [round(float(body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS]


def encode_inventory_vector(inventory: dict[str, Any]) -> list[int]:
    return [int(inventory.get(key, 0)) for key in INVENTORY_ITEMS]


def build_local_observation_package(
    vf: VisualField,
    body: dict[str, float],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    class_ids, confidences = dense_viewport_scene(vf)
    sparse_inventory = {
        key: int(value)
        for key, value in inventory.items()
        if int(value) != 0
    }
    return {
        "viewport_class_ids": class_ids,
        "viewport_confidences": confidences,
        "body_vector": encode_body_vector(body),
        "inventory_vector": encode_inventory_vector(inventory),
        "body": {key: round(float(body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS},
        "inventory": sparse_inventory,
    }


def nearest_hostile_distance(threat_distances: dict[str, int | None]) -> int | None:
    values = [
        int(value)
        for value in threat_distances.values()
        if value is not None
    ]
    return min(values) if values else None


def build_local_training_examples(
    *,
    local_trace: list[dict[str, Any]],
    final_body: dict[str, float],
    final_inventory: dict[str, Any],
    seed: int,
    episode_id: int,
    horizon: int,
) -> list[dict[str, Any]]:
    """Turn one per-step local trace into fixed-horizon supervised examples."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    final_body_clean = {key: round(float(final_body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS}
    final_inv_clean = {key: int(final_inventory.get(key, 0)) for key in INVENTORY_ITEMS}
    examples: list[dict[str, Any]] = []

    for idx, step in enumerate(local_trace):
        end_idx = min(idx + horizon, len(local_trace) - 1)
        end_step = local_trace[end_idx]
        used_terminal_fallback = idx + horizon >= len(local_trace)
        died_within_horizon = any(
            bool(candidate.get("done_after_step", False))
            for candidate in local_trace[idx:end_idx + 1]
        )
        end_body = final_body_clean if used_terminal_fallback else end_step["observation"]["body"]
        end_inventory = final_inv_clean if used_terminal_fallback else end_step["observation"]["inventory"]
        end_threats = (
            end_step.get("nearest_threat_distances", {})
            if not used_terminal_fallback
            else step.get("nearest_threat_distances", {})
        )

        start_body = step["observation"]["body"]
        start_inventory = step["observation"]["inventory"]
        start_threats = step.get("nearest_threat_distances", {})
        start_hostile = nearest_hostile_distance(start_threats)
        end_hostile = nearest_hostile_distance(end_threats)
        horizon_damage = round(
            max(0.0, float(start_body.get("health", 0.0)) - float(end_body.get("health", 0.0))),
            3,
        )
        resource_gain = 0
        inventory_delta: dict[str, int] = {}
        for key in INVENTORY_ITEMS:
            delta = int(end_inventory.get(key, 0)) - int(start_inventory.get(key, 0))
            if delta != 0:
                inventory_delta[key] = delta
            if delta > 0:
                resource_gain += delta
        health_delta = round(
            float(end_body.get("health", 0.0)) - float(start_body.get("health", 0.0)),
            3,
        )
        if start_hostile is None or end_hostile is None:
            escape_delta = None
        else:
            escape_delta = int(end_hostile - start_hostile)

        examples.append(
            {
                "seed": int(seed),
                "episode_id": int(episode_id),
                "step": int(step["step"]),
                "horizon": int(horizon),
                "action": step["action"],
                "action_index": int(step["action_index"]),
                "plan_origin": step.get("plan_origin"),
                "observation": step["observation"],
                "nearest_threat_distances": dict(start_threats),
                "label": {
                    "health_delta_h": health_delta,
                    "damage_h": horizon_damage,
                    "resource_gain_h": int(resource_gain),
                    "inventory_delta_h": inventory_delta,
                    "survived_h": not died_within_horizon,
                    "escape_delta_h": escape_delta,
                    "nearest_hostile_now": start_hostile,
                    "nearest_hostile_h": end_hostile,
                },
            }
        )

    return examples


def build_local_trace_entry(
    *,
    step: int,
    vf: VisualField,
    body: dict[str, float],
    inventory: dict[str, Any],
    primitive: str,
    plan_origin: str,
    nearest_threat_distances: dict[str, int | None],
    done_after_step: bool,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "action": primitive,
        "action_index": int(ACTION_TO_IDX.get(primitive, 0)),
        "plan_origin": plan_origin,
        "observation": build_local_observation_package(vf, body, inventory),
        "nearest_threat_distances": {
            key: (
                int(nearest_threat_distances.get(key))
                if nearest_threat_distances.get(key) is not None
                else None
            )
            for key in LOCAL_HOSTILE_KEYS
        },
        "done_after_step": bool(done_after_step),
    }


def local_dataset_metadata(horizon: int) -> dict[str, Any]:
    return {
        "viewport_rows": VIEWPORT_ROWS,
        "viewport_cols": VIEWPORT_COLS,
        "near_classes": list(NEAR_CLASSES),
        "body_keys": list(LOCAL_BODY_KEYS),
        "inventory_keys": list(INVENTORY_ITEMS),
        "action_names": list(ACTION_NAMES),
        "horizon": int(horizon),
    }
