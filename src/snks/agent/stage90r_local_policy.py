"""Stage 90R helpers for viewport-first local policy data."""

from __future__ import annotations

import json
from collections import Counter, deque
from typing import Any

from snks.agent.crafter_pixel_env import ACTION_NAMES, ACTION_TO_IDX, INVENTORY_ITEMS
from snks.agent.decode_head import NEAR_CLASSES, NEAR_TO_IDX
from snks.agent.perception import VisualField
from snks.encoder.tile_head_trainer import VIEWPORT_COLS, VIEWPORT_ROWS

LOCAL_BODY_KEYS = ("health", "food", "drink", "energy")
LOCAL_HOSTILE_KEYS = ("zombie", "skeleton", "arrow")
LOCAL_CORE_ACTIONS = ("move_left", "move_right", "move_up", "move_down", "do", "sleep")
LOCAL_RESOURCE_KEYS = ("water", "tree", "stone", "coal", "iron", "diamond", "cow")
RESOURCE_CLASS_SET = set(LOCAL_RESOURCE_KEYS)
HOSTILE_CLASS_SET = set(LOCAL_HOSTILE_KEYS)
_PLAYER_CENTER_Y = VIEWPORT_ROWS // 2
_PLAYER_CENTER_X = VIEWPORT_COLS // 2
_TEMPORAL_ACTION_WINDOW = 6
_TEMPORAL_TRANSITION_WINDOW = 4
_TEMPORAL_STREAK_CLIP = 4
_TEMPORAL_DISPLACEMENT_CLIP = 8.0
_TEMPORAL_RESOURCE_CLIP = 4.0
_TEMPORAL_DAMAGE_CLIP = 4.0
_TEMPORAL_HEALTH_DELTA_CLIP = 4.0
_TEMPORAL_FEATURE_NAMES = [
    "near_concept_streak_norm",
    "recent_displacement_norm",
    "recent_damage_norm",
    "recent_resource_gain_norm",
    "recent_health_delta_norm",
]


def _clip_signed(value: float, *, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return round(max(-1.0, min(1.0, float(value) / limit)), 3)


def _clip_positive(value: float, *, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return round(max(0.0, min(1.0, float(value) / limit)), 3)


def _streak_bucket(value: int) -> str:
    value = int(max(0, value))
    if value <= 0:
        return "none"
    if value == 1:
        return "single"
    if value == 2:
        return "short"
    if value == 3:
        return "medium"
    return "long"


def _magnitude_bucket(value: float, *, positive_only: bool) -> str:
    magnitude = float(value)
    if positive_only:
        if magnitude <= 0.0:
            return "none"
    else:
        if abs(magnitude) <= 1e-6:
            return "none"
    magnitude = abs(magnitude)
    if magnitude < 1.0:
        return "low"
    if magnitude < 2.5:
        return "medium"
    return "high"


class TemporalBeliefTracker:
    """Compact short-horizon belief state for local world-model ranking."""

    def __init__(self) -> None:
        self._prev_action: str | None = None
        self._action_streak = 0
        self._stationary_streak = 0
        self._prev_near_concept: str | None = None
        self._near_concept_streak = 0
        self._recent_actions: deque[str] = deque(maxlen=_TEMPORAL_ACTION_WINDOW)
        self._recent_displacements: deque[float] = deque(maxlen=_TEMPORAL_TRANSITION_WINDOW)
        self._recent_damage: deque[float] = deque(maxlen=_TEMPORAL_TRANSITION_WINDOW)
        self._recent_resource_gain: deque[float] = deque(maxlen=_TEMPORAL_TRANSITION_WINDOW)
        self._recent_health_delta: deque[float] = deque(maxlen=_TEMPORAL_TRANSITION_WINDOW)

    def build_context(self, *, near_concept: str | None) -> dict[str, Any]:
        current_near = str(near_concept or "empty")
        near_streak = self._near_concept_streak + 1 if current_near == self._prev_near_concept else 0
        vector = [
            _clip_positive(near_streak, limit=float(_TEMPORAL_STREAK_CLIP)),
            _clip_positive(sum(self._recent_displacements), limit=_TEMPORAL_DISPLACEMENT_CLIP),
            _clip_positive(sum(self._recent_damage), limit=_TEMPORAL_DAMAGE_CLIP),
            _clip_positive(sum(self._recent_resource_gain), limit=_TEMPORAL_RESOURCE_CLIP),
            _clip_signed(sum(self._recent_health_delta), limit=_TEMPORAL_HEALTH_DELTA_CLIP),
        ]
        return {
            "vector": vector,
            "feature_names": list(_TEMPORAL_FEATURE_NAMES),
            "signature": {
                "near_concept": current_near,
                "near_concept_streak_bucket": _streak_bucket(near_streak),
                "recent_displacement_bucket": _magnitude_bucket(
                    sum(self._recent_displacements),
                    positive_only=True,
                ),
                "recent_damage_bucket": _magnitude_bucket(
                    sum(self._recent_damage),
                    positive_only=True,
                ),
                "recent_resource_bucket": _magnitude_bucket(
                    sum(self._recent_resource_gain),
                    positive_only=True,
                ),
                "recent_health_delta_bucket": _magnitude_bucket(
                    sum(self._recent_health_delta),
                    positive_only=False,
                ),
            },
        }

    def observe_transition(
        self,
        *,
        action: str,
        near_concept: str | None,
        player_pos_before: tuple[int, int] | list[int],
        player_pos_after: tuple[int, int] | list[int],
        body_before: dict[str, float],
        body_after: dict[str, float],
        inventory_before: dict[str, Any],
        inventory_after: dict[str, Any],
    ) -> None:
        action = str(action)
        if action == self._prev_action:
            self._action_streak += 1
        else:
            self._prev_action = action
            self._action_streak = 1
        self._recent_actions.append(action)

        before_x, before_y = int(player_pos_before[0]), int(player_pos_before[1])
        after_x, after_y = int(player_pos_after[0]), int(player_pos_after[1])
        displacement = abs(after_x - before_x) + abs(after_y - before_y)
        self._recent_displacements.append(float(displacement))
        if displacement == 0:
            self._stationary_streak += 1
        else:
            self._stationary_streak = 0

        health_before = float(body_before.get("health", 0.0))
        health_after = float(body_after.get("health", 0.0))
        health_delta = health_after - health_before
        self._recent_health_delta.append(float(health_delta))
        self._recent_damage.append(max(0.0, -health_delta))

        resource_gain = 0.0
        for key in set(inventory_before.keys()) | set(inventory_after.keys()):
            delta = int(inventory_after.get(key, 0)) - int(inventory_before.get(key, 0))
            if delta > 0:
                resource_gain += float(delta)
        self._recent_resource_gain.append(resource_gain)

        current_near = str(near_concept or "empty")
        if current_near == self._prev_near_concept:
            self._near_concept_streak += 1
        else:
            self._prev_near_concept = current_near
            self._near_concept_streak = 1


def _with_temporal_context(
    observation: dict[str, Any],
    temporal_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if temporal_context is None:
        return {
            **observation,
            "temporal_vector": [],
            "temporal_feature_names": list(_TEMPORAL_FEATURE_NAMES),
            "temporal_signature": {},
        }
    return {
        **observation,
        "temporal_vector": list(temporal_context.get("vector", [])),
        "temporal_feature_names": list(temporal_context.get("feature_names", _TEMPORAL_FEATURE_NAMES)),
        "temporal_signature": dict(temporal_context.get("signature", {})),
    }


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
    *,
    temporal_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    class_ids, confidences = dense_viewport_scene(vf)
    sparse_inventory = {
        key: int(value)
        for key, value in inventory.items()
        if int(value) != 0
    }
    return _with_temporal_context(
        {
            "viewport_class_ids": class_ids,
            "viewport_confidences": confidences,
            "body_vector": encode_body_vector(body),
            "inventory_vector": encode_inventory_vector(inventory),
            "body": {key: round(float(body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS},
            "inventory": sparse_inventory,
        },
        temporal_context,
    )


def _class_name(class_id: int) -> str:
    if 0 <= class_id < len(NEAR_CLASSES):
        return str(NEAR_CLASSES[class_id])
    return "empty"


def _iter_named_tiles(class_ids: list[list[int]]) -> list[tuple[str, int, int]]:
    tiles: list[tuple[str, int, int]] = []
    for gy, row in enumerate(class_ids):
        for gx, class_id in enumerate(row):
            name = _class_name(int(class_id))
            if name == "empty":
                continue
            tiles.append((name, gy, gx))
    return tiles


def _tile_names_within_radius(
    class_ids: list[list[int]],
    *,
    max_distance: int,
    allowed: set[str] | None = None,
) -> list[str]:
    names: set[str] = set()
    for name, gy, gx in _iter_named_tiles(class_ids):
        if allowed is not None and name not in allowed:
            continue
        if abs(gy - _PLAYER_CENTER_Y) + abs(gx - _PLAYER_CENTER_X) <= max_distance:
            names.add(name)
    return sorted(names)


def _relative_tile_descriptors(
    class_ids: list[list[int]],
    *,
    allowed: set[str],
    max_distance: int,
    limit: int,
) -> list[dict[str, int | str]]:
    descriptors: list[dict[str, int | str]] = []
    for name, gy, gx in _iter_named_tiles(class_ids):
        if name not in allowed:
            continue
        rel_y = gy - _PLAYER_CENTER_Y
        rel_x = gx - _PLAYER_CENTER_X
        manhattan = abs(rel_y) + abs(rel_x)
        if manhattan > max_distance:
            continue
        descriptors.append(
            {
                "name": name,
                "rel_y": int(rel_y),
                "rel_x": int(rel_x),
                "manhattan": int(manhattan),
            }
        )
    descriptors.sort(
        key=lambda item: (
            int(item["manhattan"]),
            abs(int(item["rel_y"])),
            abs(int(item["rel_x"])),
            str(item["name"]),
            int(item["rel_y"]),
            int(item["rel_x"]),
        )
    )
    return descriptors[: max(1, limit)]


def _body_buckets(observation: dict[str, Any]) -> list[int]:
    return [
        int(round(float(value)))
        for value in observation.get("body_vector", [])
    ]


def _inventory_presence(observation: dict[str, Any]) -> list[str]:
    inventory_vector = observation.get("inventory_vector", [])
    present: list[str] = []
    for key, value in zip(INVENTORY_ITEMS, inventory_vector, strict=False):
        if float(value) > 0:
            present.append(key)
    return present


def _nearest_hostile_bucket(nearest_threat_distances: dict[str, int | None]) -> str:
    distance = nearest_hostile_distance(nearest_threat_distances)
    if distance is None:
        return "none"
    if distance <= 1:
        return "contact"
    if distance <= 3:
        return "near"
    return "far"


def infer_local_regime(
    observation: dict[str, Any],
    nearest_threat_distances: dict[str, int | None],
) -> tuple[list[str], str]:
    class_ids = observation["viewport_class_ids"]
    labels: list[str] = []
    nearest_bucket = _nearest_hostile_bucket(nearest_threat_distances)
    if nearest_bucket == "contact":
        labels.append("hostile_contact")
    elif nearest_bucket == "near":
        labels.append("hostile_near")

    body_vector = observation.get("body_vector", [])
    if body_vector and min(float(value) for value in body_vector) <= 4.0:
        labels.append("low_vitals")

    nearby_resources = _tile_names_within_radius(
        class_ids,
        max_distance=2,
        allowed=RESOURCE_CLASS_SET,
    )
    if nearby_resources:
        labels.append("local_resource_facing")

    if not labels:
        labels.append("neutral")

    priority = (
        "hostile_contact",
        "hostile_near",
        "low_vitals",
        "local_resource_facing",
        "neutral",
    )
    primary = next((label for label in priority if label in labels), "neutral")
    return labels, primary


def build_state_signature(
    observation: dict[str, Any],
    nearest_threat_distances: dict[str, int | None],
) -> dict[str, Any]:
    class_ids = observation["viewport_class_ids"]
    center_patch = [
        row[max(0, _PLAYER_CENTER_X - 1):_PLAYER_CENTER_X + 2]
        for row in class_ids[max(0, _PLAYER_CENTER_Y - 1):_PLAYER_CENTER_Y + 2]
    ]
    adjacent_tiles = {
        "left": int(class_ids[_PLAYER_CENTER_Y][_PLAYER_CENTER_X - 1]),
        "right": int(class_ids[_PLAYER_CENTER_Y][_PLAYER_CENTER_X + 1]),
        "up": int(class_ids[_PLAYER_CENTER_Y - 1][_PLAYER_CENTER_X]),
        "down": int(class_ids[_PLAYER_CENTER_Y + 1][_PLAYER_CENTER_X]),
    }
    resource_tiles = _tile_names_within_radius(
        class_ids,
        max_distance=2,
        allowed=RESOURCE_CLASS_SET,
    )
    visible_hostiles = _tile_names_within_radius(
        class_ids,
        max_distance=VIEWPORT_ROWS + VIEWPORT_COLS,
        allowed=HOSTILE_CLASS_SET,
    )
    hostile_geometry = _relative_tile_descriptors(
        class_ids,
        allowed=HOSTILE_CLASS_SET,
        max_distance=VIEWPORT_ROWS + VIEWPORT_COLS,
        limit=3,
    )
    resource_geometry = _relative_tile_descriptors(
        class_ids,
        allowed=RESOURCE_CLASS_SET,
        max_distance=4,
        limit=3,
    )
    regime_labels, primary_regime = infer_local_regime(observation, nearest_threat_distances)
    return {
        "center_patch_ids": center_patch,
        "adjacent_tiles": adjacent_tiles,
        "body_buckets": _body_buckets(observation),
        "inventory_presence": _inventory_presence(observation),
        "nearest_hostile_bucket": _nearest_hostile_bucket(nearest_threat_distances),
        "visible_hostiles": visible_hostiles,
        "resource_tiles": resource_tiles,
        "hostile_geometry": hostile_geometry,
        "resource_geometry": resource_geometry,
        "regime_labels": regime_labels,
        "primary_regime": primary_regime,
        "temporal_signature": dict(observation.get("temporal_signature", {})),
    }


def _state_signature_key(signature: dict[str, Any]) -> str:
    return json.dumps(signature, sort_keys=True, separators=(",", ":"))


def nearest_hostile_distance(threat_distances: dict[str, int | None]) -> int | None:
    values = [
        int(value)
        for value in threat_distances.values()
        if value is not None
    ]
    return min(values) if values else None


def _horizon_label(
    *,
    step: dict[str, Any],
    end_step: dict[str, Any],
    final_body_clean: dict[str, float],
    final_inv_clean: dict[str, int],
    horizon_slice: list[dict[str, Any]],
    used_terminal_fallback: bool,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    died_within_horizon = any(bool(candidate.get("done_after_step", False)) for candidate in horizon_slice)
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
    label = {
        "health_delta_h": health_delta,
        "damage_h": horizon_damage,
        "resource_gain_h": int(resource_gain),
        "inventory_delta_h": inventory_delta,
        "survived_h": not died_within_horizon,
        "escape_delta_h": escape_delta,
        "nearest_hostile_now": start_hostile,
        "nearest_hostile_h": end_hostile,
    }
    return label, dict(start_threats)


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
    belief_tracker = TemporalBeliefTracker()

    for idx, step in enumerate(local_trace):
        end_idx = min(idx + horizon, len(local_trace) - 1)
        end_step = local_trace[end_idx]
        used_terminal_fallback = idx + horizon >= len(local_trace)
        label, start_threats = _horizon_label(
            step=step,
            end_step=end_step,
            final_body_clean=final_body_clean,
            final_inv_clean=final_inv_clean,
            horizon_slice=local_trace[idx:end_idx + 1],
            used_terminal_fallback=used_terminal_fallback,
        )
        temporal_context = belief_tracker.build_context(
            near_concept=str(step.get("near_concept", "empty"))
        )
        observation = _with_temporal_context(step["observation"], temporal_context)
        state_signature = build_state_signature(observation, start_threats)
        regime_labels, primary_regime = infer_local_regime(observation, start_threats)

        examples.append(
            {
                "seed": int(seed),
                "episode_id": int(episode_id),
                "step": int(step["step"]),
                "horizon": int(horizon),
                "action": step["action"],
                "action_index": int(step["action_index"]),
                "plan_origin": step.get("plan_origin"),
                "observation": observation,
                "nearest_threat_distances": dict(start_threats),
                "regime_labels": regime_labels,
                "primary_regime": primary_regime,
                "state_signature": state_signature,
                "state_signature_key": _state_signature_key(state_signature),
                "label_source": "observed_chosen_action",
                "label": label,
                "counterfactual_outcomes": list(step.get("counterfactual_outcomes", [])),
            }
        )
        belief_tracker.observe_transition(
            action=str(step["action"]),
            near_concept=str(step.get("near_concept", "empty")),
            player_pos_before=tuple(step.get("player_pos_before", (32, 32))),
            player_pos_after=tuple(step.get("player_pos_after", step.get("player_pos_before", (32, 32)))),
            body_before=dict(step["observation"].get("body", {})),
            body_after=dict(step.get("body_after", step["observation"].get("body", {}))),
            inventory_before=dict(step["observation"].get("inventory", {})),
            inventory_after=dict(step.get("inventory_after", step["observation"].get("inventory", {}))),
        )

    return examples


def _aggregate_candidate_labels(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("candidate aggregation requires at least one sample")

    escape_values = [
        float(sample["label"]["escape_delta_h"])
        for sample in samples
        if sample["label"]["escape_delta_h"] is not None
    ]
    inventory_delta: dict[str, float] = {}
    for key in INVENTORY_ITEMS:
        deltas = [
            int(sample["label"]["inventory_delta_h"].get(key, 0))
            for sample in samples
        ]
        mean_delta = sum(deltas) / len(deltas)
        if abs(mean_delta) > 1e-6:
            inventory_delta[key] = round(mean_delta, 3)

    return {
        "health_delta_h": round(
            sum(float(sample["label"]["health_delta_h"]) for sample in samples) / len(samples),
            3,
        ),
        "damage_h": round(
            sum(float(sample["label"]["damage_h"]) for sample in samples) / len(samples),
            3,
        ),
        "resource_gain_h": round(
            sum(float(sample["label"]["resource_gain_h"]) for sample in samples) / len(samples),
            3,
        ),
        "inventory_delta_h": inventory_delta,
        "survived_h": round(
            sum(float(sample["label"]["survived_h"]) for sample in samples) / len(samples),
            3,
        ),
        "escape_delta_h": (
            round(sum(escape_values) / len(escape_values), 3)
            if escape_values
            else None
        ),
        "escape_valid_fraction": round(len(escape_values) / len(samples), 3),
        "nearest_hostile_now": samples[0]["label"].get("nearest_hostile_now"),
        "nearest_hostile_h": round(
            sum(
                float(sample["label"]["nearest_hostile_h"])
                for sample in samples
                if sample["label"].get("nearest_hostile_h") is not None
            )
            / max(
                1,
                sum(1 for sample in samples if sample["label"].get("nearest_hostile_h") is not None),
            ),
            3,
        )
        if any(sample["label"].get("nearest_hostile_h") is not None for sample in samples)
        else None,
    }


def _aggregate_counterfactual_outcomes(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    if not outcomes:
        raise ValueError("counterfactual aggregation requires at least one outcome")
    label_wrapped = [{"label": outcome["label"]} for outcome in outcomes]
    aggregated = _aggregate_candidate_labels(label_wrapped)
    confidences = [float(outcome.get("mean_confidence", 0.0)) for outcome in outcomes]
    aggregated["counterfactual_mean_confidence"] = round(
        sum(confidences) / max(len(confidences), 1),
        4,
    )
    aggregated["counterfactual_supported_fraction"] = round(
        sum(1.0 for confidence in confidences if confidence >= 0.2) / max(len(confidences), 1),
        3,
    )
    return aggregated


def _counterfactual_label_is_supported(
    label: dict[str, Any],
    *,
    min_supported_fraction: float = 0.001,
) -> bool:
    return float(label.get("counterfactual_supported_fraction", 0.0)) >= min_supported_fraction


def build_state_centered_training_examples(
    samples: list[dict[str, Any]],
    *,
    candidate_actions: tuple[str, ...] = LOCAL_CORE_ACTIONS,
) -> list[dict[str, Any]]:
    """Group flat chosen-action samples into state-centered action-comparison buckets."""
    grouped: dict[str, dict[str, Any]] = {}

    for sample in samples:
        key = str(sample["state_signature_key"])
        group = grouped.setdefault(
            key,
            {
                "state_signature_key": key,
                "state_signature": sample["state_signature"],
                "observation": sample["observation"],
                "nearest_threat_distances": sample["nearest_threat_distances"],
                "regime_labels": list(sample["regime_labels"]),
                "primary_regime": sample["primary_regime"],
                "members": [],
                "actions": {action: [] for action in candidate_actions},
                "counterfactuals": {action: [] for action in candidate_actions},
            },
        )
        group["members"].append(sample)
        if sample["action"] in group["actions"]:
            group["actions"][sample["action"]].append(sample)
        for outcome in sample.get("counterfactual_outcomes", []):
            action = str(outcome.get("action"))
            if action not in group["counterfactuals"]:
                continue
            group["counterfactuals"][action].append(
                {
                    **outcome,
                    "seed": int(sample["seed"]),
                    "episode_id": int(sample["episode_id"]),
                    "step": int(sample["step"]),
                }
            )

    state_samples: list[dict[str, Any]] = []
    for state_id, group in enumerate(grouped.values()):
        representative = min(
            group["members"],
            key=lambda item: (int(item["seed"]), int(item["episode_id"]), int(item["step"])),
        )
        candidate_rows: list[dict[str, Any]] = []
        for action in candidate_actions:
            action_samples = group["actions"][action]
            counterfactual_outcomes = group["counterfactuals"][action]
            if counterfactual_outcomes:
                observed_fallback = (
                    _aggregate_candidate_labels(action_samples)
                    if action_samples
                    else None
                )
                aggregated_counterfactual = _aggregate_counterfactual_outcomes(counterfactual_outcomes)
                if _counterfactual_label_is_supported(aggregated_counterfactual):
                    candidate_rows.append(
                        {
                            "action": action,
                            "action_index": int(ACTION_TO_IDX[action]),
                            "label": aggregated_counterfactual,
                            "support": len(counterfactual_outcomes),
                            "source": "counterfactual_local_rollout",
                            "comparison_priority": "counterfactual",
                            "counterfactual_support": len(counterfactual_outcomes),
                            "observed_support": len(action_samples),
                            "observed_label_fallback": observed_fallback,
                            "support_refs": [
                                {
                                    "seed": int(outcome["seed"]),
                                    "episode_id": int(outcome["episode_id"]),
                                    "step": int(outcome["step"]),
                                }
                                for outcome in counterfactual_outcomes[:5]
                            ],
                        }
                    )
                    continue
                if observed_fallback is None:
                    continue
            if not action_samples:
                continue
            candidate_rows.append(
                {
                    "action": action,
                    "action_index": int(ACTION_TO_IDX[action]),
                    "label": _aggregate_candidate_labels(action_samples),
                    "support": len(action_samples),
                    "counterfactual_support": 0,
                    "observed_support": len(action_samples),
                    "source": (
                        "matched_state_bucket"
                        if len(action_samples) > 1 or len(group["members"]) > 1
                        else "observed_singleton"
                    ),
                    "comparison_priority": "observed",
                    "support_refs": [
                        {
                            "seed": int(sample["seed"]),
                            "episode_id": int(sample["episode_id"]),
                            "step": int(sample["step"]),
                        }
                        for sample in action_samples[:5]
                    ],
                }
            )
        candidate_rows.sort(key=lambda item: item["action_index"])
        state_samples.append(
            {
                "state_id": int(state_id),
                "state_signature_key": group["state_signature_key"],
                "state_signature": group["state_signature"],
                "observation": group["observation"],
                "nearest_threat_distances": group["nearest_threat_distances"],
                "regime_labels": group["regime_labels"],
                "primary_regime": group["primary_regime"],
                "support": len(group["members"]),
                "chosen_action_support": {
                    action: len(group["actions"][action])
                    for action in candidate_actions
                    if group["actions"][action]
                },
                "candidate_actions": candidate_rows,
                "comparison_coverage": {
                    "n_candidate_actions": len(candidate_rows),
                    "n_counterfactual_actions": sum(
                        1
                        for candidate in candidate_rows
                        if candidate.get("comparison_priority") == "counterfactual"
                    ),
                    "missing_actions": [
                        action
                        for action in candidate_actions
                        if not group["actions"][action] and not group["counterfactuals"][action]
                    ],
                },
                "representative_ref": {
                    "seed": int(representative["seed"]),
                    "episode_id": int(representative["episode_id"]),
                    "step": int(representative["step"]),
                    "chosen_action": representative["action"],
                    "plan_origin": representative.get("plan_origin"),
                },
            }
        )

    state_samples.sort(
        key=lambda item: (
            item["representative_ref"]["seed"],
            item["representative_ref"]["episode_id"],
            item["representative_ref"]["step"],
        )
    )
    return state_samples


def build_local_trace_entry(
    *,
    step: int,
    vf: VisualField,
    body: dict[str, float],
    inventory: dict[str, Any],
    primitive: str,
    plan_origin: str,
    nearest_threat_distances: dict[str, int | None],
    near_concept: str,
    player_pos_before: tuple[int, int],
    player_pos_after: tuple[int, int],
    body_after: dict[str, float],
    inventory_after: dict[str, Any],
    counterfactual_outcomes: list[dict[str, Any]] | None = None,
    done_after_step: bool,
) -> dict[str, Any]:
    return {
        "step": int(step),
        "action": primitive,
        "action_index": int(ACTION_TO_IDX.get(primitive, 0)),
        "plan_origin": plan_origin,
        "observation": build_local_observation_package(vf, body, inventory),
        "near_concept": str(near_concept),
        "player_pos_before": [int(player_pos_before[0]), int(player_pos_before[1])],
        "player_pos_after": [int(player_pos_after[0]), int(player_pos_after[1])],
        "body_after": {key: round(float(body_after.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS},
        "inventory_after": {
            key: int(value)
            for key, value in inventory_after.items()
            if int(value) != 0
        },
        "nearest_threat_distances": {
            key: (
                int(nearest_threat_distances.get(key))
                if nearest_threat_distances.get(key) is not None
                else None
            )
            for key in LOCAL_HOSTILE_KEYS
        },
        "counterfactual_outcomes": list(counterfactual_outcomes or []),
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
        "core_action_names": list(LOCAL_CORE_ACTIONS),
        "temporal_feature_names": list(_TEMPORAL_FEATURE_NAMES),
        "regime_labels": [
            "hostile_contact",
            "hostile_near",
            "low_vitals",
            "local_resource_facing",
            "neutral",
        ],
        "horizon": int(horizon),
    }
