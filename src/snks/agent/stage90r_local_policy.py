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
_BELIEF_TRANSITION_WINDOW = 4
_BELIEF_STREAK_CLIP = 4
_BELIEF_PROGRESS_CLIP = 4.0
_BELIEF_STALL_CLIP = 4.0
_BELIEF_THREAT_TREND_CLIP = 4.0
_BELIEF_RESOURCE_CLIP = 4.0
_BELIEF_DAMAGE_CLIP = 4.0
_BELIEF_FEATURE_NAMES = [
    "belief_affordance_stability_norm",
    "belief_progress_norm",
    "belief_stall_risk_norm",
    "belief_threat_trend_norm",
    "belief_resource_flow_norm",
    "belief_damage_pressure_norm",
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


class BeliefStateEncoder:
    """Compact state-evolution-centric belief state for local world-model ranking."""

    def __init__(self) -> None:
        self._prev_near_concept: str | None = None
        self._near_concept_streak = 0
        self._recent_progress: deque[float] = deque(maxlen=_BELIEF_TRANSITION_WINDOW)
        self._recent_stall: deque[float] = deque(maxlen=_BELIEF_TRANSITION_WINDOW)
        self._recent_threat_trend: deque[float] = deque(maxlen=_BELIEF_TRANSITION_WINDOW)
        self._recent_resource_flow: deque[float] = deque(maxlen=_BELIEF_TRANSITION_WINDOW)
        self._recent_damage_pressure: deque[float] = deque(maxlen=_BELIEF_TRANSITION_WINDOW)
        self._prev_nearest_threat: int | None = None

    def build_context(self, *, near_concept: str | None) -> dict[str, Any]:
        current_near = str(near_concept or "empty")
        near_streak = self._near_concept_streak + 1 if current_near == self._prev_near_concept else 0
        affordance_stability = (
            _clip_positive(near_streak, limit=float(_BELIEF_STREAK_CLIP))
            if current_near in RESOURCE_CLASS_SET
            else 0.0
        )
        vector = [
            affordance_stability,
            _clip_positive(sum(self._recent_progress), limit=_BELIEF_PROGRESS_CLIP),
            _clip_positive(sum(self._recent_stall), limit=_BELIEF_STALL_CLIP),
            _clip_signed(sum(self._recent_threat_trend), limit=_BELIEF_THREAT_TREND_CLIP),
            _clip_positive(sum(self._recent_resource_flow), limit=_BELIEF_RESOURCE_CLIP),
            _clip_positive(sum(self._recent_damage_pressure), limit=_BELIEF_DAMAGE_CLIP),
        ]
        return {
            "vector": vector,
            "feature_names": list(_BELIEF_FEATURE_NAMES),
            "signature": {
                "affordance_stability_bucket": _magnitude_bucket(
                    affordance_stability * float(_BELIEF_STREAK_CLIP),
                    positive_only=True,
                ),
                "progress_bucket": _magnitude_bucket(
                    sum(self._recent_progress),
                    positive_only=True,
                ),
                "stall_bucket": _magnitude_bucket(
                    sum(self._recent_stall),
                    positive_only=True,
                ),
                "threat_trend_bucket": _magnitude_bucket(
                    sum(self._recent_threat_trend),
                    positive_only=False,
                ),
                "resource_flow_bucket": _magnitude_bucket(
                    sum(self._recent_resource_flow),
                    positive_only=True,
                ),
                "damage_pressure_bucket": _magnitude_bucket(
                    sum(self._recent_damage_pressure),
                    positive_only=True,
                ),
            },
        }

    def observe_transition(
        self,
        *,
        near_concept: str | None,
        player_pos_before: tuple[int, int] | list[int],
        player_pos_after: tuple[int, int] | list[int],
        body_before: dict[str, float],
        body_after: dict[str, float],
        inventory_before: dict[str, Any],
        inventory_after: dict[str, Any],
        nearest_threat_distance_before: int | None = None,
    ) -> None:
        before_x, before_y = int(player_pos_before[0]), int(player_pos_before[1])
        after_x, after_y = int(player_pos_after[0]), int(player_pos_after[1])
        displacement = abs(after_x - before_x) + abs(after_y - before_y)

        health_before = float(body_before.get("health", 0.0))
        health_after = float(body_after.get("health", 0.0))
        health_delta = health_after - health_before
        damage_pressure = max(0.0, -health_delta)

        resource_gain = 0.0
        for key in set(inventory_before.keys()) | set(inventory_after.keys()):
            delta = int(inventory_after.get(key, 0)) - int(inventory_before.get(key, 0))
            if delta > 0:
                resource_gain += float(delta)
        threat_trend = 0.0
        if nearest_threat_distance_before is not None and self._prev_nearest_threat is not None:
            threat_trend = float(nearest_threat_distance_before - self._prev_nearest_threat)
        if nearest_threat_distance_before is not None:
            self._prev_nearest_threat = int(nearest_threat_distance_before)

        progress_signal = 0.0
        if displacement > 0:
            progress_signal += 0.5
        if resource_gain > 0.0:
            progress_signal += min(0.5, 0.25 * resource_gain)
        if threat_trend > 0.0:
            progress_signal += min(0.25, 0.1 * threat_trend)
        if health_delta > 0.0:
            progress_signal += min(0.25, 0.1 * health_delta)
        progress_signal = min(1.0, progress_signal)
        stall_signal = 1.0 if displacement == 0 and resource_gain <= 0.0 and threat_trend <= 0.0 else 0.0

        self._recent_progress.append(progress_signal)
        self._recent_stall.append(stall_signal)
        self._recent_threat_trend.append(threat_trend)
        self._recent_resource_flow.append(resource_gain)
        self._recent_damage_pressure.append(damage_pressure)

        current_near = str(near_concept or "empty")
        if current_near == self._prev_near_concept:
            self._near_concept_streak += 1
        else:
            self._prev_near_concept = current_near
            self._near_concept_streak = 1


TemporalBeliefTracker = BeliefStateEncoder


def _with_belief_state(
    observation: dict[str, Any],
    belief_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if belief_context is None:
        return {
            **observation,
            "belief_state_vector": [],
            "belief_state_feature_names": list(_BELIEF_FEATURE_NAMES),
            "belief_state_signature": {},
        }
    return {
        **observation,
        "belief_state_vector": list(belief_context.get("vector", [])),
        "belief_state_feature_names": list(belief_context.get("feature_names", _BELIEF_FEATURE_NAMES)),
        "belief_state_signature": dict(belief_context.get("signature", {})),
    }


def _belief_state_record(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "vector": list(observation.get("belief_state_vector", [])),
        "feature_names": list(observation.get("belief_state_feature_names", _BELIEF_FEATURE_NAMES)),
        "signature": dict(observation.get("belief_state_signature", {})),
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
    belief_context: dict[str, Any] | None = None,
    temporal_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    class_ids, confidences = dense_viewport_scene(vf)
    sparse_inventory = {
        key: int(value)
        for key, value in inventory.items()
        if int(value) != 0
    }
    if belief_context is None and temporal_context is not None:
        belief_context = temporal_context
    return _with_belief_state(
        {
            "viewport_class_ids": class_ids,
            "viewport_confidences": confidences,
            "body_vector": encode_body_vector(body),
            "inventory_vector": encode_inventory_vector(inventory),
            "body": {key: round(float(body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS},
            "inventory": sparse_inventory,
        },
        belief_context,
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
        "belief_state_signature": dict(observation.get("belief_state_signature", {})),
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
    start_pos = tuple(step.get("player_pos_before", (32, 32)))
    end_pos = tuple(end_step.get("player_pos_after", end_step.get("player_pos_before", (32, 32))))
    displacement_total = (
        abs(int(end_pos[0]) - int(start_pos[0]))
        + abs(int(end_pos[1]) - int(start_pos[1]))
    )
    threat_trend = (
        round(float(end_hostile - start_hostile), 3)
        if start_hostile is not None and end_hostile is not None
        else 0.0
    )
    progress_delta = 0.0
    if displacement_total > 0:
        progress_delta += 0.5
    if resource_gain > 0:
        progress_delta += min(1.0, 0.5 * float(resource_gain))
    if threat_trend > 0.0:
        progress_delta += min(0.5, 0.25 * float(threat_trend))
    if health_delta > 0.0:
        progress_delta += min(0.25, 0.1 * float(health_delta))
    start_near = str(step.get("near_concept", "empty"))
    horizon_near = [str(candidate.get("near_concept", "empty")) for candidate in horizon_slice[1:]]
    label = {
        "health_delta_h": health_delta,
        "damage_h": horizon_damage,
        "resource_gain_h": int(resource_gain),
        "inventory_delta_h": inventory_delta,
        "survived_h": not died_within_horizon,
        "escape_delta_h": escape_delta,
        "nearest_hostile_now": start_hostile,
        "nearest_hostile_h": end_hostile,
        "progress_delta_h": round(progress_delta, 3),
        "stall_risk_h": (
            1.0
            if displacement_total == 0 and resource_gain == 0 and threat_trend <= 0.0 and health_delta <= 0.0
            else 0.0
        ),
        "affordance_persistence_h": (
            1.0
            if start_near in RESOURCE_CLASS_SET and (resource_gain > 0 or start_near in horizon_near)
            else 0.0
        ),
        "threat_trend_h": threat_trend,
    }
    return label, dict(start_threats)


def _immediate_transition_outcome(
    *,
    step: dict[str, Any],
    next_step: dict[str, Any] | None,
    start_threats: dict[str, int | None],
    next_threats: dict[str, int | None],
) -> dict[str, Any]:
    body_before = dict(step["observation"].get("body", {}))
    body_after = dict(step.get("body_after", body_before))
    inventory_before = dict(step["observation"].get("inventory", {}))
    inventory_after = dict(step.get("inventory_after", inventory_before))
    health_delta = round(
        float(body_after.get("health", 0.0)) - float(body_before.get("health", 0.0)),
        3,
    )
    damage = round(max(0.0, -health_delta), 3)
    inventory_delta: dict[str, int] = {}
    resource_gain = 0
    for key in INVENTORY_ITEMS:
        delta = int(inventory_after.get(key, 0)) - int(inventory_before.get(key, 0))
        if delta != 0:
            inventory_delta[key] = delta
        if delta > 0:
            resource_gain += delta
    start_hostile = nearest_hostile_distance(start_threats)
    end_hostile = nearest_hostile_distance(next_threats)
    escape_delta = (
        int(end_hostile - start_hostile)
        if start_hostile is not None and end_hostile is not None
        else None
    )
    player_pos_before = tuple(step.get("player_pos_before", (32, 32)))
    player_pos_after = tuple(step.get("player_pos_after", player_pos_before))
    displacement = (
        abs(int(player_pos_after[0]) - int(player_pos_before[0]))
        + abs(int(player_pos_after[1]) - int(player_pos_before[1]))
    )
    threat_trend = (
        round(float(end_hostile - start_hostile), 3)
        if start_hostile is not None and end_hostile is not None
        else 0.0
    )
    progress_delta = 0.0
    if displacement > 0:
        progress_delta += 0.5
    if resource_gain > 0:
        progress_delta += min(1.0, 0.5 * float(resource_gain))
    if threat_trend > 0.0:
        progress_delta += min(0.5, 0.25 * float(threat_trend))
    if health_delta > 0.0:
        progress_delta += min(0.25, 0.1 * float(health_delta))
    near_concept = str(step.get("near_concept", "empty"))
    next_near_concept = str(next_step.get("near_concept", near_concept)) if next_step is not None else near_concept
    return {
        "health_delta_step": health_delta,
        "damage_step": damage,
        "resource_gain_step": int(resource_gain),
        "inventory_delta_step": inventory_delta,
        "survived_step": not bool(step.get("done_after_step", False)),
        "escape_delta_step": escape_delta,
        "nearest_hostile_now": start_hostile,
        "nearest_hostile_next": end_hostile,
        "progress_delta_step": round(progress_delta, 3),
        "stall_risk_step": (
            1.0
            if displacement == 0 and resource_gain == 0 and threat_trend <= 0.0 and health_delta <= 0.0
            else 0.0
        ),
        "affordance_persistence_step": (
            1.0
            if near_concept in RESOURCE_CLASS_SET and (resource_gain > 0 or near_concept == next_near_concept)
            else 0.0
        ),
        "threat_trend_step": threat_trend,
        "displacement_step": int(displacement),
        "done_after_step": bool(step.get("done_after_step", False)),
    }


def build_learner_transition_records(
    *,
    local_trace: list[dict[str, Any]],
    final_body: dict[str, float],
    final_inventory: dict[str, Any],
    seed: int,
    episode_id: int,
    horizon: int,
) -> list[dict[str, Any]]:
    """Build first-class learner transition records from experienced local traces."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    final_body_clean = {key: round(float(final_body.get(key, 0.0)), 3) for key in LOCAL_BODY_KEYS}
    final_inv_clean = {key: int(final_inventory.get(key, 0)) for key in INVENTORY_ITEMS}
    records: list[dict[str, Any]] = []
    belief_tracker = BeliefStateEncoder()

    for idx, step in enumerate(local_trace):
        end_idx = min(idx + horizon, len(local_trace) - 1)
        end_step = local_trace[end_idx]
        used_terminal_fallback = idx + horizon >= len(local_trace)
        horizon_outcome, start_threats = _horizon_label(
            step=step,
            end_step=end_step,
            final_body_clean=final_body_clean,
            final_inv_clean=final_inv_clean,
            horizon_slice=local_trace[idx:end_idx + 1],
            used_terminal_fallback=used_terminal_fallback,
        )
        belief_context = belief_tracker.build_context(
            near_concept=str(step.get("near_concept", "empty"))
        )
        observation = _with_belief_state(step["observation"], belief_context)
        state_signature = build_state_signature(observation, start_threats)
        regime_labels, primary_regime = infer_local_regime(observation, start_threats)

        next_step = local_trace[idx + 1] if idx + 1 < len(local_trace) else None
        next_threats = (
            dict(next_step.get("nearest_threat_distances", {}))
            if next_step is not None
            else dict(start_threats)
        )
        immediate_outcome = _immediate_transition_outcome(
            step=step,
            next_step=next_step,
            start_threats=start_threats,
            next_threats=next_threats,
        )

        belief_tracker.observe_transition(
            near_concept=str(step.get("near_concept", "empty")),
            player_pos_before=tuple(step.get("player_pos_before", (32, 32))),
            player_pos_after=tuple(step.get("player_pos_after", step.get("player_pos_before", (32, 32)))),
            body_before=dict(step["observation"].get("body", {})),
            body_after=dict(step.get("body_after", step["observation"].get("body", {}))),
            inventory_before=dict(step["observation"].get("inventory", {})),
            inventory_after=dict(step.get("inventory_after", step["observation"].get("inventory", {}))),
            nearest_threat_distance_before=nearest_hostile_distance(start_threats),
        )
        next_belief_context = belief_tracker.build_context(
            near_concept=(
                str(next_step.get("near_concept", "empty"))
                if next_step is not None
                else str(step.get("near_concept", "empty"))
            )
        )
        next_observation = (
            _with_belief_state(next_step["observation"], next_belief_context)
            if next_step is not None
            else None
        )

        records.append(
            {
                "seed": int(seed),
                "episode_id": int(episode_id),
                "step": int(step["step"]),
                "horizon": int(horizon),
                "controller": str(step.get("controller", "planner_bootstrap")),
                "action": step["action"],
                "action_index": int(step["action_index"]),
                "plan_origin": step.get("plan_origin"),
                "planner_action": step.get("planner_action", step["action"]),
                "planner_action_index": int(step.get("planner_action_index", step["action_index"])),
                "learner_action": step.get("learner_action"),
                "learner_action_index": step.get("learner_action_index"),
                "rescue_applied": bool(step.get("rescue_applied", False)),
                "rescue_trigger": step.get("rescue_trigger"),
                "observation": observation,
                "belief_state": _belief_state_record(observation),
                "next_observation": next_observation,
                "next_belief_state": (
                    _belief_state_record(next_observation)
                    if next_observation is not None
                    else {
                        "vector": list(next_belief_context.get("vector", [])),
                        "feature_names": list(next_belief_context.get("feature_names", _BELIEF_FEATURE_NAMES)),
                        "signature": dict(next_belief_context.get("signature", {})),
                    }
                ),
                "next_observation_available": next_observation is not None,
                "terminated": bool(step.get("done_after_step", False)),
                "nearest_threat_distances": dict(start_threats),
                "next_nearest_threat_distances": dict(next_threats),
                "regime_labels": regime_labels,
                "primary_regime": primary_regime,
                "state_signature": state_signature,
                "state_signature_key": _state_signature_key(state_signature),
                "immediate_outcome": immediate_outcome,
                "horizon_outcome": horizon_outcome,
                "auxiliary_counterfactual_probe_count": len(step.get("counterfactual_outcomes", [])),
                "auxiliary_counterfactual_probes": list(step.get("counterfactual_outcomes", [])),
            }
        )

    return records


def build_planner_teacher_records(
    learner_transition_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project planner-controlled transitions into teacher records for later actor bootstrap."""
    records: list[dict[str, Any]] = []
    for transition in learner_transition_records:
        records.append(
            {
                "seed": int(transition["seed"]),
                "episode_id": int(transition["episode_id"]),
                "step": int(transition["step"]),
                "teacher_policy": "planner",
                "teacher_mode": (
                    "planner_rescue"
                    if bool(transition.get("rescue_applied", False))
                    else (
                        "planner_controlled_bootstrap"
                        if str(transition.get("controller")) == "planner_bootstrap"
                        else "mixed_control_teacher_reference"
                    )
                ),
                "planner_action": str(transition.get("planner_action", transition["action"])),
                "planner_action_index": int(transition.get("planner_action_index", transition["action_index"])),
                "planner_plan_origin": transition.get("plan_origin"),
                "learner_action": transition.get("learner_action"),
                "learner_action_index": transition.get("learner_action_index"),
                "learner_action_matches_planner": (
                    None
                    if transition.get("learner_action") is None
                    else str(transition.get("learner_action")) == str(transition.get("planner_action", transition["action"]))
                ),
                "observation": transition["observation"],
                "belief_state": transition["belief_state"],
                "next_observation": transition["next_observation"],
                "next_belief_state": transition["next_belief_state"],
                "next_observation_available": bool(transition["next_observation_available"]),
                "regime_labels": list(transition.get("regime_labels", [])),
                "primary_regime": transition.get("primary_regime"),
                "state_signature_key": transition.get("state_signature_key"),
                "resulting_outcome": dict(transition["horizon_outcome"]),
                "auxiliary_counterfactual_probe_count": int(transition.get("auxiliary_counterfactual_probe_count", 0)),
            }
        )
    return records


def build_rescue_records(
    *,
    rescue_trace: list[dict[str, Any]],
    seed: int,
    episode_id: int,
) -> list[dict[str, Any]]:
    """Normalize explicit rescue events for the hybrid dataset contract."""
    records: list[dict[str, Any]] = []
    for idx, event in enumerate(rescue_trace):
        records.append(
            {
                "seed": int(seed),
                "episode_id": int(episode_id),
                "rescue_index": int(idx),
                "step": int(event.get("step", idx)),
                "trigger": str(event.get("trigger", "unknown")),
                "planner_action": event.get("planner_action"),
                "learner_action": event.get("learner_action"),
                "rescue_applied": bool(event.get("rescue_applied", True)),
                "rescue_improved_outcome": event.get("rescue_improved_outcome"),
                "pre_rescue_state": dict(event.get("pre_rescue_state", {})),
                "post_rescue_outcome": dict(event.get("post_rescue_outcome", {})),
            }
        )
    return records


def build_auxiliary_counterfactual_probe_records(
    learner_transition_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Flatten short local counterfactual probes into auxiliary-only records."""
    probes: list[dict[str, Any]] = []
    for transition in learner_transition_records:
        for probe_idx, outcome in enumerate(transition.get("auxiliary_counterfactual_probes", [])):
            probes.append(
                {
                    "seed": int(transition["seed"]),
                    "episode_id": int(transition["episode_id"]),
                    "step": int(transition["step"]),
                    "probe_index": int(probe_idx),
                    "state_signature_key": transition.get("state_signature_key"),
                    "primary_regime": transition.get("primary_regime"),
                    "observation": transition["observation"],
                    "belief_state": transition["belief_state"],
                    "plan_origin": transition.get("plan_origin"),
                    "action": str(outcome.get("action", "unknown")),
                    "action_index": int(ACTION_TO_IDX.get(str(outcome.get("action", "move_right")), 0)),
                    "mean_confidence": round(float(outcome.get("mean_confidence", 0.0)), 4),
                    "label": dict(outcome.get("label", {})),
                    "source": "auxiliary_counterfactual_probe",
                }
            )
    return probes


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
    records = build_learner_transition_records(
        local_trace=local_trace,
        final_body=final_body,
        final_inventory=final_inventory,
        seed=seed,
        episode_id=episode_id,
        horizon=horizon,
    )
    return [
        {
            "seed": int(record["seed"]),
            "episode_id": int(record["episode_id"]),
            "step": int(record["step"]),
            "horizon": int(record["horizon"]),
            "action": record["action"],
            "action_index": int(record["action_index"]),
            "plan_origin": record.get("plan_origin"),
            "observation": record["observation"],
            "nearest_threat_distances": dict(record["nearest_threat_distances"]),
            "regime_labels": list(record["regime_labels"]),
            "primary_regime": record["primary_regime"],
            "state_signature": record["state_signature"],
            "state_signature_key": record["state_signature_key"],
            "label_source": "observed_chosen_action",
            "label": dict(record["horizon_outcome"]),
            "counterfactual_outcomes": list(record.get("auxiliary_counterfactual_probes", [])),
        }
        for record in records
    ]


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
    resource_gain = round(sum(max(0.0, float(delta)) for delta in inventory_delta.values()), 3)

    return {
        "health_delta_h": round(
            sum(float(sample["label"]["health_delta_h"]) for sample in samples) / len(samples),
            3,
        ),
        "damage_h": round(
            sum(float(sample["label"]["damage_h"]) for sample in samples) / len(samples),
            3,
        ),
        "resource_gain_h": resource_gain,
        "progress_delta_h": round(
            sum(float(sample["label"].get("progress_delta_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "stall_risk_h": round(
            sum(float(sample["label"].get("stall_risk_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "affordance_persistence_h": round(
            sum(float(sample["label"].get("affordance_persistence_h", 0.0)) for sample in samples) / len(samples),
            3,
        ),
        "threat_trend_h": round(
            sum(float(sample["label"].get("threat_trend_h", 0.0)) for sample in samples) / len(samples),
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
    controller: str = "planner_bootstrap",
    planner_action: str | None = None,
    planner_action_index: int | None = None,
    learner_action: str | None = None,
    learner_action_index: int | None = None,
    rescue_applied: bool = False,
    rescue_trigger: str | None = None,
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
        "controller": str(controller),
        "planner_action": str(planner_action or primitive),
        "planner_action_index": int(
            planner_action_index
            if planner_action_index is not None
            else ACTION_TO_IDX.get(str(planner_action or primitive), 0)
        ),
        "learner_action": str(learner_action) if learner_action is not None else None,
        "learner_action_index": (
            int(learner_action_index)
            if learner_action_index is not None
            else None
        ),
        "rescue_applied": bool(rescue_applied),
        "rescue_trigger": str(rescue_trigger) if rescue_trigger is not None else None,
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
        "dataset_contract": {
            "version": "stage90r_planner_teacher_hybrid_slice_a_v1",
            "primary_record_types": [
                "learner_transition_records",
                "planner_teacher_records",
                "rescue_records",
            ],
            "auxiliary_record_types": [
                "auxiliary_counterfactual_probes",
                "auxiliary_action_samples",
                "auxiliary_state_samples",
            ],
        },
        "viewport_rows": VIEWPORT_ROWS,
        "viewport_cols": VIEWPORT_COLS,
        "near_classes": list(NEAR_CLASSES),
        "body_keys": list(LOCAL_BODY_KEYS),
        "inventory_keys": list(INVENTORY_ITEMS),
        "action_names": list(ACTION_NAMES),
        "core_action_names": list(LOCAL_CORE_ACTIONS),
        "belief_state_feature_names": list(_BELIEF_FEATURE_NAMES),
        "regime_labels": [
            "hostile_contact",
            "hostile_near",
            "low_vitals",
            "local_resource_facing",
            "neutral",
        ],
        "horizon": int(horizon),
    }
