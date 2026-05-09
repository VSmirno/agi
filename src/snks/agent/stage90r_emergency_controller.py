"""Stage 90R first-class emergency safety controller.

The controller is intentionally local and symbolic. It consumes current
runtime state, textbook/config facts, and one-step candidate evidence, then
decides whether to override the normal learner/planner path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_DAMAGE_BUCKET_VALUES = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.6,
    "high": 1.0,
}


@dataclass(frozen=True)
class EmergencyWorldFacts:
    hostile_concepts: tuple[str, ...]
    resource_concepts: tuple[str, ...] = ()
    threat_ranges: dict[str, int] = field(default_factory=dict)
    movement_behaviors: dict[str, str] = field(default_factory=dict)
    activation_threshold: float = 1.0
    default_hostile_emergency_range: int = 1
    source: str = "fallback"

    @classmethod
    def from_textbook(cls, textbook: Any | None) -> "EmergencyWorldFacts":
        data = getattr(textbook, "data", None)
        if not isinstance(data, dict):
            return cls(
                hostile_concepts=("zombie", "skeleton", "arrow"),
                resource_concepts=("water", "tree", "stone", "coal", "iron", "diamond", "cow"),
                threat_ranges={"zombie": 1, "skeleton": 5, "arrow": 1},
                movement_behaviors={"zombie": "chase_player", "skeleton": "chase_player"},
                source="fallback_no_textbook",
            )

        emergency = getattr(textbook, "emergency_control_block", None)
        block = emergency if isinstance(emergency, dict) else data.get("emergency_control", {})
        block = block if isinstance(block, dict) else {}
        hostile_categories = {
            str(category)
            for category in block.get("hostile_categories", ("enemy", "projectile"))
        }
        resource_categories = {
            str(category)
            for category in block.get("resource_categories", ("resource",))
        }
        hostile: set[str] = set()
        resources: set[str] = set()
        for entry in data.get("vocabulary", []):
            if not isinstance(entry, dict):
                continue
            cid = str(entry.get("id", ""))
            if not cid:
                continue
            category = str(entry.get("category", ""))
            if category in hostile_categories or bool(entry.get("dangerous", False)):
                hostile.add(cid)
            if category in resource_categories:
                resources.add(cid)

        threat_ranges: dict[str, int] = {}
        movement_behaviors: dict[str, str] = {}
        for rule in data.get("rules", []):
            if not isinstance(rule, dict):
                continue
            entity = str(rule.get("entity", ""))
            if not entity:
                continue
            if rule.get("passive") == "spatial":
                threat_ranges[entity] = int(rule.get("range", block.get("default_hostile_emergency_range", 1)))
            elif rule.get("passive") == "movement":
                movement_behaviors[entity] = str(rule.get("behavior", "unknown"))

        return cls(
            hostile_concepts=tuple(sorted(hostile)),
            resource_concepts=tuple(sorted(resources)),
            threat_ranges=threat_ranges,
            movement_behaviors=movement_behaviors,
            activation_threshold=float(block.get("activation_threshold", 1.0)),
            default_hostile_emergency_range=int(block.get("default_hostile_emergency_range", 1)),
            source="textbook.emergency_control",
        )

    def emergency_range(self, concept_id: str) -> int:
        return int(self.threat_ranges.get(concept_id, self.default_hostile_emergency_range))


@dataclass(frozen=True)
class EmergencyFeatures:
    score: float
    activated: bool
    primary_reason: str | None
    reasons: tuple[str, ...]
    values: dict[str, Any]


@dataclass(frozen=True)
class EmergencySelection:
    action: str
    override_source: str
    reason: str
    utility_components: dict[str, Any]
    ranked_actions: tuple[dict[str, Any], ...]


class EmergencySafetyController:
    """Explicit emergency activation and safety-first action selection."""

    def __init__(
        self,
        *,
        facts: EmergencyWorldFacts,
        low_vitals_threshold: float = 4.0,
        hostile_distance_threshold: int = 1,
        stall_streak_threshold: int = 2,
    ) -> None:
        self.facts = facts
        self.low_vitals_threshold = float(low_vitals_threshold)
        self.hostile_distance_threshold = int(hostile_distance_threshold)
        self.stall_streak_threshold = int(stall_streak_threshold)

    def evaluate(
        self,
        *,
        body: dict[str, float],
        nearest_threat_distances: dict[str, int | None],
        actor_non_progress_streak: int,
        planner_action: str,
        current_action: str,
        learner_action: str | None,
        belief_state_signature: dict[str, Any] | None = None,
        candidate_outcomes: list[dict[str, Any]] | None = None,
        predicted_baseline_loss: float = 0.0,
        predicted_selected_loss: float = 0.0,
    ) -> EmergencyFeatures:
        min_vital = (
            min(float(body.get(key, 9.0)) for key in ("health", "food", "drink", "energy"))
            if body
            else 9.0
        )
        vital_pressure = max(
            0.0,
            min(1.0, (self.low_vitals_threshold - min_vital) / max(self.low_vitals_threshold, 1e-6)),
        )

        threat_pressures: dict[str, float] = {}
        for concept in self.facts.hostile_concepts:
            distance = nearest_threat_distances.get(concept)
            if distance is None:
                continue
            emergency_range = max(
                self.hostile_distance_threshold,
                self.facts.emergency_range(concept),
            )
            if int(distance) <= emergency_range:
                pressure = 1.0
            elif int(distance) <= emergency_range + 2:
                pressure = 0.55
            else:
                pressure = 0.0
            if pressure > 0.0:
                threat_pressures[concept] = pressure
        hostile_pressure = max(threat_pressures.values(), default=0.0)
        nearest_hostile = _nearest_distance(nearest_threat_distances)

        belief_signature = dict(belief_state_signature or {})
        recent_damage_pressure = _DAMAGE_BUCKET_VALUES.get(
            str(belief_signature.get("damage_pressure_bucket", "none")),
            0.0,
        )
        no_progress_pressure = max(
            0.0,
            min(1.0, float(actor_non_progress_streak) / max(float(self.stall_streak_threshold), 1.0)),
        )
        candidate_damage = _best_candidate_damage(candidate_outcomes or [])
        evaluator_pressure = max(
            0.0,
            min(1.0, max(float(predicted_baseline_loss), candidate_damage) / 2.0),
        )
        current_action_damage = _candidate_damage_for_action(candidate_outcomes or [], current_action)
        planner_disagreement = bool(learner_action is not None and learner_action != planner_action)

        weighted_score = (
            1.25 * vital_pressure
            + 1.4 * hostile_pressure
            + 0.75 * recent_damage_pressure
            + 0.85 * no_progress_pressure
            + 0.9 * evaluator_pressure
            + (0.15 if planner_disagreement else 0.0)
        )
        reasons: list[str] = []
        if vital_pressure >= 0.01:
            reasons.append("low_vitals")
        if hostile_pressure >= 1.0:
            reasons.append("hostile_contact")
        elif hostile_pressure > 0.0:
            reasons.append("hostile_near")
        if recent_damage_pressure >= 0.25:
            reasons.append("recent_damage")
        if no_progress_pressure >= 1.0:
            reasons.append("repeated_non_progress")
        if evaluator_pressure >= 0.5:
            reasons.append("evaluator_predicted_damage")
        if planner_disagreement:
            reasons.append("planner_learner_disagreement_feature")

        activated = weighted_score >= self.facts.activation_threshold
        return EmergencyFeatures(
            score=round(float(weighted_score), 4),
            activated=bool(activated),
            primary_reason=reasons[0] if activated and reasons else None,
            reasons=tuple(reasons),
            values={
                "min_vital": round(float(min_vital), 3),
                "vital_pressure": round(float(vital_pressure), 4),
                "nearest_hostile": nearest_hostile,
                "nearest_threat_distances": dict(nearest_threat_distances),
                "hostile_pressure": round(float(hostile_pressure), 4),
                "threat_pressures": threat_pressures,
                "recent_damage_pressure": round(float(recent_damage_pressure), 4),
                "actor_non_progress_streak": int(actor_non_progress_streak),
                "no_progress_pressure": round(float(no_progress_pressure), 4),
                "evaluator_pressure": round(float(evaluator_pressure), 4),
                "candidate_damage_min": round(float(candidate_damage), 4),
                "current_action_predicted_damage": current_action_damage,
                "predicted_baseline_loss": round(float(predicted_baseline_loss), 4),
                "predicted_selected_loss": round(float(predicted_selected_loss), 4),
                "planner_disagreement_feature": planner_disagreement,
                "planner_action": planner_action,
                "learner_action": learner_action,
                "current_action": current_action,
                "facts_source": self.facts.source,
            },
        )

    def select_action(
        self,
        *,
        current_action: str,
        planner_action: str,
        learner_action: str | None,
        candidate_outcomes: list[dict[str, Any]] | None = None,
        advisory_ranked: list[dict[str, Any]] | None = None,
        allowed_actions: list[str] | tuple[str, ...] = (),
    ) -> EmergencySelection:
        allowed = [str(action) for action in allowed_actions] or [
            "move_left",
            "move_right",
            "move_up",
            "move_down",
            "do",
            "sleep",
        ]
        outcome_by_action = {
            str(outcome.get("action")): outcome
            for outcome in candidate_outcomes or []
            if str(outcome.get("action")) in allowed
        }
        advisory_rank_by_action = {
            str(candidate.get("action")): rank
            for rank, candidate in enumerate(advisory_ranked or [])
        }
        ranked: list[dict[str, Any]] = []
        for action in allowed:
            label = dict(outcome_by_action.get(action, {}).get("label", {}))
            survived = bool(label.get("survived_h", True))
            damage = float(label.get("damage_h", 0.0))
            health_delta = float(label.get("health_delta_h", 0.0))
            escape_delta_raw = label.get("escape_delta_h")
            escape_delta = 0.0 if escape_delta_raw is None else float(escape_delta_raw)
            nearest_h = label.get("nearest_hostile_h")
            effective_displacement = float(label.get("effective_displacement_h", 0.0))
            blocked = bool(label.get("blocked_h", False))
            adjacent_after = bool(label.get("adjacent_hostile_after_h", False))
            resource_gain = float(label.get("resource_gain_h", 0.0))
            advisory_rank = advisory_rank_by_action.get(action)
            advisory_bonus = 0.0 if advisory_rank is None else max(0.0, 0.6 - 0.15 * advisory_rank)
            planner_bonus = 0.2 if action == planner_action else 0.0
            learner_penalty = -0.1 if learner_action is not None and action == learner_action else 0.0
            sleep_threat_penalty = -1.5 if action == "sleep" and label.get("nearest_hostile_now") is not None else 0.0
            blocked_penalty = -3.0 if blocked else 0.0
            adjacent_penalty = -2.0 if adjacent_after else 0.0
            displacement_bonus = min(1.0, 0.5 * effective_displacement)
            utility = (
                (6.0 if survived else -12.0)
                - 4.0 * damage
                + 1.75 * escape_delta
                + 0.5 * health_delta
                + min(0.5, 0.1 * resource_gain)
                + advisory_bonus
                + planner_bonus
                + learner_penalty
                + sleep_threat_penalty
                + blocked_penalty
                + adjacent_penalty
                + displacement_bonus
            )
            ranked.append(
                {
                    "action": action,
                    "utility": round(float(utility), 4),
                    "components": {
                        "survived_h": survived,
                        "damage_h": round(float(damage), 4),
                        "health_delta_h": round(float(health_delta), 4),
                        "escape_delta_h": None if escape_delta_raw is None else round(float(escape_delta), 4),
                        "nearest_hostile_h": nearest_h,
                        "effective_displacement_h": round(float(effective_displacement), 4),
                        "blocked_h": blocked,
                        "adjacent_hostile_after_h": adjacent_after,
                        "resource_gain_h": round(float(resource_gain), 4),
                        "advisory_rank": advisory_rank,
                        "planner_aligned": action == planner_action,
                        "learner_aligned": learner_action is not None and action == learner_action,
                    },
                }
            )
        ranked.sort(key=lambda item: float(item["utility"]), reverse=True)
        selected = str(ranked[0]["action"]) if ranked else current_action
        advisory_best = str((advisory_ranked or [{}])[0].get("action", ""))
        if selected == planner_action:
            source = "planner_aligned_safety"
        elif advisory_best and selected == advisory_best:
            source = "advisory_aligned_safety"
        elif selected == learner_action:
            source = "learner_aligned_safety"
        else:
            source = "independent_emergency_choice"
        return EmergencySelection(
            action=selected,
            override_source=source,
            reason="safety_first_candidate_ranking",
            utility_components=dict(ranked[0].get("components", {})) if ranked else {},
            ranked_actions=tuple(ranked[:4]),
        )


def _nearest_distance(distances: dict[str, int | None]) -> int | None:
    values = [int(value) for value in distances.values() if value is not None]
    return min(values) if values else None


def _best_candidate_damage(candidate_outcomes: list[dict[str, Any]]) -> float:
    damages = [
        float(dict(outcome.get("label", {})).get("damage_h", 0.0))
        for outcome in candidate_outcomes
    ]
    return min(damages) if damages else 0.0


def _candidate_damage_for_action(
    candidate_outcomes: list[dict[str, Any]],
    action: str,
) -> float | None:
    for outcome in candidate_outcomes:
        if str(outcome.get("action")) != str(action):
            continue
        return round(float(dict(outcome.get("label", {})).get("damage_h", 0.0)), 4)
    return None
