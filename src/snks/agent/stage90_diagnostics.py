"""Stage 90 diagnostics helpers: death-bundle summaries and taxonomy.

This module stays passive: it only summarizes planner state and classifies
already-recorded death bundles. It does not alter planning behavior.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from snks.agent.vector_sim import DynamicEntityState, VectorPlan, VectorTrajectory

DEFAULT_DEATH_TRACE_HORIZON = 8
DEFAULT_CANDIDATE_LIMIT = 5

FAILURE_BUCKETS = (
    "missed_imminent_threat",
    "bad_tradeoff_under_threat",
    "no_escape_plan_generated",
    "state_desync_or_execution_failure",
    "resource_vitals_commitment_error",
    "unknown",
)

ERROR_LABELS = ("prediction", "ranking", "generation", "execution", "unknown")

RESOURCE_DEATH_CAUSES = {"starvation", "dehydration"}


def _round_float(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def summarize_dynamic_entities(entities: list[DynamicEntityState]) -> list[dict[str, Any]]:
    summary = []
    for entity in entities:
        summary.append(
            {
                "concept_id": entity.concept_id,
                "position": list(entity.position),
                "velocity": list(entity.velocity) if entity.velocity is not None else None,
                "age": int(entity.age),
                "last_seen_step": int(entity.last_seen_step),
            }
        )
    return summary


def summarize_plan(plan: VectorPlan) -> dict[str, Any]:
    return {
        "origin": plan.origin,
        "steps": [
            {"action": step.action, "target": step.target}
            for step in plan.steps
        ],
        "first_action": plan.steps[0].action if plan.steps else None,
        "first_target": plan.steps[0].target if plan.steps else None,
        "n_steps": len(plan.steps),
    }


def summarize_trajectory(traj: VectorTrajectory) -> dict[str, Any]:
    final_state = traj.final_state
    final_body = final_state.body if final_state is not None else {}
    return {
        "terminated": bool(traj.terminated),
        "terminated_reason": traj.terminated_reason,
        "predicted_health": _round_float(final_body.get("health")),
        "predicted_food": _round_float(final_body.get("food")),
        "predicted_drink": _round_float(final_body.get("drink")),
        "predicted_energy": _round_float(final_body.get("energy")),
        "predicted_player_pos": (
            list(final_state.player_pos) if final_state is not None else None
        ),
        "health_delta": _round_float(traj.vital_delta("health")),
        "inventory_gain": int(traj.total_inventory_gain()),
        "avg_surprise": _round_float(traj.avg_surprise()),
    }


def summarize_scored_candidates(
    scored: list[tuple[tuple, VectorPlan, VectorTrajectory]],
    body_before: dict[str, float],
    limit: int = DEFAULT_CANDIDATE_LIMIT,
) -> list[dict[str, Any]]:
    health_before = float(body_before.get("health", 0.0))
    result: list[dict[str, Any]] = []
    for rank, (score, plan, traj) in enumerate(scored[:limit], start=1):
        traj_summary = summarize_trajectory(traj)
        predicted_health_raw = traj_summary.get("predicted_health")
        predicted_health = (
            health_before
            if predicted_health_raw is None
            else float(predicted_health_raw)
        )
        result.append(
            {
                "rank": rank,
                "score": list(score),
                "plan": summarize_plan(plan),
                "trajectory": traj_summary,
                "predicted_loss": _round_float(max(0.0, health_before - predicted_health)),
            }
        )
    return result


def infer_error_label(step_snapshot: dict[str, Any]) -> str:
    """Classify one pre-death decision into a coarse error label."""
    actual_damage = float(step_snapshot.get("actual_damage", 0.0) or 0.0)
    if actual_damage <= 0.0:
        return "unknown"

    if bool(step_snapshot.get("blocked_move")):
        return "execution"

    predicted_loss = float(step_snapshot.get("chosen_predicted_loss", 0.0) or 0.0)
    better_safe_candidate = bool(step_snapshot.get("better_safe_candidate_exists"))
    better_move_candidate = bool(step_snapshot.get("better_move_candidate_exists"))
    move_candidates_present = bool(step_snapshot.get("move_candidates_present"))

    if predicted_loss <= 0.0:
        return "prediction"
    if better_safe_candidate:
        return "ranking"
    if better_move_candidate:
        return "ranking"
    if not move_candidates_present and bool(step_snapshot.get("hostile_present")):
        return "generation"
    return "unknown"


def build_death_trace_bundle(
    *,
    episode_steps: int,
    death_cause: str,
    env_cause: str,
    final_body: dict[str, float],
    final_inventory: dict[str, Any],
    capture_horizon: int,
    recent_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    error_counts = Counter(step.get("error_label", "unknown") for step in recent_steps)
    primary_error = "unknown"
    for label, _count in error_counts.most_common():
        if label != "unknown":
            primary_error = label
            break
    if primary_error == "unknown" and recent_steps:
        primary_error = str(recent_steps[-1].get("error_label", "unknown"))

    return {
        "schema_version": 1,
        "episode_steps": int(episode_steps),
        "death_cause": death_cause,
        "env_cause": env_cause,
        "capture_horizon": int(capture_horizon),
        "n_recent_steps": len(recent_steps),
        "primary_error_label": primary_error,
        "error_label_counts": dict(error_counts),
        "final_body": {key: _round_float(value) for key, value in final_body.items()},
        "final_inventory": dict(final_inventory),
        "recent_steps": recent_steps,
    }


def classify_failure_bucket(bundle: dict[str, Any]) -> str:
    """Map one death bundle into the Stage 90 primary failure taxonomy."""
    death_cause = str(bundle.get("death_cause", "unknown"))
    if death_cause in RESOURCE_DEATH_CAUSES:
        return "resource_vitals_commitment_error"

    recent_steps = list(bundle.get("recent_steps", []))
    if not recent_steps:
        return "unknown"

    if any(bool(step.get("blocked_move")) for step in recent_steps):
        return "state_desync_or_execution_failure"

    if any(step.get("error_label") == "execution" for step in recent_steps):
        return "state_desync_or_execution_failure"

    primary_error = str(bundle.get("primary_error_label", "unknown"))
    if primary_error == "generation":
        return "no_escape_plan_generated"
    if primary_error == "ranking":
        return "bad_tradeoff_under_threat"
    if primary_error == "prediction":
        return "missed_imminent_threat"

    final_body = bundle.get("final_body", {})
    if (
        float(final_body.get("food", 1.0) or 1.0) <= 0.0
        or float(final_body.get("drink", 1.0) or 1.0) <= 0.0
    ):
        return "resource_vitals_commitment_error"

    return "unknown"


def summarize_failure_buckets(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(classify_failure_bucket(bundle) for bundle in bundles)
    n = max(len(bundles), 1)
    seed_coverage: dict[str, set[int]] = {bucket: set() for bucket in FAILURE_BUCKETS}
    representatives: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in FAILURE_BUCKETS}

    for bundle in bundles:
        bucket = classify_failure_bucket(bundle)
        seed = bundle.get("seed")
        if isinstance(seed, int):
            seed_coverage[bucket].add(seed)
        if len(representatives[bucket]) < 3:
            representatives[bucket].append(
                {
                    "seed": seed,
                    "episode_id": bundle.get("episode_id"),
                    "death_cause": bundle.get("death_cause"),
                    "primary_error_label": bundle.get("primary_error_label", "unknown"),
                }
            )

    dominant_bucket = counts.most_common(1)[0][0] if counts else "unknown"
    return {
        "n_bundles": len(bundles),
        "bucket_counts": {bucket: int(counts.get(bucket, 0)) for bucket in FAILURE_BUCKETS},
        "bucket_shares": {
            bucket: _round_float(counts.get(bucket, 0) / n) for bucket in FAILURE_BUCKETS
        },
        "seed_coverage": {
            bucket: sorted(seed_coverage[bucket]) for bucket in FAILURE_BUCKETS
        },
        "representative_episodes": representatives,
        "dominant_bucket": dominant_bucket,
        "dominant_bucket_share": _round_float(counts.get(dominant_bucket, 0) / n),
    }
