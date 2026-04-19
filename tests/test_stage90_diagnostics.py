from __future__ import annotations

from snks.agent.stage90_diagnostics import (
    build_death_trace_bundle,
    classify_failure_bucket,
    infer_error_label,
    summarize_failure_buckets,
)


def test_infer_error_label_prediction_when_damage_was_unpredicted():
    snapshot = {
        "actual_damage": 2.0,
        "chosen_predicted_loss": 0.0,
        "better_safe_candidate_exists": False,
        "better_move_candidate_exists": False,
        "move_candidates_present": True,
        "hostile_present": True,
        "blocked_move": False,
    }
    assert infer_error_label(snapshot) == "prediction"


def test_infer_error_label_ranking_when_safer_candidate_exists():
    snapshot = {
        "actual_damage": 2.0,
        "chosen_predicted_loss": 2.0,
        "better_safe_candidate_exists": True,
        "better_move_candidate_exists": False,
        "move_candidates_present": True,
        "hostile_present": True,
        "blocked_move": False,
    }
    assert infer_error_label(snapshot) == "ranking"


def test_infer_error_label_execution_when_move_was_blocked():
    snapshot = {
        "actual_damage": 1.0,
        "chosen_predicted_loss": 1.0,
        "better_safe_candidate_exists": False,
        "better_move_candidate_exists": False,
        "move_candidates_present": True,
        "hostile_present": True,
        "blocked_move": True,
    }
    assert infer_error_label(snapshot) == "execution"


def test_classify_failure_bucket_resource_death():
    bundle = build_death_trace_bundle(
        episode_steps=100,
        death_cause="starvation",
        env_cause="done",
        final_body={"health": 1.0, "food": 0.0, "drink": 2.0, "energy": 3.0},
        final_inventory={},
        capture_horizon=3,
        recent_steps=[],
    )
    assert classify_failure_bucket(bundle) == "resource_vitals_commitment_error"


def test_classify_failure_bucket_from_primary_error_label():
    bundle = build_death_trace_bundle(
        episode_steps=120,
        death_cause="zombie",
        env_cause="health",
        final_body={"health": 0.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        final_inventory={},
        capture_horizon=3,
        recent_steps=[
            {"error_label": "ranking", "blocked_move": False},
            {"error_label": "ranking", "blocked_move": False},
        ],
    )
    assert classify_failure_bucket(bundle) == "bad_tradeoff_under_threat"


def test_summarize_failure_buckets_reports_dominant_bucket():
    bundles = [
        {
            "seed": 1,
            "episode_id": 0,
            "death_cause": "zombie",
            "primary_error_label": "prediction",
            "recent_steps": [{"error_label": "prediction", "blocked_move": False}],
        },
        {
            "seed": 2,
            "episode_id": 1,
            "death_cause": "arrow",
            "primary_error_label": "prediction",
            "recent_steps": [{"error_label": "prediction", "blocked_move": False}],
        },
        {
            "seed": 3,
            "episode_id": 2,
            "death_cause": "starvation",
            "primary_error_label": "unknown",
            "recent_steps": [],
            "final_body": {"food": 0.0, "drink": 3.0},
        },
    ]
    summary = summarize_failure_buckets(bundles)
    assert summary["dominant_bucket"] == "missed_imminent_threat"
    assert summary["bucket_counts"]["missed_imminent_threat"] == 2
