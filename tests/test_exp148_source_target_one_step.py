from __future__ import annotations

import pytest

from experiments.exp148_source_target_one_step import (
    _aggregate_split,
    _outcome_label,
)


def _summary(
    layout: str,
    *,
    contact_failure: bool,
    blocked_noop_failure: bool,
    interact_ratios: list[float | None],
    free_ratio: float | None,
    blocked_mse: float,
):
    return {
        "layout": layout,
        "contact_failure": contact_failure,
        "blocked_noop_failure": blocked_noop_failure,
        "interact_prediction_persistence_ratios": interact_ratios,
        "free_forward_prediction_persistence_ratio": free_ratio,
        "blocked_forward_prediction_mse": blocked_mse,
    }


def test_split_aggregation_and_outcome_labels_are_deterministic():
    source = [
        _summary(
            "east_row2",
            contact_failure=False,
            blocked_noop_failure=False,
            interact_ratios=[1.0, None],
            free_ratio=0.5,
            blocked_mse=0.2,
        ),
        _summary(
            "west_row3",
            contact_failure=True,
            blocked_noop_failure=False,
            interact_ratios=[3.0, 5.0],
            free_ratio=1.5,
            blocked_mse=0.4,
        ),
    ]
    source_summary = _aggregate_split(source)

    assert source_summary["layout_count"] == 2
    assert source_summary["contact_failure_layouts"] == 1
    assert source_summary["blocked_noop_failure_layouts"] == 0
    assert source_summary["medians"] == {
        "interact_prediction_persistence_ratio": 3.0,
        "free_forward_prediction_persistence_ratio": 1.0,
        "blocked_forward_prediction_mse": 0.3,
    }

    clean = _aggregate_split(
        [
            _summary(
                "a",
                contact_failure=False,
                blocked_noop_failure=False,
                interact_ratios=[0.5],
                free_ratio=0.5,
                blocked_mse=0.01,
            )
        ]
    )
    all_failed = _aggregate_split(
        [
            _summary(
                "a",
                contact_failure=True,
                blocked_noop_failure=True,
                interact_ratios=[2.0],
                free_ratio=2.0,
                blocked_mse=1.0,
            )
        ]
    )

    assert _outcome_label(clean, clean) == "mixed_or_inconclusive"
    assert _outcome_label(clean, all_failed) == "unseen_only_failure_evidence"
    assert _outcome_label(all_failed, all_failed) == "shared_one_step_failure_evidence"
    assert _outcome_label(source_summary, all_failed) == "mixed_or_inconclusive"

