"""Focused artifact-row contract for exp170."""

from __future__ import annotations

import pytest

from experiments import exp170_event_mode_row_audit as exp


def _row(*, changed: bool, prediction: float, persistence: float):
    return {
        "split": "source",
        "layout": "layout",
        "step": 1,
        "real_history": [3, 2],
        "canonical_action": 2,
        "action": 2,
        "action_name": "forward",
        "rgb_changed": changed,
        "predicted_vs_actual_next_z_mse": prediction,
        "persistence_vs_actual_next_z_mse": persistence,
    }


def test_matching_rows_apply_oracle_event_without_changing_targets():
    changed_event = _row(changed=True, prediction=0.8, persistence=2.0)
    changed_event.update(event_probability=0.1, predicted_change=False)
    changed_frozen = _row(changed=True, prediction=0.25, persistence=2.0)

    changed_oracle = exp.oracle_event_row(changed_event, changed_frozen)
    assert changed_oracle["predicted_vs_actual_next_z_mse"] == 0.25
    assert changed_oracle["prediction_to_persistence_ratio"] == 0.125
    assert changed_oracle["persistence_vs_actual_next_z_mse"] == 2.0

    static_event = _row(changed=False, prediction=0.9, persistence=0.0)
    static_event.update(event_probability=0.9, predicted_change=True)
    static_frozen = _row(changed=False, prediction=0.7, persistence=0.0)
    static_oracle = exp.oracle_event_row(static_event, static_frozen)
    assert static_oracle["predicted_vs_actual_next_z_mse"] == 0.0
    assert static_oracle["prediction_to_persistence_ratio"] is None
    assert static_oracle["oracle_event_uses_vector"] is False

    mismatch = dict(static_frozen, action=3)
    with pytest.raises(AssertionError, match="row keys"):
        exp.oracle_event_row(static_event, mismatch)
