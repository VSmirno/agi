"""Contracts for the exp159 independent-member amplitude oracle audit."""

from __future__ import annotations

import importlib
import importlib.util

import torch


def _experiment():
    name = "experiments.exp159_independent_amplitude_oracle"
    assert importlib.util.find_spec(name) is not None, "exp159 audit is missing"
    return importlib.import_module(name)


def test_independent_amplitudes_project_clip_and_handle_zero_direction():
    exp = _experiment()
    raw_deltas = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.1, 0.0], [-1.0, 0.0], [0.0, 0.0]]
    )
    target = torch.tensor([0.25, 0.5])

    gates = exp.independent_member_amplitudes(raw_deltas, target)

    torch.testing.assert_close(gates, torch.tensor([0.25, 0.5, 1.0, 0.0, 0.0]))


def test_gate_requires_exact_clean_source_and_unseen_motion_gain():
    exp = _experiment()
    passing = {
        split: {
            "contact_failure_layouts": 0,
            "blocked_noop_failure_layouts": 0,
            "medians": {"free_forward_prediction_persistence_ratio": 0.8},
        }
        for split in ("source", "unseen")
    }
    assert exp.independent_target_upper_bound_gate(passing, exact_protocol=True)
    assert not exp.independent_target_upper_bound_gate(passing, exact_protocol=False)
    assert not exp.independent_target_upper_bound_gate(
        {**passing, "unseen": {**passing["unseen"], "contact_failure_layouts": 1}},
        exact_protocol=True,
    )
    assert not exp.independent_target_upper_bound_gate(
        {
            **passing,
            "source": {
                **passing["source"],
                "medians": {"free_forward_prediction_persistence_ratio": 1.0},
            },
        },
        exact_protocol=True,
    )
