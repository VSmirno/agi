"""Exact bounded least-squares contract for the exp155 oracle diagnostic."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _experiment():
    name = "experiments.exp155_oracle_residual_gate"
    assert importlib.util.find_spec(name) is not None, "exp155 oracle experiment is missing"
    return importlib.import_module(name)


def test_oracle_solvers_recover_interior_and_shared_optima():
    exp = _experiment()
    deltas = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    target = torch.tensor([0.5, 1.0])

    shared_gate, shared_mse = exp.solve_shared_scalar_gate(deltas, target)
    member_gates, member_mse = exp.solve_per_member_scalar_gates(deltas, target)

    assert shared_gate == pytest.approx(0.75)
    assert shared_mse == pytest.approx(0.0625)
    torch.testing.assert_close(member_gates, torch.tensor([0.5, 1.0]))
    assert member_mse == pytest.approx(0.0, abs=1e-12)


def test_per_member_solver_honors_box_boundaries_and_is_deterministic():
    exp = _experiment()
    deltas = torch.tensor([[1.0], [1.0]])

    low, low_mse = exp.solve_per_member_scalar_gates(deltas, torch.tensor([-1.0]))
    high, high_mse = exp.solve_per_member_scalar_gates(deltas, torch.tensor([2.0]))

    torch.testing.assert_close(low, torch.zeros(2))
    torch.testing.assert_close(high, torch.ones(2))
    assert low_mse == pytest.approx(1.0)
    assert high_mse == pytest.approx(1.0)


def test_falsification_gate_requires_exact_clean_source_and_motion_gain():
    exp = _experiment()
    passing = {
        "contact_failure_layouts": 0,
        "blocked_noop_failure_layouts": 0,
        "medians": {"free_forward_prediction_persistence_ratio": 0.8},
    }
    assert exp.oracle_falsification_gate(passing, exact_protocol=True)
    assert not exp.oracle_falsification_gate(passing, exact_protocol=False)
    assert not exp.oracle_falsification_gate(
        {**passing, "contact_failure_layouts": 1}, exact_protocol=True
    )
    assert not exp.oracle_falsification_gate(
        {**passing, "medians": {"free_forward_prediction_persistence_ratio": 1.0}},
        exact_protocol=True,
    )
