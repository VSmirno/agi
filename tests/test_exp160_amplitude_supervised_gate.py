"""Focused objective contracts for exp160 amplitude-supervised gate training."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _experiment():
    name = "experiments.exp160_amplitude_supervised_gate"
    assert importlib.util.find_spec(name) is not None, "exp160 experiment is missing"
    return importlib.import_module(name)


def test_analytic_targets_are_detached_bounded_and_zero_safe():
    exp = _experiment()
    raw = torch.tensor(
        [[[1.0, 0.0]], [[0.1, 0.0]], [[-1.0, 0.0]], [[0.0, 0.0]]],
        requires_grad=True,
    )
    displacement = torch.tensor([[0.25, 0.5]], requires_grad=True)

    target = exp.analytic_amplitude_targets(raw, displacement)

    torch.testing.assert_close(target[:, 0], torch.tensor([0.25, 1.0, 0.0, 0.0]))
    assert not target.requires_grad


def test_weighted_amplitude_mse_uses_ordinary_member_denominator():
    exp = _experiment()
    predicted = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    target = torch.zeros(2, 2)
    valid = torch.tensor([True, True])
    actions = torch.tensor([2, 2])
    changed = torch.tensor([False, True])
    weights = torch.ones(5, 2)
    weights[2] = torch.tensor([1.0, 3.0])

    loss = exp.weighted_amplitude_mse(
        predicted, target, valid, actions, changed, weights
    )

    # Weighted squared sum is 1 + 3; ordinary denominator is 2 members * 2 rows.
    assert loss == pytest.approx(1.0)
