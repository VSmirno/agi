"""Focused objective contract for exp158 balanced latent gate training."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _experiment():
    name = "experiments.exp158_balanced_latent_gate"
    assert importlib.util.find_spec(name) is not None, "exp158 experiment is missing"
    return importlib.import_module(name)


def test_weighted_latent_mse_applies_rare_class_weight_without_batch_renormalizing():
    exp = _experiment()
    member_z = torch.tensor([[[1.0], [1.0]]])
    target_z = torch.zeros(2, 1)
    valid = torch.tensor([True, True])
    actions = torch.tensor([2, 2])
    changed = torch.tensor([False, True])
    weights = torch.ones(5, 2)
    weights[2] = torch.tensor([1.0, 3.0])

    loss = exp.weighted_latent_mse(
        member_z, target_z, valid, actions, changed, weights
    )

    # Ordinary valid-element denominator: (1*1^2 + 3*1^2) / 2.
    assert loss == pytest.approx(2.0)
