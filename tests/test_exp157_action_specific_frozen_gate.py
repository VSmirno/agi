"""Contracts for the exp157 frozen-backbone action-specific gate intervention."""

from __future__ import annotations

import importlib
import importlib.util

import torch

from experiments.exp153_change_gated_dynamics import ChangeGatedResidualWorldModel
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder


def _experiment():
    name = "experiments.exp157_action_specific_frozen_gate"
    assert importlib.util.find_spec(name) is not None, "exp157 experiment is missing"
    return importlib.import_module(name)


def _baseline():
    return ChangeGatedResidualWorldModel(
        CoreEncoder(4), {"grid-v1": (5, 1)}, h_dim=3, heads=2
    )


def _state(batch: int = 2):
    return LatentState(
        z=torch.tensor([[2.0, -1.0, 0.5, 3.0]]).repeat(batch, 1),
        sensors=torch.zeros(batch, 1),
        sensor_mask=torch.ones(batch, 1, dtype=torch.bool),
        hidden=torch.zeros(batch, 3),
        schema="grid-v1",
    )


def test_actions_select_distinct_state_linear_boundaries():
    exp = _experiment()
    model = exp.ActionSpecificGateWorldModel(
        CoreEncoder(4), {"grid-v1": (5, 1)}, h_dim=3, heads=2
    )
    with torch.no_grad():
        for member in model.action_gate_heads:
            for gate in member:
                gate.weight.zero_()
                gate.bias.zero_()
        model.action_gate_heads[0][2].weight[0, 0] = 1.0
        model.action_gate_heads[0][3].weight[0, 0] = -1.0

    gates = model.change_gates(_state(), torch.tensor([2, 3]))

    torch.testing.assert_close(gates[0, :, 0], torch.tensor([2.0, -2.0]).sigmoid())
    torch.testing.assert_close(gates[1, :, 0], torch.full((2,), 0.5))


def test_checkpoint_transfer_freezes_everything_except_new_gate_parameters():
    exp = _experiment()
    baseline = _baseline()

    candidate, counts = exp.transfer_frozen_backbone(baseline)

    trainable = {
        name for name, parameter in candidate.named_parameters() if parameter.requires_grad
    }
    assert trainable
    assert all(name.startswith("action_gate_heads.") for name in trainable)
    assert counts["trainable"] == sum(
        parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad
    )
    assert counts["frozen"] == sum(
        parameter.numel() for parameter in candidate.parameters() if not parameter.requires_grad
    )
    baseline_state = baseline.state_dict()
    for name, value in candidate.state_dict().items():
        if not name.startswith("action_gate_heads."):
            torch.testing.assert_close(value, baseline_state[name])


def test_transfer_gate_requires_both_splits_clean_and_free_motion_better():
    exp = _experiment()
    passing = {
        "contact_failure_layouts": 0,
        "blocked_noop_failure_layouts": 0,
        "medians": {"free_forward_prediction_persistence_ratio": 0.9},
    }
    assert exp.one_step_transfer_gate(passing, passing, exact_protocol=True)
    assert not exp.one_step_transfer_gate(passing, passing, exact_protocol=False)
    assert not exp.one_step_transfer_gate(
        passing,
        {**passing, "contact_failure_layouts": 1},
        exact_protocol=True,
    )
    assert not exp.one_step_transfer_gate(
        passing,
        {**passing, "medians": {"free_forward_prediction_persistence_ratio": 1.0}},
        exact_protocol=True,
    )
