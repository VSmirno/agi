"""Contracts for the exp156 pre-gate raw-delta oracle audit."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch

from experiments.exp153_change_gated_dynamics import ChangeGatedResidualWorldModel
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder


def _experiment():
    name = "experiments.exp156_gated_delta_oracle"
    assert importlib.util.find_spec(name) is not None, "exp156 audit is missing"
    return importlib.import_module(name)


def test_raw_deltas_are_latent_head_outputs_before_native_gate():
    exp = _experiment()
    model = ChangeGatedResidualWorldModel(
        CoreEncoder(4), {"grid-v1": (5, 1)}, h_dim=3, heads=2
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.latent_heads[0].bias.fill_(0.25)
        model.latent_heads[1].bias.fill_(-0.5)
        for gate in model.gate_heads:
            gate.bias.fill_(-12.0)
    state = LatentState(
        z=torch.full((1, 4), 3.0),
        sensors=torch.zeros(1, 1),
        sensor_mask=torch.ones(1, 1, dtype=torch.bool),
        hidden=torch.zeros(1, 3),
        schema="grid-v1",
    )

    prediction, raw = exp.native_prediction_and_raw_deltas(
        model, state, torch.tensor([3], dtype=torch.long)
    )

    torch.testing.assert_close(raw[0], torch.full((1, 4), 0.25))
    torch.testing.assert_close(raw[1], torch.full((1, 4), -0.5))
    native_delta = prediction.member_z - state.z.unsqueeze(0)
    assert not torch.allclose(native_delta, raw)


def test_upper_bound_gate_requires_exact_clean_source_and_motion_gain():
    exp = _experiment()
    passing = {
        "contact_failure_layouts": 0,
        "blocked_noop_failure_layouts": 0,
        "medians": {"free_forward_prediction_persistence_ratio": 0.8},
    }
    assert exp.raw_delta_upper_bound_gate(passing, exact_protocol=True)
    assert not exp.raw_delta_upper_bound_gate(passing, exact_protocol=False)
    assert not exp.raw_delta_upper_bound_gate(
        {**passing, "blocked_noop_failure_layouts": 1}, exact_protocol=True
    )
    assert not exp.raw_delta_upper_bound_gate(
        {**passing, "medians": {"free_forward_prediction_persistence_ratio": 1.0}},
        exact_protocol=True,
    )


def test_auxiliary_loader_rejects_incomplete_v4_checkpoint(tmp_path):
    exp = _experiment()
    checkpoint = tmp_path / "unsafe.pt"
    torch.save(
        {
            "format_version": 4,
            "latent_parameterization": "gated_residual_zero_init",
            "event_supervision": True,
            "modules": {"model": {"class": exp.auxiliary.AUXILIARY_CLASS}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError):
        exp.load_auxiliary_checkpoint(checkpoint)
