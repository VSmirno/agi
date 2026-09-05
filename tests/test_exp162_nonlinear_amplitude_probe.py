"""Focused contracts for the exp162 nonlinear amplitude probe."""

from __future__ import annotations

import importlib
import importlib.util

import torch

from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder


def _experiment():
    name = "experiments.exp162_nonlinear_amplitude_probe"
    assert importlib.util.find_spec(name) is not None, "exp162 experiment is missing"
    return importlib.import_module(name)


def test_probe_has_exact_per_action_mlp_and_forward_shape():
    exp = _experiment()
    probe = exp.NonlinearAmplitudeProbe(z_dim=2, h_dim=1, heads=3)

    assert len(probe.by_action) == 5
    for head in probe.by_action:
        assert isinstance(head[0], torch.nn.Linear)
        assert head[0].in_features == 3 and head[0].out_features == 128
        assert isinstance(head[1], torch.nn.ReLU)
        assert isinstance(head[2], torch.nn.Linear)
        assert head[2].in_features == 128 and head[2].out_features == 3
    output = probe(torch.zeros(2, 2), torch.zeros(2, 1), torch.tensor([0, 4]))
    assert output.shape == (3, 2)
    assert torch.all((output > 0) & (output < 1))


def test_installed_gate_can_depend_on_z_hidden_interaction():
    exp = _experiment()
    probe = exp.NonlinearAmplitudeProbe(z_dim=2, h_dim=1, heads=1)
    with torch.no_grad():
        for parameter in probe.parameters():
            parameter.zero_()
        head = probe.by_action[3]
        head[0].weight[0, 0] = 1.0
        head[0].weight[0, 2] = 1.0
        head[0].bias[0] = -1.5
        head[2].weight[0, 0] = 8.0
    model = exp.NonlinearProbeGatedWorldModel(
        CoreEncoder(2), {"grid-v1": (5, 1)}, h_dim=1, heads=1,
        amplitude_probe=probe,
    )
    state = LatentState(
        z=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        sensors=torch.zeros(2, 1),
        sensor_mask=torch.ones(2, 1, dtype=torch.bool),
        hidden=torch.tensor([[0.0], [1.0]]),
        schema="grid-v1",
    )

    gates = model.change_gates(state, torch.tensor([3, 3]))

    assert gates.shape == (1, 2, 1)
    assert gates[0, 0, 0] == 0.5
    assert gates[0, 1, 0] > 0.95


def test_defaults_lock_protocol_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.collection_steps == 64
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.progress_interval == 30
