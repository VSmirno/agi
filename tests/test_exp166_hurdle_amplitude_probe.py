"""Focused contracts for the exp166 hurdle amplitude diagnostic."""

from __future__ import annotations

import importlib
import importlib.util

import torch


def _experiment():
    name = "experiments.exp166_hurdle_amplitude_probe"
    assert importlib.util.find_spec(name) is not None, "exp166 experiment is missing"
    return importlib.import_module(name)


def test_hurdle_gate_is_exact_zero_below_atom_boundary():
    exp = _experiment()
    logits = torch.tensor([[-0.1, 0.0, 0.1]])
    conditional = torch.tensor([[0.2, 0.3, 0.4]])

    gate = exp.hurdle_gate(logits, conditional)

    torch.testing.assert_close(gate, torch.tensor([[0.0, 0.3, 0.4]]))
    assert gate[0, 0].item() == 0.0


def test_atom_weights_and_loss_handle_single_class_member_groups():
    exp = _experiment()
    actions = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    target = torch.tensor(
        [
            [0.2, 0.4, 0.0, 0.0],
            [0.0, 0.0, 0.3, 0.6],
        ]
    )
    weights, counts = exp.atom_class_weights(actions, target, action_count=2)

    torch.testing.assert_close(weights[0, 0], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(weights[0, 1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights[1, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(weights[1, 1], torch.tensor([0.0, 1.0]))
    assert counts.shape == (2, 2, 2)
    assert torch.isfinite(weights).all()

    logits = torch.zeros_like(target)
    conditional = torch.full_like(target, 0.5)
    total, components = exp.hurdle_amplitude_loss(
        logits, conditional, target, actions, weights
    )
    expected_conditional = torch.tensor(
        ((0.5 - 0.2) ** 2 + (0.5 - 0.4) ** 2
         + (0.5 - 0.3) ** 2 + (0.5 - 0.6) ** 2) / 4
    )
    torch.testing.assert_close(components["conditional_mse"], expected_conditional)
    torch.testing.assert_close(
        total, components["atom_bce"] + components["conditional_mse"]
    )


def test_defaults_lock_protocol_reference_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.collection_steps == 64
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.progress_interval == 30
    assert args.exp165_reference == exp.DEFAULT_EXP165_REFERENCE
