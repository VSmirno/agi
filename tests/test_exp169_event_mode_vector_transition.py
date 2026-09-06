"""Focused contracts for the exp169 event-mode vector transition."""

from __future__ import annotations

import importlib
import importlib.util

import torch

from snks.agent.core_world_model import LatentState, Prediction


def _experiment():
    name = "experiments.exp169_event_mode_vector_transition"
    assert importlib.util.find_spec(name) is not None, "exp169 experiment is missing"
    return importlib.import_module(name)


def _prediction() -> Prediction:
    state = LatentState(
        z=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        sensors=torch.zeros(2, 1),
        sensor_mask=torch.ones(2, 1, dtype=torch.bool),
        hidden=torch.ones(2, 1),
        schema="grid-v1",
    )
    return Prediction(
        next_state=state,
        terminated_prob=torch.tensor([0.25, 0.75]),
        uncertainty=torch.tensor([99.0, 99.0]),
        member_z=torch.zeros(2, 2, 2),
    )


def test_event_mode_is_literal_persistence_or_exact_frozen_vector_delta():
    exp = _experiment()
    native = _prediction()
    current = native.next_state.z
    frozen_delta = torch.tensor(
        [
            [[10.0, 20.0], [30.0, 40.0]],
            [[11.0, 21.0], [31.0, 41.0]],
        ]
    )
    installed = exp.apply_event_mode(
        native, current, frozen_delta, torch.tensor([0.49, 0.50])
    )

    torch.testing.assert_close(installed.member_z[:, 0], current[0].expand(2, -1))
    torch.testing.assert_close(
        installed.member_z[:, 1], current[1].unsqueeze(0) + frozen_delta[:, 1]
    )
    torch.testing.assert_close(installed.next_state.hidden, native.next_state.hidden)
    torch.testing.assert_close(installed.terminated_prob, native.terminated_prob)


def test_balanced_event_bce_handles_actions_with_only_one_observed_class():
    exp = _experiment()
    counts = {
        "0": {"total": 3, "rgb_changed": 3, "rgb_no_change": 0},
        "1": {"total": 2, "rgb_changed": 0, "rgb_no_change": 2},
    }
    weights = exp.event_class_weights(counts, action_count=2)
    torch.testing.assert_close(weights, torch.tensor([[0.0, 1.0], [1.0, 0.0]]))

    logits = torch.zeros(5, requires_grad=True)
    actions = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    changed = torch.tensor([True, True, True, False, False])
    loss = exp.balanced_event_bce(logits, actions, changed, weights)

    torch.testing.assert_close(loss, torch.tensor(torch.log(torch.tensor(2.0))))
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_defaults_lock_frozen_vector_reference_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.exp168_checkpoint == exp.DEFAULT_EXP168_CHECKPOINT
    assert args.exp168_reference == exp.DEFAULT_EXP168_REFERENCE
    assert args.progress_interval == 30
