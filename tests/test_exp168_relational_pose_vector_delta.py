"""Focused contracts for the exp168 direct vector-transition diagnostic."""

from __future__ import annotations

import importlib
import importlib.util

import torch

from snks.agent.core_world_model import LatentState, Prediction


def _experiment():
    name = "experiments.exp168_relational_pose_vector_delta"
    assert importlib.util.find_spec(name) is not None, "exp168 experiment is missing"
    return importlib.import_module(name)


def test_vector_probe_and_prediction_use_direct_member_deltas():
    exp = _experiment()
    probe = exp.RelationalPoseVectorDelta(z_dim=2, h_dim=1, heads=2)
    output = probe(
        torch.zeros(1, 2), torch.zeros(1, 1), torch.zeros(1, 8),
        torch.tensor([2], dtype=torch.long),
    )
    assert output.shape == (2, 1, 2)

    state = LatentState(
        z=torch.tensor([[1.0, 2.0]]), sensors=torch.zeros(1, 1),
        sensor_mask=torch.ones(1, 1, dtype=torch.bool), hidden=torch.ones(1, 1),
        schema="grid-v1",
    )
    native = Prediction(
        next_state=state, terminated_prob=torch.tensor([0.25]),
        uncertainty=torch.tensor([99.0]), member_z=torch.zeros(2, 1, 2),
    )
    delta = torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]])
    replaced = exp.apply_member_deltas(native, state.z, delta)

    torch.testing.assert_close(
        replaced.member_z,
        torch.tensor([[[2.0, 2.0]], [[1.0, 4.0]]]),
    )
    torch.testing.assert_close(replaced.next_state.z, torch.tensor([[1.5, 3.0]]))
    torch.testing.assert_close(replaced.next_state.hidden, state.hidden)
    torch.testing.assert_close(replaced.terminated_prob, native.terminated_prob)


def test_detached_vector_target_and_weighted_loss_keep_member_shape():
    exp = _experiment()
    current = torch.zeros(2, 2, requires_grad=True)
    actual = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target = exp.detached_vector_target(actual, current, heads=2)

    assert target.shape == (2, 2, 2)
    assert not target.requires_grad
    prediction = torch.zeros_like(target, requires_grad=True)
    actions = torch.tensor([0, 1], dtype=torch.long)
    changed = torch.tensor([False, True])
    weights = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    loss = exp.weighted_vector_mse(
        prediction, target, actions, changed, weights
    )
    expected = (target.square() * torch.tensor([2.0, 7.0])[None, :, None]).mean()

    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is not None


def test_defaults_lock_protocol_references_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.probe_updates == 400
    assert args.probe_batch_size == 256
    assert args.exp165_reference == exp.DEFAULT_EXP165_REFERENCE
    assert args.exp166_reference == exp.DEFAULT_EXP166_REFERENCE
    assert args.progress_interval == 30
