"""Behavior checks for causal recurrent learning, independent of an environment."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_objective import masked_mse, sigreg
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline.core_config import CoreConfig


def make_batch(burn_in: int = 0) -> SequenceBatch:
    # Action 0 decreases the sensor; action 1 increases it on each step.
    return SequenceBatch(
        rgb=torch.zeros(2, 3, 3, 64, 64),
        sensors=torch.tensor([[[0.0], [-0.5], [-1.0]],
                              [[0.0], [0.5], [1.0]]]),
        sensor_mask=torch.ones(2, 3, 1, dtype=torch.bool),
        actions=torch.tensor([[0, 0], [1, 1]]),
        terminated=torch.zeros(2, 2),
        valid=torch.ones(2, 2, dtype=torch.bool),
        schema="toy", burn_in=burn_in,
    )


def make_model() -> CoreWorldModel:
    torch.manual_seed(7)
    return CoreWorldModel(CoreEncoder(8), {"toy": (2, 1)}, 16, 3)


def test_sigreg_rejects_collapse_and_masked_nan_has_finite_gradients():
    # Catches a variance-only substitute and multiplication-based NaN masking.
    torch.manual_seed(11)
    normal = torch.randn(512, 8, requires_grad=True)
    directions = F.normalize(torch.randn(8, 16), dim=0)
    good = sigreg(normal, directions)
    assert good < sigreg(torch.zeros_like(normal), directions)
    good.backward()
    assert torch.isfinite(normal.grad).all()
    with pytest.raises(ValueError):
        sigreg(normal[:1], directions)
    pred = torch.tensor([2.0, float("nan")], requires_grad=True)
    loss = masked_mse(pred, torch.tensor([1.0, float("nan")]),
                      torch.tensor([True, False]))
    assert loss.item() == 1.0
    loss.backward()
    assert torch.equal(pred.grad, torch.tensor([2.0, 0.0]))
    assert masked_mse(pred, pred, torch.zeros(2, dtype=torch.bool)).item() == 0


def test_rollout_preserves_root_uses_history_and_registers_paired_heads():
    # Catches in-place root updates, discarded history, and RNG drift at transfer.
    model = make_model()
    batch = make_batch()
    root = model.initial_from_tensors(batch.rgb[:, 0], batch.sensors[:, 0],
                                      batch.sensor_mask[:, 0], "toy")
    saved = [x.clone() for x in (root.z, root.sensors, root.hidden)]
    left = model.step(root, torch.tensor([0, 0])).next_state
    right = model.step(root, torch.tensor([1, 1])).next_state
    assert not torch.allclose(left.z, right.z)
    history_only = replace(root, hidden=left.hidden)
    assert not torch.allclose(model.step(history_only, batch.actions[:, 0]).next_state.z,
                              model.step(root, batch.actions[:, 0]).next_state.z)
    model.rollout(root, batch.actions)
    assert all(torch.equal(a, b) for a, b in zip(saved, (root.z, root.sensors, root.hidden)))
    old = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    model.register_schema("new", (3, 2), seed=91)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in old.items())
    other = make_model()
    other.register_schema("new", (3, 2), seed=91)
    assert torch.equal(model.action_embeddings["new"].weight,
                       other.action_embeddings["new"].weight)


def test_future_targets_do_not_enter_rollout_but_receive_encoder_gradients():
    # Changing unseen observations must change targets, never recurrent inputs.
    model = make_model()
    trainer = CoreTrainer(model, CoreConfig(device="cpu", sigreg_weight=0.0))
    batch = make_batch()
    observed_inputs = []
    hook = model.recurrent.register_forward_pre_hook(
        lambda _module, args: observed_inputs.append(args[0].detach().clone()))
    trainer.compute_loss(batch)
    baseline = list(observed_inputs)
    observed_inputs.clear()
    changed_rgb = batch.rgb.clone()
    changed_rgb[:, 1:] = 0.7
    changed_rgb.requires_grad_()
    changed_sensors = batch.sensors.clone()
    changed_sensors[:, 1:] = 9.0
    changed_mask = batch.sensor_mask.clone()
    changed_mask[:, 1:] = False
    loss = trainer.compute_loss(replace(batch, rgb=changed_rgb,
                                       sensors=changed_sensors, sensor_mask=changed_mask))
    hook.remove()
    assert len(baseline) == len(observed_inputs) == 2
    assert all(torch.equal(a, b) for a, b in zip(baseline, observed_inputs))
    loss.backward()
    assert changed_rgb.grad[:, 1:].abs().sum() > 0


def test_training_learns_action_effect_for_all_heads_with_frozen_encoder():
    # Catches missing optimization, ensemble-mean-only loss, and action blindness.
    model = make_model()
    model.encoder.requires_grad_(False)
    trainer = CoreTrainer(model, CoreConfig(device="cpu", learning_rate=0.02,
                                           sensor_weight=5.0, sigreg_weight=0.0))
    batch = make_batch(burn_in=1)
    initial_loss = trainer.compute_loss(batch).item()
    for _ in range(60):
        trainer.update(batch, Mode.TRAIN)
    assert trainer.compute_loss(batch).item() < initial_loss * 0.25
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(head.weight.grad is not None and head.weight.grad.abs().sum() > 0
               for head in model.latent_heads)
    with torch.no_grad():
        root = model.initial_from_tensors(batch.rgb[:, 0], batch.sensors[:, 0],
                                          batch.sensor_mask[:, 0], "toy")
        prediction = model.rollout(root, batch.actions)[-1]
    assert prediction.next_state.sensors[0, 0] < -0.3
    assert prediction.next_state.sensors[1, 0] > 0.3


def test_evaluation_prohibits_update_without_mutating_weights_or_optimizer():
    # Catches an evaluation guard placed after zero_grad or optimizer.step.
    model = make_model()
    trainer = CoreTrainer(model, CoreConfig(device="cpu"))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sentinel = next(model.parameters())
    sentinel.grad = torch.ones_like(sentinel)
    with pytest.raises(PermissionError):
        trainer.update(make_batch(), Mode.EVALUATE)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert torch.equal(sentinel.grad, torch.ones_like(sentinel))
    assert not trainer.optimizer.state
