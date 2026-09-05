"""Self-supervised change labels, autoregressive alignment and public artifacts."""

from dataclasses import replace
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from experiments import exp153_change_gated_dynamics as gated
from experiments.exp146_temporal_mpc_physics import TemporalProbe
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline.core_config import CoreConfig


def _experiment():
    name = "experiments.exp154_auxiliary_change_gate"
    assert importlib.util.find_spec(name) is not None, "auxiliary change-gate experiment is missing"
    return importlib.import_module(name)


def _model():
    torch.manual_seed(154)
    return _experiment().AuxiliaryChangeGatedWorldModel(
        CoreEncoder(4), {"grid-v1": (5, 1)}, 3, 2,
    )


def _batch():
    rgb = torch.zeros(2, 5, 3, 64, 64)
    rgb[0, 1:3] = 0.2
    rgb[0, 3:] = 0.7
    rgb[1, 2:] = 0.4
    return SequenceBatch(
        rgb, torch.zeros(2, 5, 1), torch.ones(2, 5, 1, dtype=torch.bool),
        torch.tensor([[0, 3, 2, 4], [0, 2, 4, 4]]), torch.zeros(2, 4),
        torch.tensor([[True, True, True, True], [True, True, False, False]]),
        "grid-v1", 1,
    )


def _config():
    return CoreConfig(device="cpu", z_dim=4, h_dim=3, ensemble_size=2,
                      train_horizon=2, burn_in=1, sigreg_weight=0.1)


def test_inverse_class_weights_balance_present_classes_without_inventing_missing_class():
    # Wrong inverse-frequency normalization would change effective per-action scale.
    exp = _experiment()
    counts = {"0": {"rgb_no_change": 0, "rgb_changed": 8},
              "1": {"rgb_no_change": 6, "rgb_changed": 2},
              "2": {"rgb_no_change": 5, "rgb_changed": 0}}
    weights = exp.action_class_weights(counts, n_actions=3)
    torch.testing.assert_close(weights, torch.tensor([[0., 1.], [2 / 3, 2.], [1., 0.]]))
    assert float(weights[1, 0] * 6) == pytest.approx(float(weights[1, 1] * 2))


def test_auxiliary_target_and_mask_align_with_burn_in_and_autoregressive_state():
    # Teacher forcing at supervised steps or selecting padded targets breaks this.
    exp, model, batch = _experiment(), _model(), _batch()
    with torch.no_grad():
        for gate, delta in zip(model.gate_heads, model.latent_heads):
            gate.weight[0, 0] = 1.0
            delta.bias.fill_(2.0)
    logits, labels, actions = exp.auxiliary_gate_examples(model, batch, train_horizon=2)
    assert labels.tolist() == [0., 1., 1.]
    assert actions.tolist() == [3, 2, 2]
    assert logits.shape == (2, 3)
    encoded = model.encoder(batch.rgb[:, 1])[:, 0]
    expected_next = encoded[0] + 2 * encoded[0].sigmoid()
    torch.testing.assert_close(logits[0], torch.stack([encoded[0], encoded[1], expected_next]))
    mutated_rgb = batch.rgb.clone()
    mutated_rgb[:, 2:] = batch.rgb[:, 1:2]
    changed_logits, changed_labels, _ = exp.auxiliary_gate_examples(
        model, replace(batch, rgb=mutated_rgb), train_horizon=2,
    )
    torch.testing.assert_close(changed_logits, logits, rtol=0, atol=0)
    assert changed_labels.tolist() == [0., 0., 0.]


def test_auxiliary_forward_preserves_exp153_initialization_and_predictions():
    model = _model()
    torch.manual_seed(154)
    baseline = gated.ChangeGatedResidualWorldModel(CoreEncoder(4), {"grid-v1": (5, 1)}, 3, 2)
    assert all(torch.equal(value, baseline.state_dict()[name])
               for name, value in model.state_dict().items())
    batch = _batch()
    state = model.initial_from_tensors(batch.rgb[:, 0], batch.sensors[:, 0],
                                       batch.sensor_mask[:, 0], batch.schema)
    with torch.no_grad():
        for candidate in (model, baseline):
            candidate.gate_heads[0].bias.fill_(0.4)
            candidate.latent_heads[0].bias.fill_(2.0)
    torch.testing.assert_close(model.change_gate_logits(state, batch.actions[:, 0]).sigmoid(),
                               baseline.change_gates(state, batch.actions[:, 0]), rtol=0, atol=0)
    assert torch.equal(model.step(state, batch.actions[:, 0]).member_z,
                       baseline.step(state, batch.actions[:, 0]).member_z)


def test_auxiliary_repeats_predictive_recurrent_inputs_after_real_burn_in():
    # Dropped burn-in hidden or teacher-forced future state changes these inputs.
    exp, model, batch = _experiment(), _model(), _batch()
    with torch.no_grad():
        for delta in model.latent_heads:
            delta.weight.normal_(0, 0.1)
    inputs = []
    hook = model.recurrent.register_forward_pre_hook(
        lambda _module, args: inputs.append(tuple(value.detach().clone() for value in args)))
    try:
        CoreTrainer(model, _config()).compute_loss(batch)
        predictive_inputs = list(inputs)
        inputs.clear()
        exp.auxiliary_gate_examples(model, batch, train_horizon=2)
    finally:
        hook.remove()
    assert len(inputs) == len(predictive_inputs) == 3
    for actual, expected in zip(inputs, predictive_inputs):
        for tensor, reference in zip(actual, expected):
            torch.testing.assert_close(tensor, reference, rtol=1e-6, atol=1e-7)


def test_total_loss_keeps_base_predictive_objective_and_backpropagates_auxiliary():
    exp, model, batch, config = _experiment(), _model(), _batch(), _config()
    weights = torch.ones(5, 2)
    trainer = exp.AuxiliaryGateTrainer(model, config, weights, gate_aux_weight=1.0)
    torch.manual_seed(42)
    predictive = CoreTrainer(model, config).compute_loss(batch)
    torch.manual_seed(42)
    total = trainer.compute_loss(batch)
    torch.testing.assert_close(total, predictive + math.log(2), rtol=1e-6, atol=1e-7)
    assert trainer.loss_components["predictive_loss"] == float(predictive.detach())
    assert trainer.loss_components["gate_aux_loss"] == pytest.approx(math.log(2))
    total.backward()
    assert all(head.bias.grad is not None and head.bias.grad.abs().sum() > 0
               for head in model.gate_heads)
    metrics = trainer.update(batch, Mode.ADAPT)
    assert metrics["loss"] == pytest.approx(metrics["predictive_loss"] + metrics["gate_aux_loss"])
    off = exp.AuxiliaryGateTrainer(model, config, weights, gate_aux_weight=0.0)
    torch.manual_seed(43)
    base = CoreTrainer(model, config).compute_loss(batch)
    torch.manual_seed(43)
    assert torch.equal(off.compute_loss(batch), base)
    assert off.loss_components["gate_aux_loss"] == 0.0


def test_weighted_bce_uses_all_members_and_valid_targets_only():
    exp, model, batch = _experiment(), _model(), _batch()
    with torch.no_grad():
        model.gate_heads[0].bias.fill_(1.)
        model.gate_heads[1].bias.fill_(-1.)
    weights = torch.ones(5, 2)
    weights[2, 1] = 3.0
    trainer = exp.AuxiliaryGateTrainer(model, _config(), weights)
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([[1., 1., 1.], [-1., -1., -1.]]),
        torch.tensor([[0., 1., 1.], [0., 1., 1.]]), reduction="none",
    ) * torch.tensor([1., 3., 3.])
    torch.testing.assert_close(trainer.compute_auxiliary_loss(batch), expected.mean())


def test_safe_v3_loader_and_explicit_v4_auxiliary_checkpoint(tmp_path):
    exp, model = _experiment(), _model()
    probe = TemporalProbe(4, width=5)
    config = _config()
    manifest = {"analysis_git_head": "exp154-head", "baseline_checkpoint_git_head": "exp153-head",
                "budgets": {"dynamics_updates": 1}, "auxiliary": {"gate_aux_weight": 1.0}}
    payload = gated._checkpoint_payload(model, probe, probe, config, manifest)
    path = tmp_path / "baseline.pt"
    torch.save(payload, path)
    restored, _, head, metadata = exp._load_gated_checkpoint(path)
    assert head == "exp154-head" and metadata["event_supervision"] is False
    assert all(torch.equal(value, restored.state_dict()[name])
               for name, value in model.state_dict().items())
    auxiliary = exp._checkpoint_payload(model, probe, probe, config, manifest)
    assert auxiliary["format_version"] == 4
    assert auxiliary["gate_auxiliary"] == manifest["auxiliary"]
    assert auxiliary["event_supervision"] is True
    for mutation in ({"format_version": 2}, {"format_version": 4},
                     {"latent_parameterization": "absolute"}, {"event_supervision": True}):
        torch.save({**payload, **mutation}, path)
        with pytest.raises(ValueError):
            exp._load_gated_checkpoint(path)


def test_one_step_gate_allows_preserved_zero_contact_but_requires_fixed_blocked():
    exp = _experiment()
    baseline = {"contact_failure_layouts": 0, "blocked_noop_failure_layouts": 4}
    candidate = {"contact_failure_layouts": 0, "blocked_noop_failure_layouts": 0,
                 "medians": {"free_forward_prediction_persistence_ratio": 0.5}}
    assert exp._one_step_gate(baseline, candidate, True)
    assert not exp._one_step_gate(baseline, candidate, False)
    assert not exp._one_step_gate(baseline, {**candidate, "contact_failure_layouts": 1}, True)
    assert not exp._one_step_gate(baseline, {**candidate, "blocked_noop_failure_layouts": 1}, True)


@pytest.mark.skipif(not os.environ.get("EXP154_SMOKE_DIR"), reason="set EXP154_SMOKE_DIR to inspect CLI smoke")
def test_smoke_public_artifacts_and_bounded_progress():
    output = Path(os.environ["EXP154_SMOKE_DIR"])
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == result["status"] == "completed"
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["argv"] and manifest["cwd"] and manifest["analysis_git_head"]
    assert manifest["baseline_checkpoint_git_head"]
    assert result["exact_protocol"] is result["auxiliary_one_step_gate"] is False
    assert result["auxiliary_composition_gate"] is False
    assert result["physics_transfer_gate"] is None
    assert result["controls"]["event_balanced_sampling"] is False
    assert result["controls"]["task_success_supervision"] is False
    assert result["controls"]["push2_not_run"] is True
    assert set(result["evaluation"]) == {"ordered_h3", "ordered_h1", "shuffled_h3", "raw_h3"}
    for name, count in (("baseline_one_step_rows.jsonl", 120),
                        ("auxiliary_one_step_rows.jsonl", 120),
                        ("auxiliary_late_fork_rows.jsonl", 125), ("gate_rows.jsonl", 120),
                        ("evaluation_traces.jsonl", 16 * manifest["budgets"]["eval_seeds"])):
        assert len((output / name).read_text().splitlines()) == count
    losses = [json.loads(line) for line in (output / "dynamics_losses.jsonl").read_text().splitlines()]
    assert len(losses) == manifest["budgets"]["dynamics_updates"]
    assert all(row["loss"] == pytest.approx(row["predictive_loss"] + row["gate_aux_loss"])
               for row in losses)
    checkpoint = torch.load(output / "auxiliary_checkpoint.pt", weights_only=True, map_location="cpu")
    assert checkpoint["format_version"] == 4
    assert checkpoint["gate_auxiliary"]["gate_aux_weight"] == 1.0
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "completed"
    assert max(right["elapsed_seconds"] - left["elapsed_seconds"]
               for left, right in zip(progress, progress[1:])) <= 31
    assert (output / "run.log").stat().st_size > 0
