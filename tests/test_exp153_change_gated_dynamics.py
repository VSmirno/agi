"""Multiplicative change gates and the public matched-protocol artifact contract."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp146_temporal_mpc_physics import TemporalProbe
from experiments.exp147_rollout_localization import _load_checkpoint as load_absolute
from experiments.exp151_event_balanced_dynamics import _load_residual_checkpoint
from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.pipeline.core_config import CoreConfig


def _experiment():
    name = "experiments.exp153_change_gated_dynamics"
    assert importlib.util.find_spec(name) is not None, "change-gated experiment is missing"
    return importlib.import_module(name)


def _model(model_type=None, **flags):
    torch.manual_seed(146)
    if model_type is None:
        model_type = _experiment().ChangeGatedResidualWorldModel
    return model_type(CoreEncoder(4), {"grid-v1": (5, 1)}, 3, 2, **flags)


def _state():
    return LatentState(
        torch.tensor([[2.0, -3.0, 0.5, 7.0]]).repeat(5, 1),
        torch.arange(5.0)[:, None],
        torch.tensor([[True], [False], [True], [False], [True]]),
        torch.ones(5, 3), "grid-v1",
    )


def test_initial_gate_is_half_and_every_action_predicts_exact_persistence():
    # Catches a lost skip connection or initialization changing the shared backbone.
    model, baseline = _model(), _model(CoreWorldModel)
    for name, value in baseline.state_dict().items():
        if not name.startswith("latent_heads."):
            assert torch.equal(model.state_dict()[name], value), name
    state, actions = _state(), torch.arange(5)
    assert torch.equal(model.change_gates(state, actions), torch.full((2, 5, 1), 0.5))
    prediction = model.step(state, actions)
    assert torch.equal(prediction.member_z, state.z[None].expand(2, -1, -1))
    assert torch.equal(prediction.next_state.z, state.z)
    assert torch.equal(prediction.uncertainty, torch.zeros(5))


def test_each_delta_is_multiplied_by_its_own_gate_before_mean_and_variance():
    # Catches additive gates, a shared gate, or uncertainty left on ungated deltas.
    model = _model()
    with torch.no_grad():
        model.latent_heads[0].bias.fill_(2.0)
        model.latent_heads[1].bias.fill_(4.0)
        model.gate_heads[0].bias.fill_(-math.log(3))
        model.gate_heads[1].bias.fill_(math.log(3))
    state = _state()
    prediction = model.step(state, torch.arange(5))
    torch.testing.assert_close(prediction.member_z[0], state.z + 0.5)
    torch.testing.assert_close(prediction.member_z[1], state.z + 3.0)
    torch.testing.assert_close(prediction.next_state.z, state.z + 1.75)
    torch.testing.assert_close(prediction.uncertainty, torch.full((5,), 1.5625))
    rollout = model.rollout(state, torch.zeros(5, 2, dtype=torch.long))
    torch.testing.assert_close(rollout[-1].next_state.z, state.z + 3.5)


@pytest.mark.parametrize("normalize_sensor_condition", [False, True])
@pytest.mark.parametrize("predict_sensor_delta", [False, True])
def test_shared_hidden_sensor_mask_and_termination_plumbing_is_unchanged(
    normalize_sensor_condition, predict_sensor_delta
):
    # Catches changes to body conditioning, sensor deltas, recurrence or termination.
    model = _model(normalize_sensor_condition=normalize_sensor_condition,
                   predict_sensor_delta=predict_sensor_delta)
    state, actions = _state(), torch.arange(5)
    original = CoreWorldModel.step(model, state, actions)
    prediction = model.step(state, actions)
    for name in ("hidden", "sensors", "sensor_mask"):
        assert torch.equal(getattr(prediction.next_state, name),
                           getattr(original.next_state, name)), name
    assert prediction.next_state.schema == state.schema
    assert torch.equal(prediction.terminated_prob, original.terminated_prob)
    assert prediction.member_z.shape == (2, 5, 4)
    assert prediction.next_state.hidden.shape == (5, 3)


def test_gate_uses_current_z_and_existing_action_embedding_directly():
    # Catches gating from hidden/body alone or dropping either current-z or action.
    model = _model()
    with torch.no_grad():
        model.action_embeddings["grid-v1"].weight.zero_()
        model.action_embeddings["grid-v1"].weight[:, 0] = torch.arange(-2.0, 3.0)
        for gate, delta in zip(model.gate_heads, model.latent_heads):
            gate.weight[0, 0] = 1.0
            gate.weight[0, 4] = 1.0
            delta.bias.fill_(2.0)
    state = replace(_state(), z=torch.zeros(5, 4))
    actions = torch.arange(5)
    expected = torch.tensor([-2., -1., 0., 1., 2.]).sigmoid()
    torch.testing.assert_close(model.change_gates(state, actions),
                               expected[None, :, None].expand(2, -1, -1))
    changed = replace(state, z=torch.tensor([[2., 0., 0., 0.]]).repeat(5, 1),
                      hidden=state.hidden * 99, sensors=state.sensors + 100)
    changed_expected = torch.tensor([0., 1., 2., 3., 4.]).sigmoid()
    torch.testing.assert_close(model.change_gates(changed, actions),
                               changed_expected[None, :, None].expand(2, -1, -1))
    torch.testing.assert_close(model.step(changed, actions).next_state.z,
                               changed.z + 2 * changed_expected[:, None])


def test_checkpoint_safely_reconstructs_gated_semantics_and_old_loaders_refuse(tmp_path):
    # Catches ambiguous checkpoint tags or missing trained gate tensors/metadata.
    exp, model = _experiment(), _model()
    with torch.no_grad():
        model.latent_heads[0].bias.fill_(2.)
        model.gate_heads[0].bias.fill_(1.)
    probe = TemporalProbe(4, width=5)
    config = CoreConfig(device="cpu", z_dim=4, h_dim=3, ensemble_size=2)
    manifest = {"analysis_git_head": "exp153-head", "baseline_checkpoint_git_head": "exp150-head",
                "budgets": {"dynamics_updates": 1}}
    path = tmp_path / "gated.pt"
    torch.save(exp._checkpoint_payload(model, probe, probe, config, manifest), path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    assert payload["format_version"] == 3
    assert payload["latent_parameterization"] == "gated_residual_zero_init"
    meta = payload["modules"]["model"]
    assert meta["class"] == "experiments.exp153_change_gated_dynamics.ChangeGatedResidualWorldModel"
    assert meta["gate"]["input"] == "concat(current_z, existing_action_embedding)"
    assert meta["gate"]["weight_init"] == meta["gate"]["bias_init"] == 0.0
    assert meta["gate"]["initial_probability"] == 0.5
    restored = exp.ChangeGatedResidualWorldModel(
        CoreEncoder(meta["z_dim"]), {key: tuple(value) for key, value in meta["schemas"].items()},
        meta["h_dim"], meta["ensemble_size"],
        normalize_sensor_condition=meta["normalize_sensor_condition"],
        predict_sensor_delta=meta["predict_sensor_delta"],
    )
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    torch.testing.assert_close(restored.step(_state(), torch.arange(5)).member_z,
                               model.step(_state(), torch.arange(5)).member_z)
    for loader in (load_absolute, _load_residual_checkpoint):
        with pytest.raises(ValueError, match="format_version"):
            loader(path)


@pytest.mark.skipif(not os.environ.get("EXP153_SMOKE_DIR"),
                    reason="set EXP153_SMOKE_DIR to inspect a completed CLI smoke")
def test_smoke_public_artifacts_and_bounded_progress():
    # Catches incomplete diagnostics/progress or scientific claims from reduced budgets.
    output = Path(os.environ["EXP153_SMOKE_DIR"])
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == result["status"] == "completed"
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["argv"] and manifest["cwd"] and manifest["budgets"]
    assert manifest["analysis_git_head"] and manifest["baseline_checkpoint_git_head"]
    assert manifest["baseline_checkpoint_metadata"]["latent_parameterization"] == "residual_zero_init"
    assert result["exact_protocol"] is False
    assert result["gated_one_step_gate"] is False
    assert result["gated_composition_gate"] is False
    assert result["physics_transfer_gate"] is None
    assert result["controls"]["event_balanced_sampling"] is False
    assert result["controls"]["event_supervision"] is False
    assert result["controls"]["push2_not_run"] is True
    assert set(result["evaluation"]) == {"ordered_h3", "ordered_h1", "shuffled_h3", "raw_h3"}
    for name, count in (("baseline_one_step_rows.jsonl", 120),
                        ("gated_one_step_rows.jsonl", 120),
                        ("gated_late_fork_rows.jsonl", 125), ("gate_rows.jsonl", 120),
                        ("evaluation_traces.jsonl", 16 * manifest["budgets"]["eval_seeds"])):
        assert len((output / name).read_text().splitlines()) == count, name
    for arm in ("baseline_one_step", "gated_one_step"):
        assert set(result[arm]["splits"]) == {"source", "unseen"}
    gate_rows = [json.loads(line) for line in (output / "gate_rows.jsonl").read_text().splitlines()]
    assert all(0 <= row["min"] <= row["mean"] <= row["max"] <= 1 for row in gate_rows)
    assert all(len(row["by_member"]) == 3 for row in gate_rows)
    checkpoint = torch.load(output / "gated_checkpoint.pt", weights_only=True, map_location="cpu")
    assert checkpoint["latent_parameterization"] == "gated_residual_zero_init"
    assert checkpoint["analysis_git_head"] == manifest["analysis_git_head"]
    meta = checkpoint["modules"]["model"]
    model = _experiment().ChangeGatedResidualWorldModel(
        CoreEncoder(meta["z_dim"]), {key: tuple(value) for key, value in meta["schemas"].items()},
        meta["h_dim"], meta["ensemble_size"],
        normalize_sensor_condition=meta["normalize_sensor_condition"],
        predict_sensor_delta=meta["predict_sensor_delta"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["ordered_probe_state_dict"] and checkpoint["shuffled_probe_state_dict"]
    assert any(torch.count_nonzero(value) for name, value in model.state_dict().items()
               if name.startswith("gate_heads."))
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "completed"
    assert all(set(row) >= {"stage", "completed", "total", "elapsed_seconds"}
               for row in progress[:-1])
    assert max(b["elapsed_seconds"] - a["elapsed_seconds"]
               for a, b in zip(progress, progress[1:])) < 31
    assert (output / "run.log").stat().st_size > 0
