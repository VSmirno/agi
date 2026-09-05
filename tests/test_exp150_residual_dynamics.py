"""Residual transition semantics and opt-in public experiment artifacts."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder


def _experiment():
    name = "experiments.exp150_residual_dynamics"
    assert importlib.util.find_spec(name) is not None, "residual experiment is missing"
    return importlib.import_module(name)


def _model(model_type):
    torch.manual_seed(146)
    return model_type(CoreEncoder(4), {"grid-v1": (5, 1)}, 3, 2)


def _state():
    return LatentState(
        torch.tensor([[2.0, -3.0, 0.5, 7.0]]).repeat(5, 1),
        torch.arange(5.0)[:, None],
        torch.tensor([[True], [False], [True], [False], [True]]),
        torch.ones(5, 3),
        "grid-v1",
    )


def test_zero_init_predicts_exact_persistence_for_every_action():
    # Catches absolute prediction or accidentally zeroing the shared backbone.
    residual = _model(_experiment().ResidualLatentWorldModel)
    baseline = _model(CoreWorldModel)
    for name, value in residual.state_dict().items():
        if name.startswith("latent_heads."):
            assert torch.count_nonzero(value) == 0
        else:
            assert torch.equal(value, baseline.state_dict()[name]), name
    state = _state()
    actions = torch.arange(5)
    prediction = residual.step(state, actions)
    assert torch.equal(prediction.member_z, state.z[None].expand(2, -1, -1))
    assert torch.equal(prediction.next_state.z, state.z)
    assert torch.equal(prediction.uncertainty, torch.zeros(5))
    original = CoreWorldModel.step(residual, state, actions)
    assert torch.equal(prediction.next_state.hidden, original.next_state.hidden)
    assert torch.equal(prediction.next_state.sensors, original.next_state.sensors)
    assert torch.equal(prediction.next_state.sensor_mask, state.sensor_mask)
    assert torch.equal(prediction.terminated_prob, original.terminated_prob)


def test_learned_heads_add_delta_and_report_member_variance():
    # Catches forgetting the skip connection, averaging before addition, or stale spread.
    model = _model(_experiment().ResidualLatentWorldModel)
    with torch.no_grad():
        model.latent_heads[0].bias.fill_(1.0)
        model.latent_heads[1].bias.fill_(3.0)
    state = _state()
    prediction = model.step(state, torch.arange(5))
    assert torch.equal(prediction.member_z[0], state.z + 1.0)
    assert torch.equal(prediction.member_z[1], state.z + 3.0)
    assert torch.equal(prediction.next_state.z, state.z + 2.0)
    assert torch.equal(prediction.uncertainty, torch.ones(5))
    rollout = model.rollout(state, torch.zeros(5, 2, dtype=torch.long))
    assert torch.equal(rollout[-1].next_state.z, state.z + 4.0)


@pytest.mark.parametrize(
    "exact,contacts,blocked,free,baseline_contacts,baseline_blocked,want",
    [
        (True, 0, 0, 0.9, 4, 4, True),
        (False, 0, 0, 0.9, 4, 4, False),
        (True, 1, 0, 0.9, 4, 4, False),
        (True, 0, 1, 0.9, 4, 4, False),
        (True, 0, 0, 1.0, 4, 4, False),
        (True, 0, 0, None, 4, 4, False),
        (True, 0, 0, 0.9, 0, 4, False),
        (True, 0, 0, 0.9, 4, 0, False),
    ],
)
def test_one_step_gate_requires_exact_protocol_and_both_improvements(
    exact, contacts, blocked, free, baseline_contacts, baseline_blocked, want
):
    baseline = {"contact_failure_layouts": baseline_contacts,
                "blocked_noop_failure_layouts": baseline_blocked}
    residual = {"contact_failure_layouts": contacts,
                "blocked_noop_failure_layouts": blocked,
                "medians": {"free_forward_prediction_persistence_ratio": free}}
    assert _experiment()._one_step_gate(baseline, residual, exact) is want


@pytest.mark.skipif(not os.environ.get("EXP150_SMOKE_DIR"),
                    reason="set EXP150_SMOKE_DIR to inspect a completed CLI smoke")
def test_smoke_public_artifact_contract():
    # Catches missing phases/rows, wrong checkpoint type, and unjustified smoke PASS.
    output = Path(os.environ["EXP150_SMOKE_DIR"])
    results = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["status"] == results["status"] == "completed"
    assert manifest["analysis_git_head"] and manifest["baseline_checkpoint_git_head"]
    assert manifest["argv"] and manifest["cwd"] and manifest["budgets"]
    for name, count in (
        ("baseline_one_step_rows.jsonl", 120),
        ("residual_one_step_rows.jsonl", 120),
        ("residual_late_fork_rows.jsonl", 125),
        ("evaluation_traces.jsonl", 16 * manifest["budgets"]["eval_seeds"]),
    ):
        rows = [json.loads(line) for line in (output / name).read_text().splitlines()]
        assert len(rows) == count, name
    checkpoint = torch.load(output / "residual_checkpoint.pt", weights_only=True,
                            map_location="cpu")
    assert checkpoint["latent_parameterization"] == "residual_zero_init"
    assert checkpoint["analysis_git_head"] == manifest["analysis_git_head"]
    metadata = checkpoint["modules"]["model"]
    model = _experiment().ResidualLatentWorldModel(
        CoreEncoder(metadata["z_dim"]),
        {name: tuple(shape) for name, shape in metadata["schemas"].items()},
        metadata["h_dim"], metadata["ensemble_size"],
        normalize_sensor_condition=metadata["normalize_sensor_condition"],
        predict_sensor_delta=metadata["predict_sensor_delta"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["ordered_probe_state_dict"] and checkpoint["shuffled_probe_state_dict"]
    assert results["physics_transfer_gate"] is None
    assert results["residual_one_step_gate"] is False
    assert results["residual_composition_gate"] is False
    assert set(results["evaluation"]) == {
        "ordered_h3", "ordered_h1", "shuffled_h3", "raw_h3"
    }
    for arm in ("baseline_one_step", "residual_one_step"):
        assert set(results[arm]["splits"]) == {"source", "unseen"}
        assert results[arm]["outcome_label"]
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "completed"
    assert all(set(row) >= {"stage", "completed", "total", "elapsed_seconds"}
               for row in progress[:-1])
    assert (output / "run.log").stat().st_size > 0
