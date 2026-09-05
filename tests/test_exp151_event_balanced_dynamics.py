"""Event-window alignment, residual loading, and public exp151 artifacts."""

from __future__ import annotations

from dataclasses import asdict
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp146_temporal_mpc_physics import TemporalProbe
from experiments.exp147_rollout_localization import _load_checkpoint as load_absolute
from experiments.exp150_residual_dynamics import ResidualLatentWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Mode, Observation, Transition
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import tensorize
from snks.pipeline.core_config import CoreConfig


def _experiment():
    name = "experiments.exp151_event_balanced_dynamics"
    assert importlib.util.find_spec(name) is not None, "event-balanced experiment is missing"
    return importlib.import_module(name)


def _episode(uid, actions, pixels, schema="grid-v1"):
    observations = [
        Observation(np.full((3, 64, 64), pixel, dtype=np.uint8),
                    np.array([step], dtype=np.float32), np.array([True]), schema, step)
        for step, pixel in enumerate(pixels)
    ]
    return Episode(uid, "adapt", "fixture", "r1", tuple(
        Transition(observations[i], action, observations[i + 1],
                   False, i == len(actions) - 1)
        for i, action in enumerate(actions)
    ))


def _replay(*episodes):
    replay = SequenceReplay(8, 145)
    for episode in episodes:
        replay.append(episode, Mode.ADAPT)
    return replay


@pytest.mark.parametrize("burn_in,event_start,ordinary_starts", [
    (0, 1, {0, 2, 3}), (1, 0, {1, 2}),
])
def test_event_pool_uses_first_loss_transition_and_preserves_window_support(
    burn_in, event_start, ordinary_starts
):
    # Catches burn-in/last-target labeling and excluding later events from ordinary.
    replay = _replay(
        _episode("six", [3, 3, 2, 4, 3, 3], [0, 0, 1, 2, 2, 3, 4]),
        _episode("wrong-schema", [3, 4, 4, 4], [0, 1, 1, 1, 1], "other"),
    )
    sampler = _experiment().EventBalancedSampler(replay, 3, burn_in, 145)
    assert sampler.report()["pool_sizes"] == {"event": 1, "ordinary": len(ordinary_starts)}
    observed_ordinary = set()
    for _ in range(12):
        windows = sampler.sample(8)
        batch = tensorize(windows, burn_in, "cpu")
        anchors = [window.transitions[burn_in] for window in windows]
        flags = [t.action == 3 and not np.array_equal(t.before.rgb, t.after.rgb)
                 for t in anchors]
        assert sum(flags) == 4
        assert batch.valid[:, burn_in].all()
        assert batch.actions.shape == (8, 3 + burn_in)
        assert all(window.uid == "six" for window in windows)
        for window, event in zip(windows, flags):
            start = window.transitions[0].before.step
            if event:
                assert start == event_start
            else:
                observed_ordinary.add(start)
    assert observed_ordinary == ordinary_starts


def test_sampling_is_seeded_with_replacement_and_counts_all_supervised_targets():
    # One eligible window per stratum forces replacement; later events count in loss.
    replay = _replay(_episode("event", [3, 4, 3], [0, 1, 1, 2]),
                     _episode("ordinary", [4, 3, 4], [0, 0, 1, 1]))
    first = _experiment().EventBalancedSampler(replay, 3, 0, 145)
    second = _experiment().EventBalancedSampler(replay, 3, 0, 145)
    for _ in range(3):
        left, right = first.sample(8), second.sample(8)
        assert [window.uid for window in left] == [window.uid for window in right]
        assert sum(window.uid == "event" for window in left) == 4
    report = first.report()
    assert report["batches"] == 3
    assert report["sampled_anchors"] == {"event": 12, "ordinary": 12}
    assert report["supervised_transitions"] == {"event": 36, "ordinary": 36}
    assert report["supervised_by_position"] == [
        {"event": 12, "ordinary": 12}, {"event": 12, "ordinary": 12},
        {"event": 12, "ordinary": 12},
    ]
    with pytest.raises(ValueError, match="even"):
        first.sample(7)


def test_short_windows_only_count_real_targets_and_empty_strata_fail():
    # Catches padding counted as ordinary and silent fallback to unbalanced replay.
    replay = _replay(_episode("short", [3], [0, 1]),
                     _episode("ordinary", [4, 4, 4], [0, 0, 0, 0]))
    sampler = _experiment().EventBalancedSampler(replay, 3, 0, 145)
    batch = tensorize(sampler.sample(8), 0, "cpu")
    assert int(batch.valid.sum()) == 16
    assert sampler.report()["supervised_transitions"] == {"event": 4, "ordinary": 12}
    with pytest.raises(ValueError, match="event.*empty"):
        _experiment().EventBalancedSampler(replay, 3, 1, 145)


def _payload():
    config = CoreConfig(device="cpu", z_dim=4, h_dim=3, ensemble_size=2)
    model = ResidualLatentWorldModel(CoreEncoder(4), {"grid-v1": (5, 1)}, 3, 2)
    probe = TemporalProbe(4, width=5)
    return model, {
        "format_version": 2, "latent_parameterization": "residual_zero_init",
        "git_head": "exp150-training-head", "config": asdict(config), "budgets": {},
        "modules": {
            "model": {"class": "experiments.exp150_residual_dynamics.ResidualLatentWorldModel",
                      "schemas": {"grid-v1": [5, 1]}, "z_dim": 4, "h_dim": 3,
                      "ensemble_size": 2, "normalize_sensor_condition": False,
                      "predict_sensor_delta": False},
            "probe": {"z_dim": 4, "width": 5},
        },
        "model_state_dict": model.state_dict(),
        "ordered_probe_state_dict": probe.state_dict(),
        "shuffled_probe_state_dict": probe.state_dict(),
    }


def test_safe_residual_loader_reconstructs_delta_and_absolute_loader_refuses(tmp_path):
    original, payload = _payload()
    path = tmp_path / "residual.pt"
    torch.save(payload, path)
    loaded, probe, head, metadata = _experiment()._load_residual_checkpoint(path)
    assert isinstance(loaded, ResidualLatentWorldModel)
    assert head == "exp150-training-head"
    assert metadata["latent_parameterization"] == "residual_zero_init"
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    state = original.initial(_episode("one", [3], [12, 13]).transitions[0].before)
    prediction = loaded.step(state, torch.tensor([3]))
    assert torch.equal(prediction.next_state.z, state.z)
    assert not any(p.requires_grad for p in probe.parameters())
    with pytest.raises(ValueError, match="format_version"):
        load_absolute(path)


@pytest.mark.parametrize("bad", ["version", "parameterization", "class", "config", "shape"])
def test_residual_loader_rejects_ambiguous_or_inconsistent_payload(tmp_path, bad):
    _, payload = _payload()
    if bad == "version":
        payload["format_version"] = 1
    elif bad == "parameterization":
        payload["latent_parameterization"] = "absolute"
    elif bad == "class":
        payload["modules"]["model"]["class"] = "snks.agent.core_world_model.CoreWorldModel"
    elif bad == "config":
        payload["config"]["z_dim"] = 8
    else:
        payload["model_state_dict"]["latent_heads.0.bias"] = torch.zeros(8)
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError):
        _experiment()._load_residual_checkpoint(path)


@pytest.mark.skipif(not os.environ.get("EXP151_SMOKE_DIR"),
                    reason="set EXP151_SMOKE_DIR to inspect a completed CLI smoke")
def test_smoke_public_artifact_contract():
    # Catches missing phases, mismatched checkpoint semantics, or smoke PASS claims.
    output = Path(os.environ["EXP151_SMOKE_DIR"])
    results = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["status"] == results["status"] == "completed"
    assert manifest["analysis_git_head"] and manifest["baseline_checkpoint_git_head"]
    assert manifest["argv"] and manifest["cwd"] and manifest["budgets"]
    assert manifest["fixed_protocol"]["event_fraction"] == 0.5
    assert manifest["fixed_corpus"]["transitions"] == 130676
    assert manifest["fixed_corpus"]["rgb_changing_interact_transitions"] == 1925
    for name, count in (("baseline_one_step_rows.jsonl", 120),
                        ("event_balanced_one_step_rows.jsonl", 120),
                        ("event_balanced_late_fork_rows.jsonl", 125),
                        ("evaluation_traces.jsonl", 16 * manifest["budgets"]["eval_seeds"])):
        assert len((output / name).read_text().splitlines()) == count
    checkpoint = torch.load(output / "event_balanced_checkpoint.pt", weights_only=True,
                            map_location="cpu")
    assert checkpoint["event_balanced"] is True
    assert checkpoint["latent_parameterization"] == "residual_zero_init"
    assert checkpoint["analysis_git_head"] == manifest["analysis_git_head"]
    assert checkpoint["sampling"] == results["sampling"]
    _experiment()._load_residual_checkpoint(output / "event_balanced_checkpoint.pt")
    sampling = results["sampling"]
    expected = manifest["budgets"]["dynamics_updates"] * 4
    assert sampling["sampled_anchors"] == {"event": expected, "ordinary": expected}
    assert all(sampling["pool_sizes"].values())
    assert sampling["batches"] == manifest["budgets"]["dynamics_updates"]
    assert results["physics_transfer_gate"] is None
    assert results["event_balanced_one_step_gate"] is False
    assert results["event_balanced_composition_gate"] is False
    for arm in ("baseline_one_step", "event_balanced_one_step"):
        assert set(results[arm]["splits"]) == {"source", "unseen"}
    assert set(results["evaluation"]) == {"ordered_h3", "ordered_h1", "shuffled_h3", "raw_h3"}
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "completed"
    assert all(set(row) >= {"stage", "completed", "total", "elapsed_seconds"}
               for row in progress[:-1])
    assert max(b["elapsed_seconds"] - a["elapsed_seconds"]
               for a, b in zip(progress, progress[1:])) < 31
    assert (output / "run.log").stat().st_size > 0
