"""Frozen before-state probes: split, labels, sampling, and diagnostic artifacts."""

from __future__ import annotations

from dataclasses import replace
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Observation, Transition


def _experiment():
    name = "experiments.exp152_representation_separability"
    assert importlib.util.find_spec(name) is not None, "representation diagnostic is missing"
    return importlib.import_module(name)


def _episode(uid: str) -> Episode:
    before = Observation(np.full((3, 64, 64), 20, dtype=np.uint8),
                         np.array([0.0], dtype=np.float32), np.array([True]),
                         "grid-v1", 0)
    after = replace(before, sensors=np.array([99.0], dtype=np.float32), step=1)
    return Episode(uid, "adapt", "push_box", "fixture", (
        Transition(before, 3, after, False, True),
    ))


def test_episode_split_uses_first_three_quarters_and_has_no_transition_leakage():
    exp = _experiment()
    episodes = {name: [_episode(f"{name}:{i}") for i in range(8)]
                for name in ("east", "west")}
    train, heldout, cutoff = exp._split_episodes(episodes, 8)
    assert cutoff == 6
    assert [ep.uid for ep in train["east"]] == [f"east:{i}" for i in range(6)]
    assert [ep.uid for ep in heldout["east"]] == ["east:6", "east:7"]
    fit_transitions = {id(t) for rows in train.values() for ep in rows for t in ep.transitions}
    held_transitions = {id(t) for rows in heldout.values() for ep in rows for t in ep.transitions}
    assert not fit_transitions & held_transitions
    episodes["west"][-1] = episodes["east"][0]
    with pytest.raises(ValueError, match="duplicate|overlap"):
        exp._split_episodes(episodes, 8)


def test_after_rgb_changes_label_but_never_changes_encoded_before_input(tmp_path):
    exp = _experiment()
    original = _episode("same-before")
    transition = original.transitions[0]
    changed = replace(transition, after=replace(
        transition.after, rgb=np.full((3, 64, 64), 250, dtype=np.uint8)))
    forward = replace(changed, action=2)
    episode = replace(original, transitions=(transition, changed, forward))
    encoder = CoreEncoder(4).eval().requires_grad_(False)
    with exp.ProgressJournal(tmp_path / "progress.jsonl", 1) as journal:
        x, labels = exp._encode_task(encoder, [episode], 3, 2, "cpu",
                                     time.monotonic() + 30, journal, "train")
        forward_x, forward_labels = exp._encode_task(
            encoder, [episode], 2, 2, "cpu", time.monotonic() + 30, journal, "train")
    assert labels.tolist() == [0, 1]
    assert forward_labels.tolist() == [1]
    assert torch.equal(x[0], x[1])
    assert torch.allclose(x[0], forward_x[0], atol=1e-7)
    expected = encoder(torch.tensor(transition.before.rgb[None]).float() / 255)
    assert torch.allclose(x[0], expected[0], atol=1e-7)
    assert not x.requires_grad
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_balanced_batches_are_seeded_and_sample_with_replacement():
    exp = _experiment()
    labels = np.array([0, 0, 0, 0, 1])
    left, right = np.random.default_rng(152), np.random.default_rng(152)
    for _ in range(3):
        first = exp._balanced_indices(labels, 8, left)
        second = exp._balanced_indices(labels, 8, right)
        assert np.array_equal(first, second)
        assert np.bincount(labels[first], minlength=2).tolist() == [4, 4]
        assert int(np.count_nonzero(first == 4)) == 4
    with pytest.raises(ValueError, match="even"):
        exp._balanced_indices(labels, 7, left)
    with pytest.raises(ValueError, match="both classes"):
        exp._balanced_indices(np.zeros(4, dtype=int), 8, left)


def test_metrics_report_hand_counted_confusion_and_class_recalls():
    metrics = _experiment()._metrics(
        torch.tensor([0, 0, 0, 1, 1]), torch.tensor([-2.0, 1.0, -1.0, 0.0, -3.0]))
    assert metrics["class_counts"] == {"0": 3, "1": 2}
    assert metrics["confusion"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert metrics["minority_class"] == 1
    assert metrics["minority_recall"] == 0.5
    assert metrics["majority_recall"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(7 / 12)


@pytest.mark.parametrize("exact,interact,forward,want", [
    (True, True, True, "representation_signal_evidence"),
    (True, False, True, "contact_representation_bottleneck_evidence"),
    (True, False, False, "encoder_objective_bottleneck_evidence"),
    (True, True, False, "mixed_or_inconclusive"),
    (False, True, True, "non_preregistered_protocol"),
])
def test_interpretation_requires_exact_protocol_and_both_tasks(exact, interact, forward, want):
    exp = _experiment()
    assert exp._outcome_label({"interact_rgb_changed": interact,
                               "forward_rgb_changed": forward}, exact) == want


@pytest.mark.parametrize("accuracy,recall0,recall1,control,want", [
    (0.8, 0.7, 0.9, 0.6, True),
    (0.799, 0.7, 0.898, 0.5, False),
    (0.85, 0.699, 1.0, 0.5, False),
    (0.85, 1.0, 0.699, 0.5, False),
    (0.8, 0.7, 0.9, 0.601, False),
    (float("nan"), 0.8, 0.8, 0.5, False),
])
def test_task_signal_requires_accuracy_each_recall_and_shuffled_margin(
    accuracy, recall0, recall1, control, want
):
    ordered = {"balanced_accuracy": accuracy, "recall_by_class": {"0": recall0, "1": recall1}}
    assert _experiment()._task_signal(ordered, {"balanced_accuracy": control}) is want


@pytest.mark.skipif(not os.environ.get("EXP152_SMOKE_DIR"),
                    reason="set EXP152_SMOKE_DIR to inspect a completed CLI smoke")
def test_smoke_public_artifacts_and_bounded_progress():
    output = Path(os.environ["EXP152_SMOKE_DIR"])
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == result["status"] == "completed"
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["argv"] and manifest["cwd"] and manifest["budgets"]
    assert manifest["analysis_git_head"] and manifest["checkpoint_git_head"]
    assert manifest["checkpoint_metadata"]["latent_parameterization"] == "residual_zero_init"
    assert result["exact_protocol"] is False
    assert result["representation_signal_evidence"] is False
    assert result["outcome_label"] == "non_preregistered_protocol"
    assert result["controls"]["encoder_frozen"] is True
    assert result["controls"]["input"] == "z(before.rgb) only"
    assert result["controls"]["push2_run"] is False
    assert result["controls"]["goal_success_evaluated"] is False
    assert set(result["tasks"]) == {"interact_rgb_changed", "forward_rgb_changed"}
    for task in result["tasks"].values():
        assert task["train_counts"] == task["shuffled_train_counts"]
        for arm in ("ordered", "shuffled"):
            assert task[arm]["updates"] == manifest["budgets"]["probe_updates"]
            count = manifest["budgets"]["probe_updates"] * manifest["budgets"]["probe_batch_size"] // 2
            assert task[arm]["sampled_class_counts"] == {"0": count, "1": count}
            assert task[arm]["heldout"]["class_counts"] == task["heldout_counts"]
            assert 0 <= task[arm]["heldout"]["balanced_accuracy"] <= 1
    progress = [json.loads(line) for line in (output / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "completed"
    assert all(set(row) >= {"stage", "completed", "total", "elapsed_seconds"}
               for row in progress[:-1])
    assert max(b["elapsed_seconds"] - a["elapsed_seconds"]
               for a, b in zip(progress, progress[1:])) < 31
    assert (output / "run.log").stat().st_size > 0
