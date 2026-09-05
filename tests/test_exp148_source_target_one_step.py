from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from experiments.exp143_temporal_proximity import TemporalProbe
from experiments.exp148_source_target_one_step import (
    _aggregate_split,
    main,
    _outcome_label,
)
from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(148)
    model = CoreWorldModel(
        CoreEncoder(4),
        {"grid-v1": (5, 1)},
        h_dim=3,
        heads=2,
        normalize_sensor_condition=False,
        predict_sensor_delta=False,
    )
    probe = TemporalProbe(4, width=5)
    torch.save(
        {
            "format_version": 1,
            "git_head": "checkpoint-training-head",
            "budgets": {},
            "config": {
                "device": "cpu",
                "z_dim": 4,
                "h_dim": 3,
                "ensemble_size": 2,
                "normalize_sensor_condition": False,
                "predict_sensor_delta": False,
            },
            "modules": {
                "model": {
                    "schemas": {"grid-v1": [5, 1]},
                    "z_dim": 4,
                    "h_dim": 3,
                    "ensemble_size": 2,
                    "normalize_sensor_condition": False,
                    "predict_sensor_delta": False,
                },
                "probe": {"z_dim": 4, "width": 5},
            },
            "model_state_dict": model.state_dict(),
            "ordered_probe_state_dict": probe.state_dict(),
            "shuffled_probe_state_dict": probe.state_dict(),
        },
        path,
    )


def _summary(
    layout: str,
    *,
    contact_failure: bool,
    blocked_noop_failure: bool,
    interact_ratios: list[float | None],
    free_ratio: float | None,
    blocked_mse: float,
):
    return {
        "layout": layout,
        "contact_failure": contact_failure,
        "blocked_noop_failure": blocked_noop_failure,
        "interact_prediction_persistence_ratios": interact_ratios,
        "free_forward_prediction_persistence_ratio": free_ratio,
        "blocked_forward_prediction_mse": blocked_mse,
    }


def test_split_aggregation_and_outcome_labels_are_deterministic():
    source = [
        _summary(
            "east_row2",
            contact_failure=False,
            blocked_noop_failure=False,
            interact_ratios=[1.0, None],
            free_ratio=0.5,
            blocked_mse=0.2,
        ),
        _summary(
            "west_row3",
            contact_failure=True,
            blocked_noop_failure=False,
            interact_ratios=[3.0, 5.0],
            free_ratio=1.5,
            blocked_mse=0.4,
        ),
    ]
    source_summary = _aggregate_split(source)

    assert source_summary["layout_count"] == 2
    assert source_summary["contact_failure_layouts"] == 1
    assert source_summary["blocked_noop_failure_layouts"] == 0
    assert source_summary["medians"]["interact_prediction_persistence_ratio"] == pytest.approx(3.0)
    assert source_summary["medians"]["free_forward_prediction_persistence_ratio"] == pytest.approx(1.0)
    assert source_summary["medians"]["blocked_forward_prediction_mse"] == pytest.approx(0.3)

    clean = _aggregate_split(
        [
            _summary(
                "a",
                contact_failure=False,
                blocked_noop_failure=False,
                interact_ratios=[0.5],
                free_ratio=0.5,
                blocked_mse=0.01,
            )
        ]
    )
    all_failed = _aggregate_split(
        [
            _summary(
                "a",
                contact_failure=True,
                blocked_noop_failure=True,
                interact_ratios=[2.0],
                free_ratio=2.0,
                blocked_mse=1.0,
            )
        ]
    )

    assert _outcome_label(clean, clean) == "mixed_or_inconclusive"
    assert _outcome_label(clean, all_failed) == "unseen_only_failure_evidence"
    assert _outcome_label(all_failed, all_failed) == "shared_one_step_failure_evidence"
    assert _outcome_label(source_summary, all_failed) == "mixed_or_inconclusive"


def test_cli_writes_exact_manifest_progress_and_120_rows(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "diagnostic"
    _write_checkpoint(checkpoint)

    assert (
        main(
            [
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(output),
                "--progress-interval",
                "1",
            ]
        )
        == 0
    )

    manifest = json.loads((output / "manifest.json").read_text())
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    rows = (output / "diagnostic_rows.jsonl").read_text().splitlines()

    assert manifest["argv"] == list(sys.orig_argv)
    assert manifest["analysis_git_head"]
    assert manifest["checkpoint_git_head"] == "checkpoint-training-head"
    assert manifest["cwd"] == str(Path.cwd())
    assert manifest["exit_code"] == manifest["exit_status"] == 0
    assert manifest["status"] == "completed"
    assert len(rows) == 120
    assert progress[-1]["status"] == "completed"
