from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from experiments.exp143_temporal_proximity import TemporalProbe
from experiments.exp147_rollout_localization import build_parser, main
from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder


def _write_checkpoint(path: Path) -> None:
    torch.manual_seed(147)
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


def test_checkpoint_diagnostic_writes_teacher_forced_and_rollout_evidence(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    output = tmp_path / "diagnostic"
    _write_checkpoint(checkpoint)

    assert main(
        [
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(output),
            "--progress-interval",
            "1",
        ]
    ) == 0

    results = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (output / "diagnostic_rows.jsonl").read_text().splitlines()
    ]
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]

    assert manifest["status"] == "completed"
    assert manifest["exit_code"] == 0
    assert manifest["checkpoint_git_head"] == "checkpoint-training-head"
    assert manifest["analysis_git_head"]
    assert len([row for row in rows if row["row_type"] == "teacher_forced"]) == 15
    assert len([row for row in rows if row["row_type"] == "autoregressive"]) == 3
    blocked = next(
        row
        for row in rows
        if row["row_type"] == "teacher_forced"
        and row["step"] == 0
        and row["action"] == 2
    )
    assert blocked["action_name"] == "forward"
    assert blocked["rgb_changed"] is False
    assert blocked["persistence_vs_actual_next_z_mse"] == pytest.approx(0.0)
    assert [row["canonical_action"] for row in results["teacher_forced"]] == [3, 2, 3]
    assert all(1 <= row["prediction_error_rank_among_actions"] <= 5
               for row in results["teacher_forced"])
    assert [row["depth"] for row in results["autoregressive"]] == [1, 2, 3]
    assert results["classification"]["label"] in {
        "one_step_failure_evidence",
        "autoregressive_compounding_evidence",
        "mixed",
        "inconclusive",
    }
    assert set(results["classification"]["numeric_predicates"]) >= {
        "one_step_changed_not_better_than_persistence",
        "blocked_forward_material_departure",
        "autoregressive_material_growth",
    }
    assert progress[-1]["status"] == "completed"


def test_bad_checkpoint_metadata_writes_failure_manifest(tmp_path):
    checkpoint = tmp_path / "bad.pt"
    output = tmp_path / "diagnostic"
    torch.save({"format_version": 1}, checkpoint)

    with pytest.raises(ValueError, match="checkpoint missing required metadata"):
        main(["--checkpoint", str(checkpoint), "--out", str(output)])

    manifest = json.loads((output / "manifest.json").read_text())
    progress = [
        json.loads(line)
        for line in (output / "progress.jsonl").read_text().splitlines()
    ]
    assert manifest["status"] == "failed"
    assert manifest["exit_code"] == 1
    assert manifest["analysis_git_head"]
    assert manifest["checkpoint_git_head"] is None
    assert progress[-1]["status"] == "failed"


def test_cli_requires_checkpoint_and_bounds_progress_interval():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--out", "run"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--checkpoint", "checkpoint.pt", "--out", "run", "--progress-interval", "31"]
        )
