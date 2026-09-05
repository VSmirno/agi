"""Focused contracts for exp163 frozen amplitude threshold calibration."""

from __future__ import annotations

import importlib
import importlib.util

import pytest
import torch


def _experiment():
    name = "experiments.exp163_frozen_amplitude_calibration"
    assert importlib.util.find_spec(name) is not None, "exp163 experiment is missing"
    return importlib.import_module(name)


class _FixedProbe(torch.nn.Module):
    def forward(self, z, hidden, actions):
        return z.transpose(0, 1)


def test_threshold_is_shared_by_action_and_zeroes_scores_at_boundary():
    exp = _experiment()
    calibrated = exp.ThresholdCalibratedProbe(
        _FixedProbe(), torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
    )
    z = torch.tensor([[0.2, 0.21], [0.6, 0.61]])

    output = calibrated(z, torch.zeros(2, 1), torch.tensor([0, 2]))

    torch.testing.assert_close(output[:, 0], torch.tensor([0.0, 0.21]))
    torch.testing.assert_close(output[:, 1], torch.tensor([0.0, 0.61]))


def test_calibration_split_audit_rejects_any_non_heldout_leakage():
    exp = _experiment()
    audit = exp.calibration_split_audit(
        train_episode_ids={"train-a", "train-b"},
        heldout_episode_ids={"held-a", "held-b"},
    )
    assert audit["selection_source"] == "heldout_episodes_only"
    assert audit["canonical_audit_rows_used_for_selection"] == 0
    assert audit["overlap"] == 0

    with pytest.raises(ValueError, match="overlap"):
        exp.calibration_split_audit(
            train_episode_ids={"shared"}, heldout_episode_ids={"shared"}
        )


def test_defaults_lock_exact_protocol_and_observability():
    exp = _experiment()
    args = exp.build_parser().parse_args(
        ["--config", "config.yaml", "--baseline-checkpoint", "base.pt", "--out", "run"]
    )

    assert args.episodes_per_layout == 512
    assert args.collection_steps == 64
    assert args.progress_interval == 30
    assert args.nonlinear_checkpoint == exp.DEFAULT_NONLINEAR_CHECKPOINT
