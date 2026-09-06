"""Focused coverage contract for the exp171 event-context audit."""

from __future__ import annotations

import importlib
import importlib.util


def test_coverage_counts_labels_without_crossing_orientation_or_action():
    name = "experiments.exp171_event_context_audit"
    assert importlib.util.find_spec(name) is not None, "exp171 experiment is missing"
    exp = importlib.import_module(name)

    east = (1, 0, 4, 0, 0)
    north = (1, 0, 4, 0, 3)
    train_rows = [
        {"pose_key": east, "action": 2, "rgb_changed": True},
        {"pose_key": east, "action": 2, "rgb_changed": True},
        {"pose_key": east, "action": 2, "rgb_changed": False},
        {"pose_key": north, "action": 2, "rgb_changed": False},
        {"pose_key": east, "action": 3, "rgb_changed": False},
    ]
    canonical_rows = [
        {"pose_key": east, "action": 2, "rgb_changed": True},
        {"pose_key": east, "action": 2, "rgb_changed": False},
        {"pose_key": north, "action": 3, "rgb_changed": False},
    ]

    covered = exp.coverage_counts(train_rows, canonical_rows)

    assert covered[0]["train_pose_action_count"] == 3
    assert covered[0]["same_label_train_count"] == 2
    assert covered[0]["opposite_label_train_count"] == 1
    assert covered[1]["same_label_train_count"] == 1
    assert covered[1]["opposite_label_train_count"] == 2
    assert covered[2]["train_pose_action_count"] == 0
    assert covered[2]["same_label_train_count"] == 0
    assert covered[2]["opposite_label_train_count"] == 0
