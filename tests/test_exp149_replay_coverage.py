from __future__ import annotations

import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "experiments"))

from snks.env.core_types import Observation
from experiments.exp149_replay_coverage import (
    _observation_changes,
    _validate_terminal_counts,
)


def _observation(value: int, sensor: float, mask: bool) -> Observation:
    return Observation(
        np.full((3, 64, 64), value, dtype=np.uint8),
        np.array([sensor], dtype=np.float32),
        np.array([mask], dtype=bool),
        "grid-v1",
        0,
    )


def test_observation_changes_reports_rgb_sensor_and_mask_independently():
    before = _observation(1, 0.0, False)
    after = _observation(2, 1.0, True)

    assert _observation_changes(before, after) == {
        "rgb": True,
        "sensors": True,
        "sensor_mask": True,
        "exact": True,
    }


def test_scientific_terminal_counts_validate_total_and_fit_cutoff():
    assert _validate_terminal_counts(
        {"east_row2": 2, "west_row3": 7, "south_col4": 2, "north_col5": 7},
        {"east_row2": 2, "west_row3": 4, "south_col4": 2, "north_col5": 5},
        episodes_per_layout=512,
    )
