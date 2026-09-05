from __future__ import annotations

import numpy as np

from snks.env.core_types import Observation
from experiments.exp149_replay_coverage import _observation_changes


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
