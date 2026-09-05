"""Data crossing the learned agent boundary; diagnostics stay with the runner."""

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Mode(Enum):
    TRAIN = "train"
    ADAPT = "adapt"
    EVALUATE = "evaluate"


@dataclass(frozen=True, slots=True)
class Observation:
    rgb: np.ndarray
    sensors: np.ndarray
    sensor_mask: np.ndarray
    schema: str
    step: int

    def __post_init__(self) -> None:
        rgb = np.array(self.rgb, dtype=np.uint8, copy=True)
        sensors = np.array(self.sensors, dtype=np.float32, copy=True)
        mask = np.array(self.sensor_mask, dtype=bool, copy=True)
        if rgb.shape != (3, 64, 64) or sensors.ndim != 1 or mask.shape != sensors.shape:
            raise ValueError("expected CHW 64x64 RGB and matching sensor vectors")
        if not np.isfinite(sensors[mask]).all():
            raise ValueError("observed sensors must be finite")
        sensors[~mask] = 0.0
        for name, value in (("rgb", rgb), ("sensors", sensors), ("sensor_mask", mask)):
            value.flags.writeable = False
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    schema: str
    names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GoalSpec:
    image: Observation | None
    ranges: dict[int, tuple[float, float]]


@dataclass(frozen=True, slots=True)
class Transition:
    before: Observation
    action: int
    after: Observation
    terminated: bool
    truncated: bool


@dataclass(frozen=True, slots=True)
class Episode:
    uid: str
    split: str
    family: str
    ruleset: str
    transitions: tuple[Transition, ...]
