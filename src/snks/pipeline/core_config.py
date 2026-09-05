"""Small, explicit profile for the first learning-core experiments."""

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CoreConfig:
    profile: str = "development"
    device: str = "cpu"
    seed: int = 0
    z_dim: int = 64
    h_dim: int = 32
    ensemble_size: int = 3
    batch_size: int = 8
    burn_in: int = 1
    train_horizon: int = 3
    planner_horizon: int = 3
    beam_width: int = 4
    max_model_calls: int = 128
    replay_capacity: int = 64
    recent_fraction: float = 0.5
    learning_rate: float = 0.001
    sigreg_weight: float = 0.1
    sensor_weight: float = 1.0
    termination_weight: float = 1.0
    exploration_fraction: float = 0.2

    def __post_init__(self) -> None:
        positive = (self.z_dim, self.h_dim, self.ensemble_size, self.batch_size,
                    self.train_horizon, self.planner_horizon, self.beam_width,
                    self.max_model_calls, self.replay_capacity, self.learning_rate)
        if any(value <= 0 for value in positive) or self.burn_in < 0:
            raise ValueError("model sizes and budgets must be positive; burn_in >= 0")
        if not 0 <= self.recent_fraction <= 1 or not 0 <= self.exploration_fraction <= 1:
            raise ValueError("sampling/exploration fractions must be in [0, 1]")


def load_core_config(path: Path) -> CoreConfig:
    """Read one YAML profile, rejecting misspelled configuration fields."""
    with Path(path).open() as handle:
        values = yaml.safe_load(handle) or {}
    unknown = set(values) - {field.name for field in fields(CoreConfig)}
    if unknown:
        raise ValueError(f"unknown core configuration fields: {sorted(unknown)}")
    return CoreConfig(**values)
