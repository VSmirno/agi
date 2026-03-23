"""Configuration loading from YAML files."""

from pathlib import Path

import yaml

from snks.daf.types import DafConfig, DcamConfig, EncoderConfig, PipelineConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load PipelineConfig from a YAML file."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    daf_raw = raw.get("daf", {})
    encoder_raw = raw.get("encoder", {})
    dcam_raw = raw.get("dcam", {})
    pipeline_raw = raw.get("pipeline", {})

    config = PipelineConfig(
        daf=DafConfig(**{k: v for k, v in daf_raw.items() if k in DafConfig.__dataclass_fields__}),
        encoder=EncoderConfig(**{k: v for k, v in encoder_raw.items() if k in EncoderConfig.__dataclass_fields__}),
        dcam=DcamConfig(**{k: v for k, v in dcam_raw.items() if k in DcamConfig.__dataclass_fields__}),
        steps_per_cycle=pipeline_raw.get("steps_per_cycle", 100),
        device=raw.get("device", "auto"),
    )
    return config
