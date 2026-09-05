"""Trusted, atomic checkpoints for the small learning-core experiment."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import torch

from snks.learning.core_replay import SequenceReplay


def save_checkpoint(path, model, trainer, replay: SequenceReplay, metadata: dict) -> str:
    """Save model/trainer/RNG state and a separately hashed replay snapshot."""
    target = Path(path)
    replay_path = target.with_suffix(target.suffix + ".replay.npz")
    if target.exists() or replay_path.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    replay.save(replay_path)
    payload = {
        "model": model.state_dict(),
        "trainer": trainer.state_dict() if hasattr(trainer, "state_dict") else trainer.optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "metadata": dict(metadata),
        "replay_path": replay_path.name,
        "replay_hash": replay.manifest()["hash"],
    }
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _file_hash(target)


def load_checkpoint(path, model, trainer, replay: SequenceReplay, expected_schema_hash: str) -> dict:
    """Validate schema and replay before changing the supplied live objects."""
    target = Path(path)
    payload = torch.load(target, map_location="cpu", weights_only=True)
    metadata = dict(payload["metadata"])
    if metadata.get("schema_hash") != expected_schema_hash:
        raise ValueError("checkpoint schema hash does not match target")
    snapshot = SequenceReplay.load(target.with_name(payload["replay_path"]))
    if snapshot.manifest()["hash"] != payload["replay_hash"]:
        raise ValueError("checkpoint replay hash does not match snapshot")
    model.load_state_dict(payload["model"])
    if hasattr(trainer, "load_state_dict"):
        trainer.load_state_dict(payload["trainer"])
    else:
        trainer.optimizer.load_state_dict(payload["trainer"])
    torch.set_rng_state(payload["rng"])
    if payload.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    replay._restore_from(snapshot)
    return metadata


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
