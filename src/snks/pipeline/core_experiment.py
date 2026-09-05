"""Runnable, bounded smoke, fixed-encoder pilot, and transfer probes."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch

import snks
from snks.agent.core_agent import CoreAgent
from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Mode
from snks.learning.core_checkpoint import load_checkpoint, save_checkpoint
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.pipeline.core_config import CoreConfig, load_core_config
from snks.pipeline.core_controls import (
    build_dynamics_controls,
    prediction_probe,
    train_dynamics_controls,
)
from snks.pipeline.core_metrics import normalized_auc
from snks.pipeline.core_runner import EpisodeResult, model_digest, run_episode
from snks.pipeline.core_tasks import make_task
from snks.pipeline.core_transfer import TransferCondition, prepare_transfer


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CHECKPOINT = "source_checkpoint.pt"
SOURCE_FAMILIES = {"resource_acquisition", "resource_recovery"}
TRANSFER_TARGETS = (
    ("door_key", "key_consumed"),
    ("push_box", "push_2"),
)


class TraceWriter:
    """Flush one auditable episode record at a time."""

    def __init__(self, path: Path) -> None:
        self._handle = path.open("x", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


class RandomActionAgent:
    """Seeded primitive sampler used only for held-out and shared corpora."""

    def __init__(self, model: CoreWorldModel, config: CoreConfig, seed: int,
                 n_actions: int) -> None:
        self.model = model
        self.config = config
        self._rng = random.Random(seed)
        self._n_actions = n_actions
        self.last_model_calls = 0
        self.last_trace: list[dict[str, Any]] = []

    def start(self, _obs: Any, _goal: Any) -> None:
        self.last_trace = []

    def act(self, _exploration_fraction: float = 0.0) -> int:
        action = self._rng.randrange(self._n_actions)
        self.last_trace = [{"random_corpus": True, "action": action}]
        return action

    def observe(self, _transition: Any) -> None:
        return None


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("budget must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one bounded transferable-learning-core probe."
    )
    parser.add_argument("--stage", required=True, choices=("smoke", "pilot", "transfer"))
    parser.add_argument("--config", type=Path, default=Path("configs/core_pilot.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--family", default="resource_acquisition")
    parser.add_argument("--ruleset", default="default")
    parser.add_argument("--episodes", type=_positive, default=8)
    parser.add_argument("--eval-episodes", type=_positive, default=4)
    parser.add_argument("--steps", type=_positive, default=32)
    parser.add_argument("--updates", type=_positive, default=100)
    parser.add_argument("--max-seconds", type=_positive, default=180)
    parser.add_argument("--checkpoint", type=Path)
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _check_deadline(deadline: float, boundary: str) -> None:
    if time.monotonic() > deadline:
        raise TimeoutError(f"wall-clock budget exceeded at {boundary}")


def _seed_lists(seed: int, episodes: int, eval_episodes: int) -> dict[str, Any]:
    return {
        "source_train": [seed + 1_000 + index for index in range(episodes)],
        "source_validation": [seed + 2_000 + index for index in range(eval_episodes)],
        "source_audit": [seed + 3_000 + index for index in range(eval_episodes)],
        "transfer": {
            f"{family}/{ruleset}": {
                "adapt": [seed + 10_000 + index for index in range(episodes)],
                "validation": [
                    seed + 20_000 + index for index in range(eval_episodes)
                ],
            }
            for family, ruleset in TRANSFER_TARGETS
        },
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(config_path: Path) -> dict[str, str]:
    candidates = list((REPOSITORY_ROOT / "src" / "snks").glob("**/core_*.py"))
    candidates.extend(
        [
            REPOSITORY_ROOT / "experiments" / "exp138_learning_core.py",
            config_path.resolve(),
        ]
    )
    hashes: dict[str, str] = {}
    for path in sorted({item.resolve() for item in candidates if item.is_file()}):
        try:
            name = str(path.relative_to(REPOSITORY_ROOT))
        except ValueError:
            name = str(path)
        hashes[name] = _file_hash(path)
    return hashes


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    commit = completed.stdout.strip()
    return commit or "unknown"


def _preflight(config: CoreConfig, config_path: Path) -> dict[str, Any]:
    package_path = Path(snks.__file__).resolve()
    source_root = (REPOSITORY_ROOT / "src").resolve()
    if not package_path.is_relative_to(source_root):
        raise RuntimeError(
            f"snks imported from {package_path}; expected checkout source under {source_root}"
        )
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but torch.cuda.is_available() is false")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {device.index} is unavailable")
        gpu_index = torch.cuda.current_device() if device.index is None else device.index
        gpu = torch.cuda.get_device_name(gpu_index)
    else:
        gpu = None
    hashes = _source_hashes(config_path)
    snapshot = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "repository_root": str(REPOSITORY_ROOT),
        "snks_file": str(package_path),
        "hostname": socket.gethostname(),
        "requested_device": config.device,
        "actual_device": str(device),
        "gpu": gpu,
        "torch_version": torch.__version__,
        "git_commit": _git_commit(),
        "source_sha256": hashes,
        "source_snapshot_sha256": snapshot,
    }


def _schema_hash(schemas: dict[str, tuple[int, int]]) -> str:
    canonical = {name: list(shape) for name, shape in sorted(schemas.items())}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _new_source(
    config: CoreConfig,
    family: str,
    ruleset: str,
    seed: int,
    steps: int,
) -> tuple[CoreWorldModel, CoreTrainer, SequenceReplay, dict[str, tuple[int, int]]]:
    adapter, _ = make_task(family, ruleset, seed, split="train", max_steps=steps)
    try:
        observation = adapter.reset(seed)
        schemas = {
            observation.schema: (len(adapter.actions.names), len(observation.sensors))
        }
    finally:
        adapter.close()
    model = CoreWorldModel(
        CoreEncoder(config.z_dim), schemas, config.h_dim, config.ensemble_size
    ).to(config.device)
    trainer = CoreTrainer(model, config)
    replay = SequenceReplay(config.replay_capacity, config.seed)
    return model, trainer, replay, schemas


def _result_record(result: EpisodeResult) -> dict[str, Any]:
    terminal = bool(
        result.episode.transitions
        and (
            result.episode.transitions[-1].terminated
            or result.episode.transitions[-1].truncated
        )
    )
    return {
        "uid": result.episode.uid,
        "split": result.episode.split,
        "family": result.episode.family,
        "ruleset": result.episode.ruleset,
        "steps": result.steps,
        "transitions": len(result.episode.transitions),
        "terminal_or_truncated": terminal,
        "success": result.success,
        "agent_failed": result.agent_failed,
        "infrastructure_failed": result.infrastructure_failed,
        "model_calls": result.model_calls,
    }


def _run_cases(
    *,
    model: CoreWorldModel,
    config: CoreConfig,
    replay: SequenceReplay,
    family: str,
    ruleset: str,
    seeds: Sequence[int],
    split: str,
    mode: Mode,
    steps: int,
    deadline: float,
    trace: TraceWriter,
    role: str,
    random_actions: bool = False,
) -> tuple[list[EpisodeResult], list[Episode]]:
    results: list[EpisodeResult] = []
    complete: list[Episode] = []
    for seed in seeds:
        _check_deadline(deadline, f"{role} episode {seed}")
        adapter, case = make_task(family, ruleset, seed, split=split, max_steps=steps)
        try:
            episode_config = replace(config, seed=seed)
            if random_actions:
                agent: Any = RandomActionAgent(
                    model, episode_config, seed, len(adapter.actions.names)
                )
                exploration = 0.0 if mode is Mode.EVALUATE else None
            else:
                agent = CoreAgent(model, episode_config)
                exploration = 1.0 if mode is not Mode.EVALUATE else 0.0
            result = run_episode(
                adapter,
                agent,
                case,
                mode,
                replay,
                None,
                exploration=exploration,
            )
        finally:
            adapter.close()
        results.append(result)
        record = _result_record(result)
        trace.write({"role": role, "seed": seed, **record, "audit": result.audit})
        if record["terminal_or_truncated"] and not result.agent_failed:
            complete.append(result.episode)
    return results, complete


def _summarize_episodes(results: Sequence[EpisodeResult]) -> dict[str, Any]:
    count = len(results)
    successes = sum(result.success for result in results)
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count if count else None,
        "agent_failures": sum(result.agent_failed for result in results),
        "infrastructure_failures": sum(result.infrastructure_failed for result in results),
        "steps": [result.steps for result in results],
        "total_steps": sum(result.steps for result in results),
        "model_calls": [result.model_calls for result in results],
        "total_model_calls": sum(result.model_calls for result in results),
        "records": [_result_record(result) for result in results],
    }


def _train_updates(
    model: CoreWorldModel,
    trainer: CoreTrainer,
    replay: SequenceReplay,
    config: CoreConfig,
    updates: int,
    mode: Mode,
    deadline: float,
    *,
    schema: str | None,
    require_mixed: tuple[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if require_mixed is not None and updates < 2:
        raise ValueError("WEIGHTS_REPLAY requires at least two updates to consume A and B")
    metrics: list[dict[str, Any]] = []
    schema_counts: dict[str, int] = {}
    for index in range(updates):
        _check_deadline(deadline, f"update {index}")
        required_schema = None
        if require_mixed is not None and index < len(require_mixed):
            required_schema = require_mixed[index]
        attempts = 0
        while True:
            samples = replay.sample(
                config.batch_size,
                config.train_horizon,
                config.burn_in,
                config.recent_fraction,
                schema=schema,
            )
            if not samples:
                raise RuntimeError("no completed replay episode is available for training")
            sampled_schema = samples[0].transitions[0].before.schema
            if required_schema is None or sampled_schema == required_schema:
                break
            attempts += 1
            if attempts >= 256:
                raise RuntimeError(
                    f"mixed replay did not yield required schema {required_schema}"
                )
        batch = tensorize(samples, config.burn_in, torch.device(config.device))
        update_metrics = trainer.update(batch, mode)
        metrics.append({key: _jsonable(value) for key, value in update_metrics.items()})
        schema_counts[sampled_schema] = schema_counts.get(sampled_schema, 0) + 1
    if require_mixed is not None and any(schema_counts.get(name, 0) == 0 for name in require_mixed):
        raise RuntimeError("mixed replay failed to train on both source and target schemas")
    model.eval()
    return metrics, schema_counts


def _sensor_event_counts(episodes: Sequence[Episode]) -> dict[str, Any]:
    by_sensor: dict[str, int] = {}
    transitions = 0
    changed_transitions = 0
    for episode in episodes:
        for transition in episode.transitions:
            mask = transition.before.sensor_mask & transition.after.sensor_mask
            changed = (transition.before.sensors != transition.after.sensors) & mask
            transitions += 1
            changed_transitions += int(changed.any())
            for index in np.flatnonzero(changed):
                key = str(int(index))
                by_sensor[key] = by_sensor.get(key, 0) + 1
    return {
        "transitions": transitions,
        "changed_transitions": changed_transitions,
        "events_by_sensor_index": by_sensor,
        "positive_signal": bool(changed_transitions),
    }


def _save_source(
    out: Path,
    model: CoreWorldModel,
    trainer: CoreTrainer,
    replay: SequenceReplay,
    config: CoreConfig,
    schemas: dict[str, tuple[int, int]],
    family: str,
    ruleset: str,
    seeds: dict[str, Any],
    source_cost: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "format": "snks-core-source-v1",
        "config": asdict(config),
        "schemas": schemas,
        "schema_hash": _schema_hash(schemas),
        "source_family": family,
        "source_ruleset": ruleset,
        "source_schema": next(iter(schemas)),
        "seed_lists": seeds,
        "source_cost": source_cost,
    }
    path = out / SOURCE_CHECKPOINT
    digest = save_checkpoint(path, model, trainer, replay, metadata)
    return {
        "path": path.name,
        "replay_path": path.with_suffix(path.suffix + ".replay.npz").name,
        "sha256": digest,
        "metadata": metadata,
    }


def _source_stage(
    *,
    stage: str,
    out: Path,
    config: CoreConfig,
    family: str,
    ruleset: str,
    episodes: int,
    eval_episodes: int,
    steps: int,
    updates: int,
    deadline: float,
    seeds: dict[str, Any],
    trace: TraceWriter,
) -> dict[str, Any]:
    if family not in SOURCE_FAMILIES:
        raise ValueError("smoke/pilot source family must be a Crafter resource family")
    model, trainer, replay, schemas = _new_source(
        config, family, ruleset, seeds["source_train"][0], steps
    )
    train_results, complete = _run_cases(
        model=model,
        config=config,
        replay=replay,
        family=family,
        ruleset=ruleset,
        seeds=seeds["source_train"],
        split="train",
        mode=Mode.TRAIN,
        steps=steps,
        deadline=deadline,
        trace=trace,
        role="source_collection",
    )
    if len(complete) != episodes:
        raise RuntimeError(
            f"source collection completed {len(complete)}/{episodes} episodes; "
            "failed episodes remain in traces and were not replayed"
        )
    update_metrics, sampled = _train_updates(
        model,
        trainer,
        replay,
        config,
        updates,
        Mode.TRAIN,
        deadline,
        schema=next(iter(schemas)),
    )
    evaluation, _ = _run_cases(
        model=model,
        config=config,
        replay=replay,
        family=family,
        ruleset=ruleset,
        seeds=seeds["source_validation"],
        split="validation",
        mode=Mode.EVALUATE,
        steps=steps,
        deadline=deadline,
        trace=trace,
        role="source_end_to_end_validation",
    )
    source_cost = {
        "collection_steps_including_reset": sum(item.steps for item in train_results),
        "completed_collection_episodes": len(complete),
        "gradient_updates": updates,
        "sampled_schema_updates": sampled,
    }
    checkpoint = _save_source(
        out, model, trainer, replay, config, schemas, family, ruleset, seeds, source_cost
    )
    result: dict[str, Any] = {
        "stage": stage,
        "interpretation": "small development probe; not a causal or capability gate",
        "limitations": [
            "Crafter uses a controlled native local-source fixture, not natural-spawn evidence",
            "single development configuration; no confidence interval",
        ],
        "source": {
            "family": family,
            "ruleset": ruleset,
            "schemas": schemas,
            "collection": _summarize_episodes(train_results),
            "updates": update_metrics,
            "replay": replay.manifest(),
            "end_to_end_validation": _summarize_episodes(evaluation),
            "cost": source_cost,
        },
        "checkpoint": checkpoint,
        "not_run": [
            "five_seed_confirmation",
            "causal_interventions",
            "G3/G6 gate claims",
        ],
    }
    if stage == "smoke":
        return result

    _check_deadline(deadline, "fixed-encoder control construction")
    variants = build_dynamics_controls(model)
    encoder_hashes_before = {
        name: model_digest(variant.encoder) for name, variant in variants.items()
    }
    losses = train_dynamics_controls(variants, replay, config, updates, deadline)
    encoder_hashes_after = {
        name: model_digest(variant.encoder) for name, variant in variants.items()
    }
    control_eval: dict[str, Any] = {}
    for name, variant in variants.items():
        evaluated, _ = _run_cases(
            model=variant,
            config=config,
            replay=replay,
            family=family,
            ruleset=ruleset,
            seeds=seeds["source_validation"],
            split="validation",
            mode=Mode.EVALUATE,
            steps=steps,
            deadline=deadline,
            trace=trace,
            role=f"fixed_encoder_validation/{name}",
        )
        control_eval[name] = _summarize_episodes(evaluated)

    _, audit_episodes = _run_cases(
        model=model,
        config=config,
        replay=replay,
        family=family,
        ruleset=ruleset,
        seeds=seeds["source_audit"],
        split="validation",
        mode=Mode.EVALUATE,
        steps=steps,
        deadline=deadline,
        trace=trace,
        role="common_random_audit_corpus",
        random_actions=True,
    )
    if len(audit_episodes) != eval_episodes:
        raise RuntimeError("held-out random audit corpus contains an incomplete episode")
    probes: dict[str, Any] = {}
    for name, variant in variants.items():
        _check_deadline(deadline, f"prediction probe {name}")
        probes[name] = prediction_probe(variant, audit_episodes)
    event_counts = _sensor_event_counts(audit_episodes)
    result["fixed_encoder_controls"] = {
        "design": (
            "paired new validation seeds and one shared random-action held-out corpus; "
            "end-to-end success remains separate from the fixed-E prediction comparison"
        ),
        "encoder_hashes_before": encoder_hashes_before,
        "encoder_hashes_after": encoder_hashes_after,
        "encoder_hashes_unchanged": encoder_hashes_before == encoder_hashes_after,
        "losses": losses,
        "validation": control_eval,
        "common_corpus": {
            "episodes": len(audit_episodes),
            "replay_append": False,
            "sensor_events": event_counts,
        },
        "prediction": probes,
        "positive_signal_warning": (
            None
            if event_counts["positive_signal"]
            else "No observed sensor changes: sensor-error differences have no positive-event support."
        ),
    }
    return result


def _trusted_checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("source checkpoint metadata is missing")
    return dict(metadata)


def _load_source(
    checkpoint: Path, requested_config: CoreConfig
) -> tuple[CoreWorldModel, SequenceReplay, CoreConfig, dict[str, Any]]:
    metadata = _trusted_checkpoint_metadata(checkpoint)
    if metadata.get("format") != "snks-core-source-v1":
        raise ValueError("checkpoint is not an owned learning-core source checkpoint")
    config_values = dict(metadata["config"])
    source_config = replace(
        CoreConfig(**config_values),
        device=requested_config.device,
        seed=requested_config.seed,
    )
    schemas = {
        str(name): tuple(int(value) for value in shape)
        for name, shape in dict(metadata["schemas"]).items()
    }
    if metadata.get("source_schema") != "crafter-v1" or not (
        set(schemas) == {"crafter-v1"}
    ):
        raise ValueError("A->B transfer requires a Crafter-only source checkpoint")
    model = CoreWorldModel(
        CoreEncoder(source_config.z_dim),
        schemas,
        source_config.h_dim,
        source_config.ensemble_size,
    )
    trainer = CoreTrainer(model, source_config)
    replay = SequenceReplay(source_config.replay_capacity, source_config.seed)
    loaded = load_checkpoint(
        checkpoint, model, trainer, replay, str(metadata["schema_hash"])
    )
    model.to(source_config.device)
    model.eval()
    return model, replay, source_config, loaded


def _append_complete(episodes: Sequence[Episode], replay: SequenceReplay) -> None:
    for episode in episodes:
        replay.append(episode, Mode.ADAPT)


def _evaluate(
    model: CoreWorldModel,
    config: CoreConfig,
    replay: SequenceReplay,
    family: str,
    ruleset: str,
    seeds: Sequence[int],
    steps: int,
    deadline: float,
    trace: TraceWriter,
    role: str,
) -> dict[str, Any]:
    results, _ = _run_cases(
        model=model,
        config=config,
        replay=replay,
        family=family,
        ruleset=ruleset,
        seeds=seeds,
        split="validation",
        mode=Mode.EVALUATE,
        steps=steps,
        deadline=deadline,
        trace=trace,
        role=role,
    )
    return _summarize_episodes(results)


def _transfer_stage(
    *,
    checkpoint: Path,
    requested_config: CoreConfig,
    episodes: int,
    eval_episodes: int,
    steps: int,
    updates: int,
    deadline: float,
    seeds: dict[str, Any],
    trace: TraceWriter,
) -> dict[str, Any]:
    if updates < 2:
        raise ValueError("transfer requires at least two updates for mixed A/B replay")
    source, source_replay, config, metadata = _load_source(
        checkpoint.resolve(), requested_config
    )
    source_family = str(metadata["source_family"])
    source_ruleset = str(metadata["source_ruleset"])
    source_validation_seeds = seeds["source_validation"]
    result: dict[str, Any] = {
        "stage": "transfer",
        "source_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _file_hash(checkpoint.resolve()),
            "metadata": metadata,
        },
        "conditions": {},
        "limitations": [
            "single development seed; no confidence interval",
            "fixed single B layout; seeds vary push-box appearance but are not held-out maps",
            "SOURCE_CONTROL not run",
            "no causal intervention run",
            "no G3 or G6 PASS claim",
        ],
        "not_run": [
            TransferCondition.SOURCE_CONTROL.value,
            "five_seed_confirmation",
            "held_out_B_maps",
            "causal_interventions",
        ],
    }

    for target_family, target_ruleset in TRANSFER_TARGETS:
        key = f"{target_family}/{target_ruleset}"
        target_seeds = seeds["transfer"][key]
        branches: dict[str, tuple[CoreWorldModel, CoreTrainer, SequenceReplay]] = {}
        for condition in (
            TransferCondition.FRESH,
            TransferCondition.WEIGHTS,
            TransferCondition.WEIGHTS_REPLAY,
        ):
            branches[condition.value] = prepare_transfer(
                source,
                source_replay,
                condition,
                target_schema="grid-v1",
                target_shape=(5, 1),
                seed=config.seed,
                config=config,
            )

        family_result: dict[str, Any] = {}
        for name, (model, _trainer, branch_replay) in branches.items():
            family_result[name] = {
                "A_before": _evaluate(
                    model,
                    config,
                    branch_replay,
                    source_family,
                    source_ruleset,
                    source_validation_seeds,
                    steps,
                    deadline,
                    trace,
                    f"transfer/{key}/{name}/A_before",
                ),
                "B_checkpoint0": _evaluate(
                    model,
                    config,
                    branch_replay,
                    target_family,
                    target_ruleset,
                    target_seeds["validation"],
                    steps,
                    deadline,
                    trace,
                    f"transfer/{key}/{name}/B_checkpoint0",
                ),
            }

        collector_model = branches[TransferCondition.FRESH.value][0]
        collector_replay = SequenceReplay(config.replay_capacity, config.seed + 71)
        adapt_results, adapt_episodes = _run_cases(
            model=collector_model,
            config=config,
            replay=collector_replay,
            family=target_family,
            ruleset=target_ruleset,
            seeds=target_seeds["adapt"],
            split="adapt",
            mode=Mode.ADAPT,
            steps=steps,
            deadline=deadline,
            trace=trace,
            role=f"transfer/{key}/shared_random_B_adaptation",
            random_actions=True,
        )
        if len(adapt_episodes) != episodes:
            raise RuntimeError(
                f"shared B corpus for {key} completed {len(adapt_episodes)}/{episodes} episodes"
            )
        b_steps = sum(item.steps for item in adapt_results)

        for name, (model, trainer, branch_replay) in branches.items():
            _append_complete(adapt_episodes, branch_replay)
            mixed = name == TransferCondition.WEIGHTS_REPLAY.value
            update_metrics, schema_counts = _train_updates(
                model,
                trainer,
                branch_replay,
                config,
                updates,
                Mode.ADAPT,
                deadline,
                schema=None if mixed else "grid-v1",
                require_mixed=("crafter-v1", "grid-v1") if mixed else None,
            )
            b_after = _evaluate(
                model,
                config,
                branch_replay,
                target_family,
                target_ruleset,
                target_seeds["validation"],
                steps,
                deadline,
                trace,
                f"transfer/{key}/{name}/B_after",
            )
            a_after = _evaluate(
                model,
                config,
                branch_replay,
                source_family,
                source_ruleset,
                source_validation_seeds,
                steps,
                deadline,
                trace,
                f"transfer/{key}/{name}/A_after",
            )
            branch = family_result[name]
            before_score = float(branch["B_checkpoint0"]["success_rate"])
            after_score = float(b_after["success_rate"])
            curve_steps = [0.0, float(b_steps)]
            curve_scores = [before_score, after_score]
            branch.update(
                B_after=b_after,
                A_after=a_after,
                A_retention_success_delta=(
                    float(a_after["success_rate"])
                    - float(branch["A_before"]["success_rate"])
                ),
                B_curve={
                    "actual_B_steps_including_reset": curve_steps,
                    "success_rate": curve_scores,
                    "normalized_auc": normalized_auc(curve_steps, curve_scores),
                },
                adaptation={
                    "shared_completed_B_episodes": len(adapt_episodes),
                    "actual_B_steps_including_reset": b_steps,
                    "gradient_updates": updates,
                    "sampled_schema_updates": schema_counts,
                    "metrics": update_metrics,
                },
                lifetime_training_cost={
                    "source": metadata.get("source_cost", "unknown"),
                    "B_steps_including_reset": b_steps,
                    "B_gradient_updates": updates,
                },
            )
        result["conditions"][key] = family_result
    return result


def run(args: argparse.Namespace, trace: TraceWriter, manifest: dict[str, Any]) -> dict[str, Any]:
    config = load_core_config(args.config)
    _seed_everything(config.seed)
    manifest.update(
        config=asdict(config),
        config_path=str(args.config.resolve()),
        stage=args.stage,
        budgets={
            "episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
            "steps": args.steps,
            "updates": args.updates,
            "max_seconds": args.max_seconds,
        },
    )
    seeds = _seed_lists(config.seed, args.episodes, args.eval_episodes)
    manifest["seed_lists"] = seeds
    manifest["preflight"] = _preflight(config, args.config)
    deadline = time.monotonic() + args.max_seconds
    if args.stage in {"smoke", "pilot"}:
        return _source_stage(
            stage=args.stage,
            out=args.out,
            config=config,
            family=args.family,
            ruleset=args.ruleset,
            episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            steps=args.steps,
            updates=args.updates,
            deadline=deadline,
            seeds=seeds,
            trace=trace,
        )
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for transfer")
    return _transfer_stage(
        checkpoint=args.checkpoint,
        requested_config=config,
        episodes=args.episodes,
        eval_episodes=args.eval_episodes,
        steps=args.steps,
        updates=args.updates,
        deadline=deadline,
        seeds=seeds,
        trace=trace,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.out = args.out.resolve()
    manifest: dict[str, Any] = {
        "status": "running",
        "started_unix": time.time(),
        "argv": list(sys.argv if argv is None else argv),
    }
    results: dict[str, Any] = {"status": "running", "stage": args.stage}
    try:
        args.out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"--out must be a new directory: {args.out}")
    trace = TraceWriter(args.out / "traces.jsonl")
    try:
        results = run(args, trace, manifest)
        results["status"] = "completed"
        manifest["status"] = "completed"
        return_code = 0
    except Exception as error:  # The artifact must retain honest partial failure state.
        results["status"] = "error"
        results["partial_artifacts"] = ["traces.jsonl", "manifest.json"]
        results["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        manifest["status"] = "error"
        manifest["error"] = results["error"]
        return_code = 1
    finally:
        trace.close()
        manifest["finished_unix"] = time.time()
        manifest["elapsed_seconds"] = (
            manifest["finished_unix"] - manifest["started_unix"]
        )
        _write_json(args.out / "results.json", results)
        _write_json(args.out / "manifest.json", manifest)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
