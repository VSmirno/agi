"""Checkpoint-only localization of one-step versus rollout error at the exp146 fork."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import torch

from experiments.exp143_temporal_proximity import TemporalProbe
from experiments.exp145_physics_transfer import (
    TARGET_LAYOUTS,
    _adapter,
    _goal_observation,
)
from experiments.exp146_temporal_mpc_physics import ProgressJournal
from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


LAYOUT_NAME = "east_row4_left"
SEED = 20000
REAL_PREFIX = (0, 3, 2, 3, 2)
CANONICAL = (3, 2, 3)
HORIZON = 3
SCHEMA = "grid-v1"
ACTION_COUNT = 5
MATERIAL_FRACTION_OF_MOVEMENT = 0.10
MATERIAL_GROWTH_RATIO = 1.50
NUMERIC_MSE_FLOOR = 1e-8


def _progress_interval(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 30:
        raise argparse.ArgumentTypeError("progress interval must be in [1, 30]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-interval", type=_progress_interval, default=30)
    return parser


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"checkpoint {field} must be a positive integer")
    return value


def _required_mapping(container: Mapping[str, Any], key: str, field: str):
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {field} must be a mapping")
    return value


def _validate_state_dict(
    name: str,
    state_dict: Any,
    expected: Mapping[str, torch.Tensor],
) -> None:
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint {name} must be a state_dict mapping")
    missing = sorted(set(expected) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"checkpoint {name} keys do not match rebuilt module: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for key, wanted in expected.items():
        loaded = state_dict[key]
        if not torch.is_tensor(loaded):
            raise ValueError(f"checkpoint {name}.{key} is not a tensor")
        if loaded.shape != wanted.shape:
            raise ValueError(
                f"checkpoint {name}.{key} shape {tuple(loaded.shape)} does not "
                f"match metadata-derived shape {tuple(wanted.shape)}"
            )


def _load_checkpoint(path: Path):
    """Load the trusted local state-dict payload through PyTorch's safe loader."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not safely load checkpoint: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    required = {
        "format_version",
        "git_head",
        "config",
        "modules",
        "model_state_dict",
        "ordered_probe_state_dict",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint missing required metadata: {missing}")
    if payload["format_version"] != 1:
        raise ValueError(
            f"unsupported checkpoint format_version: {payload['format_version']!r}"
        )
    checkpoint_git_head = payload["git_head"]
    if not isinstance(checkpoint_git_head, str) or not checkpoint_git_head:
        raise ValueError("checkpoint git_head must be a non-empty string")

    config = _required_mapping(payload, "config", "config")
    modules = _required_mapping(payload, "modules", "modules")
    model_metadata = _required_mapping(modules, "model", "modules.model")
    probe_metadata = _required_mapping(modules, "probe", "modules.probe")
    schemas_metadata = _required_mapping(
        model_metadata, "schemas", "modules.model.schemas"
    )
    schemas: dict[str, tuple[int, int]] = {}
    for name, shape in schemas_metadata.items():
        if (
            not isinstance(name, str)
            or not isinstance(shape, (list, tuple))
            or len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in shape)
        ):
            raise ValueError(f"checkpoint schema {name!r} has invalid dimensions")
        schemas[name] = (shape[0], shape[1])
    if schemas.get(SCHEMA) != (ACTION_COUNT, 1):
        raise ValueError(
            "checkpoint grid-v1 schema must have exactly 5 actions and 1 sensor"
        )

    z_dim = _positive_int(model_metadata.get("z_dim"), "modules.model.z_dim")
    h_dim = _positive_int(model_metadata.get("h_dim"), "modules.model.h_dim")
    heads = _positive_int(
        model_metadata.get("ensemble_size"), "modules.model.ensemble_size"
    )
    probe_z_dim = _positive_int(
        probe_metadata.get("z_dim"), "modules.probe.z_dim"
    )
    probe_width = _positive_int(
        probe_metadata.get("width"), "modules.probe.width"
    )
    if probe_z_dim != z_dim:
        raise ValueError("checkpoint model and ordered probe z_dim metadata disagree")
    for field, expected in (
        ("z_dim", z_dim),
        ("h_dim", h_dim),
        ("ensemble_size", heads),
    ):
        if config.get(field) != expected:
            raise ValueError(f"checkpoint config.{field} disagrees with module metadata")
    flags = {}
    for field in ("normalize_sensor_condition", "predict_sensor_delta"):
        value = model_metadata.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"checkpoint modules.model.{field} must be boolean")
        if config.get(field) is not value:
            raise ValueError(f"checkpoint config.{field} disagrees with module metadata")
        flags[field] = value
    device_name = config.get("device")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("checkpoint config.device must be a non-empty string")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("checkpoint requests CUDA but CUDA is unavailable")

    model = CoreWorldModel(
        CoreEncoder(z_dim),
        schemas,
        h_dim,
        heads,
        normalize_sensor_condition=flags["normalize_sensor_condition"],
        predict_sensor_delta=flags["predict_sensor_delta"],
    )
    ordered = TemporalProbe(probe_z_dim, width=probe_width)
    _validate_state_dict("model_state_dict", payload["model_state_dict"], model.state_dict())
    _validate_state_dict(
        "ordered_probe_state_dict",
        payload["ordered_probe_state_dict"],
        ordered.state_dict(),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    ordered.load_state_dict(payload["ordered_probe_state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    ordered.to(device).eval().requires_grad_(False)
    return model, ordered, checkpoint_git_head, {
        "device": str(device),
        "schemas": schemas,
        "z_dim": z_dim,
        "h_dim": h_dim,
        "ensemble_size": heads,
        "probe_width": probe_width,
        **flags,
    }


def _fresh_real_fork(history: tuple[int, ...], action: int):
    layout, _push_one, _push_two = TARGET_LAYOUTS[LAYOUT_NAME]
    adapter = _adapter(layout, 1, SEED, 32)
    try:
        observation = adapter.reset(SEED)
        for previous in history:
            transition = adapter.step(previous)
            if transition.terminated or transition.truncated:
                raise RuntimeError("real history unexpectedly ended before the fork")
            observation = transition.after
        before = observation
        transition = adapter.step(action)
        return before, transition.after, adapter.diagnostic_snapshot()
    finally:
        adapter.close()


def _costs(z: torch.Tensor, goal_z: torch.Tensor, ordered: TemporalProbe):
    horizon = torch.ones(len(z), device=z.device)
    ordered_cost = -ordered(z, goal_z.expand(len(z), -1), horizon)
    raw_cost = (z - goal_z.expand(len(z), -1)).square().mean(-1)
    return float(ordered_cost[0]), float(raw_cost[0])


def _teacher_forced_next(predicted, actual) -> LatentState:
    return LatentState(
        actual.z,
        actual.sensors,
        actual.sensor_mask,
        predicted.next_state.hidden.detach(),
        actual.schema,
    )


def _classification(teacher_rows, rollout_rows):
    canonical_rows = [
        row for row in teacher_rows if row["action"] == row["canonical_action"]
    ]
    changed = [row for row in canonical_rows if row["rgb_changed"]]
    movement = [
        row["persistence_vs_actual_next_z_mse"]
        for row in changed
        if row["persistence_vs_actual_next_z_mse"] > 0.0
    ]
    if not movement:
        raise RuntimeError("canonical protocol contained no changed latent transition")
    movement_scale = float(statistics.median(movement))
    material_threshold = max(
        NUMERIC_MSE_FLOOR,
        MATERIAL_FRACTION_OF_MOVEMENT * movement_scale,
    )
    not_better = [
        row["step"]
        for row in changed
        if row["predicted_vs_actual_next_z_mse"]
        >= row["persistence_vs_actual_next_z_mse"]
    ]
    blocked = next(
        row for row in teacher_rows if row["step"] == 0 and row["action"] == 2
    )
    if blocked["rgb_changed"]:
        raise RuntimeError("protocol error: step-0 forward action was not blocked")
    blocked_departure = (
        blocked["predicted_vs_actual_next_z_mse"] > material_threshold
    )
    first_error = rollout_rows[0]["predicted_vs_actual_z_mse"]
    final_error = rollout_rows[-1]["predicted_vs_actual_z_mse"]
    growth_difference = final_error - first_error
    growth_ratio = final_error / max(first_error, NUMERIC_MSE_FLOOR)
    material_growth = bool(
        growth_difference > material_threshold
        and growth_ratio >= MATERIAL_GROWTH_RATIO
    )
    all_changed_better = bool(changed) and not not_better
    one_step_evidence = bool(not_better or blocked_departure)
    compounding_evidence = bool(all_changed_better and material_growth)
    if one_step_evidence and compounding_evidence:
        label = "mixed"
    elif one_step_evidence:
        label = "one_step_failure_evidence"
    elif compounding_evidence:
        label = "autoregressive_compounding_evidence"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "one_step_failure_evidence": one_step_evidence,
        "autoregressive_compounding_evidence": compounding_evidence,
        "numeric_predicates": {
            "material_mse_threshold": material_threshold,
            "movement_scale_median_changed_canonical_persistence_mse": movement_scale,
            "one_step_changed_not_better_than_persistence": {
                "predicate": "prediction_mse >= persistence_mse on an RGB-changed canonical transition",
                "matching_steps": not_better,
                "value": bool(not_better),
            },
            "blocked_forward_material_departure": {
                "predicate": "step_0_forward_prediction_mse > material_mse_threshold while actual RGB is unchanged",
                "prediction_mse": blocked["predicted_vs_actual_next_z_mse"],
                "value": blocked_departure,
            },
            "canonical_changed_transitions_all_beat_persistence": {
                "predicate": "prediction_mse < persistence_mse for every RGB-changed canonical transition",
                "value": all_changed_better,
            },
            "autoregressive_material_growth": {
                "predicate": (
                    "H3_mse - H1_mse > material_mse_threshold and "
                    "H3_mse / max(H1_mse, 1e-8) >= 1.5"
                ),
                "difference": growth_difference,
                "ratio": growth_ratio,
                "value": material_growth,
            },
        },
        "scope": (
            "Conservative classification of one deterministic three-step fork; "
            "three rollout points do not establish a general growth law."
        ),
    }


@torch.inference_mode()
def _diagnose(model, ordered, journal: ProgressJournal, rows_path: Path):
    layout, _push_one, _push_two = TARGET_LAYOUTS[LAYOUT_NAME]
    goal_z = model.initial(_goal_observation(layout, 1, SEED, 32)).z

    root_adapter = _adapter(layout, 1, SEED, 32)
    try:
        root = model.initial(root_adapter.reset(SEED))
        for action in REAL_PREFIX:
            transition = root_adapter.step(action)
            prediction = model.step(
                root, torch.tensor([action], device=root.z.device, dtype=torch.long)
            )
            if transition.terminated or transition.truncated:
                raise RuntimeError("real prefix unexpectedly ended the episode")
            root = _teacher_forced_next(prediction, model.initial(transition.after))
        root_diagnostic = root_adapter.diagnostic_snapshot()
    finally:
        root_adapter.close()

    writer = core.TraceWriter(rows_path)
    all_teacher_rows = []
    canonical_summaries = []
    actual_endpoints = []
    state = root
    try:
        journal.update("teacher_forced", 0, HORIZON * ACTION_COUNT)
        completed = 0
        for step, canonical_action in enumerate(CANONICAL):
            history = (*REAL_PREFIX, *CANONICAL[:step])
            step_rows = []
            canonical_actual = None
            canonical_prediction = None
            for action in range(ACTION_COUNT):
                before, after, diagnostic = _fresh_real_fork(history, action)
                prediction = model.step(
                    state,
                    torch.tensor([action], device=state.z.device, dtype=torch.long),
                )
                actual = model.initial(after)
                predicted_ordered, predicted_raw = _costs(
                    prediction.next_state.z, goal_z, ordered
                )
                actual_ordered, actual_raw = _costs(actual.z, goal_z, ordered)
                row = {
                    "row_type": "teacher_forced",
                    "step": step,
                    "real_history": list(history),
                    "canonical_action": canonical_action,
                    "action": action,
                    "action_name": GRID_ACTIONS[action],
                    "actual_diagnostic": diagnostic,
                    "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                    "predicted_vs_actual_next_z_mse": float(
                        (prediction.next_state.z - actual.z).square().mean()
                    ),
                    "persistence_vs_actual_next_z_mse": float(
                        (state.z - actual.z).square().mean()
                    ),
                    "predicted_ordered_goal_cost": predicted_ordered,
                    "actual_ordered_goal_cost": actual_ordered,
                    "predicted_raw_goal_cost": predicted_raw,
                    "actual_raw_goal_cost": actual_raw,
                }
                step_rows.append(row)
                completed += 1
                journal.update(
                    "teacher_forced",
                    completed,
                    HORIZON * ACTION_COUNT,
                    step=step,
                    action=action,
                    action_name=GRID_ACTIONS[action],
                )
                if action == canonical_action:
                    canonical_actual = actual
                    canonical_prediction = prediction
            ordered_errors = sorted(
                step_rows,
                key=lambda row: (row["predicted_vs_actual_next_z_mse"], row["action"]),
            )
            for rank, row in enumerate(ordered_errors, start=1):
                row["prediction_error_rank_among_actions"] = rank
            for row in step_rows:
                writer.write(row)
            all_teacher_rows.extend(step_rows)
            canonical_row = next(
                row for row in step_rows if row["action"] == canonical_action
            )
            prediction_mse = canonical_row["predicted_vs_actual_next_z_mse"]
            persistence_mse = canonical_row["persistence_vs_actual_next_z_mse"]
            canonical_summaries.append(
                {
                    "step": step,
                    "canonical_action": canonical_action,
                    "canonical_action_name": GRID_ACTIONS[canonical_action],
                    "rgb_changed": canonical_row["rgb_changed"],
                    "one_step_mse": prediction_mse,
                    "persistence_mse": persistence_mse,
                    "prediction_minus_persistence_mse": (
                        prediction_mse - persistence_mse
                    ),
                    "prediction_to_persistence_ratio": (
                        prediction_mse / persistence_mse
                        if persistence_mse > 0.0
                        else None
                    ),
                    "prediction_error_rank_among_actions": canonical_row[
                        "prediction_error_rank_among_actions"
                    ],
                }
            )
            if canonical_actual is None or canonical_prediction is None:
                raise RuntimeError("canonical action was not evaluated")
            actual_endpoints.append(canonical_actual)
            state = _teacher_forced_next(canonical_prediction, canonical_actual)

        journal.update("autoregressive", 0, HORIZON)
        rollout_rows = []
        predicted_state = root
        for depth, (action, actual) in enumerate(
            zip(CANONICAL, actual_endpoints, strict=True), start=1
        ):
            prediction = model.step(
                predicted_state,
                torch.tensor([action], device=root.z.device, dtype=torch.long),
            )
            predicted_state = prediction.next_state
            predicted_ordered, predicted_raw = _costs(
                predicted_state.z, goal_z, ordered
            )
            actual_ordered, actual_raw = _costs(actual.z, goal_z, ordered)
            row = {
                "row_type": "autoregressive",
                "depth": depth,
                "actions": list(CANONICAL[:depth]),
                "action": action,
                "action_name": GRID_ACTIONS[action],
                "predicted_vs_actual_z_mse": float(
                    (predicted_state.z - actual.z).square().mean()
                ),
                "predicted_ordered_goal_cost": predicted_ordered,
                "actual_ordered_goal_cost": actual_ordered,
                "predicted_raw_goal_cost": predicted_raw,
                "actual_raw_goal_cost": actual_raw,
            }
            rollout_rows.append(row)
            writer.write(row)
            journal.update(
                "autoregressive", depth, HORIZON, depth=depth, action=action
            )
    finally:
        writer.close()

    classification = _classification(all_teacher_rows, rollout_rows)
    return {
        "status": "completed",
        "claim": "bounded checkpoint-only rollout-error localization",
        "protocol": {
            "layout": LAYOUT_NAME,
            "push_distance": 1,
            "seed": SEED,
            "real_prefix": list(REAL_PREFIX),
            "canonical_continuation": list(CANONICAL),
            "horizon": HORIZON,
            "teacher_forcing": "real z/sensors/mask with carried predicted hidden",
            "fresh_environment_replay_per_action_fork": True,
            "reward_or_success_used_for_fitting": False,
            "push_2_run": False,
        },
        "late_fork_root_diagnostic": root_diagnostic,
        "teacher_forced": canonical_summaries,
        "autoregressive": rollout_rows,
        "error_growth": {
            "h1_to_h3_mse_difference": (
                rollout_rows[-1]["predicted_vs_actual_z_mse"]
                - rollout_rows[0]["predicted_vs_actual_z_mse"]
            ),
            "h1_to_h3_mse_ratio": (
                rollout_rows[-1]["predicted_vs_actual_z_mse"]
                / max(
                    rollout_rows[0]["predicted_vs_actual_z_mse"],
                    NUMERIC_MSE_FLOOR,
                )
            ),
            "interpretation_limit": (
                "Three fixed depths describe this fork only; they do not establish "
                "a general error-growth trend."
            ),
        },
        "classification": classification,
        "artifacts": {"rows": rows_path.name},
    }


def _exit_code(error: BaseException) -> int:
    if isinstance(error, KeyboardInterrupt):
        return 130
    if isinstance(error, SystemExit) and isinstance(error.code, int):
        return error.code
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "argv": (
            list(sys.orig_argv)
            if argv is None
            else [sys.executable, str(Path(__file__)), *argv]
        ),
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "checkpoint_git_head": None,
        "checkpoint": str(args.checkpoint),
        "arguments": core._jsonable(vars(args)),
    }
    with ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 1, operation="safe_checkpoint_load")
            model, ordered, checkpoint_head, metadata = _load_checkpoint(
                args.checkpoint
            )
            manifest["checkpoint_git_head"] = checkpoint_head
            manifest["checkpoint_metadata"] = metadata
            journal.update("initialize", 1, 1, device=metadata["device"])
            results = _diagnose(
                model,
                ordered,
                journal,
                args.out / "diagnostic_rows.jsonl",
            )
            results["checkpoint"] = {
                "path": str(args.checkpoint),
                "git_head": checkpoint_head,
                "metadata": metadata,
                "load_policy": "torch.load(weights_only=True, map_location='cpu')",
            }
            journal.update("artifacts", 0, 2, operation="write_results")
            core._write_json(args.out / "results.json", results)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            core._write_json(
                args.out / "manifest.json",
                {**manifest, "exit_code": 0, "status": "completed"},
            )
            journal.update("artifacts", 2, 2, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            core._write_json(
                args.out / "manifest.json",
                {
                    **manifest,
                    "exit_code": _exit_code(error),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
