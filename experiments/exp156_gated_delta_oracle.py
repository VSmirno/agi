"""Pre-gate raw-delta oracle audit for frozen exp153 and exp154 checkpoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import time
from typing import Any

import numpy as np
import torch

from experiments import exp148_source_target_one_step as one_step
from experiments import exp153_change_gated_dynamics as gated
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp155_oracle_residual_gate as oracle
from experiments.exp147_rollout_localization import _exit_code
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


CHECKPOINTS = {
    "exp153": {
        "path": Path(
            "output_to_user/core/exp153-change-gated-dynamics-001/"
            "gated_checkpoint.pt"
        ),
        "git_head": "49877e40f45156da2971b1802d748c551b2abc56",
        "format_version": 3,
        "protocol": gated.PROTOCOL,
    },
    "exp154": {
        "path": Path(
            "output_to_user/core/exp154-auxiliary-change-gate-001/"
            "auxiliary_checkpoint.pt"
        ),
        "git_head": "9f896a336f9186ea2129306afdda06313b261908",
        "format_version": 4,
        "protocol": auxiliary.PROTOCOL,
    },
}
VARIANTS = ("native", "raw_delta_per_member_oracle")


def native_prediction_and_raw_deltas(model, state, actions):
    """Return native gated output and latent-head directions before sigmoid gates."""

    prediction = model.step(state, actions)
    raw_deltas = torch.stack(
        [head(prediction.next_state.hidden) for head in model.latent_heads]
    )
    return prediction, raw_deltas


def raw_delta_upper_bound_gate(source_summary: Mapping, exact_protocol: bool) -> bool:
    ratio = source_summary["medians"]["free_forward_prediction_persistence_ratio"]
    return bool(
        exact_protocol
        and source_summary["contact_failure_layouts"] == 0
        and source_summary["blocked_noop_failure_layouts"] == 0
        and ratio is not None
        and math.isfinite(ratio)
        and ratio < 1.0
    )


def load_auxiliary_checkpoint(path: Path):
    """Safely reconstruct the explicitly tagged exp154 v4 state-dict payload."""

    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ValueError(f"could not safely load auxiliary checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("format_version") != 4:
        raise ValueError("auxiliary checkpoint requires format_version 4")
    if payload.get("latent_parameterization") != "gated_residual_zero_init":
        raise ValueError("checkpoint requires gated_residual_zero_init parameterization")
    if payload.get("event_supervision") is not True:
        raise ValueError("auxiliary checkpoint requires event_supervision=true")
    if payload.get("event_balanced") is not False:
        raise ValueError("auxiliary checkpoint requires event_balanced=false")
    head = payload.get("git_head")
    if not isinstance(head, str) or not head:
        raise ValueError("checkpoint git_head must be a non-empty string")
    config = auxiliary.checkpoint_io._required_mapping(payload, "config", "config")
    modules = auxiliary.checkpoint_io._required_mapping(payload, "modules", "modules")
    meta = auxiliary.checkpoint_io._required_mapping(
        modules, "model", "modules.model"
    )
    probe_meta = auxiliary.checkpoint_io._required_mapping(
        modules, "probe", "modules.probe"
    )
    if meta.get("class") != auxiliary.AUXILIARY_CLASS:
        raise ValueError("checkpoint must identify the exp154 auxiliary class")
    if meta.get("gate") != gated.GATE_DEFINITION:
        raise ValueError("checkpoint gate definition does not match exp153/154")
    if meta.get("schemas") != {"grid-v1": [5, 1]}:
        raise ValueError("checkpoint requires grid-v1 with 5 actions and 1 sensor")
    dimensions = {}
    for field in ("z_dim", "h_dim", "ensemble_size"):
        dimensions[field] = auxiliary.checkpoint_io._positive_int(
            meta.get(field), field
        )
        if config.get(field) != dimensions[field]:
            raise ValueError(f"checkpoint config.{field} disagrees with metadata")
    flags = {}
    for field in ("normalize_sensor_condition", "predict_sensor_delta"):
        flags[field] = meta.get(field)
        if not isinstance(flags[field], bool) or config.get(field) is not flags[field]:
            raise ValueError(f"checkpoint {field} metadata must agree and be boolean")
    if probe_meta.get("z_dim") != dimensions["z_dim"]:
        raise ValueError("checkpoint probe/model z_dim disagree")
    width = auxiliary.checkpoint_io._positive_int(
        probe_meta.get("width"), "probe.width"
    )
    device_name = config.get("device")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("checkpoint device must be a non-empty string")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("checkpoint requests CUDA but CUDA is unavailable")
    model = auxiliary.AuxiliaryChangeGatedWorldModel(
        CoreEncoder(dimensions["z_dim"]),
        {"grid-v1": (5, 1)},
        dimensions["h_dim"],
        dimensions["ensemble_size"],
        **flags,
    )
    probe = auxiliary.temporal.TemporalProbe(dimensions["z_dim"], width=width)
    for name, module in (
        ("model_state_dict", model),
        ("ordered_probe_state_dict", probe),
    ):
        state = payload.get(name)
        auxiliary.checkpoint_io._validate_state_dict(name, state, module.state_dict())
        module.load_state_dict(state, strict=True)
        module.to(device).eval().requires_grad_(False)
    auxiliary.checkpoint_io._validate_state_dict(
        "shuffled_probe_state_dict",
        payload.get("shuffled_probe_state_dict"),
        probe.state_dict(),
    )
    gate_auxiliary = payload.get("gate_auxiliary")
    if not isinstance(gate_auxiliary, Mapping):
        raise ValueError("checkpoint gate_auxiliary must be a mapping")
    return model, probe, head, {
        "device": str(device),
        **dimensions,
        **flags,
        "probe_width": width,
        "latent_parameterization": payload["latent_parameterization"],
        "event_supervision": True,
        "event_balanced": False,
        "gate_auxiliary": dict(gate_auxiliary),
        "config": dict(config),
        "budgets": dict(payload.get("budgets", {})),
        "load_policy": "torch.load(weights_only=True, map_location='cpu')",
    }


def _variant_row(common: dict, prediction_mse: float) -> dict:
    persistence_mse = common["persistence_vs_actual_next_z_mse"]
    return {
        **common,
        "predicted_vs_actual_next_z_mse": prediction_mse,
        "prediction_to_persistence_ratio": (
            prediction_mse / persistence_mse if persistence_mse > 0.0 else None
        ),
    }


@torch.inference_mode()
def diagnose_checkpoint(label: str, model, journal, rows_path: Path):
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * one_step.HORIZON * 5
    completed = 0
    layouts = {
        variant: {split: [] for split in one_step.SPLITS} for variant in VARIANTS
    }
    writer = core.TraceWriter(rows_path)
    journal.update(f"{label}_raw_delta_oracle", 0, total)
    try:
        for split in one_step.SPLITS:
            for layout_name, spec in specs[split].items():
                layout, actions = spec[:2]
                prefix, continuation = one_step._validate_protocol(
                    split, layout_name, layout, actions, one_step.SEED
                )
                state, prefix_diagnostic = one_step._replay_prefix(
                    model, layout, prefix, one_step.SEED
                )
                rows = {variant: [] for variant in VARIANTS}
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, diagnostic = one_step._fresh_real_fork(
                            layout, history, action, one_step.SEED
                        )
                        action_tensor = torch.tensor(
                            [action], device=state.z.device, dtype=torch.long
                        )
                        prediction, raw_deltas = native_prediction_and_raw_deltas(
                            model, state, action_tensor
                        )
                        actual = model.initial(after)
                        target = actual.z[0] - state.z[0]
                        member_gates, solver_mse = oracle.solve_per_member_scalar_gates(
                            raw_deltas[:, 0], target
                        )
                        oracle_mse = float(
                            (
                                (member_gates[:, None] * raw_deltas[:, 0]).mean(0)
                                - target
                            )
                            .square()
                            .mean()
                        )
                        if not math.isclose(
                            oracle_mse, solver_mse, rel_tol=1e-5, abs_tol=1e-8
                        ):
                            raise AssertionError("raw-delta oracle solver/report mismatch")
                        native_mse = float(
                            (prediction.next_state.z[0] - actual.z[0]).square().mean()
                        )
                        common = {
                            "checkpoint": label,
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "real_history": list(history),
                            "canonical_action": canonical_action,
                            "action": action,
                            "action_name": GRID_ACTIONS[action],
                            "actual_diagnostic": diagnostic,
                            "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                            "persistence_vs_actual_next_z_mse": float(target.square().mean()),
                        }
                        rows["native"].append(_variant_row(common, native_mse))
                        rows["raw_delta_per_member_oracle"].append(
                            _variant_row(common, oracle_mse)
                        )
                        writer.write(
                            {
                                **common,
                                "native": {"mse": native_mse},
                                "raw_delta_per_member_oracle": {
                                    "gates": member_gates.tolist(),
                                    "mse": oracle_mse,
                                },
                            }
                        )
                        completed += 1
                        journal.update(
                            f"{label}_raw_delta_oracle",
                            completed,
                            total,
                            split=split,
                            layout=layout_name,
                            step=step,
                            action=action,
                        )
                        if action == canonical_action:
                            canonical_prediction, canonical_actual = prediction, actual
                    if canonical_prediction is None or canonical_actual is None:
                        raise RuntimeError("canonical action was not evaluated")
                    state = one_step._teacher_forced_next(
                        canonical_prediction, canonical_actual
                    )
                for variant in VARIANTS:
                    summary = one_step._layout_summary(
                        rows[variant], layout_name, split
                    )
                    summary.update(
                        prefix=list(prefix),
                        continuation=list(continuation),
                        prefix_diagnostic=prefix_diagnostic,
                    )
                    layouts[variant][split].append(summary)
    finally:
        writer.close()
    return {
        variant: {
            "layouts": layouts[variant],
            "splits": {
                split: one_step._aggregate_split(layouts[variant][split])
                for split in one_step.SPLITS
            },
        }
        for variant in VARIANTS
    }, total


def _gate_distribution(rows_path: Path) -> dict:
    values = []
    for line in rows_path.read_text().splitlines():
        values.extend(json.loads(line)["raw_delta_per_member_oracle"]["gates"])
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _interpret(gates: Mapping[str, bool]) -> tuple[str, str]:
    exp153, exp154 = gates["exp153"], gates["exp154"]
    if not exp153 and not exp154:
        return (
            "both_raw_delta_bounds_fail",
            "Abandon scalar-gated current delta directions; the next minimal mechanism "
            "must learn context-conditioned delta directions, not another gate tune.",
        )
    if not exp153:
        return (
            "exp153_raw_delta_bound_fails",
            "Exp153's contact pass was a suppressive identity illusion and its raw "
            "delta directions are unusable under the registered scalar bound.",
        )
    if not exp154:
        return (
            "auxiliary_corrupted_shared_deltas",
            "Exp153 raw directions pass while exp154 fails; auxiliary supervision "
            "corrupted the shared delta directions.",
        )
    return (
        "raw_directions_expressive_in_both",
        "Both raw bounds pass; learnability, objective, or gate expressivity is the "
        "remaining bottleneck.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp153-checkpoint", type=Path, default=CHECKPOINTS["exp153"]["path"])
    parser.add_argument("--exp154-checkpoint", type=Path, default=CHECKPOINTS["exp154"]["path"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--progress-interval", type=one_step._progress_interval, default=30
    )
    return parser


def _argv(argv) -> list[str]:
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def _prepare_output(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    unexpected = sorted(item.name for item in path.iterdir() if item.name != "run.log")
    if unexpected:
        raise FileExistsError(f"output directory is not empty: {unexpected}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _prepare_output(args.out)
    started = time.monotonic()
    command = os.environ.get("EXP156_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    checkpoint_paths = {
        "exp153": args.exp153_checkpoint,
        "exp154": args.exp154_checkpoint,
    }
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "checkpoint_git_heads": {},
        "checkpoints": {name: str(path) for name, path in checkpoint_paths.items()},
        "arguments": core._jsonable(vars(args)),
        "status": "running",
        "exit_code": None,
        "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with gated.temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            models, metadata_by_name, matching_by_name = {}, {}, {}
            for completed, name in enumerate(("exp153", "exp154"), 1):
                journal.update("initialize", completed - 1, 2, checkpoint=name)
                if name == "exp153":
                    model, probe, head, metadata = auxiliary._load_gated_checkpoint(
                        checkpoint_paths[name]
                    )
                else:
                    model, probe, head, metadata = load_auxiliary_checkpoint(
                        checkpoint_paths[name]
                    )
                del probe
                spec = CHECKPOINTS[name]
                matching = {
                    "canonical_checkpoint_path": checkpoint_paths[name] == spec["path"],
                    "checkpoint_git_head": head == spec["git_head"],
                    "checkpoint_format_version": spec["format_version"] == (3 if name == "exp153" else 4),
                    "checkpoint_budgets": all(
                        metadata["budgets"].get(key) == value
                        for key, value in spec["protocol"].items()
                    ),
                    "checkpoint_config": metadata["config"] == FIXED_CONFIG,
                    "gated_residual": metadata["latent_parameterization"]
                    == "gated_residual_zero_init",
                    "event_supervision": metadata["event_supervision"] is (name == "exp154"),
                }
                models[name] = model
                metadata_by_name[name] = metadata
                matching_by_name[name] = matching
                manifest["checkpoint_git_heads"][name] = head
                manifest.setdefault("checkpoint_metadata", {})[name] = metadata
                manifest.setdefault("protocol_match", {})[name] = matching
                core._write_json(args.out / "manifest.json", manifest)
                journal.update(
                    "initialize",
                    completed,
                    2,
                    checkpoint=name,
                    exact_protocol=all(matching.values()),
                )
            checkpoint_results, gates = {}, {}
            for name in ("exp153", "exp154"):
                rows_path = args.out / f"{name}_raw_delta_oracle_rows.jsonl"
                variants, row_count = diagnose_checkpoint(
                    name, models[name], journal, rows_path
                )
                exact = all(matching_by_name[name].values())
                source = variants["raw_delta_per_member_oracle"]["splits"]["source"]
                gates[name] = raw_delta_upper_bound_gate(source, exact)
                checkpoint_results[name] = {
                    "checkpoint": str(checkpoint_paths[name]),
                    "checkpoint_git_head": manifest["checkpoint_git_heads"][name],
                    "exact_protocol": exact,
                    "protocol_match": matching_by_name[name],
                    "raw_delta_upper_bound_gate": gates[name],
                    "variants": variants,
                    "gate_distribution": _gate_distribution(rows_path),
                    "rows": row_count,
                    "artifacts": {"rows": rows_path.name},
                }
            outcome, conclusion = _interpret(gates)
            journal.update("artifacts", 0, 2, operation="write_results")
            results = {
                "status": "completed",
                "claim": "pre-gate raw-delta scalar expressivity upper bound only",
                "interpretation_limit": (
                    "No learnability, composition, transfer, or AGI claim."
                ),
                "analysis_git_head": manifest["analysis_git_head"],
                "checkpoint_git_heads": manifest["checkpoint_git_heads"],
                "exact_command": command,
                "per_checkpoint": checkpoint_results,
                "upper_bound_gates": gates,
                "outcome": outcome,
                "conclusion": conclusion,
                "protocol": {
                    "source_and_unseen_layouts": "exp148 exact",
                    "fresh_environment_replay_per_action_fork": True,
                    "teacher_forcing": (
                        "actual z/sensors/mask with native prediction hidden"
                    ),
                    "raw_delta": (
                        "latent_heads(native recurrent hidden), before sigmoid gate"
                    ),
                    "per_member_bound": (
                        "independent g_i in [0,1] via exp155 active-set enumeration"
                    ),
                    "rows_per_checkpoint": 120,
                },
                "artifacts": {
                    "progress": "progress.jsonl",
                    "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            core._write_json(args.out / "results.json", results)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            runtime = time.monotonic() - started
            manifest.update(
                status="completed",
                exit_code=0,
                exit_status=0,
                runtime_seconds=runtime,
                exact_protocol=all(
                    all(matching.values()) for matching in matching_by_name.values()
                ),
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 2, 2, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = _exit_code(error)
            manifest.update(
                status="failed",
                exit_code=code,
                exit_status=code,
                runtime_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
            )
            core._write_json(args.out / "manifest.json", manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
