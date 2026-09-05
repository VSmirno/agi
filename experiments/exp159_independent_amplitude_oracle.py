"""Checkpoint-only independent-member raw-delta amplitude oracle audit."""

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

import numpy as np
import torch

from experiments import exp148_source_target_one_step as one_step
from experiments import exp153_change_gated_dynamics as gated
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp155_oracle_residual_gate as joint_oracle
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments.exp147_rollout_localization import _exit_code
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


CHECKPOINT = raw_oracle.CHECKPOINTS["exp153"]
VARIANTS = ("native", "independent_amplitude_oracle", "joint_active_set_oracle")


def independent_member_amplitudes(
    raw_deltas: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Return each member's independently optimal scalar in [0, 1]."""

    numerators = (raw_deltas * target.unsqueeze(0)).sum(dim=-1)
    denominators = raw_deltas.square().sum(dim=-1)
    gates = torch.where(
        denominators > 0,
        numerators / denominators.clamp_min(torch.finfo(raw_deltas.dtype).tiny),
        torch.zeros_like(numerators),
    )
    return gates.clamp(0.0, 1.0)


def independent_target_upper_bound_gate(
    splits: Mapping[str, Mapping], exact_protocol: bool
) -> bool:
    """Apply the preregistered one-step predicate to both transfer splits."""

    if not exact_protocol:
        return False
    for split in one_step.SPLITS:
        summary = splits[split]
        ratio = summary["medians"]["free_forward_prediction_persistence_ratio"]
        if not (
            summary["contact_failure_layouts"] == 0
            and summary["blocked_noop_failure_layouts"] == 0
            and ratio is not None
            and math.isfinite(ratio)
            and ratio < 1.0
        ):
            return False
    return True


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
def diagnose(model, journal, rows_path: Path):
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * one_step.HORIZON * 5
    completed = 0
    layouts = {
        variant: {split: [] for split in one_step.SPLITS} for variant in VARIANTS
    }
    writer = core.TraceWriter(rows_path)
    journal.update("independent_amplitude_oracle", 0, total)
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
                        prediction, raw_deltas = (
                            raw_oracle.native_prediction_and_raw_deltas(
                                model, state, action_tensor
                            )
                        )
                        actual = model.initial(after)
                        target = actual.z[0] - state.z[0]
                        deltas = raw_deltas[:, 0]
                        independent_gates = independent_member_amplitudes(deltas, target)
                        independent_delta = (
                            independent_gates[:, None] * deltas
                        ).mean(dim=0)
                        independent_mse = float(
                            (independent_delta - target).square().mean()
                        )
                        joint_gates, joint_mse = (
                            joint_oracle.solve_per_member_scalar_gates(deltas, target)
                        )
                        joint_reported_mse = float(
                            ((joint_gates[:, None] * deltas).mean(0) - target)
                            .square()
                            .mean()
                        )
                        if not math.isclose(
                            joint_reported_mse,
                            joint_mse,
                            rel_tol=1e-5,
                            abs_tol=1e-8,
                        ):
                            raise AssertionError("joint active-set solver/report mismatch")
                        native_mse = float(
                            (prediction.next_state.z[0] - actual.z[0]).square().mean()
                        )
                        common = {
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "real_history": list(history),
                            "canonical_action": canonical_action,
                            "action": action,
                            "action_name": GRID_ACTIONS[action],
                            "actual_diagnostic": diagnostic,
                            "rgb_changed": bool(
                                not np.array_equal(before.rgb, after.rgb)
                            ),
                            "persistence_vs_actual_next_z_mse": float(
                                target.square().mean()
                            ),
                        }
                        rows["native"].append(_variant_row(common, native_mse))
                        rows["independent_amplitude_oracle"].append(
                            _variant_row(common, independent_mse)
                        )
                        rows["joint_active_set_oracle"].append(
                            _variant_row(common, joint_reported_mse)
                        )
                        writer.write(
                            {
                                **common,
                                "native": {"mse": native_mse},
                                "independent_amplitude_oracle": {
                                    "gates": independent_gates.tolist(),
                                    "mse": independent_mse,
                                },
                                "joint_active_set_oracle": {
                                    "gates": joint_gates.tolist(),
                                    "mse": joint_reported_mse,
                                },
                            }
                        )
                        completed += 1
                        journal.update(
                            "independent_amplitude_oracle",
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


def _gate_distribution(rows_path: Path, variant: str) -> dict:
    values = []
    for line in rows_path.read_text().splitlines():
        values.extend(json.loads(line)[variant]["gates"])
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _interpret(independent_pass: bool, joint_pass: bool) -> tuple[str, str]:
    if independent_pass:
        return (
            "independent_amplitude_target_licensed",
            "The cheap self-supervised per-member amplitude regression target is "
            "licensed for the next causal training test.",
        )
    if joint_pass:
        return (
            "independent_scalars_require_coordination",
            "Independent scalar supervision is insufficient even though the joint "
            "oracle passes; the next mechanism needs a joint target or vector "
            "prediction, not a longer scalar regression run.",
        )
    return (
        "independent_and_joint_bounds_fail",
        "Neither scalar upper bound passes; independent amplitude regression is not "
        "licensed and the current raw delta directions remain insufficient.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT["path"])
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
    command = os.environ.get("EXP159_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_git_head": None,
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
            journal.update("initialize", 0, 1, operation="safe_exp153_load")
            model, probe, checkpoint_head, metadata = (
                auxiliary._load_gated_checkpoint(args.checkpoint)
            )
            del probe
            matching = {
                "canonical_checkpoint_path": args.checkpoint == CHECKPOINT["path"],
                "checkpoint_git_head": checkpoint_head == CHECKPOINT["git_head"],
                "checkpoint_format_version": CHECKPOINT["format_version"] == 3,
                "checkpoint_budgets": all(
                    metadata["budgets"].get(key) == value
                    for key, value in CHECKPOINT["protocol"].items()
                ),
                "checkpoint_config": metadata["config"] == FIXED_CONFIG,
                "gated_residual": metadata["latent_parameterization"]
                == "gated_residual_zero_init",
                "event_supervision": metadata["event_supervision"] is False,
            }
            exact = all(matching.values())
            manifest.update(
                checkpoint_git_head=checkpoint_head,
                checkpoint_metadata=metadata,
                protocol_match=matching,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("initialize", 1, 1, exact_protocol=exact)

            rows_path = args.out / "independent_amplitude_oracle_rows.jsonl"
            variants, row_count = diagnose(model, journal, rows_path)
            independent_pass = independent_target_upper_bound_gate(
                variants["independent_amplitude_oracle"]["splits"], exact
            )
            joint_pass = independent_target_upper_bound_gate(
                variants["joint_active_set_oracle"]["splits"], exact
            )
            outcome, conclusion = _interpret(independent_pass, joint_pass)
            journal.update("artifacts", 0, 2, operation="write_results")
            results = {
                "status": "completed",
                "claim": "checkpoint-only independent-member amplitude upper bound",
                "interpretation_limit": "No training, composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "checkpoint": str(args.checkpoint),
                "checkpoint_git_head": checkpoint_head,
                "exact_command": command,
                "exact_protocol": exact,
                "protocol_match": matching,
                "variants": variants,
                "independent_target_upper_bound_gate": independent_pass,
                "joint_active_set_control_gate": joint_pass,
                "outcome": outcome,
                "conclusion": conclusion,
                "gate_distributions": {
                    variant: _gate_distribution(rows_path, variant)
                    for variant in (
                        "independent_amplitude_oracle",
                        "joint_active_set_oracle",
                    )
                },
                "protocol": {
                    "source_and_unseen_layouts": "exp148 exact",
                    "fresh_environment_replay_per_action_fork": True,
                    "teacher_forcing": "actual z/sensors/mask with native prediction hidden",
                    "raw_delta": "latent_heads(native recurrent hidden), before sigmoid gate",
                    "independent_target": "clip(dot(d_i,t)/dot(d_i,d_i),0,1); zero direction -> 0",
                    "candidate_prediction": "current_z + mean_i(g_i * d_i)",
                    "joint_control": "exp156 active-set oracle on identical rows",
                    "rows": row_count,
                },
                "artifacts": {
                    "rows": rows_path.name,
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
                exact_protocol=exact,
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
