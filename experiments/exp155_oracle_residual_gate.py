"""Oracle scalar-gate expressivity bound for frozen exp150 residual deltas."""

from __future__ import annotations

import argparse
import itertools
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
from experiments import exp150_residual_dynamics as residual
from experiments.exp147_rollout_localization import _exit_code
from experiments.exp151_event_balanced_dynamics import (
    FIXED_CONFIG,
    _load_residual_checkpoint,
)
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


DEFAULT_CHECKPOINT = Path(
    "output_to_user/core/exp150-residual-dynamics-001/residual_checkpoint.pt"
)
EXPECTED_CHECKPOINT_HEAD = "afdf53ea2a50cc3e798662727baa789da31c3c2f"
VARIANTS = ("persistence", "ungated", "shared_scalar", "per_member_scalar")


def _solver_inputs(deltas: torch.Tensor, target: torch.Tensor):
    if deltas.ndim != 2 or target.ndim != 1 or deltas.shape[1] != target.shape[0]:
        raise ValueError("deltas must be [members, z_dim] and target must be [z_dim]")
    if deltas.shape[0] < 1:
        raise ValueError("at least one ensemble member is required")
    if not torch.isfinite(deltas).all() or not torch.isfinite(target).all():
        raise ValueError("oracle least-squares inputs must be finite")
    return deltas.detach().to(device="cpu", dtype=torch.float64), target.detach().to(
        device="cpu", dtype=torch.float64
    )


def solve_shared_scalar_gate(
    deltas: torch.Tensor, target: torch.Tensor
) -> tuple[float, float]:
    """Minimize ||g * mean(deltas) - target||^2 for exactly bounded g."""

    work_deltas, work_target = _solver_inputs(deltas, target)
    direction = work_deltas.mean(0)
    denominator = float(direction.square().sum())
    gate = 0.0 if denominator == 0.0 else float(direction.dot(work_target)) / denominator
    gate = min(1.0, max(0.0, gate))
    mse = float((gate * direction - work_target).square().mean())
    return gate, mse


def solve_per_member_scalar_gates(
    deltas: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, float]:
    """Solve the tiny box-constrained ensemble problem by active-set enumeration."""

    work_deltas, work_target = _solver_inputs(deltas, target)
    members = work_deltas.shape[0]
    design = work_deltas.T / members
    best_gates = torch.zeros(members, dtype=torch.float64)
    best_mse = math.inf
    tolerance = 1e-10
    # Each coordinate is fixed at 0, fixed at 1, or free. This is exact for a
    # convex box-constrained least-squares problem and only costs 3^ensemble.
    for active_set in itertools.product((0, 1, None), repeat=members):
        gates = torch.zeros(members, dtype=torch.float64)
        fixed_one = [index for index, status in enumerate(active_set) if status == 1]
        free = [index for index, status in enumerate(active_set) if status is None]
        if fixed_one:
            gates[fixed_one] = 1.0
        if free:
            residual_target = work_target - design @ gates
            solution = torch.linalg.lstsq(design[:, free], residual_target).solution
            if bool(((solution < -tolerance) | (solution > 1.0 + tolerance)).any()):
                continue
            gates[free] = solution.clamp(0.0, 1.0)
        mse = float((design @ gates - work_target).square().mean())
        if mse < best_mse - 1e-15:
            best_mse = mse
            best_gates = gates
    return best_gates.to(device=deltas.device, dtype=deltas.dtype), best_mse


def oracle_falsification_gate(source_summary: dict, exact_protocol: bool) -> bool:
    free_ratio = source_summary["medians"]["free_forward_prediction_persistence_ratio"]
    return bool(
        exact_protocol
        and source_summary["contact_failure_layouts"] == 0
        and source_summary["blocked_noop_failure_layouts"] == 0
        and free_ratio is not None
        and math.isfinite(free_ratio)
        and free_ratio < 1.0
    )


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
def diagnose_oracle(model, journal, rows_path: Path):
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * one_step.HORIZON * 5
    completed = 0
    layouts_by_variant = {
        variant: {split: [] for split in one_step.SPLITS} for variant in VARIANTS
    }
    writer = core.TraceWriter(rows_path)
    journal.update("oracle_forks", 0, total)
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
                rows_by_variant = {variant: [] for variant in VARIANTS}
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
                        prediction = model.step(state, action_tensor)
                        actual = model.initial(after)
                        deltas = prediction.member_z[:, 0] - state.z[0]
                        target = actual.z[0] - state.z[0]
                        shared_gate, shared_solver_mse = solve_shared_scalar_gate(
                            deltas, target
                        )
                        member_gates, member_solver_mse = solve_per_member_scalar_gates(
                            deltas, target
                        )
                        mean_delta = deltas.mean(0)
                        persistence_mse = float(target.square().mean())
                        ungated_mse = float((mean_delta - target).square().mean())
                        shared_mse = float(
                            (shared_gate * mean_delta - target).square().mean()
                        )
                        member_mse = float(
                            (
                                (member_gates[:, None] * deltas).mean(0) - target
                            ).square().mean()
                        )
                        if not math.isclose(shared_mse, shared_solver_mse, rel_tol=1e-5, abs_tol=1e-8):
                            raise AssertionError("shared oracle solver/report mismatch")
                        if not math.isclose(member_mse, member_solver_mse, rel_tol=1e-5, abs_tol=1e-8):
                            raise AssertionError("per-member oracle solver/report mismatch")
                        common = {
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "real_history": list(history),
                            "canonical_action": canonical_action,
                            "action": action,
                            "action_name": GRID_ACTIONS[action],
                            "actual_diagnostic": diagnostic,
                            "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                            "persistence_vs_actual_next_z_mse": persistence_mse,
                        }
                        mse_by_variant = {
                            "persistence": persistence_mse,
                            "ungated": ungated_mse,
                            "shared_scalar": shared_mse,
                            "per_member_scalar": member_mse,
                        }
                        for variant, mse in mse_by_variant.items():
                            rows_by_variant[variant].append(_variant_row(common, mse))
                        writer.write(
                            {
                                **common,
                                "ungated": {"mse": ungated_mse},
                                "shared_scalar": {"gate": shared_gate, "mse": shared_mse},
                                "per_member_scalar": {
                                    "gates": member_gates.tolist(),
                                    "mse": member_mse,
                                },
                            }
                        )
                        completed += 1
                        journal.update(
                            "oracle_forks",
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
                    # Exactly exp148: actual z/sensors/mask with the frozen base
                    # prediction's recurrent hidden state. Gates do not alter hidden.
                    state = one_step._teacher_forced_next(
                        canonical_prediction, canonical_actual
                    )
                for variant in VARIANTS:
                    summary = one_step._layout_summary(
                        rows_by_variant[variant], layout_name, split
                    )
                    summary.update(
                        prefix=list(prefix),
                        continuation=list(continuation),
                        prefix_diagnostic=prefix_diagnostic,
                    )
                    layouts_by_variant[variant][split].append(summary)
    finally:
        writer.close()
    variants = {}
    for variant in VARIANTS:
        variants[variant] = {
            "layouts": layouts_by_variant[variant],
            "splits": {
                split: one_step._aggregate_split(layouts_by_variant[variant][split])
                for split in one_step.SPLITS
            },
        }
    return variants, total


def _gate_distribution(rows_path: Path) -> dict:
    shared, per_member = [], []
    for line in rows_path.read_text().splitlines():
        row = json.loads(line)
        shared.append(row["shared_scalar"]["gate"])
        per_member.extend(row["per_member_scalar"]["gates"])
    return {
        "shared_scalar": {
            "count": len(shared),
            "min": min(shared),
            "median": statistics.median(shared),
            "max": max(shared),
        },
        "per_member_scalar": {
            "count": len(per_member),
            "min": min(per_member),
            "median": statistics.median(per_member),
            "max": max(per_member),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
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
    command = os.environ.get("EXP155_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "checkpoint_git_head": None,
        "checkpoint": str(args.checkpoint),
        "arguments": core._jsonable(vars(args)),
        "status": "running",
        "exit_code": None,
        "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with residual.temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 1, operation="safe_residual_checkpoint_load")
            model, probe, checkpoint_head, metadata = _load_residual_checkpoint(
                args.checkpoint
            )
            del probe
            manifest.update(
                checkpoint_git_head=checkpoint_head,
                checkpoint_metadata=metadata,
            )
            matching = {
                "canonical_checkpoint_path": args.checkpoint == DEFAULT_CHECKPOINT,
                "checkpoint_git_head": checkpoint_head == EXPECTED_CHECKPOINT_HEAD,
                "checkpoint_budgets": all(
                    metadata["budgets"].get(name) == value
                    for name, value in residual.PROTOCOL.items()
                ),
                "checkpoint_config": metadata["config"] == FIXED_CONFIG,
                "uniform_residual": metadata["event_balanced"] is False,
            }
            exact_protocol = all(matching.values())
            manifest["protocol_match"] = matching
            core._write_json(args.out / "manifest.json", manifest)
            journal.update(
                "initialize", 1, 1, device=metadata["device"], exact_protocol=exact_protocol
            )
            rows_path = args.out / "oracle_rows.jsonl"
            variants, row_count = diagnose_oracle(model, journal, rows_path)
            per_member_source = variants["per_member_scalar"]["splits"]["source"]
            per_member_gate = oracle_falsification_gate(
                per_member_source, exact_protocol
            )
            shared_gate = oracle_falsification_gate(
                variants["shared_scalar"]["splits"]["source"], exact_protocol
            )
            if per_member_gate:
                conclusion = (
                    "Per-member scalar gating can rescue the frozen exp150 delta directions; "
                    "the next test is a learnable action-specific gate."
                )
            elif per_member_source["contact_failure_layouts"]:
                conclusion = (
                    "Fixed exp150 residual delta directions cannot be rescued by scalar gates. "
                    "This does not falsify jointly learned or different delta directions."
                )
            else:
                conclusion = (
                    "Scalar gates do not satisfy the full exp148 one-step bound for frozen exp150 "
                    "deltas. This does not falsify jointly learned or different delta directions."
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            results = {
                "status": "completed",
                "claim": "oracle expressivity upper bound only",
                "interpretation_limit": (
                    "No learnability, generalization, composition, transfer, or AGI evidence."
                ),
                "analysis_git_head": manifest["analysis_git_head"],
                "checkpoint_git_head": checkpoint_head,
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "per_member_oracle_falsification_gate": per_member_gate,
                "shared_scalar_support_gate": shared_gate,
                "variants": variants,
                "gate_distribution": _gate_distribution(rows_path),
                "conclusion": conclusion,
                "protocol": {
                    "source_and_unseen_layouts": "exp148 exact",
                    "fresh_environment_replay_per_action_fork": True,
                    "teacher_forcing": (
                        "actual z/sensors/mask with frozen ungated prediction hidden"
                    ),
                    "oracle_target": "actual_next_z - current_z",
                    "shared_bound": "one g in [0,1] for all ensemble members",
                    "per_member_bound": "independent g_i in [0,1] via active-set enumeration",
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
                exact_protocol=exact_protocol,
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
