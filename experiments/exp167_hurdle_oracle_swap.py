"""Checkpoint-only 2x2 oracle swap for the frozen exp166 hurdle probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments import exp159_independent_amplitude_oracle as amplitude_oracle
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments import exp164_relational_slot_probe as relational
from experiments import exp165_relational_pose_probe as pose_probe
from experiments import exp166_hurdle_amplitude_probe as hurdle
from experiments.exp147_rollout_localization import _exit_code
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG
from snks.pipeline import core_experiment as core


DEFAULT_BASELINE = hurdle.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = hurdle.EXPECTED_BASELINE_HEAD
DEFAULT_CHECKPOINT = Path(
    "output_to_user/core/exp166-hurdle-amplitude-probe-001/"
    "hurdle_amplitude_probe.pt"
)
DEFAULT_EXP166_REFERENCE = Path(
    "output_to_user/core/exp166-hurdle-amplitude-probe-001/results.json"
)
EXPECTED_EXP166_HEAD = "e1e1bc4fc80f2844eebf1f2537518fd007bdfc52"
DEFAULT_EXP159_REFERENCE = Path(
    "output_to_user/core/exp159-independent-amplitude-oracle-001/results.json"
)
EXPECTED_EXP159_HEAD = "2449e374fc456ea48abff0579d7c849efa28bf6f"
ARMS = ("PP", "PO", "OP", "OO")


def oracle_swap_gates(atom_logits, conditional, oracle):
    """Return the four predicted/oracle atom and conditional combinations."""

    if atom_logits.shape != conditional.shape or oracle.shape != conditional.shape:
        raise ValueError("all hurdle components must have equal shape")
    predicted_atom = atom_logits.sigmoid() >= hurdle.ATOM_BOUNDARY
    oracle_atom = oracle > 0
    zero = torch.zeros_like(conditional)
    return {
        "PP": torch.where(predicted_atom, conditional, zero),
        "PO": torch.where(predicted_atom, oracle, zero),
        "OP": torch.where(oracle_atom, conditional, zero),
        "OO": oracle,
    }


def interpret_swap(po_pass: bool, op_pass: bool, oo_pass: bool):
    """Interpret the preregistered component swap truth table."""

    if not oo_pass:
        return (
            "invalid_oracle_audit",
            "The independent oracle control failed; the swap audit is invalid.",
        )
    if not po_pass and op_pass:
        return (
            "atom_bottleneck",
            "Oracle atom decisions repair the audit while oracle conditional values do not; "
            "atom prediction is the bottleneck.",
        )
    if po_pass and not op_pass:
        return (
            "conditional_bottleneck",
            "Oracle conditional amplitudes repair the audit while oracle atom decisions do "
            "not; conditional amplitude is the bottleneck.",
        )
    if not po_pass and not op_pass:
        return (
            "both_components_fail",
            "Neither single oracle component repairs the audit while OO passes; both atom "
            "and conditional components contribute.",
        )
    return (
        "error_interaction",
        "Both single-component swaps pass; the original failure comes from interaction "
        "between their prediction errors.",
    )


def _load_hurdle_checkpoint(path: Path, baseline):
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ValueError(f"could not safely load exp166 checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("exp166 checkpoint requires format_version 1")
    expected = {
        "analysis_git_head": EXPECTED_EXP166_HEAD,
        "baseline_checkpoint_git_head": EXPECTED_BASELINE_HEAD,
        "objective": hurdle.OBJECTIVE,
        "z_dim": baseline.encoder.z_dim,
        "h_dim": baseline.h_dim,
        "pose_dim": hurdle.POSE_DIM,
        "ensemble_size": baseline.heads,
        "hidden_width": hurdle.HIDDEN_WIDTH,
        "atom_boundary": hurdle.ATOM_BOUNDARY,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"exp166 checkpoint {key} mismatch")
    state_dict = payload.get("probe_state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("exp166 checkpoint lacks probe state dict")
    parameter = next(baseline.parameters())
    probe = hurdle.HurdleAmplitudeProbe(
        baseline.encoder.z_dim, baseline.h_dim, baseline.heads
    ).to(device=parameter.device, dtype=parameter.dtype)
    probe.load_state_dict(state_dict, strict=True)
    probe.eval().requires_grad_(False)
    return probe, {
        key: payload[key] for key in (
            "format_version", "analysis_git_head", "baseline_checkpoint_git_head",
            "z_dim", "h_dim", "pose_dim", "ensemble_size", "hidden_width",
            "atom_boundary",
        )
    }


def _load_reference(path: Path, expected_head: str, role: str):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {role} reference: {error}") from error
    if payload.get("analysis_git_head") != expected_head:
        raise ValueError(f"{role} reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError(f"{role} reference is not exact protocol")
    rows_name = payload.get("artifacts", {}).get("rows")
    rows_path = path.parent / str(rows_name)
    try:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {role} rows: {error}") from error
    if len(rows) != 120:
        raise ValueError(f"{role} reference requires exactly 120 rows")
    return payload, rows


def _variant_row(common, prediction_mse):
    persistence = common["persistence_vs_actual_next_z_mse"]
    return {
        **common,
        "predicted_vs_actual_next_z_mse": prediction_mse,
        "prediction_to_persistence_ratio": (
            prediction_mse / persistence if persistence > 0 else None
        ),
    }


@torch.inference_mode()
def diagnose(model, journal, rows_path: Path):
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * 3 * 5
    completed = 0
    layouts = {arm: {split: [] for split in one_step.SPLITS} for arm in ARMS}
    mismatches = defaultdict(lambda: {"match": 0, "false_positive": 0, "false_negative": 0})
    writer = core.TraceWriter(rows_path)
    journal.update("oracle_swap", 0, total)
    try:
        for split in one_step.SPLITS:
            for layout_name, spec in specs[split].items():
                layout, actions = spec[:2]
                prefix, continuation = one_step._validate_protocol(
                    split, layout_name, layout, actions, one_step.SEED
                )
                state, prefix_diagnostic = pose_probe._replay_pose_prefix(
                    model, layout, prefix, one_step.SEED
                )
                rows = {arm: [] for arm in ARMS}
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, diagnostic, pose = pose_probe._fresh_pose_fork(
                            layout, history, action, one_step.SEED
                        )
                        pose = pose[None].to(state.z.device, state.z.dtype)
                        model.set_relations(pose)
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
                        oracle = amplitude_oracle.independent_member_amplitudes(
                            deltas, target
                        )[:, None]
                        logits, conditional = model.amplitude_probe.components(
                            state.z, state.hidden, pose, action_tensor
                        )
                        gates = oracle_swap_gates(logits, conditional, oracle)
                        native_gate = model.change_gates(state, action_tensor).squeeze(-1)
                        torch.testing.assert_close(gates["PP"], native_gate)
                        predicted_atom = (logits.sigmoid() >= hurdle.ATOM_BOUNDARY).flatten()
                        oracle_atom = (oracle > 0).flatten()
                        key = f"{split}/action{action}"
                        mismatches[key]["match"] += int((predicted_atom == oracle_atom).sum())
                        mismatches[key]["false_positive"] += int(
                            (predicted_atom & ~oracle_atom).sum()
                        )
                        mismatches[key]["false_negative"] += int(
                            (~predicted_atom & oracle_atom).sum()
                        )
                        common = {
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "real_history": list(history),
                            "canonical_action": canonical_action,
                            "action": action,
                            "action_name": one_step.GRID_ACTIONS[action],
                            "actual_diagnostic": diagnostic,
                            "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                            "persistence_vs_actual_next_z_mse": float(target.square().mean()),
                            "privileged_pose": pose[0].tolist(),
                        }
                        arm_payload = {}
                        for arm in ARMS:
                            delta = (gates[arm][:, 0, None] * deltas).mean(dim=0)
                            mse = float((delta - target).square().mean())
                            rows[arm].append(_variant_row(common, mse))
                            arm_payload[arm] = {"gates": gates[arm].flatten().tolist(), "mse": mse}
                        pp_mse = float(
                            (prediction.next_state.z[0] - actual.z[0]).square().mean()
                        )
                        if not math.isclose(pp_mse, arm_payload["PP"]["mse"], rel_tol=1e-5, abs_tol=1e-8):
                            raise AssertionError("PP formula does not match installed exp166 model")
                        writer.write({
                            **common,
                            "predicted_atom": predicted_atom.tolist(),
                            "oracle_atom": oracle_atom.tolist(),
                            "predicted_atom_probability": logits.sigmoid().flatten().tolist(),
                            "predicted_conditional": conditional.flatten().tolist(),
                            "oracle_amplitude": oracle.flatten().tolist(),
                            "arms": arm_payload,
                        })
                        completed += 1
                        journal.update(
                            "oracle_swap", completed, total,
                            split=split, layout=layout_name, step=step, action=action,
                        )
                        if action == canonical_action:
                            canonical_prediction, canonical_actual = prediction, actual
                    state = one_step._teacher_forced_next(
                        canonical_prediction, canonical_actual
                    )
                for arm in ARMS:
                    summary = one_step._layout_summary(rows[arm], layout_name, split)
                    summary.update(
                        prefix=list(prefix), continuation=list(continuation),
                        prefix_diagnostic=prefix_diagnostic,
                    )
                    layouts[arm][split].append(summary)
    finally:
        writer.close()
    variants = {
        arm: {
            "layouts": layouts[arm],
            "splits": {
                split: one_step._aggregate_split(layouts[arm][split])
                for split in one_step.SPLITS
            },
        }
        for arm in ARMS
    }
    return variants, dict(sorted(mismatches.items()))


def _row_key(row):
    return (
        row["split"], row["layout"], row["step"], tuple(row["real_history"]),
        row["canonical_action"], row["action"], row["action_name"], row["rgb_changed"],
    )


def _assert_reference_alignment(rows, exp166_rows, exp159_rows):
    keys = [_row_key(row) for row in rows]
    if keys != [_row_key(row) for row in exp166_rows]:
        raise AssertionError("PP row signature differs from exp166")
    if keys != [_row_key(row) for row in exp159_rows]:
        raise AssertionError("OO row signature differs from exp159")
    pp_mse = [
        abs(row["arms"]["PP"]["mse"] - reference["predicted_vs_actual_next_z_mse"])
        for row, reference in zip(rows, exp166_rows)
    ]
    oo_mse = [
        abs(row["arms"]["OO"]["mse"] - reference["independent_amplitude_oracle"]["mse"])
        for row, reference in zip(rows, exp159_rows)
    ]
    oo_gate = [
        abs(value - expected)
        for row, reference in zip(rows, exp159_rows)
        for value, expected in zip(
            row["arms"]["OO"]["gates"],
            reference["independent_amplitude_oracle"]["gates"],
        )
    ]
    alignment = {
        "rows": len(rows),
        "ordered_protocol_rows_equal": True,
        "pp_max_abs_mse_difference": max(pp_mse),
        "oo_max_abs_mse_difference": max(oo_mse),
        "oo_max_abs_gate_difference": max(oo_gate),
    }
    if max(pp_mse) > 1e-7 or max(oo_mse) > 1e-7 or max(oo_gate) > 1e-7:
        raise AssertionError(f"oracle swap reference mismatch: {alignment}")
    return alignment


def _arm_statistics(rows):
    result = {}
    for arm in ARMS:
        gates = [value for row in rows for value in row["arms"][arm]["gates"]]
        result[arm] = {
            "member_values": len(gates),
            "min": min(gates),
            "median": statistics.median(gates),
            "mean": sum(gates) / len(gates),
            "max": max(gates),
            "exact_zero": sum(value == 0 for value in gates),
            "exact_zero_rate": sum(value == 0 for value in gates) / len(gates),
            "by_split": {
                split: {
                    "member_values": len(values),
                    "exact_zero": sum(value == 0 for value in values),
                    "exact_zero_rate": sum(value == 0 for value in values) / len(values),
                    "mean": sum(values) / len(values),
                }
                for split in one_step.SPLITS
                for values in [[
                    value for row in rows if row["split"] == split
                    for value in row["arms"][arm]["gates"]
                ]]
            },
        }
    return result


def _metric_signature(variant):
    return {
        split: {
            "contact_failure_layouts": variant["splits"][split]["contact_failure_layouts"],
            "blocked_noop_failure_layouts": variant["splits"][split]["blocked_noop_failure_layouts"],
            "medians": variant["splits"][split]["medians"],
        }
        for split in one_step.SPLITS
    }


def metric_signatures_match(candidate, reference, tolerance: float = 1e-7) -> bool:
    """Compare discrete metrics exactly and finite aggregates within roundoff."""

    if candidate.keys() != reference.keys():
        return False
    for split in candidate:
        current, expected = candidate[split], reference[split]
        for key in ("contact_failure_layouts", "blocked_noop_failure_layouts"):
            if current.get(key) != expected.get(key):
                return False
        current_medians = current.get("medians", {})
        expected_medians = expected.get("medians", {})
        if current_medians.keys() != expected_medians.keys():
            return False
        for key, value in current_medians.items():
            expected_value = expected_medians[key]
            if value is None or expected_value is None:
                if value is not expected_value:
                    return False
            elif not (
                math.isfinite(value)
                and math.isfinite(expected_value)
                and abs(value - expected_value) <= tolerance
            ):
                return False
    return True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--exp166-reference", type=Path, default=DEFAULT_EXP166_REFERENCE)
    parser.add_argument("--exp159-reference", type=Path, default=DEFAULT_EXP159_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--progress-interval", type=one_step._progress_interval, default=30
    )
    return parser


def _argv(argv):
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    amplitude_oracle._prepare_output(args.out)
    started = time.monotonic()
    command = os.environ.get("EXP167_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "checkpoint": str(args.checkpoint),
        "exp166_reference": str(args.exp166_reference),
        "exp159_reference": str(args.exp159_reference),
        "arguments": core._jsonable(vars(args)),
        "status": "running",
        "exit_code": None,
        "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with hurdle.temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 4, operation="safe_exp153_load")
            baseline, _ordered, baseline_head, baseline_metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            journal.update("initialize", 1, 4, operation="safe_exp166_load")
            probe, checkpoint_metadata = _load_hurdle_checkpoint(
                args.checkpoint, baseline
            )
            model = hurdle._installed_model(baseline, probe)
            journal.update("initialize", 2, 4, operation="load_exp166_reference")
            exp166_reference, exp166_rows = _load_reference(
                args.exp166_reference, EXPECTED_EXP166_HEAD, "exp166"
            )
            journal.update("initialize", 3, 4, operation="load_exp159_reference")
            exp159_reference, exp159_rows = _load_reference(
                args.exp159_reference, EXPECTED_EXP159_HEAD, "exp159"
            )
            matching = {
                "canonical_baseline": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "baseline_config": baseline_metadata["config"] == FIXED_CONFIG,
                "canonical_checkpoint": args.checkpoint == DEFAULT_CHECKPOINT,
                "checkpoint_head": checkpoint_metadata["analysis_git_head"] == EXPECTED_EXP166_HEAD,
                "canonical_exp166_reference": args.exp166_reference == DEFAULT_EXP166_REFERENCE,
                "exp166_reference_head": exp166_reference["analysis_git_head"] == EXPECTED_EXP166_HEAD,
                "canonical_exp159_reference": args.exp159_reference == DEFAULT_EXP159_REFERENCE,
                "exp159_reference_head": exp159_reference["analysis_git_head"] == EXPECTED_EXP159_HEAD,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=baseline_metadata,
                checkpoint_metadata=checkpoint_metadata,
                protocol_match=matching,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("initialize", 4, 4)
            rows_path = args.out / "hurdle_oracle_swap_rows.jsonl"
            variants, mismatches = diagnose(model, journal, rows_path)
            rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
            alignment = _assert_reference_alignment(
                rows, exp166_rows, exp159_rows
            )
            matching["canonical_row_alignment"] = alignment["ordered_protocol_rows_equal"]
            matching["pp_matches_exp166"] = alignment["pp_max_abs_mse_difference"] <= 1e-7
            matching["oo_matches_exp159"] = (
                alignment["oo_max_abs_mse_difference"] <= 1e-7
                and alignment["oo_max_abs_gate_difference"] <= 1e-7
            )
            matching["pp_metric_signature"] = metric_signatures_match(
                _metric_signature(variants["PP"]),
                _metric_signature({"splits": exp166_reference["one_step"]["splits"]}),
            )
            matching["oo_metric_signature"] = metric_signatures_match(
                _metric_signature(variants["OO"]),
                _metric_signature(
                    exp159_reference["variants"]["independent_amplitude_oracle"]
                ),
            )
            exact = all(matching.values())
            gates = {
                arm: nonlinear.nonlinear_probe_gate(
                    variants[arm]["splits"]["source"],
                    variants[arm]["splits"]["unseen"],
                    exact,
                )
                for arm in ARMS
            }
            outcome, conclusion = interpret_swap(
                gates["PO"], gates["OP"], gates["OO"]
            )
            statistics_by_arm = _arm_statistics(rows)
            totals = {"match": 0, "false_positive": 0, "false_negative": 0}
            for values in mismatches.values():
                for key in totals:
                    totals[key] += values[key]
            mismatch_report = {
                "member_decisions": sum(totals.values()),
                **totals,
                "by_split_action": mismatches,
            }
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "checkpoint-only hurdle component oracle swap",
                "interpretation_limit": (
                    "Actual-next data is used only to construct PO/OP/OO oracle "
                    "components; no training, composition, transfer, or AGI claim."
                ),
                "analysis_git_head": manifest["analysis_git_head"],
                "exact_command": command,
                "exact_protocol": exact,
                "protocol_match": matching,
                "reference_alignment": alignment,
                "protocol": {
                    "source_and_unseen_layouts": "exp148 exact 120 rows",
                    "teacher_forcing": "actual z/sensors/mask with carried predicted hidden",
                    "pose": "current BEFORE relations plus agent_dir one-hot",
                    "raw_delta": "frozen exp153 heads before gate",
                    "oracle": "exp159 independent analytic member amplitude",
                    "actual_next_use": "oracle components and scoring only",
                },
                "arms": variants,
                "arm_gates": gates,
                "arm_gate_statistics": statistics_by_arm,
                "atom_mismatches": mismatch_report,
                "pp_reference_metric_signature": _metric_signature(variants["PP"]),
                "oo_reference_metric_signature": _metric_signature(variants["OO"]),
                "exp166_reference": {
                    "path": str(args.exp166_reference),
                    "analysis_git_head": exp166_reference["analysis_git_head"],
                    "gate": exp166_reference["hurdle_amplitude_gate"],
                },
                "exp159_reference": {
                    "path": str(args.exp159_reference),
                    "analysis_git_head": exp159_reference["analysis_git_head"],
                    "gate": exp159_reference["independent_target_upper_bound_gate"],
                },
                "outcome": outcome,
                "conclusion": conclusion,
                "controls": {
                    "checkpoint_only": True,
                    "new_weights": False,
                    "corpus_training": False,
                    "posthoc_threshold": False,
                    "pp_uses_future_data": False,
                    "oracle_arms_use_actual_next": True,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "rows": rows_path.name,
                    "progress": "progress.jsonl",
                    "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            core._write_json(args.out / "results.json", result)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            manifest.update(
                status="completed", exit_code=0, exit_status=0,
                runtime_seconds=time.monotonic() - started,
                exact_protocol=exact, protocol_match=matching,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 2, 2, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = _exit_code(error)
            manifest.update(
                status="failed", exit_code=code, exit_status=code,
                runtime_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
            )
            core._write_json(args.out / "manifest.json", manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
