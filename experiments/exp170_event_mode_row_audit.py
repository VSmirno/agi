"""Artifact-only row audit of exp169 event selection over frozen exp168 vectors."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp148_source_target_one_step as one_step
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments import exp167_hurdle_oracle_swap as signature
from snks.pipeline import core_experiment as core


DEFAULT_EXP168_DIR = Path(
    "output_to_user/core/exp168-relational-pose-vector-delta-001"
)
DEFAULT_EXP169_DIR = Path(
    "output_to_user/core/exp169-event-mode-vector-transition-001"
)
EXPECTED_EXP168_HEAD = "a5349328a1541f4c80ab1c19ea534961dcf7eea8"
EXPECTED_EXP169_HEAD = "d063acaa8b93fa84d2bad4cdfeb5fe45e0f6d092"
ROW_FIELDS = (
    "split", "layout", "step", "real_history", "canonical_action",
    "action", "action_name", "rgb_changed",
)


def row_key(row: Mapping):
    """Return the exact canonical fork identity, including real history."""

    return tuple(
        tuple(row[name]) if name == "real_history" else row[name]
        for name in ROW_FIELDS
    )


def oracle_event_row(event_row: Mapping, frozen_row: Mapping) -> dict:
    """Select frozen vector MSE on observed change and exact persistence otherwise."""

    if row_key(event_row) != row_key(frozen_row):
        raise AssertionError("event and frozen row keys differ")
    event_persistence = event_row["persistence_vs_actual_next_z_mse"]
    frozen_persistence = frozen_row["persistence_vs_actual_next_z_mse"]
    if not math.isclose(
        event_persistence, frozen_persistence, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError("event and frozen persistence targets differ")
    changed = bool(event_row["rgb_changed"])
    prediction = (
        float(frozen_row["predicted_vs_actual_next_z_mse"])
        if changed else 0.0
    )
    persistence = float(event_persistence)
    return {
        **event_row,
        "predicted_vs_actual_next_z_mse": prediction,
        "prediction_to_persistence_ratio": (
            prediction / persistence if persistence > 0.0 else None
        ),
        "oracle_event_uses_vector": changed,
        "oracle_event_source": (
            "frozen_exp168_vector" if changed else "literal_persistence"
        ),
    }


def _read_json(path: Path, role: str):
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {role}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return value


def _read_rows(path: Path, role: str):
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {role}: {error}") from error
    if len(rows) != 120 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{role} must contain exactly 120 object rows")
    return rows


def _metric_match(candidate, reference):
    return signature.metric_signatures_match(
        signature._metric_signature(candidate),
        signature._metric_signature(reference),
    )


def _summarize_rows(rows):
    layouts = {split: [] for split in one_step.SPLITS}
    per_layout = {split: {} for split in one_step.SPLITS}
    for split in one_step.SPLITS:
        names = list(one_step._layout_specs()[split])
        for layout in names:
            selected = [
                row for row in rows
                if row["split"] == split and row["layout"] == layout
            ]
            if len(selected) != 15:
                raise AssertionError(f"{split}/{layout} requires 15 rows")
            summary = one_step._layout_summary(selected, layout, split)
            layouts[split].append(summary)
            free = summary["free_forward"]
            blocked = summary["blocked_forward"]
            contacts = summary["interact"]
            critical = {
                "free_step1_action2": {
                    **free,
                    "failure_ratio_ge_1": (
                        free["prediction_to_persistence_ratio"] is not None
                        and free["prediction_to_persistence_ratio"] >= 1.0
                    ),
                },
                "blocked_step0_action2": {
                    **blocked,
                    "material_mse_threshold": summary["material_mse_threshold"],
                    "failure_above_material_threshold": bool(
                        blocked["predicted_vs_actual_next_z_mse"]
                        > summary["material_mse_threshold"]
                    ),
                },
                "contacts_step0_2_action3": [
                    {
                        **row,
                        "failure_ratio_ge_1": (
                            row["prediction_to_persistence_ratio"] is not None
                            and row["prediction_to_persistence_ratio"] >= 1.0
                        ),
                    }
                    for row in contacts
                ],
            }
            per_layout[split][layout] = {
                "all_rows": selected,
                "critical": critical,
                "exp148_layout_summary": summary,
            }
    return {
        "layouts": layouts,
        "splits": {
            split: one_step._aggregate_split(layouts[split])
            for split in one_step.SPLITS
        },
        "per_layout": per_layout,
    }


def _critical_counts(summary):
    by_split = {}
    totals = {"free_failures": 0, "contact_row_failures": 0, "blocked_failures": 0}
    for split in one_step.SPLITS:
        counts = {key: 0 for key in totals}
        for payload in summary["per_layout"][split].values():
            critical = payload["critical"]
            counts["free_failures"] += int(
                critical["free_step1_action2"]["failure_ratio_ge_1"]
            )
            counts["blocked_failures"] += int(
                critical["blocked_step0_action2"]["failure_above_material_threshold"]
            )
            counts["contact_row_failures"] += sum(
                int(row["failure_ratio_ge_1"])
                for row in critical["contacts_step0_2_action3"]
            )
        by_split[split] = counts
        for key, value in counts.items():
            totals[key] += value
    return {
        "by_split": by_split,
        "total": totals,
        "all_critical_rows_pass": not any(totals.values()),
        "status": "descriptive_only_not_registered_gate",
    }


def _classification_errors(candidate_rows):
    false_negatives = []
    false_positives = []
    for row in candidate_rows:
        payload = {
            key: row[key]
            for key in (
                "split", "layout", "step", "real_history", "canonical_action",
                "action", "action_name", "rgb_changed", "event_probability",
                "predicted_change", "predicted_vs_actual_next_z_mse",
                "persistence_vs_actual_next_z_mse",
            )
        }
        if row["rgb_changed"] and not row["predicted_change"]:
            false_negatives.append(payload)
        elif not row["rgb_changed"] and row["predicted_change"]:
            false_positives.append(payload)
    return {
        "false_negative_count": len(false_negatives),
        "false_positive_count": len(false_positives),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def _max_difference(rows_a, rows_b, field):
    return max(abs(float(a[field]) - float(b[field])) for a, b in zip(rows_a, rows_b, strict=True))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp168-dir", type=Path, default=DEFAULT_EXP168_DIR)
    parser.add_argument("--exp169-dir", type=Path, default=DEFAULT_EXP169_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-interval", type=float, default=30.0)
    return parser


def _argv(argv):
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.out.exists():
        unexpected = [item for item in args.out.iterdir() if item.name != "run.log"]
        if unexpected:
            raise FileExistsError(f"output directory is not empty: {unexpected}")
    else:
        args.out.mkdir(parents=True)
    started = time.monotonic()
    command = os.environ.get("EXP170_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv), "exact_command": command, "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "exp168_dir": str(args.exp168_dir), "exp169_dir": str(args.exp169_dir),
        "status": "running", "exit_code": None, "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("load_artifacts", 0, 6)
            result168 = _read_json(args.exp168_dir / "results.json", "exp168 results")
            journal.update("load_artifacts", 1, 6)
            manifest168 = _read_json(args.exp168_dir / "manifest.json", "exp168 manifest")
            journal.update("load_artifacts", 2, 6)
            rows168 = _read_rows(
                args.exp168_dir / result168["artifacts"]["rows"], "exp168 rows"
            )
            journal.update("load_artifacts", 3, 6)
            result169 = _read_json(args.exp169_dir / "results.json", "exp169 results")
            journal.update("load_artifacts", 4, 6)
            manifest169 = _read_json(args.exp169_dir / "manifest.json", "exp169 manifest")
            candidate_rows = _read_rows(
                args.exp169_dir / result169["artifacts"]["rows"], "exp169 candidate rows"
            )
            frozen_rows = _read_rows(
                args.exp169_dir / result169["artifacts"]["frozen_vector_rows"],
                "exp169 frozen rows",
            )
            base_rows = _read_rows(
                args.exp169_dir / result169["artifacts"]["base_rows"],
                "exp169 base rows",
            )
            journal.update("load_artifacts", 6, 6)

            journal.update("align_rows", 0, 120)
            keys_equal = (
                [row_key(row) for row in rows168]
                == [row_key(row) for row in frozen_rows]
                == [row_key(row) for row in candidate_rows]
                == [row_key(row) for row in base_rows]
            )
            if not keys_equal:
                raise AssertionError("canonical artifact row keys differ")
            persistence_max_diff = max(
                _max_difference(rows168, frozen_rows, "persistence_vs_actual_next_z_mse"),
                _max_difference(rows168, candidate_rows, "persistence_vs_actual_next_z_mse"),
                _max_difference(rows168, base_rows, "persistence_vs_actual_next_z_mse"),
            )
            if persistence_max_diff > 1e-12:
                raise AssertionError("canonical persistence targets differ")
            frozen_max_mse_diff = _max_difference(
                rows168, frozen_rows, "predicted_vs_actual_next_z_mse"
            )
            candidate_base_max_mse_diff = _max_difference(
                candidate_rows, base_rows, "predicted_vs_actual_next_z_mse"
            )
            if frozen_max_mse_diff > 1e-7 or candidate_base_max_mse_diff > 1e-12:
                raise AssertionError("stored reference rows do not reproduce")
            oracle_rows = []
            writer = core.TraceWriter(args.out / "oracle_event_rows.jsonl")
            try:
                for index, (event_row, frozen_row) in enumerate(
                    zip(candidate_rows, frozen_rows, strict=True), start=1
                ):
                    row = oracle_event_row(event_row, frozen_row)
                    writer.write(row)
                    oracle_rows.append(row)
                    journal.update("align_rows", index, 120)
            finally:
                writer.close()

            journal.update("summarize", 0, 4)
            exp168_summary = _summarize_rows(rows168)
            journal.update("summarize", 1, 4)
            exp169_summary = _summarize_rows(candidate_rows)
            journal.update("summarize", 2, 4)
            oracle_summary = _summarize_rows(oracle_rows)
            journal.update("summarize", 3, 4)
            errors = _classification_errors(candidate_rows)
            critical = _critical_counts(oracle_summary)
            journal.update("summarize", 4, 4)

            matching = {
                "canonical_exp168_dir": args.exp168_dir == DEFAULT_EXP168_DIR,
                "canonical_exp169_dir": args.exp169_dir == DEFAULT_EXP169_DIR,
                "exp168_result_head": result168.get("analysis_git_head") == EXPECTED_EXP168_HEAD,
                "exp168_manifest_head": manifest168.get("analysis_git_head") == EXPECTED_EXP168_HEAD,
                "exp169_result_head": result169.get("analysis_git_head") == EXPECTED_EXP169_HEAD,
                "exp169_manifest_head": manifest169.get("analysis_git_head") == EXPECTED_EXP169_HEAD,
                "exp168_completed_exact": result168.get("status") == "completed" and result168.get("exact_protocol") is True and manifest168.get("status") == "completed" and manifest168.get("exit_code") == 0,
                "exp169_completed_exact": result169.get("status") == "completed" and result169.get("exact_protocol") is True and manifest169.get("status") == "completed" and manifest169.get("exit_code") == 0,
                "ordered_row_keys_equal": keys_equal,
                "persistence_targets_equal": persistence_max_diff <= 1e-12,
                "frozen_rows_equal_exp168": frozen_max_mse_diff <= 1e-7,
                "candidate_rows_equal_base_trace": candidate_base_max_mse_diff <= 1e-12,
                "exp168_rows_match_result": _metric_match(
                    exp168_summary, result168["one_step"]
                ),
                "exp169_rows_match_result": _metric_match(
                    exp169_summary, result169["one_step"]
                ),
            }
            exact_protocol = all(matching.values())
            oracle_source_gate = nonlinear.nonlinear_probe_gate(
                oracle_summary["splits"]["source"],
                oracle_summary["splits"]["source"], exact_protocol,
            )
            oracle_unseen_gate = nonlinear.nonlinear_probe_gate(
                oracle_summary["splits"]["unseen"],
                oracle_summary["splits"]["unseen"], exact_protocol,
            )
            registered = {
                "source_split_gate": result169["source_split_gate"],
                "unseen_split_gate": result169["unseen_split_gate"],
                "event_mode_vector_gate": result169["event_mode_vector_gate"],
                "status": "preserved_from_exp169_not_recomputed_or_changed",
            }
            oracle_registered = {
                "source_split_gate": oracle_source_gate,
                "unseen_split_gate": oracle_unseen_gate,
                "oracle_event_gate": bool(oracle_source_gate and oracle_unseen_gate),
                "predicate": "original median gate; contact0 blocked0 median free<1 interact<1",
            }
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "artifact-only event-mode row audit",
                "interpretation_limit": "No training, architecture, composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "exact_command": command, "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "row_alignment": {
                    "rows": len(oracle_rows),
                    "key_fields": list(ROW_FIELDS),
                    "ordered_keys_equal": keys_equal,
                    "persistence_max_abs_diff": persistence_max_diff,
                    "exp168_vs_exp169_frozen_max_abs_mse_diff": frozen_max_mse_diff,
                    "candidate_vs_base_max_abs_mse_diff": candidate_base_max_mse_diff,
                },
                "exp169_registered_gate": registered,
                "oracle_event_registered_median_gate": oracle_registered,
                "oracle_event_all_critical_rows": critical,
                "predicted_event_errors": errors,
                "exp168_frozen_vector_splits": exp168_summary["splits"],
                "exp169_candidate_splits": exp169_summary["splits"],
                "oracle_event_splits": oracle_summary["splits"],
                "oracle_event_per_layout": oracle_summary["per_layout"],
                "controls": {
                    "training": False, "new_weights": False,
                    "source_artifacts_only": True,
                    "oracle_event_uses_actual_rgb_change": True,
                    "oracle_event_changed_uses_frozen_vector": True,
                    "oracle_event_nochange_mse_exact_zero": True,
                    "registered_exp169_gate_mutated": False,
                    "stricter_check_descriptive_only": True,
                },
                "artifacts": {
                    "rows": "oracle_event_rows.jsonl",
                    "progress": "progress.jsonl", "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            core._write_json(args.out / "results.json", result)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            manifest.update(
                status="completed", exit_code=0, exit_status=0,
                exact_protocol=exact_protocol, protocol_match=matching,
                runtime_seconds=time.monotonic() - started,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 2, 2, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = temporal._exit_code(error)
            manifest.update(
                status="failed", exit_code=code, exit_status=code,
                runtime_seconds=time.monotonic() - started,
                error=f"{type(error).__name__}: {error}",
            )
            core._write_json(args.out / "manifest.json", manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
