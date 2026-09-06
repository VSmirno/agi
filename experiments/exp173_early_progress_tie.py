"""Re-evaluate exp172 with an exact-tie early-progress ordering."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
import time

import torch

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp172_observation_only_transition as observation_only
from experiments.exp172_behavior_eval import evaluate_behavior
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_EXP172_DIR = Path(
    "output_to_user/core/exp172-observation-only-transition-001"
)
EXPECTED_EXP172_HEAD = "728092fcbd75f90f7a01d00de60608d26ba5f220"
WEST_TIE_DECISION = 6
DELAYED = (2, 3, 0)
IMMEDIATE = (3, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, role: str):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {role}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must be a JSON object")
    return payload


def _load_heads(path: Path, baseline):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"could not safely load exp172 heads: {error}") from error
    required = {
        "format_version",
        "analysis_git_head",
        "baseline_checkpoint_git_head",
        "objective",
        "z_dim",
        "h_dim",
        "ensemble_size",
        "hidden_width",
        "train_action_counts",
        "class_weights",
        "vector_probe_state_dict",
        "event_probe_state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("exp172 checkpoint fields mismatch")
    expected = {
        "format_version": 1,
        "analysis_git_head": EXPECTED_EXP172_HEAD,
        "baseline_checkpoint_git_head": frozen.EXPECTED_BASELINE_HEAD,
        "z_dim": baseline.encoder.z_dim,
        "h_dim": baseline.h_dim,
        "ensemble_size": baseline.heads,
        "hidden_width": observation_only.HIDDEN_WIDTH,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("exp172 checkpoint metadata mismatch")
    parameter = next(baseline.parameters())
    vector = observation_only.ObservationVectorDelta(
        baseline.encoder.z_dim, baseline.h_dim, baseline.heads
    ).to(device=parameter.device, dtype=parameter.dtype)
    event = observation_only.ObservationEventHead(
        baseline.encoder.z_dim, baseline.h_dim
    ).to(device=parameter.device, dtype=parameter.dtype)
    vector.load_state_dict(payload["vector_probe_state_dict"], strict=True)
    event.load_state_dict(payload["event_probe_state_dict"], strict=True)
    vector.eval().requires_grad_(False)
    event.eval().requires_grad_(False)
    return observation_only._installed_model(baseline, vector, event), payload


def _layout(behavior, arm: str, name: str):
    return next(
        row for row in behavior[arm]["layouts"] if row["layout"] == name
    )


def _candidate(trace, actions):
    return next(row for row in trace if tuple(row["actions"]) == tuple(actions))


def _tie_evidence(old_evaluation, new_evaluation):
    old = _layout(old_evaluation["behavior"], "actual", "west_row3")
    if len(old["decision_traces"]) <= WEST_TIE_DECISION:
        raise ValueError("exp172 west_row3 trace is missing decision 6")
    old_trace = old["decision_traces"][WEST_TIE_DECISION]
    old_delayed = _candidate(old_trace, DELAYED)
    old_immediate = _candidate(old_trace, IMMEDIATE)
    exact_tie = old_delayed["cost"] == old_immediate["cost"]
    if not exact_tie or old["selected_plans"][WEST_TIE_DECISION] != list(DELAYED):
        raise ValueError("exp172 artifact does not contain the reproduced exact tie")

    new = _layout(new_evaluation["behavior"], "actual", "west_row3")
    if len(new["decision_traces"]) <= WEST_TIE_DECISION:
        raise ValueError("exp173 west_row3 ended before reproduced decision 6")
    new_trace = new["decision_traces"][WEST_TIE_DECISION]
    new_delayed = _candidate(new_trace, DELAYED)
    new_immediate = _candidate(new_trace, IMMEDIATE)
    selected = new["selected_plans"][WEST_TIE_DECISION]
    return {
        "layout": "west_row3",
        "zero_based_decision": WEST_TIE_DECISION,
        "exp172": {
            "selected": old["selected_plans"][WEST_TIE_DECISION],
            "delayed_endpoint_cost": old_delayed["cost"],
            "immediate_endpoint_cost": old_immediate["cost"],
            "exact_endpoint_tie": exact_tie,
            "success": old["success"],
            "steps": old["steps"],
        },
        "exp173": {
            "selected": selected,
            "delayed_endpoint_cost": new_delayed["cost"],
            "delayed_prefix_costs": new_delayed["prefix_costs"],
            "immediate_endpoint_cost": new_immediate["cost"],
            "immediate_prefix_costs": new_immediate["prefix_costs"],
            "success": new["success"],
            "steps": new["steps"],
        },
        "tie_corrected": bool(
            new_delayed["cost"] == new_immediate["cost"]
            and selected == list(IMMEDIATE)
            and new_immediate["prefix_costs"] < new_delayed["prefix_costs"]
        ),
    }


def _comparison(old_evaluation, new_evaluation):
    compared = {}
    for arm in ("original", "learned", "actual"):
        old = old_evaluation["behavior"][arm]
        new = new_evaluation["behavior"][arm]
        old_layouts = {row["layout"]: row for row in old["layouts"]}
        new_layouts = {row["layout"]: row for row in new["layouts"]}
        compared[arm] = {
            "exp172_successes": old["summary"]["successes"],
            "exp173_successes": new["summary"]["successes"],
            "success_delta": (
                new["summary"]["successes"] - old["summary"]["successes"]
            ),
            "changed_action_traces": [
                {
                    "layout": name,
                    "exp172_success": old_layouts[name]["success"],
                    "exp173_success": new_layouts[name]["success"],
                    "exp172_actions": old_layouts[name]["actions"],
                    "exp173_actions": new_layouts[name]["actions"],
                }
                for name in old_layouts
                if old_layouts[name]["actions"] != new_layouts[name]["actions"]
            ],
        }
    return compared


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/core_pilot.yaml"))
    parser.add_argument(
        "--baseline-checkpoint", type=Path, default=frozen.DEFAULT_BASELINE
    )
    parser.add_argument("--exp172-dir", type=Path, default=DEFAULT_EXP172_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--progress-interval",
        type=temporal._progress_interval,
        default=30,
    )
    return parser


def _argv(argv):
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frozen._prepare_output(args.out)
    started = time.monotonic()
    command = os.environ.get("EXP173_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    head_path = args.exp172_dir / "observation_only_heads.pt"
    old_results_path = args.exp172_dir / "results.json"
    input_hashes = {
        "baseline_checkpoint_sha256": _sha256(args.baseline_checkpoint),
        "exp172_heads_sha256": _sha256(head_path),
        "exp172_results_sha256": _sha256(old_results_path),
    }
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "inputs": {
            "baseline_checkpoint": str(args.baseline_checkpoint),
            "exp172_heads": str(head_path),
            "exp172_results": str(old_results_path),
            **input_hashes,
        },
        "training": False,
        "tie_break": (
            "exact endpoint ties only: lexicographic prefix costs, then actions"
        ),
        "status": "running",
        "exit_code": None,
        "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 3, operation="safe_baseline_load")
            baseline, ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            ordered.eval().requires_grad_(False)
            journal.update("initialize", 1, 3, operation="safe_exp172_head_load")
            model, checkpoint = _load_heads(head_path, baseline)
            journal.update("initialize", 2, 3, operation="load_exp172_results")
            old_results = _read_json(old_results_path, "exp172 results")
            if (
                old_results.get("status") != "completed"
                or old_results.get("analysis_git_head") != EXPECTED_EXP172_HEAD
            ):
                raise ValueError("exp172 results identity mismatch")
            config = replace(
                load_core_config(args.config),
                seed=0,
                z_dim=baseline.encoder.z_dim,
                h_dim=baseline.h_dim,
                ensemble_size=baseline.heads,
                burn_in=0,
            )
            journal.update("initialize", 3, 3, device=str(next(baseline.parameters()).device))
            evaluation = evaluate_behavior(
                model,
                baseline,
                ordered,
                config,
                journal,
                args.out,
                early_progress_tie_break=True,
            )
            comparison = _comparison(old_results["evaluation"], evaluation)
            tie_evidence = _tie_evidence(old_results["evaluation"], evaluation)
            final_hashes = {
                "baseline_checkpoint_sha256": _sha256(args.baseline_checkpoint),
                "exp172_heads_sha256": _sha256(head_path),
                "exp172_results_sha256": _sha256(old_results_path),
            }
            inputs_unchanged = final_hashes == input_hashes
            if not inputs_unchanged:
                raise AssertionError("exp173 input artifacts changed during evaluation")
            result = {
                "status": "completed",
                "claim": "exact-tie early-progress development comparison",
                "analysis_git_head": manifest["analysis_git_head"],
                "training": False,
                "protocol": {
                    "episodes": 24,
                    "exp172_behavior_protocol_reused": True,
                    "endpoint_cost_primary": True,
                    "exact_ties_only": True,
                    "prefix_costs_lexicographic": True,
                    "action_ids_final_tie_break": True,
                    "path_cost_sum": False,
                    "epsilon_or_weight": False,
                    "production_core_planner_changed": False,
                },
                "source": {
                    "exp172_analysis_git_head": old_results["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "checkpoint_metadata": {
                        "z_dim": checkpoint["z_dim"],
                        "h_dim": checkpoint["h_dim"],
                        "ensemble_size": checkpoint["ensemble_size"],
                    },
                    **input_hashes,
                    "inputs_unchanged": inputs_unchanged,
                },
                "tie_evidence": tie_evidence,
                "comparison": comparison,
                "evaluation": evaluation,
                "interpretation_limit": (
                    "One development-only tie-break run on the same layouts and "
                    "saved heads; remaining failures are not model-tuning evidence."
                ),
                "artifacts": {
                    "behavior_rows": "behavior_rows.jsonl",
                    "canonical_rollout_rows": "canonical_rollout_rows.jsonl",
                    "progress": "progress.jsonl",
                    "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            journal.update("artifacts", 0, 2, operation="write_results")
            core._write_json(args.out / "results.json", result)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            manifest.update(
                status="completed",
                exit_code=0,
                exit_status=0,
                runtime_seconds=time.monotonic() - started,
                baseline_checkpoint_git_head=baseline_head,
                input_hashes_unchanged=inputs_unchanged,
                tie_corrected=tie_evidence["tie_corrected"],
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 2, 2, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = temporal._exit_code(error)
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
