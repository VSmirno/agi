"""Checkpoint-only source versus unseen one-step Push-1 diagnostic."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import statistics
import sys

import numpy as np
import torch

from experiments.exp145_physics_transfer import (
    SOURCE_LAYOUTS,
    TARGET_LAYOUTS,
    _adapter,
)
from experiments.exp147_rollout_localization import (
    ACTION_COUNT,
    MATERIAL_FRACTION_OF_MOVEMENT,
    NUMERIC_MSE_FLOOR,
    ProgressJournal,
    _load_checkpoint,
    _teacher_forced_next,
    _exit_code,
    _progress_interval,
)
from snks.env.core_grid import GRID_ACTIONS
from snks.pipeline import core_experiment as core


SEED = 20000
PUSH_ONE_CONTINUATION = (3, 2, 3)
SOURCE_PREFIX_LENGTH = 4
TARGET_PREFIX_LENGTH = 5
HORIZON = 3
SPLITS = ("source", "unseen")


def _layout_specs():
    return {
        "source": {
            name: (layout, actions)
            for name, (layout, actions) in SOURCE_LAYOUTS.items()
        },
        "unseen": {
            name: (layout, actions, _push_two)
            for name, (layout, actions, _push_two) in TARGET_LAYOUTS.items()
        },
    }


def _finite_median(values):
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(value)
    ]
    return float(statistics.median(finite)) if finite else None


def _aggregate_split(layout_summaries):
    """Aggregate deterministic per-layout failure evidence and medians."""

    summaries = list(layout_summaries)
    return {
        "layout_count": len(summaries),
        "contact_failure_layouts": sum(
            bool(summary["contact_failure"]) for summary in summaries
        ),
        "blocked_noop_failure_layouts": sum(
            bool(summary["blocked_noop_failure"]) for summary in summaries
        ),
        "medians": {
            "interact_prediction_persistence_ratio": _finite_median(
                ratio
                for summary in summaries
                for ratio in summary["interact_prediction_persistence_ratios"]
            ),
            "free_forward_prediction_persistence_ratio": _finite_median(
                summary["free_forward_prediction_persistence_ratio"]
                for summary in summaries
            ),
            "blocked_forward_prediction_mse": _finite_median(
                summary["blocked_forward_prediction_mse"]
                for summary in summaries
            ),
        },
    }


def _outcome_label(source_summary, unseen_summary):
    """Return the conservative split-level evidence label."""

    def all_failed(summary):
        return (
            summary["layout_count"] > 0
            and summary["contact_failure_layouts"] == summary["layout_count"]
            and summary["blocked_noop_failure_layouts"] == summary["layout_count"]
        )

    def none_failed(summary):
        return (
            summary["contact_failure_layouts"] == 0
            and summary["blocked_noop_failure_layouts"] == 0
        )

    if all_failed(source_summary) and all_failed(unseen_summary):
        return "shared_one_step_failure_evidence"
    if none_failed(source_summary) and all_failed(unseen_summary):
        return "unseen_only_failure_evidence"
    return "mixed_or_inconclusive"


def _fresh_real_fork(layout, history: tuple[int, ...], action: int, seed: int):
    adapter = _adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
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


def _replay_prefix(model, layout, prefix: tuple[int, ...], seed: int):
    adapter = _adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        state = model.initial(observation)
        for action in prefix:
            transition = adapter.step(action)
            if transition.terminated or transition.truncated:
                raise RuntimeError("canonical prefix unexpectedly ended the episode")
            prediction = model.step(
                state,
                torch.tensor([action], device=state.z.device, dtype=torch.long),
            )
            actual = model.initial(transition.after)
            state = _teacher_forced_next(prediction, actual)
        return state, adapter.diagnostic_snapshot()
    finally:
        adapter.close()


def _validate_protocol(split: str, layout_name: str, layout, actions, seed: int):
    prefix_length = (
        SOURCE_PREFIX_LENGTH if split == "source" else TARGET_PREFIX_LENGTH
    )
    prefix = tuple(actions[:prefix_length])
    continuation = tuple(actions[prefix_length:])
    if continuation != PUSH_ONE_CONTINUATION:
        raise RuntimeError(
            f"{split}/{layout_name} continuation must be {PUSH_ONE_CONTINUATION}, "
            f"got {continuation}"
        )
    adapter = _adapter(layout, 1, seed, 32)
    try:
        adapter.reset(seed)
        for action in prefix:
            transition = adapter.step(action)
            if transition.terminated or transition.truncated:
                raise RuntimeError(
                    f"{split}/{layout_name} canonical prefix is terminal"
                )
    finally:
        adapter.close()
    return prefix, continuation


def _layout_summary(rows, layout_name, split):
    canonical_rows = [
        row for row in rows if row["action"] == row["canonical_action"]
    ]
    changed_persistence = [
        row["persistence_vs_actual_next_z_mse"]
        for row in canonical_rows
        if row["rgb_changed"]
        and row["persistence_vs_actual_next_z_mse"] > 0.0
    ]
    if not changed_persistence:
        raise RuntimeError(f"{split}/{layout_name} has no changed canonical transition")
    material_threshold = max(
        NUMERIC_MSE_FLOOR,
        MATERIAL_FRACTION_OF_MOVEMENT * float(statistics.median(changed_persistence)),
    )
    interacts = [
        row for row in canonical_rows if row["step"] in (0, 2)
    ]
    if len(interacts) != 2 or any(row["canonical_action"] != 3 for row in interacts):
        raise RuntimeError(f"{split}/{layout_name} canonical interact protocol is invalid")
    free_forward = next(
        row
        for row in canonical_rows
        if row["step"] == 1 and row["canonical_action"] == 2
    )
    blocked_forward = next(
        row for row in rows if row["step"] == 0 and row["action"] == 2
    )
    if blocked_forward["rgb_changed"]:
        raise RuntimeError(f"{split}/{layout_name} step-0 forward was not blocked")
    contact_failure = any(
        row["predicted_vs_actual_next_z_mse"]
        >= row["persistence_vs_actual_next_z_mse"]
        for row in interacts
    )
    blocked_noop_failure = (
        not blocked_forward["rgb_changed"]
        and blocked_forward["predicted_vs_actual_next_z_mse"] > material_threshold
    )
    return {
        "split": split,
        "layout": layout_name,
        "contact_failure": bool(contact_failure),
        "blocked_noop_failure": bool(blocked_noop_failure),
        "material_mse_threshold": material_threshold,
        "interact": interacts,
        "free_forward": free_forward,
        "blocked_forward": blocked_forward,
        "interact_prediction_persistence_ratios": [
            row["prediction_to_persistence_ratio"] for row in interacts
        ],
        "free_forward_prediction_persistence_ratio": free_forward[
            "prediction_to_persistence_ratio"
        ],
        "blocked_forward_prediction_mse": blocked_forward[
            "predicted_vs_actual_next_z_mse"
        ],
    }


@torch.inference_mode()
def _diagnose(model, journal: ProgressJournal, rows_path: Path):
    layout_specs = _layout_specs()
    total = (
        sum(len(layouts) for layouts in layout_specs.values())
        * HORIZON
        * ACTION_COUNT
    )
    completed = 0
    layout_summaries = {split: [] for split in SPLITS}
    writer = core.TraceWriter(rows_path)
    try:
        for split in SPLITS:
            for layout_name, spec in layout_specs[split].items():
                layout, actions = spec[:2]
                prefix, continuation = _validate_protocol(
                    split, layout_name, layout, actions, SEED
                )
                state, prefix_diagnostic = _replay_prefix(
                    model, layout, prefix, SEED
                )
                layout_rows = []
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    for action in range(ACTION_COUNT):
                        before, after, diagnostic = _fresh_real_fork(
                            layout, history, action, SEED
                        )
                        prediction = model.step(
                            state,
                            torch.tensor(
                                [action], device=state.z.device, dtype=torch.long
                            ),
                        )
                        actual = model.initial(after)
                        persistence_mse = float((state.z - actual.z).square().mean())
                        prediction_mse = float(
                            (prediction.next_state.z - actual.z).square().mean()
                        )
                        row = {
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
                            "predicted_vs_actual_next_z_mse": prediction_mse,
                            "persistence_vs_actual_next_z_mse": persistence_mse,
                            "prediction_to_persistence_ratio": (
                                prediction_mse / persistence_mse
                                if persistence_mse > 0.0
                                else None
                            ),
                        }
                        writer.write(row)
                        layout_rows.append(row)
                        completed += 1
                        journal.update(
                            "teacher_forced",
                            completed,
                            total,
                            split=split,
                            layout=layout_name,
                            step=step,
                            action=action,
                            action_name=GRID_ACTIONS[action],
                        )
                        if action == canonical_action:
                            canonical_prediction = prediction
                            canonical_actual = actual
                    if canonical_prediction is None or canonical_actual is None:
                        raise RuntimeError("canonical action was not evaluated")
                    state = _teacher_forced_next(canonical_prediction, canonical_actual)
                summary = _layout_summary(
                    layout_rows,
                    layout_name,
                    split,
                )
                summary["prefix"] = list(prefix)
                summary["continuation"] = list(continuation)
                summary["prefix_diagnostic"] = prefix_diagnostic
                layout_summaries[split].append(summary)
    finally:
        writer.close()
    source = _aggregate_split(layout_summaries["source"])
    unseen = _aggregate_split(layout_summaries["unseen"])
    return {
        "status": "completed",
        "claim": "bounded checkpoint-only source versus unseen one-step failure evidence",
        "protocol": {
            "push_distance": 1,
            "seed": SEED,
            "source_prefix_length": SOURCE_PREFIX_LENGTH,
            "target_prefix_length": TARGET_PREFIX_LENGTH,
            "continuation": list(PUSH_ONE_CONTINUATION),
            "teacher_forcing": "real z/sensors/mask with carried predicted hidden",
            "fresh_environment_replay_per_action_fork": True,
            "push_2_run": False,
            "rows": total,
        },
        "layouts": layout_summaries,
        "splits": {"source": source, "unseen": unseen},
        "outcome_label": _outcome_label(source, unseen),
        "interpretation_limit": (
            "Diagnostic evidence only; this split does not prove representation capacity."
        ),
        "artifacts": {"rows": rows_path.name},
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "argv": list(sys.orig_argv),
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "checkpoint_git_head": None,
        "checkpoint": str(args.checkpoint),
        "arguments": core._jsonable(vars(args)),
    }
    with ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("initialize", 0, 1, operation="safe_checkpoint_load")
            model, ordered, checkpoint_head, metadata = _load_checkpoint(
                args.checkpoint
            )
            del ordered
            manifest["checkpoint_git_head"] = checkpoint_head
            manifest["checkpoint_metadata"] = metadata
            journal.update("initialize", 1, 1, device=metadata["device"])
            results = _diagnose(model, journal, args.out / "diagnostic_rows.jsonl")
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
                {
                    **manifest,
                    "exit_code": 0,
                    "exit_status": 0,
                    "status": "completed",
                },
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
                    "exit_status": _exit_code(error),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-interval", type=_progress_interval, default=30)
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
