"""Checkpoint-only coverage and matched-context audit for exp169's event head."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import torch

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp148_source_target_one_step as one_step
from experiments import exp150_residual_dynamics as residual
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp161_amplitude_input_probe as input_probe
from experiments import exp165_relational_pose_probe as pose_probe
from experiments import exp169_event_mode_vector_transition as event_mode
from experiments import exp170_event_mode_row_audit as row_audit
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_EXP169_DIR = Path(
    "output_to_user/core/exp169-event-mode-vector-transition-001"
)
EXPECTED_EXP169_HEAD = row_audit.EXPECTED_EXP169_HEAD
EVENT_BOUNDARY = event_mode.EVENT_BOUNDARY
PROTOCOL = dict(residual.PROTOCOL)
OBJECTIVE = {
    "coverage": (
        "exact integer [box-agent, goal-box, orientation] BEFORE pose plus action; "
        "train side of the established 75/25 episode split only"
    ),
    "matching": "source donor to unseen recipient by exact pose/action/canonical step",
    "interventions": (
        "hold recipient pose/action; replace hidden only, z only, or z+hidden"
    ),
    "training": False,
}


def _json_pose_key(value) -> tuple[int, int, int, int, int]:
    try:
        key = tuple(value)
    except TypeError as error:
        raise ValueError("pose_key must be an iterable of five integers") from error
    if len(key) != 5 or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in key
    ):
        raise ValueError("pose_key must contain five integers")
    result = tuple(int(item) for item in key)
    if result[-1] not in range(4):
        raise ValueError("pose orientation must be in [0,3]")
    return result


def exact_pose_key(snapshot: Mapping, orientation: int):
    """Return collision-free integer relative offsets and orientation."""

    if isinstance(orientation, bool) or not isinstance(
        orientation, (int, np.integer)
    ):
        raise ValueError("orientation must be an integer in [0,3]")
    try:
        coordinates = (
            *snapshot["agent_pos"], *snapshot["box_pos"], *snapshot["goal_pos"]
        )
        agent_x, agent_y = snapshot["agent_pos"]
        box_x, box_y = snapshot["box_pos"]
        goal_x, goal_y = snapshot["goal_pos"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("snapshot requires 2D agent, box, and goal positions") from error
    if len(coordinates) != 6 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer))
        for value in coordinates
    ):
        raise ValueError("snapshot positions must contain integers")
    return _json_pose_key((
        int(box_x - agent_x), int(box_y - agent_y),
        int(goal_x - box_x), int(goal_y - box_y), int(orientation),
    ))


def coverage_counts(
    train_rows: Iterable[Mapping], canonical_rows: Iterable[Mapping]
) -> list[dict]:
    """Attach exact pose/action same- and opposite-event train counts."""

    counts = Counter()
    for row in train_rows:
        key = _json_pose_key(row["pose_key"])
        action = int(row["action"])
        label = bool(row["rgb_changed"])
        counts[(key, action, label)] += 1

    covered = []
    for row in canonical_rows:
        key = _json_pose_key(row["pose_key"])
        action = int(row["action"])
        label = bool(row["rgb_changed"])
        same = counts[(key, action, label)]
        opposite = counts[(key, action, not label)]
        covered.append({
            **row,
            "pose_key": list(key),
            "train_pose_action_count": same + opposite,
            "same_label_train_count": same,
            "opposite_label_train_count": opposite,
        })
    return covered


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
    if not isinstance(payload, Mapping):
        raise ValueError(f"{role} must be a JSON object")
    return payload


def _read_reference(exp169_dir: Path):
    result = _read_json(exp169_dir / "results.json", "exp169 results")
    if result.get("analysis_git_head") != EXPECTED_EXP169_HEAD:
        raise ValueError("exp169 results analysis head mismatch")
    try:
        rows_path = exp169_dir / result["artifacts"]["rows"]
        checkpoint_path = exp169_dir / result["artifacts"]["checkpoint"]
        rows = [
            json.loads(line) for line in rows_path.read_text().splitlines() if line
        ]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp169 artifacts: {error}") from error
    if len(rows) != 120 or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("exp169 reference must contain exactly 120 object rows")
    return result, rows, checkpoint_path


def _load_event_checkpoint(path: Path, baseline):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"could not safely load exp169 checkpoint: {error}") from error
    required = {
        "format_version", "analysis_git_head", "baseline_checkpoint_git_head",
        "exp168_checkpoint_git_head", "objective", "z_dim", "h_dim",
        "pose_dim", "hidden_width", "train_action_counts", "class_weights",
        "event_probe_state_dict",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("exp169 checkpoint fields mismatch")
    expected = {
        "format_version": 1,
        "analysis_git_head": EXPECTED_EXP169_HEAD,
        "baseline_checkpoint_git_head": EXPECTED_BASELINE_HEAD,
        "z_dim": baseline.encoder.z_dim,
        "h_dim": baseline.h_dim,
        "pose_dim": event_mode.POSE_DIM,
        "hidden_width": event_mode.HIDDEN_WIDTH,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("exp169 checkpoint metadata mismatch")
    state = payload["event_probe_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("exp169 event state must contain tensors only")
    parameter = next(baseline.parameters())
    probe = event_mode.RelationalPoseEventHead(
        baseline.encoder.z_dim, baseline.h_dim
    ).to(device=parameter.device, dtype=parameter.dtype)
    probe.load_state_dict(state, strict=True)
    return probe.eval().requires_grad_(False), payload


def _train_coverage_rows(train_episodes, args, journal):
    total = sum(map(len, train_episodes.values()))
    rows = []
    completed = 0
    journal.update("train_coverage", 0, total)
    for layout_index, (layout_name, (layout, _actions)) in enumerate(
        temporal.SOURCE_LAYOUTS.items()
    ):
        for offset, episode in enumerate(train_episodes[layout_name]):
            seed = 10000 + layout_index * 100000 + offset
            if not episode.uid.endswith(f":{seed}"):
                raise AssertionError(f"unexpected episode UID/seed: {episode.uid}")
            adapter = temporal._adapter(layout, 1, seed, args.collection_steps)
            try:
                observation = adapter.reset(seed)
                for step, transition in enumerate(episode.transitions):
                    if not np.array_equal(observation.rgb, transition.before.rgb):
                        raise AssertionError(
                            f"{episode.uid}/{step} before observation mismatch"
                        )
                    pose_key = exact_pose_key(
                        adapter.diagnostic_snapshot(), adapter.world.agent_dir
                    )
                    replayed = adapter.step(transition.action)
                    if replayed.action != transition.action:
                        raise AssertionError(f"{episode.uid}/{step} action mismatch")
                    if not np.array_equal(replayed.before.rgb, transition.before.rgb):
                        raise AssertionError(f"{episode.uid}/{step} replay before mismatch")
                    if not np.array_equal(replayed.after.rgb, transition.after.rgb):
                        raise AssertionError(f"{episode.uid}/{step} replay after mismatch")
                    rows.append({
                        "pose_key": pose_key,
                        "action": int(transition.action),
                        "rgb_changed": bool(
                            not np.array_equal(transition.before.rgb, transition.after.rgb)
                        ),
                    })
                    observation = replayed.after
            finally:
                adapter.close()
            completed += 1
            journal.update(
                "train_coverage", completed, total,
                layout=layout_name, episode=offset,
            )
    return rows


def _fresh_context_fork(layout, history, action, seed):
    adapter = temporal._adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        for previous in history:
            transition = adapter.step(previous)
            if transition.terminated or transition.truncated:
                raise RuntimeError("real history unexpectedly ended before the fork")
            observation = transition.after
        snapshot = adapter.diagnostic_snapshot()
        orientation = adapter.world.agent_dir
        pose = pose_probe.pose_vector(snapshot, orientation)
        pose_key = exact_pose_key(snapshot, orientation)
        transition = adapter.step(action)
        return observation, transition.after, pose, pose_key
    finally:
        adapter.close()


@torch.inference_mode()
def _canonical_context_rows(baseline, probe, journal):
    rows = []
    completed = 0
    total = 120
    journal.update("canonical_native", 0, total)
    device = next(baseline.parameters()).device
    for split in one_step.SPLITS:
        for layout_name, spec in one_step._layout_specs()[split].items():
            layout, actions = spec[:2]
            prefix, continuation = one_step._validate_protocol(
                split, layout_name, layout, actions, one_step.SEED
            )
            adapter = temporal._adapter(layout, 1, one_step.SEED, 32)
            try:
                observation = adapter.reset(one_step.SEED)
                state = baseline.initial(observation)
                for action in prefix:
                    transition = adapter.step(action)
                    prediction = baseline.step(
                        state, torch.tensor([action], device=device, dtype=torch.long)
                    )
                    actual = baseline.initial(transition.after)
                    state = one_step._teacher_forced_next(prediction, actual)
            finally:
                adapter.close()

            for step, canonical_action in enumerate(continuation):
                history = (*prefix, *continuation[:step])
                canonical_after = None
                for action in range(5):
                    before, after, pose, pose_key = _fresh_context_fork(
                        layout, history, action, one_step.SEED
                    )
                    action_tensor = torch.tensor(
                        [action], device=device, dtype=torch.long
                    )
                    probability = probe(
                        state.z, state.hidden, pose[None].to(device), action_tensor
                    )
                    row = {
                        "split": split,
                        "layout": layout_name,
                        "step": step,
                        "real_history": list(history),
                        "canonical_action": canonical_action,
                        "action": action,
                        "action_name": one_step.GRID_ACTIONS[action],
                        "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                        "pose_key": pose_key,
                        "privileged_pose": pose.tolist(),
                        "event_probability": float(probability.item()),
                        "predicted_change": bool(probability.item() >= EVENT_BOUNDARY),
                        "_z": state.z.detach().cpu().clone(),
                        "_hidden": state.hidden.detach().cpu().clone(),
                    }
                    rows.append(row)
                    completed += 1
                    journal.update(
                        "canonical_native", completed, total,
                        split=split, layout=layout_name, step=step, action=action,
                    )
                    if action == canonical_action:
                        canonical_after = after
                if canonical_after is None:
                    raise RuntimeError("canonical action was not evaluated")
                action_tensor = torch.tensor(
                    [canonical_action], device=device, dtype=torch.long
                )
                prediction = baseline.step(state, action_tensor)
                actual = baseline.initial(canonical_after)
                state = one_step._teacher_forced_next(prediction, actual)
    if len(rows) != total:
        raise AssertionError(f"canonical evaluator produced {len(rows)} rows")
    return rows


def _native_reproduction(rows, reference_rows, tolerance=1e-6):
    if [row_audit.row_key(row) for row in rows] != [
        row_audit.row_key(row) for row in reference_rows
    ]:
        raise AssertionError("canonical native row identities differ from exp169")
    differences = [
        abs(row["event_probability"] - reference["event_probability"])
        for row, reference in zip(rows, reference_rows, strict=True)
    ]
    maximum = max(differences)
    result = {
        "rows": len(rows),
        "absolute_tolerance": tolerance,
        "max_abs_probability_difference": maximum,
        "matches": maximum <= tolerance,
    }
    if not result["matches"]:
        raise AssertionError(f"exp169 native probabilities differ by {maximum}")
    return result


def _public_row(row):
    return {
        key: (list(value) if key == "pose_key" else value)
        for key, value in row.items()
        if not key.startswith("_")
    }


def _identity(row):
    return {
        key: row[key]
        for key in ("split", "layout", "step", "action", "action_name")
    }


@torch.inference_mode()
def _probability(probe, z, hidden, pose, action, device):
    action_tensor = torch.tensor([action], device=device, dtype=torch.long)
    return float(probe(
        z.to(device), hidden.to(device), pose[None].to(device), action_tensor
    ).item())


def _matched_context_rows(rows, probe, journal):
    source = defaultdict(list)
    unseen = defaultdict(list)
    for row in rows:
        key = (_json_pose_key(row["pose_key"]), int(row["action"]), int(row["step"]))
        (source if row["split"] == "source" else unseen)[key].append(row)
    pair_total = sum(
        len(source[key]) * len(unseen[key]) for key in source.keys() & unseen.keys()
    )
    unmatched_total = sum(
        len(values) for key, values in source.items() if key not in unseen
    ) + sum(len(values) for key, values in unseen.items() if key not in source)
    total = pair_total + unmatched_total
    journal.update("matched_context", 0, max(total, 1))
    device = next(probe.parameters()).device
    output = []
    completed = 0
    for key in sorted(source.keys() | unseen.keys()):
        source_rows = source.get(key, [])
        unseen_rows = unseen.get(key, [])
        if source_rows and unseen_rows:
            for donor in source_rows:
                for recipient in unseen_rows:
                    pose = torch.tensor(recipient["privileged_pose"], dtype=torch.float32)
                    probabilities = {
                        "recipient_native": recipient["event_probability"],
                        "donor_hidden_only": _probability(
                            probe, recipient["_z"], donor["_hidden"], pose,
                            recipient["action"], device,
                        ),
                        "donor_z_only": _probability(
                            probe, donor["_z"], recipient["_hidden"], pose,
                            recipient["action"], device,
                        ),
                        "donor_z_hidden": _probability(
                            probe, donor["_z"], donor["_hidden"], pose,
                            recipient["action"], device,
                        ),
                    }
                    native_decision = probabilities["recipient_native"] >= EVENT_BOUNDARY
                    flips = {
                        name: (probability >= EVENT_BOUNDARY) != native_decision
                        for name, probability in probabilities.items()
                        if name != "recipient_native"
                    }
                    output.append({
                        "match_status": "paired",
                        "match_key": {
                            "pose_key": list(key[0]), "action": key[1], "step": key[2]
                        },
                        "donor": _identity(donor),
                        "recipient": _identity(recipient),
                        "donor_rgb_changed": donor["rgb_changed"],
                        "recipient_rgb_changed": recipient["rgb_changed"],
                        "probabilities": probabilities,
                        "flips": flips,
                    })
                    completed += 1
                    journal.update("matched_context", completed, max(total, 1))
        else:
            role, values = (
                ("unmatched_source", source_rows)
                if source_rows else ("unmatched_unseen", unseen_rows)
            )
            for row in values:
                output.append({
                    "match_status": role,
                    "match_key": {
                        "pose_key": list(key[0]), "action": key[1], "step": key[2]
                    },
                    "row": _identity(row),
                    "rgb_changed": row["rgb_changed"],
                    "event_probability": row["event_probability"],
                })
                completed += 1
                journal.update("matched_context", completed, max(total, 1))
    if completed != total:
        raise AssertionError("matched-context row accounting mismatch")
    if total == 0:
        journal.update("matched_context", 1, 1)
    return output


def _write_rows(path: Path, rows):
    writer = core.TraceWriter(path)
    try:
        for row in rows:
            writer.write(row)
    finally:
        writer.close()


def _swap_summary(rows):
    paired = [row for row in rows if row["match_status"] == "paired"]
    variants = ("donor_hidden_only", "donor_z_only", "donor_z_hidden")
    return {
        "paired_rows": len(paired),
        "unmatched_source_rows": sum(
            row["match_status"] == "unmatched_source" for row in rows
        ),
        "unmatched_unseen_rows": sum(
            row["match_status"] == "unmatched_unseen" for row in rows
        ),
        "flips": {
            variant: sum(row["flips"][variant] for row in paired)
            for variant in variants
        },
        "flip_rates": {
            variant: (
                sum(row["flips"][variant] for row in paired) / len(paired)
                if paired else None
            )
            for variant in variants
        },
    }


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    parser.set_defaults(baseline_checkpoint=DEFAULT_BASELINE)
    parser.add_argument("--exp169-dir", type=Path, default=DEFAULT_EXP169_DIR)
    return parser


def _argv(argv):
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frozen._prepare_output(args.out)
    started = time.monotonic()
    deadline = started + args.max_seconds
    command = os.environ.get("EXP171_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv), "exact_command": command, "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp169_dir": str(args.exp169_dir),
        "budgets": core._jsonable(vars(args)), "fixed_protocol": PROTOCOL,
        "objective": OBJECTIVE, "status": "running",
        "exit_code": None, "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 3, operation="safe_exp153_load")
            baseline, _ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            initial_backbone = {
                name: value.detach().clone()
                for name, value in baseline.state_dict().items()
            }
            journal.update("initialize", 1, 3, operation="load_exp169_reference")
            exp169_result, reference_rows, event_checkpoint_path = _read_reference(
                args.exp169_dir
            )
            event_checkpoint_sha256 = _sha256(event_checkpoint_path)
            probe, checkpoint_metadata = _load_event_checkpoint(
                event_checkpoint_path, baseline
            )
            initial_event = {
                name: value.detach().clone() for name, value in probe.state_dict().items()
            }
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config), seed=args.seed,
                z_dim=args.z_dim, h_dim=args.h_dim, burn_in=0,
                replay_capacity=len(temporal.SOURCE_LAYOUTS) * args.episodes_per_layout,
                termination_weight=0.0, salient_fraction=0.0,
            )
            if next(baseline.parameters()).device.type != torch.device(config.device).type:
                raise ValueError("checkpoint device and requested config disagree")
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 2, 3, operation="reconstruct_replay")
            corpus, _fit, _validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            episodes = input_probe._ordered_episodes(replay, corpus)
            train_episodes, heldout_episodes = input_probe.episode_disjoint_split(episodes)
            train_ids = {
                episode.uid for values in train_episodes.values() for episode in values
            }
            heldout_ids = {
                episode.uid for values in heldout_episodes.values() for episode in values
            }
            if train_ids & heldout_ids or len(train_ids | heldout_ids) != corpus["episodes"]:
                raise AssertionError("75/25 episode split is not complete and disjoint")
            journal.update("initialize", 3, 3, device=config.device)

            train_rows = _train_coverage_rows(train_episodes, args, journal)
            canonical_rows = _canonical_context_rows(baseline, probe, journal)
            reproduction = _native_reproduction(canonical_rows, reference_rows)
            covered_rows = coverage_counts(train_rows, canonical_rows)
            public_coverage = [_public_row(row) for row in covered_rows]
            journal.update("artifacts", 0, 3, operation="write_coverage")
            _write_rows(args.out / "canonical_coverage_rows.jsonl", public_coverage)

            matched_rows = _matched_context_rows(canonical_rows, probe, journal)
            swap_summary = _swap_summary(matched_rows)
            journal.update("artifacts", 1, 3, operation="write_swaps")
            _write_rows(args.out / "matched_context_rows.jsonl", matched_rows)

            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            event_head_unchanged = all(
                torch.equal(probe.state_dict()[name], value)
                for name, value in initial_event.items()
            )
            checkpoint_unchanged = _sha256(event_checkpoint_path) == event_checkpoint_sha256
            if not (backbone_unchanged and event_head_unchanged and checkpoint_unchanged):
                raise AssertionError("frozen backbone or exp169 event checkpoint changed")
            exact_protocol = bool(
                baseline_head == EXPECTED_BASELINE_HEAD
                and checkpoint_metadata["analysis_git_head"] == EXPECTED_EXP169_HEAD
                and exp169_result["analysis_git_head"] == EXPECTED_EXP169_HEAD
                and core._jsonable(asdict(config)) == FIXED_CONFIG
                and corpus["default_corpus_verified"]
                and all(
                    len(train_episodes[name]) == 384
                    and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                )
                and reproduction["matches"]
            )
            coverage_summary = {
                "canonical_rows": len(public_coverage),
                "train_transitions": len(train_rows),
                "rows_with_pose_action_coverage": sum(
                    row["train_pose_action_count"] > 0 for row in public_coverage
                ),
                "rows_with_opposite_label_coverage": sum(
                    row["opposite_label_train_count"] > 0 for row in public_coverage
                ),
                "zero_coverage_rows": sum(
                    row["train_pose_action_count"] == 0 for row in public_coverage
                ),
            }
            result = {
                "status": "completed",
                "claim": "exp169 event-head coverage and matched-context diagnostic",
                "analysis_git_head": manifest["analysis_git_head"],
                "exact_command": command, "exact_protocol": exact_protocol,
                "objective": OBJECTIVE, "corpus": corpus,
                "episode_split": {
                    "train_episodes": len(train_ids),
                    "heldout_episodes": len(heldout_ids), "overlap": 0,
                    "train_uid_digest": input_probe._uid_digest(train_episodes),
                    "heldout_uid_digest": input_probe._uid_digest(heldout_episodes),
                },
                "native_reproduction": reproduction,
                "coverage": coverage_summary,
                "matched_context": swap_summary,
                "frozen_backbone_unchanged": backbone_unchanged,
                "frozen_event_head_unchanged": event_head_unchanged,
                "event_checkpoint_sha256": event_checkpoint_sha256,
                "event_checkpoint_file_unchanged": checkpoint_unchanged,
                "interpretation_limits": [
                    "Pose/action coverage is coarse: walls and observations are absent from the key.",
                    "A swap flip proves sensitivity under this intervention, not a harmful shortcut; hybrid inputs may be out of distribution.",
                    "Source and unseen layouts are development evidence, not a sealed transfer test.",
                ],
                "controls": {
                    "training": False, "weights_updated": False,
                    "threshold_tuned": False, "planner_changed": False,
                    "representation_changed": False,
                    "coverage_uses_train_split_only": True,
                    "coverage_float_rounding": False,
                },
                "artifacts": {
                    "coverage_rows": "canonical_coverage_rows.jsonl",
                    "matched_context_rows": "matched_context_rows.jsonl",
                    "progress": "progress.jsonl", "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            journal.update("artifacts", 2, 3, operation="write_results")
            core._write_json(args.out / "results.json", result)
            manifest.update(
                status="completed", exit_code=0, exit_status=0,
                exact_protocol=exact_protocol,
                baseline_checkpoint_git_head=baseline_head,
                exp169_checkpoint_git_head=checkpoint_metadata["analysis_git_head"],
                event_checkpoint_sha256=event_checkpoint_sha256,
                frozen_backbone_unchanged=backbone_unchanged,
                frozen_event_head_unchanged=event_head_unchanged,
                runtime_seconds=time.monotonic() - started,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("artifacts", 3, 3, operation="complete")
            journal.close(status="completed")
            return 0
        except BaseException as error:
            code = temporal._exit_code(error)
            failure = {
                "status": "failed", "exit_code": code,
                "error": f"{type(error).__name__}: {error}",
                "analysis_git_head": manifest["analysis_git_head"],
                "exact_command": command,
            }
            core._write_json(args.out / "results.json", failure)
            manifest.update(
                status="failed", exit_code=code, exit_status=code,
                runtime_seconds=time.monotonic() - started,
                error=failure["error"],
            )
            core._write_json(args.out / "manifest.json", manifest)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
