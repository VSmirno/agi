"""Heldout-only zero-threshold calibration of the frozen exp162 amplitudes."""

from __future__ import annotations

from collections.abc import Mapping
from collections import defaultdict
from dataclasses import asdict, replace
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
from experiments import exp153_change_gated_dynamics as gated
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp159_independent_amplitude_oracle as amplitude_oracle
from experiments import exp161_amplitude_input_probe as input_probe
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_NONLINEAR_CHECKPOINT = Path(
    "output_to_user/core/exp162-nonlinear-amplitude-probe-001/"
    "nonlinear_amplitude_probe.pt"
)
EXPECTED_NONLINEAR_HEAD = "8edec06cca48b34a8285dec7d943f5ff4332082e"
PROTOCOL = dict(residual.PROTOCOL)
CALIBRATION = {
    "selection_source": "exact exp161/162 heldout episodes only",
    "intervention": "g'=0 if g<=tau_action else g",
    "parameters": "one scalar threshold per action shared across ensemble members",
    "objective": "teacher-forced actual-next-z latent MSE",
    "candidate_search": "exact frozen-score breakpoints plus tau=0",
}


class ThresholdCalibratedProbe(torch.nn.Module):
    """Apply a shared per-action hard zero threshold to frozen amplitudes."""

    def __init__(self, probe: torch.nn.Module, thresholds: torch.Tensor):
        super().__init__()
        if thresholds.shape != (5,):
            raise ValueError("thresholds must have shape [5]")
        if not torch.isfinite(thresholds).all() or torch.any(
            (thresholds < 0) | (thresholds > 1)
        ):
            raise ValueError("thresholds must be finite and in [0,1]")
        self.probe = probe
        self.register_buffer("thresholds", thresholds.detach().clone())

    def forward(self, z, hidden, actions):
        amplitudes = self.probe(z, hidden, actions)
        selected = self.thresholds[actions].unsqueeze(0)
        return torch.where(amplitudes <= selected, 0.0, amplitudes)


class CalibratedWorldModel(gated.ChangeGatedResidualWorldModel):
    """Install a frozen threshold-calibrated probe over exp153 raw deltas."""

    def __init__(self, *args, amplitude_probe: ThresholdCalibratedProbe, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gate_heads
        self.amplitude_probe = amplitude_probe

    def change_gates(self, state: LatentState, actions: torch.Tensor):
        return self.amplitude_probe(state.z, state.hidden, actions).unsqueeze(-1)


def calibration_split_audit(train_episode_ids: set[str], heldout_episode_ids: set[str]):
    overlap = train_episode_ids & heldout_episode_ids
    if overlap:
        raise ValueError(f"calibration episode overlap: {sorted(overlap)[:3]}")
    return {
        "selection_source": "heldout_episodes_only",
        "train_episodes_used_for_selection": 0,
        "canonical_audit_rows_used_for_selection": 0,
        "heldout_episodes": len(heldout_episode_ids),
        "overlap": 0,
    }


def _load_nonlinear_probe(path: Path, device: torch.device):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"could not safely load exp162 checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("format_version") != 1:
        raise ValueError("exp162 checkpoint requires format_version 1")
    if payload.get("analysis_git_head") != EXPECTED_NONLINEAR_HEAD:
        raise ValueError("exp162 checkpoint analysis head mismatch")
    if payload.get("baseline_checkpoint_git_head") != EXPECTED_BASELINE_HEAD:
        raise ValueError("exp162 checkpoint baseline head mismatch")
    dimensions = {}
    for name, expected in (("z_dim", 256), ("h_dim", 128), ("ensemble_size", 3)):
        value = payload.get(name)
        if value != expected:
            raise ValueError(f"exp162 checkpoint {name} must be {expected}")
        dimensions[name] = value
    if payload.get("hidden_width") != nonlinear.HIDDEN_WIDTH:
        raise ValueError("exp162 checkpoint hidden width mismatch")
    objective = payload.get("objective")
    if not isinstance(objective, Mapping) or objective.get("architecture") != nonlinear.OBJECTIVE["architecture"]:
        raise ValueError("exp162 checkpoint objective/architecture mismatch")
    probe = nonlinear.NonlinearAmplitudeProbe(
        dimensions["z_dim"], dimensions["h_dim"], dimensions["ensemble_size"]
    )
    state = payload.get("probe_state_dict")
    auxiliary.checkpoint_io._validate_state_dict(
        "probe_state_dict", state, probe.state_dict()
    )
    probe.load_state_dict(state, strict=True)
    return probe.to(device).eval().requires_grad_(False), payload


def _installed_model(baseline, probe):
    parameter = next(baseline.parameters())
    model = CalibratedWorldModel(
        nonlinear.CoreEncoder(baseline.encoder.z_dim),
        dict(baseline.schemas),
        baseline.h_dim,
        baseline.heads,
        normalize_sensor_condition=baseline.normalize_sensor_condition,
        predict_sensor_delta=baseline.predict_sensor_delta,
        amplitude_probe=probe,
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate_keys = set(model.state_dict())
    transferable = {
        name: value
        for name, value in baseline.state_dict().items()
        if name in candidate_keys
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("amplitude_probe.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected frozen transfer mismatch: {incompatible}")
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _extract_calibration_dataset(model, episodes_by_layout, journal):
    episode_total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    names = ("z", "hidden", "target", "raw", "displacement", "actions", "changed")
    chunks = {name: [] for name in names}
    journal.update("extract_heldout", 0, episode_total)
    parameter = next(model.parameters())
    for layout in temporal.SOURCE_LAYOUTS:
        for episode in episodes_by_layout[layout]:
            transitions = episode.transitions
            observations = [transitions[0].before, *[row.after for row in transitions]]
            rgb = torch.tensor(
                np.stack([obs.rgb for obs in observations]),
                device=parameter.device,
                dtype=parameter.dtype,
            ) / 255
            z_sequence = model.encoder(rgb)
            sensors = torch.tensor(
                np.stack([obs.sensors for obs in observations]),
                device=parameter.device,
                dtype=parameter.dtype,
            )
            masks = torch.tensor(
                np.stack([obs.sensor_mask for obs in observations]),
                device=parameter.device,
                dtype=torch.bool,
            )
            state = LatentState(
                z_sequence[0:1],
                torch.where(masks[0:1], sensors[0:1], 0.0),
                masks[0:1],
                z_sequence.new_zeros(1, model.h_dim),
                transitions[0].before.schema,
            )
            rows = {name: [] for name in names}
            for index, transition in enumerate(transitions):
                action = torch.tensor(
                    [transition.action], device=parameter.device, dtype=torch.long
                )
                prediction, raw_deltas = raw_oracle.native_prediction_and_raw_deltas(
                    model, state, action
                )
                displacement = z_sequence[index + 1 : index + 2] - state.z
                target = amplitude_oracle.independent_member_amplitudes(
                    raw_deltas[:, 0], displacement[0]
                ).detach()
                rows["z"].append(state.z[0])
                rows["hidden"].append(state.hidden[0])
                rows["target"].append(target)
                rows["raw"].append(raw_deltas[:, 0])
                rows["displacement"].append(displacement[0])
                rows["actions"].append(action[0])
                rows["changed"].append(
                    torch.tensor(
                        not np.array_equal(transition.before.rgb, transition.after.rgb),
                        device=parameter.device,
                        dtype=torch.bool,
                    )
                )
                actual = LatentState(
                    z_sequence[index + 1 : index + 2],
                    torch.where(
                        masks[index + 1 : index + 2],
                        sensors[index + 1 : index + 2],
                        0.0,
                    ),
                    masks[index + 1 : index + 2],
                    z_sequence.new_zeros(1, model.h_dim),
                    state.schema,
                )
                state = one_step._teacher_forced_next(prediction, actual)
            for name, values in rows.items():
                chunks[name].append(torch.stack(values).cpu())
            completed += 1
            journal.update("extract_heldout", completed, episode_total, layout=layout)
    return {name: torch.cat(values) for name, values in chunks.items()}


@torch.inference_mode()
def _score_dataset(probe, dataset, device, journal):
    outputs = []
    total = len(dataset["actions"])
    batch = 4096
    journal.update("score_heldout", 0, total)
    for start in range(0, total, batch):
        stop = min(start + batch, total)
        outputs.append(
            probe(
                dataset["z"][start:stop].to(device),
                dataset["hidden"][start:stop].to(device),
                dataset["actions"][start:stop].to(device),
            ).transpose(0, 1).cpu()
        )
        journal.update("score_heldout", stop, total)
    return torch.cat(outputs)


def _select_threshold(scores, raw_deltas, displacement) -> dict:
    """Exactly minimize ensemble latent MSE over frozen score breakpoints."""

    scores_np = scores.detach().double().numpy()
    raw_np = raw_deltas.detach().double().numpy()
    target_np = displacement.detach().double().numpy()
    members = scores_np.shape[1]
    contributions = scores_np[..., None] * raw_np / members
    errors = contributions.sum(axis=1) - target_np
    total_sse = float(np.square(errors).sum())
    denominator = errors.size
    best_sse, best_tau = total_sse, 0.0
    flat_scores = scores_np.reshape(-1)
    order = np.argsort(flat_scores, kind="stable")
    cursor = 0
    evaluated = 1
    while cursor < len(order):
        value = float(flat_scores[order[cursor]])
        stop = cursor + 1
        while stop < len(order) and flat_scores[order[stop]] == value:
            stop += 1
        for flat_index in order[cursor:stop]:
            row, member = divmod(int(flat_index), members)
            contribution = contributions[row, member]
            before = errors[row]
            after = before - contribution
            total_sse += float(np.dot(after, after) - np.dot(before, before))
            errors[row] = after
        evaluated += 1
        if total_sse < best_sse - 1e-12:
            best_sse, best_tau = total_sse, value
        cursor = stop
    return {
        "threshold": best_tau,
        "initial_mse": float(np.square(contributions.sum(axis=1) - target_np).mean()),
        "calibrated_mse": best_sse / denominator,
        "score_breakpoints_evaluated": evaluated,
        "member_scores": flat_scores.size,
        "suppressed_member_fraction": float((scores_np <= best_tau).mean()),
    }


def _calibration_mse(scores, raw, displacement, actions, thresholds):
    selected = thresholds[actions].unsqueeze(1)
    amplitudes = torch.where(scores <= selected, 0.0, scores)
    predicted = (amplitudes[..., None] * raw).mean(dim=1)
    return float((predicted - displacement).square().mean())


def _distribution(values: torch.Tensor) -> dict:
    if not values.numel():
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    array = values.detach().double().numpy()
    return {
        "count": len(array),
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def _overlap_statistics(scores, targets, actions, thresholds):
    groups = {}
    for label, mask in [("overall", torch.ones_like(actions, dtype=torch.bool))] + [
        (f"action{action}", actions == action) for action in range(5)
    ]:
        member_mask = mask.unsqueeze(1).expand_as(scores)
        zero = scores[member_mask & (targets == 0)]
        positive = scores[member_mask & (targets > 0)]
        zero_dist, positive_dist = _distribution(zero), _distribution(positive)
        tau = None if label == "overall" else float(thresholds[int(label[6:])])
        row = {
            "target_zero": zero_dist,
            "target_positive": positive_dist,
            "support_overlap_width": (
                max(0.0, min(zero_dist["max"], positive_dist["max"])
                    - max(zero_dist["min"], positive_dist["min"]))
                if zero.numel() and positive.numel()
                else None
            ),
        }
        if tau is not None:
            row.update(
                threshold=tau,
                zero_suppressed_fraction=float((zero <= tau).float().mean())
                if zero.numel()
                else None,
                positive_retained_fraction=float((positive > tau).float().mean())
                if positive.numel()
                else None,
            )
        groups[label] = row
    return groups


def _split_pass(summary: Mapping) -> bool:
    medians = summary["medians"]
    free = medians["free_forward_prediction_persistence_ratio"]
    interact = medians["interact_prediction_persistence_ratio"]
    return bool(
        summary["contact_failure_layouts"] == 0
        and summary["blocked_noop_failure_layouts"] == 0
        and free is not None
        and math.isfinite(free)
        and free < 1.0
        and interact is not None
        and math.isfinite(interact)
        and interact < 1.0
    )


@torch.inference_mode()
def _gate_statistics(model, output_path, deadline, journal):
    """Log gates on the exact exp148 carried-hidden state for every audit fork."""

    grouped = defaultdict(list)
    writer = core.TraceWriter(output_path)
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * 3 * 5
    completed = 0
    journal.update("gate_statistics", 0, total)
    try:
        for split in one_step.SPLITS:
            for layout_name, spec in specs[split].items():
                layout, actions = spec[:2]
                prefix, continuation = one_step._validate_protocol(
                    split, layout_name, layout, actions, one_step.SEED
                )
                state, _diagnostic = one_step._replay_prefix(
                    model, layout, prefix, one_step.SEED
                )
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, _diagnostic = one_step._fresh_real_fork(
                            layout, history, action, one_step.SEED
                        )
                        action_tensor = torch.tensor(
                            [action], device=state.z.device, dtype=torch.long
                        )
                        values = model.change_gates(state, action_tensor).flatten().tolist()
                        record = {
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "action": action,
                            "action_name": one_step.GRID_ACTIONS[action],
                            "rgb_changed": bool(
                                not np.array_equal(before.rgb, after.rgb)
                            ),
                            "real_history": list(history),
                            "by_member": values,
                            "mean": sum(values) / len(values),
                            "min": min(values),
                            "max": max(values),
                        }
                        writer.write(record)
                        grouped[(split, action, step, record["rgb_changed"])].extend(
                            values
                        )
                        if action == canonical_action:
                            canonical_prediction = model.step(state, action_tensor)
                            canonical_actual = model.initial(after)
                        completed += 1
                        journal.update(
                            "gate_statistics", completed, total,
                            split=split, layout=layout_name, step=step, action=action,
                        )
                    state = one_step._teacher_forced_next(
                        canonical_prediction, canonical_actual
                    )
    finally:
        writer.close()
    return {
        "diagnostic_only": True,
        "rows": total,
        "input": "exact exp148 teacher-forced real state with carried predicted hidden",
        "by_action_context": [
            {
                "split": split,
                "action": action,
                "step": step,
                "rgb_changed": changed,
                "member_values": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for (split, action, step, changed), values in sorted(grouped.items())
        ],
        "artifacts": {"rows": output_path.name},
    }


def _canonical_contexts(statistics_by_model):
    result = {}
    for model, statistics_row in statistics_by_model.items():
        rows = statistics_row["by_action_context"]
        result[model] = {
            split: {
                "blocked": next(
                    row for row in rows
                    if row["split"] == split and row["action"] == 2
                    and row["step"] == 0 and not row["rgb_changed"]
                ),
                "free": next(
                    row for row in rows
                    if row["split"] == split and row["action"] == 2
                    and row["step"] == 1 and row["rgb_changed"]
                ),
                "contact": [
                    row for row in rows
                    if row["split"] == split and row["action"] == 3
                    and row["step"] in (0, 2) and row["rgb_changed"]
                ],
            }
            for split in one_step.SPLITS
        }
    return result


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--nonlinear-checkpoint", type=Path, default=DEFAULT_NONLINEAR_CHECKPOINT
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
    deadline = started + args.max_seconds
    command = os.environ.get("EXP163_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "nonlinear_checkpoint": str(args.nonlinear_checkpoint),
        "budgets": core._jsonable(vars(args)),
        "fixed_protocol": PROTOCOL,
        "calibration": CALIBRATION,
        "status": "running",
        "exit_code": None,
        "exit_status": None,
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
            device = next(baseline.parameters()).device
            journal.update("initialize", 1, 3, operation="safe_exp162_load")
            probe, probe_payload = _load_nonlinear_probe(
                args.nonlinear_checkpoint, device
            )
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config),
                seed=args.seed,
                z_dim=args.z_dim,
                h_dim=args.h_dim,
                burn_in=0,
                replay_capacity=len(temporal.SOURCE_LAYOUTS)
                * args.episodes_per_layout,
                termination_weight=0.0,
                salient_fraction=0.0,
            )
            if device.type != torch.device(config.device).type:
                raise ValueError("checkpoint device and requested config disagree")
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 3, 3, device=config.device)
            corpus, _fit, _validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            episodes = input_probe._ordered_episodes(replay, corpus)
            train_episodes, heldout_episodes = input_probe.episode_disjoint_split(
                episodes
            )
            train_ids = {
                episode.uid for values in train_episodes.values() for episode in values
            }
            heldout_ids = {
                episode.uid for values in heldout_episodes.values() for episode in values
            }
            split_audit = calibration_split_audit(train_ids, heldout_ids)
            coverage = _audit_counts({"all": replay._episodes()})
            counts = {
                action: {
                    key: row[key]
                    for key in ("total", "rgb_changed", "rgb_no_change")
                }
                for action, row in coverage["actions"].items()
            }
            fixed_counts = counts == FIXED_CORPUS["action_counts"]
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "canonical_nonlinear_checkpoint": args.nonlinear_checkpoint
                == DEFAULT_NONLINEAR_CHECKPOINT,
                "nonlinear_checkpoint_head": probe_payload["analysis_git_head"]
                == EXPECTED_NONLINEAR_HEAD,
                "default_budgets": all(
                    getattr(args, key) == value for key, value in PROTOCOL.items()
                ),
                "baseline_config": metadata["config"] == FIXED_CONFIG,
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_and_counts": corpus["default_corpus_verified"]
                and fixed_counts,
                "episode_split_75_25": all(
                    len(train_episodes[name]) == 384
                    and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                ),
                "heldout_only_selection": split_audit[
                    "canonical_audit_rows_used_for_selection"
                ] == 0,
            }
            exact_protocol = all(matching.values())
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                nonlinear_checkpoint_git_head=probe_payload["analysis_git_head"],
                protocol_match=matching,
                calibration_split=split_audit,
                action_counts=counts,
            )
            core._write_json(args.out / "manifest.json", manifest)

            dataset = _extract_calibration_dataset(
                baseline, heldout_episodes, journal
            )
            scores = _score_dataset(probe, dataset, device, journal)
            threshold_rows = {}
            thresholds = torch.zeros(5)
            journal.update("select_thresholds", 0, 5)
            for action in range(5):
                selected = dataset["actions"] == action
                row = _select_threshold(
                    scores[selected],
                    dataset["raw"][selected],
                    dataset["displacement"][selected],
                )
                threshold_rows[str(action)] = row
                thresholds[action] = row["threshold"]
                journal.update(
                    "select_thresholds", action + 1, 5,
                    action=action, threshold=row["threshold"],
                )
            calibration_metrics = {
                "uncalibrated_mse": _calibration_mse(
                    scores,
                    dataset["raw"],
                    dataset["displacement"],
                    dataset["actions"],
                    torch.zeros(5),
                ),
                "calibrated_mse": _calibration_mse(
                    scores,
                    dataset["raw"],
                    dataset["displacement"],
                    dataset["actions"],
                    thresholds,
                ),
                "by_action": threshold_rows,
            }
            overlap = _overlap_statistics(
                scores, dataset["target"], dataset["actions"], thresholds
            )
            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            if not backbone_unchanged:
                raise AssertionError("frozen exp153 backbone changed")

            uncalibrated_model = nonlinear._installed_model(baseline, probe)
            calibrated_probe = ThresholdCalibratedProbe(
                probe, thresholds.to(device)
            ).eval().requires_grad_(False)
            calibrated_model = _installed_model(baseline, calibrated_probe)
            diagnostics, gate_statistics = {}, {}
            for role, model in (
                ("exp162_uncalibrated", uncalibrated_model),
                ("calibrated", calibrated_model),
            ):
                rows_path = args.out / f"{role}_one_step_rows.jsonl"
                journal.update(f"one_step_{role}", 0, 120)
                diagnostics[role] = one_step._diagnose(model, journal, rows_path)
                journal.update(f"one_step_{role}", 120, 120)
                gate_statistics[role] = _gate_statistics(
                    model,
                    args.out / f"{role}_gate_rows.jsonl",
                    deadline,
                    journal,
                )
            source_pass = _split_pass(
                diagnostics["calibrated"]["splits"]["source"]
            )
            unseen_pass = _split_pass(
                diagnostics["calibrated"]["splits"]["unseen"]
            )
            gate = bool(exact_protocol and source_pass and unseen_pass)
            if gate:
                outcome = "zero_atom_calibration_passes"
                conclusion = (
                    "A heldout-only zero atom/calibration objective repairs both local splits."
                )
            elif source_pass and not unseen_pass:
                outcome = "source_only_calibration"
                conclusion = (
                    "Calibration repairs source only; the remaining failure is "
                    "generalization or state quality."
                )
            else:
                outcome = "score_overlap_or_state_failure"
                conclusion = (
                    "Frozen scores still overlap the required zero/positive amplitudes; "
                    "the remaining failure is ranking/state quality, not calibration length."
                )
            journal.update("calibration_checkpoint", 0, 1)
            checkpoint_path = args.out / "calibrated_amplitude_probe.pt"
            torch.save(
                {
                    "format_version": 1,
                    "analysis_git_head": manifest["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "nonlinear_checkpoint_git_head": probe_payload["analysis_git_head"],
                    "calibration": CALIBRATION,
                    "thresholds": thresholds,
                    "probe_state_dict": probe.state_dict(),
                },
                checkpoint_path,
            )
            journal.update("calibration_checkpoint", 1, 1)
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "heldout-only frozen amplitude calibration diagnostic",
                "interpretation_limit": "No training, composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "nonlinear_checkpoint_git_head": probe_payload["analysis_git_head"],
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "calibration": CALIBRATION,
                "calibration_split": split_audit,
                "corpus": corpus,
                "action_counts": counts,
                "target_distribution": input_probe._target_summary(dataset),
                "thresholds": thresholds.tolist(),
                "heldout_latent_metrics": calibration_metrics,
                "score_overlap": overlap,
                "one_step": diagnostics,
                "gate_statistics": gate_statistics,
                "canonical_gate_contexts": _canonical_contexts(gate_statistics),
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "frozen_calibration_gate": gate,
                "outcome": outcome,
                "conclusion": conclusion,
                "frozen_backbone_unchanged": backbone_unchanged,
                "controls": {
                    "new_weights_or_training": False,
                    "threshold_count": 5,
                    "threshold_shared_across_members": True,
                    "heldout_only_selection": True,
                    "canonical_audit_rows_used_for_selection": 0,
                    "object_or_task_labels": False,
                    "crafter_specific_branches": False,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "uncalibrated_rows": "exp162_uncalibrated_one_step_rows.jsonl",
                    "calibrated_rows": "calibrated_one_step_rows.jsonl",
                    "uncalibrated_gate_rows": "exp162_uncalibrated_gate_rows.jsonl",
                    "calibrated_gate_rows": "calibrated_gate_rows.jsonl",
                    "progress": "progress.jsonl",
                    "manifest": "manifest.json",
                    "run_log": "run.log",
                },
            }
            core._write_json(args.out / "results.json", result)
            journal.update("artifacts", 1, 2, operation="write_manifest")
            manifest.update(
                status="completed",
                exit_code=0,
                exit_status=0,
                exact_protocol=exact_protocol,
                runtime_seconds=time.monotonic() - started,
                thresholds=thresholds.tolist(),
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
