"""Privileged relational-pose hurdle probe for sparse amplitude atoms."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp148_source_target_one_step as one_step
from experiments import exp150_residual_dynamics as residual
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp161_amplitude_input_probe as linear_probe
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments import exp164_relational_slot_probe as relational
from experiments import exp165_relational_pose_probe as pose_probe
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_EXP162_REFERENCE = relational.DEFAULT_EXP162_REFERENCE
EXPECTED_EXP162_HEAD = relational.EXPECTED_EXP162_HEAD
DEFAULT_EXP165_REFERENCE = Path(
    "output_to_user/core/exp165-relational-pose-probe-001/results.json"
)
EXPECTED_EXP165_HEAD = "001505f3a681b746598738d67a88f3f466425afb"
PROTOCOL = dict(residual.PROTOCOL)
POSE_DIM = pose_probe.POSE_DIM
HIDDEN_WIDTH = pose_probe.HIDDEN_WIDTH
ATOM_BOUNDARY = 0.5
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "input": "frozen z + carried hidden + four relations + agent_dir one-hot",
    "architecture": (
        "per-action shared Linear(input,128)->ReLU torso with separate "
        "member atom logits and conditional sigmoid amplitudes"
    ),
    "atom": "target > 0; installed when probability >= 0.5",
    "atom_loss": "fixed train-only class-balanced per-action/member BCE",
    "conditional_loss": "MSE over positive target members only",
    "total_loss": "mean weighted atom BCE + mean positive conditional MSE",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_diagnostic": True,
}


def hurdle_gate(atom_logits: torch.Tensor, conditional: torch.Tensor) -> torch.Tensor:
    """Install exact-zero atoms using the fixed probability boundary."""

    if atom_logits.shape != conditional.shape:
        raise ValueError("atom logits and conditional amplitudes must have equal shape")
    if not torch.isfinite(atom_logits).all() or not torch.isfinite(conditional).all():
        raise ValueError("hurdle components must be finite")
    return torch.where(
        atom_logits.sigmoid() >= ATOM_BOUNDARY,
        conditional,
        torch.zeros_like(conditional),
    )


def atom_class_weights(
    actions: torch.Tensor, target: torch.Tensor, action_count: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute fixed balanced atom weights for each action and ensemble member."""

    if actions.ndim != 1 or actions.dtype != torch.long:
        raise ValueError("actions must be long with shape [batch]")
    if target.ndim != 2 or target.shape[1] != len(actions):
        raise ValueError("target must have shape [members,batch]")
    if action_count <= 0 or bool(((actions < 0) | (actions >= action_count)).any()):
        raise ValueError("actions must fit action_count")
    positive = target > 0
    members = target.shape[0]
    counts = torch.zeros(
        action_count, members, 2, dtype=torch.long, device=target.device
    )
    weights = torch.zeros(
        action_count, members, 2, dtype=target.dtype, device=target.device
    )
    for action in range(action_count):
        selected = actions == action
        for member in range(members):
            positives = int((positive[member] & selected).sum())
            negatives = int((~positive[member] & selected).sum())
            counts[action, member] = torch.tensor(
                [negatives, positives], device=target.device
            )
            total = negatives + positives
            if negatives and positives:
                weights[action, member, 0] = total / (2.0 * negatives)
                weights[action, member, 1] = total / (2.0 * positives)
            elif negatives:
                weights[action, member, 0] = 1.0
            elif positives:
                weights[action, member, 1] = 1.0
    return weights, counts


def hurdle_amplitude_loss(
    atom_logits: torch.Tensor,
    conditional: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return fixed weighted atom BCE plus positive-only conditional MSE."""

    if atom_logits.shape != target.shape or conditional.shape != target.shape:
        raise ValueError("predictions and target must share [members,batch] shape")
    if actions.shape != (target.shape[1],) or actions.dtype != torch.long:
        raise ValueError("actions must be long with shape [batch]")
    if weights.shape != (int(weights.shape[0]), target.shape[0], 2):
        raise ValueError("weights must have shape [actions,members,2]")
    labels = target > 0
    labels_bm = labels.transpose(0, 1)
    selected_weights = torch.gather(
        weights[actions], 2, labels_bm.long().unsqueeze(-1)
    ).squeeze(-1).transpose(0, 1)
    bce = functional.binary_cross_entropy_with_logits(
        atom_logits, labels.to(atom_logits.dtype), reduction="none"
    )
    atom_bce = (bce * selected_weights).mean()
    if bool(labels.any()):
        conditional_mse = (conditional[labels] - target[labels]).square().mean()
    else:
        conditional_mse = conditional.sum() * 0.0
    total = atom_bce + conditional_mse
    return total, {"atom_bce": atom_bce, "conditional_mse": conditional_mse}


class HurdleActionHead(torch.nn.Module):
    def __init__(self, width: int, members: int):
        super().__init__()
        self.torso = torch.nn.Sequential(
            torch.nn.Linear(width, HIDDEN_WIDTH), torch.nn.ReLU()
        )
        self.atom = torch.nn.Linear(HIDDEN_WIDTH, members)
        self.conditional = torch.nn.Linear(HIDDEN_WIDTH, members)

    def forward(self, features: torch.Tensor):
        encoded = self.torso(features)
        return self.atom(encoded), self.conditional(encoded).sigmoid()


class HurdleAmplitudeProbe(torch.nn.Module):
    """Per-action pose-aware torso with separate atom and amplitude outputs."""

    def __init__(self, z_dim: int, h_dim: int, heads: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        width = z_dim + h_dim + POSE_DIM
        self.by_action = torch.nn.ModuleList(
            HurdleActionHead(width, heads) for _ in range(5)
        )

    def components(self, z, hidden, pose, actions):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if pose.shape != (len(z), POSE_DIM):
            raise ValueError("pose must have shape [batch,8]")
        if actions.shape != (len(z),) or actions.dtype != torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden, pose), dim=-1)
        all_components = [head(features) for head in self.by_action]
        logits = torch.stack([row[0] for row in all_components], dim=1)
        conditional = torch.stack([row[1] for row in all_components], dim=1)
        gather = actions[:, None, None].expand(-1, 1, self.heads)
        selected_logits = logits.gather(1, gather).squeeze(1).transpose(0, 1)
        selected_conditional = (
            conditional.gather(1, gather).squeeze(1).transpose(0, 1)
        )
        return selected_logits, selected_conditional

    def forward(self, z, hidden, pose, actions):
        logits, conditional = self.components(z, hidden, pose, actions)
        return hurdle_gate(logits, conditional)


class HurdlePoseWorldModel(pose_probe.RelationalPoseWorldModel):
    """Frozen raw exp153 deltas gated by the fitted hurdle probe."""

    def change_gates(self, state: LatentState, actions: torch.Tensor):
        pose = self._current_relations
        if pose is None or pose.shape[0] != state.z.shape[0]:
            raise RuntimeError("pose context was not installed for this state")
        pose = pose.to(device=state.z.device, dtype=state.z.dtype)
        return self.amplitude_probe(
            state.z, state.hidden, pose, actions
        ).unsqueeze(-1)


def _installed_model(baseline, probe):
    parameter = next(baseline.parameters())
    model = HurdlePoseWorldModel(
        CoreEncoder(baseline.encoder.z_dim),
        dict(baseline.schemas),
        baseline.h_dim,
        baseline.heads,
        normalize_sensor_condition=baseline.normalize_sensor_condition,
        predict_sensor_delta=baseline.predict_sensor_delta,
        amplitude_probe=probe,
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate_keys = set(model.state_dict())
    transferable = {
        name: value for name, value in baseline.state_dict().items()
        if name in candidate_keys
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("amplitude_probe.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected frozen transfer mismatch: {incompatible}")
    return model.eval().requires_grad_(False)


def _fit_probe(train, heldout, weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = HurdleAmplitudeProbe(
            config.z_dim, config.h_dim, config.ensemble_size
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_hurdle", 0, args.probe_updates)
    probe.train()
    for update in range(1, args.probe_updates + 1):
        indices = torch.randint(
            len(train["actions"]), (args.probe_batch_size,), generator=generator
        )
        actions = train["actions"][indices].to(device)
        logits, conditional = probe.components(
            train["z"][indices].to(device),
            train["hidden"][indices].to(device),
            train["relations"][indices].to(device),
            actions,
        )
        target = train["target"][indices].to(device).transpose(0, 1)
        loss, parts = hurdle_amplitude_loss(
            logits, conditional, target, actions, weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        row = {
            "update": update,
            "loss": float(loss.detach()),
            "atom_bce": float(parts["atom_bce"].detach()),
            "conditional_mse": float(parts["conditional_mse"].detach()),
        }
        losses.append(row)
        trace.write(row)
        journal.update("fit_hurdle", update, args.probe_updates, **row)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _probe_metrics(probe, train, weights, device),
        "heldout": _probe_metrics(probe, heldout, weights, device),
    }


@torch.inference_mode()
def _probe_metrics(probe, dataset, weights, device):
    logits_chunks, conditional_chunks = [], []
    for start in range(0, len(dataset["actions"]), 4096):
        stop = min(start + 4096, len(dataset["actions"]))
        logits, conditional = probe.components(
            dataset["z"][start:stop].to(device),
            dataset["hidden"][start:stop].to(device),
            dataset["relations"][start:stop].to(device),
            dataset["actions"][start:stop].to(device),
        )
        logits_chunks.append(logits.cpu())
        conditional_chunks.append(conditional.cpu())
    logits = torch.cat(logits_chunks, dim=1)
    conditional = torch.cat(conditional_chunks, dim=1)
    target = dataset["target"].transpose(0, 1)
    actions = dataset["actions"]
    labels = target > 0
    predicted = logits.sigmoid() >= ATOM_BOUNDARY
    installed = hurdle_gate(logits, conditional)
    groups = []
    for action in range(5):
        action_mask = actions == action
        for member in range(target.shape[0]):
            truth = labels[member, action_mask]
            guess = predicted[member, action_mask]
            negatives = int((~truth).sum())
            positives = int(truth.sum())
            recall_zero = float((~guess[~truth]).float().mean()) if negatives else None
            recall_positive = float(guess[truth].float().mean()) if positives else None
            recalls = [value for value in (recall_zero, recall_positive) if value is not None]
            groups.append({
                "action": action,
                "member": member,
                "zero": negatives,
                "positive": positives,
                "recall_zero": recall_zero,
                "recall_positive": recall_positive,
                "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
            })
    negative = ~labels
    positive = labels
    recall_zero = float((~predicted[negative]).float().mean()) if bool(negative.any()) else None
    recall_positive = float(predicted[positive].float().mean()) if bool(positive.any()) else None
    conditional_mse = (
        float((conditional[positive] - target[positive]).square().mean())
        if bool(positive.any()) else None
    )
    loss, parts = hurdle_amplitude_loss(
        logits, conditional, target, actions, weights.cpu()
    )
    return {
        "transitions": len(actions),
        "member_elements": target.numel(),
        "zero_targets": int(negative.sum()),
        "positive_targets": int(positive.sum()),
        "recall_zero": recall_zero,
        "recall_positive": recall_positive,
        "balanced_accuracy": (
            (recall_zero + recall_positive) / 2
            if recall_zero is not None and recall_positive is not None else None
        ),
        "conditional_positive_mse": conditional_mse,
        "installed_exact_zero_rate": float((installed == 0).float().mean()),
        "installed_nonzero_rate": float((installed > 0).float().mean()),
        "objective": {
            "total": float(loss),
            "atom_bce": float(parts["atom_bce"]),
            "conditional_mse": float(parts["conditional_mse"]),
        },
        "by_action_member": groups,
    }


def _load_exp165_reference(path: Path):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp165 reference: {error}") from error
    if payload.get("analysis_git_head") != EXPECTED_EXP165_HEAD:
        raise ValueError("exp165 reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError("exp165 reference is not exact protocol")
    if not isinstance(payload.get("one_step"), Mapping):
        raise ValueError("exp165 reference lacks one-step results")
    if not isinstance(payload.get("probe_metrics"), Mapping):
        raise ValueError("exp165 reference lacks probe metrics")
    return {
        "path": str(path),
        "analysis_git_head": payload["analysis_git_head"],
        "probe_metrics": payload["probe_metrics"],
        "one_step": payload["one_step"],
        "gate": payload["relational_pose_gate"],
    }


@torch.inference_mode()
def _gate_statistics(model, output_path, journal):
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
                state, _ = pose_probe._replay_pose_prefix(
                    model, layout, prefix, one_step.SEED
                )
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, _, pose = pose_probe._fresh_pose_fork(
                            layout, history, action, one_step.SEED
                        )
                        pose = pose[None].to(state.z.device, state.z.dtype)
                        model.set_relations(pose)
                        action_tensor = torch.tensor(
                            [action], device=state.z.device, dtype=torch.long
                        )
                        logits, conditional = model.amplitude_probe.components(
                            state.z, state.hidden, pose, action_tensor
                        )
                        gates = hurdle_gate(logits, conditional).flatten()
                        probabilities = logits.sigmoid().flatten()
                        amplitudes = conditional.flatten()
                        row = {
                            "split": split,
                            "layout": layout_name,
                            "step": step,
                            "action": action,
                            "action_name": one_step.GRID_ACTIONS[action],
                            "rgb_changed": bool(not np.array_equal(before.rgb, after.rgb)),
                            "real_history": list(history),
                            "by_member": gates.tolist(),
                            "atom_probability_by_member": probabilities.tolist(),
                            "conditional_by_member": amplitudes.tolist(),
                            "exact_zero_fraction": float((gates == 0).float().mean()),
                        }
                        writer.write(row)
                        grouped[(split, action, step, row["rgb_changed"])].append(row)
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
    groups = []
    for (split, action, step, changed), rows in sorted(grouped.items()):
        gates = [value for row in rows for value in row["by_member"]]
        atoms = [value for row in rows for value in row["atom_probability_by_member"]]
        amplitudes = [value for row in rows for value in row["conditional_by_member"]]
        groups.append({
            "split": split,
            "action": action,
            "step": step,
            "rgb_changed": changed,
            "member_values": len(gates),
            "gate_mean": sum(gates) / len(gates),
            "gate_min": min(gates),
            "gate_max": max(gates),
            "exact_zero_rate": sum(value == 0 for value in gates) / len(gates),
            "atom_probability_mean": sum(atoms) / len(atoms),
            "conditional_mean": sum(amplitudes) / len(amplitudes),
        })
    return {
        "diagnostic_only": True,
        "rows": total,
        "input": "exact exp148 state plus current privileged pose",
        "atom_boundary": ATOM_BOUNDARY,
        "by_action_context": groups,
        "artifacts": {"rows": output_path.name},
    }


def _canonical_contexts(statistics):
    rows = statistics["by_action_context"]
    return {
        split: {
            "blocked": next(
                row for row in rows if row["split"] == split
                and row["action"] == 2 and row["step"] == 0
                and not row["rgb_changed"]
            ),
            "free": next(
                row for row in rows if row["split"] == split
                and row["action"] == 2 and row["step"] == 1
                and row["rgb_changed"]
            ),
            "contact": [
                row for row in rows if row["split"] == split
                and row["action"] == 3 and row["step"] in (0, 2)
                and row["rgb_changed"]
            ],
        }
        for split in one_step.SPLITS
    }


def build_parser():
    parser = relational.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--exp165-reference", type=Path, default=DEFAULT_EXP165_REFERENCE
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
    command = os.environ.get("EXP166_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp162_protocol_reference": str(args.exp162_reference),
        "exp165_reference": str(args.exp165_reference),
        "budgets": core._jsonable(vars(args)),
        "fixed_protocol": PROTOCOL,
        "objective": OBJECTIVE,
        "status": "running",
        "exit_code": None,
        "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(
        args.out / "progress.jsonl", args.progress_interval
    ) as journal:
        try:
            journal.update("initialize", 0, 4, operation="safe_exp153_load")
            baseline, _ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            initial_backbone = {
                name: value.detach().clone()
                for name, value in baseline.state_dict().items()
            }
            journal.update("initialize", 1, 4, operation="load_exp162_protocol")
            exp162_reference, reference_rows = relational._load_exp162_reference(
                args.exp162_reference
            )
            journal.update("initialize", 2, 4, operation="load_exp165_reference")
            exp165_reference = _load_exp165_reference(args.exp165_reference)
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
            if next(baseline.parameters()).device.type != torch.device(config.device).type:
                raise ValueError("checkpoint device and requested config disagree")
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 4, 4, device=config.device)
            corpus, _fit, _validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            episodes = linear_probe._ordered_episodes(replay, corpus)
            train_episodes, heldout_episodes = linear_probe.episode_disjoint_split(
                episodes
            )
            train_ids = {
                episode.uid for values in train_episodes.values() for episode in values
            }
            heldout_ids = {
                episode.uid for values in heldout_episodes.values() for episode in values
            }
            if train_ids & heldout_ids or len(train_ids | heldout_ids) != corpus["episodes"]:
                raise AssertionError("75/25 episode split is not complete and disjoint")
            sidecar = pose_probe._build_pose_sidecar(episodes, args, journal)
            if len(sidecar) != corpus["transitions"]:
                raise AssertionError("pose sidecar does not cover every transition")
            coverage = _audit_counts({"all": replay._episodes()})
            action_counts = {
                action: {key: row[key] for key in ("total", "rgb_changed", "rgb_no_change")}
                for action, row in coverage["actions"].items()
            }
            train = relational._extract_dataset(
                baseline, train_episodes, sidecar, journal, "extract_train"
            )
            heldout = relational._extract_dataset(
                baseline, heldout_episodes, sidecar, journal, "extract_heldout"
            )
            weights, atom_counts = atom_class_weights(
                train["actions"], train["target"].transpose(0, 1), 5
            )
            weights = weights.to(config.device)
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "canonical_exp162_protocol_reference": args.exp162_reference == DEFAULT_EXP162_REFERENCE,
                "exp162_reference_head": exp162_reference["analysis_git_head"] == EXPECTED_EXP162_HEAD,
                "canonical_exp165_reference": args.exp165_reference == DEFAULT_EXP165_REFERENCE,
                "exp165_reference_head": exp165_reference["analysis_git_head"] == EXPECTED_EXP165_HEAD,
                "default_budgets": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "baseline_config": metadata["config"] == FIXED_CONFIG,
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_and_counts": corpus["default_corpus_verified"] and action_counts == FIXED_CORPUS["action_counts"],
                "episode_split_75_25": all(
                    len(train_episodes[name]) == 384 and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                ),
                "episode_disjoint": not bool(train_ids & heldout_ids),
                "sidecar_exact_coverage": len(sidecar) == corpus["transitions"],
                "probe_budget": args.probe_updates == 400 and args.probe_batch_size == 256,
                "atom_weights_train_only": int(atom_counts.sum()) == train["target"].numel(),
            }
            split_metadata = {
                "train_episodes": len(train_ids),
                "heldout_episodes": len(heldout_ids),
                "train_uid_digest": linear_probe._uid_digest(train_episodes),
                "heldout_uid_digest": linear_probe._uid_digest(heldout_episodes),
                "overlap": 0,
            }
            sidecar_metadata = {
                "key": "exact episode UID + zero-based transition step",
                "snapshot_timing": "before action",
                "rows": len(sidecar),
                "digest": relational._sidecar_digest(sidecar),
                "features": [
                    "box_x-agent_x", "box_y-agent_y", "goal_x-box_x", "goal_y-box_y",
                    "agent_dir_0", "agent_dir_1", "agent_dir_2", "agent_dir_3",
                ],
                "relation_normalization_grid_span": relational.GRID_SPAN,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                exp162_protocol_reference_metadata=exp162_reference,
                exp165_reference_metadata=exp165_reference,
                protocol_match=matching,
                episode_split=split_metadata,
                pose_sidecar=sidecar_metadata,
                action_counts=action_counts,
                atom_counts=atom_counts.tolist(),
                atom_class_weights=weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)
            datasets = {
                "train": linear_probe._target_summary(train),
                "heldout": linear_probe._target_summary(heldout),
            }
            trace = core.TraceWriter(args.out / "probe_losses.jsonl")
            try:
                probe, metrics = _fit_probe(
                    train, heldout, weights, config, args, journal, trace
                )
            finally:
                trace.close()
            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            if not backbone_unchanged:
                raise AssertionError("frozen exp153 backbone changed")
            journal.update("probe_checkpoint", 0, 1)
            checkpoint_path = args.out / "hurdle_amplitude_probe.pt"
            torch.save({
                "format_version": 1,
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "objective": OBJECTIVE,
                "z_dim": config.z_dim,
                "h_dim": config.h_dim,
                "pose_dim": POSE_DIM,
                "ensemble_size": config.ensemble_size,
                "hidden_width": HIDDEN_WIDTH,
                "atom_boundary": ATOM_BOUNDARY,
                "atom_counts": atom_counts,
                "atom_class_weights": weights.cpu(),
                "probe_state_dict": probe.state_dict(),
            }, checkpoint_path)
            journal.update("probe_checkpoint", 1, 1)
            model = _installed_model(baseline, probe)
            journal.update("one_step_hurdle", 0, 120)
            diagnostic, candidate_rows = pose_probe._diagnose(
                model, journal, args.out / "hurdle_one_step_rows.jsonl"
            )
            journal.update("one_step_hurdle", 120, 120)
            alignment = relational._assert_protocol_alignment(
                diagnostic, exp162_reference["one_step"], candidate_rows, reference_rows
            )
            matching["canonical_evaluator_rows"] = alignment["ordered_protocol_rows_equal"]
            exact_protocol = all(matching.values())
            source_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["source"], diagnostic["splits"]["source"], exact_protocol
            )
            unseen_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["unseen"], diagnostic["splits"]["unseen"], exact_protocol
            )
            gate = bool(source_pass and unseen_pass)
            statistics = _gate_statistics(
                model, args.out / "canonical_gate_rows.jsonl", journal
            )
            comparison = pose_probe._comparison(diagnostic, exp165_reference["one_step"])
            heldout_atom = metrics["heldout"]
            atom_good = bool(
                heldout_atom["balanced_accuracy"] is not None
                and heldout_atom["balanced_accuracy"] >= 0.8
                and heldout_atom["recall_zero"] >= 0.8
                and heldout_atom["recall_positive"] >= 0.8
            )
            if gate:
                outcome = "hurdle_pose_passes"
                conclusion = (
                    "The hurdle transition target/objective is sufficient with privileged "
                    "Markov pose; the next experiment should remove the privilege."
                )
            elif not atom_good:
                outcome = "atom_classification_failure"
                conclusion = (
                    "Heldout atom classification is poor and both physics gates do not pass; "
                    "the next bottleneck is state/target classification."
                )
            else:
                outcome = "conditional_amplitude_delta_failure"
                conclusion = (
                    "Heldout atom classification is good but physics still fails; conditional "
                    "amplitude or raw-delta interaction remains the bottleneck."
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "privileged evaluator-only hurdle amplitude diagnostic",
                "interpretation_limit": "No deployable solution, composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "protocol_alignment": alignment,
                "objective": OBJECTIVE,
                "episode_split": split_metadata,
                "pose_sidecar": sidecar_metadata,
                "corpus": corpus,
                "action_counts": action_counts,
                "atom_counts": atom_counts.tolist(),
                "atom_class_weights": weights.tolist(),
                "target_datasets": datasets,
                "probe_metrics": metrics,
                "atom_quality_descriptive_floor": 0.8,
                "heldout_atom_quality_good": atom_good,
                "one_step": diagnostic,
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "hurdle_amplitude_gate": gate,
                "gate_statistics": statistics,
                "canonical_gate_contexts": _canonical_contexts(statistics),
                "exp165_reference": exp165_reference,
                "exp165_comparison": comparison,
                "frozen_backbone_unchanged": backbone_unchanged,
                "outcome": outcome,
                "conclusion": conclusion,
                "controls": {
                    "only_causal_change_from_exp165": "hurdle target/output/loss",
                    "privileged_pose_unchanged": True,
                    "snapshot_before_transition": True,
                    "sidecar_keyed_by_episode_uid_and_step": True,
                    "atom_weights_train_only": True,
                    "fixed_atom_boundary": ATOM_BOUNDARY,
                    "posthoc_threshold_selection": False,
                    "source_or_unseen_leakage": False,
                    "action_rules_encoded": False,
                    "outcome_or_blocked_labels_encoded": False,
                    "object_condition_branches": False,
                    "raw_deltas_before_native_gate": True,
                    "analytic_targets_detached": True,
                    "backbone_frozen": True,
                    "exp165_retrained": False,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "rows": "hurdle_one_step_rows.jsonl",
                    "gate_rows": "canonical_gate_rows.jsonl",
                    "progress": "progress.jsonl",
                    "manifest": "manifest.json",
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
