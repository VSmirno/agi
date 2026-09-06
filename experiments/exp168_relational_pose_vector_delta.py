"""Privileged direct vector-delta transition probe after scalar-gate failure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import json
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
from experiments import exp161_amplitude_input_probe as linear_probe
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments import exp164_relational_slot_probe as relational
from experiments import exp165_relational_pose_probe as pose_probe
from experiments import exp166_hurdle_amplitude_probe as hurdle
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import CoreWorldModel, LatentState, Prediction
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_EXP162_REFERENCE = relational.DEFAULT_EXP162_REFERENCE
EXPECTED_EXP162_HEAD = relational.EXPECTED_EXP162_HEAD
DEFAULT_EXP165_REFERENCE = hurdle.DEFAULT_EXP165_REFERENCE
EXPECTED_EXP165_HEAD = hurdle.EXPECTED_EXP165_HEAD
DEFAULT_EXP166_REFERENCE = Path(
    "output_to_user/core/exp166-hurdle-amplitude-probe-001/results.json"
)
EXPECTED_EXP166_HEAD = "e1e1bc4fc80f2844eebf1f2537518fd007bdfc52"
PROTOCOL = dict(residual.PROTOCOL)
POSE_DIM = pose_probe.POSE_DIM
HIDDEN_WIDTH = pose_probe.HIDDEN_WIDTH
OBJECTIVE = {
    "target": "detached actual_next_z - current_z, broadcast to ensemble members",
    "input": "frozen current z + carried hidden + privileged 8D BEFORE pose",
    "architecture": (
        "separate per-action Linear(z_dim+h_dim+8,128)->ReLU->"
        "Linear(128,ensemble_size*z_dim) direct member deltas"
    ),
    "weight": "fixed train-only weight[action, observed_rgb_change]",
    "denominator": "ordinary member*batch*latent count; no sampled renormalization",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_diagnostic": True,
    "jepa_style": "frozen encoder next-latent target with no target gradient",
}


class RelationalPoseVectorDelta(torch.nn.Module):
    """Predict a full latent displacement for every member and primitive action."""

    def __init__(self, z_dim: int, h_dim: int, heads: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        width = z_dim + h_dim + POSE_DIM
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, heads * z_dim),
            )
            for _ in range(5)
        )

    def forward(self, z, hidden, pose, actions):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if pose.shape != (len(z), POSE_DIM):
            raise ValueError("pose must have shape [batch,8]")
        if actions.shape != (len(z),) or actions.dtype != torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden, pose), dim=-1)
        all_actions = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = all_actions.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads * self.z_dim)
        ).squeeze(1)
        return selected.reshape(len(z), self.heads, self.z_dim).permute(1, 0, 2)


def apply_member_deltas(
    prediction: Prediction, current_z: torch.Tensor, member_delta: torch.Tensor
) -> Prediction:
    """Replace only latent output with current state plus direct member deltas."""

    if member_delta.ndim != 3 or member_delta.shape[1:] != current_z.shape:
        raise ValueError("member_delta must have shape [members,batch,z_dim]")
    member_z = current_z.unsqueeze(0) + member_delta
    return replace(
        prediction,
        member_z=member_z,
        next_state=replace(prediction.next_state, z=member_z.mean(0)),
        uncertainty=member_z.var(0, unbiased=False).mean(-1),
    )


def detached_vector_target(actual_z, current_z, heads: int):
    """Return a no-gradient member-broadcast displacement target."""

    if actual_z.shape != current_z.shape or actual_z.ndim != 2 or heads <= 0:
        raise ValueError("actual/current z must share [batch,z_dim] and heads be positive")
    return (actual_z - current_z).detach().unsqueeze(0).expand(heads, -1, -1)


def weighted_vector_mse(prediction, target, actions, changed, class_weights):
    """Fixed population-weighted direct vector error with ordinary denominator."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction/target must share [members,batch,z_dim]")
    if actions.shape != (prediction.shape[1],) or actions.dtype != torch.long:
        raise ValueError("actions must be long with shape [batch]")
    if changed.shape != actions.shape or changed.dtype != torch.bool:
        raise ValueError("changed must be bool with shape [batch]")
    if class_weights.ndim != 2 or class_weights.shape[1] != 2:
        raise ValueError("class weights must have shape [actions,2]")
    weights = class_weights[actions, changed.long()]
    return ((prediction - target).square() * weights[None, :, None]).mean()


class VectorDeltaWorldModel(CoreWorldModel):
    """Frozen exp153 plumbing with direct vector latent transitions only."""

    def __init__(self, *args, vector_probe: RelationalPoseVectorDelta, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_probe = vector_probe
        self._current_pose = None

    def set_relations(self, pose: torch.Tensor) -> None:
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
            raise ValueError("current pose must have shape [batch,8]")
        self._current_pose = pose.detach()

    def step(self, state: LatentState, actions: torch.Tensor) -> Prediction:
        pose = self._current_pose
        if pose is None or pose.shape[0] != state.z.shape[0]:
            raise RuntimeError("pose context was not installed for this state")
        native = super().step(state, actions)
        member_delta = self.vector_probe(
            state.z,
            state.hidden,
            pose.to(device=state.z.device, dtype=state.z.dtype),
            actions,
        )
        return apply_member_deltas(native, state.z, member_delta)


def _installed_model(baseline, probe):
    parameter = next(baseline.parameters())
    model = VectorDeltaWorldModel(
        CoreEncoder(baseline.encoder.z_dim),
        dict(baseline.schemas),
        baseline.h_dim,
        baseline.heads,
        normalize_sensor_condition=baseline.normalize_sensor_condition,
        predict_sensor_delta=baseline.predict_sensor_delta,
        vector_probe=probe,
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate_keys = set(model.state_dict())
    transferable = {
        name: value for name, value in baseline.state_dict().items()
        if name in candidate_keys
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("vector_probe.") for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected frozen transfer mismatch: {incompatible}")
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _extract_dataset(model, episodes_by_layout, sidecar, journal, stage):
    names = ("z", "hidden", "pose", "target", "actions", "changed")
    chunks = {name: [] for name in names}
    total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    parameter = next(model.parameters())
    journal.update(stage, 0, total)
    for layout in temporal.SOURCE_LAYOUTS:
        for episode in episodes_by_layout[layout]:
            transitions = episode.transitions
            observations = [transitions[0].before, *[row.after for row in transitions]]
            rgb = torch.tensor(
                np.stack([obs.rgb for obs in observations]),
                device=parameter.device, dtype=parameter.dtype,
            ) / 255
            z_sequence = model.encoder(rgb)
            sensors = torch.tensor(
                np.stack([obs.sensors for obs in observations]),
                device=parameter.device, dtype=parameter.dtype,
            )
            masks = torch.tensor(
                np.stack([obs.sensor_mask for obs in observations]),
                device=parameter.device, dtype=torch.bool,
            )
            state = LatentState(
                z_sequence[0:1], torch.where(masks[0:1], sensors[0:1], 0.0),
                masks[0:1], z_sequence.new_zeros(1, model.h_dim),
                transitions[0].before.schema,
            )
            rows = {name: [] for name in names}
            for index, transition in enumerate(transitions):
                action = torch.tensor(
                    [transition.action], device=parameter.device, dtype=torch.long
                )
                prediction = model.step(state, action)
                target = (z_sequence[index + 1] - state.z[0]).detach()
                pose = sidecar.get((episode.uid, index))
                if pose is None:
                    raise AssertionError(f"missing pose row: {episode.uid}/{index}")
                rows["z"].append(state.z[0])
                rows["hidden"].append(state.hidden[0])
                rows["pose"].append(pose.to(parameter.device))
                rows["target"].append(target)
                rows["actions"].append(action[0])
                rows["changed"].append(torch.tensor(
                    not np.array_equal(transition.before.rgb, transition.after.rgb),
                    device=parameter.device, dtype=torch.bool,
                ))
                actual = LatentState(
                    z_sequence[index + 1:index + 2],
                    torch.where(
                        masks[index + 1:index + 2],
                        sensors[index + 1:index + 2], 0.0,
                    ),
                    masks[index + 1:index + 2],
                    z_sequence.new_zeros(1, model.h_dim), state.schema,
                )
                state = one_step._teacher_forced_next(prediction, actual)
            for name, values in rows.items():
                chunks[name].append(torch.stack(values).cpu())
            completed += 1
            journal.update(stage, completed, total, layout=layout)
    return {name: torch.cat(values) for name, values in chunks.items()}


def _action_change_counts(dataset):
    counts = {}
    for action in range(5):
        selected = dataset["actions"] == action
        changed = int((dataset["changed"] & selected).sum())
        total = int(selected.sum())
        counts[str(action)] = {
            "total": total,
            "rgb_changed": changed,
            "rgb_no_change": total - changed,
        }
    return counts


@torch.inference_mode()
def _vector_metrics(probe, dataset, class_weights, device):
    squared = weighted = persistence = 0.0
    count = 0
    for start in range(0, len(dataset["actions"]), 4096):
        stop = min(start + 4096, len(dataset["actions"]))
        actions = dataset["actions"][start:stop].to(device)
        changed = dataset["changed"][start:stop].to(device)
        prediction = probe(
            dataset["z"][start:stop].to(device),
            dataset["hidden"][start:stop].to(device),
            dataset["pose"][start:stop].to(device), actions,
        )
        target = detached_vector_target(
            dataset["target"][start:stop].to(device),
            torch.zeros_like(dataset["target"][start:stop], device=device),
            probe.heads,
        )
        error = (prediction - target).square()
        squared += float(error.sum())
        weighted += float(
            (error * class_weights[actions, changed.long()][None, :, None]).sum()
        )
        persistence += float(target.square().sum())
        count += error.numel()
    groups = {}
    for action in range(5):
        for event in (False, True):
            mask = (dataset["actions"] == action) & (dataset["changed"] == event)
            key = f"action{action}_{'changed' if event else 'nochange'}"
            if not int(mask.sum()):
                groups[key] = {"transitions": 0, "mse": None, "persistence_mse": None}
                continue
            actions = dataset["actions"][mask].to(device)
            prediction = probe(
                dataset["z"][mask].to(device),
                dataset["hidden"][mask].to(device),
                dataset["pose"][mask].to(device), actions,
            )
            target = dataset["target"][mask].to(device).unsqueeze(0).expand(
                probe.heads, -1, -1
            )
            mse = float((prediction - target).square().mean())
            baseline = float(target.square().mean())
            groups[key] = {
                "transitions": int(mask.sum()),
                "mse": mse,
                "persistence_mse": baseline,
                "prediction_to_persistence_ratio": mse / baseline if baseline > 0 else None,
            }
    return {
        "mse": squared / count,
        "weighted_mse": weighted / count,
        "persistence_mse": persistence / count,
        "prediction_to_persistence_ratio": squared / persistence if persistence > 0 else None,
        "member_latent_elements": count,
        "groups": groups,
    }


def _fit_probe(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = RelationalPoseVectorDelta(
            config.z_dim, config.h_dim, config.ensemble_size
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_vector", 0, args.probe_updates)
    probe.train()
    for update in range(1, args.probe_updates + 1):
        indices = torch.randint(
            len(train["actions"]), (args.probe_batch_size,), generator=generator
        )
        actions = train["actions"][indices].to(device)
        changed = train["changed"][indices].to(device)
        prediction = probe(
            train["z"][indices].to(device),
            train["hidden"][indices].to(device),
            train["pose"][indices].to(device), actions,
        )
        target = train["target"][indices].to(device).unsqueeze(0).expand(
            config.ensemble_size, -1, -1
        ).detach()
        loss = weighted_vector_mse(
            prediction, target, actions, changed, class_weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        trace.write({"update": update, "loss": value})
        journal.update("fit_vector", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0], "loss_last": losses[-1],
        "updates": args.probe_updates, "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _vector_metrics(probe, train, class_weights, device),
        "heldout": _vector_metrics(probe, heldout, class_weights, device),
    }


def _load_exp166_reference(path: Path):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp166 reference: {error}") from error
    if payload.get("analysis_git_head") != EXPECTED_EXP166_HEAD:
        raise ValueError("exp166 reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError("exp166 reference is not exact protocol")
    if not isinstance(payload.get("one_step"), Mapping):
        raise ValueError("exp166 reference lacks one-step results")
    return {
        "path": str(path),
        "analysis_git_head": payload["analysis_git_head"],
        "probe_metrics": payload["probe_metrics"],
        "one_step": payload["one_step"],
        "gate": payload["hurdle_amplitude_gate"],
    }


def build_parser():
    parser = relational.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--exp165-reference", type=Path, default=DEFAULT_EXP165_REFERENCE
    )
    parser.add_argument(
        "--exp166-reference", type=Path, default=DEFAULT_EXP166_REFERENCE
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
    command = os.environ.get("EXP168_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv), "exact_command": command, "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp162_protocol_reference": str(args.exp162_reference),
        "exp165_reference": str(args.exp165_reference),
        "exp166_reference": str(args.exp166_reference),
        "budgets": core._jsonable(vars(args)), "fixed_protocol": PROTOCOL,
        "objective": OBJECTIVE, "status": "running",
        "exit_code": None, "exit_status": None,
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
            exp165_reference = hurdle._load_exp165_reference(args.exp165_reference)
            journal.update("initialize", 3, 4, operation="load_exp166_reference")
            exp166_reference = _load_exp166_reference(args.exp166_reference)
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
            full_coverage = _audit_counts({"all": replay._episodes()})
            full_counts = {
                action: {key: row[key] for key in ("total", "rgb_changed", "rgb_no_change")}
                for action, row in full_coverage["actions"].items()
            }
            train = _extract_dataset(
                baseline, train_episodes, sidecar, journal, "extract_train"
            )
            heldout = _extract_dataset(
                baseline, heldout_episodes, sidecar, journal, "extract_heldout"
            )
            train_counts = _action_change_counts(train)
            class_weights = auxiliary.action_class_weights(train_counts).to(config.device)
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "canonical_exp162_protocol_reference": args.exp162_reference == DEFAULT_EXP162_REFERENCE,
                "exp162_reference_head": exp162_reference["analysis_git_head"] == EXPECTED_EXP162_HEAD,
                "canonical_exp165_reference": args.exp165_reference == DEFAULT_EXP165_REFERENCE,
                "exp165_reference_head": exp165_reference["analysis_git_head"] == EXPECTED_EXP165_HEAD,
                "canonical_exp166_reference": args.exp166_reference == DEFAULT_EXP166_REFERENCE,
                "exp166_reference_head": exp166_reference["analysis_git_head"] == EXPECTED_EXP166_HEAD,
                "default_budgets": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "baseline_config": metadata["config"] == FIXED_CONFIG,
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_and_counts": corpus["default_corpus_verified"] and full_counts == FIXED_CORPUS["action_counts"],
                "episode_split_75_25": all(
                    len(train_episodes[name]) == 384 and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                ),
                "episode_disjoint": not bool(train_ids & heldout_ids),
                "sidecar_exact_coverage": len(sidecar) == corpus["transitions"],
                "class_weights_train_only": sum(row["total"] for row in train_counts.values()) == len(train["actions"]),
                "probe_budget": args.probe_updates == 400 and args.probe_batch_size == 256,
            }
            split_metadata = {
                "train_episodes": len(train_ids), "heldout_episodes": len(heldout_ids),
                "train_uid_digest": linear_probe._uid_digest(train_episodes),
                "heldout_uid_digest": linear_probe._uid_digest(heldout_episodes),
                "overlap": 0,
            }
            sidecar_metadata = {
                "key": "exact episode UID + zero-based transition step",
                "snapshot_timing": "before action", "rows": len(sidecar),
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
                exp166_reference_metadata=exp166_reference,
                protocol_match=matching, episode_split=split_metadata,
                pose_sidecar=sidecar_metadata, full_action_counts=full_counts,
                train_action_counts=train_counts, class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)
            trace = core.TraceWriter(args.out / "probe_losses.jsonl")
            try:
                probe, metrics = _fit_probe(
                    train, heldout, class_weights, config, args, journal, trace
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
            checkpoint_path = args.out / "relational_pose_vector_delta.pt"
            torch.save({
                "format_version": 1,
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "objective": OBJECTIVE,
                "z_dim": config.z_dim, "h_dim": config.h_dim,
                "pose_dim": POSE_DIM, "ensemble_size": config.ensemble_size,
                "hidden_width": HIDDEN_WIDTH,
                "train_action_counts": train_counts,
                "class_weights": class_weights.cpu(),
                "probe_state_dict": probe.state_dict(),
            }, checkpoint_path)
            journal.update("probe_checkpoint", 1, 1)
            model = _installed_model(baseline, probe)
            journal.update("one_step_vector", 0, 120)
            diagnostic, candidate_rows = pose_probe._diagnose(
                model, journal, args.out / "vector_delta_one_step_rows.jsonl"
            )
            journal.update("one_step_vector", 120, 120)
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
            comparison165 = pose_probe._comparison(
                diagnostic, exp165_reference["one_step"]
            )
            comparison166 = pose_probe._comparison(
                diagnostic, exp166_reference["one_step"]
            )
            if gate:
                outcome = "direct_vector_transition_passes"
                conclusion = (
                    "A direct privileged vector transition passes both local splits; "
                    "the scalar amplitude factorization was the wall."
                )
            elif comparison166["categorical_unchanged"]:
                outcome = "vector_categorical_failures_unchanged"
                conclusion = (
                    "Direct vectors leave exp166 categorical failures unchanged; current "
                    "representation/objective/training is insufficient and an explicit "
                    "object-transition bottleneck is licensed."
                )
            else:
                outcome = "vector_improvement_only"
                conclusion = (
                    "Direct vectors change categorical behavior without passing both splits; "
                    "report exact deltas before the object-transition follow-up."
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "privileged direct vector latent-transition diagnostic",
                "interpretation_limit": "No deployable solution, composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exact_command": command, "exact_protocol": exact_protocol,
                "protocol_match": matching, "protocol_alignment": alignment,
                "objective": OBJECTIVE, "episode_split": split_metadata,
                "pose_sidecar": sidecar_metadata, "corpus": corpus,
                "full_action_counts": full_counts,
                "train_action_counts": train_counts,
                "class_weights": class_weights.tolist(),
                "probe_metrics": metrics,
                "one_step": diagnostic,
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "direct_vector_transition_gate": gate,
                "exp165_reference": exp165_reference,
                "exp166_reference": exp166_reference,
                "exp165_comparison": comparison165,
                "exp166_comparison": comparison166,
                "frozen_backbone_unchanged": backbone_unchanged,
                "outcome": outcome, "conclusion": conclusion,
                "controls": {
                    "only_causal_change_from_exp166": "direct member vector delta target/output/loss",
                    "scalar_amplitude": False, "atom_logit": False,
                    "threshold": False, "hurdle": False, "mixture": False,
                    "privileged_pose_unchanged": True,
                    "snapshot_before_transition": True,
                    "class_weights_train_only": True,
                    "future_rgb_only_selects_fixed_weight": True,
                    "source_or_unseen_leakage": False,
                    "rules_or_outcome_labels": False,
                    "target_detached": True, "backbone_frozen": True,
                    "exp165_retrained": False, "exp166_retrained": False,
                    "mpc": False, "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "rows": "vector_delta_one_step_rows.jsonl",
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
