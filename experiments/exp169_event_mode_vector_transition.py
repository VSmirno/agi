"""Compose a frozen direct vector transition with a learned generic event mode."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import shlex
import sys
import time

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
from experiments import exp167_hurdle_oracle_swap as oracle_swap
from experiments import exp168_relational_pose_vector_delta as vector_delta
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
DEFAULT_EXP168_CHECKPOINT = Path(
    "output_to_user/core/exp168-relational-pose-vector-delta-001/"
    "relational_pose_vector_delta.pt"
)
DEFAULT_EXP168_REFERENCE = Path(
    "output_to_user/core/exp168-relational-pose-vector-delta-001/results.json"
)
EXPECTED_EXP168_HEAD = "a5349328a1541f4c80ab1c19ea534961dcf7eea8"
PROTOCOL = dict(residual.PROTOCOL)
POSE_DIM = pose_probe.POSE_DIM
HIDDEN_WIDTH = pose_probe.HIDDEN_WIDTH
EVENT_BOUNDARY = 0.5
OBJECTIVE = {
    "target": "generic observed RGB changed versus byte-identical transition",
    "input": "frozen current z + carried hidden + privileged 8D BEFORE pose",
    "architecture": (
        "separate per-action Linear(z_dim+h_dim+8,128)->ReLU->Linear(128,1)"
    ),
    "loss": "fixed train-only per-action class-balanced BCE",
    "installed_transition": (
        "event probability >=0.5 uses frozen exp168 member delta; otherwise literal z persistence"
    ),
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_diagnostic": True,
}


def event_class_weights(counts: Mapping, action_count: int = 5) -> torch.Tensor:
    """Return deterministic balanced weights, including safe single-class actions."""

    return auxiliary.action_class_weights(counts, n_actions=action_count)


def balanced_event_bce(
    logits: torch.Tensor,
    actions: torch.Tensor,
    changed: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Fixed population-weighted event BCE with an ordinary batch denominator."""

    if logits.ndim != 1:
        raise ValueError("logits must have shape [batch]")
    if actions.shape != logits.shape or actions.dtype != torch.long:
        raise ValueError("actions must be long with shape [batch]")
    if changed.shape != logits.shape or changed.dtype != torch.bool:
        raise ValueError("changed must be bool with shape [batch]")
    if class_weights.ndim != 2 or class_weights.shape[1] != 2:
        raise ValueError("class weights must have shape [actions,2]")
    if bool(((actions < 0) | (actions >= len(class_weights))).any()):
        raise ValueError("action is outside class weights")
    weights = class_weights[actions, changed.long()]
    losses = functional.binary_cross_entropy_with_logits(
        logits, changed.to(logits.dtype), reduction="none"
    )
    return (losses * weights).mean()


class RelationalPoseEventHead(torch.nn.Module):
    """Predict one generic change probability for each action and state."""

    def __init__(self, z_dim: int, h_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        width = z_dim + h_dim + POSE_DIM
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, 1),
            )
            for _ in range(5)
        )

    def logits(self, z, hidden, pose, actions):
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
        return all_actions.gather(1, actions[:, None, None]).flatten()

    def forward(self, z, hidden, pose, actions):
        return self.logits(z, hidden, pose, actions).sigmoid()


def apply_event_mode(
    prediction: Prediction,
    current_z: torch.Tensor,
    frozen_member_delta: torch.Tensor,
    event_probability: torch.Tensor,
) -> Prediction:
    """Use the frozen vector exactly on change, otherwise persist z literally."""

    if frozen_member_delta.ndim != 3 or frozen_member_delta.shape[1:] != current_z.shape:
        raise ValueError("frozen_member_delta must have shape [members,batch,z_dim]")
    if event_probability.shape != (len(current_z),):
        raise ValueError("event_probability must have shape [batch]")
    if not torch.isfinite(event_probability).all():
        raise ValueError("event_probability must be finite")
    selected = torch.where(
        (event_probability >= EVENT_BOUNDARY)[None, :, None],
        frozen_member_delta,
        torch.zeros_like(frozen_member_delta),
    )
    result = vector_delta.apply_member_deltas(prediction, current_z, selected)
    change = event_probability >= EVENT_BOUNDARY
    exact_next_z = torch.where(change[:, None], result.next_state.z, current_z)
    return replace(result, next_state=replace(result.next_state, z=exact_next_z))


class EventModeVectorWorldModel(CoreWorldModel):
    """Frozen exp153 plumbing and exp168 deltas composed with one event bit."""

    def __init__(self, *args, vector_probe, event_probe, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_probe = vector_probe
        self.event_probe = event_probe
        self._current_pose = None
        self.event_trace: list[dict] | None = None

    def set_relations(self, pose: torch.Tensor) -> None:
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
            raise ValueError("current pose must have shape [batch,8]")
        self._current_pose = pose.detach()

    def step(self, state: LatentState, actions: torch.Tensor) -> Prediction:
        pose = self._current_pose
        if pose is None or pose.shape[0] != state.z.shape[0]:
            raise RuntimeError("pose context was not installed for this state")
        pose = pose.to(device=state.z.device, dtype=state.z.dtype)
        native = CoreWorldModel.step(self, state, actions)
        frozen_delta = self.vector_probe(state.z, state.hidden, pose, actions)
        probability = self.event_probe(state.z, state.hidden, pose, actions)
        if self.event_trace is not None:
            if len(actions) != 1:
                raise RuntimeError("canonical event trace requires batch size one")
            self.event_trace.append({
                "action": int(actions.item()),
                "event_probability": float(probability.item()),
            })
        return apply_event_mode(native, state.z, frozen_delta, probability)


def _load_exp168_checkpoint(path: Path, baseline) -> tuple[torch.nn.Module, dict]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"could not safely load exp168 checkpoint: {error}") from error
    required = {
        "format_version", "analysis_git_head", "baseline_checkpoint_git_head",
        "objective", "z_dim", "h_dim", "pose_dim", "ensemble_size",
        "hidden_width", "train_action_counts", "class_weights", "probe_state_dict",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("exp168 checkpoint fields mismatch")
    expected = {
        "format_version": 1,
        "analysis_git_head": EXPECTED_EXP168_HEAD,
        "baseline_checkpoint_git_head": EXPECTED_BASELINE_HEAD,
        "z_dim": baseline.encoder.z_dim,
        "h_dim": baseline.h_dim,
        "pose_dim": POSE_DIM,
        "ensemble_size": baseline.heads,
        "hidden_width": HIDDEN_WIDTH,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("exp168 checkpoint metadata mismatch")
    state = payload["probe_state_dict"]
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("exp168 probe state must contain tensors only")
    parameter = next(baseline.parameters())
    probe = vector_delta.RelationalPoseVectorDelta(
        baseline.encoder.z_dim, baseline.h_dim, baseline.heads
    ).to(device=parameter.device, dtype=parameter.dtype)
    probe.load_state_dict(state, strict=True)
    return probe.eval().requires_grad_(False), payload


def _load_exp168_reference(path: Path):
    try:
        payload = json.loads(path.read_text())
        rows_path = path.parent / payload["artifacts"]["rows"]
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp168 reference: {error}") from error
    if payload.get("analysis_git_head") != EXPECTED_EXP168_HEAD:
        raise ValueError("exp168 reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError("exp168 reference is not exact protocol")
    if not isinstance(payload.get("one_step"), Mapping) or len(rows) != 120:
        raise ValueError("exp168 reference requires one-step result and 120 rows")
    return {
        "path": str(path),
        "analysis_git_head": payload["analysis_git_head"],
        "probe_metrics": payload["probe_metrics"],
        "one_step": payload["one_step"],
        "gate": payload["direct_vector_transition_gate"],
    }, rows


def _installed_vector_model(baseline, vector_probe):
    return vector_delta._installed_model(baseline, vector_probe)


def _installed_event_model(baseline, vector_probe, event_probe):
    parameter = next(baseline.parameters())
    model = EventModeVectorWorldModel(
        CoreEncoder(baseline.encoder.z_dim),
        dict(baseline.schemas),
        baseline.h_dim,
        baseline.heads,
        normalize_sensor_condition=baseline.normalize_sensor_condition,
        predict_sensor_delta=baseline.predict_sensor_delta,
        vector_probe=vector_probe,
        event_probe=event_probe,
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate_keys = set(model.state_dict())
    transferable = {
        name: value for name, value in baseline.state_dict().items()
        if name in candidate_keys
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith(("vector_probe.", "event_probe."))
        for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected frozen transfer mismatch: {incompatible}")
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def _event_metrics(probe, dataset, class_weights, device):
    logits = []
    for start in range(0, len(dataset["actions"]), 4096):
        stop = min(start + 4096, len(dataset["actions"]))
        actions = dataset["actions"][start:stop].to(device)
        logits.append(probe.logits(
            dataset["z"][start:stop].to(device),
            dataset["hidden"][start:stop].to(device),
            dataset["pose"][start:stop].to(device),
            actions,
        ).cpu())
    logits = torch.cat(logits)
    probabilities = logits.sigmoid()
    predicted = probabilities >= EVENT_BOUNDARY
    changed = dataset["changed"]

    def summarize(mask):
        count = int(mask.sum())
        if not count:
            return {
                "transitions": 0, "balanced_accuracy": None,
                "changed_recall": None, "nochange_recall": None,
                "vector_prediction_rate": None, "literal_persistence_rate": None,
                "mean_event_probability": None,
            }
        actual = changed[mask]
        guess = predicted[mask]
        positive = int(actual.sum())
        negative = count - positive
        changed_recall = (
            float((guess & actual).sum()) / positive if positive else None
        )
        nochange_recall = (
            float((~guess & ~actual).sum()) / negative if negative else None
        )
        recalls = [value for value in (changed_recall, nochange_recall) if value is not None]
        return {
            "transitions": count,
            "rgb_changed": positive,
            "rgb_no_change": negative,
            "balanced_accuracy": sum(recalls) / len(recalls),
            "changed_recall": changed_recall,
            "nochange_recall": nochange_recall,
            "vector_prediction_rate": float(guess.float().mean()),
            "literal_persistence_rate": float((~guess).float().mean()),
            "mean_event_probability": float(probabilities[mask].mean()),
        }

    groups = {f"action{action}": summarize(dataset["actions"] == action) for action in range(5)}
    loss = balanced_event_bce(
        logits, dataset["actions"], changed, class_weights.cpu()
    )
    return {"loss": float(loss), "overall": summarize(torch.ones_like(changed)), "by_action": groups}


def _fit_event_probe(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = RelationalPoseEventHead(config.z_dim, config.h_dim).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_event", 0, args.probe_updates)
    probe.train()
    for update in range(1, args.probe_updates + 1):
        indices = torch.randint(
            len(train["actions"]), (args.probe_batch_size,), generator=generator
        )
        actions = train["actions"][indices].to(device)
        changed = train["changed"][indices].to(device)
        logits = probe.logits(
            train["z"][indices].to(device),
            train["hidden"][indices].to(device),
            train["pose"][indices].to(device),
            actions,
        )
        loss = balanced_event_bce(logits, actions, changed, class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        trace.write({"update": update, "loss": value})
        journal.update("fit_event", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0], "loss_last": losses[-1],
        "updates": args.probe_updates, "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _event_metrics(probe, train, class_weights, device),
        "heldout": _event_metrics(probe, heldout, class_weights, device),
    }


def _row_mse_alignment(candidate_rows, reference_rows, tolerance=1e-7):
    if len(candidate_rows) != len(reference_rows):
        return {"matches": False, "max_abs_mse_diff": None}
    maximum = 0.0
    for current, expected in zip(candidate_rows, reference_rows, strict=True):
        difference = abs(
            current["predicted_vs_actual_next_z_mse"]
            - expected["predicted_vs_actual_next_z_mse"]
        )
        maximum = max(maximum, difference)
    return {"matches": maximum <= tolerance, "max_abs_mse_diff": maximum}


def _evaluation_trace(trace):
    cursor = 0
    selected = []
    for split in one_step.SPLITS:
        for layout_name, spec in one_step._layout_specs()[split].items():
            layout, actions = spec[:2]
            prefix, continuation = one_step._validate_protocol(
                split, layout_name, layout, actions, one_step.SEED
            )
            for expected_action in prefix:
                record = trace[cursor]
                if record["action"] != expected_action:
                    raise AssertionError("event trace prefix action mismatch")
                cursor += 1
            for step, _canonical_action in enumerate(continuation):
                for action in range(5):
                    record = trace[cursor]
                    if record["action"] != action:
                        raise AssertionError("event trace fork action mismatch")
                    selected.append({
                        **record, "split": split, "layout": layout_name,
                        "step": step,
                    })
                    cursor += 1
    if cursor != len(trace) or len(selected) != 120:
        raise AssertionError("event trace does not match canonical evaluator")
    return selected


def _enrich_rows(rows, trace_rows, output_path):
    writer = core.TraceWriter(output_path)
    enriched = []
    try:
        for row, event in zip(rows, trace_rows, strict=True):
            keys = ("split", "layout", "step", "action")
            if any(row[key] != event[key] for key in keys):
                raise AssertionError("event probability row alignment mismatch")
            predicted = event["event_probability"] >= EVENT_BOUNDARY
            enriched_row = {
                **row,
                "event_probability": event["event_probability"],
                "predicted_change": predicted,
                "literal_persistence": not predicted,
            }
            if not predicted and not math.isclose(
                row["predicted_vs_actual_next_z_mse"],
                row["persistence_vs_actual_next_z_mse"],
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise AssertionError("no-event transition was not literal persistence")
            writer.write(enriched_row)
            enriched.append(enriched_row)
    finally:
        writer.close()
    return enriched


def _canonical_event_stats(rows):
    result = {}
    for split in one_step.SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        result[split] = {}
        for action in range(5):
            selected = [row for row in split_rows if row["action"] == action]
            probabilities = [row["event_probability"] for row in selected]
            predicted = [row["predicted_change"] for row in selected]
            result[split][f"action{action}"] = {
                "rows": len(selected),
                "actual_rgb_change_rate": sum(row["rgb_changed"] for row in selected) / len(selected),
                "vector_prediction_rate": sum(predicted) / len(predicted),
                "literal_persistence_rate": sum(not value for value in predicted) / len(predicted),
                "probability_min": min(probabilities),
                "probability_mean": sum(probabilities) / len(probabilities),
                "probability_max": max(probabilities),
            }
    return result


def build_parser():
    parser = relational.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--exp168-checkpoint", type=Path, default=DEFAULT_EXP168_CHECKPOINT
    )
    parser.add_argument(
        "--exp168-reference", type=Path, default=DEFAULT_EXP168_REFERENCE
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
    command = os.environ.get("EXP169_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv), "exact_command": command, "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp162_protocol_reference": str(args.exp162_reference),
        "exp168_checkpoint": str(args.exp168_checkpoint),
        "exp168_reference": str(args.exp168_reference),
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
            journal.update("initialize", 1, 4, operation="safe_exp168_load")
            vector_probe, vector_metadata = _load_exp168_checkpoint(
                args.exp168_checkpoint, baseline
            )
            initial_vector = {
                name: value.detach().clone()
                for name, value in vector_probe.state_dict().items()
            }
            journal.update("initialize", 2, 4, operation="load_exp162_protocol")
            exp162_reference, exp162_rows = relational._load_exp162_reference(
                args.exp162_reference
            )
            journal.update("initialize", 3, 4, operation="load_exp168_reference")
            exp168_reference, exp168_rows = _load_exp168_reference(
                args.exp168_reference
            )
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
            train_episodes, heldout_episodes = linear_probe.episode_disjoint_split(episodes)
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
            train = vector_delta._extract_dataset(
                baseline, train_episodes, sidecar, journal, "extract_train"
            )
            heldout = vector_delta._extract_dataset(
                baseline, heldout_episodes, sidecar, journal, "extract_heldout"
            )
            train_counts = vector_delta._action_change_counts(train)
            class_weights = event_class_weights(train_counts).to(config.device)
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "canonical_exp162_reference": args.exp162_reference == DEFAULT_EXP162_REFERENCE,
                "exp162_reference_head": exp162_reference["analysis_git_head"] == EXPECTED_EXP162_HEAD,
                "canonical_exp168_checkpoint": args.exp168_checkpoint == DEFAULT_EXP168_CHECKPOINT,
                "exp168_checkpoint_head": vector_metadata["analysis_git_head"] == EXPECTED_EXP168_HEAD,
                "canonical_exp168_reference": args.exp168_reference == DEFAULT_EXP168_REFERENCE,
                "exp168_reference_head": exp168_reference["analysis_git_head"] == EXPECTED_EXP168_HEAD,
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
                "event_budget": args.probe_updates == 400 and args.probe_batch_size == 256,
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
                exp168_checkpoint_metadata={
                    key: core._jsonable(value)
                    for key, value in vector_metadata.items()
                    if key != "probe_state_dict"
                },
                exp168_reference_metadata=exp168_reference,
                protocol_match=matching, episode_split=split_metadata,
                pose_sidecar=sidecar_metadata, full_action_counts=full_counts,
                train_action_counts=train_counts, class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)
            trace = core.TraceWriter(args.out / "event_losses.jsonl")
            try:
                event_probe, metrics = _fit_event_probe(
                    train, heldout, class_weights, config, args, journal, trace
                )
            finally:
                trace.close()
            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            vector_unchanged = all(
                torch.equal(vector_probe.state_dict()[name], value)
                for name, value in initial_vector.items()
            )
            if not backbone_unchanged or not vector_unchanged:
                raise AssertionError("frozen backbone or vector head changed")
            journal.update("event_checkpoint", 0, 1)
            checkpoint_path = args.out / "event_mode_head.pt"
            torch.save({
                "format_version": 1,
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exp168_checkpoint_git_head": vector_metadata["analysis_git_head"],
                "objective": OBJECTIVE,
                "z_dim": config.z_dim, "h_dim": config.h_dim,
                "pose_dim": POSE_DIM, "hidden_width": HIDDEN_WIDTH,
                "train_action_counts": train_counts,
                "class_weights": class_weights.cpu(),
                "event_probe_state_dict": event_probe.state_dict(),
            }, checkpoint_path)
            journal.update("event_checkpoint", 1, 1)

            frozen_model = _installed_vector_model(baseline, vector_probe)
            journal.update("one_step_frozen_vector", 0, 120)
            frozen_diagnostic, frozen_rows = pose_probe._diagnose(
                frozen_model, journal, args.out / "frozen_vector_one_step_rows.jsonl"
            )
            journal.update("one_step_frozen_vector", 120, 120)
            frozen_alignment = relational._assert_protocol_alignment(
                frozen_diagnostic, exp162_reference["one_step"],
                frozen_rows, exp162_rows,
            )
            frozen_row_match = _row_mse_alignment(frozen_rows, exp168_rows)
            frozen_metric_match = oracle_swap.metric_signatures_match(
                oracle_swap._metric_signature(frozen_diagnostic),
                oracle_swap._metric_signature(exp168_reference["one_step"]),
            )
            matching["frozen_vector_protocol_rows"] = frozen_alignment["ordered_protocol_rows_equal"]
            matching["frozen_vector_row_mse"] = frozen_row_match["matches"]
            matching["frozen_vector_metric_signature"] = frozen_metric_match

            model = _installed_event_model(baseline, vector_probe, event_probe)
            model.event_trace = []
            journal.update("one_step_event_mode", 0, 120)
            diagnostic, base_rows = pose_probe._diagnose(
                model, journal, args.out / "event_mode_base_rows.jsonl"
            )
            journal.update("one_step_event_mode", 120, 120)
            event_rows = _enrich_rows(
                base_rows, _evaluation_trace(model.event_trace),
                args.out / "event_mode_one_step_rows.jsonl",
            )
            alignment = relational._assert_protocol_alignment(
                diagnostic, exp162_reference["one_step"], event_rows, exp162_rows
            )
            matching["canonical_evaluator_rows"] = alignment["ordered_protocol_rows_equal"]
            exact_protocol = all(matching.values())
            source_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["source"], diagnostic["splits"]["source"],
                exact_protocol,
            )
            unseen_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["unseen"], diagnostic["splits"]["unseen"],
                exact_protocol,
            )
            gate = bool(source_pass and unseen_pass)
            comparison = pose_probe._comparison(
                diagnostic, exp168_reference["one_step"]
            )
            if gate:
                outcome = "event_mode_vector_transition_passes"
                conclusion = (
                    "A generic persistence/change mode composed with frozen direct vectors "
                    "passes both local splits; next remove privileged pose and test transfer."
                )
            else:
                outcome = "event_mode_vector_transition_fails"
                conclusion = (
                    "The event-mode composition fails the exact gate; use heldout event "
                    "classification versus changed-row physics to distinguish classifier "
                    "failure from frozen-vector conditional mismatch."
                )
            canonical_stats = _canonical_event_stats(event_rows)
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "privileged generic event-mode plus frozen vector diagnostic",
                "interpretation_limit": "No deployable solution, composition transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exp168_checkpoint_git_head": vector_metadata["analysis_git_head"],
                "exact_command": command, "exact_protocol": exact_protocol,
                "protocol_match": matching, "protocol_alignment": alignment,
                "frozen_vector_alignment": frozen_alignment,
                "frozen_vector_row_match": frozen_row_match,
                "objective": OBJECTIVE, "episode_split": split_metadata,
                "pose_sidecar": sidecar_metadata, "corpus": corpus,
                "full_action_counts": full_counts,
                "train_action_counts": train_counts,
                "class_weights": class_weights.tolist(),
                "event_metrics": metrics,
                "frozen_vector_one_step": frozen_diagnostic,
                "one_step": diagnostic,
                "canonical_event_stats": canonical_stats,
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "event_mode_vector_gate": gate,
                "exp168_reference": exp168_reference,
                "exp168_comparison": comparison,
                "frozen_backbone_unchanged": backbone_unchanged,
                "frozen_vector_head_unchanged": vector_unchanged,
                "parameter_counts": {
                    "event_trainable": sum(p.numel() for p in event_probe.parameters()),
                    "vector_frozen": sum(p.numel() for p in vector_probe.parameters()),
                    "backbone_frozen": sum(p.numel() for p in baseline.parameters()),
                },
                "outcome": outcome, "conclusion": conclusion,
                "controls": {
                    "only_causal_change_from_exp168": "generic RGB-change event head",
                    "vector_retrained": False, "backbone_frozen": True,
                    "vector_head_frozen": True, "literal_persistence": True,
                    "scalar_amplitude": False, "raw_direction": False,
                    "member_atom": False, "posthoc_threshold": False,
                    "rules_or_outcome_labels": False,
                    "source_or_unseen_leakage": False,
                    "class_weights_train_only": True,
                    "snapshot_before_transition": True,
                    "mpc": False, "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "event_losses.jsonl",
                    "frozen_vector_rows": "frozen_vector_one_step_rows.jsonl",
                    "base_rows": "event_mode_base_rows.jsonl",
                    "rows": "event_mode_one_step_rows.jsonl",
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
                frozen_backbone_unchanged=backbone_unchanged,
                frozen_vector_head_unchanged=vector_unchanged,
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
