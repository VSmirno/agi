"""Privileged relational-position plus agent-orientation amplitude diagnostic."""

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
from experiments import exp160_amplitude_supervised_gate as supervised
from experiments import exp161_amplitude_input_probe as linear_probe
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments import exp164_relational_slot_probe as relational
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
DEFAULT_EXP164_REFERENCE = Path(
    "output_to_user/core/exp164-relational-slot-probe-001/results.json"
)
EXPECTED_EXP164_HEAD = "7a6a6cab987c286e5980262230078681227ac3c6"
PROTOCOL = dict(residual.PROTOCOL)
RELATION_DIM = relational.RELATION_DIM
ORIENTATION_DIM = 4
POSE_DIM = RELATION_DIM + ORIENTATION_DIM
HIDDEN_WIDTH = nonlinear.HIDDEN_WIDTH
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "input": (
        "frozen z + carried hidden + four normalized relational positions + "
        "privileged pre-transition agent_dir one-hot"
    ),
    "architecture": (
        "separate per-action Linear(z_dim+h_dim+8,128)->ReLU->"
        "Linear(128,ensemble_size)->sigmoid"
    ),
    "weight": "fixed full-corpus weight[action, observed_rgb_change]",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_diagnostic": True,
}


def pose_vector(snapshot: Mapping, agent_dir: int) -> torch.Tensor:
    """Append a validated direction one-hot to exp164's relational positions."""

    if isinstance(agent_dir, bool) or not isinstance(agent_dir, (int, np.integer)):
        raise ValueError("agent_dir must be an integer in [0,3]")
    direction = int(agent_dir)
    if direction not in range(ORIENTATION_DIM):
        raise ValueError("agent_dir must be an integer in [0,3]")
    one_hot = torch.zeros(ORIENTATION_DIM, dtype=torch.float32)
    one_hot[direction] = 1.0
    return torch.cat((relational.relational_vector(snapshot), one_hot))


def aligned_episode_pose(episode, adapter, seed: int):
    """Build UID/step pose rows from the world immediately before each action."""

    observation = adapter.reset(seed)
    sidecar = {}
    for step, transition in enumerate(episode.transitions):
        if not np.array_equal(observation.rgb, transition.before.rgb):
            raise AssertionError(f"{episode.uid}/{step} before observation mismatch")
        key = (episode.uid, step)
        if key in sidecar:
            raise AssertionError(f"duplicate pose sidecar key: {key}")
        sidecar[key] = pose_vector(
            adapter.diagnostic_snapshot(), adapter.world.agent_dir
        )
        replayed = adapter.step(transition.action)
        if replayed.action != transition.action:
            raise AssertionError(f"{episode.uid}/{step} action mismatch")
        if not np.array_equal(replayed.before.rgb, transition.before.rgb):
            raise AssertionError(f"{episode.uid}/{step} replay before mismatch")
        if not np.array_equal(replayed.after.rgb, transition.after.rgb):
            raise AssertionError(f"{episode.uid}/{step} replay after mismatch")
        observation = replayed.after
    return sidecar


class RelationalPoseProbe(torch.nn.Module):
    """Exp164 MLP with only the current agent orientation appended."""

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
                torch.nn.Linear(HIDDEN_WIDTH, heads),
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
        if actions.shape != (len(z),) or actions.dtype is not torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden, pose), dim=-1)
        logits = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = logits.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads)
        ).squeeze(1)
        return selected.sigmoid().transpose(0, 1)


class RelationalPoseWorldModel(relational.RelationalProbeWorldModel):
    """Experiment-only frozen dynamics with evaluator-supplied pose context."""

    def set_relations(self, pose: torch.Tensor) -> None:
        if pose.ndim != 2 or pose.shape[1] != POSE_DIM:
            raise ValueError("current pose must have shape [batch,8]")
        self._current_relations = pose.detach()


def _installed_model(baseline, probe):
    parameter = next(baseline.parameters())
    model = RelationalPoseWorldModel(
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


def _build_pose_sidecar(episodes_by_layout, args, journal):
    total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    sidecar = {}
    journal.update("pose_sidecar", 0, total)
    for layout_index, (layout_name, (layout, _actions)) in enumerate(
        temporal.SOURCE_LAYOUTS.items()
    ):
        for offset, episode in enumerate(episodes_by_layout[layout_name]):
            seed = 10000 + layout_index * 100000 + offset
            if not episode.uid.endswith(f":{seed}"):
                raise AssertionError(f"unexpected episode UID/seed: {episode.uid}")
            adapter = temporal._adapter(layout, 1, seed, args.collection_steps)
            try:
                rows = aligned_episode_pose(episode, adapter, seed)
            finally:
                adapter.close()
            overlap = sidecar.keys() & rows.keys()
            if overlap:
                raise AssertionError(f"duplicate sidecar keys: {sorted(overlap)[:3]}")
            sidecar.update(rows)
            completed += 1
            journal.update(
                "pose_sidecar", completed, total,
                layout=layout_name, episode=offset,
            )
    return sidecar


def _fit_probe(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = RelationalPoseProbe(
            config.z_dim, config.h_dim, config.ensemble_size
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_pose", 0, args.probe_updates)
    probe.train()
    for update in range(1, args.probe_updates + 1):
        indices = torch.randint(
            len(train["actions"]),
            (args.probe_batch_size,),
            generator=generator,
        )
        actions = train["actions"][indices].to(device)
        changed = train["changed"][indices].to(device)
        prediction = probe(
            train["z"][indices].to(device),
            train["hidden"][indices].to(device),
            train["relations"][indices].to(device),
            actions,
        )
        target = train["target"][indices].to(device).transpose(0, 1)
        valid = torch.ones(len(indices), device=device, dtype=torch.bool)
        loss = supervised.weighted_amplitude_mse(
            prediction, target, valid, actions, changed, class_weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        trace.write({"update": update, "loss": value})
        journal.update("fit_pose", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": relational._probe_metrics(probe, train, class_weights, device),
        "heldout": relational._probe_metrics(
            probe, heldout, class_weights, device
        ),
    }


def _load_exp164_reference(path: Path):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp164 reference: {error}") from error
    if payload.get("analysis_git_head") != EXPECTED_EXP164_HEAD:
        raise ValueError("exp164 reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError("exp164 reference is not exact protocol")
    if not isinstance(payload.get("one_step"), Mapping):
        raise ValueError("exp164 reference lacks one-step results")
    if not isinstance(payload.get("probe_metrics"), Mapping):
        raise ValueError("exp164 reference lacks probe metrics")
    return {
        "path": str(path),
        "analysis_git_head": payload["analysis_git_head"],
        "probe_metrics": payload["probe_metrics"],
        "one_step": payload["one_step"],
        "gate": payload["relational_probe_gate"],
    }


def _fresh_pose_fork(layout, history, action, seed):
    adapter = temporal._adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        for previous in history:
            transition = adapter.step(previous)
            if transition.terminated or transition.truncated:
                raise RuntimeError("real history unexpectedly ended before the fork")
            observation = transition.after
        before = observation
        pose = pose_vector(
            adapter.diagnostic_snapshot(), adapter.world.agent_dir
        )
        transition = adapter.step(action)
        return before, transition.after, adapter.diagnostic_snapshot(), pose
    finally:
        adapter.close()


def _replay_pose_prefix(model, layout, prefix, seed):
    adapter = temporal._adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        state = model.initial(observation)
        for action in prefix:
            pose = pose_vector(
                adapter.diagnostic_snapshot(), adapter.world.agent_dir
            )
            model.set_relations(pose[None].to(state.z.device))
            transition = adapter.step(action)
            if transition.terminated or transition.truncated:
                raise RuntimeError("canonical prefix unexpectedly ended the episode")
            prediction = model.step(
                state,
                torch.tensor([action], device=state.z.device, dtype=torch.long),
            )
            actual = model.initial(transition.after)
            state = one_step._teacher_forced_next(prediction, actual)
        return state, adapter.diagnostic_snapshot()
    finally:
        adapter.close()


@torch.inference_mode()
def _diagnose(model, journal, rows_path):
    specs = one_step._layout_specs()
    total = sum(len(layouts) for layouts in specs.values()) * 3 * 5
    completed = 0
    summaries = {split: [] for split in one_step.SPLITS}
    written_rows = []
    writer = core.TraceWriter(rows_path)
    try:
        for split in one_step.SPLITS:
            for layout_name, spec in specs[split].items():
                layout, actions = spec[:2]
                prefix, continuation = one_step._validate_protocol(
                    split, layout_name, layout, actions, one_step.SEED
                )
                state, prefix_diagnostic = _replay_pose_prefix(
                    model, layout, prefix, one_step.SEED
                )
                layout_rows = []
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, diagnostic, pose = _fresh_pose_fork(
                            layout, history, action, one_step.SEED
                        )
                        model.set_relations(pose[None].to(state.z.device))
                        action_tensor = torch.tensor(
                            [action], device=state.z.device, dtype=torch.long
                        )
                        prediction = model.step(state, action_tensor)
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
                            "action_name": one_step.GRID_ACTIONS[action],
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
                            "privileged_pose": pose.tolist(),
                        }
                        writer.write(row)
                        written_rows.append(row)
                        layout_rows.append(row)
                        completed += 1
                        journal.update(
                            "teacher_forced_pose", completed, total,
                            split=split, layout=layout_name, step=step, action=action,
                        )
                        if action == canonical_action:
                            canonical_prediction, canonical_actual = prediction, actual
                    state = one_step._teacher_forced_next(
                        canonical_prediction, canonical_actual
                    )
                summary = one_step._layout_summary(layout_rows, layout_name, split)
                summary.update(
                    prefix=list(prefix),
                    continuation=list(continuation),
                    prefix_diagnostic=prefix_diagnostic,
                )
                summaries[split].append(summary)
    finally:
        writer.close()
    source = one_step._aggregate_split(summaries["source"])
    unseen = one_step._aggregate_split(summaries["unseen"])
    return {
        "status": "completed",
        "claim": "privileged relational-pose one-step diagnostic",
        "protocol": {
            "push_distance": 1,
            "seed": one_step.SEED,
            "source_prefix_length": one_step.SOURCE_PREFIX_LENGTH,
            "target_prefix_length": one_step.TARGET_PREFIX_LENGTH,
            "continuation": list(one_step.PUSH_ONE_CONTINUATION),
            "teacher_forcing": "real z/sensors/mask with carried predicted hidden",
            "fresh_environment_replay_per_action_fork": True,
            "push_2_run": False,
            "rows": total,
        },
        "layouts": summaries,
        "splits": {"source": source, "unseen": unseen},
        "outcome_label": one_step._outcome_label(source, unseen),
        "interpretation_limit": "Privileged evaluator evidence only; not deployable.",
        "artifacts": {"rows": rows_path.name},
    }, written_rows


def _comparison(candidate, reference):
    result = {}
    categorical_unchanged = True
    for split in one_step.SPLITS:
        current = candidate["splits"][split]
        prior = reference["splits"][split]
        contact_delta = (
            current["contact_failure_layouts"] - prior["contact_failure_layouts"]
        )
        blocked_delta = (
            current["blocked_noop_failure_layouts"]
            - prior["blocked_noop_failure_layouts"]
        )
        categorical_unchanged &= contact_delta == 0 and blocked_delta == 0
        result[split] = {
            "contact_failure_layout_delta": contact_delta,
            "blocked_failure_layout_delta": blocked_delta,
            "free_ratio_delta": (
                current["medians"]["free_forward_prediction_persistence_ratio"]
                - prior["medians"]["free_forward_prediction_persistence_ratio"]
            ),
            "interact_ratio_delta": (
                current["medians"]["interact_prediction_persistence_ratio"]
                - prior["medians"]["interact_prediction_persistence_ratio"]
            ),
        }
    return {"categorical_unchanged": categorical_unchanged, "by_split": result}


def build_parser():
    parser = relational.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--exp164-reference", type=Path, default=DEFAULT_EXP164_REFERENCE
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
    command = os.environ.get("EXP165_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp162_reference": str(args.exp162_reference),
        "exp164_reference": str(args.exp164_reference),
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
            journal.update("initialize", 1, 4, operation="load_exp162_reference")
            exp162_reference, reference_rows = relational._load_exp162_reference(
                args.exp162_reference
            )
            journal.update("initialize", 2, 4, operation="load_exp164_reference")
            exp164_reference = _load_exp164_reference(args.exp164_reference)
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
                episode.uid
                for values in heldout_episodes.values()
                for episode in values
            }
            if train_ids & heldout_ids or len(train_ids | heldout_ids) != corpus["episodes"]:
                raise AssertionError("75/25 episode split is not complete and disjoint")
            sidecar = _build_pose_sidecar(episodes, args, journal)
            if len(sidecar) != corpus["transitions"]:
                raise AssertionError("pose sidecar does not cover every transition")
            coverage = _audit_counts({"all": replay._episodes()})
            counts = {
                action: {
                    key: row[key]
                    for key in ("total", "rgb_changed", "rgb_no_change")
                }
                for action, row in coverage["actions"].items()
            }
            class_weights = auxiliary.action_class_weights(counts).to(config.device)
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "canonical_exp162_reference": args.exp162_reference
                == DEFAULT_EXP162_REFERENCE,
                "exp162_reference_head": exp162_reference["analysis_git_head"]
                == EXPECTED_EXP162_HEAD,
                "canonical_exp164_reference": args.exp164_reference
                == DEFAULT_EXP164_REFERENCE,
                "exp164_reference_head": exp164_reference["analysis_git_head"]
                == EXPECTED_EXP164_HEAD,
                "default_budgets": all(
                    getattr(args, key) == value for key, value in PROTOCOL.items()
                ),
                "baseline_config": metadata["config"] == FIXED_CONFIG,
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_and_counts": corpus["default_corpus_verified"]
                and counts == FIXED_CORPUS["action_counts"],
                "episode_split_75_25": all(
                    len(train_episodes[name]) == 384
                    and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                ),
                "episode_disjoint": not bool(train_ids & heldout_ids),
                "sidecar_exact_coverage": len(sidecar) == corpus["transitions"],
                "probe_budget": args.probe_updates == 400
                and args.probe_batch_size == 256,
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
                    "box_x-agent_x", "box_y-agent_y",
                    "goal_x-box_x", "goal_y-box_y",
                    "agent_dir_0", "agent_dir_1", "agent_dir_2", "agent_dir_3",
                ],
                "relation_normalization_grid_span": relational.GRID_SPAN,
                "agent_dir_source": "adapter.world.agent_dir before action",
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                exp162_reference_metadata=exp162_reference,
                exp164_reference_metadata=exp164_reference,
                protocol_match=matching,
                episode_split=split_metadata,
                pose_sidecar=sidecar_metadata,
                action_counts=counts,
                class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)

            train = relational._extract_dataset(
                baseline, train_episodes, sidecar, journal, "extract_train"
            )
            heldout = relational._extract_dataset(
                baseline, heldout_episodes, sidecar, journal, "extract_heldout"
            )
            datasets = {
                "train": linear_probe._target_summary(train),
                "heldout": linear_probe._target_summary(heldout),
            }
            loss_trace = core.TraceWriter(args.out / "probe_losses.jsonl")
            try:
                probe, metrics = _fit_probe(
                    train, heldout, class_weights, config, args, journal, loss_trace
                )
            finally:
                loss_trace.close()
            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            if not backbone_unchanged:
                raise AssertionError("frozen exp153 backbone changed")

            journal.update("probe_checkpoint", 0, 1)
            checkpoint_path = args.out / "relational_pose_probe.pt"
            torch.save(
                {
                    "format_version": 1,
                    "analysis_git_head": manifest["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "objective": OBJECTIVE,
                    "z_dim": config.z_dim,
                    "h_dim": config.h_dim,
                    "pose_dim": POSE_DIM,
                    "ensemble_size": config.ensemble_size,
                    "hidden_width": HIDDEN_WIDTH,
                    "probe_state_dict": probe.state_dict(),
                },
                checkpoint_path,
            )
            journal.update("probe_checkpoint", 1, 1)
            model = _installed_model(baseline, probe)
            journal.update("one_step_pose", 0, 120)
            diagnostic, candidate_rows = _diagnose(
                model, journal, args.out / "relational_pose_one_step_rows.jsonl"
            )
            journal.update("one_step_pose", 120, 120)
            alignment = relational._assert_protocol_alignment(
                diagnostic,
                exp162_reference["one_step"],
                candidate_rows,
                reference_rows,
            )
            matching["canonical_evaluator_rows"] = alignment[
                "ordered_protocol_rows_equal"
            ]
            exact_protocol = all(matching.values())
            source_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["source"],
                diagnostic["splits"]["source"],
                exact_protocol,
            )
            unseen_pass = nonlinear.nonlinear_probe_gate(
                diagnostic["splits"]["unseen"],
                diagnostic["splits"]["unseen"],
                exact_protocol,
            )
            gate = bool(source_pass and unseen_pass)
            comparison = _comparison(diagnostic, exp164_reference["one_step"])
            if gate:
                outcome = "relational_pose_passes"
                conclusion = (
                    "Privileged full object pose repairs both local splits; z+hidden "
                    "lacked accessible transferable pose geometry."
                )
            elif source_pass:
                outcome = "source_only_relational_pose"
                conclusion = (
                    "Privileged pose repairs source only; generalization remains deficient."
                )
            elif comparison["categorical_unchanged"]:
                outcome = "pose_categorical_failures_unchanged"
                conclusion = (
                    "Full position plus orientation leaves categorical failures unchanged; "
                    "a richer transition target is licensed."
                )
            else:
                outcome = "pose_improvement_only"
                conclusion = (
                    "Pose changes local metrics without passing the gate; report the exact "
                    "deltas before choosing a richer transition target."
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "privileged evaluator-only relational pose diagnostic",
                "interpretation_limit": (
                    "Not a deployable solution and no composition, transfer, or AGI evidence."
                ),
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
                "action_counts": counts,
                "class_weights": class_weights.tolist(),
                "target_datasets": datasets,
                "probe_metrics": metrics,
                "one_step": diagnostic,
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "relational_pose_gate": gate,
                "exp162_reference": exp162_reference,
                "exp164_reference": exp164_reference,
                "exp164_comparison": comparison,
                "frozen_backbone_unchanged": backbone_unchanged,
                "outcome": outcome,
                "conclusion": conclusion,
                "controls": {
                    "privileged_diagnostic": True,
                    "snapshot_and_agent_dir_before_transition": True,
                    "sidecar_keyed_by_episode_uid_and_step": True,
                    "only_causal_change_from_exp164": "agent_dir one-hot",
                    "action_rules_encoded": False,
                    "outcome_or_blocked_labels_encoded": False,
                    "object_condition_branches": False,
                    "raw_deltas_before_native_gate": True,
                    "analytic_targets_detached": True,
                    "backbone_frozen": True,
                    "exp164_retrained": False,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "rows": "relational_pose_one_step_rows.jsonl",
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
                protocol_match=matching,
                runtime_seconds=time.monotonic() - started,
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
