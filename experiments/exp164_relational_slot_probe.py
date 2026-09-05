"""Privileged evaluator probe for object-relative geometry in amplitude prediction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import hashlib
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
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp159_independent_amplitude_oracle as amplitude_oracle
from experiments import exp160_amplitude_supervised_gate as supervised
from experiments import exp161_amplitude_input_probe as linear_probe
from experiments import exp162_nonlinear_amplitude_probe as nonlinear
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_EXP162_REFERENCE = Path(
    "output_to_user/core/exp162-nonlinear-amplitude-probe-001/results.json"
)
EXPECTED_EXP162_HEAD = "8edec06cca48b34a8285dec7d943f5ff4332082e"
PROTOCOL = dict(residual.PROTOCOL)
GRID_SPAN = 5.0
RELATION_DIM = 4
HIDDEN_WIDTH = nonlinear.HIDDEN_WIDTH
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "input": (
        "frozen z + carried hidden + privileged normalized "
        "[box-agent, goal-box] geometry"
    ),
    "architecture": (
        "separate per-action Linear(z_dim+h_dim+4,128)->ReLU->"
        "Linear(128,ensemble_size)->sigmoid"
    ),
    "weight": "fixed full-corpus weight[action, observed_rgb_change]",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_diagnostic": True,
}


def relational_vector(snapshot: Mapping) -> torch.Tensor:
    """Return normalized box-agent and goal-box offsets, without outcome labels."""

    try:
        agent_x, agent_y = snapshot["agent_pos"]
        box_x, box_y = snapshot["box_pos"]
        goal_x, goal_y = snapshot["goal_pos"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("snapshot requires 2D agent, box, and goal positions") from error
    values = torch.tensor(
        [
            box_x - agent_x,
            box_y - agent_y,
            goal_x - box_x,
            goal_y - box_y,
        ],
        dtype=torch.float32,
    ) / GRID_SPAN
    if values.shape != (RELATION_DIM,) or not torch.isfinite(values).all():
        raise ValueError("relational features must be four finite values")
    return values


def aligned_episode_relations(episode, adapter, seed: int):
    """Build one UID/step sidecar while proving every snapshot is pre-action."""

    observation = adapter.reset(seed)
    sidecar = {}
    for step, transition in enumerate(episode.transitions):
        if not np.array_equal(observation.rgb, transition.before.rgb):
            raise AssertionError(f"{episode.uid}/{step} before observation mismatch")
        key = (episode.uid, step)
        if key in sidecar:
            raise AssertionError(f"duplicate relational sidecar key: {key}")
        sidecar[key] = relational_vector(adapter.diagnostic_snapshot())
        replayed = adapter.step(transition.action)
        if replayed.action != transition.action:
            raise AssertionError(f"{episode.uid}/{step} action mismatch")
        if not np.array_equal(replayed.before.rgb, transition.before.rgb):
            raise AssertionError(f"{episode.uid}/{step} replay before mismatch")
        if not np.array_equal(replayed.after.rgb, transition.after.rgb):
            raise AssertionError(f"{episode.uid}/{step} replay after mismatch")
        observation = replayed.after
    return sidecar


class RelationalSlotProbe(torch.nn.Module):
    """Exp162 head with only four privileged relative-position inputs added."""

    def __init__(self, z_dim: int, h_dim: int, heads: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        width = z_dim + h_dim + RELATION_DIM
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, heads),
            )
            for _ in range(5)
        )

    def forward(self, z, hidden, relations, actions):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if relations.shape != (len(z), RELATION_DIM):
            raise ValueError("relations must have shape [batch,4]")
        if actions.shape != (len(z),) or actions.dtype is not torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden, relations), dim=-1)
        logits = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = logits.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads)
        ).squeeze(1)
        return selected.sigmoid().transpose(0, 1)


class RelationalProbeWorldModel(nonlinear.gated.ChangeGatedResidualWorldModel):
    """Experiment-only model whose evaluator supplies the current geometry sidecar."""

    def __init__(self, *args, amplitude_probe: RelationalSlotProbe, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gate_heads
        self.amplitude_probe = amplitude_probe
        self._current_relations = None

    def set_relations(self, relations: torch.Tensor) -> None:
        if relations.ndim != 2 or relations.shape[1] != RELATION_DIM:
            raise ValueError("current relations must have shape [batch,4]")
        self._current_relations = relations.detach()

    def change_gates(self, state: LatentState, actions: torch.Tensor):
        relations = self._current_relations
        if relations is None or relations.shape[0] != state.z.shape[0]:
            raise RuntimeError("relational context was not installed for this state")
        relations = relations.to(device=state.z.device, dtype=state.z.dtype)
        return self.amplitude_probe(
            state.z, state.hidden, relations, actions
        ).unsqueeze(-1)


def _installed_model(baseline, probe):
    parameter = next(baseline.parameters())
    model = RelationalProbeWorldModel(
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


def _build_relational_sidecar(episodes_by_layout, args, journal):
    total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    sidecar = {}
    journal.update("relational_sidecar", 0, total)
    for layout_index, (layout_name, (layout, _actions)) in enumerate(
        temporal.SOURCE_LAYOUTS.items()
    ):
        for offset, episode in enumerate(episodes_by_layout[layout_name]):
            seed = 10000 + layout_index * 100000 + offset
            if not episode.uid.endswith(f":{seed}"):
                raise AssertionError(f"unexpected episode UID/seed: {episode.uid}")
            adapter = temporal._adapter(layout, 1, seed, args.collection_steps)
            try:
                rows = aligned_episode_relations(episode, adapter, seed)
            finally:
                adapter.close()
            overlap = sidecar.keys() & rows.keys()
            if overlap:
                raise AssertionError(f"duplicate sidecar keys: {sorted(overlap)[:3]}")
            sidecar.update(rows)
            completed += 1
            journal.update(
                "relational_sidecar", completed, total,
                layout=layout_name, episode=offset,
            )
    return sidecar


def _sidecar_digest(sidecar) -> str:
    digest = hashlib.sha256()
    for (uid, step), value in sorted(sidecar.items()):
        digest.update(uid.encode())
        digest.update(int(step).to_bytes(4, "little"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def _extract_dataset(model, episodes_by_layout, sidecar, journal, stage):
    names = ("z", "hidden", "relations", "target", "actions", "changed")
    chunks = {name: [] for name in names}
    episode_total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    journal.update(stage, 0, episode_total)
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
                relation = sidecar.get((episode.uid, index))
                if relation is None:
                    raise AssertionError(f"missing sidecar row: {episode.uid}/{index}")
                rows["z"].append(state.z[0])
                rows["hidden"].append(state.hidden[0])
                rows["relations"].append(relation.to(parameter.device))
                rows["target"].append(target)
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
            journal.update(stage, completed, episode_total, layout=layout)
    return {name: torch.cat(values) for name, values in chunks.items()}


def _probe_metrics(probe, dataset, class_weights, device):
    squared = weighted = 0.0
    count = 0
    groups = {}
    with torch.inference_mode():
        for start in range(0, len(dataset["actions"]), 4096):
            stop = min(start + 4096, len(dataset["actions"]))
            actions = dataset["actions"][start:stop].to(device)
            changed = dataset["changed"][start:stop].to(device)
            prediction = probe(
                dataset["z"][start:stop].to(device),
                dataset["hidden"][start:stop].to(device),
                dataset["relations"][start:stop].to(device),
                actions,
            )
            target = dataset["target"][start:stop].to(device).transpose(0, 1)
            errors = (prediction - target).square()
            squared += float(errors.sum())
            weighted += float(
                (errors * class_weights[actions, changed.long()].unsqueeze(0)).sum()
            )
            count += errors.numel()
    for action in (2, 3):
        for event in (False, True):
            mask = (dataset["actions"] == action) & (dataset["changed"] == event)
            key = f"action{action}_{'changed' if event else 'nochange'}"
            if not int(mask.sum()):
                groups[key] = {"transitions": 0, "mse": None}
                continue
            with torch.inference_mode():
                prediction = probe(
                    dataset["z"][mask].to(device),
                    dataset["hidden"][mask].to(device),
                    dataset["relations"][mask].to(device),
                    dataset["actions"][mask].to(device),
                )
                target = dataset["target"][mask].to(device).transpose(0, 1)
            groups[key] = {
                "transitions": int(mask.sum()),
                "mse": float((prediction - target).square().mean()),
            }
    return {
        "mse": squared / count,
        "weighted_mse": weighted / count,
        "member_elements": count,
        "groups": groups,
    }


def _fit_probe(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = RelationalSlotProbe(
            config.z_dim, config.h_dim, config.ensemble_size
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_relational", 0, args.probe_updates)
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
        journal.update("fit_relational", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _probe_metrics(probe, train, class_weights, device),
        "heldout": _probe_metrics(probe, heldout, class_weights, device),
    }


def _load_exp162_reference(path: Path):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp162 reference: {error}") from error
    if payload.get("analysis_git_head") != EXPECTED_EXP162_HEAD:
        raise ValueError("exp162 reference analysis head mismatch")
    if payload.get("exact_protocol") is not True:
        raise ValueError("exp162 reference is not exact protocol")
    diagnostic = payload.get("one_step")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("exp162 reference lacks one-step results")
    rows_name = payload.get("artifacts", {}).get("rows")
    rows_path = path.parent / str(rows_name)
    try:
        rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp162 reference rows: {error}") from error
    if len(rows) != 120:
        raise ValueError("exp162 reference requires exactly 120 rows")
    return {
        "path": str(path),
        "analysis_git_head": payload["analysis_git_head"],
        "probe_metrics": payload["probe_metrics"],
        "one_step": diagnostic,
        "gate": payload["nonlinear_probe_gate"],
        "rows_artifact": rows_path.name,
    }, rows


def _fresh_relational_fork(layout, history, action, seed):
    adapter = temporal._adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        for previous in history:
            transition = adapter.step(previous)
            if transition.terminated or transition.truncated:
                raise RuntimeError("real history unexpectedly ended before the fork")
            observation = transition.after
        before = observation
        relations = relational_vector(adapter.diagnostic_snapshot())
        transition = adapter.step(action)
        return before, transition.after, adapter.diagnostic_snapshot(), relations
    finally:
        adapter.close()


def _replay_relational_prefix(model, layout, prefix, seed):
    adapter = temporal._adapter(layout, 1, seed, 32)
    try:
        observation = adapter.reset(seed)
        state = model.initial(observation)
        for action in prefix:
            relations = relational_vector(adapter.diagnostic_snapshot())
            model.set_relations(relations[None].to(state.z.device))
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
                state, prefix_diagnostic = _replay_relational_prefix(
                    model, layout, prefix, one_step.SEED
                )
                layout_rows = []
                for step, canonical_action in enumerate(continuation):
                    history = (*prefix, *continuation[:step])
                    canonical_prediction = canonical_actual = None
                    for action in range(5):
                        before, after, diagnostic, relations = _fresh_relational_fork(
                            layout, history, action, one_step.SEED
                        )
                        model.set_relations(relations[None].to(state.z.device))
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
                            "privileged_relations": relations.tolist(),
                        }
                        writer.write(row)
                        written_rows.append(row)
                        layout_rows.append(row)
                        completed += 1
                        journal.update(
                            "teacher_forced_relational", completed, total,
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
    result = {
        "status": "completed",
        "claim": "privileged relational checkpoint one-step diagnostic",
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
    }
    return result, written_rows


def _assert_protocol_alignment(candidate, reference, candidate_rows, reference_rows):
    if candidate["protocol"] != reference["protocol"]:
        raise AssertionError("candidate and exp162 one-step protocols differ")
    fields = (
        "split", "layout", "step", "real_history", "canonical_action",
        "action", "action_name", "rgb_changed",
    )
    candidate_keys = [tuple(row[name] if name != "real_history" else tuple(row[name]) for name in fields) for row in candidate_rows]
    reference_keys = [tuple(row[name] if name != "real_history" else tuple(row[name]) for name in fields) for row in reference_rows]
    if candidate_keys != reference_keys:
        raise AssertionError("candidate rows do not match exp162 canonical audit rows")
    return {"rows": len(candidate_rows), "ordered_protocol_rows_equal": True}


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--exp162-reference", type=Path, default=DEFAULT_EXP162_REFERENCE
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
    command = os.environ.get("EXP164_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "exp162_reference": str(args.exp162_reference),
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
            journal.update("initialize", 0, 3, operation="safe_exp153_load")
            baseline, _ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            initial_backbone = {
                name: value.detach().clone()
                for name, value in baseline.state_dict().items()
            }
            journal.update("initialize", 1, 3, operation="load_exp162_reference")
            reference, reference_rows = _load_exp162_reference(args.exp162_reference)
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
            journal.update("initialize", 3, 3, device=config.device)
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
            sidecar = _build_relational_sidecar(episodes, args, journal)
            if len(sidecar) != corpus["transitions"]:
                raise AssertionError("sidecar does not cover every exact corpus transition")
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
                "exp162_reference_head": reference["analysis_git_head"]
                == EXPECTED_EXP162_HEAD,
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
                "digest": _sidecar_digest(sidecar),
                "features": [
                    "box_x-agent_x", "box_y-agent_y",
                    "goal_x-box_x", "goal_y-box_y",
                ],
                "normalization_grid_span": GRID_SPAN,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                exp162_reference_metadata=reference,
                protocol_match=matching,
                episode_split=split_metadata,
                relational_sidecar=sidecar_metadata,
                action_counts=counts,
                class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)

            train = _extract_dataset(
                baseline, train_episodes, sidecar, journal, "extract_train"
            )
            heldout = _extract_dataset(
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
            checkpoint_path = args.out / "relational_slot_probe.pt"
            torch.save(
                {
                    "format_version": 1,
                    "analysis_git_head": manifest["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "objective": OBJECTIVE,
                    "z_dim": config.z_dim,
                    "h_dim": config.h_dim,
                    "relation_dim": RELATION_DIM,
                    "ensemble_size": config.ensemble_size,
                    "hidden_width": HIDDEN_WIDTH,
                    "probe_state_dict": probe.state_dict(),
                },
                checkpoint_path,
            )
            journal.update("probe_checkpoint", 1, 1)
            model = _installed_model(baseline, probe)
            journal.update("one_step_relational", 0, 120)
            diagnostic, candidate_rows = _diagnose(
                model, journal, args.out / "relational_one_step_rows.jsonl"
            )
            journal.update("one_step_relational", 120, 120)
            alignment = _assert_protocol_alignment(
                diagnostic, reference["one_step"], candidate_rows, reference_rows
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
            if gate:
                outcome = "relational_geometry_passes"
                conclusion = (
                    "Privileged relative geometry makes amplitudes locally transferable; "
                    "z+hidden lacked accessible object-relative state."
                )
            elif source_pass:
                outcome = "source_only_relational_geometry"
                conclusion = (
                    "Privileged geometry repairs source only; generalization remains deficient."
                )
            else:
                outcome = "relational_geometry_insufficient"
                conclusion = (
                    "Four relative slots are insufficient; next diagnose a richer object "
                    "transition target rather than fit this probe longer."
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "privileged evaluator-only relational state diagnostic",
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
                "relational_sidecar": sidecar_metadata,
                "corpus": corpus,
                "action_counts": counts,
                "class_weights": class_weights.tolist(),
                "target_datasets": datasets,
                "probe_metrics": metrics,
                "one_step": diagnostic,
                "source_split_gate": source_pass,
                "unseen_split_gate": unseen_pass,
                "relational_probe_gate": gate,
                "exp162_reference": reference,
                "frozen_backbone_unchanged": backbone_unchanged,
                "outcome": outcome,
                "conclusion": conclusion,
                "controls": {
                    "privileged_diagnostic": True,
                    "snapshot_before_transition": True,
                    "sidecar_keyed_by_episode_uid_and_step": True,
                    "only_relational_geometry_added_to_exp162_head": True,
                    "action_rules_encoded": False,
                    "outcome_or_blocked_labels_encoded": False,
                    "object_condition_branches": False,
                    "raw_deltas_before_native_gate": True,
                    "analytic_targets_detached": True,
                    "backbone_frozen": True,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "rows": "relational_one_step_rows.jsonl",
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
