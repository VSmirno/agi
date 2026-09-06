"""Train observation-only vector/event heads and evaluate autonomous behavior."""

from __future__ import annotations

from dataclasses import asdict, replace
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
from experiments import exp168_relational_pose_vector_delta as vector_delta
from experiments import exp169_event_mode_vector_transition as event_mode
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
PROTOCOL = dict(residual.PROTOCOL)
HIDDEN_WIDTH = 128
EVENT_BOUNDARY = 0.5
OBJECTIVE = {
    "target": {
        "vector": "detached actual_next_z - current_z, broadcast to members",
        "event": "observed RGB changed versus byte-identical transition",
    },
    "input": "frozen current z + carried hidden only",
    "architecture": (
        "separate per-action z+hidden MLPs with width 128 for vector and event"
    ),
    "loss": {
        "vector": "fixed train-only action/change weighted MSE",
        "event": "fixed train-only per-action class-balanced BCE",
    },
    "installed_transition": (
        "event probability >=0.5 uses learned member delta; otherwise literal z persistence"
    ),
    "updates_each": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
    "privileged_pose": False,
}


class ObservationVectorDelta(torch.nn.Module):
    """Predict direct member latent displacements from observation state only."""

    def __init__(self, z_dim: int, h_dim: int, heads: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(z_dim + h_dim, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, heads * z_dim),
            )
            for _ in range(5)
        )

    def forward(self, z, hidden, actions):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if actions.shape != (len(z),) or actions.dtype != torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden), dim=-1)
        all_actions = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = all_actions.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads * self.z_dim)
        ).squeeze(1)
        return selected.reshape(len(z), self.heads, self.z_dim).permute(1, 0, 2)


class ObservationEventHead(torch.nn.Module):
    """Predict generic RGB change from observation state only."""

    def __init__(self, z_dim: int, h_dim: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(z_dim + h_dim, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, 1),
            )
            for _ in range(5)
        )

    def logits(self, z, hidden, actions):
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if actions.shape != (len(z),) or actions.dtype != torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden), dim=-1)
        all_actions = torch.stack([head(features) for head in self.by_action], dim=1)
        return all_actions.gather(1, actions[:, None, None]).flatten()

    def forward(self, z, hidden, actions):
        return self.logits(z, hidden, actions).sigmoid()


class ObservationOnlyWorldModel(CoreWorldModel):
    """Frozen native plumbing with pose-free learned event/vector transitions."""

    def __init__(self, *args, vector_probe, event_probe, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_probe = vector_probe
        self.event_probe = event_probe

    def step(self, state: LatentState, actions: torch.Tensor):
        native = CoreWorldModel.step(self, state, actions)
        member_delta = self.vector_probe(state.z, state.hidden, actions)
        probability = self.event_probe(state.z, state.hidden, actions)
        return event_mode.apply_event_mode(
            native, state.z, member_delta, probability
        )


def _installed_model(baseline, vector_probe, event_probe):
    parameter = next(baseline.parameters())
    model = ObservationOnlyWorldModel(
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
        name: value
        for name, value in baseline.state_dict().items()
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
def _extract_dataset(model, episodes_by_layout, journal, stage):
    names = ("z", "hidden", "target", "actions", "changed")
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
                prediction = model.step(state, action)
                rows["z"].append(state.z[0])
                rows["hidden"].append(state.hidden[0])
                rows["target"].append(
                    (z_sequence[index + 1] - state.z[0]).detach()
                )
                rows["actions"].append(action[0])
                rows["changed"].append(torch.tensor(
                    not np.array_equal(transition.before.rgb, transition.after.rgb),
                    device=parameter.device,
                    dtype=torch.bool,
                ))
                actual = LatentState(
                    z_sequence[index + 1:index + 2],
                    torch.where(
                        masks[index + 1:index + 2],
                        sensors[index + 1:index + 2],
                        0.0,
                    ),
                    masks[index + 1:index + 2],
                    z_sequence.new_zeros(1, model.h_dim),
                    state.schema,
                )
                state = one_step._teacher_forced_next(prediction, actual)
            for name, values in rows.items():
                chunks[name].append(torch.stack(values).cpu())
            completed += 1
            journal.update(stage, completed, total, layout=layout)
    return {name: torch.cat(values) for name, values in chunks.items()}


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
            actions,
        )
        target = dataset["target"][start:stop].to(device).unsqueeze(0).expand(
            probe.heads, -1, -1
        )
        error = (prediction - target).square()
        squared += float(error.sum())
        weighted += float(
            (error * class_weights[actions, changed.long()][None, :, None]).sum()
        )
        persistence += float(target.square().sum())
        count += error.numel()
    return {
        "mse": squared / count,
        "weighted_mse": weighted / count,
        "persistence_mse": persistence / count,
        "prediction_to_persistence_ratio": (
            squared / persistence if persistence > 0 else None
        ),
        "member_latent_elements": count,
    }


@torch.inference_mode()
def _event_metrics(probe, dataset, class_weights, device):
    logits = []
    for start in range(0, len(dataset["actions"]), 4096):
        stop = min(start + 4096, len(dataset["actions"]))
        logits.append(probe.logits(
            dataset["z"][start:stop].to(device),
            dataset["hidden"][start:stop].to(device),
            dataset["actions"][start:stop].to(device),
        ).cpu())
    logits = torch.cat(logits)
    probabilities = logits.sigmoid()
    predicted = probabilities >= EVENT_BOUNDARY
    changed = dataset["changed"]

    def summarize(mask):
        count = int(mask.sum())
        actual = changed[mask]
        guess = predicted[mask]
        positive = int(actual.sum())
        negative = count - positive
        changed_recall = float((guess & actual).sum()) / positive if positive else None
        nochange_recall = (
            float((~guess & ~actual).sum()) / negative if negative else None
        )
        recalls = [
            value for value in (changed_recall, nochange_recall) if value is not None
        ]
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

    loss = event_mode.balanced_event_bce(
        logits, dataset["actions"], changed, class_weights.cpu()
    )
    return {
        "loss": float(loss),
        "overall": summarize(torch.ones_like(changed)),
        "by_action": {
            f"action{action}": summarize(dataset["actions"] == action)
            for action in range(5)
        },
    }


def _fit_vector(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = ObservationVectorDelta(
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
            actions,
        )
        target = train["target"][indices].to(device).unsqueeze(0).expand(
            config.ensemble_size, -1, -1
        ).detach()
        loss = vector_delta.weighted_vector_mse(
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
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _vector_metrics(probe, train, class_weights, device),
        "heldout": _vector_metrics(probe, heldout, class_weights, device),
    }


def _fit_event(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = ObservationEventHead(config.z_dim, config.h_dim).to(device)
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
            actions,
        )
        loss = event_mode.balanced_event_bce(
            logits, actions, changed, class_weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        trace.write({"update": update, "loss": value})
        journal.update("fit_event", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": _event_metrics(probe, train, class_weights, device),
        "heldout": _event_metrics(probe, heldout, class_weights, device),
    }


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
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
    command = os.environ.get("EXP172_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
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
            journal.update("initialize", 0, 2, operation="safe_exp153_load")
            baseline, ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
            ordered.eval().requires_grad_(False)
            initial_backbone = {
                name: value.detach().clone()
                for name, value in baseline.state_dict().items()
            }
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config),
                seed=args.seed,
                z_dim=args.z_dim,
                h_dim=args.h_dim,
                burn_in=0,
                replay_capacity=(
                    len(temporal.SOURCE_LAYOUTS) * args.episodes_per_layout
                ),
                termination_weight=0.0,
                salient_fraction=0.0,
            )
            if next(baseline.parameters()).device.type != torch.device(config.device).type:
                raise ValueError("checkpoint device and requested config disagree")
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 2, 2, device=config.device)
            corpus, _fit, _validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            episodes = linear_probe._ordered_episodes(replay, corpus)
            train_episodes, heldout_episodes = linear_probe.episode_disjoint_split(
                episodes
            )
            train_ids = {
                episode.uid
                for values in train_episodes.values()
                for episode in values
            }
            heldout_ids = {
                episode.uid
                for values in heldout_episodes.values()
                for episode in values
            }
            if train_ids & heldout_ids or len(train_ids | heldout_ids) != corpus["episodes"]:
                raise AssertionError("75/25 episode split is not complete and disjoint")
            full_coverage = _audit_counts({"all": replay._episodes()})
            full_counts = {
                action: {
                    key: row[key]
                    for key in ("total", "rgb_changed", "rgb_no_change")
                }
                for action, row in full_coverage["actions"].items()
            }
            train = _extract_dataset(
                baseline, train_episodes, journal, "extract_train"
            )
            heldout = _extract_dataset(
                baseline, heldout_episodes, journal, "extract_heldout"
            )
            train_counts = vector_delta._action_change_counts(train)
            class_weights = event_mode.event_class_weights(train_counts).to(
                config.device
            )
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
                "default_budgets": all(
                    getattr(args, key) == value for key, value in PROTOCOL.items()
                ),
                "baseline_config": metadata["config"] == FIXED_CONFIG,
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_and_counts": (
                    corpus["default_corpus_verified"]
                    and full_counts == FIXED_CORPUS["action_counts"]
                ),
                "episode_split_75_25": all(
                    len(train_episodes[name]) == 384
                    and len(heldout_episodes[name]) == 128
                    for name in temporal.SOURCE_LAYOUTS
                ),
                "episode_disjoint": not bool(train_ids & heldout_ids),
                "class_weights_train_only": (
                    sum(row["total"] for row in train_counts.values())
                    == len(train["actions"])
                ),
                "head_budgets": (
                    args.probe_updates == 400 and args.probe_batch_size == 256
                ),
                "observation_only": set(train) == {
                    "z", "hidden", "target", "actions", "changed"
                },
            }
            split_metadata = {
                "train_episodes": len(train_ids),
                "heldout_episodes": len(heldout_ids),
                "train_uid_digest": linear_probe._uid_digest(train_episodes),
                "heldout_uid_digest": linear_probe._uid_digest(heldout_episodes),
                "overlap": 0,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                protocol_match=matching,
                episode_split=split_metadata,
                full_action_counts=full_counts,
                train_action_counts=train_counts,
                class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)

            vector_trace = core.TraceWriter(args.out / "vector_losses.jsonl")
            try:
                vector_probe, vector_metrics = _fit_vector(
                    train, heldout, class_weights, config, args, journal, vector_trace
                )
            finally:
                vector_trace.close()
            event_trace = core.TraceWriter(args.out / "event_losses.jsonl")
            try:
                event_probe, event_metrics = _fit_event(
                    train, heldout, class_weights, config, args, journal, event_trace
                )
            finally:
                event_trace.close()

            backbone_unchanged = all(
                torch.equal(baseline.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            if not backbone_unchanged:
                raise AssertionError("frozen exp153 backbone changed")
            journal.update("checkpoint", 0, 1)
            checkpoint_path = args.out / "observation_only_heads.pt"
            torch.save({
                "format_version": 1,
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "objective": OBJECTIVE,
                "z_dim": config.z_dim,
                "h_dim": config.h_dim,
                "ensemble_size": config.ensemble_size,
                "hidden_width": HIDDEN_WIDTH,
                "train_action_counts": train_counts,
                "class_weights": class_weights.cpu(),
                "vector_probe_state_dict": vector_probe.state_dict(),
                "event_probe_state_dict": event_probe.state_dict(),
            }, checkpoint_path)
            journal.update("checkpoint", 1, 1)

            model = _installed_model(baseline, vector_probe, event_probe)
            from experiments.exp172_behavior_eval import evaluate_behavior

            evaluation = evaluate_behavior(
                model, baseline, ordered, config, journal, args.out
            )
            exact_protocol = all(matching.values())
            result = {
                "status": "completed",
                "claim": "observation-only transition development comparison",
                "interpretation_limit": (
                    "One development run; no independent generalization or AGI claim."
                ),
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "objective": OBJECTIVE,
                "episode_split": split_metadata,
                "corpus": corpus,
                "full_action_counts": full_counts,
                "train_action_counts": train_counts,
                "class_weights": class_weights.tolist(),
                "vector_metrics": vector_metrics,
                "event_metrics": event_metrics,
                "evaluation": evaluation,
                "frozen_backbone_unchanged": backbone_unchanged,
                "parameter_counts": {
                    "vector_trainable": sum(
                        parameter.numel() for parameter in vector_probe.parameters()
                    ),
                    "event_trainable": sum(
                        parameter.numel() for parameter in event_probe.parameters()
                    ),
                    "backbone_frozen": sum(
                        parameter.numel() for parameter in baseline.parameters()
                    ),
                },
                "controls": {
                    "backbone_frozen": True,
                    "heads_fresh": True,
                    "pose_features": False,
                    "snapshot_features": False,
                    "sidecar": False,
                    "literal_persistence": True,
                    "class_weights_train_only": True,
                    "source_or_unseen_leakage": False,
                    "posthoc_threshold": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "vector_losses": "vector_losses.jsonl",
                    "event_losses": "event_losses.jsonl",
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
                exact_protocol=exact_protocol,
                protocol_match=matching,
                runtime_seconds=time.monotonic() - started,
                frozen_backbone_unchanged=backbone_unchanged,
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
