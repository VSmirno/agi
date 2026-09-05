"""Teacher-forced diagnostic of inputs needed to predict exp159 amplitudes."""

from __future__ import annotations

import argparse
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
from experiments import exp153_change_gated_dynamics as gated
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp159_independent_amplitude_oracle as amplitude_oracle
from experiments import exp160_amplitude_supervised_gate as supervised
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
PROTOCOL = dict(residual.PROTOCOL)
ARMS = {"z": False, "z_hidden": True}
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "inputs": {"z": "current actual z", "z_hidden": "current actual z + carried hidden"},
    "architecture": "separate per-action Linear(input, ensemble_size) + sigmoid",
    "weight": "fixed full-corpus weight[action, observed_rgb_change]",
    "denominator": "ordinary sampled valid member count; no batch renormalization",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
}


def episode_disjoint_split(episodes_by_layout: Mapping[str, list]):
    """Return the complete established 75/25 split, rejecting UID leakage."""

    train, heldout = {}, {}
    for layout, episodes in episodes_by_layout.items():
        cutoff = round(0.75 * len(episodes))
        train[layout] = list(episodes[:cutoff])
        heldout[layout] = list(episodes[cutoff:])
        train_ids = {episode.uid for episode in train[layout]}
        heldout_ids = {episode.uid for episode in heldout[layout]}
        if train_ids & heldout_ids:
            raise ValueError(f"episode leakage in {layout}")
        if len(train_ids | heldout_ids) != len(episodes):
            raise ValueError(f"duplicate episode UIDs in {layout}")
    return train, heldout


class AmplitudeInputProbe(torch.nn.Module):
    """Action-specific linear amplitude predictor for one candidate input set."""

    def __init__(self, z_dim: int, h_dim: int, heads: int, use_hidden: bool):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        self.use_hidden = use_hidden
        width = z_dim + (h_dim if use_hidden else 0)
        self.by_action = torch.nn.ModuleList(
            torch.nn.Linear(width, heads) for _ in range(5)
        )

    def forward(
        self, z: torch.Tensor, hidden: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if z.ndim != 2 or z.shape[1] != self.z_dim:
            raise ValueError("z must have shape [batch,z_dim]")
        if hidden.shape != (len(z), self.h_dim):
            raise ValueError("hidden must have shape [batch,h_dim]")
        if actions.shape != (len(z),) or actions.dtype is not torch.long:
            raise ValueError("actions must be long with shape [batch]")
        features = torch.cat((z, hidden), dim=-1) if self.use_hidden else z
        logits = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = logits.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads)
        ).squeeze(1)
        return selected.sigmoid().transpose(0, 1)


class ProbeGatedWorldModel(gated.ChangeGatedResidualWorldModel):
    """Install a fitted diagnostic probe as the frozen model's gate."""

    def __init__(self, *args, amplitude_probe: AmplitudeInputProbe, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gate_heads
        self.amplitude_probe = amplitude_probe

    def change_gates(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        return self.amplitude_probe(state.z, state.hidden, actions).unsqueeze(-1)


def _installed_model(baseline, probe: AmplitudeInputProbe):
    parameter = next(baseline.parameters())
    model = ProbeGatedWorldModel(
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


def _ordered_episodes(replay, corpus):
    by_uid = {episode.uid: episode for episode in replay._episodes()}
    ordered = {
        layout: [by_uid[uid] for uid in corpus["episode_uids_by_layout"][layout]]
        for layout in temporal.SOURCE_LAYOUTS
    }
    if sum(map(len, ordered.values())) != corpus["episodes"]:
        raise AssertionError("could not reconstruct the exact collected episode order")
    return ordered


@torch.inference_mode()
def _extract_dataset(model, episodes_by_layout, journal, stage: str):
    episode_total = sum(map(len, episodes_by_layout.values()))
    completed = 0
    chunks = {name: [] for name in ("z", "hidden", "target", "actions", "changed")}
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
            episode_rows = {name: [] for name in chunks}
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
                episode_rows["z"].append(state.z[0])
                episode_rows["hidden"].append(state.hidden[0])
                episode_rows["target"].append(target)
                episode_rows["actions"].append(action[0])
                episode_rows["changed"].append(
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
            for name, values in episode_rows.items():
                chunks[name].append(torch.stack(values).cpu())
            completed += 1
            journal.update(stage, completed, episode_total, layout=layout)
    return {name: torch.cat(values) for name, values in chunks.items()}


def _target_summary(dataset) -> dict:
    target = dataset["target"]
    actions, changed = dataset["actions"], dataset["changed"]
    groups = {}
    for action in (2, 3):
        for event in (False, True):
            selected = target[(actions == action) & (changed == event)]
            groups[f"action{action}_{'changed' if event else 'nochange'}"] = {
                "transitions": len(selected),
                "members": selected.numel(),
                "mean": float(selected.mean()) if selected.numel() else None,
                "mse_to_half": float((selected - 0.5).square().mean())
                if selected.numel()
                else None,
            }
    return {
        "transitions": len(target),
        "members": target.numel(),
        "mean": float(target.mean()),
        "min": float(target.min()),
        "max": float(target.max()),
        "zero": int((target == 0).sum()),
        "one": int((target == 1).sum()),
        "interior": int(((target > 0) & (target < 1)).sum()),
        "groups": groups,
    }


def _probe_metrics(probe, dataset, class_weights, device) -> dict:
    squared, weighted, count = 0.0, 0.0, 0
    group_values = {}
    with torch.inference_mode():
        for start in range(0, len(dataset["actions"]), 4096):
            stop = min(start + 4096, len(dataset["actions"]))
            z = dataset["z"][start:stop].to(device)
            hidden = dataset["hidden"][start:stop].to(device)
            actions = dataset["actions"][start:stop].to(device)
            changed = dataset["changed"][start:stop].to(device)
            target = dataset["target"][start:stop].to(device).transpose(0, 1)
            prediction = probe(z, hidden, actions)
            errors = (prediction - target).square()
            squared += float(errors.sum())
            weights = class_weights[actions, changed.long()]
            weighted += float((errors * weights.unsqueeze(0)).sum())
            count += errors.numel()
    for action in (2, 3):
        for event in (False, True):
            mask = (dataset["actions"] == action) & (dataset["changed"] == event)
            selected = {name: value[mask] for name, value in dataset.items()}
            key = f"action{action}_{'changed' if event else 'nochange'}"
            if not len(selected["actions"]):
                group_values[key] = {"transitions": 0, "mse": None}
                continue
            with torch.inference_mode():
                prediction = probe(
                    selected["z"].to(device),
                    selected["hidden"].to(device),
                    selected["actions"].to(device),
                )
                target = selected["target"].to(device).transpose(0, 1)
                mse = float((prediction - target).square().mean())
            group_values[key] = {"transitions": len(selected["actions"]), "mse": mse}
    return {
        "mse": squared / count,
        "weighted_mse": weighted / count,
        "member_elements": count,
        "groups": group_values,
    }


def _fit_probe(
    name, use_hidden, train, heldout, class_weights, config, args, journal, trace
):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 161)
        probe = AmplitudeInputProbe(
            config.z_dim, config.h_dim, config.ensemble_size, use_hidden
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 161)
    losses = []
    journal.update(f"fit_{name}", 0, args.probe_updates)
    probe.train()
    for update in range(1, args.probe_updates + 1):
        indices = torch.randint(
            len(train["actions"]),
            (args.probe_batch_size,),
            generator=generator,
        )
        z = train["z"][indices].to(device)
        hidden = train["hidden"][indices].to(device)
        actions = train["actions"][indices].to(device)
        changed = train["changed"][indices].to(device)
        target = train["target"][indices].to(device).transpose(0, 1)
        predicted = probe(z, hidden, actions)
        valid = torch.ones(len(indices), device=device, dtype=torch.bool)
        loss = supervised.weighted_amplitude_mse(
            predicted, target, valid, actions, changed, class_weights
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        trace.write({"arm": name, "update": update, "loss": value})
        journal.update(f"fit_{name}", update, args.probe_updates, loss=value)
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


def _uid_digest(by_layout) -> str:
    payload = [episode.uid for layout in temporal.SOURCE_LAYOUTS for episode in by_layout[layout]]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _interpret(gates: Mapping[str, bool]) -> tuple[str, str]:
    z_pass, hidden_pass = gates["z"], gates["z_hidden"]
    if z_pass and hidden_pass:
        return (
            "both_linear_inputs_pass",
            "Teacher-forced probe learning is sufficient; exp160 failed because of its "
            "autoregressive training distribution or optimization path.",
        )
    if hidden_pass:
        return (
            "recurrent_context_required",
            "Recurrent context is necessary; hidden-conditioned gate training is licensed.",
        )
    if z_pass:
        return (
            "teacher_forced_z_probe_passes",
            "Direct teacher-forced z-probe learning is sufficient; exp160 failed from "
            "rollout-objective mismatch.",
        )
    return (
        "both_linear_inputs_fail",
        "Linear state inputs cannot predict the required amplitudes; next test "
        "nonlinearity or an object-centric state target, not longer retraining.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = residual.build_parser()
    parser.description = __doc__
    return parser


def _argv(argv) -> list[str]:
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frozen._prepare_output(args.out)
    started = time.monotonic()
    deadline = started + args.max_seconds
    command = os.environ.get("EXP161_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "baseline_checkpoint_git_head": None,
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
            baseline, _ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            baseline.eval().requires_grad_(False)
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
                replay_capacity=len(temporal.SOURCE_LAYOUTS)
                * args.episodes_per_layout,
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
            episodes = _ordered_episodes(replay, corpus)
            train_episodes, heldout_episodes = episode_disjoint_split(episodes)
            train_ids = {
                episode.uid for values in train_episodes.values() for episode in values
            }
            heldout_ids = {
                episode.uid for values in heldout_episodes.values() for episode in values
            }
            if train_ids & heldout_ids or len(train_ids | heldout_ids) != corpus["episodes"]:
                raise AssertionError("75/25 episode split is not complete and disjoint")
            coverage = _audit_counts({"all": replay._episodes()})
            counts = {
                action: {
                    key: row[key]
                    for key in ("total", "rgb_changed", "rgb_no_change")
                }
                for action, row in coverage["actions"].items()
            }
            fixed_counts = counts == FIXED_CORPUS["action_counts"]
            class_weights = auxiliary.action_class_weights(counts).to(config.device)
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head == EXPECTED_BASELINE_HEAD,
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
                "episode_disjoint": not bool(train_ids & heldout_ids),
                "probe_budget": args.probe_updates == 400
                and args.probe_batch_size == 256,
            }
            exact_protocol = all(matching.values())
            split_metadata = {
                "train_episodes": len(train_ids),
                "heldout_episodes": len(heldout_ids),
                "train_uid_digest": _uid_digest(train_episodes),
                "heldout_uid_digest": _uid_digest(heldout_episodes),
                "overlap": 0,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                protocol_match=matching,
                episode_split=split_metadata,
                action_counts=counts,
                class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)

            train = _extract_dataset(
                baseline, train_episodes, journal, "extract_train"
            )
            heldout = _extract_dataset(
                baseline, heldout_episodes, journal, "extract_heldout"
            )
            datasets = {
                "train": _target_summary(train),
                "heldout": _target_summary(heldout),
            }
            loss_trace = core.TraceWriter(args.out / "probe_losses.jsonl")
            probes, metrics = {}, {}
            try:
                for name, use_hidden in ARMS.items():
                    probes[name], metrics[name] = _fit_probe(
                        name,
                        use_hidden,
                        train,
                        heldout,
                        class_weights,
                        config,
                        args,
                        journal,
                        loss_trace,
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
            checkpoint_path = args.out / "amplitude_input_probes.pt"
            torch.save(
                {
                    "format_version": 1,
                    "analysis_git_head": manifest["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "objective": OBJECTIVE,
                    "z_dim": config.z_dim,
                    "h_dim": config.h_dim,
                    "ensemble_size": config.ensemble_size,
                    "arms": ARMS,
                    "probe_state_dicts": {
                        name: probe.state_dict() for name, probe in probes.items()
                    },
                },
                checkpoint_path,
            )
            journal.update("probe_checkpoint", 1, 1)

            diagnostics = {}
            candidates = {"native": baseline}
            candidates.update(
                {
                    name: _installed_model(baseline, probe)
                    for name, probe in probes.items()
                }
            )
            for name, model in candidates.items():
                journal.update(f"one_step_{name}", 0, 120)
                diagnostics[name] = one_step._diagnose(
                    model, journal, args.out / f"{name}_one_step_rows.jsonl"
                )
                journal.update(f"one_step_{name}", 120, 120)
            gates = {
                name: frozen.one_step_transfer_gate(
                    diagnostics[name]["splits"]["source"],
                    diagnostics[name]["splits"]["unseen"],
                    exact_protocol,
                )
                for name in ARMS
            }
            outcome, conclusion = _interpret(gates)
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "teacher-forced linear amplitude input diagnostic only",
                "interpretation_limit": "No composition, transfer, or AGI claim.",
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "objective": OBJECTIVE,
                "episode_split": split_metadata,
                "corpus": corpus,
                "action_counts": counts,
                "class_weights": class_weights.tolist(),
                "target_datasets": datasets,
                "probe_metrics": metrics,
                "one_step": diagnostics,
                "arm_gates": gates,
                "outcome": outcome,
                "conclusion": conclusion,
                "frozen_backbone_unchanged": backbone_unchanged,
                "controls": {
                    "teacher_forced_actual_z_sensors_mask": True,
                    "prediction_hidden_carried": True,
                    "raw_deltas_before_native_gate": True,
                    "analytic_targets_detached": True,
                    "train_transitions_only_for_probe_fit": True,
                    "heldout_episode_leakage": False,
                    "mlp": False,
                    "mpc": False,
                    "push2": False,
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "native_rows": "native_one_step_rows.jsonl",
                    "z_rows": "z_one_step_rows.jsonl",
                    "z_hidden_rows": "z_hidden_one_step_rows.jsonl",
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
