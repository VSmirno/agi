"""Nonlinear z+carried-hidden probe for independent amplitude targets."""

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

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp148_source_target_one_step as one_step
from experiments import exp150_residual_dynamics as residual
from experiments import exp153_change_gated_dynamics as gated
from experiments import exp154_auxiliary_change_gate as auxiliary
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp160_amplitude_supervised_gate as supervised
from experiments import exp161_amplitude_input_probe as linear_probe
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
DEFAULT_LINEAR_REFERENCE = Path(
    "output_to_user/core/exp161-amplitude-input-probe-001/results.json"
)
EXPECTED_LINEAR_HEAD = "419fe0018fc7d6584fc2c42d1ffdf74dab3e5494"
PROTOCOL = dict(residual.PROTOCOL)
HIDDEN_WIDTH = 128
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "input": "teacher-forced current z + carried frozen recurrent hidden",
    "architecture": (
        "separate per-action Linear(z_dim+h_dim,128)->ReLU->"
        "Linear(128,ensemble_size)->sigmoid"
    ),
    "weight": "fixed full-corpus weight[action, observed_rgb_change]",
    "denominator": "ordinary sampled member count; no batch renormalization",
    "updates": 400,
    "batch_size": 256,
    "split": "first 75% episodes per layout train; final 25% held out",
}


class NonlinearAmplitudeProbe(torch.nn.Module):
    """One fixed-width nonlinear z+hidden head for each primitive action."""

    def __init__(self, z_dim: int, h_dim: int, heads: int):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.heads = heads
        self.by_action = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(z_dim + h_dim, HIDDEN_WIDTH),
                torch.nn.ReLU(),
                torch.nn.Linear(HIDDEN_WIDTH, heads),
            )
            for _ in range(5)
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
        features = torch.cat((z, hidden), dim=-1)
        logits = torch.stack([head(features) for head in self.by_action], dim=1)
        selected = logits.gather(
            1, actions[:, None, None].expand(-1, 1, self.heads)
        ).squeeze(1)
        return selected.sigmoid().transpose(0, 1)


class NonlinearProbeGatedWorldModel(gated.ChangeGatedResidualWorldModel):
    """Use the fitted nonlinear probe as an experiment-only residual gate."""

    def __init__(self, *args, amplitude_probe: NonlinearAmplitudeProbe, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gate_heads
        self.amplitude_probe = amplitude_probe

    def change_gates(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        return self.amplitude_probe(state.z, state.hidden, actions).unsqueeze(-1)


def _installed_model(baseline, probe: NonlinearAmplitudeProbe):
    parameter = next(baseline.parameters())
    model = NonlinearProbeGatedWorldModel(
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


def nonlinear_probe_gate(source: Mapping, unseen: Mapping, exact: bool) -> bool:
    """Apply exact source/unseen local predicates, including interaction gain."""

    if not frozen.one_step_transfer_gate(source, unseen, exact):
        return False
    for summary in (source, unseen):
        ratio = summary["medians"]["interact_prediction_persistence_ratio"]
        if ratio is None or not math.isfinite(ratio) or ratio >= 1.0:
            return False
    return True


def _fit_probe(train, heldout, class_weights, config, args, journal, trace):
    device = torch.device(config.device)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + 162)
        probe = NonlinearAmplitudeProbe(
            config.z_dim, config.h_dim, config.ensemble_size
        ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed + 162)
    losses = []
    journal.update("fit_nonlinear", 0, args.probe_updates)
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
        prediction = probe(z, hidden, actions)
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
        journal.update("fit_nonlinear", update, args.probe_updates, loss=value)
    probe.eval().requires_grad_(False)
    return probe, {
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "updates": args.probe_updates,
        "batch_size": args.probe_batch_size,
        "learning_rate": config.learning_rate,
        "train": linear_probe._probe_metrics(
            probe, train, class_weights, device
        ),
        "heldout": linear_probe._probe_metrics(
            probe, heldout, class_weights, device
        ),
    }


def _load_linear_reference(path: Path) -> dict:
    try:
        reference = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load exp161 reference: {error}") from error
    if reference.get("analysis_git_head") != EXPECTED_LINEAR_HEAD:
        raise ValueError("exp161 reference analysis head mismatch")
    if reference.get("exact_protocol") is not True:
        raise ValueError("exp161 reference is not exact protocol")
    metrics = reference.get("probe_metrics", {}).get("z_hidden")
    diagnostic = reference.get("one_step", {}).get("z_hidden")
    gate = reference.get("arm_gates", {}).get("z_hidden")
    if not isinstance(metrics, Mapping) or not isinstance(diagnostic, Mapping):
        raise ValueError("exp161 reference lacks z_hidden metrics")
    if not isinstance(gate, bool):
        raise ValueError("exp161 reference lacks z_hidden gate")
    return {
        "path": str(path),
        "analysis_git_head": reference["analysis_git_head"],
        "probe_metrics": dict(metrics),
        "one_step": dict(diagnostic),
        "gate": gate,
    }


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--linear-reference", type=Path, default=DEFAULT_LINEAR_REFERENCE
    )
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
    command = os.environ.get("EXP162_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "linear_reference": str(args.linear_reference),
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
            journal.update("initialize", 1, 3, operation="load_exp161_reference")
            reference = _load_linear_reference(args.linear_reference)
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
                "canonical_linear_reference": args.linear_reference
                == DEFAULT_LINEAR_REFERENCE,
                "linear_reference_head": reference["analysis_git_head"]
                == EXPECTED_LINEAR_HEAD,
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
                "train_uid_digest": linear_probe._uid_digest(train_episodes),
                "heldout_uid_digest": linear_probe._uid_digest(heldout_episodes),
                "overlap": 0,
            }
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
                linear_reference_metadata={
                    "analysis_git_head": reference["analysis_git_head"],
                    "gate": reference["gate"],
                },
                protocol_match=matching,
                episode_split=split_metadata,
                action_counts=counts,
                class_weights=class_weights.tolist(),
            )
            core._write_json(args.out / "manifest.json", manifest)

            train = linear_probe._extract_dataset(
                baseline, train_episodes, journal, "extract_train"
            )
            heldout = linear_probe._extract_dataset(
                baseline, heldout_episodes, journal, "extract_heldout"
            )
            datasets = {
                "train": linear_probe._target_summary(train),
                "heldout": linear_probe._target_summary(heldout),
            }
            loss_trace = core.TraceWriter(args.out / "probe_losses.jsonl")
            try:
                probe, metrics = _fit_probe(
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
            checkpoint_path = args.out / "nonlinear_amplitude_probe.pt"
            torch.save(
                {
                    "format_version": 1,
                    "analysis_git_head": manifest["analysis_git_head"],
                    "baseline_checkpoint_git_head": baseline_head,
                    "objective": OBJECTIVE,
                    "z_dim": config.z_dim,
                    "h_dim": config.h_dim,
                    "ensemble_size": config.ensemble_size,
                    "hidden_width": HIDDEN_WIDTH,
                    "probe_state_dict": probe.state_dict(),
                },
                checkpoint_path,
            )
            journal.update("probe_checkpoint", 1, 1)
            model = _installed_model(baseline, probe)
            journal.update("one_step_nonlinear", 0, 120)
            diagnostic = one_step._diagnose(
                model,
                journal,
                args.out / "nonlinear_one_step_rows.jsonl",
            )
            journal.update("one_step_nonlinear", 120, 120)
            gate = nonlinear_probe_gate(
                diagnostic["splits"]["source"],
                diagnostic["splits"]["unseen"],
                exact_protocol,
            )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "checkpoint-only nonlinear amplitude input diagnostic",
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
                "one_step": diagnostic,
                "nonlinear_probe_gate": gate,
                "exp161_z_hidden_reference": reference,
                "frozen_backbone_unchanged": backbone_unchanged,
                "controls": {
                    "only_causal_change_from_exp161": "nonlinear probe head",
                    "teacher_forced_actual_z_sensors_mask": True,
                    "prediction_hidden_carried": True,
                    "raw_deltas_before_native_gate": True,
                    "analytic_targets_detached": True,
                    "linear_arm_retrained": False,
                    "mpc": False,
                    "push2": False,
                },
                "conclusion": (
                    "The nonlinear state interaction passes the registered local gate; "
                    "nonlinear amplitude training is licensed, while composition remains separate."
                    if gate
                    else "The nonlinear z+hidden MLP still fails the registered local gate; "
                    "next test an object-centric state or target rather than longer probe fitting."
                ),
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "probe_losses.jsonl",
                    "rows": "nonlinear_one_step_rows.jsonl",
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
