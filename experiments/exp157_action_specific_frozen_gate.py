"""Train only action-specific state gates over frozen exp153 raw deltas."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
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
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG
from snks.agent.core_agent import CoreAgent
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import GoalSpec, Mode
from snks.learning.core_objective import masked_mse
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config
from snks.pipeline.core_tasks import TaskCase


DEFAULT_BASELINE = Path(
    "output_to_user/core/exp153-change-gated-dynamics-001/gated_checkpoint.pt"
)
EXPECTED_BASELINE_HEAD = "49877e40f45156da2971b1802d748c551b2abc56"
PROTOCOL = dict(gated.PROTOCOL)
GATE_DEFINITION = {
    "input": "current_z only",
    "heads": "one Linear(z_dim, 1) for each ensemble member and primitive action",
    "selection": "action selects a distinct state-space decision boundary",
    "weight_init": 0.0,
    "bias_init": 0.0,
    "initial_probability": 0.5,
    "member_transition": "current_z + sigmoid(action_gate_i(z)) * frozen_delta_i(hidden)",
}
PREREGISTERED_GATES = {
    "one_step_transfer": (
        "exact protocol; BOTH source and unseen contact failures == 0, blocked "
        "failures == 0, median free-forward persistence ratio < 1"
    ),
    "source_composition": (
        "8 steps x 6 seeds x 4 layouts; natural-terminal fit coverage in every "
        "source layout; ordered_h3 >= 18/24 and >= 3/6 each layout and >= each "
        "control + 4 successes"
    ),
    "composition": "one_step_transfer and source_composition",
    "physics": "run Push-2 only if composition passes",
}


class ActionSpecificGateWorldModel(gated.ChangeGatedResidualWorldModel):
    """Use an independent current-z linear gate for each member/action pair."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.gate_heads
        action_counts = {shape[0] for shape in self.schemas.values()}
        if len(action_counts) != 1:
            raise ValueError("action-specific experiment requires one shared action count")
        self.gate_action_count = action_counts.pop()
        self.action_gate_heads = torch.nn.ModuleList(
            torch.nn.ModuleList(
                torch.nn.Linear(self.encoder.z_dim, 1)
                for _ in range(self.gate_action_count)
            )
            for _ in range(self.heads)
        )
        for member in self.action_gate_heads:
            for head in member:
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)

    def change_gates(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        by_member = []
        for member in self.action_gate_heads:
            all_logits = torch.cat([head(state.z) for head in member], dim=1)
            selected = all_logits.gather(1, actions[:, None])
            by_member.append(selected.sigmoid())
        return torch.stack(by_member)


def transfer_frozen_backbone(baseline):
    """Copy every non-gate exp153 tensor, then expose only new gates to Adam."""

    parameter = next(baseline.parameters())
    model = ActionSpecificGateWorldModel(
        CoreEncoder(baseline.encoder.z_dim),
        dict(baseline.schemas),
        baseline.h_dim,
        baseline.heads,
        normalize_sensor_condition=baseline.normalize_sensor_condition,
        predict_sensor_delta=baseline.predict_sensor_delta,
    ).to(device=parameter.device, dtype=parameter.dtype)
    candidate_keys = set(model.state_dict())
    transferable = {
        name: value
        for name, value in baseline.state_dict().items()
        if name in candidate_keys
    }
    incompatible = model.load_state_dict(transferable, strict=False)
    if incompatible.unexpected_keys or any(
        not name.startswith("action_gate_heads.")
        for name in incompatible.missing_keys
    ):
        raise RuntimeError(f"unexpected frozen transfer mismatch: {incompatible}")
    model.requires_grad_(False)
    model.action_gate_heads.requires_grad_(True)
    counts = {
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen": sum(p.numel() for p in model.parameters() if not p.requires_grad),
    }
    return model, counts


class LatentOnlyGateTrainer(CoreTrainer):
    """Preserve CoreTrainer rollout, optimizing only ensemble latent MSE."""

    def compute_loss(self, batch: SequenceBatch) -> torch.Tensor:
        batch_size, steps = batch.actions.shape
        stop = min(steps, batch.burn_in + self.config.train_horizon)
        if batch.burn_in < 0 or batch.burn_in >= stop:
            raise ValueError("burn-in must leave at least one prediction target")
        if not batch.valid[:, batch.burn_in:stop].any():
            raise ValueError("batch has no valid prediction targets")
        if (batch.valid[:, 1:] & ~batch.valid[:, :-1]).any():
            raise ValueError("padding must be at the sequence tail")
        rgb = batch.rgb[:, : stop + 1]
        z = self.model.encoder(rgb.reshape(-1, 3, 64, 64)).reshape(
            batch_size, stop + 1, -1
        )
        mask = batch.sensor_mask[:, 0]
        state = LatentState(
            z[:, 0],
            torch.where(mask, batch.sensors[:, 0], 0.0),
            mask,
            z.new_zeros(batch_size, self.model.h_dim),
            batch.schema,
        )
        latent_sum = z.new_zeros(())
        latent_count = 0
        for index in range(stop):
            valid = batch.valid[:, index]
            actions = torch.where(valid, batch.actions[:, index], 0)
            prediction = self.model.step(state, actions)
            if index < batch.burn_in:
                real_mask = batch.sensor_mask[:, index + 1]
                state = LatentState(
                    z[:, index + 1],
                    torch.where(real_mask, batch.sensors[:, index + 1], 0.0),
                    real_mask,
                    prediction.next_state.hidden,
                    batch.schema,
                )
                continue
            count = int(valid.sum()) * self.model.heads * z.shape[-1]
            latent_sum = latent_sum + masked_mse(
                prediction.member_z, z[:, index + 1], valid[None, :, None]
            ) * count
            latent_count += count
            state = prediction.next_state
        return latent_sum / max(latent_count, 1)


def one_step_transfer_gate(
    source: Mapping, unseen: Mapping, exact_protocol: bool
) -> bool:
    def split_passes(summary: Mapping) -> bool:
        ratio = summary["medians"]["free_forward_prediction_persistence_ratio"]
        return bool(
            summary["contact_failure_layouts"] == 0
            and summary["blocked_noop_failure_layouts"] == 0
            and ratio is not None
            and math.isfinite(ratio)
            and ratio < 1.0
        )

    return bool(exact_protocol and split_passes(source) and split_passes(unseen))


def _source_composition_gate(evaluation: Mapping, corpus: Mapping, args) -> bool:
    successes = evaluation["ordered_h3"]["overall"]["successes"]
    return bool(
        args.eval_steps == 8
        and args.eval_seeds == 6
        and all(corpus["natural_terminal_fit_episodes_by_layout"].values())
        and successes >= 18
        and min(
            row["successes"]
            for row in evaluation["ordered_h3"]["by_layout"].values()
        )
        >= 3
        and all(
            successes >= evaluation[control]["overall"]["successes"] + 4
            for control in ("ordered_h1", "shuffled_h3", "raw_h3")
        )
    )


def _physics_gate(source: Mapping, target: Mapping, args) -> bool:
    source_success = source["ordered_h3"]["overall"]["successes"]
    target_success = target["ordered_h3"]["overall"]["successes"]
    return bool(
        args.eval_steps == 8
        and args.eval_seeds == 6
        and target_success >= 18
        and min(
            row["successes"]
            for row in target["ordered_h3"]["by_layout"].values()
        )
        >= 3
        and all(
            target_success >= target[control]["overall"]["successes"] + 4
            for control in ("ordered_h1", "shuffled_h3", "raw_h3")
        )
        and target_success / max(source_success, 1) >= 0.75
    )


def _evaluate_push2(
    model, probes, config, seeds, steps, deadline, trace, journal
):
    """Run the established temporal MPC arms under Push-2 with Push-1 goal pixels."""

    evaluation = {}
    arms = (
        ("ordered_h3", "ordered", 3),
        ("ordered_h1", "ordered", 1),
        ("shuffled_h3", "shuffled", 3),
        ("raw_h3", None, 3),
    )
    total = len(temporal.TARGET_LAYOUTS) * len(seeds) * len(arms)
    completed = 0
    journal.update("evaluate_push2", 0, total)
    neutral = temporal.TerminationNeutralModel(model)
    for role, probe_name, horizon in arms:
        by_layout, all_results = {}, []
        for layout_name, (layout, _push1, _push2) in temporal.TARGET_LAYOUTS.items():
            layout_results = []
            for seed in seeds:
                core._check_deadline(deadline, f"push2/{role}/{layout_name}/{seed}")
                adapter = temporal._adapter(layout, 2, seed, steps)
                try:
                    goal = temporal._goal_observation(layout, 1, seed, steps)
                    case = TaskCase(
                        uid=f"exp157:push2:{role}:{layout_name}:{seed}",
                        family="push_box",
                        ruleset=f"push2:{layout_name}",
                        seed=seed,
                        split="validation",
                        goal=GoalSpec(goal, {}),
                        max_steps=steps,
                    )
                    episode_config = replace(
                        config, seed=seed, planner_horizon=horizon, beam_width=5
                    )
                    agent = (
                        CoreAgent(neutral, episode_config)
                        if probe_name is None
                        else temporal.TemporalAgent(
                            neutral, episode_config, probes[probe_name]
                        )
                    )
                    result = temporal._episode_result(adapter, agent, case)
                finally:
                    adapter.close()
                layout_results.append(result)
                all_results.append(result)
                trace.write(
                    {
                        "role": role,
                        "layout": layout_name,
                        "push_distance": 2,
                        "goal_push_distance": 1,
                        "seed": seed,
                        **core._result_record(result),
                        "audit": result.audit,
                    }
                )
                completed += 1
                journal.update(
                    "evaluate_push2",
                    completed,
                    total,
                    role=role,
                    layout=layout_name,
                    seed=seed,
                )
            by_layout[layout_name] = core._summarize_episodes(layout_results)
        evaluation[role] = {
            "overall": core._summarize_episodes(all_results),
            "by_layout": by_layout,
        }
    return evaluation


def _checkpoint_payload(model, ordered, shuffled, config, manifest, counts):
    return {
        "format_version": 5,
        "latent_parameterization": "frozen_exp153_delta_action_specific_gate",
        "event_balanced": False,
        "event_supervision": False,
        "analysis_git_head": manifest["analysis_git_head"],
        "git_head": manifest["analysis_git_head"],
        "baseline_checkpoint_git_head": manifest["baseline_checkpoint_git_head"],
        "budgets": manifest["budgets"],
        "config": core._jsonable(asdict(config)),
        "parameter_counts": counts,
        "modules": {
            "model": {
                "class": (
                    "experiments.exp157_action_specific_frozen_gate."
                    "ActionSpecificGateWorldModel"
                ),
                "schemas": core._jsonable(model.schemas),
                "z_dim": config.z_dim,
                "h_dim": config.h_dim,
                "ensemble_size": config.ensemble_size,
                "normalize_sensor_condition": config.normalize_sensor_condition,
                "predict_sensor_delta": config.predict_sensor_delta,
                "gate": GATE_DEFINITION,
            },
            "probe": {
                "z_dim": config.z_dim,
                "width": ordered.network[0].out_features,
            },
        },
        "model_state_dict": model.state_dict(),
        "ordered_probe_state_dict": ordered.state_dict(),
        "shuffled_probe_state_dict": shuffled.state_dict(),
    }


def _load_shuffled_probe(path: Path, metadata: Mapping):
    """Load the already-validated exp153 shuffled probe without refitting it."""

    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ValueError(f"could not safely reload shuffled probe: {error}") from error
    probe = temporal.TemporalProbe(
        metadata["z_dim"], width=metadata["probe_width"]
    )
    auxiliary.checkpoint_io._validate_state_dict(
        "shuffled_probe_state_dict",
        payload.get("shuffled_probe_state_dict"),
        probe.state_dict(),
    )
    probe.load_state_dict(payload["shuffled_probe_state_dict"], strict=True)
    return probe.to(metadata["device"]).eval().requires_grad_(False)


def _evaluate_frozen_probes(model, ordered, shuffled, fit, validation, args, journal):
    metrics = {"optimization_updates": 0, "source": "frozen exp153 checkpoint"}
    for completed, (role, by_layout) in enumerate(
        (("train", fit), ("validation", validation)), 1
    ):
        journal.update("probe_metrics", completed - 1, 2, split=role)
        episodes = [
            episode
            for name in temporal.SOURCE_LAYOUTS
            for episode in by_layout[name]
        ]
        encoded = temporal._encode_episodes(
            model, episodes, torch.device(next(model.parameters()).device)
        )
        pairs = temporal._pairs(encoded, args.max_horizon)
        metrics[role] = {
            "ordered": temporal._probe_metrics(ordered, pairs),
            "shuffled_endpoint": temporal._probe_metrics(shuffled, pairs),
        }
        journal.update("probe_metrics", completed, 2, split=role)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = residual.build_parser()
    parser.description = __doc__
    parser.set_defaults(baseline_checkpoint=DEFAULT_BASELINE)
    return parser


def _argv(argv) -> list[str]:
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def _prepare_output(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    unexpected = sorted(item.name for item in path.iterdir() if item.name != "run.log")
    if unexpected:
        raise FileExistsError(f"output directory is not empty: {unexpected}")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cutoff = round(0.75 * args.episodes_per_layout)
    if args.probe_episodes_per_layout > cutoff:
        parser.error("probe fit subset exceeds the source fit cutoff")
    if args.probe_validation_per_layout > args.episodes_per_layout - cutoff:
        parser.error("probe validation subset exceeds held-out source episodes")
    _prepare_output(args.out)
    started = time.monotonic()
    deadline = started + args.max_seconds
    command = os.environ.get("EXP157_LAUNCH_COMMAND") or shlex.join(_argv(argv))
    manifest = {
        "argv": _argv(argv),
        "exact_command": command,
        "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint_git_head": None,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "budgets": core._jsonable(vars(args)),
        "fixed_protocol": PROTOCOL,
        "fixed_config": FIXED_CONFIG,
        "gate_definition": GATE_DEFINITION,
        "preregistered_gates": PREREGISTERED_GATES,
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
            baseline, ordered, baseline_head, metadata = (
                auxiliary._load_gated_checkpoint(args.baseline_checkpoint)
            )
            shuffled = _load_shuffled_probe(args.baseline_checkpoint, metadata)
            manifest.update(
                baseline_checkpoint_git_head=baseline_head,
                baseline_checkpoint_metadata=metadata,
            )
            core._write_json(args.out / "manifest.json", manifest)
            model, parameter_counts = transfer_frozen_backbone(baseline)
            initial_backbone = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
                if not name.startswith("action_gate_heads.")
            }
            journal.update(
                "initialize", 1, 3, operation="frozen_transfer", **parameter_counts
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
            if next(model.parameters()).device.type != torch.device(config.device).type:
                raise ValueError("baseline checkpoint device and requested config disagree")
            trainer = LatentOnlyGateTrainer(model, config)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 3, 3, device=config.device)
            corpus, fit, validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            matching = {
                "canonical_checkpoint_path": args.baseline_checkpoint
                == DEFAULT_BASELINE,
                "baseline_checkpoint_head": baseline_head
                == EXPECTED_BASELINE_HEAD,
                "default_budgets": all(
                    getattr(args, key) == value for key, value in PROTOCOL.items()
                ),
                "baseline_budgets": all(
                    metadata["budgets"].get(key) == getattr(args, key)
                    for key in PROTOCOL
                ),
                "baseline_config": metadata["config"]
                == core._jsonable(asdict(config)),
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_counts": corpus["default_corpus_verified"],
                "baseline_unsupervised_gated": metadata["event_supervision"] is False,
                "only_new_gates_trainable": parameter_counts["trainable"]
                == config.ensemble_size * 5 * (config.z_dim + 1),
            }
            exact_protocol = all(matching.values())
            manifest.update(
                protocol_match=matching, parameter_counts=parameter_counts
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update(
                "dynamics", 0, args.dynamics_updates, exact_protocol=exact_protocol
            )
            dynamics_losses, schema_counts = [], {}
            loss_trace = core.TraceWriter(args.out / "dynamics_losses.jsonl")
            try:
                for completed in range(
                    0, args.dynamics_updates, args.dynamics_log_every
                ):
                    chunk = min(
                        args.dynamics_log_every, args.dynamics_updates - completed
                    )
                    losses, counts = core._train_updates(
                        model,
                        trainer,
                        replay,
                        config,
                        chunk,
                        Mode.ADAPT,
                        deadline,
                        schema="grid-v1",
                    )
                    for update, metrics in enumerate(losses, completed + 1):
                        loss_trace.write({"update": update, **metrics})
                    dynamics_losses.extend(losses)
                    for schema, count in counts.items():
                        schema_counts[schema] = schema_counts.get(schema, 0) + count
                    journal.update(
                        "dynamics",
                        completed + chunk,
                        args.dynamics_updates,
                        loss=losses[-1]["loss"],
                    )
            finally:
                loss_trace.close()
            backbone_unchanged = all(
                torch.equal(model.state_dict()[name], value)
                for name, value in initial_backbone.items()
            )
            if not backbone_unchanged:
                raise AssertionError("a frozen exp153 backbone tensor changed")
            model.eval()
            probe_metrics = _evaluate_frozen_probes(
                model, ordered, shuffled, fit, validation, args, journal
            )
            journal.update("action_gate_checkpoint", 0, 1)
            checkpoint_path = args.out / "action_specific_gate_checkpoint.pt"
            torch.save(
                _checkpoint_payload(
                    model,
                    ordered,
                    shuffled,
                    config,
                    manifest,
                    parameter_counts,
                ),
                checkpoint_path,
            )
            journal.update("action_gate_checkpoint", 1, 1)
            diagnostics = {}
            for role, candidate in (("baseline", baseline), ("candidate", model)):
                journal.update(f"{role}_one_step", 0, 120)
                diagnostics[role] = one_step._diagnose(
                    candidate,
                    journal,
                    args.out / f"{role}_one_step_rows.jsonl",
                )
                journal.update(f"{role}_one_step", 120, 120)
            statistics = gated._gate_statistics(
                model,
                args.out / "candidate_one_step_rows.jsonl",
                args.out / "gate_rows.jsonl",
                deadline,
                journal,
            )
            late_fork = temporal._late_fork_audit(
                model,
                ordered,
                config,
                deadline,
                journal,
                args.out / "candidate_late_fork_rows.jsonl",
            )
            total = len(temporal.TARGET_LAYOUTS) * args.eval_seeds * 4
            journal.update("evaluate", 0, total)
            evaluation_trace = core.TraceWriter(
                args.out / "evaluation_traces.jsonl"
            )
            try:
                evaluation = temporal._evaluate(
                    model,
                    {"ordered": ordered, "shuffled": shuffled},
                    config,
                    replay,
                    range(20000, 20000 + args.eval_seeds),
                    args.eval_steps,
                    deadline,
                    evaluation_trace,
                    journal,
                    [0],
                    total,
                )
            finally:
                evaluation_trace.close()
            transfer_gate = one_step_transfer_gate(
                diagnostics["candidate"]["splits"]["source"],
                diagnostics["candidate"]["splits"]["unseen"],
                exact_protocol,
            )
            source_gate = _source_composition_gate(evaluation, corpus, args)
            composition_gate = bool(transfer_gate and source_gate)
            physics_evaluation = None
            physics_transfer_gate = None
            if composition_gate:
                physics_trace = core.TraceWriter(
                    args.out / "physics_evaluation_traces.jsonl"
                )
                try:
                    physics_evaluation = _evaluate_push2(
                        model,
                        {"ordered": ordered, "shuffled": shuffled},
                        config,
                        range(20000, 20000 + args.eval_seeds),
                        args.eval_steps,
                        deadline,
                        physics_trace,
                        journal,
                    )
                finally:
                    physics_trace.close()
                physics_transfer_gate = _physics_gate(
                    evaluation, physics_evaluation, args
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "frozen-backbone action-specific gate causal intervention",
                "interpretation_limit": (
                    "No AGI, JEPA, concept, composition, or transfer proof."
                ),
                "analysis_git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "exact_command": command,
                "exact_protocol": exact_protocol,
                "protocol_match": matching,
                "preregistered_gates": PREREGISTERED_GATES,
                "one_step_transfer_gate": transfer_gate,
                "source_compositional_gate": source_gate,
                "composition_gate": composition_gate,
                "physics_transfer_gate": physics_transfer_gate,
                "corpus": corpus,
                "dynamics": {
                    "updates": args.dynamics_updates,
                    "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"],
                    "schema_counts": schema_counts,
                    "objective": "ensemble latent predictive MSE only",
                    "burn_in": config.burn_in,
                    "train_horizon": config.train_horizon,
                },
                "parameter_counts": parameter_counts,
                "frozen_backbone_unchanged": backbone_unchanged,
                "probe": probe_metrics,
                "baseline_one_step": diagnostics["baseline"],
                "candidate_one_step": diagnostics["candidate"],
                "gate_statistics": statistics,
                "candidate_late_fork": late_fork,
                "evaluation": evaluation,
                "physics_evaluation": physics_evaluation,
                "controls": {
                    "baseline_checkpoint": str(args.baseline_checkpoint),
                    "backbone_and_raw_delta_heads_frozen": True,
                    "only_action_specific_gate_optimized": True,
                    "event_balanced_sampling": False,
                    "event_supervision": False,
                    "rgb_change_auxiliary": False,
                    "task_success_supervision": False,
                    "planner_changed": False,
                    "sampling_changed": False,
                    "autoregressive_horizon_preserved": True,
                    "sensor_and_termination_losses": False,
                    "push2_run": composition_gate,
                },
                "conclusion": (
                    "Action-specific gates learned the registered oracle-like one-step "
                    "amplitudes from ordinary latent prediction."
                    if transfer_gate
                    else "Exp156 established raw-direction expressivity, but action-specific "
                    "gate expressivity alone was insufficient under uniform latent MSE; "
                    "next diagnose or learn amplitude targets, not new raw directions."
                ),
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "losses": "dynamics_losses.jsonl",
                    "baseline_rows": "baseline_one_step_rows.jsonl",
                    "candidate_rows": "candidate_one_step_rows.jsonl",
                    "gate_rows": "gate_rows.jsonl",
                    "late_fork_rows": "candidate_late_fork_rows.jsonl",
                    "evaluation_traces": "evaluation_traces.jsonl",
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
                runtime_seconds=time.monotonic() - started,
                exact_protocol=exact_protocol,
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
