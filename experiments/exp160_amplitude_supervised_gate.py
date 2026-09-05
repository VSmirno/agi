"""Train frozen exp153 action-specific gates on analytic amplitude targets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
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
from experiments import exp156_gated_delta_oracle as raw_oracle
from experiments import exp157_action_specific_frozen_gate as frozen
from experiments import exp159_independent_amplitude_oracle as amplitude_oracle
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.env.core_types import Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


DEFAULT_BASELINE = frozen.DEFAULT_BASELINE
EXPECTED_BASELINE_HEAD = frozen.EXPECTED_BASELINE_HEAD
PROTOCOL = dict(frozen.PROTOCOL)
ActionSpecificGateWorldModel = frozen.ActionSpecificGateWorldModel
transfer_frozen_backbone = frozen.transfer_frozen_backbone
one_step_transfer_gate = frozen.one_step_transfer_gate
PREREGISTERED_GATES = dict(frozen.PREREGISTERED_GATES)
OBJECTIVE = {
    "target": "detached exp159 independent analytic amplitude per member",
    "target_formula": "clip(dot(d_i,t)/dot(d_i,d_i),0,1); zero direction -> 0",
    "target_displacement": "actual next_z - current predicted state.z",
    "raw_delta": "frozen latent head on native recurrent hidden before gate",
    "prediction": "sigmoid action-specific current-z gate",
    "error": "per-transition/member squared amplitude error",
    "weight": "fixed weight[action, observed_rgb_change] from exact full corpus",
    "denominator": "ordinary count of valid member elements",
    "sampled_batch_renormalization": False,
    "rgb_change_role": "weight selection only; not a BCE or prediction target",
}


def analytic_amplitude_targets(
    raw_deltas: torch.Tensor, displacement: torch.Tensor
) -> torch.Tensor:
    """Return detached exp159 targets for every member and batch row."""

    return amplitude_oracle.independent_member_amplitudes(
        raw_deltas, displacement
    ).detach()


def weighted_amplitude_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    actions: torch.Tensor,
    changed: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """Fixed-population amplitude MSE with an ordinary member denominator."""

    if predicted.ndim != 2 or target.shape != predicted.shape:
        raise ValueError("predicted and target must match [members,batch]")
    batch = predicted.shape[1]
    if any(value.shape != (batch,) for value in (valid, actions, changed)):
        raise ValueError("valid, actions and changed must have shape [batch]")
    if valid.dtype is not torch.bool or changed.dtype is not torch.bool:
        raise ValueError("valid and changed must be boolean")
    if actions.dtype is not torch.long:
        raise ValueError("actions must be long")
    if class_weights.shape != (5, 2):
        raise ValueError("class_weights must be 5x2")
    if not torch.isfinite(class_weights).all() or (class_weights < 0).any():
        raise ValueError("class_weights must be finite and non-negative")
    count = int(valid.sum()) * predicted.shape[0]
    if count == 0:
        return predicted.new_zeros(())
    weights = class_weights[actions, changed.long()]
    errors = (predicted - target).square() * weights.unsqueeze(0)
    return errors.masked_select(valid.unsqueeze(0).expand_as(errors)).sum() / count


class AmplitudeSupervisedGateTrainer(CoreTrainer):
    """Train only gates against detached per-member analytic amplitudes."""

    def __init__(self, model, config, class_weights: torch.Tensor):
        if class_weights.shape != (5, 2):
            raise ValueError("class_weights must be 5x2")
        self.class_weights = class_weights.detach().to(next(model.parameters()).device)
        self._target_count = 0
        self._target_sum = 0.0
        self._target_min = float("inf")
        self._target_max = float("-inf")
        self._target_zero = 0
        self._target_one = 0
        super().__init__(model, config)

    def target_distribution(self) -> dict:
        count = self._target_count
        return {
            "scope": "valid sampled supervised member targets across optimizer updates",
            "count": count,
            "mean": self._target_sum / count if count else None,
            "min": self._target_min if count else None,
            "max": self._target_max if count else None,
            "zero": self._target_zero,
            "one": self._target_one,
            "interior": count - self._target_zero - self._target_one,
        }

    def _record_targets(self, target: torch.Tensor, valid: torch.Tensor) -> None:
        selected = target.masked_select(valid.unsqueeze(0).expand_as(target))
        if not selected.numel():
            return
        self._target_count += selected.numel()
        self._target_sum += float(selected.sum())
        self._target_min = min(self._target_min, float(selected.min()))
        self._target_max = max(self._target_max, float(selected.max()))
        self._target_zero += int((selected == 0).sum())
        self._target_one += int((selected == 1).sum())

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
        labels = (rgb[:, 1:] != rgb[:, :-1]).flatten(2).any(-1)
        mask = batch.sensor_mask[:, 0]
        state = LatentState(
            z[:, 0],
            torch.where(mask, batch.sensors[:, 0], 0.0),
            mask,
            z.new_zeros(batch_size, self.model.h_dim),
            batch.schema,
        )
        amplitude_sum = z.new_zeros(())
        ordinary_count = 0
        for index in range(stop):
            valid = batch.valid[:, index]
            actions = torch.where(valid, batch.actions[:, index], 0)
            prediction, raw_deltas = raw_oracle.native_prediction_and_raw_deltas(
                self.model, state, actions
            )
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
            target = analytic_amplitude_targets(
                raw_deltas, z[:, index + 1] - state.z
            )
            predicted = self.model.change_gates(state, actions).squeeze(-1)
            self._record_targets(target, valid)
            count = int(valid.sum()) * self.model.heads
            amplitude_sum = amplitude_sum + weighted_amplitude_mse(
                predicted,
                target,
                valid,
                actions,
                labels[:, index],
                self.class_weights,
            ) * count
            ordinary_count += count
            state = prediction.next_state
        return amplitude_sum / max(ordinary_count, 1)


def _checkpoint_payload(
    model, ordered, shuffled, config, manifest, counts, supervision
):
    payload = frozen._checkpoint_payload(
        model, ordered, shuffled, config, manifest, counts
    )
    payload.update(
        format_version=7,
        latent_parameterization=(
            "frozen_exp153_delta_action_specific_gate_amplitude_supervised"
        ),
        amplitude_supervision=supervision,
    )
    return payload


def build_parser():
    parser = frozen.build_parser()
    parser.description = __doc__
    return parser


def _argv(argv) -> list[str]:
    if argv is None:
        return list(sys.orig_argv)
    return [sys.executable, str(Path(__file__)), *argv]


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cutoff = round(0.75 * args.episodes_per_layout)
    if args.probe_episodes_per_layout > cutoff:
        parser.error("probe fit subset exceeds the source fit cutoff")
    if args.probe_validation_per_layout > args.episodes_per_layout - cutoff:
        parser.error("probe validation subset exceeds held-out source episodes")
    frozen._prepare_output(args.out)
    started = time.monotonic()
    deadline = started + args.max_seconds
    command = os.environ.get("EXP160_LAUNCH_COMMAND") or shlex.join(_argv(argv))
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
        "objective": OBJECTIVE,
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
            shuffled = frozen._load_shuffled_probe(
                args.baseline_checkpoint, metadata
            )
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
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 3, 3, device=config.device)
            corpus, fit, validation = residual._collect_corpus(
                args, replay, deadline, journal
            )
            journal.update("class_counts", 0, 1)
            coverage = _audit_counts({"all": replay._episodes()})
            counts = {
                action: {
                    key: row[key]
                    for key in ("total", "rgb_changed", "rgb_no_change")
                }
                for action, row in coverage["actions"].items()
            }
            fixed_counts = counts == FIXED_CORPUS["action_counts"]
            if corpus["default_corpus_verified"] and not fixed_counts:
                raise AssertionError(f"exp149 action/change counts mismatch: {counts}")
            class_weights = auxiliary.action_class_weights(counts)
            supervision = {
                **OBJECTIVE,
                "action_counts": counts,
                "class_weights": class_weights.tolist(),
                "weight_formula": "N_action / (K_present * N_action_class)",
                "computed_once_before_training": True,
            }
            trainer = AmplitudeSupervisedGateTrainer(model, config, class_weights)
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
                "fixed_corpus_and_class_counts": corpus["default_corpus_verified"]
                and fixed_counts,
                "baseline_unsupervised_gated": metadata["event_supervision"] is False,
                "only_new_gates_trainable": parameter_counts["trainable"]
                == config.ensemble_size * 5 * (config.z_dim + 1),
            }
            exact_protocol = all(matching.values())
            manifest.update(
                protocol_match=matching,
                parameter_counts=parameter_counts,
                amplitude_supervision=supervision,
            )
            core._write_json(args.out / "manifest.json", manifest)
            journal.update(
                "class_counts",
                1,
                1,
                class_weights=class_weights.tolist(),
                exact_protocol=exact_protocol,
            )
            journal.update("dynamics", 0, args.dynamics_updates)
            dynamics_losses, schema_counts = [], {}
            loss_trace = core.TraceWriter(args.out / "dynamics_losses.jsonl")
            try:
                for completed in range(
                    0, args.dynamics_updates, args.dynamics_log_every
                ):
                    chunk = min(
                        args.dynamics_log_every, args.dynamics_updates - completed
                    )
                    losses, by_schema = core._train_updates(
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
                    for schema, count in by_schema.items():
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
            probe_metrics = frozen._evaluate_frozen_probes(
                model, ordered, shuffled, fit, validation, args, journal
            )
            target_distribution = trainer.target_distribution()
            journal.update("amplitude_gate_checkpoint", 0, 1)
            checkpoint_path = args.out / "amplitude_supervised_gate_checkpoint.pt"
            torch.save(
                _checkpoint_payload(
                    model,
                    ordered,
                    shuffled,
                    config,
                    manifest,
                    parameter_counts,
                    supervision,
                ),
                checkpoint_path,
            )
            journal.update("amplitude_gate_checkpoint", 1, 1)
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
            evaluation_trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
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
            source_gate = frozen._source_composition_gate(evaluation, corpus, args)
            composition_gate = bool(transfer_gate and source_gate)
            physics_evaluation = None
            physics_transfer_gate = None
            if composition_gate:
                physics_trace = core.TraceWriter(
                    args.out / "physics_evaluation_traces.jsonl"
                )
                try:
                    physics_evaluation = frozen._evaluate_push2(
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
                physics_transfer_gate = frozen._physics_gate(
                    evaluation, physics_evaluation, args
                )
            journal.update("artifacts", 0, 2, operation="write_results")
            result = {
                "status": "completed",
                "claim": "direct analytic-amplitude supervision causal intervention",
                "interpretation_limit": "No AGI, JEPA, composition, or transfer proof.",
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
                "amplitude_supervision": {
                    **supervision,
                    "sampled_target_distribution": target_distribution,
                },
                "corpus": corpus,
                "dynamics": {
                    "updates": args.dynamics_updates,
                    "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"],
                    "schema_counts": schema_counts,
                    "objective": "fixed-weight analytic amplitude MSE only",
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
                    "architecture_changed_from_exp157": False,
                    "backbone_and_raw_delta_heads_frozen": True,
                    "only_action_specific_gate_optimized": True,
                    "sampling_changed": False,
                    "planner_changed": False,
                    "probe_refit": False,
                    "event_supervision": False,
                    "rgb_change_used_only_for_fixed_loss_weight": True,
                    "analytic_target_detached": True,
                    "autoregressive_horizon_preserved": True,
                    "task_success_supervision": False,
                    "sensor_and_termination_losses": False,
                    "push2_run": composition_gate,
                },
                "conclusion": (
                    "Direct analytic amplitude learning fixes registered local dynamics; "
                    "composition remains a separate gate."
                    if transfer_gate
                    else "Direct supervision failed despite exp159's independent oracle; "
                    "the z-only gate did not learn or generalize the amplitude mapping, "
                    "or autoregressive targets shifted. Next inspect hidden/context input "
                    "or a state-target probe, not a weight sweep or raw-delta replacement."
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
