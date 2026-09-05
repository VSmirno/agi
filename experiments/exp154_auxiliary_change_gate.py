"""Matched exp153 gated dynamics with an RGB-change self-supervised objective."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp147_rollout_localization as checkpoint_io
from experiments import exp148_source_target_one_step as one_step
from experiments import exp150_residual_dynamics as residual
from experiments import exp153_change_gated_dynamics as gated
from experiments.exp149_replay_coverage import _audit_counts
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, FIXED_CORPUS
from snks.agent.core_world_model import LatentState
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, SequenceBatch
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import CoreConfig, load_core_config


PROTOCOL = {**gated.PROTOCOL, "gate_aux_weight": 1.0}
GATED_CLASS = "experiments.exp153_change_gated_dynamics.ChangeGatedResidualWorldModel"
AUXILIARY_CLASS = "experiments.exp154_auxiliary_change_gate.AuxiliaryChangeGatedWorldModel"
PREREGISTERED_GATES = {
    "one_step": (
        "exact protocol; source contact failures == 0 and <= frozen exp153 baseline; "
        "source blocked failures == 0 and < baseline; median free-forward ratio < 1"
    ),
    "baseline_reference_counts": {"contact_failure_layouts": 0, "blocked_noop_failure_layouts": 4},
    "contact_non_regression_reason": "exp153 already has zero contact failures",
    "source_compositional": (
        "8 steps x 6 seeds x 4 layouts; natural-terminal fit coverage in every source layout; "
        "ordered_h3 >= 18/24 and >= 3/6 each layout and >= each control + 4 successes"
    ),
    "composition": "one_step and source_compositional",
    "gate_diagnostic": (
        "separately in source and unseen: mean action-2 changed gate exceeds both "
        "blocked step-0 and blocked step-2 gates by >= 0.15; "
        "mean action-3 changed gate exceeds no-change gate by >= 0.15"
    ),
    "gate_diagnostic_role": "supporting only; not part of one_step or composition gates",
}


class AuxiliaryChangeGatedWorldModel(gated.ChangeGatedResidualWorldModel):
    """Expose the same exp153 gate logits without adding parameters or inputs."""

    def change_gate_logits(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        condition = torch.cat((state.z, self.action_embeddings[state.schema](actions)), dim=-1)
        return torch.stack([head(condition) for head in self.gate_heads])

    def change_gates(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        return self.change_gate_logits(state, actions).sigmoid()


def action_class_weights(counts: Mapping, n_actions: int = 5) -> torch.Tensor:
    """Weight present class c by N_action / (K_present * N_action,c)."""
    weights = torch.zeros(n_actions, 2)
    for action in range(n_actions):
        row = counts.get(str(action), {})
        frequencies = [row.get("rgb_no_change", 0), row.get("rgb_changed", 0)]
        if any(not isinstance(count, int) or count < 0 for count in frequencies):
            raise ValueError("class counts must be non-negative integers")
        total, present = sum(frequencies), sum(count > 0 for count in frequencies)
        for label, count in enumerate(frequencies):
            if count:
                weights[action, label] = total / (present * count)
    return weights


def auxiliary_gate_examples(
    model: AuxiliaryChangeGatedWorldModel, batch: SequenceBatch, train_horizon: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repeat CoreTrainer's burn-in then rollout; future RGB supplies labels only.

    Re-encoding the real prefix and repeating the short autoregressive pass avoids
    hooks or cached graph tensors. After burn-in, inputs are prediction.next_state,
    exactly as in the predictive objective, never real supervised future frames.
    """
    batch_size, steps = batch.actions.shape
    stop = min(steps, batch.burn_in + train_horizon)
    if batch.burn_in < 0 or batch.burn_in >= stop:
        raise ValueError("burn-in must leave at least one prediction target")
    if not batch.valid[:, batch.burn_in:stop].any():
        raise ValueError("batch has no valid prediction targets")
    if (batch.valid[:, 1:] & ~batch.valid[:, :-1]).any():
        raise ValueError("padding must be at the sequence tail")
    prefix = batch.rgb[:, :batch.burn_in + 1]
    z = model.encoder(prefix.reshape(-1, 3, 64, 64)).reshape(batch_size, batch.burn_in + 1, -1)
    mask = batch.sensor_mask[:, 0]
    state = LatentState(z[:, 0], torch.where(mask, batch.sensors[:, 0], 0.0),
                        mask, z.new_zeros(batch_size, model.h_dim), batch.schema)
    labels = (batch.rgb[:, 1:stop + 1] != batch.rgb[:, :stop]).flatten(2).any(-1)
    logits, targets, selected_actions = [], [], []
    for index in range(stop):
        valid = batch.valid[:, index]
        actions = torch.where(valid, batch.actions[:, index], 0)
        if index >= batch.burn_in:
            logits.append(model.change_gate_logits(state, actions).squeeze(-1)[:, valid])
            targets.append(labels[:, index][valid].to(z.dtype))
            selected_actions.append(actions[valid])
        prediction = model.step(state, actions)
        if index < batch.burn_in:
            real_mask = batch.sensor_mask[:, index + 1]
            state = LatentState(z[:, index + 1],
                                torch.where(real_mask, batch.sensors[:, index + 1], 0.0),
                                real_mask, prediction.next_state.hidden, batch.schema)
        else:
            state = prediction.next_state
    return torch.cat(logits, dim=1), torch.cat(targets), torch.cat(selected_actions)


class AuxiliaryGateTrainer(CoreTrainer):
    """Add weighted ensemble BCE while preserving CoreTrainer loss/update code."""

    def __init__(self, model: AuxiliaryChangeGatedWorldModel, config: CoreConfig,
                 class_weights: torch.Tensor, gate_aux_weight: float = 1.0):
        if not math.isfinite(gate_aux_weight) or gate_aux_weight < 0:
            raise ValueError("gate_aux_weight must be finite and non-negative")
        if class_weights.shape != (5, 2) or not torch.isfinite(class_weights).all() or (class_weights < 0).any():
            raise ValueError("class_weights must be a finite non-negative 5x2 tensor")
        super().__init__(model, config)
        self.class_weights = class_weights.detach().to(next(model.parameters()).device)
        self.gate_aux_weight = gate_aux_weight
        # Reporting contains scalars only, never graph tensors or gate inputs.
        self.loss_components: dict[str, float] = {}

    def compute_auxiliary_loss(self, batch: SequenceBatch) -> torch.Tensor:
        logits, labels, actions = auxiliary_gate_examples(self.model, batch, self.config.train_horizon)
        weights = self.class_weights[actions, labels.long()]
        errors = F.binary_cross_entropy_with_logits(logits, labels.expand_as(logits), reduction="none")
        # Mean across every valid transition and ensemble member. Fixed weights
        # have corpus expectation one within action; no sampled-batch renormalizing.
        return (errors * weights).mean()

    def compute_loss(self, batch: SequenceBatch) -> torch.Tensor:
        predictive = super().compute_loss(batch)
        auxiliary = self.compute_auxiliary_loss(batch) if self.gate_aux_weight else predictive.new_zeros(())
        self.loss_components = {"predictive_loss": float(predictive.detach()),
                                "gate_aux_loss": float(auxiliary.detach())}
        return predictive + self.gate_aux_weight * auxiliary if self.gate_aux_weight else predictive

    def update(self, batch: SequenceBatch, mode: Mode) -> dict[str, float]:
        metrics = super().update(batch, mode)
        return {**metrics, **self.loss_components}


def _load_gated_checkpoint(path: Path):
    """Reconstruct only frozen exp153 v3 with explicitly unsupervised gate tags."""
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ValueError(f"could not safely load gated checkpoint: {error}") from error
    if not isinstance(payload, Mapping) or payload.get("format_version") != 3:
        raise ValueError("gated baseline requires format_version 3")
    if payload.get("latent_parameterization") != "gated_residual_zero_init":
        raise ValueError("checkpoint requires gated_residual_zero_init parameterization")
    if payload.get("event_supervision") is not False or payload.get("event_balanced") is not False:
        raise ValueError("baseline requires event_supervision=false and event_balanced=false")
    head = payload.get("git_head")
    if not isinstance(head, str) or not head:
        raise ValueError("checkpoint git_head must be a non-empty string")
    config = checkpoint_io._required_mapping(payload, "config", "config")
    modules = checkpoint_io._required_mapping(payload, "modules", "modules")
    meta = checkpoint_io._required_mapping(modules, "model", "modules.model")
    probe_meta = checkpoint_io._required_mapping(modules, "probe", "modules.probe")
    if meta.get("class") != GATED_CLASS or meta.get("gate") != gated.GATE_DEFINITION:
        raise ValueError("checkpoint must identify the exp153 gated class and gate definition")
    if meta.get("schemas") != {"grid-v1": [5, 1]}:
        raise ValueError("checkpoint requires grid-v1 with 5 actions and 1 sensor")
    dimensions = {}
    for field in ("z_dim", "h_dim", "ensemble_size"):
        dimensions[field] = checkpoint_io._positive_int(meta.get(field), field)
        if config.get(field) != dimensions[field]:
            raise ValueError(f"checkpoint config.{field} disagrees with module metadata")
    flags = {}
    for field in ("normalize_sensor_condition", "predict_sensor_delta"):
        flags[field] = meta.get(field)
        if not isinstance(flags[field], bool) or config.get(field) is not flags[field]:
            raise ValueError(f"checkpoint {field} metadata must agree and be boolean")
    if probe_meta.get("z_dim") != dimensions["z_dim"]:
        raise ValueError("checkpoint probe/model z_dim disagree")
    width = checkpoint_io._positive_int(probe_meta.get("width"), "probe.width")
    device_name = config.get("device")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("checkpoint device must be a non-empty string")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("checkpoint requests CUDA but CUDA is unavailable")
    model = gated.ChangeGatedResidualWorldModel(
        CoreEncoder(dimensions["z_dim"]), {"grid-v1": (5, 1)},
        dimensions["h_dim"], dimensions["ensemble_size"], **flags,
    )
    ordered = temporal.TemporalProbe(dimensions["z_dim"], width=width)
    for name, module in (("model_state_dict", model), ("ordered_probe_state_dict", ordered)):
        state = payload.get(name)
        checkpoint_io._validate_state_dict(name, state, module.state_dict())
        module.load_state_dict(state, strict=True)
        module.to(device).eval().requires_grad_(False)
    checkpoint_io._validate_state_dict(
        "shuffled_probe_state_dict", payload.get("shuffled_probe_state_dict"), ordered.state_dict()
    )
    return model, ordered, head, {
        "device": str(device), **dimensions, **flags, "probe_width": width,
        "latent_parameterization": payload["latent_parameterization"],
        "event_supervision": False, "event_balanced": False,
        "config": dict(config), "budgets": dict(payload.get("budgets", {})),
        "load_policy": "torch.load(weights_only=True, map_location='cpu')",
    }


def _checkpoint_payload(model, ordered, shuffled, config, manifest) -> dict[str, Any]:
    payload = gated._checkpoint_payload(model, ordered, shuffled, config, manifest)
    payload.update(format_version=4, event_supervision=True, gate_auxiliary=manifest["auxiliary"])
    payload["modules"]["model"]["class"] = AUXILIARY_CLASS
    return payload


def _one_step_gate(baseline: Mapping, candidate: Mapping, exact_protocol: bool) -> bool:
    ratio = candidate["medians"]["free_forward_prediction_persistence_ratio"]
    return bool(
        exact_protocol and candidate["contact_failure_layouts"] == 0
        and candidate["contact_failure_layouts"] <= baseline["contact_failure_layouts"]
        and candidate["blocked_noop_failure_layouts"] == 0
        and candidate["blocked_noop_failure_layouts"] < baseline["blocked_noop_failure_layouts"]
        and ratio is not None and math.isfinite(ratio) and ratio < 1.0
    )


def _gate_diagnostic(statistics: Mapping) -> dict[str, Any]:
    comparisons = []
    for split in one_step.SPLITS:
        rows = [row for row in statistics["by_action_context"] if row["split"] == split]
        def mean(action: int, changed: bool, step: int | None = None) -> float:
            selected = [row for row in rows if row["action"] == action
                        and row["rgb_changed"] == changed and (step is None or row["step"] == step)]
            return sum(row["mean"] * row["member_values"] for row in selected) / sum(
                row["member_values"] for row in selected)
        forward = mean(2, True)
        margins = {"forward_changed_minus_blocked_step0": forward - mean(2, False, 0),
                   "forward_changed_minus_blocked_step2": forward - mean(2, False, 2),
                   "interact_changed_minus_no_change": mean(3, True) - mean(3, False)}
        comparisons.append({"split": split, "margins": margins,
                            "threshold_met": all(value >= 0.15 for value in margins.values())})
    return {"threshold": 0.15, "supporting_only": True, "by_split": comparisons,
            "threshold_met": all(row["threshold_met"] for row in comparisons)}


def build_parser() -> argparse.ArgumentParser:
    parser = residual.build_parser()
    parser.description = __doc__
    parser.add_argument("--gate-aux-weight", type=float, choices=(1.0,), default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cutoff = round(0.75 * args.episodes_per_layout)
    if args.probe_episodes_per_layout > cutoff:
        parser.error("probe fit subset exceeds the source fit cutoff")
    if args.probe_validation_per_layout > args.episodes_per_layout - cutoff:
        parser.error("probe validation subset exceeds held-out source episodes")
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    manifest = {
        "argv": list(sys.orig_argv) if argv is None else [sys.executable, str(Path(__file__)), *argv],
        "cwd": str(Path.cwd()), "analysis_git_head": core._git_commit(),
        "baseline_checkpoint_git_head": None, "baseline_checkpoint": str(args.baseline_checkpoint),
        "budgets": core._jsonable(vars(args)), "fixed_protocol": PROTOCOL,
        "fixed_config": FIXED_CONFIG, "fixed_corpus": FIXED_CORPUS,
        "gate_definition": gated.GATE_DEFINITION, "preregistered_gates": PREREGISTERED_GATES,
        "status": "running", "exit_code": None, "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("initialize", 0, 2, operation="safe_gated_baseline_load")
            baseline, baseline_probe, baseline_head, metadata = _load_gated_checkpoint(args.baseline_checkpoint)
            del baseline_probe
            manifest.update(baseline_checkpoint_git_head=baseline_head, baseline_checkpoint_metadata=metadata)
            core._write_json(args.out / "manifest.json", manifest)
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config), seed=args.seed, z_dim=args.z_dim, h_dim=args.h_dim,
                burn_in=0, replay_capacity=len(temporal.SOURCE_LAYOUTS) * args.episodes_per_layout,
                termination_weight=0.0, salient_fraction=0.0,
            )
            device = torch.device(config.device)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(config.seed)
                model = AuxiliaryChangeGatedWorldModel(
                    CoreEncoder(config.z_dim), {"grid-v1": (5, 1)}, config.h_dim, config.ensemble_size,
                    normalize_sensor_condition=config.normalize_sensor_condition,
                    predict_sensor_delta=config.predict_sensor_delta,
                ).to(device)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 2, 2, device=str(device))
            corpus, fit, validation = residual._collect_corpus(args, replay, deadline, journal)
            journal.update("class_counts", 0, 1)
            coverage = _audit_counts({"all": replay._episodes()})
            counts = {action: {key: row[key] for key in ("total", "rgb_changed", "rgb_no_change")}
                      for action, row in coverage["actions"].items()}
            fixed_counts = counts == FIXED_CORPUS["action_counts"]
            if corpus["default_corpus_verified"] and not fixed_counts:
                raise AssertionError(f"exp149 RGB/action corpus counts mismatch: {counts}")
            weights = action_class_weights(counts)
            auxiliary = {
                "gate_aux_weight": args.gate_aux_weight,
                "target": "any(batch.rgb[:, t+1] != batch.rgb[:, t]) on valid supervised transitions",
                "loss": "mean_over_valid_transitions_and_members(weight[action,label] * BCEWithLogits)",
                "class_weight_formula": "N_action / (K_present_action * N_action_class); absent class = 0",
                "class_order": ["rgb_no_change", "rgb_changed"],
                "action_counts": counts, "class_weights": weights.tolist(),
                "count_source": "exact collected replay once before training; no sampler RNG consumed",
                "input": "same burn-in and autoregressive state as CoreTrainer; z plus existing action embedding",
                "implementation_cost": "one extra prefix encode and short autoregressive pass per update",
                "future_rgb_gate_input": False, "task_success_supervision": False,
            }
            trainer = AuxiliaryGateTrainer(model, config, weights, args.gate_aux_weight)
            matching = {
                "default_budgets": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "baseline_budgets": all(metadata["budgets"].get(key) == getattr(args, key) for key in gated.PROTOCOL),
                "baseline_config": metadata["config"] == core._jsonable(asdict(config)),
                "fixed_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_counts": corpus["default_corpus_verified"] and fixed_counts,
                "baseline_unsupervised_gated": metadata["event_supervision"] is False,
            }
            exact = all(matching.values())
            manifest.update(protocol_match=matching, auxiliary=auxiliary)
            corpus["action_counts"] = counts
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("class_counts", 1, 1, class_weights=weights.tolist(), exact_protocol=exact)
            journal.update("dynamics", 0, args.dynamics_updates)
            dynamics_losses, schema_counts = [], {}
            loss_trace = core.TraceWriter(args.out / "dynamics_losses.jsonl")
            try:
                for completed in range(0, args.dynamics_updates, args.dynamics_log_every):
                    chunk = min(args.dynamics_log_every, args.dynamics_updates - completed)
                    losses, counts_by_schema = core._train_updates(
                        model, trainer, replay, config, chunk, Mode.ADAPT, deadline, schema="grid-v1",
                    )
                    for index, metrics in enumerate(losses, completed + 1):
                        loss_trace.write({"update": index, **metrics})
                    dynamics_losses.extend(losses)
                    for schema, count in counts_by_schema.items():
                        schema_counts[schema] = schema_counts.get(schema, 0) + count
                    journal.update("dynamics", completed + chunk, args.dynamics_updates, **losses[-1])
            finally:
                loss_trace.close()
            model.eval().requires_grad_(False)
            ordered, shuffled, probe_metrics = residual._fit_probes(
                model, config, args, fit, validation, deadline, journal,
            )
            journal.update("auxiliary_checkpoint", 0, 1)
            checkpoint_path = args.out / "auxiliary_checkpoint.pt"
            torch.save(_checkpoint_payload(model, ordered, shuffled, config, manifest), checkpoint_path)
            journal.update("auxiliary_checkpoint", 1, 1)
            diagnostics = {}
            for role, candidate in (("baseline", baseline), ("auxiliary", model)):
                journal.update(f"{role}_one_step", 0, 120)
                core._check_deadline(deadline, f"{role}_one_step")
                diagnostics[role] = one_step._diagnose(candidate, journal, args.out / f"{role}_one_step_rows.jsonl")
                core._check_deadline(deadline, f"{role}_one_step")
                journal.update(f"{role}_one_step", 120, 120)
            statistics = gated._gate_statistics(model, args.out / "auxiliary_one_step_rows.jsonl",
                                                args.out / "gate_rows.jsonl", deadline, journal)
            late_fork = temporal._late_fork_audit(
                model, ordered, config, deadline, journal, args.out / "auxiliary_late_fork_rows.jsonl",
            )
            total = len(temporal.TARGET_LAYOUTS) * args.eval_seeds * 4
            journal.update("evaluate", 0, total)
            trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
            try:
                evaluation = temporal._evaluate(
                    model, {"ordered": ordered, "shuffled": shuffled}, config, replay,
                    range(20000, 20000 + args.eval_seeds), args.eval_steps, deadline, trace, journal, [0], total,
                )
            finally:
                trace.close()
            one_step_gate = _one_step_gate(diagnostics["baseline"]["splits"]["source"],
                                           diagnostics["auxiliary"]["splits"]["source"], exact)
            successes = evaluation["ordered_h3"]["overall"]["successes"]
            source_gate = bool(
                args.eval_steps == 8 and args.eval_seeds == 6
                and all(corpus["natural_terminal_fit_episodes_by_layout"].values()) and successes >= 18
                and min(row["successes"] for row in evaluation["ordered_h3"]["by_layout"].values()) >= 3
                and all(successes >= evaluation[control]["overall"]["successes"] + 4
                        for control in ("ordered_h1", "shuffled_h3", "raw_h3"))
            )
            journal.update("artifacts", 0, 1)
            result = {
                "status": "completed", "claim": "single self-supervised gate objective development comparison",
                "exact_protocol": exact, "protocol_match": matching,
                "preregistered_gates": PREREGISTERED_GATES,
                "auxiliary_one_step_gate": one_step_gate, "source_compositional_gate": source_gate,
                "auxiliary_composition_gate": bool(one_step_gate and source_gate), "physics_transfer_gate": None,
                "corpus": corpus, "auxiliary": auxiliary,
                "dynamics": {"updates": args.dynamics_updates, "first": dynamics_losses[0],
                             "last": dynamics_losses[-1], "schema_counts": schema_counts,
                             "burn_in": config.burn_in, "train_horizon": config.train_horizon,
                             "termination_weight": config.termination_weight, "salient_fraction": config.salient_fraction},
                "probe": probe_metrics, "baseline_one_step": diagnostics["baseline"],
                "auxiliary_one_step": diagnostics["auxiliary"], "auxiliary_late_fork": late_fork,
                "gate_statistics": statistics, "gate_diagnostic": _gate_diagnostic(statistics),
                "replay_heldout_gate_metrics": {"performed": False,
                    "reason": "canonical gate diagnostic sufficient; dynamics trains on the entire matched replay"},
                "evaluation": evaluation,
                "controls": {
                    "latent_parameterization": "gated_residual_zero_init", "gate_definition": gated.GATE_DEFINITION,
                    "event_balanced_sampling": False, "event_supervision": True,
                    "self_supervised_rgb_change_auxiliary": True, "task_success_supervision": False,
                    "sampling": "unchanged SequenceReplay and core._train_updates from exp153/150",
                    "architecture_initialization_predictive_objective_unchanged": True,
                    "seed_and_chunked_updates_match_exp153": True,
                    "baseline_frozen_checkpoint": str(args.baseline_checkpoint), "baseline_checkpoint_git_head": baseline_head,
                    "baseline_training_budgets": metadata["budgets"], "baseline_training_config": metadata["config"],
                    "auxiliary_training_config": core._jsonable(asdict(config)), "source_only_training": True,
                    "push_distance": 1, "goal_push_distance": 1, "push2_not_run": True,
                    "termination_neutral_planning": True, "beam_width": 5,
                    "canonical_actions_excluded_from_fit_data": True,
                    "one_step_protocol": "unchanged exp148: 8 layouts x 3 steps x 5 actions",
                },
                "limitations": [
                    "one seed, the unchanged exp153 architecture and one Push-1 task family",
                    "RGB inequality is a visual-change proxy, not a semantic box or task-success label",
                    "extra self-supervised objective changes shared representations as well as gate heads",
                    "weights balance classes under full-corpus action frequencies, not every sampled batch/window position",
                    "the additional prefix encode/rollout increases training cost; no teacher forcing after burn-in",
                    "gate/delta scale is not identifiable; amplitude is not a calibrated change probability",
                    "a reduced-budget smoke cannot pass the preregistered one-step/composition gates",
                    "probe fit includes naturally terminal source episodes as in exp146/150/153",
                    "CUDA training is not guaranteed bitwise deterministic despite seeded replay sampling",
                    "baseline and candidate encode different latent spaces; compare persistence ratios and failures",
                    "no sampling or planner change, weight sweep, or Push-2 physics test",
                    "not AGI, JEPA, concept, representation-capacity or physics-transfer proof",
                ],
                "artifacts": {"checkpoint": checkpoint_path.name, "manifest": "manifest.json",
                              "progress": "progress.jsonl", "losses": "dynamics_losses.jsonl", "external_log": "run.log"},
            }
            core._write_json(args.out / "results.json", result)
            core._write_json(args.out / "manifest.json", {**manifest, "exit_code": 0, "exit_status": 0, "status": "completed"})
            journal.update("artifacts", 1, 1)
            return 0
        except BaseException as error:
            code = temporal._exit_code(error)
            core._write_json(args.out / "manifest.json", {
                **manifest, "exit_code": code, "exit_status": code, "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            })
            raise


if __name__ == "__main__":
    raise SystemExit(main())
