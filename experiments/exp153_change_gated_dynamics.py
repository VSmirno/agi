"""Matched exp150 uniform training with a current-state multiplicative change gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

import torch

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp148_source_target_one_step as one_step
from experiments import exp150_residual_dynamics as residual
from experiments.exp151_event_balanced_dynamics import FIXED_CONFIG, _load_residual_checkpoint
from snks.agent.core_world_model import CoreWorldModel, LatentState, Prediction
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


PROTOCOL = dict(residual.PROTOCOL)
GATE_DEFINITION = {
    "input": "concat(current_z, existing_action_embedding)",
    "head": "one Linear(z_dim + h_dim, 1) per ensemble member followed by sigmoid",
    "weight_init": 0.0, "bias_init": 0.0, "initial_probability": 0.5,
    "latent_delta_weight_init": 0.0, "latent_delta_bias_init": 0.0,
    "member_transition": "current_z + sigmoid(gate_i(input)) * delta_i(hidden)",
}


class ChangeGatedResidualWorldModel(CoreWorldModel):
    """Gate existing delta heads directly from current z and the action embedding."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_heads = torch.nn.ModuleList(
            torch.nn.Linear(self.encoder.z_dim + self.h_dim, 1) for _ in range(self.heads)
        )
        for head in (*self.latent_heads, *self.gate_heads):
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def change_gates(self, state: LatentState, actions: torch.Tensor) -> torch.Tensor:
        condition = torch.cat((state.z, self.action_embeddings[state.schema](actions)), dim=-1)
        return torch.stack([head(condition).sigmoid() for head in self.gate_heads])

    def step(self, state: LatentState, actions: torch.Tensor) -> Prediction:
        prediction = super().step(state, actions)
        member_delta = self.change_gates(state, actions) * prediction.member_z
        member_z = state.z.unsqueeze(0) + member_delta
        return replace(
            prediction, member_z=member_z,
            # Algebraically the member mean, preserving exact identity when delta=0.
            next_state=replace(prediction.next_state, z=state.z + member_delta.mean(0)),
            uncertainty=member_z.var(0, unbiased=False).mean(-1),
        )


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    return parser


def _checkpoint_payload(model, ordered, shuffled, config, manifest):
    return {
        # Both old absolute v1 and ungated residual v2 loaders must refuse this file.
        "format_version": 3, "latent_parameterization": "gated_residual_zero_init",
        "event_balanced": False, "event_supervision": False,
        "analysis_git_head": manifest["analysis_git_head"],
        "git_head": manifest["analysis_git_head"],
        "baseline_checkpoint_git_head": manifest["baseline_checkpoint_git_head"],
        "budgets": manifest["budgets"], "config": core._jsonable(asdict(config)),
        "modules": {
            "model": {
                "class": "experiments.exp153_change_gated_dynamics.ChangeGatedResidualWorldModel",
                "schemas": core._jsonable(model.schemas),
                "z_dim": config.z_dim, "h_dim": config.h_dim,
                "ensemble_size": config.ensemble_size,
                "normalize_sensor_condition": config.normalize_sensor_condition,
                "predict_sensor_delta": config.predict_sensor_delta,
                "gate": GATE_DEFINITION,
            },
            "probe": {"z_dim": config.z_dim, "width": ordered.network[0].out_features},
        },
        "model_state_dict": model.state_dict(),
        "ordered_probe_state_dict": ordered.state_dict(),
        "shuffled_probe_state_dict": shuffled.state_dict(),
    }


@torch.inference_mode()
def _gate_statistics(model, diagnostic_path, output_path, deadline, journal):
    """Read gates at exp148's exact real before states; labels only group the report."""
    rows = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    specs = one_step._layout_specs()
    cached = {}
    grouped = defaultdict(list)
    writer = core.TraceWriter(output_path)
    journal.update("gate_statistics", 0, len(rows))
    try:
        for completed, row in enumerate(rows, 1):
            core._check_deadline(deadline, f"gate_statistics/{completed}")
            key = row["split"], row["layout"], tuple(row["real_history"])
            if key not in cached:
                layout = specs[row["split"]][row["layout"]][0]
                before, _after, _diagnostic = one_step._fresh_real_fork(
                    layout, key[2], row["action"], one_step.SEED
                )
                # The gate has no recurrent/body input; real before z is sufficient.
                cached[key] = model.initial(before)
            state = cached[key]
            values = model.change_gates(
                state, torch.tensor([row["action"]], device=state.z.device)
            ).flatten().tolist()
            record = {name: row[name] for name in (
                "split", "layout", "step", "action", "action_name", "rgb_changed", "real_history"
            )}
            record.update(by_member=values, mean=sum(values) / len(values),
                          min=min(values), max=max(values))
            writer.write(record)
            grouped[(row["split"], row["action"], row["step"], row["rgb_changed"])].extend(values)
            journal.update("gate_statistics", completed, len(rows),
                           split=row["split"], action=row["action"], step=row["step"])
    finally:
        writer.close()
    summaries = [
        {"split": split, "action": action, "step": step, "rgb_changed": changed,
         "member_values": len(values), "mean": sum(values) / len(values),
         "min": min(values), "max": max(values)}
        for (split, action, step, changed), values in sorted(grouped.items())
    ]
    return {
        "diagnostic_only": True, "rows": len(rows),
        "input": "exp148 real before-state z and existing action embedding",
        "grouping": ["split", "action", "canonical_continuation_step", "target_rgb_changed"],
        "by_action_context": summaries, "artifacts": {"rows": output_path.name},
        "interpretation_limit": "gate is a multiplicative amplitude, not a calibrated event probability",
    }


def main(argv=None) -> int:
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
        "argv": list(sys.orig_argv), "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(), "baseline_checkpoint_git_head": None,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "budgets": core._jsonable(vars(args)), "fixed_protocol": PROTOCOL,
        "fixed_residual_config": FIXED_CONFIG, "gate_definition": GATE_DEFINITION,
        "status": "running", "exit_code": None, "exit_status": None,
    }
    core._write_json(args.out / "manifest.json", manifest)
    with temporal.ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("initialize", 0, 2, operation="safe_residual_baseline_load")
            baseline, baseline_probe, baseline_head, metadata = _load_residual_checkpoint(
                args.baseline_checkpoint
            )
            del baseline_probe
            manifest.update(baseline_checkpoint_git_head=baseline_head,
                            baseline_checkpoint_metadata=metadata)
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
                model = ChangeGatedResidualWorldModel(
                    CoreEncoder(config.z_dim), {"grid-v1": (5, 1)}, config.h_dim,
                    config.ensemble_size, normalize_sensor_condition=config.normalize_sensor_condition,
                    predict_sensor_delta=config.predict_sensor_delta,
                ).to(device)
            trainer = CoreTrainer(model, config)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 2, 2, device=str(device))
            corpus, fit, validation = residual._collect_corpus(args, replay, deadline, journal)
            matching = {
                "default_budgets": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "baseline_budgets": all(metadata["budgets"].get(key) == getattr(args, key)
                                        for key in PROTOCOL),
                "baseline_config": metadata["config"] == core._jsonable(asdict(config)),
                "fixed_residual_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_counts": corpus["default_corpus_verified"],
                "baseline_uniform_residual": metadata["event_balanced"] is False,
            }
            exact_protocol = all(matching.values())
            manifest["protocol_match"] = matching
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("dynamics", 0, args.dynamics_updates, exact_protocol=exact_protocol)
            dynamics_losses, schema_counts = [], {}
            for completed in range(0, args.dynamics_updates, args.dynamics_log_every):
                chunk = min(args.dynamics_log_every, args.dynamics_updates - completed)
                losses, counts = core._train_updates(
                    model, trainer, replay, config, chunk, Mode.ADAPT,
                    deadline, schema="grid-v1",
                )
                dynamics_losses.extend(losses)
                for schema, count in counts.items():
                    schema_counts[schema] = schema_counts.get(schema, 0) + count
                journal.update("dynamics", completed + chunk, args.dynamics_updates,
                               loss=losses[-1]["loss"])
            model.eval().requires_grad_(False)
            ordered, shuffled, probe_metrics = residual._fit_probes(
                model, config, args, fit, validation, deadline, journal
            )
            journal.update("gated_checkpoint", 0, 1)
            checkpoint_path = args.out / "gated_checkpoint.pt"
            torch.save(_checkpoint_payload(model, ordered, shuffled, config, manifest), checkpoint_path)
            journal.update("gated_checkpoint", 1, 1)
            diagnostics = {}
            for role, candidate in (("baseline", baseline), ("gated", model)):
                journal.update(f"{role}_one_step", 0, 120)
                core._check_deadline(deadline, f"{role}_one_step")
                diagnostics[role] = one_step._diagnose(
                    candidate, journal, args.out / f"{role}_one_step_rows.jsonl"
                )
                core._check_deadline(deadline, f"{role}_one_step")
                journal.update(f"{role}_one_step", 120, 120)
            gates = _gate_statistics(model, args.out / "gated_one_step_rows.jsonl",
                                     args.out / "gate_rows.jsonl", deadline, journal)
            late_fork = temporal._late_fork_audit(
                model, ordered, config, deadline, journal, args.out / "gated_late_fork_rows.jsonl"
            )
            total = len(temporal.TARGET_LAYOUTS) * args.eval_seeds * 4
            journal.update("evaluate", 0, total)
            trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
            try:
                evaluation = temporal._evaluate(
                    model, {"ordered": ordered, "shuffled": shuffled}, config, replay,
                    range(20000, 20000 + args.eval_seeds), args.eval_steps,
                    deadline, trace, journal, [0], total,
                )
            finally:
                trace.close()
            one_step_gate = residual._one_step_gate(
                diagnostics["baseline"]["splits"]["source"],
                diagnostics["gated"]["splits"]["source"], exact_protocol,
            )
            successes = evaluation["ordered_h3"]["overall"]["successes"]
            source_gate = bool(
                args.eval_steps == 8 and args.eval_seeds == 6
                and all(corpus["natural_terminal_fit_episodes_by_layout"].values())
                and successes >= 18
                and min(row["successes"] for row in evaluation["ordered_h3"]["by_layout"].values()) >= 3
                and all(successes >= evaluation[control]["overall"]["successes"] + 4
                        for control in ("ordered_h1", "shuffled_h3", "raw_h3"))
            )
            journal.update("artifacts", 0, 1)
            result = {
                "status": "completed", "claim": "single change-gated parameterization development comparison",
                "exact_protocol": exact_protocol, "protocol_match": matching,
                "gated_one_step_gate": one_step_gate, "source_compositional_gate": source_gate,
                "gated_composition_gate": bool(one_step_gate and source_gate),
                "physics_transfer_gate": None, "corpus": corpus,
                "dynamics": {
                    "updates": args.dynamics_updates, "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"], "schema_counts": schema_counts,
                    "burn_in": config.burn_in, "train_horizon": config.train_horizon,
                    "termination_weight": config.termination_weight,
                    "salient_fraction": config.salient_fraction,
                },
                "probe": probe_metrics, "baseline_one_step": diagnostics["baseline"],
                "gated_one_step": diagnostics["gated"], "gated_late_fork": late_fork,
                "gate_statistics": gates, "evaluation": evaluation,
                "controls": {
                    "latent_parameterization": "gated_residual_zero_init", "gate_definition": GATE_DEFINITION,
                    "event_balanced_sampling": False, "event_supervision": False, "auxiliary_loss": False,
                    "sampling": "unchanged SequenceReplay and core._train_updates from exp150",
                    "encoder_gru_actions_sensor_heads_objectives_unchanged": True,
                    "seed_and_chunked_updates_match_exp150": True,
                    "baseline_frozen_checkpoint": str(args.baseline_checkpoint),
                    "baseline_checkpoint_git_head": baseline_head,
                    "baseline_training_budgets": metadata["budgets"],
                    "baseline_training_config": metadata["config"],
                    "gated_training_config": core._jsonable(asdict(config)),
                    "source_only_training": True, "push_distance": 1, "goal_push_distance": 1,
                    "push2_not_run": True, "termination_neutral_planning": True, "beam_width": 5,
                    "canonical_actions_excluded_from_fit_data": True,
                    "one_step_protocol": "unchanged exp148: 8 layouts x 3 steps x 5 actions",
                },
                "limitations": [
                    "one architecture, one declared training seed, one Push-1 task family",
                    "a reduced-budget smoke cannot pass the preregistered change-gated gates",
                    "no event supervision: RGB-change labels only group the post-training diagnostic",
                    "gate/delta scale is not identifiable; gate values are not calibrated change probabilities",
                    "baseline and candidate encode their own latent spaces; compare persistence ratios and failures",
                    "CUDA training is not guaranteed bitwise deterministic despite seeded replay sampling",
                    "temporal proximity depends on collection policy; no shortest-path guarantee",
                    "probe fit includes naturally terminal source episodes as in exp146/exp150",
                    "no event balancing, auxiliary loss, planner tuning or Push-2 physics test",
                    "not AGI, JEPA, concept, representation-capacity or physics-transfer proof",
                ],
                "artifacts": {"checkpoint": checkpoint_path.name, "manifest": "manifest.json",
                              "progress": "progress.jsonl", "external_log": "run.log"},
            }
            core._write_json(args.out / "results.json", result)
            core._write_json(args.out / "manifest.json", {
                **manifest, "exit_code": 0, "exit_status": 0, "status": "completed",
            })
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
