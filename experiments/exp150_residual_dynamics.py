"""Matched exp146 training with a zero-initialized residual latent transition."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
import time

import torch

from experiments import exp146_temporal_mpc_physics as temporal
from experiments.exp147_rollout_localization import _load_checkpoint
from experiments.exp148_source_target_one_step import _diagnose
from experiments.exp149_replay_coverage import _validate_terminal_counts
from snks.agent.core_world_model import CoreWorldModel, LatentState, Prediction
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


# These are exp146's declared scientific budgets, not fitted thresholds.
PROTOCOL = {
    "episodes_per_layout": 512,
    "collection_steps": 64,
    "dynamics_updates": 2000,
    "dynamics_log_every": 100,
    "probe_updates": 400,
    "probe_batch_size": 256,
    "probe_episodes_per_layout": 64,
    "probe_validation_per_layout": 16,
    "max_horizon": 3,
    "eval_seeds": 6,
    "eval_steps": 8,
    "z_dim": 256,
    "h_dim": 128,
    "seed": 0,
}


class ResidualLatentWorldModel(CoreWorldModel):
    """Keep CoreWorldModel plumbing; interpret each latent head as a delta."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for head in self.latent_heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)

    def step(self, state: LatentState, actions: torch.Tensor) -> Prediction:
        prediction = super().step(state, actions)
        member_z = state.z.unsqueeze(0) + prediction.member_z
        return replace(
            prediction,
            member_z=member_z,
            next_state=replace(prediction.next_state, z=member_z.mean(0)),
            uncertainty=member_z.var(0, unbiased=False).mean(-1),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "baseline-checkpoint", "out"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    for name, default in PROTOCOL.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", default=default,
            type=int if name == "seed" else temporal._positive,
        )
    parser.add_argument("--collection-log-every", type=temporal._positive, default=32)
    parser.add_argument("--max-seconds", type=temporal._positive, default=3600)
    parser.add_argument("--progress-interval", type=temporal._progress_interval, default=30)
    return parser


def _one_step_gate(baseline, residual, exact_protocol: bool) -> bool:
    free_ratio = residual["medians"]["free_forward_prediction_persistence_ratio"]
    return bool(
        exact_protocol
        and residual["contact_failure_layouts"] == 0
        and residual["blocked_noop_failure_layouts"] == 0
        and free_ratio is not None and math.isfinite(free_ratio) and free_ratio < 1.0
        and residual["contact_failure_layouts"] < baseline["contact_failure_layouts"]
        and residual["blocked_noop_failure_layouts"] < baseline["blocked_noop_failure_layouts"]
    )


def _collect_corpus(args, replay, deadline, journal):
    episodes = {name: [] for name in temporal.SOURCE_LAYOUTS}
    total = len(episodes) * args.episodes_per_layout
    completed = 0
    journal.update("collect", 0, total)
    for offset in range(args.episodes_per_layout):
        for index, (name, (layout, _actions)) in enumerate(temporal.SOURCE_LAYOUTS.items()):
            core._check_deadline(deadline, f"collect/{name}/{offset}")
            seed = 10000 + index * 100000 + offset
            episode = temporal._collect(name, layout, seed, args.collection_steps)
            episodes[name].append(episode)
            replay.append(episode, Mode.ADAPT)
            completed += 1
            if completed % args.collection_log_every == 0 or completed == total:
                journal.update("collect", completed, total, layout=name, offset=offset)
    fit, validation, cutoff = temporal._probe_split(episodes, args)
    terminals = {
        name: sum(bool(ep.transitions and ep.transitions[-1].terminated) for ep in items)
        for name, items in episodes.items()
    }
    fit_terminals = {
        name: sum(bool(ep.transitions and ep.transitions[-1].terminated) for ep in items)
        for name, items in fit.items()
    }
    cutoff_terminals = {
        name: sum(bool(ep.transitions and ep.transitions[-1].terminated)
                  for ep in items[:cutoff])
        for name, items in episodes.items()
    }
    transitions = sum(len(ep.transitions) for items in episodes.values() for ep in items)
    fixed = args.episodes_per_layout == 512 and args.collection_steps == 64
    if fixed and (transitions != 130676 or not _validate_terminal_counts(
        terminals, cutoff_terminals, episodes_per_layout=args.episodes_per_layout
    )):
        raise AssertionError(
            f"scientific protocol mismatch: {transitions=}, {terminals=}, {cutoff_terminals=}"
        )
    corpus = {
        "episodes": completed,
        "transitions": transitions,
        "default_corpus_verified": fixed,
        "source_push_distance": 1,
        "collection_interleaved_by_offset": True,
        "collection_seed_scheme": "10000 + layout_index * 100000 + offset",
        "source_layouts_insertion_order": list(episodes),
        "by_layout": {
            name: {"episodes": len(items), "natural_terminals": terminals[name],
                   "transitions": sum(len(ep.transitions) for ep in items)}
            for name, items in episodes.items()
        },
        "episode_uids_by_layout": {
            name: [ep.uid for ep in items] for name, items in episodes.items()
        },
        "fit_cutoff_per_layout": cutoff,
        "natural_terminals_by_layout": terminals,
        "natural_terminals_fit_cutoff_by_layout": cutoff_terminals,
        "natural_terminal_fit_episodes_by_layout": fit_terminals,
        "fit_episodes_by_layout": {name: len(items) for name, items in fit.items()},
        "validation_episodes_by_layout": {
            name: len(items) for name, items in validation.items()
        },
    }
    return corpus, fit, validation


def _fit_probes(model, config, args, fit, validation, deadline, journal):
    device = torch.device(config.device)
    pairs = {}
    journal.update("probe_encode", 0, 2, split="fit")
    encoded = {}
    for completed, (split, by_layout) in enumerate(
        (("fit", fit), ("validation", validation)), start=1
    ):
        episodes = [ep for name in temporal.SOURCE_LAYOUTS for ep in by_layout[name]]
        encoded[split] = temporal._encode_episodes(model, episodes, device)
        journal.update("probe_encode", completed, 2, split=split)
        core._check_deadline(deadline, f"probe_encode/{split}")
    journal.update("probe_pairs", 0, 2)
    for completed, split in enumerate(("fit", "validation"), start=1):
        pairs[split] = temporal._pairs(encoded[split], args.max_horizon)
        journal.update("probe_pairs", completed, 2, split=split)
        core._check_deadline(deadline, f"probe_pairs/{split}")
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(config.seed + 146)
        ordered = temporal.TemporalProbe(config.z_dim).to(device)
        shuffled = temporal.TemporalProbe(config.z_dim).to(device)
        shuffled.load_state_dict(ordered.state_dict())
    journal.update("probe_fit", 0, args.probe_updates)
    losses = temporal._fit_pair(
        ordered, shuffled, pairs["fit"], args.probe_updates,
        args.probe_batch_size, config.seed + 146,
    )
    core._check_deadline(deadline, "probe_fit")
    journal.update("probe_fit", args.probe_updates, args.probe_updates)
    journal.update("probe_metrics", 0, 2)
    metrics = {"losses": losses}
    for completed, (role, split) in enumerate((("train", "fit"), ("validation", "validation")), 1):
        metrics[role] = {
            "ordered": temporal._probe_metrics(ordered, pairs[split]),
            "shuffled_endpoint": temporal._probe_metrics(shuffled, pairs[split]),
        }
        journal.update("probe_metrics", completed, 2, split=split)
    ordered.requires_grad_(False)
    shuffled.requires_grad_(False)
    return ordered, shuffled, metrics


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.probe_episodes_per_layout > round(0.75 * args.episodes_per_layout):
        parser.error("probe fit subset exceeds the source fit cutoff")
    if args.probe_validation_per_layout > args.episodes_per_layout - round(
        0.75 * args.episodes_per_layout
    ):
        parser.error("probe validation subset exceeds held-out source episodes")
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    manifest = {
        "argv": list(sys.orig_argv), "cwd": str(Path.cwd()),
        "analysis_git_head": core._git_commit(),
        "baseline_checkpoint_git_head": None,
        "baseline_checkpoint": str(args.baseline_checkpoint),
        "budgets": core._jsonable(vars(args)), "fixed_protocol": PROTOCOL,
    }
    with temporal.ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            journal.update("initialize", 0, 2, operation="safe_baseline_load")
            baseline, baseline_probe, baseline_head, metadata = _load_checkpoint(args.baseline_checkpoint)
            del baseline_probe
            manifest["baseline_checkpoint_git_head"] = baseline_head
            manifest["baseline_checkpoint_metadata"] = metadata
            # The established loader validates reconstruction; retain saved training
            # metadata as evidence that this is a matched causal comparison.
            baseline_payload = torch.load(args.baseline_checkpoint, weights_only=True, map_location="cpu")
            baseline_config = baseline_payload["config"]
            baseline_budgets = baseline_payload.get("budgets", {})
            del baseline_payload
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config), seed=args.seed,
                z_dim=args.z_dim, h_dim=args.h_dim, burn_in=0,
                replay_capacity=len(temporal.SOURCE_LAYOUTS) * args.episodes_per_layout,
                termination_weight=0.0, salient_fraction=0.0,
            )
            device = torch.device(config.device)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(config.seed)
                model = ResidualLatentWorldModel(
                    CoreEncoder(config.z_dim), {"grid-v1": (5, 1)},
                    config.h_dim, config.ensemble_size,
                    normalize_sensor_condition=config.normalize_sensor_condition,
                    predict_sensor_delta=config.predict_sensor_delta,
                ).to(device)
            trainer = CoreTrainer(model, config)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            matching = {
                "default_budgets": all(getattr(args, name) == value for name, value in PROTOCOL.items()),
                "baseline_budgets": all(baseline_budgets.get(name) == getattr(args, name) for name in PROTOCOL),
                "baseline_config": baseline_config == core._jsonable(asdict(config)),
            }
            exact_protocol = all(matching.values())
            manifest["protocol_match"] = matching
            journal.update("initialize", 2, 2, device=str(device), exact_protocol=exact_protocol)
            corpus, fit, validation = _collect_corpus(args, replay, deadline, journal)

            journal.update("dynamics", 0, args.dynamics_updates)
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
                journal.update("dynamics", completed + chunk, args.dynamics_updates, loss=losses[-1]["loss"])
            model.eval().requires_grad_(False)
            ordered, shuffled, probe_metrics = _fit_probes(
                model, config, args, fit, validation, deadline, journal
            )

            journal.update("residual_checkpoint", 0, 1)
            checkpoint_path = args.out / "residual_checkpoint.pt"
            torch.save({
                # Version 2 prevents exp147's absolute-only loader from silently
                # interpreting these identical-shaped tensors as absolute heads.
                "format_version": 2,
                "latent_parameterization": "residual_zero_init",
                "analysis_git_head": manifest["analysis_git_head"],
                "git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "budgets": manifest["budgets"], "config": core._jsonable(asdict(config)),
                "modules": {
                    "model": {
                        "class": "experiments.exp150_residual_dynamics.ResidualLatentWorldModel",
                        "schemas": core._jsonable(model.schemas),
                        "z_dim": config.z_dim, "h_dim": config.h_dim,
                        "ensemble_size": config.ensemble_size,
                        "normalize_sensor_condition": config.normalize_sensor_condition,
                        "predict_sensor_delta": config.predict_sensor_delta,
                    },
                    "probe": {"z_dim": config.z_dim, "width": ordered.network[0].out_features},
                },
                "model_state_dict": model.state_dict(),
                "ordered_probe_state_dict": ordered.state_dict(),
                "shuffled_probe_state_dict": shuffled.state_dict(),
            }, checkpoint_path)
            journal.update("residual_checkpoint", 1, 1)
            one_step = {}
            for role, candidate in (("baseline", baseline), ("residual", model)):
                journal.update(f"{role}_one_step", 0, 120)
                core._check_deadline(deadline, f"{role}_one_step")
                one_step[role] = _diagnose(candidate, journal, args.out / f"{role}_one_step_rows.jsonl")
                core._check_deadline(deadline, f"{role}_one_step")
                journal.update(f"{role}_one_step", 120, 120)
            late_fork = temporal._late_fork_audit(
                model, ordered, config, deadline, journal, args.out / "residual_late_fork_rows.jsonl"
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
            one_step_gate = _one_step_gate(
                one_step["baseline"]["splits"]["source"],
                one_step["residual"]["splits"]["source"], exact_protocol,
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
                "status": "completed",
                "claim": "single-parameterization development comparison; no concept proof",
                "exact_protocol": exact_protocol, "protocol_match": matching,
                "residual_one_step_gate": one_step_gate,
                "source_compositional_gate": source_gate,
                "residual_composition_gate": bool(one_step_gate and source_gate),
                "physics_transfer_gate": None,
                "corpus": corpus,
                "dynamics": {
                    "updates": args.dynamics_updates,
                    "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"],
                    "schema_counts": schema_counts, "burn_in": config.burn_in,
                    "termination_weight": config.termination_weight,
                    "salient_fraction": config.salient_fraction,
                },
                "probe": probe_metrics,
                "baseline_one_step": one_step["baseline"],
                "residual_one_step": one_step["residual"],
                "residual_late_fork": late_fork, "evaluation": evaluation,
                "controls": {
                    "latent_parameterization": "residual_zero_init",
                    "only_latent_heads_zero_initialized": True,
                    "encoder_gru_actions_sensor_heads_objectives_unchanged": True,
                    "seed_and_chunked_updates_match_exp146": True,
                    "baseline_frozen_checkpoint": str(args.baseline_checkpoint),
                    "baseline_checkpoint_git_head": baseline_head,
                    "baseline_training_budgets": baseline_budgets,
                    "baseline_training_config": baseline_config,
                    "residual_training_config": core._jsonable(asdict(config)),
                    "event_balanced_sampling": False,
                    "source_only_training": True, "push_distance": 1,
                    "goal_push_distance": 1, "push2_not_run": True,
                    "termination_neutral_planning": True,
                    "beam_width": 5, "canonical_actions_excluded_from_fit_data": True,
                    "one_step_protocol": "unchanged exp148: 8 layouts x 3 steps x 5 actions",
                },
                "limitations": [
                    "one latent parameterization, one declared training seed and one Push-1 task family",
                    "a reduced-budget smoke cannot pass the predeclared residual gates",
                    "temporal proximity depends on the collection policy; no shortest-path guarantee",
                    "probe fit includes naturally terminal source episodes as in exp146",
                    "baseline and residual encode their own latent spaces; compare persistence ratios and failures",
                    "no event balancing, objective change, planner tuning or Push-2 physics test",
                    "not AGI, JEPA, general representation-capacity or physics-transfer proof",
                ],
                "artifacts": {"checkpoint": checkpoint_path.name},
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
