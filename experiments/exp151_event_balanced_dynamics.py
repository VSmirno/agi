"""Matched exp150 residual training with 50:50 event/ordinary window anchors."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

from experiments import exp146_temporal_mpc_physics as temporal
from experiments import exp147_rollout_localization as checkpoint_io
from experiments import exp150_residual_dynamics as residual
from experiments.exp148_source_target_one_step import _diagnose
from experiments.exp149_replay_coverage import _audit_counts
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config


PROTOCOL = {**residual.PROTOCOL, "event_fraction": 0.5}
RESIDUAL_CLASS = "experiments.exp150_residual_dynamics.ResidualLatentWorldModel"
# Recorded by exp149; these checks precede fitting and are not fitted thresholds.
FIXED_CORPUS = {
    "episodes": 2048, "transitions": 130676,
    "natural_terminals_by_layout": {
        "east_row2": 2, "west_row3": 7, "south_col4": 2, "north_col5": 7,
    },
    "natural_terminals_fit_cutoff_by_layout": {
        "east_row2": 2, "west_row3": 4, "south_col4": 2, "north_col5": 5,
    },
    "rgb_changing_interact_transitions": 1925,
    "episodes_with_rgb_changing_interact": 1281,
    "action_counts": {
        "0": {"total": 26346, "rgb_changed": 26346, "rgb_no_change": 0},
        "1": {"total": 26001, "rgb_changed": 26001, "rgb_no_change": 0},
        "2": {"total": 26122, "rgb_changed": 16764, "rgb_no_change": 9358},
        "3": {"total": 26125, "rgb_changed": 1925, "rgb_no_change": 24200},
        "4": {"total": 26082, "rgb_changed": 0, "rgb_no_change": 26082},
    },
}
FIXED_CONFIG = {
    "batch_size": 8, "beam_width": 4, "burn_in": 0, "device": "cuda",
    "ensemble_size": 3, "exploration_fraction": 0.2, "h_dim": 128,
    "learning_rate": 0.001, "max_model_calls": 128,
    "normalize_sensor_condition": False, "planner_horizon": 3,
    "predict_sensor_delta": False, "profile": "pilot", "recent_fraction": 0.5,
    "replay_capacity": 2048, "salient_fraction": 0.0, "seed": 0,
    "sensor_weight": 1.0, "sigreg_weight": 0.1, "termination_weight": 0.0,
    "train_horizon": 3, "z_dim": 256,
}


def _event(transition) -> bool:
    return transition.action == 3 and not np.array_equal(
        transition.before.rgb, transition.after.rgb
    )


class EventBalancedSampler:
    """Experiment-local window indices over the existing fixed replay contents.

    Match SequenceReplay._window's support: full windows when possible, otherwise
    start zero. The anchor is the FIRST supervised transition, at burn_in, not an
    event elsewhere in the window. Ordinary anchors can have later event targets.
    """

    def __init__(self, replay, length, burn_in, seed, event_fraction=0.5):
        if length < 1 or burn_in < 0:
            raise ValueError("length must be positive and burn_in non-negative")
        if event_fraction != 0.5:
            raise ValueError("event_fraction is preregistered at 0.5")
        self.length, self.burn_in, self.seed = length, burn_in, seed
        self.width = length + burn_in
        self.rng = np.random.default_rng(seed)
        self.pools = {"event": [], "ordinary": []}
        # This private read is confined to the experiment; no arrays are copied.
        for episode in replay._episodes():
            if episode.transitions[0].before.schema != "grid-v1":
                continue
            count = len(episode.transitions)
            if count <= burn_in:
                continue
            for start in range(max(1, count - self.width + 1)):
                role = "event" if _event(episode.transitions[start + burn_in]) else "ordinary"
                self.pools[role].append((episode, start))
        for role, pool in self.pools.items():
            if not pool:
                raise ValueError(f"{role} window pool is empty; cannot balance batches")
        self.batches = 0
        self.anchors = {"event": 0, "ordinary": 0}
        self.positions = [{"event": 0, "ordinary": 0} for _ in range(length)]

    def sample(self, batch_size: int) -> list[Episode]:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("50:50 sampling requires a positive even batch_size")
        windows = []
        for role, pool in self.pools.items():
            for _ in range(batch_size // 2):
                episode, start = pool[int(self.rng.integers(len(pool)))]
                transitions = episode.transitions[start:start + self.width]
                windows.append(replace(episode, transitions=transitions))
                self.anchors[role] += 1
                for position, transition in enumerate(transitions[self.burn_in:]):
                    target_role = "event" if _event(transition) else "ordinary"
                    self.positions[position][target_role] += 1
        self.rng.shuffle(windows)
        self.batches += 1
        return windows

    def report(self):
        return {
            "event_fraction": 0.5, "seed": self.seed, "with_replacement": True,
            "anchor": "first supervised transition at window index burn_in",
            "ordinary": "all other eligible anchors, including windows with later events",
            "window_support": "SequenceReplay full windows or start-zero short episodes",
            "burn_in": self.burn_in, "train_horizon": self.length,
            "pool_sizes": {role: len(pool) for role, pool in self.pools.items()},
            "batches": self.batches, "sampled_anchors": dict(self.anchors),
            "supervised_transitions": {
                role: sum(row[role] for row in self.positions) for role in self.anchors
            },
            "supervised_by_position": [dict(row) for row in self.positions],
        }


def _load_residual_checkpoint(path: Path):
    """Rebuild only explicitly tagged residual v2 state dicts via the safe loader."""
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except Exception as error:
        raise ValueError(f"could not safely load residual checkpoint: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if payload.get("format_version") != 2:
        raise ValueError("residual checkpoint requires format_version 2")
    if payload.get("latent_parameterization") != "residual_zero_init":
        raise ValueError("checkpoint requires residual_zero_init parameterization")
    head = payload.get("git_head")
    if not isinstance(head, str) or not head:
        raise ValueError("checkpoint git_head must be a non-empty string")
    config = checkpoint_io._required_mapping(payload, "config", "config")
    modules = checkpoint_io._required_mapping(payload, "modules", "modules")
    model_meta = checkpoint_io._required_mapping(modules, "model", "modules.model")
    probe_meta = checkpoint_io._required_mapping(modules, "probe", "modules.probe")
    if model_meta.get("class") != RESIDUAL_CLASS:
        raise ValueError("checkpoint model class must explicitly identify residual dynamics")
    if model_meta.get("schemas") != {"grid-v1": [5, 1]}:
        raise ValueError("checkpoint schemas must be grid-v1 with 5 actions and 1 sensor")
    dimensions = {}
    for field in ("z_dim", "h_dim", "ensemble_size"):
        dimensions[field] = checkpoint_io._positive_int(model_meta.get(field), field)
        if config.get(field) != dimensions[field]:
            raise ValueError(f"checkpoint config.{field} disagrees with module metadata")
    flags = {}
    for field in ("normalize_sensor_condition", "predict_sensor_delta"):
        flags[field] = model_meta.get(field)
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
    model = residual.ResidualLatentWorldModel(
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
        "event_balanced": payload.get("event_balanced", False),
        "config": dict(config), "budgets": dict(payload.get("budgets", {})),
        "load_policy": "torch.load(weights_only=True, map_location='cpu')",
    }


def build_parser():
    parser = residual.build_parser()
    parser.description = __doc__
    parser.add_argument("--event-fraction", type=float, choices=(0.5,), default=0.5)
    return parser


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
        "fixed_corpus": FIXED_CORPUS, "fixed_residual_config": FIXED_CONFIG,
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
            manifest["baseline_checkpoint_git_head"] = baseline_head
            manifest["baseline_checkpoint_metadata"] = metadata
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config), seed=args.seed, z_dim=args.z_dim, h_dim=args.h_dim,
                burn_in=0, replay_capacity=len(temporal.SOURCE_LAYOUTS) * args.episodes_per_layout,
                termination_weight=0.0, salient_fraction=0.0,
            )
            if config.train_horizon != 3:
                raise ValueError("exp151 requires the unchanged train_horizon=3")
            if config.batch_size < 2 or config.batch_size % 2:
                raise ValueError("50:50 sampling requires an even batch_size")
            device = torch.device(config.device)
            with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
                torch.manual_seed(config.seed)
                model = residual.ResidualLatentWorldModel(
                    CoreEncoder(config.z_dim), {"grid-v1": (5, 1)}, config.h_dim,
                    config.ensemble_size, normalize_sensor_condition=config.normalize_sensor_condition,
                    predict_sensor_delta=config.predict_sensor_delta,
                ).to(device)
            trainer = CoreTrainer(model, config)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)
            journal.update("initialize", 2, 2, device=str(device))
            corpus, fit, validation = residual._collect_corpus(args, replay, deadline, journal)
            journal.update("stratify", 0, 1)
            coverage = _audit_counts({"all": replay._episodes()})
            observed_corpus = {
                key: corpus[key] for key in ("episodes", "transitions",
                    "natural_terminals_by_layout", "natural_terminals_fit_cutoff_by_layout")
            }
            observed_corpus.update({key: coverage[key] for key in (
                "rgb_changing_interact_transitions", "episodes_with_rgb_changing_interact")})
            observed_corpus["action_counts"] = {
                action: {key: row[key] for key in ("total", "rgb_changed", "rgb_no_change")}
                for action, row in coverage["actions"].items()
            }
            corpus["observable_counts"] = observed_corpus
            fixed_counts = observed_corpus == FIXED_CORPUS
            if corpus["default_corpus_verified"] and not fixed_counts:
                raise AssertionError(f"exp149 corpus coverage mismatch: {observed_corpus}")
            sampler = EventBalancedSampler(
                replay, config.train_horizon, config.burn_in, config.seed + 145, args.event_fraction
            )
            matching = {
                "default_budgets": all(getattr(args, key) == value for key, value in PROTOCOL.items()),
                "baseline_budgets": all(metadata["budgets"].get(key) == getattr(args, key)
                                        for key in residual.PROTOCOL),
                "baseline_config": metadata["config"] == core._jsonable(asdict(config)),
                "fixed_residual_config": core._jsonable(asdict(config)) == FIXED_CONFIG,
                "fixed_corpus_counts": fixed_counts,
                "baseline_uniform_residual": metadata["event_balanced"] is False,
            }
            exact_protocol = all(matching.values())
            manifest["protocol_match"] = matching
            manifest["sampling"] = sampler.report()
            core._write_json(args.out / "manifest.json", manifest)
            journal.update("stratify", 1, 1, pool_sizes=sampler.report()["pool_sizes"],
                           exact_protocol=exact_protocol)
            journal.update("dynamics", 0, args.dynamics_updates)
            dynamics_losses = []
            for completed in range(0, args.dynamics_updates, args.dynamics_log_every):
                chunk = min(args.dynamics_log_every, args.dynamics_updates - completed)
                for index in range(chunk):
                    core._check_deadline(deadline, f"update {completed + index}")
                    batch = tensorize(sampler.sample(config.batch_size), config.burn_in, device)
                    dynamics_losses.append(trainer.update(batch, Mode.ADAPT))
                # Match core._train_updates' mode boundary from exp150.
                model.eval()
                journal.update("dynamics", completed + chunk, args.dynamics_updates,
                               loss=dynamics_losses[-1]["loss"], sampling=sampler.report())
            model.requires_grad_(False)
            sampling = sampler.report()
            ordered, shuffled, probe_metrics = residual._fit_probes(
                model, config, args, fit, validation, deadline, journal
            )
            journal.update("event_balanced_checkpoint", 0, 1)
            checkpoint_path = args.out / "event_balanced_checkpoint.pt"
            torch.save({
                "format_version": 2, "latent_parameterization": "residual_zero_init",
                "event_balanced": True, "sampling": sampling,
                "analysis_git_head": manifest["analysis_git_head"],
                "git_head": manifest["analysis_git_head"],
                "baseline_checkpoint_git_head": baseline_head,
                "budgets": manifest["budgets"], "config": core._jsonable(asdict(config)),
                "modules": {
                    "model": {
                        "class": RESIDUAL_CLASS, "schemas": core._jsonable(model.schemas),
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
            journal.update("event_balanced_checkpoint", 1, 1)
            one_step = {}
            for role, candidate in (("baseline", baseline), ("event_balanced", model)):
                journal.update(f"{role}_one_step", 0, 120)
                core._check_deadline(deadline, f"{role}_one_step")
                one_step[role] = _diagnose(candidate, journal, args.out / f"{role}_one_step_rows.jsonl")
                core._check_deadline(deadline, f"{role}_one_step")
                journal.update(f"{role}_one_step", 120, 120)
            late_fork = temporal._late_fork_audit(
                model, ordered, config, deadline, journal, args.out / "event_balanced_late_fork_rows.jsonl"
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
                one_step["baseline"]["splits"]["source"],
                one_step["event_balanced"]["splits"]["source"], exact_protocol,
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
                "status": "completed", "claim": "single sampling-distribution development comparison",
                "exact_protocol": exact_protocol, "protocol_match": matching,
                "event_balanced_one_step_gate": one_step_gate,
                "source_compositional_gate": source_gate,
                "event_balanced_composition_gate": bool(one_step_gate and source_gate),
                "physics_transfer_gate": None, "corpus": corpus, "sampling": sampling,
                "dynamics": {
                    "updates": args.dynamics_updates, "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"],
                    "schema_counts": {"grid-v1": args.dynamics_updates},
                    "burn_in": config.burn_in, "train_horizon": config.train_horizon,
                    "termination_weight": config.termination_weight,
                    "salient_fraction": config.salient_fraction,
                },
                "probe": probe_metrics, "baseline_one_step": one_step["baseline"],
                "event_balanced_one_step": one_step["event_balanced"],
                "event_balanced_late_fork": late_fork, "evaluation": evaluation,
                "controls": {
                    "latent_parameterization": "residual_zero_init", "event_balanced_sampling": True,
                    "only_sampling_distribution_changed": exact_protocol,
                    "encoder_gru_actions_sensor_heads_objectives_unchanged": True,
                    "seed_and_chunked_updates_match_exp150": True,
                    "baseline_frozen_checkpoint": str(args.baseline_checkpoint),
                    "baseline_checkpoint_git_head": baseline_head,
                    "baseline_training_budgets": metadata["budgets"],
                    "baseline_training_config": metadata["config"],
                    "event_balanced_training_config": core._jsonable(asdict(config)),
                    "source_only_training": True, "push_distance": 1, "goal_push_distance": 1,
                    "push2_not_run": True, "termination_neutral_planning": True, "beam_width": 5,
                    "canonical_actions_excluded_from_fit_data": True,
                    "one_step_protocol": "unchanged exp148: 8 layouts x 3 steps x 5 actions",
                },
                "limitations": [
                    "RGB-changing action 3 is a domain-local Push proxy, not a semantic box label",
                    "50:50 applies to first supervised anchors, not all targets in a horizon-3 loss",
                    "ordinary anchors may contain later events; realized loss exposure is reported separately",
                    "full-window support excludes the last horizon-1 transitions as anchors in long episodes",
                    "uniform sampling within strata replaces exp150 episode/recent-reservoir weighting",
                    "one training seed, one residual parameterization, one Push-1 task family",
                    "a reduced-budget smoke cannot pass the preregistered scientific gates",
                    "baseline and candidate encode their own latent spaces; compare persistence ratios and failures",
                    "CUDA training is not guaranteed bitwise deterministic despite seeded window sampling",
                    "probe fit includes naturally terminal source episodes as in exp146/exp150",
                    "no planner, parameterization, loss-weight tuning or Push-2 physics test",
                    "not AGI, JEPA, representation-capacity or physics-transfer proof",
                ],
                "artifacts": {"checkpoint": checkpoint_path.name},
            }
            manifest["sampling"] = sampling
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
