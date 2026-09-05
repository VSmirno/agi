"""Source-only temporal MPC composition probe for Push-1 layouts."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

import torch

from experiments.exp143_temporal_proximity import (
    TemporalAgent,
    TemporalProbe,
    _encode_episodes,
    _fit_pair,
    _pairs,
    _probe_metrics,
)
from experiments.exp145_physics_transfer import (
    SOURCE_LAYOUTS,
    TARGET_LAYOUTS,
    _adapter,
    _canonical_check,
    _collect,
    _goal_observation,
)
from snks.agent.core_agent import CoreAgent
from snks.agent.core_world_model import Prediction
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, GoalSpec, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config
from snks.pipeline.core_runner import EpisodeResult
from snks.pipeline.core_tasks import TaskCase, score_episode


class TerminationNeutralModel:
    """Expose dynamics while making planner terminal handling inactive."""

    def __init__(self, model):
        self._model = model

    @property
    def schemas(self):
        return self._model.schemas

    def initial(self, observation):
        return self._model.initial(observation)

    def step(self, state, actions) -> Prediction:
        prediction = self._model.step(state, actions)
        return replace(
            prediction,
            terminated_prob=torch.zeros_like(prediction.terminated_prob),
        )


class ProgressJournal:
    """Append-only progress records with a bounded quiet period."""

    def __init__(self, output_path: Path, interval: float = 30.0):
        if interval <= 0:
            raise ValueError("progress interval must be positive")
        self.output_path = Path(output_path)
        self._interval = interval
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._closed = False
        self._latest: dict[str, Any] | None = None
        self._handle = self.output_path.open("x", encoding="utf-8")
        self._thread = threading.Thread(
            target=self._heartbeat,
            name="exp146-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, _traceback):
        if exc_value is None:
            self.close(status="completed")
        else:
            self.close(
                status="failed",
                error=f"{exc_type.__name__}: {exc_value}",
            )
        return False

    def _elapsed(self) -> float:
        return time.monotonic() - self._started

    def _write(self, record: dict[str, Any]) -> None:
        encoded = json.dumps(record, sort_keys=True)
        self._handle.write(encoded + "\n")
        self._handle.flush()
        print(encoded, flush=True)

    def update(self, stage: str, completed: int, total: int, **metrics: Any) -> None:
        if completed < 0 or total <= 0 or completed > total:
            raise ValueError("progress requires 0 <= completed <= total and total > 0")
        record = {
            "stage": stage,
            "completed": completed,
            "total": total,
            "elapsed_seconds": self._elapsed(),
            **metrics,
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("progress journal is closed")
            self._latest = record
            self._write(record)

    def _heartbeat(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                if self._closed:
                    return
                if self._latest is None:
                    continue
                record = {
                    **self._latest,
                    "elapsed_seconds": self._elapsed(),
                    "heartbeat": True,
                }
                self._write(record)

    def close(self, *, status: str = "completed", error: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            record: dict[str, Any] = {
                "status": status,
                "elapsed_seconds": self._elapsed(),
            }
            if error is not None:
                record["error"] = error
            self._write(record)
            self._closed = True
            self._stop.set()
            self._handle.close()
        self._thread.join()


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    core._write_json(path, payload)


def _episode_result(adapter, agent, case: TaskCase) -> EpisodeResult:
    """Evaluate an existing beam-planning agent without adapting its model."""

    observation = adapter.reset(case.seed)
    steps = adapter.reset_transitions
    if steps >= case.max_steps:
        raise ValueError("reset exhausted episode transition budget")
    agent.start(observation, case.goal)
    audit = [
        {
            "step": steps,
            "sensors": observation.sensors.tolist(),
            "sensor_mask": observation.sensor_mask.tolist(),
            "diagnostic": adapter.diagnostic_snapshot(),
        }
    ]
    transitions = []
    calls = 0
    failed = False
    while steps < case.max_steps:
        try:
            action = agent.act(0.0)
            calls += agent.last_model_calls
            if not 0 <= action < len(adapter.actions.names):
                raise ValueError("agent returned invalid primitive")
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failed = True
            audit.append({"step": steps, "agent_failure": str(error)})
            break
        transition = adapter.step(action)
        steps += 1
        if steps == case.max_steps and not transition.terminated:
            transition = replace(transition, truncated=True)
        transitions.append(transition)
        try:
            agent.observe(transition)
        except (FloatingPointError, RuntimeError, ValueError) as error:
            failed = True
            audit.append({"step": steps, "agent_failure": str(error)})
            break
        audit.append(
            {
                "step": steps,
                "action": action,
                "sensors": transition.after.sensors.tolist(),
                "sensor_mask": transition.after.sensor_mask.tolist(),
                "diagnostic": adapter.diagnostic_snapshot(),
                "candidates": agent.last_trace,
            }
        )
        if transition.terminated or transition.truncated:
            break
    episode = Episode(
        case.uid,
        case.split,
        case.family,
        case.ruleset,
        tuple(transitions),
    )
    return EpisodeResult(
        episode=episode,
        steps=steps,
        agent_failed=failed,
        infrastructure_failed=False,
        audit=audit,
        success=not failed and score_episode(case, audit),
        model_calls=calls,
    )


def _evaluate(
    model,
    probes: dict[str, TemporalProbe],
    config,
    replay,
    seeds: range,
    steps: int,
    deadline: float,
    trace,
    journal: ProgressJournal,
    progress: list[int],
    total: int,
):
    evaluation = {}
    arms = (
        ("ordered_h3", "ordered", 3),
        ("ordered_h1", "ordered", 1),
        ("shuffled_h3", "shuffled", 3),
        ("raw_h3", None, 3),
    )
    neutral_model = TerminationNeutralModel(model)
    for role, probe_name, horizon in arms:
        by_layout = {}
        all_results = []
        for layout_name, (layout, _canonical_push1, _push2) in TARGET_LAYOUTS.items():
            layout_results = []
            for seed in seeds:
                core._check_deadline(deadline, f"{role}/{layout_name}/{seed}")
                adapter = _adapter(layout, 1, seed, steps)
                try:
                    goal = _goal_observation(layout, 1, seed, steps)
                    case = TaskCase(
                        uid=f"temporal-mpc:{role}:{layout_name}:{seed}",
                        family="push_box",
                        ruleset=f"push1:{layout_name}",
                        seed=seed,
                        split="validation",
                        goal=GoalSpec(goal, {}),
                        max_steps=steps,
                    )
                    episode_config = replace(
                        config,
                        seed=seed,
                        planner_horizon=horizon,
                        beam_width=5,
                    )
                    agent = (
                        CoreAgent(neutral_model, episode_config)
                        if probe_name is None
                        else TemporalAgent(
                            neutral_model,
                            episode_config,
                            probes[probe_name],
                        )
                    )
                    result = _episode_result(adapter, agent, case)
                finally:
                    adapter.close()
                layout_results.append(result)
                all_results.append(result)
                trace.write(
                    {
                        "role": role,
                        "layout": layout_name,
                        "push_distance": 1,
                        "goal_push_distance": 1,
                        "seed": seed,
                        **core._result_record(result),
                        "audit": result.audit,
                    }
                )
                progress[0] += 1
                journal.update(
                    "evaluate",
                    progress[0],
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


def _probe_split(episodes_by_layout, args):
    fit_by_layout = {}
    validation_by_layout = {}
    cutoff = round(0.75 * args.episodes_per_layout)
    for name, episodes in episodes_by_layout.items():
        selected = {episode.uid: episode for episode in episodes[:args.probe_episodes_per_layout]}
        for episode in episodes[:cutoff]:
            if episode.transitions and episode.transitions[-1].terminated:
                selected[episode.uid] = episode
        fit = list(selected.values())
        validation = episodes[cutoff:cutoff + args.probe_validation_per_layout]
        if {episode.uid for episode in fit} & {episode.uid for episode in validation}:
            raise RuntimeError("probe fit and validation episodes overlap")
        fit_by_layout[name] = fit
        validation_by_layout[name] = validation
    return fit_by_layout, validation_by_layout, cutoff


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-layout", type=_positive, default=512)
    parser.add_argument("--collection-steps", type=_positive, default=64)
    parser.add_argument("--eval-steps", type=_positive, default=8)
    parser.add_argument("--dynamics-updates", type=_positive, default=2000)
    parser.add_argument("--dynamics-log-every", type=_positive, default=100)
    parser.add_argument("--probe-updates", type=_positive, default=400)
    parser.add_argument("--probe-batch-size", type=_positive, default=256)
    parser.add_argument("--probe-episodes-per-layout", type=_positive, default=64)
    parser.add_argument("--probe-validation-per-layout", type=_positive, default=16)
    parser.add_argument("--max-horizon", type=_positive, default=3)
    parser.add_argument("--eval-seeds", type=_positive, default=6)
    parser.add_argument("--z-dim", type=_positive, default=256)
    parser.add_argument("--h-dim", type=_positive, default=128)
    parser.add_argument("--max-seconds", type=_positive, default=3600)
    parser.add_argument("--progress-interval", type=_positive, default=30)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.probe_episodes_per_layout > args.episodes_per_layout:
        parser.error("--probe-episodes-per-layout exceeds --episodes-per-layout")
    if args.probe_validation_per_layout > args.episodes_per_layout - round(
        0.75 * args.episodes_per_layout
    ):
        parser.error("probe validation subset exceeds held-out source episodes")
    args.out.mkdir(parents=True, exist_ok=False)
    invocation = list(argv) if argv is not None else sys.argv[1:]
    deadline = time.monotonic() + args.max_seconds
    manifest_base = {
        "argv": invocation,
        "git_head": core._git_commit(),
        "budgets": core._jsonable(vars(args)),
    }
    with ProgressJournal(args.out / "progress.jsonl", args.progress_interval) as journal:
        try:
            core._seed_everything(args.seed)
            config = replace(
                load_core_config(args.config),
                seed=args.seed,
                z_dim=args.z_dim,
                h_dim=args.h_dim,
                burn_in=0,
                replay_capacity=len(SOURCE_LAYOUTS) * args.episodes_per_layout,
                termination_weight=0.0,
                salient_fraction=0.0,
            )
            with torch.random.fork_rng(
                devices=[torch.device(config.device).index or 0]
                if torch.device(config.device).type == "cuda"
                else []
            ):
                torch.manual_seed(config.seed)
                model = core.CoreWorldModel(
                    CoreEncoder(config.z_dim),
                    {"grid-v1": (5, 1)},
                    config.h_dim,
                    config.ensemble_size,
                    normalize_sensor_condition=config.normalize_sensor_condition,
                    predict_sensor_delta=config.predict_sensor_delta,
                ).to(config.device)
            trainer = CoreTrainer(model, config)
            replay = SequenceReplay(config.replay_capacity, config.seed + 145)

            total_collection = len(SOURCE_LAYOUTS) * args.episodes_per_layout
            journal.update("collect", 0, total_collection)
            episodes_by_layout = {name: [] for name in SOURCE_LAYOUTS}
            corpus = {
                name: {"episodes": 0, "natural_terminals": 0, "transitions": 0}
                for name in SOURCE_LAYOUTS
            }
            collection_completed = 0
            for offset in range(args.episodes_per_layout):
                for layout_index, (name, (layout, _actions)) in enumerate(
                    SOURCE_LAYOUTS.items()
                ):
                    core._check_deadline(deadline, f"collect/{name}/{offset}")
                    seed = 10000 + layout_index * 100000 + offset
                    episode = _collect(name, layout, seed, args.collection_steps)
                    episodes_by_layout[name].append(episode)
                    replay.append(episode, Mode.ADAPT)
                    row = corpus[name]
                    row["episodes"] += 1
                    row["natural_terminals"] += int(
                        bool(episode.transitions and episode.transitions[-1].terminated)
                    )
                    row["transitions"] += len(episode.transitions)
                    collection_completed += 1
                    journal.update(
                        "collect",
                        collection_completed,
                        total_collection,
                        layout=name,
                        offset=offset,
                    )

            journal.update("dynamics", 0, args.dynamics_updates)
            dynamics_losses = []
            schema_counts = {}
            for completed in range(0, args.dynamics_updates, args.dynamics_log_every):
                core._check_deadline(deadline, f"dynamics/{completed}")
                chunk = min(args.dynamics_log_every, args.dynamics_updates - completed)
                losses, chunk_counts = core._train_updates(
                    model,
                    trainer,
                    replay,
                    config,
                    chunk,
                    Mode.ADAPT,
                    deadline,
                    schema="grid-v1",
                )
                dynamics_losses.extend(losses)
                for schema, count in chunk_counts.items():
                    schema_counts[schema] = schema_counts.get(schema, 0) + count
                journal.update(
                    "dynamics",
                    completed + chunk,
                    args.dynamics_updates,
                    loss=losses[-1]["loss"],
                )
            model.eval()
            model.requires_grad_(False)

            fit_by_layout, validation_by_layout, cutoff = _probe_split(
                episodes_by_layout, args
            )
            fit_episodes = [
                episode for name in SOURCE_LAYOUTS for episode in fit_by_layout[name]
            ]
            validation_episodes = [
                episode
                for name in SOURCE_LAYOUTS
                for episode in validation_by_layout[name]
            ]
            terminal_fit = {
                name: sum(
                    bool(episode.transitions and episode.transitions[-1].terminated)
                    for episode in fit_by_layout[name]
                )
                for name in SOURCE_LAYOUTS
            }

            device = torch.device(config.device)
            journal.update("probe_encode", 0, 2, split="fit")
            fit_encoded = _encode_episodes(model, fit_episodes, device)
            journal.update("probe_encode", 1, 2, split="validation")
            validation_encoded = _encode_episodes(model, validation_episodes, device)
            journal.update("probe_encode", 2, 2)
            journal.update("probe_pairs", 0, 2, split="fit")
            fit_pairs = _pairs(fit_encoded, args.max_horizon)
            journal.update("probe_pairs", 1, 2, split="validation")
            validation_pairs = _pairs(validation_encoded, args.max_horizon)
            journal.update("probe_pairs", 2, 2)

            with torch.random.fork_rng(
                devices=[device.index or 0] if device.type == "cuda" else []
            ):
                torch.manual_seed(config.seed + 146)
                ordered = TemporalProbe(config.z_dim).to(device)
                shuffled = TemporalProbe(config.z_dim).to(device)
                shuffled.load_state_dict(ordered.state_dict())
            journal.update("probe_fit", 0, args.probe_updates)
            probe_losses = _fit_pair(
                ordered,
                shuffled,
                fit_pairs,
                args.probe_updates,
                args.probe_batch_size,
                config.seed + 146,
            )
            core._check_deadline(deadline, "probe_fit")
            journal.update("probe_fit", args.probe_updates, args.probe_updates)
            probe_metrics = {
                "train": {
                    "ordered": _probe_metrics(ordered, fit_pairs),
                    "shuffled_endpoint": _probe_metrics(shuffled, fit_pairs),
                },
                "validation": {
                    "ordered": _probe_metrics(ordered, validation_pairs),
                    "shuffled_endpoint": _probe_metrics(shuffled, validation_pairs),
                },
                "losses": probe_losses,
            }

            total_evaluation = len(TARGET_LAYOUTS) * args.eval_seeds * 4
            journal.update("evaluate", 0, total_evaluation)
            trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
            try:
                evaluation = _evaluate(
                    model,
                    {"ordered": ordered, "shuffled": shuffled},
                    config,
                    replay,
                    range(20000, 20000 + args.eval_seeds),
                    args.eval_steps,
                    deadline,
                    trace,
                    journal,
                    [0],
                    total_evaluation,
                )
            finally:
                trace.close()

            journal.update("artifacts", 0, 1, operation="canonical_push1_audit")
            canonical_push1 = {
                name: {
                    "actions": list(actions),
                    "reachable": _canonical_check(layout, 1, actions, 30000),
                }
                for name, (layout, actions, _push2) in TARGET_LAYOUTS.items()
            }
            journal.update("artifacts", 1, 1, operation="write_results")
            ordered_success = evaluation["ordered_h3"]["overall"]["successes"]
            controls = (
                "ordered_h1",
                "shuffled_h3",
                "raw_h3",
            )
            per_layout_floor = min(
                summary["successes"]
                for summary in evaluation["ordered_h3"]["by_layout"].values()
            )
            source_compositional_gate = bool(
                all(count > 0 for count in terminal_fit.values())
                and ordered_success >= 18
                and per_layout_floor >= 3
                and all(
                    ordered_success
                    >= evaluation[control]["overall"]["successes"] + 4
                    for control in controls
                )
            )
            result = {
                "status": "completed",
                "claim": (
                    "development compositional source qualification; not physics "
                    "transfer or AGI evidence"
                ),
                "source_compositional_gate": source_compositional_gate,
                "physics_transfer_gate": None,
                "corpus": {
                    "source_push_distance": 1,
                    "collection_interleaved_by_offset": True,
                    "collection_seed_scheme": "10000 + layout_index * 100000 + offset",
                    "episodes": sum(len(items) for items in episodes_by_layout.values()),
                    "transitions": sum(row["transitions"] for row in corpus.values()),
                    "by_layout": corpus,
                    "source_layouts": {
                        name: {"layout": repr(layout), "actions": list(actions)}
                        for name, (layout, actions) in SOURCE_LAYOUTS.items()
                    },
                    "episode_uids_by_layout": {
                        name: [episode.uid for episode in items]
                        for name, items in episodes_by_layout.items()
                    },
                    "fit_cutoff_per_layout": cutoff,
                    "fit_episodes_by_layout": {
                        name: len(items) for name, items in fit_by_layout.items()
                    },
                    "validation_episodes_by_layout": {
                        name: len(items) for name, items in validation_by_layout.items()
                    },
                    "natural_terminal_fit_episodes_by_layout": terminal_fit,
                },
                "dynamics": {
                    "updates": args.dynamics_updates,
                    "loss_first": dynamics_losses[0]["loss"],
                    "loss_last": dynamics_losses[-1]["loss"],
                    "schema_counts": schema_counts,
                    "burn_in": config.burn_in,
                    "termination_weight": config.termination_weight,
                    "salient_fraction": config.salient_fraction,
                },
                "probe": probe_metrics,
                "evaluation": evaluation,
                "canonical_push1_reachability": canonical_push1,
                "controls": {
                    "source_only_training": True,
                    "source_only_evaluation": True,
                    "push_distance": 1,
                    "goal_push_distance": 1,
                    "termination_neutral_planning": True,
                    "ordered_h3": {"planner_horizon": 3, "beam_width": 5},
                    "ordered_h1": {"planner_horizon": 1, "beam_width": 5},
                    "shuffled_h3": {"planner_horizon": 3, "beam_width": 5},
                    "raw_h3": {"planner_horizon": 3, "beam_width": 5},
                    "canonical_actions_excluded_from_fit_data": True,
                    "push2_not_run": True,
                },
                "limitations": [
                    "source-only Push-1 geometry; no target physics was constructed",
                    "one task family and one declared training seed",
                    "temporal proximity is policy-dependent, not a proof of shortest paths",
                    "the bounded result is not physics transfer or AGI evidence",
                ],
            }
            _write_json(args.out / "results.json", result)
            _write_json(
                args.out / "manifest.json",
                {**manifest_base, "exit_code": 0, "status": "completed"},
            )
            journal.close(status="completed")
            return 0
        except Exception as error:
            _write_json(
                args.out / "manifest.json",
                {
                    **manifest_base,
                    "exit_code": 1,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


if __name__ == "__main__":
    raise SystemExit(main())
