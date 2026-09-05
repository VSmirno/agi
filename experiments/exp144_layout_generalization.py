"""Layout-disjoint falsification of the learned temporal image-goal score."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from exp143_temporal_proximity import (
    TemporalAgent,
    TemporalProbe,
    _encode_episodes,
    _fit_pair,
    _pairs,
    _probe_metrics,
)
from snks.agent.core_agent import CoreAgent
from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_grid import CoreGridWorld, GridCoreAdapter, GridRules, PushLayout
from snks.env.core_types import Episode, GoalSpec, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config
from snks.pipeline.core_runner import run_episode
from snks.pipeline.core_tasks import TaskCase


TRAIN_LAYOUTS = {
    "east_facing_east": (PushLayout((2, 3), 0, (3, 3), (5, 3)), (3, 2, 3)),
    "east_facing_north": (PushLayout((2, 3), 3, (3, 3), (5, 3)), (1, 3, 2, 3)),
    "west_facing_west": (PushLayout((5, 3), 2, (4, 3), (2, 3)), (3, 2, 3)),
    "south_facing_south": (PushLayout((3, 2), 1, (3, 3), (3, 5)), (3, 2, 3)),
}

TEST_LAYOUTS = {
    "east_facing_south": (PushLayout((2, 3), 1, (3, 3), (5, 3)), (0, 3, 2, 3)),
    "west_facing_north": (PushLayout((5, 3), 3, (4, 3), (2, 3)), (0, 3, 2, 3)),
    "north_facing_east": (PushLayout((3, 5), 0, (3, 4), (3, 2)), (0, 3, 2, 3)),
    "south_facing_east": (PushLayout((3, 2), 0, (3, 3), (3, 5)), (1, 3, 2, 3)),
}


class ProbeDifference:
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative

    def __call__(self, anchor, target, horizon):
        return self.positive(anchor, target, horizon) - self.negative(
            anchor, target, horizon
        )


def _adapter(layout: PushLayout, seed: int, steps: int):
    return GridCoreAdapter(CoreGridWorld(
        "push_box", GridRules(push_distance=1), seed, steps, layout=layout
    ))


def _collect(layout_name: str, layout: PushLayout, seed: int, steps: int) -> Episode:
    adapter = _adapter(layout, seed, steps)
    rng = random.Random(seed + 144000)
    transitions = []
    try:
        adapter.reset(seed)
        for _ in range(steps):
            transition = adapter.step(rng.randrange(5))
            transitions.append(transition)
            if transition.terminated or transition.truncated:
                break
    finally:
        adapter.close()
    return Episode(
        f"push-layout:{layout_name}:adapt:{seed}",
        "adapt",
        "push_box",
        layout_name,
        tuple(transitions),
    )


def _canonical_check(layout: PushLayout, actions: tuple[int, ...], seed: int):
    adapter = _adapter(layout, seed, 32)
    try:
        adapter.reset(seed)
        goal = adapter.goal_observation()
        for action in actions:
            transition = adapter.step(action)
        return bool(
            transition.terminated
            and np.array_equal(transition.after.rgb, goal.rgb)
        )
    finally:
        adapter.close()


def _evaluate_arm(
    model,
    probe,
    config,
    replay,
    layouts,
    seeds,
    steps,
    deadline,
    trace,
    role,
):
    by_layout = {}
    all_results = []
    for layout_name, (layout, _) in layouts.items():
        results = []
        for seed in seeds:
            core._check_deadline(deadline, f"{role}/{layout_name}/{seed}")
            adapter = _adapter(layout, seed, steps)
            try:
                case = TaskCase(
                    f"push-layout:{layout_name}:validation:{seed}",
                    "push_box",
                    layout_name,
                    seed,
                    "validation",
                    GoalSpec(adapter.goal_observation(), {}),
                    steps,
                )
                episode_config = replace(config, seed=seed, beam_width=5)
                agent = (CoreAgent(model, episode_config) if probe is None
                         else TemporalAgent(model, episode_config, probe))
                result = run_episode(
                    adapter, agent, case, Mode.EVALUATE, replay, None, exploration=0.0
                )
            finally:
                adapter.close()
            results.append(result)
            all_results.append(result)
            trace.write({
                "role": role,
                "layout": layout_name,
                "seed": seed,
                **core._result_record(result),
                "audit": result.audit,
            })
        by_layout[layout_name] = core._summarize_episodes(results)
    return {"overall": core._summarize_episodes(all_results), "by_layout": by_layout}


def _write(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-layout", type=int, default=64)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--dynamics-updates", type=int, default=2000)
    parser.add_argument("--probe-updates", type=int, default=400)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--eval-seeds", type=int, default=6)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--max-horizon", type=int, default=3)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    config = replace(
        load_core_config(args.config),
        seed=args.seed,
        z_dim=args.z_dim,
        h_dim=args.h_dim,
        burn_in=0,
        replay_capacity=len(TRAIN_LAYOUTS) * args.episodes_per_layout,
    )
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        model = CoreWorldModel(
            CoreEncoder(config.z_dim),
            {"grid-v1": (5, 1)},
            config.h_dim,
            config.ensemble_size,
            normalize_sensor_condition=config.normalize_sensor_condition,
            predict_sensor_delta=config.predict_sensor_delta,
        ).to(config.device)
    trainer = CoreTrainer(model, config)
    replay = SequenceReplay(config.replay_capacity, config.seed + 144)

    episodes = []
    by_layout_corpus = {name: {"episodes": 0, "successes": 0, "steps": 0}
                        for name in TRAIN_LAYOUTS}
    for offset in range(args.episodes_per_layout):
        for layout_index, (name, (layout, _)) in enumerate(TRAIN_LAYOUTS.items()):
            seed = 10000 + layout_index * 1000 + offset
            episode = _collect(name, layout, seed, args.steps)
            episodes.append(episode)
            replay.append(episode, Mode.ADAPT)
            row = by_layout_corpus[name]
            row["episodes"] += 1
            row["successes"] += int(episode.transitions[-1].terminated)
            row["steps"] += len(episode.transitions)

    losses, schema_counts = core._train_updates(
        model,
        trainer,
        replay,
        config,
        args.dynamics_updates,
        Mode.ADAPT,
        deadline,
        schema="grid-v1",
    )
    model.eval()
    model.requires_grad_(False)

    fit_episodes, validation_episodes = [], []
    cutoff = round(0.75 * args.episodes_per_layout)
    for name in TRAIN_LAYOUTS:
        selected = [episode for episode in episodes if episode.ruleset == name]
        fit_episodes.extend(selected[:cutoff])
        validation_episodes.extend(selected[cutoff:])
    device = torch.device(config.device)
    fit_pairs = _pairs(
        _encode_episodes(model, fit_episodes, device), args.max_horizon
    )
    validation_pairs = _pairs(
        _encode_episodes(model, validation_episodes, device), args.max_horizon
    )
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(config.seed + 144)
        ordered = TemporalProbe(config.z_dim).to(device)
        shuffled = TemporalProbe(config.z_dim).to(device)
        shuffled.load_state_dict(ordered.state_dict())
    probe_losses = _fit_pair(
        ordered,
        shuffled,
        fit_pairs,
        args.probe_updates,
        args.probe_batch_size,
        config.seed + 144,
    )

    trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
    seeds = range(20000, 20000 + args.eval_seeds)
    try:
        evaluation = {
            "raw": _evaluate_arm(
                model, None, config, replay, TEST_LAYOUTS, seeds, args.steps,
                deadline, trace, "raw",
            ),
            "ordered": _evaluate_arm(
                model, ordered, config, replay, TEST_LAYOUTS, seeds, args.steps,
                deadline, trace, "ordered",
            ),
            "shuffled_endpoint": _evaluate_arm(
                model, shuffled, config, replay, TEST_LAYOUTS, seeds, args.steps,
                deadline, trace, "shuffled_endpoint",
            ),
            "debiased_ordered_minus_shuffled": _evaluate_arm(
                model, ProbeDifference(ordered, shuffled), config, replay,
                TEST_LAYOUTS, seeds, args.steps, deadline, trace,
                "debiased_ordered_minus_shuffled",
            ),
            "reverse_shuffled_minus_ordered": _evaluate_arm(
                model, ProbeDifference(shuffled, ordered), config, replay,
                TEST_LAYOUTS, seeds, args.steps, deadline, trace,
                "reverse_shuffled_minus_ordered",
            ),
        }
    finally:
        trace.close()
    ordered_success = evaluation["ordered"]["overall"]["successes"]
    raw_success = evaluation["raw"]["overall"]["successes"]
    shuffled_success = evaluation["shuffled_endpoint"]["overall"]["successes"]
    successful_layouts = sum(
        summary["successes"] > 0
        for summary in evaluation["ordered"]["by_layout"].values()
    )
    informative = all(row["successes"] > 0 for row in by_layout_corpus.values())
    debiased_success = evaluation["debiased_ordered_minus_shuffled"]["overall"][
        "successes"
    ]
    reverse_success = evaluation["reverse_shuffled_minus_ordered"]["overall"][
        "successes"
    ]
    debiased_layouts = sum(
        summary["successes"] > 0
        for summary in evaluation["debiased_ordered_minus_shuffled"][
            "by_layout"
        ].values()
    )
    result = {
        "status": "completed",
        "claim": "layout-disjoint development test; not transfer or AGI evidence",
        "informative_training_coverage": informative,
        "F3_layout_gate": bool(
            informative
            and ordered_success >= raw_success + 4
            and ordered_success >= shuffled_success + 4
            and successful_layouts >= 3
        ),
        "F3_debiased_layout_gate": bool(
            informative
            and debiased_success >= raw_success + 4
            and debiased_success >= reverse_success + 4
            and debiased_layouts >= 3
        ),
        "evaluation": evaluation,
        "corpus": {
            "by_layout": by_layout_corpus,
            "episodes": len(episodes),
            "transitions": sum(len(episode.transitions) for episode in episodes),
            "interleaved_layout_insertion": True,
        },
        "probe": {
            "fit_episodes": len(fit_episodes),
            "validation_episodes": len(validation_episodes),
            "fit_pairs": len(fit_pairs),
            "validation_pairs": len(validation_pairs),
            "losses": probe_losses,
            "validation": {
                "ordered": _probe_metrics(ordered, validation_pairs),
                "shuffled_endpoint": _probe_metrics(shuffled, validation_pairs),
            },
        },
        "dynamics": {
            "z_dim": config.z_dim,
            "h_dim": config.h_dim,
            "updates": args.dynamics_updates,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "schema_counts": schema_counts,
        },
        "layouts": {
            "train": {name: {"layout": repr(layout), "canonical": list(actions)}
                      for name, (layout, actions) in TRAIN_LAYOUTS.items()},
            "test": {name: {"layout": repr(layout), "canonical": list(actions)}
                     for name, (layout, actions) in TEST_LAYOUTS.items()},
            "canonical_reachable": {
                name: _canonical_check(layout, actions, 30000)
                for name, (layout, actions) in {**TRAIN_LAYOUTS, **TEST_LAYOUTS}.items()
            },
        },
        "controls": {
            "layout_split_before_pair_construction": True,
            "same_frozen_world_model": True,
            "same_beam_width": 5,
            "same_eval_seeds": list(seeds),
            "goal_score_arms": [
                "raw", "ordered", "shuffled_endpoint",
                "debiased_ordered_minus_shuffled",
                "reverse_shuffled_minus_ordered",
            ],
        },
        "limitations": [
            "one push family and one push-distance rule",
            "four handcrafted train and four handcrafted test layouts",
            "goal template still has one solved pose",
            "single training seed and no confidence interval",
        ],
    }
    _write(args.out / "results.json", result)
    _write(
        args.out / "manifest.json",
        {"argv": sys.argv[1:], "budgets": {k: str(v) for k, v in vars(args).items()},
         "status": "completed"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
