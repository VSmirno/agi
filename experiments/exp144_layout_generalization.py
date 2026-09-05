"""Layout-disjoint falsification of the learned temporal image-goal score."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


class HindsightGoalPolicy(nn.Module):
    def __init__(self, z_dim: int, width: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * z_dim, width), nn.ReLU(), nn.Linear(width, 5)
        )

    def forward(self, state, goal):
        return self.network(torch.cat((state, goal, (state - goal).abs()), dim=-1))


class HindsightAgent:
    def __init__(self, model, policy, config, goal_blind=False):
        self.model = model
        self.policy = policy
        self.config = config
        self.goal_blind = goal_blind
        self.last_trace = []
        self.last_model_calls = 0

    @torch.no_grad()
    def start(self, obs, goal):
        if goal.image is None:
            raise ValueError("hindsight policy requires an image goal")
        self.state = self.model.initial(obs)
        self.goal = self.model.initial(goal.image).z
        self.last_trace = []

    @torch.no_grad()
    def act(self, exploration_fraction=0.0):
        if exploration_fraction:
            raise ValueError("hindsight evaluation does not explore")
        goal = torch.zeros_like(self.goal) if self.goal_blind else self.goal
        logits = self.policy(self.state.z, goal)
        action = int(logits.argmax(-1).item())
        self.last_model_calls = 0
        self.last_trace = [{"policy_logits": logits.squeeze(0).cpu().tolist()}]
        return action

    @torch.no_grad()
    def observe(self, transition):
        self.state = self.model.initial(transition.after)


def _hindsight_examples(encoded, episodes, max_distance, terminal_only=False):
    anchor, goal, action, weight = [], [], [], []
    terminal_goal_pairs = 0
    for states, episode in zip(encoded, episodes, strict=True):
        terminal = episode.transitions[-1].terminated
        if terminal_only and not terminal:
            continue
        for start, transition in enumerate(episode.transitions):
            maximum = min(max_distance, len(states) - start - 1)
            terminal_distance = len(states) - start - 1
            if terminal_only:
                distances = (
                    [terminal_distance] if terminal_distance <= max_distance else []
                )
            else:
                distances = range(1, maximum + 1)
            for distance in distances:
                anchor.append(states[start])
                goal.append(states[start + distance])
                action.append(transition.action)
                weight.append(1.0 / distance)
                terminal_goal_pairs += int(
                    terminal and start + distance == len(states) - 1
                )
    if not anchor:
        raise RuntimeError(
            "hindsight policy has no examples; need eligible fit episodes"
        )
    return (
        torch.stack(anchor),
        torch.stack(goal),
        torch.tensor(action, dtype=torch.long, device=anchor[0].device),
        torch.tensor(weight, device=anchor[0].device),
        terminal_goal_pairs,
    )


def _fit_hindsight_policies(examples, z_dim, updates, batch_size, seed):
    anchor, goal, action, weight, terminal_goal_pairs = examples
    device = anchor.device
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        real = HindsightGoalPolicy(z_dim).to(device)
        shuffled_action = HindsightGoalPolicy(z_dim).to(device)
        goal_blind = HindsightGoalPolicy(z_dim).to(device)
        shuffled_action.load_state_dict(real.state_dict())
        goal_blind.load_state_dict(real.state_dict())
    heads = {
        "real": real,
        "shuffled_action": shuffled_action,
        "goal_blind": goal_blind,
    }
    optimizers = {name: torch.optim.Adam(head.parameters(), lr=3e-4)
                  for name, head in heads.items()}
    generator = torch.Generator().manual_seed(seed)
    losses = {name: [] for name in heads}
    for _ in range(updates):
        indices_cpu = torch.randint(
            len(action), (batch_size,), generator=generator, device="cpu"
        )
        shuffle_cpu = torch.randperm(batch_size, generator=generator)
        indices = indices_cpu.to(device)
        shuffled_labels = action[indices_cpu[shuffle_cpu].to(device)]
        for name, head in heads.items():
            optimizers[name].zero_grad(set_to_none=True)
            selected_goal = torch.zeros_like(goal[indices]) if name == "goal_blind" else goal[indices]
            labels = shuffled_labels if name == "shuffled_action" else action[indices]
            per_example = F.cross_entropy(
                head(anchor[indices], selected_goal), labels, reduction="none"
            )
            loss = (per_example * weight[indices]).sum() / weight[indices].sum()
            loss.backward()
            optimizers[name].step()
            losses[name].append(float(loss.detach()))
    for head in heads.values():
        head.eval()
    return heads, {
        "examples": len(action),
        "terminal_goal_pairs": terminal_goal_pairs,
        "losses": {name: {"first": values[0], "last": values[-1]}
                   for name, values in losses.items()},
        "same_initialization_and_batches": True,
    }


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
    policy=None,
    goal_blind=False,
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
                if policy is not None:
                    agent = HindsightAgent(
                        model, policy, episode_config, goal_blind=goal_blind
                    )
                else:
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
    parser.add_argument("--hindsight-updates", type=int, default=0)
    parser.add_argument("--hindsight-batch-size", type=int, default=128)
    parser.add_argument("--hindsight-max-distance", type=int, default=8)
    parser.add_argument("--hindsight-terminal-only", action="store_true")
    parser.add_argument("--hindsight-random-encoder-control", action="store_true")
    args = parser.parse_args(argv)
    if args.hindsight_updates < 0:
        parser.error("--hindsight-updates must be non-negative")
    if args.hindsight_updates and args.hindsight_batch_size <= 0:
        parser.error("--hindsight-batch-size must be positive")
    if args.hindsight_updates and args.hindsight_max_distance <= 0:
        parser.error("--hindsight-max-distance must be positive")
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
    random_encoder_model = None
    if args.hindsight_random_encoder_control:
        random_encoder_model = copy.deepcopy(model).eval()
        random_encoder_model.requires_grad_(False)
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
    fit_encoded = _encode_episodes(model, fit_episodes, device)
    validation_encoded = _encode_episodes(model, validation_episodes, device)
    fit_pairs = _pairs(fit_encoded, args.max_horizon)
    validation_pairs = _pairs(validation_encoded, args.max_horizon)
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
    hindsight_heads = None
    hindsight_training = None
    random_encoder_heads = None
    random_encoder_training = None
    if args.hindsight_updates:
        hindsight_heads, hindsight_training = _fit_hindsight_policies(
            _hindsight_examples(
                fit_encoded,
                fit_episodes,
                args.hindsight_max_distance,
                terminal_only=args.hindsight_terminal_only,
            ),
            config.z_dim,
            args.hindsight_updates,
            args.hindsight_batch_size,
            config.seed + 5144,
        )
        if random_encoder_model is not None:
            random_fit_encoded = _encode_episodes(
                random_encoder_model, fit_episodes, device
            )
            random_encoder_heads, random_encoder_training = (
                _fit_hindsight_policies(
                    _hindsight_examples(
                        random_fit_encoded,
                        fit_episodes,
                        args.hindsight_max_distance,
                        terminal_only=args.hindsight_terminal_only,
                    ),
                    config.z_dim,
                    args.hindsight_updates,
                    args.hindsight_batch_size,
                    config.seed + 5144,
                )
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
        if hindsight_heads is not None:
            evaluation.update({
                "hindsight_real": _evaluate_arm(
                    model, None, config, replay, TEST_LAYOUTS, seeds, args.steps,
                    deadline, trace, "hindsight_real", policy=hindsight_heads["real"],
                ),
                "hindsight_shuffled_action": _evaluate_arm(
                    model, None, config, replay, TEST_LAYOUTS, seeds, args.steps,
                    deadline, trace, "hindsight_shuffled_action",
                    policy=hindsight_heads["shuffled_action"],
                ),
                "hindsight_goal_blind": _evaluate_arm(
                    model, None, config, replay, TEST_LAYOUTS, seeds, args.steps,
                    deadline, trace, "hindsight_goal_blind",
                    policy=hindsight_heads["goal_blind"], goal_blind=True,
                ),
            })
            if random_encoder_heads is not None:
                evaluation["hindsight_random_encoder"] = _evaluate_arm(
                    random_encoder_model,
                    None,
                    config,
                    replay,
                    TEST_LAYOUTS,
                    seeds,
                    args.steps,
                    deadline,
                    trace,
                    "hindsight_random_encoder",
                    policy=random_encoder_heads["real"],
                )
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
        "hindsight": {
            "training": hindsight_training,
            "random_encoder_training": random_encoder_training,
            "terminal_only": args.hindsight_terminal_only,
            "supervision_disclosure": (
                "policy selects successful episodes via Grid termination==success; "
                "frozen encoder also received termination supervision"
                if args.hindsight_terminal_only
                else "policy objective is reward-free, but frozen world-model "
                     "encoder was trained with Grid termination==success supervision"
            ),
        },
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
            (
                "hindsight distance and temporal MPC horizon are independently "
                "configured; comparisons are not horizon-matched when they differ"
            ),
        ],
    }
    if hindsight_heads is not None:
        real_success = evaluation["hindsight_real"]["overall"]["successes"]
        policy_controls = (
            evaluation["hindsight_shuffled_action"]["overall"]["successes"],
            evaluation["raw"]["overall"]["successes"],
            evaluation["ordered"]["overall"]["successes"],
        )
        goal_blind_success = evaluation["hindsight_goal_blind"]["overall"][
            "successes"
        ]
        policy_layouts = sum(
            summary["successes"] > 0
            for summary in evaluation["hindsight_real"]["by_layout"].values()
        )
        terminal_fit_coverage = all(
            any(episode.transitions[-1].terminated for episode in fit_episodes
                if episode.ruleset == layout_name)
            for layout_name in TRAIN_LAYOUTS
        )
        result["hindsight"].update({
            "terminal_fit_coverage": terminal_fit_coverage,
            "terminal_fit_episodes_by_layout": {
                layout_name: sum(
                    episode.transitions[-1].terminated
                    for episode in fit_episodes
                    if episode.ruleset == layout_name
                )
                for layout_name in TRAIN_LAYOUTS
            },
            "development_gate": bool(
                terminal_fit_coverage
                and all(real_success >= control + 4 for control in policy_controls)
                and policy_layouts >= 3
            ),
            "goal_input_diagnostic": bool(
                real_success >= goal_blind_success + 4
            ),
            "goal_input_diagnostic_limitation": (
                "current RGB already exposes the goal tile, so goal-blind does "
                "not isolate task knowledge"
            ),
            "representation_diagnostic": (
                None
                if random_encoder_heads is None
                else real_success >= evaluation["hindsight_random_encoder"][
                    "overall"
                ]["successes"] + 4
            ),
            "successful_test_layouts": policy_layouts,
        })
    _write(args.out / "results.json", result)
    _write(
        args.out / "manifest.json",
        {"argv": sys.argv[1:], "budgets": {k: str(v) for k, v in vars(args).items()},
         "status": "completed"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
