"""Zero-shot Push-1 to Push-2 transfer with a predictive visual controller."""

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

from exp144_layout_generalization import (
    HindsightAgent,
    _encode_episodes,
    _fit_hindsight_policies,
    _hindsight_examples,
)
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


PUSH_ONE = (3, 2, 3, 2, 3, 2, 3)
PUSH_TWO = (3, 2, 2, 3)

SOURCE_LAYOUTS = {
    "east_row2": (PushLayout((1, 2), 0, (2, 2), (6, 2)), PUSH_ONE),
    "west_row3": (PushLayout((6, 3), 2, (5, 3), (1, 3)), PUSH_ONE),
    "south_col4": (PushLayout((4, 1), 1, (4, 2), (4, 6)), PUSH_ONE),
    "north_col5": (PushLayout((5, 6), 3, (5, 5), (5, 1)), PUSH_ONE),
}

TARGET_LAYOUTS = {
    "east_row4_left": (
        PushLayout((1, 4), 1, (2, 4), (6, 4)),
        (0,) + PUSH_ONE,
        (0,) + PUSH_TWO,
    ),
    "west_row5_left": (
        PushLayout((6, 5), 3, (5, 5), (1, 5)),
        (0,) + PUSH_ONE,
        (0,) + PUSH_TWO,
    ),
    "north_col2_left": (
        PushLayout((2, 6), 0, (2, 5), (2, 1)),
        (0,) + PUSH_ONE,
        (0,) + PUSH_TWO,
    ),
    "south_col5_right": (
        PushLayout((5, 1), 0, (5, 2), (5, 6)),
        (1,) + PUSH_ONE,
        (1,) + PUSH_TWO,
    ),
}


def _adapter(layout, push_distance, seed, steps):
    return GridCoreAdapter(
        CoreGridWorld(
            "push_box",
            GridRules(push_distance=push_distance),
            seed,
            steps,
            layout=layout,
        )
    )


def _goal_observation(layout, push_distance, seed, steps):
    adapter = _adapter(layout, push_distance, seed, steps)
    try:
        return adapter.goal_observation()
    finally:
        adapter.close()


def _collect(layout_name, layout, seed, steps):
    adapter = _adapter(layout, 1, seed, steps)
    rng = random.Random(seed + 145000)
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
        f"push-physics:{layout_name}:push1:adapt:{seed}",
        "adapt",
        "push_box",
        f"push1:{layout_name}",
        tuple(transitions),
    )


def _canonical_check(layout, push_distance, actions, seed):
    adapter = _adapter(layout, push_distance, seed, 32)
    try:
        adapter.reset(seed)
        native_goal = adapter.goal_observation()
        for action in actions:
            transition = adapter.step(action)
        return bool(
            transition.terminated
            and np.array_equal(transition.after.rgb, native_goal.rgb)
        )
    finally:
        adapter.close()


def _goal_intervention_check(layout, seed):
    push1 = _adapter(layout, 1, seed, 32)
    push2 = _adapter(layout, 2, seed, 32)
    try:
        initial1 = push1.reset(seed)
        initial2 = push2.reset(seed)
        goal1 = push1.goal_observation()
        goal2 = push2.goal_observation()
        return {
            "initial_rgb_equal": bool(np.array_equal(initial1.rgb, initial2.rgb)),
            "native_goal_rgb_equal": bool(np.array_equal(goal1.rgb, goal2.rgb)),
        }
    finally:
        push1.close()
        push2.close()


def _evaluate(
    model,
    policy,
    config,
    replay,
    push_distance,
    goal_push_distance,
    seeds,
    steps,
    deadline,
    trace,
    role,
):
    by_layout = {}
    all_results = []
    for layout_name, (layout, _, _) in TARGET_LAYOUTS.items():
        results = []
        for seed in seeds:
            core._check_deadline(deadline, f"{role}/{layout_name}/{seed}")
            adapter = _adapter(layout, push_distance, seed, steps)
            try:
                goal = _goal_observation(
                    layout, goal_push_distance, seed, steps
                )
                case = TaskCase(
                    f"push-physics:{layout_name}:{role}:{seed}",
                    "push_box",
                    f"push{push_distance}:{layout_name}",
                    seed,
                    "validation",
                    GoalSpec(goal, {}),
                    steps,
                )
                episode_config = replace(config, seed=seed)
                agent = HindsightAgent(model, policy, episode_config)
                result = run_episode(
                    adapter,
                    agent,
                    case,
                    Mode.EVALUATE,
                    replay,
                    None,
                    exploration=0.0,
                )
            finally:
                adapter.close()
            results.append(result)
            all_results.append(result)
            trace.write({
                "role": role,
                "layout": layout_name,
                "push_distance": push_distance,
                "goal_push_distance": goal_push_distance,
                "seed": seed,
                **core._result_record(result),
                "audit": result.audit,
            })
        by_layout[layout_name] = core._summarize_episodes(results)
    return {
        "overall": core._summarize_episodes(all_results),
        "by_layout": by_layout,
    }


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-layout", type=int, default=2048)
    parser.add_argument("--collection-steps", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=32)
    parser.add_argument("--dynamics-updates", type=int, default=2000)
    parser.add_argument("--policy-updates", type=int, default=600)
    parser.add_argument("--policy-batch-size", type=int, default=128)
    parser.add_argument("--policy-horizon", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, default=6)
    parser.add_argument("--z-dim", type=int, default=256)
    parser.add_argument("--h-dim", type=int, default=128)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for name in (
        "episodes_per_layout",
        "collection_steps",
        "eval_steps",
        "dynamics_updates",
        "policy_updates",
        "policy_batch_size",
        "policy_horizon",
        "eval_seeds",
        "z_dim",
        "h_dim",
        "max_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
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
    random_model = copy.deepcopy(model).eval()
    random_model.requires_grad_(False)
    trainer = CoreTrainer(model, config)
    replay = SequenceReplay(config.replay_capacity, config.seed + 145)

    episodes = []
    corpus = {
        name: {"episodes": 0, "successes": 0, "steps": 0}
        for name in SOURCE_LAYOUTS
    }
    for offset in range(args.episodes_per_layout):
        for layout_index, (name, (layout, _)) in enumerate(
            SOURCE_LAYOUTS.items()
        ):
            seed = 10000 + layout_index * 100000 + offset
            episode = _collect(name, layout, seed, args.collection_steps)
            episodes.append(episode)
            replay.append(episode, Mode.ADAPT)
            row = corpus[name]
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

    cutoff = round(0.75 * args.episodes_per_layout)
    fit_episodes = []
    for name in SOURCE_LAYOUTS:
        selected = [
            episode
            for episode in episodes
            if episode.ruleset == f"push1:{name}"
        ]
        fit_episodes.extend(selected[:cutoff])

    device = torch.device(config.device)
    fit_encoded = _encode_episodes(model, fit_episodes, device)
    random_fit_encoded = _encode_episodes(random_model, fit_episodes, device)
    examples = _hindsight_examples(
        fit_encoded,
        fit_episodes,
        args.policy_horizon,
        terminal_only=True,
    )
    random_examples = _hindsight_examples(
        random_fit_encoded,
        fit_episodes,
        args.policy_horizon,
        terminal_only=True,
    )
    heads, policy_training = _fit_hindsight_policies(
        examples,
        config.z_dim,
        args.policy_updates,
        args.policy_batch_size,
        config.seed + 5145,
    )
    random_heads, random_policy_training = _fit_hindsight_policies(
        random_examples,
        config.z_dim,
        args.policy_updates,
        args.policy_batch_size,
        config.seed + 5145,
    )

    seeds = range(20000, 20000 + args.eval_seeds)
    trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
    try:
        evaluation = {
            "source_geometry": _evaluate(
                model, heads["real"], config, replay, 1, 1, seeds,
                args.eval_steps, deadline, trace, "source_geometry",
            ),
            "target_native_goal": _evaluate(
                model, heads["real"], config, replay, 2, 2, seeds,
                args.eval_steps, deadline, trace, "target_native_goal",
            ),
            "target_canonical_goal": _evaluate(
                model, heads["real"], config, replay, 2, 1, seeds,
                args.eval_steps, deadline, trace, "target_canonical_goal",
            ),
            "target_shuffled_action": _evaluate(
                model, heads["shuffled_action"], config, replay, 2, 1, seeds,
                args.eval_steps, deadline, trace, "target_shuffled_action",
            ),
            "target_random_encoder": _evaluate(
                random_model, random_heads["real"], config, replay, 2, 1,
                seeds, args.eval_steps, deadline, trace,
                "target_random_encoder",
            ),
        }
    finally:
        trace.close()

    source_success = evaluation["source_geometry"]["overall"]["successes"]
    target_success = evaluation["target_canonical_goal"]["overall"][
        "successes"
    ]
    shuffled_success = evaluation["target_shuffled_action"]["overall"][
        "successes"
    ]
    random_success = evaluation["target_random_encoder"]["overall"][
        "successes"
    ]
    terminal_fit = {
        name: sum(
            episode.transitions[-1].terminated
            for episode in fit_episodes
            if episode.ruleset == f"push1:{name}"
        )
        for name in SOURCE_LAYOUTS
    }
    target_layout_floor = min(
        summary["successes"]
        for summary in evaluation["target_canonical_goal"]["by_layout"].values()
    )
    gate = bool(
        all(value > 0 for value in terminal_fit.values())
        and source_success >= 18
        and target_success >= 18
        and target_layout_floor >= 3
        and target_success >= shuffled_success + 4
        and target_success >= random_success + 4
        and target_success / max(source_success, 1) >= 0.75
    )
    result = {
        "status": "completed",
        "claim": (
            "development test of reactive visual-policy transfer across a hidden "
            "physics change; not physics identification or AGI evidence"
        ),
        "physics_transfer_gate": gate,
        "evaluation": evaluation,
        "corpus": {
            "by_layout": corpus,
            "episodes": len(episodes),
            "transitions": sum(len(ep.transitions) for ep in episodes),
            "terminal_fit_episodes_by_layout": terminal_fit,
        },
        "training": {
            "encoder": {
                "termination_weight": config.termination_weight,
                "salient_fraction": config.salient_fraction,
                "updates": args.dynamics_updates,
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "schema_counts": schema_counts,
            },
            "policy": policy_training,
            "random_encoder_policy": random_policy_training,
        },
        "layouts": {
            "source": {
                name: {"layout": repr(layout), "push1": list(actions)}
                for name, (layout, actions) in SOURCE_LAYOUTS.items()
            },
            "target": {
                name: {
                    "layout": repr(layout),
                    "push1": list(push1),
                    "push2": list(push2),
                    "push1_reachable": _canonical_check(
                        layout, 1, push1, 30000
                    ),
                    "push2_reachable": _canonical_check(
                        layout, 2, push2, 30000
                    ),
                    "goal_intervention": _goal_intervention_check(
                        layout, 30000
                    ),
                }
                for name, (layout, push1, push2) in TARGET_LAYOUTS.items()
            },
        },
        "controls": {
            "source_only_training": True,
            "same_policy_examples_initialization_and_batches": True,
            "canonical_goal_uses_push1_pose_for_both_physics": True,
            "native_push2_goal_is_leak_diagnostic_only": True,
            "target_layouts_created_before_training": True,
        },
        "limitations": [
            "policy examples are selected with terminal success labels",
            "reactive policy re-encodes observations and does not identify rules",
            "four handcrafted source and four handcrafted target layouts",
            "one task family, one physics intervention and one training seed",
        ],
    }
    _write(args.out / "results.json", result)
    _write(
        args.out / "manifest.json",
        {
            "argv": sys.argv[1:],
            "budgets": {key: str(value) for key, value in vars(args).items()},
            "status": "completed",
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
