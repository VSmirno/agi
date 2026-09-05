"""Throwaway fixture diagnostic for grid action-label confusion."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product
import json
import os
from pathlib import Path
import sys

import torch

from snks.agent.core_cost import GoalCost
from snks.agent.core_planner import beam_plan
from snks.env.core_types import Mode
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config
from snks.pipeline.core_tasks import make_task
from snks.pipeline.core_transfer import TransferCondition, prepare_transfer


class CanonicalCoverageAgent:
    """Evaluator-known successful fixture trace used only as a capacity control."""

    _ACTIONS = (4,) * 13 + (3, 2, 3)

    def __init__(self, model, config, seed, n_actions):
        if n_actions != 5:
            raise ValueError("canonical push control requires the five-action grid schema")
        self.model = model
        self.config = config
        self.seed = seed
        self.last_model_calls = 0
        self.last_trace = []
        self._index = 0

    def start(self, _obs, _goal):
        self._index = 0
        self.last_trace = []

    def act(self, _exploration_fraction=0.0):
        if self._index >= len(self._ACTIONS):
            raise RuntimeError("canonical action trace exhausted before termination")
        action = self._ACTIONS[self._index]
        self._index += 1
        self.last_trace = [{"canonical_capacity_control": True, "action": action}]
        return action

    def observe(self, _transition):
        return None


def _mse(left, right):
    return float((left - right).square().mean().item())


def _state_at(model, seed, prefix):
    adapter, case = make_task("push_box", "push_1", seed, split="validation", max_steps=16)
    try:
        obs = adapter.reset(seed)
        state = model.initial(obs)
        for action in prefix:
            transition = adapter.step(action)
            predicted = model.step(state, torch.tensor([action], device=state.z.device))
            actual = model.initial(transition.after)
            state = type(state)(actual.z, actual.sensors, actual.sensor_mask,
                                predicted.next_state.hidden.detach(), actual.schema)
        return state, case
    finally:
        adapter.close()


def _real_after(seed, actions, model):
    adapter, _ = make_task("push_box", "push_1", seed, split="validation", max_steps=16)
    try:
        obs = adapter.reset(seed)
        for action in actions:
            transition = adapter.step(action)
            obs = transition.after
        return model.initial(obs).z.detach()
    finally:
        adapter.close()


def _terminal_window(episode, width, fallback):
    terminal = next(
        (index for index, transition in reversed(list(enumerate(episode.transitions)))
         if transition.terminated),
        None,
    )
    if terminal is None:
        return fallback(episode, width)
    start = max(0, terminal - width + 1)
    return replace(episode, transitions=episode.transitions[start:terminal + 1])


@torch.no_grad()
def _confusion(model, seed, prefix, correct_action):
    state, case = _state_at(model, seed, prefix)
    goal_z = model.initial(case.goal.image).z
    goal = GoalCost(goal_z, {})
    predictions = [model.step(state, torch.tensor([action], device=state.z.device))
                   for action in range(5)]
    predicted = [prediction.next_state.z.detach() for prediction in predictions]
    real = [_real_after(seed, (*prefix, action), model) for action in range(5)]
    target_spread = sum(_mse(real[i], real[j]) for i in range(5) for j in range(i + 1, 5)) / 10
    matched = [_mse(predicted[i], real[i]) for i in range(5)]
    mismatched = [_mse(predicted[i], real[j]) for i in range(5) for j in range(5) if i != j]
    rows = []
    for sequence in product(range(5), repeat=3):
        current = state
        total = 0.0
        final_cost = 0.0
        for action in sequence:
            prediction = model.step(current, torch.tensor([action], device=current.z.device))
            final_cost = float(goal(prediction).item())
            total += final_cost
            current = prediction.next_state
        rows.append((total, final_cost, sequence))
    rows.sort(key=lambda row: (row[0], row[2]))
    best_by_first = {str(action): min(total for total, _, seq in rows if seq[0] == action)
                     for action in range(5)}
    final_by_first = {str(action): min(final for _, final, seq in rows if seq[0] == action)
                      for action in range(5)}
    action_costs = sorted((cost, int(action)) for action, cost in best_by_first.items())
    correct_rank = 1 + sum(item < (best_by_first[str(correct_action)], correct_action)
                            for item in action_costs)
    final_action_costs = sorted(
        (cost, int(action)) for action, cost in final_by_first.items()
    )
    correct_final_rank = 1 + sum(
        item < (final_by_first[str(correct_action)], correct_action)
        for item in final_action_costs
    )
    one_step_costs = [float(goal(prediction).item()) for prediction in predictions]
    correct_one_step = (one_step_costs[correct_action], correct_action)
    one_step_rank = 1 + sum(
        item < correct_one_step
        for item in sorted((cost, action) for action, cost in enumerate(one_step_costs))
    )
    plan = beam_plan(model, state, goal, 5, 3, 4, 128)
    return {"prefix": list(prefix), "correct_next_action": correct_action,
            "target_spread": target_spread, "matched_mse": sum(matched) / 5,
            "mismatched_action_label_mse": sum(mismatched) / len(mismatched),
            "per_action_matched_mse": matched, "best_cost_by_first_action": best_by_first,
            "correct_action_rank": correct_rank,
            "best_final_cost_by_first_action": final_by_first,
            "correct_action_final_cost_rank": correct_final_rank,
            "one_step_goal_cost_by_action": one_step_costs,
            "correct_action_one_step_rank": one_step_rank,
            "real_goal_mse_by_action": [_mse(target, goal_z) for target in real],
            "termination_prob_by_action": [
                float(prediction.terminated_prob.item()) for prediction in predictions
            ],
            "beam_plan": {"actions": list(plan.actions),
            "selected_action": plan.actions[0], "correct": plan.actions[0] == correct_action,
            "cost": plan.cost}}


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument(
        "--adaptation-corpus",
        choices=("random", "canonical"),
        default="random",
    )
    parser.add_argument(
        "--target-replay",
        choices=("all", "terminal"),
        default="all",
    )
    parser.add_argument(
        "--window-priority",
        choices=("uniform", "terminal", "balanced"),
        default="uniform",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    requested = load_core_config(args.config)
    source, source_replay, source_config, metadata = core._load_source(args.checkpoint, requested)
    config = core._transfer_training_config(source_config)
    branches = {condition.value: prepare_transfer(
        source, source_replay, condition, "grid-v1", (5, 1), config.seed, config)
        for condition in (TransferCondition.FRESH, TransferCondition.WEIGHTS,
                          TransferCondition.WEIGHTS_REPLAY)}
    trace = core.TraceWriter(args.out / "traces.jsonl")
    deadline = __import__("time").monotonic() + args.max_seconds
    original_random_agent = core.RandomActionAgent
    if args.adaptation_corpus == "canonical":
        core.RandomActionAgent = CanonicalCoverageAgent
    try:
        adapt_results, episodes = core._run_cases(
            model=branches["fresh"][0], config=config,
            replay=__import__("snks.learning.core_replay", fromlist=["SequenceReplay"]).SequenceReplay(
                config.replay_capacity, config.seed + 71), family="push_box", ruleset="push_1",
            seeds=range(10000, 10000 + args.episodes), split="adapt", mode=Mode.ADAPT,
            steps=args.steps, deadline=deadline, trace=trace,
            role=f"diagnostic/shared_{args.adaptation_corpus}_B_adaptation",
            random_actions=True)
    finally:
        core.RandomActionAgent = original_random_agent
        trace.close()
    corpus = {"mode": args.adaptation_corpus,
              "oracle_capacity_control": args.adaptation_corpus == "canonical",
              "episodes": len(episodes), "steps": sum(x.steps for x in adapt_results),
              "actions": {}, "box_moves": 0, "successes": sum(x.success for x in adapt_results)}
    for line in (args.out / "traces.jsonl").read_text().splitlines():
        record = json.loads(line)
        for before, after in zip(record["audit"], record["audit"][1:]):
            action = after.get("action")
            corpus["actions"][str(action)] = corpus["actions"].get(str(action), 0) + 1
            if before.get("diagnostic", {}).get("box_pos") != after.get("diagnostic", {}).get("box_pos"):
                corpus["box_moves"] += 1
    training_episodes = episodes
    if args.target_replay == "terminal":
        training_episodes = [
            episode for episode in episodes if episode.transitions[-1].terminated
        ]
        if not training_episodes:
            raise RuntimeError("terminal-priority control found no terminated B episodes")
    corpus["target_replay"] = args.target_replay
    corpus["window_priority"] = args.window_priority
    corpus["training_episodes"] = len(training_episodes)
    snapshots = {}
    evaluation_trace = core.TraceWriter(args.out / "evaluation_traces.jsonl")
    try:
        for name, (model, trainer, replay) in branches.items():
            snapshots[name] = {"checkpoint0": [_confusion(model, 20000, (), 3),
                                                _confusion(model, 20000, (3,), 2),
                                                _confusion(model, 20000, (3, 2), 3)]}
            for episode in training_episodes:
                replay.append(episode, Mode.ADAPT)
            if args.window_priority == "terminal":
                uniform_window = replay._window

                def terminal_window(episode, width, fallback=uniform_window):
                    return _terminal_window(episode, width, fallback)

                replay._window = terminal_window
            elif args.window_priority == "balanced":
                terminal_episodes = [
                    episode for episode in episodes if episode.transitions[-1].terminated
                ]
                if not terminal_episodes:
                    raise RuntimeError("balanced replay found no terminated B episodes")
                uniform_sample = replay.sample
                uniform_window = replay._window

                def balanced_sample(batch_size, length, burn_in, recent_fraction,
                                    schema=None, base=uniform_sample,
                                    window=uniform_window, events=terminal_episodes,
                                    rng=replay._rng):
                    sampled = base(batch_size, length, burn_in, recent_fraction, schema)
                    if not sampled or sampled[0].transitions[0].before.schema != "grid-v1":
                        return sampled
                    for index in range(max(1, batch_size // 2)):
                        event = events[int(rng.integers(len(events)))]
                        sampled[index] = _terminal_window(event, length + burn_in, window)
                    return sampled

                replay.sample = balanced_sample
            metrics, schema_counts = core._train_updates(
                model, trainer, replay, config, args.updates, Mode.ADAPT, deadline,
                schema=None if name == "weights_replay" else "grid-v1",
                require_mixed=("crafter-v1", "grid-v1") if name == "weights_replay" else None)
            snapshots[name]["checkpoint20"] = [_confusion(model, 20000, (), 3),
                                                 _confusion(model, 20000, (3,), 2),
                                                 _confusion(model, 20000, (3, 2), 3)]
            components = []
            for before, after in zip(snapshots[name]["checkpoint0"], snapshots[name]["checkpoint20"]):
                components.append({
                    "target_spread_retained_50pct": after["target_spread"] >= 0.5 * before["target_spread"],
                    "matched_better_than_mismatched": after["matched_mse"] < after["mismatched_action_label_mse"],
                    "correct_rank1": after["correct_action_rank"] == 1,
                })
            snapshots[name]["diagnostic_pass_components"] = components
            snapshots[name]["diagnostic_pass"] = all(all(row.values()) for row in components)
            snapshots[name]["updates"] = {"losses": metrics, "schema_counts": schema_counts}
            snapshots[name]["end_to_end"] = core._evaluate(
                model, config, replay, "push_box", "push_1",
                range(20000, 20000 + args.eval_episodes), args.steps, deadline,
                evaluation_trace, f"diagnostic/{name}/B_after",
            )
    finally:
        evaluation_trace.close()
    if args.adaptation_corpus == "canonical":
        scope = "evaluator-known fixed-fixture oracle/capacity control; not candidate experience"
    elif args.target_replay == "terminal":
        scope = "fixed-fixture terminal-prioritized replay diagnostic on natural random experience"
    else:
        scope = "fixed-fixture random-experience diagnostic"
    result = {"status": "completed", "diagnostic_scope": scope,
              "corpus": corpus, "snapshots": snapshots,
              "limitations": ["single seed", "fixed layout",
                              "diagnostic is not candidate transfer evidence",
                              "Grid termination denotes success in this fixture"],
              "checkpoint_sha256": core._file_hash(args.checkpoint),
              "source_metadata": metadata}
    _write(args.out / "results.json", result)
    _write(args.out / "manifest.json", {"argv": sys.argv[1:],
                                         "budgets": {key: str(value) for key, value in vars(args).items()},
                                         "status": "completed"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
