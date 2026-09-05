"""Diagnostic: can ordered experience learn a better image-goal score than latent MSE?

This intentionally learns policy-dependent temporal proximity, not optimal
reachability.  The probe sees only frozen latent pairs and their within-episode
order; rewards, task success, rules and evaluator diagnostics are excluded.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from dataclasses import replace
import json
from itertools import product
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from snks.env.core_types import Episode, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.agent.core_world_model import CoreWorldModel, LatentState, Prediction
from snks.agent.core_agent import CoreAgent
from snks.encoder.core_encoder import CoreEncoder
from snks.pipeline.core_runner import run_episode
from snks.pipeline import core_experiment as core
from snks.pipeline.core_config import load_core_config
from snks.pipeline.core_tasks import make_task
from snks.pipeline.core_transfer import TransferCondition, prepare_transfer


@dataclass(frozen=True)
class PairSet:
    anchor: Tensor
    target: Tensor
    horizon: Tensor
    label: Tensor

    def __len__(self) -> int:
        return len(self.label)


class TemporalProbe(nn.Module):
    """Directed P(observed target occurs within H steps | anchor, dataset policy)."""

    def __init__(self, z_dim: int, width: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3 * z_dim + 1, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )

    def forward(self, anchor: Tensor, target: Tensor, horizon: Tensor) -> Tensor:
        features = torch.cat(
            (anchor, target, (anchor - target).abs(), horizon[:, None]), dim=-1
        )
        return self.network(features).squeeze(-1)


class ResidualCorrectionModel(nn.Module):
    """Zero-init correction over a frozen absolute-latent world model."""

    def __init__(self, base):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.base.requires_grad_(False)
        self.encoder = self.base.encoder
        self.h_dim = self.base.h_dim
        self.heads = self.base.heads
        self.schemas = self.base.schemas
        self.sensor_heads = self.base.sensor_heads
        self.termination_heads = self.base.termination_heads
        self.predict_sensor_delta = self.base.predict_sensor_delta
        self.correction_heads = nn.ModuleList(
            nn.Linear(self.h_dim, self.encoder.z_dim) for _ in range(self.heads)
        )
        with torch.no_grad():
            for head in self.correction_heads:
                head.weight.zero_()
                head.bias.zero_()

    def initial(self, observation):
        return self.base.initial(observation)

    def step(self, state: LatentState, actions: Tensor) -> Prediction:
        prediction = self.base.step(state, actions)
        correction = torch.stack(
            [head(prediction.next_state.hidden) for head in self.correction_heads]
        )
        member_z = prediction.member_z + correction
        next_state = LatentState(
            member_z.mean(0),
            prediction.next_state.sensors,
            prediction.next_state.sensor_mask,
            prediction.next_state.hidden,
            prediction.next_state.schema,
        )
        uncertainty = member_z.var(0, unbiased=False).mean(-1)
        return Prediction(
            next_state, prediction.terminated_prob, uncertainty, member_z
        )

    def rollout(self, state: LatentState, actions: Tensor):
        predictions = []
        for action in actions.unbind(1):
            prediction = self.step(state, action)
            predictions.append(prediction)
            state = prediction.next_state
        return predictions


class ActionContrastiveTrainer(CoreTrainer):
    """Auxiliary JEPA-style ranking of the observed action's next latent."""

    def __init__(self, model, config, weight: float, shuffled_labels: bool):
        super().__init__(model, config)
        self.weight = weight
        self.shuffled_labels = shuffled_labels
        generator = torch.Generator().manual_seed(config.seed + 4143)
        permutation = torch.randperm(5, generator=generator)
        if torch.equal(permutation, torch.arange(5)):
            permutation = permutation.roll(1)
        self.label_permutation = permutation
        self.last_action_loss = float("nan")
        self.last_changed_fraction = 0.0

    def compute_loss(self, batch):
        base_loss = super().compute_loss(batch)
        valid = batch.valid[:, 0]
        changed = (batch.rgb[:, 0] != batch.rgb[:, 1]).flatten(1).any(-1)
        selected = valid & changed
        self.last_changed_fraction = float(selected.float().mean())
        if not selected.any():
            self.last_action_loss = 0.0
            return base_loss
        state = self.model.initial_from_tensors(
            batch.rgb[:, 0], batch.sensors[:, 0], batch.sensor_mask[:, 0], batch.schema
        )
        target = self.model.encoder(batch.rgb[:, 1])
        n_actions = self.model.schemas[batch.schema][0]
        expanded = LatentState(
            state.z.repeat_interleave(n_actions, dim=0),
            state.sensors.repeat_interleave(n_actions, dim=0),
            state.sensor_mask.repeat_interleave(n_actions, dim=0),
            state.hidden.repeat_interleave(n_actions, dim=0),
            state.schema,
        )
        candidates = torch.arange(n_actions, device=state.z.device).repeat(len(state.z))
        predicted = self.model.step(expanded, candidates).next_state.z
        distance = (predicted - target.repeat_interleave(n_actions, dim=0)).square()
        logits = -distance.mean(-1).reshape(len(state.z), n_actions) / 0.1
        labels = batch.actions[:, 0]
        if self.shuffled_labels:
            labels = self.label_permutation.to(labels.device)[labels]
        action_loss = F.cross_entropy(logits[selected], labels[selected])
        self.last_action_loss = float(action_loss.detach())
        return base_loss + self.weight * action_loss


class TemporalCost:
    def __init__(self, probe: TemporalProbe, goal_z: Tensor):
        self.probe = probe
        self.goal_z = goal_z.detach().clone()

    def __call__(self, prediction: Prediction):
        state = prediction.next_state.z
        # Ask the fixed probe whether the goal is observed within its full H.
        horizon = torch.ones(len(state), device=state.device)
        return -self.probe(state, self.goal_z.expand(len(state), -1), horizon)


class TemporalAgent(CoreAgent):
    def __init__(self, model, config, probe):
        super().__init__(model, config)
        self.probe = probe

    @torch.no_grad()
    def start(self, obs, goal):
        super().start(obs, goal)
        if goal.image is None:
            raise ValueError("temporal diagnostic requires an image goal")
        self.cost = TemporalCost(self.probe, self.model.initial(goal.image).z)


def _observations(episode: Episode):
    transitions = episode.transitions
    return [transitions[0].before, *[transition.after for transition in transitions]]


@torch.no_grad()
def _encode_episodes(model, episodes: list[Episode], device: torch.device):
    encoded = []
    for episode in episodes:
        observations = _observations(episode)
        rgb = torch.as_tensor(
            np.stack([observation.rgb for observation in observations]),
            device=device,
        ).float() / 255
        encoded.append(model.encoder(rgb).detach())
    return encoded


def _pairs(encoded: list[Tensor], max_horizon: int) -> PairSet:
    anchors, targets, horizons, labels = [], [], [], []
    for states in encoded:
        # d=0 teaches that an exact state-goal match is reachable immediately.
        # d>H is an observed "not within H" label, never a claim of impossibility.
        for start in range(len(states)):
            for distance in range(0, min(2 * max_horizon, len(states) - start - 1) + 1):
                for horizon in range(1, max_horizon + 1):
                    anchors.append(states[start])
                    targets.append(states[start + distance])
                    horizons.append(horizon / max_horizon)
                    labels.append(float(distance <= horizon))
    if not labels:
        raise RuntimeError("temporal probe has no within-episode pairs")
    return PairSet(
        torch.stack(anchors),
        torch.stack(targets),
        torch.tensor(horizons, device=anchors[0].device),
        torch.tensor(labels, device=anchors[0].device),
    )


@torch.no_grad()
def _imagined_anchors(model, episodes: list[Episode], encoded: list[Tensor]):
    """One-step model outputs under real histories, aligned to real next states."""
    imagined = []
    for episode, real_states in zip(episodes, encoded, strict=True):
        observations = _observations(episode)
        state = model.initial(observations[0])
        episode_imagined = []
        for index, transition in enumerate(episode.transitions):
            action = torch.tensor([transition.action], device=state.z.device)
            prediction = model.step(state, action)
            episode_imagined.append(prediction.next_state.z.squeeze(0).detach())
            actual = model.initial(observations[index + 1])
            state = type(state)(
                real_states[index + 1:index + 2],
                actual.sensors,
                actual.sensor_mask,
                prediction.next_state.hidden.detach(),
                actual.schema,
            )
        imagined.append(torch.stack(episode_imagined))
    return imagined


def _imagined_pairs(real: list[Tensor], imagined: list[Tensor], max_horizon: int):
    anchors, targets, horizons, labels = [], [], [], []
    for real_states, imagined_states in zip(real, imagined, strict=True):
        for anchor_index in range(len(imagined_states)):
            real_index = anchor_index + 1
            maximum = min(2 * max_horizon, len(real_states) - real_index - 1)
            for distance in range(maximum + 1):
                for horizon in range(1, max_horizon + 1):
                    anchors.append(imagined_states[anchor_index])
                    targets.append(real_states[real_index + distance])
                    horizons.append(horizon / max_horizon)
                    labels.append(float(distance <= horizon))
    return PairSet(
        torch.stack(anchors),
        torch.stack(targets),
        torch.tensor(horizons, device=anchors[0].device),
        torch.tensor(labels, device=anchors[0].device),
    )


def _combine(left: PairSet, right: PairSet) -> PairSet:
    return PairSet(*(torch.cat((getattr(left, field), getattr(right, field)))
                     for field in ("anchor", "target", "horizon", "label")))


def _train_residual_controls(
    model,
    replay,
    config,
    updates: int,
    seed: int,
    deadline: float,
):
    device = torch.device(config.device)
    real = ResidualCorrectionModel(model).to(device)
    shuffled = ResidualCorrectionModel(model).to(device)
    shuffled.correction_heads.load_state_dict(real.correction_heads.state_dict())
    real_trainer = CoreTrainer(real, config)
    shuffled_trainer = CoreTrainer(shuffled, config)
    generator = torch.Generator().manual_seed(seed)
    action_permutation = torch.randperm(5, generator=generator)
    if torch.equal(action_permutation, torch.arange(5)):
        action_permutation = action_permutation.roll(1)
    losses = {"real_actions": [], "shuffled_actions": []}
    for index in range(updates):
        core._check_deadline(deadline, f"residual correction update {index}")
        samples = replay.sample(
            config.batch_size,
            config.train_horizon,
            config.burn_in,
            config.recent_fraction,
            schema="grid-v1",
            salient_fraction=config.salient_fraction,
        )
        batch = tensorize(samples, config.burn_in, device)
        losses["real_actions"].append(real_trainer.update(batch, Mode.ADAPT)["loss"])
        shuffled_actions = action_permutation[batch.actions.cpu()].to(device)
        shuffled_batch = replace(batch, actions=shuffled_actions)
        losses["shuffled_actions"].append(
            shuffled_trainer.update(shuffled_batch, Mode.ADAPT)["loss"]
        )
    real.eval()
    shuffled.eval()
    return real, shuffled, {
        "action_permutation": action_permutation.tolist(),
        "losses": {
            name: {"first": values[0], "last": values[-1]}
            for name, values in losses.items()
        },
    }


@torch.no_grad()
def _rollout_errors(model, episodes: list[Episode], horizons=(1, 3)):
    squared = {horizon: [] for horizon in horizons}
    persistence = {horizon: [] for horizon in horizons}
    for episode in episodes:
        observations = _observations(episode)
        if len(episode.transitions) < max(horizons):
            continue
        state = model.initial(observations[0])
        actions = torch.tensor(
            [[transition.action for transition in episode.transitions[:max(horizons)]]],
            device=state.z.device,
        )
        predictions = model.rollout(state, actions)
        for horizon in horizons:
            target = model.initial(observations[horizon]).z
            squared[horizon].append(float(
                (predictions[horizon - 1].next_state.z - target).square().mean()
            ))
            persistence[horizon].append(float((state.z - target).square().mean()))
    return {
        f"H{horizon}": {
            "mse": sum(values) / len(values),
            "persistence_mse": sum(persistence[horizon]) / len(persistence[horizon]),
        }
        for horizon, values in squared.items()
    }


def _shuffled_targets(pairs: PairSet, generator: torch.Generator) -> Tensor:
    permutation = torch.randperm(len(pairs), generator=generator, device="cpu")
    if len(permutation) > 1 and torch.equal(permutation, torch.arange(len(permutation))):
        permutation = permutation.roll(1)
    return pairs.target[permutation.to(pairs.target.device)]


def _fit_pair(
    real: TemporalProbe,
    shuffled: TemporalProbe,
    pairs: PairSet,
    updates: int,
    batch_size: int,
    seed: int,
):
    real.train()
    shuffled.train()
    real_optimizer = torch.optim.Adam(real.parameters(), lr=1e-3)
    shuffled_optimizer = torch.optim.Adam(shuffled.parameters(), lr=1e-3)
    cpu_generator = torch.Generator().manual_seed(seed)
    shuffled_target = _shuffled_targets(pairs, cpu_generator)
    positives = float(pairs.label.sum().item())
    negatives = len(pairs) - positives
    pos_weight = torch.tensor(negatives / max(positives, 1.0), device=pairs.label.device)
    losses = {"ordered": [], "shuffled_endpoint": []}
    for _ in range(updates):
        indices = torch.randint(
            len(pairs), (batch_size,), generator=cpu_generator, device="cpu"
        ).to(pairs.label.device)
        for name, probe, optimizer, targets in (
            ("ordered", real, real_optimizer, pairs.target),
            ("shuffled_endpoint", shuffled, shuffled_optimizer, shuffled_target),
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = probe(
                pairs.anchor[indices], targets[indices], pairs.horizon[indices]
            )
            loss = F.binary_cross_entropy_with_logits(
                logits, pairs.label[indices], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            losses[name].append(float(loss.detach()))
    real.eval()
    shuffled.eval()
    return {name: {"first": values[0], "last": values[-1]}
            for name, values in losses.items()}


@torch.no_grad()
def _probe_metrics(probe: TemporalProbe, pairs: PairSet) -> dict[str, float]:
    logits = probe(pairs.anchor, pairs.target, pairs.horizon)
    loss = F.binary_cross_entropy_with_logits(logits, pairs.label)
    prediction = logits >= 0
    positive = pairs.label.bool()
    negative = ~positive
    tpr = (prediction[positive].float().mean() if positive.any()
           else logits.new_tensor(float("nan")))
    tnr = ((~prediction[negative]).float().mean() if negative.any()
           else logits.new_tensor(float("nan")))
    return {
        "bce": float(loss),
        "balanced_accuracy": float((tpr + tnr) / 2),
        "mean_positive_logit": float(logits[positive].mean()),
        "mean_negative_logit": float(logits[negative].mean()),
    }


def _state_at(model, seed: int, prefix: tuple[int, ...]):
    adapter, case = make_task("push_box", "push_1", seed, "validation", 16)
    try:
        state = model.initial(adapter.reset(seed))
        for action in prefix:
            transition = adapter.step(action)
            predicted = model.step(
                state, torch.tensor([action], device=state.z.device)
            )
            actual = model.initial(transition.after)
            state = type(state)(
                actual.z,
                actual.sensors,
                actual.sensor_mask,
                predicted.next_state.hidden.detach(),
                actual.schema,
            )
        return state, case
    finally:
        adapter.close()


def _actual_forks(model, seed: int, prefix: tuple[int, ...]) -> Tensor:
    outcomes = []
    for candidate in range(5):
        adapter, _ = make_task("push_box", "push_1", seed, "validation", 16)
        try:
            observation = adapter.reset(seed)
            for action in (*prefix, candidate):
                observation = adapter.step(action).after
            outcomes.append(model.initial(observation).z.squeeze(0))
        finally:
            adapter.close()
    return torch.stack(outcomes)


def _rank(costs: list[float], correct: int) -> float:
    chosen = costs[correct]
    lower = sum(cost < chosen and not math.isclose(cost, chosen, rel_tol=1e-7, abs_tol=1e-9)
                for cost in costs)
    tied = sum(math.isclose(cost, chosen, rel_tol=1e-7, abs_tol=1e-9)
               for index, cost in enumerate(costs) if index != correct)
    return 1.0 + lower + 0.5 * tied


@torch.no_grad()
def _decision_rows(model, ordered, shuffled, seeds: range, max_horizon: int):
    decisions = (((), 3, 2), ((3,), 2, 1), ((3, 2), 3, 1))
    rows = []
    for seed in seeds:
        for prefix, correct, remaining in decisions:
            state, case = _state_at(model, seed, prefix)
            goal = model.initial(case.goal.image).z
            actual = _actual_forks(model, seed, prefix)
            actions = torch.arange(5, device=state.z.device)
            predicted = model.step(
                type(state)(
                    state.z.expand(5, -1),
                    state.sensors.expand(5, -1),
                    state.sensor_mask.expand(5, -1),
                    state.hidden.expand(5, -1),
                    state.schema,
                ),
                actions,
            ).next_state.z
            normalized_horizon = torch.full(
                (5,), remaining / max_horizon, device=state.z.device
            )
            goal_batch = goal.expand(5, -1)
            scores = {}
            for domain, outcomes in (("actual", actual), ("predicted", predicted)):
                scores[f"{domain}_latent_mse"] = (
                    (outcomes - goal_batch).square().mean(-1).cpu().tolist()
                )
                scores[f"{domain}_ordered"] = (
                    ordered(outcomes, goal_batch, normalized_horizon).neg().cpu().tolist()
                )
                scores[f"{domain}_shuffled"] = (
                    shuffled(outcomes, goal_batch, normalized_horizon).neg().cpu().tolist()
                )
            rows.append({
                "seed": seed,
                "prefix": list(prefix),
                "correct_action": correct,
                "remaining_horizon": remaining,
                "costs": scores,
                "ranks": {name: _rank(costs, correct) for name, costs in scores.items()},
            })
    return rows


def _aggregate(rows):
    names = sorted(rows[0]["ranks"])
    result = {}
    for name in names:
        ranks = [row["ranks"][name] for row in rows]
        result[name] = {
            "mean_rank": sum(ranks) / len(ranks),
            "mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
            "rank1_fraction": sum(rank == 1.0 for rank in ranks) / len(ranks),
            "by_prefix_mrr": {},
        }
        prefixes = sorted({tuple(row["prefix"]) for row in rows})
        for prefix in prefixes:
            selected = [row["ranks"][name] for row in rows
                        if tuple(row["prefix"]) == prefix]
            result[name]["by_prefix_mrr"][str(list(prefix))] = (
                sum(1.0 / rank for rank in selected) / len(selected)
            )
    return result


@torch.no_grad()
def _exhaustive_rows(model, ordered, shuffled, seeds: range, max_horizon: int):
    decisions = (((), 3, 3), ((3,), 2, 2), ((3, 2), 3, 1))
    rows = []
    for seed in seeds:
        for prefix, correct, rollout_horizon in decisions:
            state, case = _state_at(model, seed, prefix)
            goal = model.initial(case.goal.image).z
            by_metric = {name: [[] for _ in range(5)] for name in (
                "predicted_latent_mse", "predicted_ordered", "predicted_shuffled",
                "actual_latent_mse", "actual_ordered", "actual_shuffled",
            )}
            for sequence in product(range(5), repeat=rollout_horizon):
                current = state
                for action in sequence:
                    prediction = model.step(
                        current, torch.tensor([action], device=current.z.device)
                    )
                    current = prediction.next_state
                predicted_z = current.z
                adapter, _ = make_task(
                    "push_box", "push_1", seed, "validation", 16
                )
                try:
                    observation = adapter.reset(seed)
                    for action in (*prefix, *sequence):
                        transition = adapter.step(action)
                        observation = transition.after
                        if transition.terminated or transition.truncated:
                            break
                    actual_z = model.initial(observation).z
                finally:
                    adapter.close()
                normalized = torch.tensor(
                    [1 / max_horizon], device=state.z.device
                )
                for domain, outcome in (("predicted", predicted_z), ("actual", actual_z)):
                    by_metric[f"{domain}_latent_mse"][sequence[0]].append(float(
                        (outcome - goal).square().mean()
                    ))
                    by_metric[f"{domain}_ordered"][sequence[0]].append(float(
                        -ordered(outcome, goal, normalized)
                    ))
                    by_metric[f"{domain}_shuffled"][sequence[0]].append(float(
                        -shuffled(outcome, goal, normalized)
                    ))
            best = {name: [min(values) for values in per_action]
                    for name, per_action in by_metric.items()}
            rows.append({
                "seed": seed,
                "prefix": list(prefix),
                "correct_action": correct,
                "rollout_horizon": rollout_horizon,
                "best_final_cost_by_first_action": best,
                "ranks": {name: _rank(costs, correct) for name, costs in best.items()},
            })
    return rows


def _write(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _closed_loop(model, probe, config, replay, seeds, steps, deadline, trace, role):
    results = []
    for seed in seeds:
        core._check_deadline(deadline, f"{role} episode {seed}")
        adapter, case = make_task(
            "push_box", "push_1", seed, "validation", steps
        )
        try:
            episode_config = replace(config, seed=seed)
            agent = (CoreAgent(model, episode_config) if probe is None
                     else TemporalAgent(model, episode_config, probe))
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
        trace.write({
            "role": role,
            "seed": seed,
            **core._result_record(result),
            "audit": result.audit,
        })
    return core._summarize_episodes(results)


def _observable_salient_transitions(episodes, schema, burn_in):
    """Diagnostic salience: termination or any directly observed state change."""
    salient = []
    for episode in episodes:
        if episode.transitions[0].before.schema != schema:
            continue
        for index, transition in enumerate(episode.transitions):
            if index < burn_in:
                continue
            mask = transition.before.sensor_mask & transition.after.sensor_mask
            sensor_changed = bool(np.any(
                transition.before.sensors[mask] != transition.after.sensors[mask]
            ))
            visual_changed = not np.array_equal(
                transition.before.rgb, transition.after.rgb
            )
            if transition.terminated or sensor_changed or visual_changed:
                salient.append((episode, index))
    return salient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--dynamics-updates", type=int, default=1000)
    parser.add_argument("--probe-updates", type=int, default=400)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--eval-seeds", type=int, default=12)
    parser.add_argument("--max-horizon", type=int, default=3)
    parser.add_argument("--probe-domain", choices=("real", "mixed"), default="real")
    parser.add_argument("--visual-salience", action="store_true")
    parser.add_argument("--residual-correction-updates", type=int, default=0)
    parser.add_argument("--action-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--shuffled-action-contrastive", action="store_true")
    parser.add_argument("--fresh-z-dim", type=int)
    parser.add_argument("--fresh-h-dim", type=int)
    parser.add_argument("--closed-loop-eval", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=360)
    args = parser.parse_args(argv)
    if args.episodes < 4 or args.eval_seeds < 1 or args.max_horizon < 1:
        raise ValueError("episodes >= 4, eval-seeds and max-horizon must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + args.max_seconds
    requested = load_core_config(args.config)
    source, source_replay, source_config, metadata = core._load_source(
        args.checkpoint, requested
    )
    config = core._transfer_training_config(source_config)
    if args.fresh_z_dim or args.fresh_h_dim:
        config = replace(
            config,
            z_dim=args.fresh_z_dim or config.z_dim,
            h_dim=args.fresh_h_dim or config.h_dim,
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
        replay = SequenceReplay(config.replay_capacity, config.seed)
    else:
        model, trainer, replay = prepare_transfer(
            source,
            source_replay,
            TransferCondition.FRESH,
            "grid-v1",
            (5, 1),
            config.seed,
            config,
        )
    if args.action_contrastive_weight:
        trainer = ActionContrastiveTrainer(
            model,
            config,
            args.action_contrastive_weight,
            args.shuffled_action_contrastive,
        )
    trace = core.TraceWriter(args.out / "traces.jsonl")
    try:
        adaptation, episodes = core._run_cases(
            model=model,
            config=config,
            replay=SequenceReplay(config.replay_capacity, config.seed + 143),
            family="push_box",
            ruleset="push_1",
            seeds=range(10000, 10000 + args.episodes),
            split="adapt",
            mode=Mode.ADAPT,
            steps=args.steps,
            deadline=deadline,
            trace=trace,
            role="diagnostic/natural_random_B_adaptation",
            random_actions=True,
        )
    finally:
        trace.close()
    for episode in episodes:
        replay.append(episode, Mode.ADAPT)
    if args.visual_salience:
        replay._salient_transitions = _observable_salient_transitions
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

    split = round(0.75 * len(episodes))
    train_episodes, validation_episodes = episodes[:split], episodes[split:]
    device = torch.device(config.device)
    train_encoded = _encode_episodes(model, train_episodes, device)
    validation_encoded = _encode_episodes(model, validation_episodes, device)
    train_pairs = _pairs(train_encoded, args.max_horizon)
    validation_pairs = _pairs(validation_encoded, args.max_horizon)
    if args.probe_domain == "mixed":
        train_imagined = _imagined_anchors(model, train_episodes, train_encoded)
        validation_imagined = _imagined_anchors(
            model, validation_episodes, validation_encoded
        )
        train_pairs = _combine(
            train_pairs,
            _imagined_pairs(train_encoded, train_imagined, args.max_horizon),
        )
        validation_pairs = _combine(
            validation_pairs,
            _imagined_pairs(
                validation_encoded, validation_imagined, args.max_horizon
            ),
        )
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(config.seed + 143)
        ordered = TemporalProbe(config.z_dim).to(device)
        shuffled = TemporalProbe(config.z_dim).to(device)
        shuffled.load_state_dict(ordered.state_dict())
    probe_losses = _fit_pair(
        ordered,
        shuffled,
        train_pairs,
        args.probe_updates,
        args.probe_batch_size,
        config.seed + 143,
    )
    validation = {
        "ordered": _probe_metrics(ordered, validation_pairs),
        "shuffled_endpoint": _probe_metrics(shuffled, validation_pairs),
    }
    rows = _decision_rows(
        model,
        ordered,
        shuffled,
        range(20000, 20000 + args.eval_seeds),
        args.max_horizon,
    )
    aggregate = _aggregate(rows)
    exhaustive_rows = _exhaustive_rows(
        model,
        ordered,
        shuffled,
        range(20000, 20000 + args.eval_seeds),
        args.max_horizon,
    )
    exhaustive_aggregate = _aggregate(exhaustive_rows)
    residual_result = None
    if args.residual_correction_updates:
        residual_real, residual_shuffled, residual_training = _train_residual_controls(
            model,
            replay,
            config,
            args.residual_correction_updates,
            config.seed + 3143,
            deadline,
        )
        residual_real_rows = _decision_rows(
            residual_real,
            ordered,
            shuffled,
            range(20000, 20000 + args.eval_seeds),
            args.max_horizon,
        )
        residual_shuffled_rows = _decision_rows(
            residual_shuffled,
            ordered,
            shuffled,
            range(20000, 20000 + args.eval_seeds),
            args.max_horizon,
        )
        real_aggregate = _aggregate(residual_real_rows)
        shuffled_aggregate = _aggregate(residual_shuffled_rows)
        base_metric = aggregate["predicted_ordered"]
        real_metric = real_aggregate["predicted_ordered"]
        shuffled_metric = shuffled_aggregate["predicted_ordered"]
        key_root, key_middle = "[]", "[3]"
        ranking_pass = (
            real_metric["mrr"] >= base_metric["mrr"] + 0.05
            and real_metric["mrr"] >= shuffled_metric["mrr"] + 0.05
            and all(
                real_metric["by_prefix_mrr"][key] > base_metric["by_prefix_mrr"][key]
                and real_metric["by_prefix_mrr"][key]
                > shuffled_metric["by_prefix_mrr"][key]
                for key in (key_root, key_middle)
            )
        )
        rollout_errors = {
            "base": _rollout_errors(model, validation_episodes),
            "real_actions": _rollout_errors(residual_real, validation_episodes),
            "shuffled_actions": _rollout_errors(
                residual_shuffled, validation_episodes
            ),
        }
        dynamics_pass = all(
            rollout_errors["real_actions"][horizon]["mse"]
            < rollout_errors[control][horizon]["mse"]
            for horizon in ("H1", "H3")
            for control in ("base", "shuffled_actions")
        )
        residual_result = {
            **residual_training,
            "updates": args.residual_correction_updates,
            "zero_initialized": True,
            "base_frozen": True,
            "rollout_errors": rollout_errors,
            "aggregate": {
                "base": aggregate,
                "real_actions": real_aggregate,
                "shuffled_actions": shuffled_aggregate,
            },
            "gates": {
                "heldout_H1_H3_better_than_controls": dynamics_pass,
                "predicted_ranking_better_than_controls": ranking_pass,
                "pass": dynamics_pass and ranking_pass,
            },
        }
    primary = aggregate["actual_ordered"]
    raw = aggregate["actual_latent_mse"]
    control = aggregate["actual_shuffled"]
    improved_groups = sum(
        primary["by_prefix_mrr"][key] > raw["by_prefix_mrr"][key]
        for key in primary["by_prefix_mrr"]
    )
    f1_actual_outcome_pass = (
        primary["mrr"] > raw["mrr"]
        and primary["mrr"] > control["mrr"]
        and improved_groups >= 2
    )
    predicted = aggregate["predicted_ordered"]
    predicted_raw = aggregate["predicted_latent_mse"]
    predicted_control = aggregate["predicted_shuffled"]
    root_key, middle_key = "[]", "[3]"
    f2_imagined_outcome_pass = (
        predicted["mrr"] >= predicted_raw["mrr"] + 0.05
        and predicted["mrr"] >= predicted_control["mrr"] + 0.05
        and predicted["by_prefix_mrr"][root_key]
        >= predicted_raw["by_prefix_mrr"][root_key] + 0.05
        and predicted["by_prefix_mrr"][root_key]
        >= predicted_control["by_prefix_mrr"][root_key] + 0.05
        and predicted["by_prefix_mrr"][middle_key]
        >= predicted_raw["by_prefix_mrr"][middle_key]
    )
    terminal_future_pairs = 0
    for episode in train_episodes:
        terminal_index = next(
            (index + 1 for index, transition in enumerate(episode.transitions)
             if transition.terminated),
            None,
        )
        if terminal_index is not None:
            terminal_future_pairs += min(2 * args.max_horizon, terminal_index) + 1
    result = {
        "status": "completed",
        "claim": "policy-dependent directed temporal proximity; not optimal reachability",
        "diagnostic_pass": f1_actual_outcome_pass and f2_imagined_outcome_pass,
        "gates": {
            "F1_actual_outcomes": f1_actual_outcome_pass,
            "F2_imagined_outcomes": f2_imagined_outcome_pass,
        },
        "decision_rule": (
            "ordered actual-outcome MRR must exceed latent MSE and shuffled endpoint; "
            "must improve at least two of three prefix groups"
        ),
        "aggregate": aggregate,
        "decision_rows": rows,
        "exhaustive_final_state": {
            "aggregate": exhaustive_aggregate,
            "decision_rows": exhaustive_rows,
            "scope": "diagnostic oracle search without beam pruning",
        },
        "probe": {
            "train_episodes": len(train_episodes),
            "validation_episodes": len(validation_episodes),
            "domain": args.probe_domain,
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "terminal_endpoint_anchor_pairs_before_horizon_expansion": terminal_future_pairs,
            "losses": probe_losses,
            "validation": validation,
        },
        "residual_correction": residual_result,
        "dynamics": {
            "z_dim": config.z_dim,
            "h_dim": config.h_dim,
            "updates": args.dynamics_updates,
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "schema_counts": schema_counts,
            "visual_salience_diagnostic": args.visual_salience,
            "action_contrastive": {
                "weight": args.action_contrastive_weight,
                "shuffled_labels": args.shuffled_action_contrastive,
                "label_permutation": (
                    trainer.label_permutation.tolist()
                    if isinstance(trainer, ActionContrastiveTrainer) else None
                ),
                "last_action_loss": (
                    trainer.last_action_loss
                    if isinstance(trainer, ActionContrastiveTrainer) else None
                ),
                "last_changed_fraction": (
                    trainer.last_changed_fraction
                    if isinstance(trainer, ActionContrastiveTrainer) else None
                ),
            },
        },
        "corpus": {
            "episodes": len(episodes),
            "transitions": sum(len(episode.transitions) for episode in episodes),
            "terminated_episodes": sum(
                episode.transitions[-1].terminated for episode in episodes
            ),
            "adaptation_successes": sum(item.success for item in adaptation),
            "collection_policy": "uniform random actions",
        },
        "controls": {
            "same_probe_initialization": True,
            "same_probe_batches_and_labels": True,
            "shuffled_control": "fixed permutation of future endpoints",
            "episode_disjoint_probe_split": True,
            "encoder_and_dynamics_frozen_during_probe": True,
        },
        "limitations": [
            "fixed Push1 topology and start pose; seeds mostly vary box colour",
            "temporal labels measure the dataset policy, not shortest-path distance",
            "actual fork evaluation is diagnostic oracle evidence, not agent input",
            "goal template pose can reveal the Push ruleset",
            "no transfer or AGI claim",
        ],
        "checkpoint_sha256": core._file_hash(args.checkpoint),
        "source_metadata": metadata,
    }
    if args.closed_loop_eval:
        evaluation_trace = core.TraceWriter(args.out / "closed_loop_traces.jsonl")
        eval_seeds = range(20000, 20000 + args.eval_seeds)
        try:
            result["closed_loop"] = {
                "raw_beam4": _closed_loop(
                    model, None, replace(config, beam_width=4), replay,
                    eval_seeds, args.steps, deadline, evaluation_trace, "raw/beam4",
                ),
                "raw_beam5": _closed_loop(
                    model, None, replace(config, beam_width=5), replay,
                    eval_seeds, args.steps, deadline, evaluation_trace, "raw/beam5",
                ),
                "ordered_beam4": _closed_loop(
                    model, ordered, replace(config, beam_width=4), replay,
                    eval_seeds, args.steps, deadline, evaluation_trace, "ordered/beam4",
                ),
                "ordered_beam5": _closed_loop(
                    model, ordered, replace(config, beam_width=5), replay,
                    eval_seeds, args.steps, deadline, evaluation_trace, "ordered/beam5",
                ),
                "shuffled_beam5": _closed_loop(
                    model, shuffled, replace(config, beam_width=5), replay,
                    eval_seeds, args.steps, deadline, evaluation_trace, "shuffled/beam5",
                ),
            }
        finally:
            evaluation_trace.close()
    _write(args.out / "results.json", result)
    _write(
        args.out / "manifest.json",
        {"argv": sys.argv[1:], "budgets": {key: str(value) for key, value in vars(args).items()},
         "status": "completed"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
