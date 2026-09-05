"""Matched fixed-representation controls; no changing goal geometry mid-test."""

import copy
from dataclasses import replace
import time

import numpy as np
import torch

from snks.agent.core_world_model import CoreWorldModel
from snks.env.core_types import Episode, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, SequenceBatch, tensorize
from snks.pipeline.core_config import CoreConfig
from snks.pipeline.core_runner import model_digest


def build_dynamics_controls(model: CoreWorldModel) -> dict[str, CoreWorldModel]:
    variants = {name: copy.deepcopy(model)
                for name in ("initial", "real_actions", "shuffled_actions")}
    for variant in variants.values():
        variant.encoder.requires_grad_(False)
        variant.eval()
    return variants


def shuffle_action_labels(batch: SequenceBatch, seed: int) -> SequenceBatch:
    """Shuffle only valid action labels, not targets, order or observations."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    labels = batch.actions[batch.valid].clone()
    permutation = torch.randperm(len(labels), generator=generator).to(labels.device)
    actions = batch.actions.clone()
    actions[batch.valid] = labels[permutation]
    return replace(batch, actions=actions)


def train_dynamics_controls(variants: dict[str, CoreWorldModel], replay: SequenceReplay,
                            config: CoreConfig, updates: int,
                            deadline: float) -> dict[str, list[float]]:
    """Both trained arms consume the exact same sampled batch per update."""
    trainers = {name: CoreTrainer(variants[name], config)
                for name in ("real_actions", "shuffled_actions")}
    frozen = {name: model_digest(model.encoder) for name, model in variants.items()}
    losses = {name: [] for name in trainers}
    for index in range(updates):
        if time.monotonic() > deadline:
            raise TimeoutError("control training exceeded declared wall-clock budget")
        episodes = replay.sample(config.batch_size, config.train_horizon,
                                 config.burn_in, config.recent_fraction)
        batch = tensorize(episodes, config.burn_in, torch.device(config.device))
        for name, trainer in trainers.items():
            paired = batch if name == "real_actions" else shuffle_action_labels(batch, config.seed + index)
            torch.manual_seed(config.seed + index)
            metrics = trainer.update(paired, Mode.TRAIN)
            losses[name].append(float(metrics["loss"]))
    if any(model_digest(model.encoder) != frozen[name] for name, model in variants.items()):
        raise RuntimeError("control training changed the fixed representation")
    return losses


@torch.no_grad()
def prediction_probe(model: CoreWorldModel, episodes: list[Episode],
                     horizons: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Real-outcome error and persistence on identical prefixes/actions.

    Latent error is secondary and only comparable across a frozen encoder.
    Event counts make an empty/easy sensor metric visible.
    """
    model.eval()
    buckets = {h: {"sensor_sq": [], "persistence_sq": [], "latent_sq": [],
                   "termination_sq": [], "sensor_changes": 0, "windows": 0}
               for h in horizons}
    for episode in episodes:
        if not episode.transitions:
            continue
        root = model.initial(episode.transitions[0].before)
        for start, real in enumerate(episode.transitions):
            state = root
            for offset, transition in enumerate(episode.transitions[start:start + max(horizons)], 1):
                prediction = model.step(state, torch.tensor([transition.action], device=state.z.device))
                state = prediction.next_state
                if offset in buckets:
                    target = model.initial(transition.after)
                    mask = target.sensor_mask & root.sensor_mask
                    row = buckets[offset]
                    row["sensor_sq"].extend((state.sensors - target.sensors).square()[mask].cpu().tolist())
                    row["persistence_sq"].extend((root.sensors - target.sensors).square()[mask].cpu().tolist())
                    row["latent_sq"].append(float((state.z - target.z).square().mean()))
                    row["termination_sq"].append(float((prediction.terminated_prob - float(transition.terminated)).square().mean()))
                    row["sensor_changes"] += int(((root.sensors != target.sensors) & mask).sum())
                    row["windows"] += 1
                if transition.terminated or transition.truncated:
                    break
            predicted = model.step(root, torch.tensor([real.action], device=root.z.device))
            observed = model.initial(real.after)
            root = replace(observed, hidden=predicted.next_state.hidden)
    result = {}
    for horizon, row in buckets.items():
        result[str(horizon)] = {name: (float(np.mean(values)) if values else None)
                                for name, values in row.items() if isinstance(values, list)}
        result[str(horizon)].update(sensor_changes=row["sensor_changes"], windows=row["windows"])
    return result
