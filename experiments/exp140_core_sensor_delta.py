"""Paired from-scratch test of absolute versus sensor-delta prediction."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Episode, Mode
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.pipeline.core_config import CoreConfig
from snks.pipeline.core_controls import prediction_probe, shuffle_action_labels
from snks.pipeline.core_experiment import TraceWriter, _evaluate, _run_cases
from snks.pipeline.core_runner import model_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _corpus_hash(batches: list[list[Episode]]) -> str:
    rows = [[(episode.uid, [(t.before.step, t.action) for t in episode.transitions])
             for episode in batch] for batch in batches]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def _event_sensor_mse_h1(model: CoreWorldModel, episodes: list[Episode]) -> dict:
    errors: list[float] = []
    model.eval()
    for episode in episodes:
        for transition in episode.transitions:
            root = model.initial(transition.before)
            predicted = model.step(
                root, torch.tensor([transition.action], device=root.z.device)
            ).next_state.sensors
            target = model.initial(transition.after).sensors
            changed = ((root.sensors != target) & root.sensor_mask
                       & model.initial(transition.after).sensor_mask)
            errors.extend((predicted - target).square()[changed].cpu().tolist())
    return {"mse": float(np.mean(errors)) if errors else None, "changed_entries": len(errors)}


@torch.no_grad()
def _crafter_wood_action_probe(model: CoreWorldModel, episodes: list[Episode]) -> dict:
    rows = []
    model.eval()
    for episode in episodes:
        for transition in episode.transitions:
            if (transition.action != 5 or len(transition.before.sensors) <= 4
                    or transition.after.sensors[4] <= transition.before.sensors[4]):
                continue
            root = model.initial(transition.before)
            deltas = torch.stack([
                model.step(root, torch.tensor([action], device=root.z.device)
                           ).next_state.sensors[0, 4] - root.sensors[0, 4]
                for action in range(model.schemas[root.schema][0])
            ])
            others = torch.cat((deltas[:5], deltas[6:]))
            rows.append((float(deltas[5]), float(others.max()),
                         int((deltas > deltas[5]).sum()) + 1))
    return {"scope": "controlled-Crafter evaluator only", "events": len(rows),
            "action5_mean": float(np.mean([r[0] for r in rows])) if rows else None,
            "max_other_mean": float(np.mean([r[1] for r in rows])) if rows else None,
            "action5_rank_per_event": [r[2] for r in rows]}


def _fresh(config: CoreConfig, schemas: dict[str, tuple[int, int]]) -> CoreWorldModel:
    torch.manual_seed(config.seed)
    return CoreWorldModel(
        CoreEncoder(config.z_dim), schemas, config.h_dim, config.ensemble_size,
        normalize_sensor_condition=False,
        predict_sensor_delta=config.predict_sensor_delta,
    ).to(config.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()
    if args.updates <= 0:
        parser.error("--updates must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    replay_path = args.checkpoint.with_name(payload["replay_path"])
    source_replay = SequenceReplay.load(replay_path)
    base = CoreConfig(**metadata["config"])
    common = replace(base, device="cuda", seed=0, z_dim=64, h_dim=32,
                     batch_size=8, burn_in=1, train_horizon=3,
                     normalize_sensor_condition=False)
    configs = {
        "absolute-real": replace(common, predict_sensor_delta=False),
        "delta-real": replace(common, predict_sensor_delta=True),
        "delta-shuffled": replace(common, predict_sensor_delta=True),
        "delta-untrained": replace(common, predict_sensor_delta=True),
    }
    schemas = {str(k): tuple(map(int, v)) for k, v in metadata["schemas"].items()}
    sampler = copy.deepcopy(source_replay)
    batches = [sampler.sample(8, 3, 1, common.recent_fraction)
               for _ in range(args.updates)]
    if any(len(batch) != 8 for batch in batches):
        raise RuntimeError("source replay could not supply a full paired batch")
    models = {name: _fresh(config, schemas) for name, config in configs.items()}
    initial = {name: model_digest(model) for name, model in models.items()}
    if len(set(initial.values())) != 1:
        raise RuntimeError("fresh paired model initializations differ")
    trained = ("absolute-real", "delta-real", "delta-shuffled")
    trainers = {name: CoreTrainer(models[name], configs[name]) for name in trained}
    losses = {name: [] for name in models}
    for index, episodes in enumerate(batches):
        batch = tensorize(episodes, 1, "cuda")
        for name, trainer in trainers.items():
            paired = (shuffle_action_labels(batch, index)
                      if name == "delta-shuffled" else batch)
            torch.manual_seed(index)
            losses[name].append(float(trainer.update(paired, Mode.TRAIN)["loss"]))
    models["delta-real-zero-action"] = copy.deepcopy(models["delta-real"])
    configs["delta-real-zero-action"] = configs["delta-real"]
    losses["delta-real-zero-action"] = []
    with torch.no_grad():
        models["delta-real-zero-action"].action_embeddings[
            metadata["source_schema"]
        ].weight.zero_()
    branch_replays = {name: copy.deepcopy(source_replay) for name in models}
    before = {name: replay.manifest() for name, replay in branch_replays.items()}
    trace = TraceWriter(args.out / "traces.jsonl")
    deadline = time.monotonic() + 180
    try:
        corpus_replay = copy.deepcopy(source_replay)
        corpus_before = corpus_replay.manifest()
        _, heldout = _run_cases(
            model=models["absolute-real"], config=common, replay=corpus_replay,
            family=metadata["source_family"], ruleset=metadata["source_ruleset"],
            seeds=range(3000, 3004), split="validation", mode=Mode.EVALUATE,
            steps=32, deadline=deadline, trace=trace, role="common_random_corpus",
            random_actions=True,
        )
        if len(heldout) != 4 or corpus_replay.manifest() != corpus_before:
            raise RuntimeError("held-out corpus incomplete or appended to replay")
        validation = {name: _evaluate(
            model, configs[name], branch_replays[name], metadata["source_family"],
            metadata["source_ruleset"], range(2000, 2004), 32, deadline, trace,
            f"frozen_policy/{name}",
        ) for name, model in models.items()}
    finally:
        trace.close()
    after = {name: replay.manifest() for name, replay in branch_replays.items()}
    action_rows = [(episode.uid, [t.action for t in episode.transitions])
                   for episode in heldout]
    result = {
        "status": "completed", "seed": 0, "requested_updates": args.updates,
        "source": {"checkpoint": str(args.checkpoint.resolve()),
                   "checkpoint_sha256": _sha256(args.checkpoint),
                   "replay": str(replay_path.resolve()),
                   "replay_sha256": _sha256(replay_path),
                   "replay_manifest": source_replay.manifest(), "schemas": schemas},
        "configs": {name: asdict(config) for name, config in configs.items()},
        "paired_training": {"batch_size": 8, "burn_in": 1, "horizon": 3,
                            "corpus_sha256": _corpus_hash(batches),
                            "initial_model_sha256": initial},
        "common_action_corpus": {"episode_uids": [row[0] for row in action_rows],
                                 "sha256": hashlib.sha256(
                                     json.dumps(action_rows).encode()).hexdigest()},
        "arms": {name: {
            "loss": ({"first": values[0], "last": values[-1], "count": len(values)}
                     if values else None),
            "actual_updates": 0 if name == "delta-untrained" else args.updates,
            "success": validation[name],
            "failure_status": {
                "agent_failures": validation[name]["agent_failures"],
                "infrastructure_failures": validation[name]["infrastructure_failures"],
            },
            "prediction": prediction_probe(models[name], heldout),
            "event_sensor_h1_cold_start": {
                "history": "cold_start",
                **_event_sensor_mse_h1(models[name], heldout),
            },
            "crafter_wood_action5": _crafter_wood_action_probe(models[name], heldout),
            "replay_frozen": before[name] == after[name],
        } for name, values in losses.items()},
    }
    (args.out / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
