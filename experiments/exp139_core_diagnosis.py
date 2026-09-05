"""Small read-only diagnosis of action dependence in the learning-core pilot."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from snks.agent.core_world_model import CoreWorldModel
from snks.encoder.core_encoder import CoreEncoder
from snks.learning.core_replay import SequenceReplay
from snks.learning.core_trainer import CoreTrainer, tensorize
from snks.pipeline.core_config import CoreConfig


def summary(x):
    x = x.detach().float()
    return {"rms": float(x.square().mean().sqrt()), "std": float(x.std(unbiased=False)),
            "abs_mean": float(x.abs().mean()), "max_abs": float(x.abs().max())}


def grad_norm(module):
    grads = [p.grad.detach().float().flatten() for p in module.parameters() if p.grad is not None]
    return float(torch.cat(grads).norm()) if grads else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result_path = args.out / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    meta = payload["metadata"]
    cfg = CoreConfig(**{**meta["config"], "device": "cuda"})
    schema = meta["source_schema"]
    schemas = {k: tuple(v) for k, v in meta["schemas"].items()}
    torch.manual_seed(cfg.seed)
    fresh = CoreWorldModel(
        CoreEncoder(cfg.z_dim), schemas, cfg.h_dim, cfg.ensemble_size,
        normalize_sensor_condition=cfg.normalize_sensor_condition,
        predict_sensor_delta=cfg.predict_sensor_delta,
    ).cuda()
    source = CoreWorldModel(
        CoreEncoder(cfg.z_dim), schemas, cfg.h_dim, cfg.ensemble_size,
        normalize_sensor_condition=cfg.normalize_sensor_condition,
        predict_sensor_delta=cfg.predict_sensor_delta,
    ).cuda()
    source.load_state_dict(payload["model"])
    replay = SequenceReplay.load(args.checkpoint.with_name(payload["replay_path"]))
    episodes = replay._episodes()[:cfg.batch_size]
    batch = tensorize(episodes, cfg.burn_in, "cuda")
    rgb = batch.rgb[:, :cfg.burn_in + cfg.train_horizon + 1]
    report = {"metadata": {"checkpoint": str(args.checkpoint.resolve()),
                           "profile": cfg.profile, "config": meta["config"]},
              "batch_episode_uids": [e.uid for e in episodes], "models": {}}
    for name, model in (("fresh", fresh), ("source", source)):
        model.eval()
        z_all = model.encoder(rgb.flatten(0, 1))
        state = model.initial_from_tensors(batch.rgb[:, 0], batch.sensors[:, 0],
                                           batch.sensor_mask[:, 0], schema)
        actions = batch.actions[:, 0]
        body = torch.cat((state.sensors, state.sensor_mask.float()), -1)
        action = model.action_embeddings[schema](actions)
        sensor = model.sensor_projections[schema](body)
        if cfg.normalize_sensor_condition:
            sensor = F.layer_norm(sensor, (cfg.h_dim,))
        condition = action + sensor
        x = torch.cat((state.z, condition), -1)
        gi = F.linear(x, model.recurrent.weight_ih, model.recurrent.bias_ih)
        gh = F.linear(state.hidden, model.recurrent.weight_hh, model.recurrent.bias_hh)
        ir, iz, inn = gi.chunk(3, 1)
        hr, hz, hn = gh.chunk(3, 1)
        reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
        new = (inn + reset * hn).tanh()
        hidden = (1 - update) * new + update * state.hidden
        predictions = []
        for action_id in range(schemas[schema][0]):
            predictions.append(model.step(state, torch.full_like(actions, action_id)).next_state.sensors)
        predictions = torch.stack(predictions)
        real_prediction = model.step(state, actions).next_state.sensors
        trainer = CoreTrainer(model, cfg)
        trainer.optimizer.zero_grad(set_to_none=True)
        loss = trainer.compute_loss(batch)
        loss.backward()
        report["models"][name] = {
            "encoder_parameters": summary(torch.cat([p.detach().flatten() for p in model.encoder.parameters()])),
            "encoder_activations": summary(z_all), "initial_z": summary(state.z),
            "action_embedding": summary(action), "sensor_projection": summary(sensor),
            "combined_condition": summary(condition),
            "gru_input_contribution": {
                "z": summary(F.linear(state.z, model.recurrent.weight_ih[:, :cfg.z_dim])),
                "condition": summary(F.linear(condition, model.recurrent.weight_ih[:, cfg.z_dim:])),
            },
            "gates": {"reset": summary(reset), "update": summary(update), "candidate": summary(new),
                      "hidden": summary(hidden), "hidden_abs_gt_099": float((hidden.abs() > .99).float().mean())},
            "sensor_baseline": {"observed_mean": batch.sensors[:, 0].mean(0).tolist(),
                                "real_action_prediction_mean": real_prediction.mean(0).tolist()},
            "all_action_prediction_spread": {
                "sensor_std_rms": float(predictions.std(0, unbiased=False).square().mean().sqrt()),
                "sensor_range_by_channel": (predictions.max(0).values - predictions.min(0).values).mean(0).tolist()},
            "loss": float(loss.detach()),
            "gradient_l2": {"encoder": grad_norm(model.encoder),
                            "action_embedding": grad_norm(model.action_embeddings[schema]),
                            "sensor_projection": grad_norm(model.sensor_projections[schema]),
                            "recurrent": grad_norm(model.recurrent),
                            "sensor_heads": grad_norm(model.sensor_heads[schema])},
        }
    args.out.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
