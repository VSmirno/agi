"""Train on real sequence prefixes followed by autoregressive prediction."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from snks.agent.core_world_model import CoreWorldModel, LatentState
from snks.env.core_types import Episode, Mode
from snks.learning.core_objective import masked_mse, sigreg
from snks.pipeline.core_config import CoreConfig


@dataclass(frozen=True)
class SequenceBatch:
    rgb: Tensor
    sensors: Tensor
    sensor_mask: Tensor
    actions: Tensor
    terminated: Tensor
    valid: Tensor
    schema: str
    burn_in: int


def tensorize(episodes: list[Episode], burn_in: int,
              device: torch.device | str) -> SequenceBatch:
    """Pad complete real windows at their tails without inventing transitions."""
    if not episodes or any(not episode.transitions for episode in episodes):
        raise ValueError("tensorize needs nonempty real sequences")
    schemas = {obs.schema for episode in episodes for transition in episode.transitions
               for obs in (transition.before, transition.after)}
    if len(schemas) != 1:
        raise ValueError("mixed schemas cannot share a sequence batch")
    schema = schemas.pop()
    horizon = max(len(episode.transitions) for episode in episodes)
    if burn_in < 0 or burn_in >= horizon:
        raise ValueError("burn-in must leave at least one prediction target")
    sensor_count = len(episodes[0].transitions[0].before.sensors)
    batch_size = len(episodes)
    rgb = np.zeros((batch_size, horizon + 1, 3, 64, 64), dtype=np.uint8)
    sensors = np.zeros((batch_size, horizon + 1, sensor_count), dtype=np.float32)
    masks = np.zeros_like(sensors, dtype=bool)
    actions = np.zeros((batch_size, horizon), dtype=np.int64)
    terminated = np.zeros((batch_size, horizon), dtype=np.float32)
    valid = np.zeros((batch_size, horizon), dtype=bool)
    for row, episode in enumerate(episodes):
        transitions = episode.transitions
        if any(t.terminated or t.truncated for t in transitions[:-1]):
            raise ValueError("a sequence cannot cross an episode boundary")
        observations = [transitions[0].before] + [t.after for t in transitions]
        for index, obs in enumerate(observations):
            rgb[row, index], sensors[row, index], masks[row, index] = (
                obs.rgb, obs.sensors, obs.sensor_mask)
        length = len(transitions)
        actions[row, :length] = [t.action for t in transitions]
        terminated[row, :length] = [t.terminated for t in transitions]
        valid[row, :length] = True
    return SequenceBatch(
        torch.as_tensor(rgb, device=device).float() / 255,
        torch.as_tensor(sensors, device=device), torch.as_tensor(masks, device=device),
        torch.as_tensor(actions, device=device), torch.as_tensor(terminated, device=device),
        torch.as_tensor(valid, device=device), schema, burn_in,
    )


class CoreTrainer:
    def __init__(self, model: CoreWorldModel, config: CoreConfig):
        self.model, self.config = model, config
        self.optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config.learning_rate,
        )

    def compute_loss(self, batch: SequenceBatch) -> Tensor:
        """Supervise every member, with real future frames used only as targets."""
        batch_size, steps = batch.actions.shape
        stop = min(steps, batch.burn_in + self.config.train_horizon)
        if batch.burn_in < 0 or batch.burn_in >= stop:
            raise ValueError("burn-in must leave at least one prediction target")
        if not batch.valid[:, batch.burn_in:stop].any():
            raise ValueError("batch has no valid prediction targets")
        if (batch.valid[:, 1:] & ~batch.valid[:, :-1]).any():
            raise ValueError("padding must be at the sequence tail")
        rgb = batch.rgb[:, :stop + 1]
        z = self.model.encoder(rgb.reshape(-1, 3, 64, 64)).reshape(batch_size, stop + 1, -1)
        mask = batch.sensor_mask[:, 0]
        state = LatentState(z[:, 0], torch.where(mask, batch.sensors[:, 0], 0.0),
                            mask, z.new_zeros(batch_size, self.model.h_dim), batch.schema)
        latent_sum = z.new_zeros(())
        sensor_sum = z.new_zeros(())
        termination_sum = z.new_zeros(())
        latent_count = sensor_count = termination_count = 0
        for index in range(stop):
            valid = batch.valid[:, index]
            actions = torch.where(valid, batch.actions[:, index], 0)
            prediction = self.model.step(state, actions)
            if index < batch.burn_in:
                real_mask = batch.sensor_mask[:, index + 1]
                state = LatentState(z[:, index + 1],
                                    torch.where(real_mask, batch.sensors[:, index + 1], 0.0),
                                    real_mask, prediction.next_state.hidden, batch.schema)
                continue
            member_count = self.model.heads
            n_latent = int(valid.sum()) * member_count * z.shape[-1]
            latent_sum = latent_sum + masked_mse(
                prediction.member_z, z[:, index + 1], valid[None, :, None]) * n_latent
            latent_count += n_latent
            # Keep individual member errors: a correct mean must not conceal bad heads.
            hidden = prediction.next_state.hidden
            member_sensors = torch.stack([head(hidden)
                                          for head in self.model.sensor_heads[batch.schema]])
            if self.model.predict_sensor_delta:
                member_sensors = member_sensors + state.sensors.unsqueeze(0)
            target_mask = batch.sensor_mask[:, index + 1] & valid[:, None]
            n_sensor = int(target_mask.sum()) * member_count
            sensor_sum = sensor_sum + masked_mse(
                member_sensors, batch.sensors[:, index + 1], target_mask[None]) * n_sensor
            sensor_count += n_sensor
            logits = torch.stack([head(hidden).squeeze(-1)
                                  for head in self.model.termination_heads[batch.schema]])
            present_logits = logits[:, valid]
            targets = batch.terminated[:, index][valid].expand(member_count, -1)
            termination_sum = termination_sum + F.binary_cross_entropy_with_logits(
                present_logits, targets, reduction="sum")
            termination_count += present_logits.numel()
            state = prediction.next_state
        loss = (latent_sum / max(latent_count, 1)
                + self.config.sensor_weight * sensor_sum / max(sensor_count, 1)
                + self.config.termination_weight * termination_sum / max(termination_count, 1))
        if self.config.sigreg_weight:
            real_valid = torch.cat((batch.valid[:, :1], batch.valid[:, :stop]), dim=1)
            real_z = z[real_valid]
            if len(real_z) >= 2:
                directions = F.normalize(torch.randn(z.shape[-1], 32, device=z.device), dim=0)
                loss = loss + self.config.sigreg_weight * sigreg(real_z, directions)
        return loss

    def update(self, batch: SequenceBatch, mode: Mode) -> dict[str, float]:
        """Perform an authorized finite update; evaluation has no side effects."""
        if mode not in (Mode.TRAIN, Mode.ADAPT):
            raise PermissionError("model updates require TRAIN or ADAPT mode")
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.compute_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for group in self.optimizer.param_groups for parameter in group["params"]):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("non-finite training gradients")
        self.optimizer.step()
        return {"loss": float(loss.detach())}
