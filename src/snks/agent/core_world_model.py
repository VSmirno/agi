"""Shared recurrent dynamics with independently supervised ensemble heads."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from snks.encoder.core_encoder import CoreEncoder
from snks.env.core_types import Observation


@dataclass(frozen=True)
class LatentState:
    z: Tensor
    sensors: Tensor
    sensor_mask: Tensor
    hidden: Tensor
    schema: str


@dataclass(frozen=True)
class Prediction:
    next_state: LatentState
    terminated_prob: Tensor
    uncertainty: Tensor
    member_z: Tensor


class CoreWorldModel(nn.Module):
    def __init__(self, encoder: CoreEncoder, schemas: dict[str, tuple[int, int]],
                 h_dim: int, heads: int, *, normalize_sensor_condition: bool = False,
                 predict_sensor_delta: bool = False):
        super().__init__()
        if h_dim <= 0 or heads <= 0:
            raise ValueError("hidden dimension and ensemble size must be positive")
        self.encoder, self.h_dim, self.heads = encoder, h_dim, heads
        self.normalize_sensor_condition = normalize_sensor_condition
        self.predict_sensor_delta = predict_sensor_delta
        self.schemas: dict[str, tuple[int, int]] = {}
        self.recurrent = nn.GRUCell(encoder.z_dim + h_dim, h_dim)
        self.latent_heads = nn.ModuleList(nn.Linear(h_dim, encoder.z_dim)
                                          for _ in range(heads))
        self.action_embeddings = nn.ModuleDict()
        self.sensor_projections = nn.ModuleDict()
        self.sensor_heads = nn.ModuleDict()
        self.termination_heads = nn.ModuleDict()
        for name, shape in schemas.items():
            self._add_schema(name, shape)

    def _add_schema(self, name: str, shape: tuple[int, int]) -> None:
        n_actions, n_sensors = shape
        if not name or "." in name or n_actions <= 0 or n_sensors < 0:
            raise ValueError("schema needs a module-safe name and valid dimensions")
        parameter = next(self.encoder.parameters())
        modules = (
            nn.Embedding(n_actions, self.h_dim),
            nn.Linear(2 * n_sensors, self.h_dim),
            nn.ModuleList(nn.Linear(self.h_dim, n_sensors) for _ in range(self.heads)),
            nn.ModuleList(nn.Linear(self.h_dim, 1) for _ in range(self.heads)),
        )
        for collection, module in zip((self.action_embeddings, self.sensor_projections,
                                       self.sensor_heads, self.termination_heads), modules):
            collection[name] = module.to(device=parameter.device, dtype=parameter.dtype)
        self.schemas[name] = tuple(shape)

    def register_schema(self, name: str, shape: tuple[int, int], seed: int) -> None:
        """Initialize new CPU heads reproducibly without consuming global RNG."""
        if name in self.schemas:
            if self.schemas[name] != tuple(shape):
                raise ValueError("cannot change an existing schema's dimensions")
            return
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self._add_schema(name, shape)

    def initial(self, obs: Observation) -> LatentState:
        parameter = next(self.parameters())
        rgb = torch.tensor(obs.rgb, device=parameter.device, dtype=parameter.dtype)[None] / 255
        sensors = torch.tensor(obs.sensors, device=parameter.device, dtype=parameter.dtype)[None]
        mask = torch.tensor(obs.sensor_mask, device=parameter.device, dtype=torch.bool)[None]
        return self.initial_from_tensors(rgb, sensors, mask, obs.schema)

    def initial_from_tensors(self, rgb: Tensor, sensors: Tensor,
                             mask: Tensor, schema: str) -> LatentState:
        if schema not in self.schemas:
            raise ValueError(f"unknown schema: {schema}")
        if sensors.shape != (rgb.shape[0], self.schemas[schema][1]) or mask.shape != sensors.shape:
            raise ValueError("sensor shape does not match schema and batch")
        values = torch.where(mask, sensors, 0.0)
        if not torch.isfinite(values).all():
            raise ValueError("observed sensors must be finite")
        z = self.encoder(rgb)
        return LatentState(z, values, mask.clone(), z.new_zeros(len(z), self.h_dim), schema)

    def step(self, state: LatentState, actions: Tensor) -> Prediction:
        if state.schema not in self.schemas:
            raise ValueError(f"unknown schema: {state.schema}")
        n_actions, _ = self.schemas[state.schema]
        if actions.shape != (len(state.z),) or actions.dtype != torch.long:
            raise ValueError("actions must be a long tensor of shape B")
        if torch.any((actions < 0) | (actions >= n_actions)):
            raise ValueError("action ID outside schema")
        values = torch.where(state.sensor_mask, state.sensors, 0.0)
        body = torch.cat((values, state.sensor_mask.to(values.dtype)), dim=-1)
        projected_body = self.sensor_projections[state.schema](body)
        if self.normalize_sensor_condition:
            projected_body = F.layer_norm(projected_body, (self.h_dim,))
        condition = self.action_embeddings[state.schema](actions) + projected_body
        hidden = self.recurrent(torch.cat((state.z, condition), dim=-1), state.hidden)
        member_z = torch.stack([head(hidden) for head in self.latent_heads])
        member_sensors = torch.stack([head(hidden) for head in self.sensor_heads[state.schema]])
        if self.predict_sensor_delta:
            member_sensors = member_sensors + values.unsqueeze(0)
        terminated = torch.stack([head(hidden).squeeze(-1).sigmoid()
                                  for head in self.termination_heads[state.schema]]).mean(0)
        sensors = torch.where(state.sensor_mask, member_sensors.mean(0), 0.0)
        next_state = LatentState(member_z.mean(0), sensors, state.sensor_mask.clone(),
                                 hidden, state.schema)
        uncertainty = member_z.var(0, unbiased=False).mean(-1)
        return Prediction(next_state, terminated, uncertainty, member_z)

    def rollout(self, state: LatentState, actions: Tensor) -> list[Prediction]:
        if actions.ndim != 2 or actions.shape[0] != len(state.z):
            raise ValueError("rollout actions must have shape B,H")
        predictions = []
        for action in actions.unbind(1):
            prediction = self.step(state, action)
            predictions.append(prediction)
            state = prediction.next_state
        return predictions
