"""MotorEncoder: encodes discrete actions as SDR current patterns for DAF injection."""

from __future__ import annotations

import torch


class MotorEncoder:
    """Encodes discrete actions as SDR current patterns for DAF injection.

    Each action activates a fixed non-overlapping group of nodes (motor zone).
    """

    def __init__(
        self,
        n_actions: int,
        num_nodes: int,
        sdr_size: int = 512,
        current_strength: float = 1.0,
    ):
        self.n_actions = n_actions
        self.num_nodes = num_nodes
        self.sdr_size = sdr_size
        self.current_strength = current_strength

        # Motor zone occupies the last n_actions * sdr_size nodes
        self.motor_zone_size = n_actions * sdr_size
        self.motor_zone_start = num_nodes - self.motor_zone_size

        if self.motor_zone_start < 0:
            raise ValueError(
                f"Motor zone ({self.motor_zone_size}) exceeds num_nodes ({num_nodes})"
            )

        # Pre-compute action→node index mappings
        self._action_indices: list[torch.Tensor] = []
        for a in range(n_actions):
            start = self.motor_zone_start + a * sdr_size
            end = start + sdr_size
            self._action_indices.append(torch.arange(start, end))

    def encode(self, action: int, device: torch.device | None = None) -> torch.Tensor:
        """Action index → (num_nodes,) current injection.

        Returns a 1D tensor where only the motor zone for this action is active.
        """
        currents = torch.zeros(self.num_nodes, device=device)
        indices = self._action_indices[action]
        if device is not None:
            indices = indices.to(device)
        currents[indices] = self.current_strength
        return currents

    def decode(self, firing_rates: torch.Tensor) -> int:
        """Firing rate vector (num_nodes,) → most likely action (winner-take-all).

        Sums firing rates in each action's motor zone and returns the argmax.
        """
        scores = torch.zeros(self.n_actions)
        for a in range(self.n_actions):
            indices = self._action_indices[a]
            dev_indices = indices.to(firing_rates.device)
            scores[a] = firing_rates[dev_indices].sum()
        return int(scores.argmax().item())
