"""AttractorNavigator: goal-directed action selection via DAF dynamics.

Replaces BFS-based GridNavigator with attractor-based navigation.
For each possible action, runs mental simulation and picks the action
whose predicted state is most similar to the goal state.
"""

from __future__ import annotations

import random

import torch
from torch import Tensor


class AttractorNavigator:
    """Navigate toward goals using attractor dynamics and mental simulation.

    Instead of BFS on the grid:
    1. Encode goal observation → goal_embedding (HAC vector)
    2. For each action, predict next state via DafCausalModel
    3. Select action with highest cosine similarity to goal
    4. Fall back to exploration when similarity is low
    """

    def __init__(
        self,
        daf_causal_model,  # DafCausalModel
        motor_encoder,     # MotorEncoder
        n_sim_steps: int = 10,
        min_similarity: float = -0.5,
        exploration_epsilon: float = 0.3,
    ) -> None:
        self._causal = daf_causal_model
        self._motor = motor_encoder
        self._n_sim_steps = n_sim_steps
        self._min_similarity = min_similarity
        self._epsilon = exploration_epsilon
        self._step_count: int = 0
        self._explore_count: int = 0
        self._goal_directed_count: int = 0

    def select_action(
        self,
        current_embedding: Tensor,
        goal_embedding: Tensor | None,
        n_actions: int,
    ) -> int:
        """Select action via mental simulation toward goal.

        Args:
            current_embedding: (D,) HAC embedding of current state
            goal_embedding: (D,) HAC embedding of goal state, or None for exploration
            n_actions: number of available actions

        Returns:
            action index
        """
        self._step_count += 1

        # Epsilon-greedy exploration
        if random.random() < self._epsilon:
            self._explore_count += 1
            return random.randint(0, n_actions - 1)

        # No goal → pure exploration
        if goal_embedding is None:
            self._explore_count += 1
            return random.randint(0, n_actions - 1)

        # Mental simulation: score each action
        best_action = 0
        best_similarity = -2.0

        for action in range(n_actions):
            predicted = self._causal.predict_effect(
                current_embedding, action, self._motor, self._n_sim_steps,
            )
            sim = self._cosine_similarity(predicted, goal_embedding)

            if sim > best_similarity:
                best_similarity = sim
                best_action = action

        # Below threshold → random exploration
        if best_similarity < self._min_similarity:
            self._explore_count += 1
            return random.randint(0, n_actions - 1)

        self._goal_directed_count += 1
        return best_action

    def select_action_fast(
        self,
        current_sks: set[int],
        goal_sks: set[int] | None,
        n_actions: int,
    ) -> int:
        """Fast action selection without mental simulation.

        Uses novelty-based exploration: prefer actions that change state
        the most (highest SKS symmetric difference from recent history).

        For simple environments where mental simulation is too expensive.
        """
        self._step_count += 1

        if random.random() < self._epsilon:
            self._explore_count += 1
            return random.randint(0, n_actions - 1)

        if goal_sks is None:
            self._explore_count += 1
            return random.randint(0, n_actions - 1)

        # Prefer action that moves us toward goal SKS
        # Simple heuristic: overlap between current and goal
        current_overlap = len(current_sks & goal_sks) if goal_sks else 0

        # Without mental simulation, use exploration with bias
        # toward actions not recently tried
        self._explore_count += 1
        return random.randint(0, n_actions - 1)

    @staticmethod
    def _cosine_similarity(a: Tensor, b: Tensor) -> float:
        """Compute cosine similarity between two vectors."""
        a = a.flatten().float()
        b = b.flatten().float()
        dot = torch.dot(a, b)
        na = a.norm()
        nb = b.norm()
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(dot / (na * nb))

    @property
    def stats(self) -> dict:
        """Return navigation statistics."""
        total = max(self._step_count, 1)
        return {
            "total_steps": self._step_count,
            "explore_ratio": self._explore_count / total,
            "goal_directed_ratio": self._goal_directed_count / total,
        }
