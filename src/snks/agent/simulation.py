"""MentalSimulator: simulates action sequences using CausalWorldModel without real environment."""

from __future__ import annotations

from collections import deque

from snks.agent.causal_model import CausalWorldModel


class MentalSimulator:
    """Simulates action sequences using CausalWorldModel without real environment.

    'What happens if I do A, then B, then C?'
    Runs causal chain, returns predicted SKS trajectory.
    """

    def __init__(self, causal_model: CausalWorldModel):
        self.causal_model = causal_model

    def simulate(
        self,
        initial_sks: set[int],
        action_sequence: list[int],
    ) -> list[tuple[set[int], float]]:
        """Simulate action sequence, return (predicted_sks, confidence) per step."""
        trajectory: list[tuple[set[int], float]] = []
        current_sks = set(initial_sks)

        for action in action_sequence:
            predicted_effect, confidence = self.causal_model.predict_effect(current_sks, action)
            # New state = current + predicted new activations
            next_sks = current_sks | predicted_effect
            trajectory.append((next_sks, confidence))
            current_sks = next_sks

        return trajectory

    def find_plan(
        self,
        current_sks: set[int],
        goal_sks: set[int],
        max_depth: int = 10,
        n_actions: int = 5,
        min_confidence: float = 0.3,
    ) -> list[int] | None:
        """BFS through causal model to find plan to reach goal.

        Goal is reached when goal_sks ⊆ current_sks (all goal SKS are active).

        Returns:
            Sequence of actions to reach goal, or None if not found.
        """
        if goal_sks <= current_sks:
            return []

        # BFS: state = frozenset of active SKS
        queue: deque[tuple[frozenset[int], list[int]]] = deque()
        queue.append((frozenset(current_sks), []))
        visited: set[frozenset[int]] = {frozenset(current_sks)}

        while queue:
            state, plan = queue.popleft()

            if len(plan) >= max_depth:
                continue

            for action in range(n_actions):
                predicted_effect, confidence = self.causal_model.predict_effect(
                    set(state), action
                )
                if confidence < min_confidence:
                    continue

                next_state = state | frozenset(predicted_effect)
                new_plan = plan + [action]

                # Goal reached?
                if goal_sks <= next_state:
                    return new_plan

                if next_state not in visited:
                    visited.add(next_state)
                    queue.append((next_state, new_plan))

        return None
