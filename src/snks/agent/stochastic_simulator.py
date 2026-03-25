"""StochasticSimulator: Monte Carlo planning via CausalWorldModel (Stage 11).

Extends MentalSimulator with stochastic rollouts: samples effects from
softmax(confidence/T) distribution to estimate expected success rate.
Planning is greedy-step with N-sample scoring; transition is deterministic.
"""

from __future__ import annotations

import math

import numpy as np

from snks.agent.causal_model import CausalWorldModel


class StochasticSimulator:
    """Monte Carlo planner using CausalWorldModel.

    Uses get_all_effects_for_action() to sample stochastic futures,
    then picks best action deterministically (greedy + argmax).
    """

    def __init__(
        self,
        causal_model: CausalWorldModel,
        seed: int | None = None,
    ) -> None:
        self._causal = causal_model
        self._rng = np.random.RandomState(seed)

    def sample_effect(
        self,
        context: set[int],
        action: int,
        temperature: float = 1.0,
    ) -> tuple[set[int], float]:
        """Stochastically sample one effect from confidence distribution.

        P(effect_i) ∝ softmax(confidence_i / temperature)
        temperature → 0: deterministic (argmax).
        temperature → ∞: uniform.
        Returns (set(), 0.0) if no effects known.
        """
        effects = self._causal.get_all_effects_for_action(context, action)
        if not effects:
            return set(), 0.0

        effect_list = list(effects)  # [(effect_sks, confidence), ...]
        confidences = np.array([c for _, c in effect_list], dtype=np.float64)

        # Softmax with temperature
        if temperature <= 1e-8:
            idx = int(np.argmax(confidences))
        else:
            logits = confidences / temperature
            logits -= logits.max()  # numerical stability
            probs = np.exp(logits)
            probs /= probs.sum()
            idx = int(self._rng.choice(len(effect_list), p=probs))

        chosen_effect, chosen_conf = effect_list[idx]
        return set(chosen_effect), float(chosen_conf)

    def rollout(
        self,
        initial_sks: set[int],
        action_sequence: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[tuple[set[int], float]], float]:
        """One stochastic rollout along action_sequence.

        Returns:
            (trajectory, total_cost)
            trajectory = [(sks_state, confidence), ...] per step
            total_cost = sum(1 - confidence_i) for each step
        """
        trajectory: list[tuple[set[int], float]] = []
        state = set(initial_sks)
        total_cost = 0.0

        for action in action_sequence:
            effect, conf = self.sample_effect(state, action, temperature)
            state = state | effect
            trajectory.append((set(state), conf))
            total_cost += 1.0 - conf

        return trajectory, total_cost

    def find_plan_stochastic(
        self,
        current_sks: set[int],
        goal_sks: set[int],
        n_actions: int = 5,
        n_samples: int = 8,
        temperature: float = 1.0,
        max_depth: int = 10,
        min_confidence: float = 0.3,
    ) -> tuple[list[int] | None, float]:
        """Greedy Monte Carlo planning with N-sample scoring.

        Algorithm:
          For each step up to max_depth:
            For each action a in range(n_actions):
              Sample N rollouts from (state, a) → score = fraction reaching goal
            Pick best_action (deterministic argmax)
            Execute best_action deterministically (predict_effect, not sample)
            Append to plan

        The stochastic sampling is ONLY in the scoring phase.
        The actual plan transition uses deterministic predict_effect.

        Returns:
            (plan, expected_success_rate) or (None, 0.0) if goal unreachable.
        """
        if goal_sks <= current_sks:
            return [], 1.0

        plan: list[int] = []
        state = set(current_sks)
        best_score = 0.0

        for step in range(max_depth):
            if goal_sks <= state:
                return plan, best_score

            best_action: int | None = None
            best_score = -math.inf

            for a in range(n_actions):
                scores: list[float] = []
                for _ in range(n_samples):
                    # Sample immediate effect
                    effect, conf = self.sample_effect(state, a, temperature)
                    if conf < min_confidence:
                        scores.append(0.0)
                        continue
                    next_s = state | effect
                    # Quick goal check: if already reached → success
                    if goal_sks <= next_s:
                        scores.append(1.0)
                        continue
                    # Short rollout from next_s with random actions
                    remaining = max_depth - step - 1
                    if remaining <= 0:
                        scores.append(0.0)
                        continue
                    random_actions = self._rng.randint(0, n_actions, size=remaining).tolist()
                    traj, _ = self.rollout(next_s, random_actions, temperature)
                    final_sks = traj[-1][0] if traj else next_s
                    scores.append(1.0 if goal_sks <= final_sks else 0.0)

                score_a = float(np.mean(scores)) if scores else 0.0
                if score_a > best_score:
                    best_score = score_a
                    best_action = a

            if best_action is None or best_score <= 0.0:
                break  # no progress

            # Execute deterministically
            det_effect, det_conf = self._causal.predict_effect(state, best_action)
            if det_conf < min_confidence:
                break

            state = state | det_effect
            plan.append(best_action)

        if goal_sks <= state:
            return plan, best_score
        return None, best_score
