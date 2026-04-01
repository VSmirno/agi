"""DafCausalModel: causal learning through reward-modulated STDP.

Replaces dict-based CausalWorldModel with oscillator-native learning.
Causal relationships are encoded in STDP weight patterns, not explicit dicts.

Key mechanisms:
1. Action encoding: MotorEncoder injects action as DAF current → modulates dynamics
2. Reward modulation: positive reward amplifies recent STDP weight changes (eligibility trace)
3. State representation: HAC embeddings from SKS clusters (not hardcoded IDs)
4. Causal query: mental simulation — inject state + action, read predicted state change
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
from torch import Tensor

from snks.daf.eligibility import EligibilityTrace
from snks.daf.stdp import STDPResult


@dataclass
class CausalTrace:
    """One step of eligibility trace for reward-modulated STDP."""
    weight_snapshot: Tensor  # edge weights before action
    action: int
    sks_before: set[int]  # SKS cluster IDs before
    embedding_before: Tensor | None  # HAC embedding before
    step_idx: int


class DafCausalModel:
    """Causal learning through reward-modulated STDP.

    Instead of storing (context, action) → effect in a dict:
    - STDP naturally strengthens connections between co-active neurons
    - Motor action current modulates which neurons are active
    - Reward signal amplifies or dampens recent weight changes

    This creates action-conditioned attractors: the same visual input +
    different motor action → different attractor state → different prediction.
    """

    def __init__(
        self,
        engine,  # DafEngine
        reward_scale: float = 2.0,
        trace_length: int = 5,
        negative_scale: float = 0.5,
        trace_decay: float = 0.92,
        trace_reward_lr: float = 0.5,
    ) -> None:
        self.engine = engine
        self.reward_scale = reward_scale
        self.negative_scale = negative_scale
        self._trace: deque[CausalTrace] = deque(maxlen=trace_length)
        self._step_idx: int = 0
        self._total_reward_received: float = 0.0
        self._total_modulations: int = 0
        # Stage 41: eligibility trace for long-range credit assignment
        self._eligibility = EligibilityTrace(decay=trace_decay, reward_lr=trace_reward_lr)

    def before_action(self, action: int, current_sks: set[int],
                      current_embedding: Tensor | None = None) -> None:
        """Snapshot weights before action for later reward modulation."""
        snapshot = self.engine.graph.edge_attr[:, 0].clone()
        self._trace.append(CausalTrace(
            weight_snapshot=snapshot,
            action=action,
            sks_before=current_sks.copy(),
            embedding_before=current_embedding.clone() if current_embedding is not None else None,
            step_idx=self._step_idx,
        ))
        self._step_idx += 1

    def accumulate_stdp(self, stdp_result: STDPResult) -> None:
        """Accumulate STDP weight changes into eligibility trace (Stage 41).

        Called after each perception cycle to build up the trace.
        """
        if stdp_result.dw is not None:
            self._eligibility.accumulate(stdp_result.dw)

    def after_action(self, reward: float) -> None:
        """Modulate STDP weights based on reward.

        Uses two complementary mechanisms:
        1. Eligibility trace (Stage 41): long-range credit (20+ steps)
        2. Legacy snapshot trace: short-range precision (5 steps)

        Reward == 0: no modulation (STDP runs normally)
        """
        if abs(reward) < 1e-8 or not self._trace:
            return

        self._total_reward_received += abs(reward)
        self._total_modulations += 1

        # Stage 41: eligibility trace — long-range credit assignment
        effective_reward = reward * self.reward_scale if reward > 0 \
            else reward * self.negative_scale
        self._eligibility.apply_reward(
            effective_reward, self.engine.graph,
            self.engine.stdp.w_min, self.engine.stdp.w_max,
        )

        # Legacy snapshot trace — short-range precision
        current_weights = self.engine.graph.edge_attr[:, 0]
        applied = False

        for i, trace in enumerate(reversed(self._trace)):
            if current_weights.shape[0] != trace.weight_snapshot.shape[0]:
                continue
            decay = 0.8 ** i
            delta_w = current_weights - trace.weight_snapshot

            if reward > 0:
                modulation = decay * reward * self.reward_scale
                modulated_delta = delta_w * (1.0 + modulation)
            else:
                modulation = decay * abs(reward) * self.negative_scale
                modulated_delta = delta_w * (1.0 - modulation)

            new_weights = trace.weight_snapshot + modulated_delta
            new_weights.clamp_(
                self.engine.stdp.w_min,
                self.engine.stdp.w_max,
            )
            applied = True

        if applied:
            self.engine.graph.edge_attr[:, 0] = new_weights

    def predict_effect(
        self,
        current_embedding: Tensor,
        action: int,
        motor_encoder,
        n_sim_steps: int = 10,
    ) -> Tensor:
        """Predict next state embedding by running short mental simulation.

        Injects current state as external current + motor action,
        runs abbreviated DAF integration, reads out predicted embedding.

        Args:
            current_embedding: (2048,) HAC embedding of current state
            action: action to simulate
            motor_encoder: MotorEncoder for action encoding
            n_sim_steps: integration steps for simulation (fewer = faster)

        Returns:
            predicted_embedding: (2048,) HAC embedding of predicted state
        """
        device = self.engine.device

        # Save engine state
        saved_states = self.engine.states.clone()
        saved_currents = self.engine._external_currents.clone()

        try:
            # Inject motor action
            motor_currents = motor_encoder.encode(action, device=device)
            self.engine.set_input(motor_currents)

            # Run short simulation (no learning during simulation)
            old_learning = self.engine.enable_learning
            self.engine.enable_learning = False
            result = self.engine.step(n_sim_steps)
            self.engine.enable_learning = old_learning

            # Extract predicted embedding from coherence
            return self._extract_embedding(result)
        finally:
            # Restore engine state
            self.engine.states.copy_(saved_states)
            self.engine._external_currents.copy_(saved_currents)

    def _extract_embedding(self, step_result) -> Tensor:
        """Extract HAC-like embedding from DAF step result.

        Uses firing rate pattern as a rough embedding. Full SKS detection
        + HAC embedding is too expensive for mental simulation.
        """
        fired = step_result.fired_history  # (T, N)
        rates = fired.float().mean(dim=0)  # (N,) firing rates

        # Reduce to fixed-size embedding via chunked averaging
        embed_dim = 2048
        n = rates.shape[0]
        chunk = max(1, n // embed_dim)
        # Pad to multiple of chunk
        padded = torch.nn.functional.pad(rates, (0, chunk * embed_dim - n))
        embedding = padded[:chunk * embed_dim].reshape(embed_dim, chunk).mean(dim=1)

        # Normalize
        norm = embedding.norm()
        if norm > 1e-8:
            embedding = embedding / norm
        return embedding

    @property
    def stats(self) -> dict:
        """Return learning statistics."""
        return {
            "total_reward": self._total_reward_received,
            "total_modulations": self._total_modulations,
            "trace_length": len(self._trace),
            "step_idx": self._step_idx,
            "eligibility": self._eligibility.stats,
        }
