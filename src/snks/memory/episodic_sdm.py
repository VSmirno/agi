"""Stage 76 Phase 3: EpisodicSDM — FIFO episodic memory with action scoring.

Stores experience tuples (state, action, next_state, body_delta) and
retrieves the most similar past episodes by popcount overlap. Scores
actions by deficit-weighted aggregation over recalled episodes, then
samples with softmax (temperature-controlled exploration).

Ideology:
- No hardcoded drive list; action scoring uses tracker.observed_variables().
- No "higher is better" assumption; sign emerges from deficit × delta.
- No derived features; query is raw state SDR.
- Linear scan recall is brute force; LSH/bucketing are future work.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from snks.agent.perception import HomeostaticTracker


@dataclass
class Episode:
    """A single step of agent experience.

    Attributes:
        state_sdr: SDR of state before action, shape (n_bits,) bool.
        action: Action name (str) chosen at that state.
        next_state_sdr: SDR of state after the action.
        body_delta: {var_name: next - before} for observed body variables.
        step: Monotonic step counter (episode-level or global).
        weight: Priority for eviction when the buffer is full. Steps belonging
            to long-surviving episodes get high weight and are preserved longer.
            Low-weight steps are evicted first.
    """

    state_sdr: np.ndarray
    action: str
    next_state_sdr: np.ndarray
    body_delta: dict[str, int]
    step: int
    weight: float = 1.0


@dataclass
class EpisodicSDM:
    """Episodic buffer with priority-eviction on overflow.

    Each stored Episode has a `weight` field. When the buffer is full, the
    slot with the LOWEST weight is evicted (ties broken by first-found).
    This preserves steps from long-surviving episodes at the expense of
    short death-episodes, reducing the "all memories are deaths" failure
    mode seen in the initial FIFO implementation.

    Recall is still a linear scan with bitwise AND + popcount over state SDRs.
    """

    capacity: int = 50_000
    _buffer: list[Episode] = field(default_factory=list)

    def write(self, episode: Episode) -> int:
        """Append a new episode, returning the index it landed at.

        If the buffer is full, evict the lowest-weight slot and replace it.
        Returns the buffer index so callers can later update `weight`
        (e.g., post-episode upweight by final survival length).
        """
        if len(self._buffer) < self.capacity:
            self._buffer.append(episode)
            return len(self._buffer) - 1
        # Buffer is full — evict the slot with the lowest weight
        weights = np.fromiter(
            (ep.weight for ep in self._buffer),
            dtype=np.float64,
            count=len(self._buffer),
        )
        idx = int(np.argmin(weights))
        self._buffer[idx] = episode
        return idx

    def set_weight(self, idx: int, weight: float) -> None:
        """Update the priority weight of a previously-written episode."""
        if 0 <= idx < len(self._buffer):
            self._buffer[idx].weight = float(weight)

    def set_weights(self, indices: list[int], weight: float) -> None:
        """Bulk-update priority for a list of indices.

        Used at episode end to stamp all steps of the just-finished episode
        with its survival length (the longer the episode, the higher the
        priority of every step it contained).
        """
        w = float(weight)
        for idx in indices:
            if 0 <= idx < len(self._buffer):
                self._buffer[idx].weight = w

    def __len__(self) -> int:
        return len(self._buffer)

    def recall(
        self,
        query_sdr: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, Episode]]:
        """Return top-k episodes by popcount overlap with query.

        Args:
            query_sdr: state SDR to match against (shape (n_bits,) bool).
            top_k: maximum number of episodes to return.

        Returns:
            List of (overlap_count, episode) pairs, sorted by overlap
            descending. Empty if buffer is empty.
        """
        if not self._buffer:
            return []
        scored: list[tuple[int, Episode]] = []
        for ep in self._buffer:
            overlap = int(np.logical_and(query_sdr, ep.state_sdr).sum())
            scored.append((overlap, ep))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def count_similar(
        self,
        query_sdr: np.ndarray,
        threshold_ratio: float = 0.5,
    ) -> int:
        """Count episodes whose overlap with query ≥ threshold_ratio × popcount(query).

        Used by bootstrap gate: "≥ N similar episodes exist" → SDM path,
        else fall back to ConceptStore.
        """
        query_popcount = int(query_sdr.sum())
        if query_popcount == 0:
            return 0
        threshold = threshold_ratio * query_popcount
        count = 0
        for ep in self._buffer:
            overlap = int(np.logical_and(query_sdr, ep.state_sdr).sum())
            if overlap >= threshold:
                count += 1
        return count


def score_actions(
    recalled: list[tuple[int, Episode]],
    current_body: dict[str, int],
    tracker: "HomeostaticTracker",
) -> dict[str, float]:
    """Score each action by expected improvement of body state.

    Deficit-weighted aggregation:
    - For each body variable V in tracker.observed_variables():
      deficit[V] = max(0, tracker.observed_max[V] - current_body[V])
    - For each recalled episode with action A:
      score[A] += Σ_V deficit[V] × body_delta_V
    - Average over episodes of same action.

    Sign is emergent: if health typically goes up in "good" episodes and
    down in "bad" ones, the product deficit × delta scores restorative
    actions higher automatically. No hardcoded "higher is better".

    Works for any variable the tracker has observed — adding new body
    variables requires only tracker updates, not this function.

    Args:
        recalled: list of (overlap, episode) from EpisodicSDM.recall.
        current_body: current inventory-style dict with body variable values.
        tracker: HomeostaticTracker with observed_max and observed_variables().

    Returns:
        dict[action_name, mean_score]. Empty if no recalled episodes.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    observed_vars = tracker.observed_variables()
    for _, ep in recalled:
        contribution = 0.0
        for var, delta in ep.body_delta.items():
            if var not in observed_vars:
                continue
            obs_max = tracker.observed_max.get(var, 0)
            current = current_body.get(var, obs_max)
            deficit = max(0, obs_max - current)
            contribution += deficit * delta
        totals[ep.action] += contribution
        counts[ep.action] += 1
    return {
        action: totals[action] / counts[action]
        for action in totals
        if counts[action] > 0
    }


def select_action(
    action_scores: dict[str, float],
    temperature: float = 1.0,
    rng: np.random.RandomState | None = None,
) -> str | None:
    """Softmax sampling over action scores.

    Numerically stable (subtract max before exp). Temperature controls
    stochasticity: high → uniform, low → greedy argmax.

    Args:
        action_scores: dict[action_name, score].
        temperature: softmax temperature. Must be > 0.
        rng: RandomState for reproducibility. Defaults to numpy global.

    Returns:
        Sampled action name, or None if action_scores is empty.
    """
    if not action_scores:
        return None
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    actions = list(action_scores.keys())
    values = np.array([action_scores[a] for a in actions], dtype=np.float64)
    shifted = (values - values.max()) / temperature
    exp_values = np.exp(shifted)
    probs = exp_values / exp_values.sum()
    if rng is None:
        rng = np.random.RandomState()
    idx = rng.choice(len(actions), p=probs)
    return actions[idx]
