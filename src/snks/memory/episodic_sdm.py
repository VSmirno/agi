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
    """

    state_sdr: np.ndarray
    action: str
    next_state_sdr: np.ndarray
    body_delta: dict[str, int]
    step: int


@dataclass
class EpisodicSDM:
    """FIFO buffer of Episodes with similarity-based recall.

    Writes append to an unbounded list until capacity; then wrap-around
    overwrite (oldest episode replaced). No indexing — recall is a linear
    scan with bitwise AND and popcount over state SDRs.
    """

    capacity: int = 10_000
    _buffer: list[Episode] = field(default_factory=list)
    _write_idx: int = 0

    def write(self, episode: Episode) -> None:
        """Append or overwrite oldest slot."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(episode)
        else:
            self._buffer[self._write_idx] = episode
            self._write_idx = (self._write_idx + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def recall(
        self,
        query_sdr: np.ndarray,
        top_k: int = 20,
        mask: np.ndarray | None = None,
    ) -> list[tuple[float, Episode]]:
        """Return top-k episodes by similarity to the query.

        Args:
            query_sdr: state SDR to match against (shape (n_bits,) bool).
            top_k: maximum number of episodes to return.
            mask: optional per-bit float weight (shape (n_bits,)) from an
                AttentionWeights module. When provided, similarity becomes
                a weighted sum over matching bits instead of plain popcount.
                Bits with higher mask values dominate the ranking.

        Returns:
            List of (score, episode) pairs, sorted by score descending.
            When mask is None, score is the integer popcount; otherwise
            it is a float weighted sum. Empty list if buffer is empty.
        """
        if not self._buffer:
            return []
        scored: list[tuple[float, Episode]] = []
        for ep in self._buffer:
            matched = np.logical_and(query_sdr, ep.state_sdr)
            if mask is None:
                score = float(int(matched.sum()))
            else:
                score = float(matched.astype(np.float32).dot(mask))
            scored.append((score, ep))
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
    recalled: list[tuple[float, Episode]],
    current_body: dict[str, int],
    tracker: "HomeostaticTracker",
) -> dict[str, float]:
    """Score each action by expected improvement of body state.

    Deficit-weighted aggregation over BODY variables only:
    - body_vars = tracker.body_variables() — the subset with innate decay
      (health, food, drink, energy in Crafter). Inventory items like
      wood/sapling are excluded because their observed_max grows
      unboundedly during collection, which would make wood-positive
      actions dominate scoring regardless of actual survival relevance.
    - For each var V in body_vars:
        deficit[V] = max(0, tracker.observed_max[V] - current_body[V])
    - For each recalled episode with action A:
        score[A] += Σ_V deficit[V] × body_delta_V
    - Average over episodes of same action.

    Sign is emergent: if health typically goes up in "good" episodes and
    down in "bad" ones, the product deficit × delta scores restorative
    actions higher automatically. No hardcoded "higher is better".

    Args:
        recalled: list of (score, episode) from EpisodicSDM.recall.
        current_body: current inventory dict with body variable values.
        tracker: HomeostaticTracker with observed_max and body_variables().

    Returns:
        dict[action_name, mean_score]. Empty if no recalled episodes.
    """
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    body_vars = tracker.body_variables()
    for _, ep in recalled:
        contribution = 0.0
        for var, delta in ep.body_delta.items():
            if var not in body_vars:
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
