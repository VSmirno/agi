"""Stage 76 v2: AttentionWeights tests.

Covers:
- Update mechanics (bit activation × delta → weight change)
- Clipping behavior
- Zero-delta = no update
- Query mask composition
- Dynamic variable registration
- Integration with EpisodicSDM.recall (weighted similarity)
"""

from __future__ import annotations

import numpy as np

from snks.memory.attention import AttentionWeights
from snks.memory.episodic_sdm import Episode, EpisodicSDM


def make_sdr(n_bits: int = 4096, active: list[int] | None = None) -> np.ndarray:
    bits = np.zeros(n_bits, dtype=bool)
    if active:
        bits[np.asarray(active)] = True
    return bits


class TestAttentionUpdate:
    def test_zero_delta_no_update(self):
        attn = AttentionWeights(n_bits=100)
        sdr = make_sdr(100, active=[1, 2, 3])
        attn.update(sdr, {"health": 0})
        assert "health" not in attn.known_variables()

    def test_positive_delta_increases_active_bit_weights(self):
        attn = AttentionWeights(n_bits=100, lr=0.1)
        sdr = make_sdr(100, active=[5, 10, 15])
        attn.update(sdr, {"health": 3})
        w = attn.weights_for("health")
        assert w is not None
        # Active bits get +0.1 * 3 = +0.3
        assert np.isclose(w[5], 0.3)
        assert np.isclose(w[10], 0.3)
        assert np.isclose(w[15], 0.3)
        # Inactive bits stay zero
        assert w[0] == 0.0
        assert w[99] == 0.0

    def test_negative_delta_decreases_active_bit_weights(self):
        attn = AttentionWeights(n_bits=100, lr=0.1)
        sdr = make_sdr(100, active=[5])
        attn.update(sdr, {"health": -2})
        w = attn.weights_for("health")
        assert np.isclose(w[5], -0.2)

    def test_multiple_updates_accumulate(self):
        attn = AttentionWeights(n_bits=100, lr=0.1)
        sdr = make_sdr(100, active=[5])
        for _ in range(5):
            attn.update(sdr, {"health": 1})
        w = attn.weights_for("health")
        # 5 × 0.1 × 1 = 0.5
        assert np.isclose(w[5], 0.5)

    def test_clip_enforced(self):
        attn = AttentionWeights(n_bits=100, lr=1.0, clip=2.0)
        sdr = make_sdr(100, active=[0])
        attn.update(sdr, {"health": 100})
        w = attn.weights_for("health")
        assert w[0] == 2.0  # clipped at +clip

        attn.update(sdr, {"health": -100})
        assert w[0] == -2.0  # clipped at -clip

    def test_dynamic_variable_registration(self):
        attn = AttentionWeights(n_bits=100)
        assert attn.known_variables() == set()

        sdr = make_sdr(100, active=[0])
        attn.update(sdr, {"health": 1, "food": -1})
        assert attn.known_variables() == {"health", "food"}

        attn.update(sdr, {"mana": 5})
        assert attn.known_variables() == {"health", "food", "mana"}


class TestQueryMask:
    def test_empty_deficits_returns_none(self):
        attn = AttentionWeights(n_bits=100)
        sdr = make_sdr(100, active=[1])
        attn.update(sdr, {"health": 1})
        assert attn.query_mask({}) is None

    def test_zero_total_deficit_returns_none(self):
        attn = AttentionWeights(n_bits=100)
        sdr = make_sdr(100, active=[1])
        attn.update(sdr, {"health": 1})
        assert attn.query_mask({"health": 0}) is None

    def test_no_known_variables_returns_none(self):
        attn = AttentionWeights(n_bits=100)
        assert attn.query_mask({"health": 5}) is None

    def test_mask_has_baseline_and_relevance_components(self):
        """With learned weights, relevant bits get higher mask values."""
        attn = AttentionWeights(n_bits=100, lr=0.5, clip=5.0, mask_baseline=1.0)
        sdr = make_sdr(100, active=[10, 20])
        # Strong correlation: active bits [10, 20] → health +2
        for _ in range(3):
            attn.update(sdr, {"health": 2})
        # weights[health][10] = weights[health][20] = min(clip, 3 * 0.5 * 2) = 3.0

        mask = attn.query_mask({"health": 5})
        assert mask is not None
        assert mask.shape == (100,)
        # Bits 10 and 20 should exceed baseline
        assert mask[10] > 1.0
        assert mask[20] > 1.0
        # Bit 0 stays at baseline
        assert np.isclose(mask[0], 1.0)
        # Relevant bits weight: 1 + (5/5) * |3.0| = 1 + 3 = 4
        assert np.isclose(mask[10], 4.0)

    def test_multiple_deficits_combine(self):
        attn = AttentionWeights(n_bits=100, lr=0.5, clip=5.0)
        # health: bit 10 strongly relevant
        attn.update(make_sdr(100, [10]), {"health": 2})
        attn.update(make_sdr(100, [10]), {"health": 2})
        # food: bit 20 strongly relevant
        attn.update(make_sdr(100, [20]), {"food": 2})
        attn.update(make_sdr(100, [20]), {"food": 2})

        mask = attn.query_mask({"health": 5, "food": 5})
        # Both bits 10 and 20 should be elevated
        assert mask[10] > 1.0
        assert mask[20] > 1.0
        # Neither dominates (equal deficits → equal contribution)
        assert np.isclose(mask[10], mask[20])


class TestAttentionIntegration:
    def test_recall_with_mask_prefers_relevant_bits(self):
        """An episode matching on high-mask bits should rank ahead of one
        matching on low-mask bits."""
        sdm = EpisodicSDM(capacity=10)
        attn = AttentionWeights(n_bits=100, lr=0.5, mask_baseline=1.0)

        # Learn: bit 10 is critical for health (active when health changes)
        attn.update(make_sdr(100, [10]), {"health": 3})
        attn.update(make_sdr(100, [10]), {"health": 3})

        # Episode A: has bit 10 active (matches the critical bit)
        ep_a = Episode(
            state_sdr=make_sdr(100, [10, 50, 60]),
            action="eat",
            next_state_sdr=make_sdr(100),
            body_delta={"health": 2},
            step=0,
        )
        # Episode B: has bits 50, 60 but NOT bit 10
        ep_b = Episode(
            state_sdr=make_sdr(100, [50, 60, 70]),
            action="run",
            next_state_sdr=make_sdr(100),
            body_delta={"health": 0},
            step=1,
        )
        sdm.write(ep_a)
        sdm.write(ep_b)

        # Query: state has bit 10, 50, 60 (same as episode A and partially B)
        query = make_sdr(100, [10, 50, 60])

        # Without mask: both episodes overlap 3 and 2 → A first
        unweighted = sdm.recall(query, top_k=2)
        assert unweighted[0][1].action == "eat"

        # With mask: A's match on critical bit 10 gets amplified
        mask = attn.query_mask({"health": 5})
        weighted = sdm.recall(query, top_k=2, mask=mask)
        assert weighted[0][1].action == "eat"
        # The weighted score should be clearly higher (bit 10 weight > 1.0)
        assert weighted[0][0] > weighted[1][0]

    def test_recall_without_mask_backward_compatible(self):
        """Recall without mask behaves as before (plain popcount)."""
        sdm = EpisodicSDM(capacity=10)
        sdm.write(Episode(
            state_sdr=make_sdr(100, [0, 1, 2]),
            action="a",
            next_state_sdr=make_sdr(100),
            body_delta={},
            step=0,
        ))
        sdm.write(Episode(
            state_sdr=make_sdr(100, [0, 1]),
            action="b",
            next_state_sdr=make_sdr(100),
            body_delta={},
            step=1,
        ))
        results = sdm.recall(make_sdr(100, [0, 1, 2]), top_k=2)
        assert results[0][1].action == "a"
        assert results[0][0] == 3.0
        assert results[1][0] == 2.0
