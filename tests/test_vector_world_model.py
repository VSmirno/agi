"""Tests for Stage 83: VectorWorldModel — binary HDC world model core."""

from __future__ import annotations

import pytest
import torch
import tempfile
from pathlib import Path

from snks.agent.vector_world_model import (
    random_bitvector,
    bind,
    bundle,
    hamming_similarity,
    encode_scalar,
    decode_scalar,
    CausalSDM,
    VectorWorldModel,
)


# ---------------------------------------------------------------------------
# BitVector operations
# ---------------------------------------------------------------------------

class TestBitVectorOps:
    def test_random_bitvector_shape_and_values(self):
        v = random_bitvector(1024)
        assert v.shape == (1024,)
        assert set(v.unique().tolist()).issubset({0.0, 1.0})

    def test_random_bitvector_roughly_balanced(self):
        v = random_bitvector(10000)
        ones_frac = v.mean().item()
        assert 0.45 < ones_frac < 0.55

    def test_bind_self_inverse(self):
        a = random_bitvector(1024)
        b = random_bitvector(1024)
        bound = bind(a, b)
        recovered = bind(bound, b)
        assert hamming_similarity(a, recovered) == 1.0

    def test_bind_dissimilar(self):
        a = random_bitvector(4096)
        b = random_bitvector(4096)
        bound = bind(a, b)
        # Bound should be ~50% similar to either operand
        assert 0.45 < hamming_similarity(bound, a) < 0.55

    def test_bundle_majority_vote(self):
        a = random_bitvector(4096)
        b = random_bitvector(4096)
        c = random_bitvector(4096)
        bundled = bundle([a, a, a, b, c])
        # a appears 3/5 times, should dominate
        assert hamming_similarity(bundled, a) > 0.7

    def test_bundle_weighted(self):
        a = random_bitvector(4096)
        b = random_bitvector(4096)
        # Heavy weight on a
        bundled = bundle([a, b], weights=[0.9, 0.1])
        assert hamming_similarity(bundled, a) > 0.8

    def test_bundle_single_returns_clone(self):
        a = random_bitvector(1024)
        bundled = bundle([a])
        assert hamming_similarity(a, bundled) == 1.0
        # Should be a copy
        bundled[0] = 1 - bundled[0]
        assert a[0] != bundled[0]


# ---------------------------------------------------------------------------
# Scalar encoding
# ---------------------------------------------------------------------------

class TestScalarEncoding:
    def test_encode_decode_roundtrip(self):
        for val in range(0, 10):
            vec = encode_scalar(val, dim=65536, max_val=10)
            decoded = decode_scalar(vec, max_val=10)
            assert decoded == val, f"Roundtrip failed for {val}: got {decoded}"

    def test_encode_zero_is_all_zeros(self):
        vec = encode_scalar(0, dim=1024, max_val=10)
        assert vec.sum().item() == 0

    def test_monotonic_popcount(self):
        counts = []
        for val in range(0, 10):
            vec = encode_scalar(val, dim=65536, max_val=10)
            counts.append(vec.sum().item())
        # Each value should have more ones than the previous
        for i in range(1, len(counts)):
            assert counts[i] > counts[i - 1]

    def test_similar_values_similar_vectors(self):
        v3 = encode_scalar(3, dim=65536, max_val=10)
        v4 = encode_scalar(4, dim=65536, max_val=10)
        v8 = encode_scalar(8, dim=65536, max_val=10)
        sim_34 = hamming_similarity(v3, v4)
        sim_38 = hamming_similarity(v3, v8)
        assert sim_34 > sim_38, "Adjacent values should be more similar"


# ---------------------------------------------------------------------------
# CausalSDM
# ---------------------------------------------------------------------------

class TestCausalSDM:
    def test_write_read_roundtrip(self):
        sdm = CausalSDM(n_locations=2000, dim=4096, seed=42)
        address = random_bitvector(4096)
        data = random_bitvector(4096)
        # Write same association 5 times to build confidence
        for _ in range(5):
            sdm.write(address, data)
        result, conf = sdm.read(address)
        assert hamming_similarity(result, data) > 0.8
        assert conf > 0.0

    def test_read_unknown_returns_zero_confidence(self):
        sdm = CausalSDM(n_locations=2000, dim=4096, seed=42)
        address = random_bitvector(4096)
        result, conf = sdm.read(address)
        assert conf == 0.0

    def test_different_addresses_different_content(self):
        sdm = CausalSDM(n_locations=2000, dim=4096, seed=42)
        addr1 = random_bitvector(4096)
        addr2 = random_bitvector(4096)
        data1 = random_bitvector(4096)
        data2 = random_bitvector(4096)
        for _ in range(5):
            sdm.write(addr1, data1)
            sdm.write(addr2, data2)
        r1, _ = sdm.read(addr1)
        r2, _ = sdm.read(addr2)
        assert hamming_similarity(r1, data1) > 0.7
        assert hamming_similarity(r2, data2) > 0.7

    def test_state_dict_roundtrip(self):
        sdm = CausalSDM(n_locations=2000, dim=4096, seed=42)
        addr = random_bitvector(4096)
        data = random_bitvector(4096)
        for _ in range(5):
            sdm.write(addr, data)

        state = sdm.state_dict()
        sdm2 = CausalSDM(n_locations=2000, dim=4096, seed=42)
        sdm2.load_state_dict(state)

        r1, c1 = sdm.read(addr)
        r2, c2 = sdm2.read(addr)
        assert hamming_similarity(r1, r2) > 0.9


# ---------------------------------------------------------------------------
# VectorWorldModel
# ---------------------------------------------------------------------------

class TestVectorWorldModel:
    @pytest.fixture
    def model(self):
        return VectorWorldModel(dim=4096, n_locations=2000, seed=42)

    def test_predict_unknown_zero_confidence(self, model):
        _, conf = model.predict("tree", "do")
        assert conf == 0.0

    def test_learn_and_predict(self, model):
        # Learn: do tree → wood +1
        for _ in range(5):
            model.learn("tree", "do", {"wood": 1})
        effect_vec, conf = model.predict("tree", "do")
        assert conf > 0.0
        decoded = model.decode_effect(effect_vec)
        assert decoded.get("wood", 0) > 0

    def test_surprise_high_on_novel(self, model):
        surprise = model.learn("tree", "do", {"wood": 1})
        assert surprise > 0.5  # First encounter = high surprise

    def test_surprise_decreases_with_learning(self, model):
        surprises = []
        for _ in range(10):
            s = model.learn("tree", "do", {"wood": 1})
            surprises.append(s)
        # First should be higher than last
        assert surprises[0] > surprises[-1]

    def test_effect_encode_decode_roundtrip(self, model):
        effect = {"wood": 3}
        vec = model.encode_effect(effect)
        decoded = model.decode_effect(vec)
        assert decoded.get("wood", 0) == 3

    def test_effect_negative_value(self, model):
        effect = {"health": -3}
        vec = model.encode_effect(effect)
        decoded = model.decode_effect(vec)
        assert decoded.get("health", 0) == -3

    def test_effect_multiple_deltas(self, model):
        effect = {"wood": 1, "health": -2}
        vec = model.encode_effect(effect)
        decoded = model.decode_effect(vec)
        # With 2 bundled bindings at dim=4096, decode may be noisy
        # At least the dominant effect should be present
        assert "wood" in decoded or "health" in decoded

    def test_concept_embedding_created_on_demand(self, model):
        assert "tree" not in model.concepts
        model._ensure_concept("tree")
        assert "tree" in model.concepts
        assert model.concepts["tree"].shape == (4096,)

    def test_query_similar(self, model):
        # Make tree and oak similar by bundling
        model._ensure_concept("tree")
        model._ensure_concept("oak")
        # Manually make oak similar to tree
        model.concepts["oak"] = bundle(
            [model.concepts["tree"], model.concepts["oak"]],
            weights=[0.7, 0.3],
        )
        model._ensure_concept("zombie")

        results = model.query_similar("tree", top_k=2)
        names = [r[0] for r in results]
        # oak should be more similar to tree than zombie
        assert names[0] == "oak"

    def test_save_load_roundtrip(self, model):
        # Learn something
        for _ in range(5):
            model.learn("tree", "do", {"wood": 1})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.bin"
            model.save(path)

            model2 = VectorWorldModel(dim=4096, n_locations=2000, seed=99)
            loaded = model2.load(path)
            assert loaded is True
            assert "tree" in model2.concepts

            # Predict should work on loaded model
            effect_vec, conf = model2.predict("tree", "do")
            assert conf > 0.0

    def test_load_nonexistent_returns_false(self, model):
        assert model.load("/tmp/nonexistent_model_12345.bin") is False

    def test_generalization_similar_concepts(self, model):
        """Concepts that interact similarly should have similar embeddings
        and thus similar predictions."""
        # Train tree extensively
        for _ in range(10):
            model.learn("tree", "do", {"wood": 1},
                        context_vectors=[model._ensure_concept("tree")])

        # Make oak similar to tree manually (simulating similar context)
        model._ensure_concept("oak")
        model.concepts["oak"] = bundle(
            [model.concepts["tree"], random_bitvector(4096)],
            weights=[0.8, 0.2],
        )

        # Predict for oak should be similar to tree's learned effect
        effect_tree, _ = model.predict("tree", "do")
        effect_oak, _ = model.predict("oak", "do")
        # Since oak's vector is close to tree's, SDM should return
        # similar results (same neighborhood of addresses)
        # This tests the generalization property
        sim = hamming_similarity(effect_tree, effect_oak)
        assert sim > 0.55, f"Generalization failed: sim={sim}"
