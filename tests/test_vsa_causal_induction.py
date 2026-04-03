"""Tests for VSA+SDM few-shot causal induction."""

import torch
import pytest

from snks.agent.vsa_world_model import SDMMemory, VSACodebook


class TestVSAIdentityProperty:
    """bind(X, X) = zero vector — the foundation of generalization."""

    def test_bind_self_is_zero(self):
        cb = VSACodebook(dim=512)
        for color in ["red", "blue", "green", "yellow", "purple", "grey"]:
            v = cb.filler(f"color_{color}")
            result = VSACodebook.bind(v, v)
            assert result.sum().item() == 0, f"bind({color},{color}) != zero"

    def test_all_same_pairs_identical(self):
        cb = VSACodebook(dim=512)
        vecs = []
        for color in ["red", "blue", "green"]:
            v = cb.filler(f"color_{color}")
            vecs.append(VSACodebook.bind(v, v))
        assert torch.equal(vecs[0], vecs[1])
        assert torch.equal(vecs[1], vecs[2])

    def test_different_pairs_not_zero(self):
        cb = VSACodebook(dim=512)
        v1 = cb.filler("color_red")
        v2 = cb.filler("color_blue")
        result = VSACodebook.bind(v1, v2)
        assert result.sum().item() != 0


class TestSDMCausalLearning:
    """SDM learns same-color rule and generalizes."""

    def test_same_color_generalization(self):
        cb = VSACodebook(dim=512)
        sdm = SDMMemory(n_locations=1000, dim=512)

        # Train on red, blue
        for kc in ["red", "blue"]:
            for dc in ["red", "blue"]:
                kv = cb.filler(f"color_{kc}")
                dv = cb.filler(f"color_{dc}")
                rel = VSACodebook.bind(kv, dv)
                reward = 1.0 if kc == dc else -1.0
                for _ in range(10):
                    identity = torch.zeros(512)
                    sdm.write(rel, identity, rel, reward)

        # Test on unseen green
        gv = cb.filler("color_green")
        # Same: green-green
        rel_same = VSACodebook.bind(gv, gv)
        identity = torch.zeros(512)
        r_same = sdm.read_reward(rel_same, identity)
        # Different: green-red
        rv = cb.filler("color_red")
        rel_diff = VSACodebook.bind(gv, rv)
        r_diff = sdm.read_reward(rel_diff, identity)

        assert r_same > r_diff, f"same={r_same:.3f} should > diff={r_diff:.3f}"
        assert r_same > 0, f"same-color reward should be positive: {r_same:.3f}"

    def test_arbitrary_mapping_memorization(self):
        cb = VSACodebook(dim=512)
        sdm = SDMMemory(n_locations=1000, dim=512)

        # Train: red→blue (not same color!)
        kv = cb.filler("color_red")
        dv_correct = cb.filler("color_blue")
        dv_wrong = cb.filler("color_green")

        rel_correct = VSACodebook.bind(kv, dv_correct)
        rel_wrong = VSACodebook.bind(kv, dv_wrong)

        identity = torch.zeros(512)
        for _ in range(20):
            sdm.write(rel_correct, identity, rel_correct, 1.0)
            sdm.write(rel_wrong, identity, rel_wrong, -1.0)

        identity = torch.zeros(512)
        r_correct = sdm.read_reward(rel_correct, identity)
        r_wrong = sdm.read_reward(rel_wrong, identity)
        assert r_correct > r_wrong
