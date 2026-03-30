"""Tests for HAC role vectors (Stage 20)."""

import torch

from snks.language.roles import get_roles, random_hac_vector


class TestRandomHacVector:

    def test_unit_norm(self):
        v = random_hac_vector(2048, seed=42)
        assert abs(v.norm().item() - 1.0) < 1e-5

    def test_deterministic(self):
        v1 = random_hac_vector(2048, seed=42)
        v2 = random_hac_vector(2048, seed=42)
        assert torch.allclose(v1, v2)

    def test_different_seeds_different_vectors(self):
        v1 = random_hac_vector(2048, seed=100)
        v2 = random_hac_vector(2048, seed=101)
        cos = torch.dot(v1, v2).item()
        assert abs(cos) < 0.1  # nearly orthogonal

    def test_respects_dim(self):
        v = random_hac_vector(512, seed=42)
        assert v.shape == (512,)


class TestGetRoles:

    def test_returns_six_roles(self):
        roles = get_roles()
        assert len(roles) == 6
        expected = {"AGENT", "ACTION", "OBJECT", "LOCATION", "GOAL", "ATTR"}
        assert set(roles.keys()) == expected

    def test_all_unit_norm(self):
        roles = get_roles()
        for name, vec in roles.items():
            assert abs(vec.norm().item() - 1.0) < 1e-5, f"{name} not unit norm"

    def test_pairwise_near_orthogonal(self):
        roles = get_roles()
        names = list(roles.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cos = torch.dot(roles[names[i]], roles[names[j]]).item()
                assert abs(cos) < 0.1, f"{names[i]}-{names[j]} cosine={cos:.3f}"

    def test_custom_dim(self):
        roles = get_roles(hac_dim=512)
        for vec in roles.values():
            assert vec.shape == (512,)

    def test_deterministic_across_calls(self):
        r1 = get_roles()
        r2 = get_roles()
        for name in r1:
            assert torch.allclose(r1[name], r2[name])
