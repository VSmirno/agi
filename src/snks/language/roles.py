"""HAC role vectors for compositional understanding (Stage 20).

Roles are fixed unit vectors in HAC space, used as "envelope labels"
for bind/unbind operations. bind(ROLE, filler) encodes a role-filler pair;
unbind(ROLE, sentence_hac) extracts the filler.

In 2048-dim space, random unit vectors are nearly orthogonal (cosine ~0.02),
so roles don't interfere during bind/unbind. Capacity: ~100-200 roles.
"""

from __future__ import annotations

import torch
from torch import Tensor


def random_hac_vector(dim: int, seed: int) -> Tensor:
    """Deterministic unit vector in HAC space."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=gen)
    return v / v.norm().clamp(min=1e-8)


# Seed range 100-199 reserved for roles.
_ROLE_SEEDS: dict[str, int] = {
    "AGENT": 100,
    "ACTION": 101,
    "OBJECT": 102,
    "LOCATION": 103,
    "GOAL": 104,
    "ATTR": 105,
}


def get_roles(hac_dim: int = 2048) -> dict[str, Tensor]:
    """Return role vectors for all defined roles.

    Args:
        hac_dim: HAC space dimensionality (must match HACEngine.dim).

    Returns:
        Dict mapping role name to unit-norm (hac_dim,) tensor.
    """
    return {
        name: random_hac_vector(hac_dim, seed)
        for name, seed in _ROLE_SEEDS.items()
    }
