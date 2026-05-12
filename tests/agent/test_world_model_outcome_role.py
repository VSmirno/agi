"""Tests for VectorWorldModel.learn_outcome / predict_outcome.

The outcome-role binding lives in the same CausalSDM as physics-effect
predictions but at an XOR-distinguished address. These tests verify:
  - write+read roundtrip recovers the outcome
  - per-(concept, action) addresses are independent — knowing the outcome
    of `(tree, do)` doesn't leak into `(zombie, do)`
  - outcome writes don't pollute physics `predict()` reads at the same pair
  - save+load roundtrip preserves outcome writes
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from snks.agent.vector_world_model import VectorWorldModel


# Smoke profile: same dim as world model real config so XOR-orthogonality
# stays meaningful (smaller dim means higher crosstalk floor).
SMOKE_DIM = 8192
SMOKE_LOC = 10000


def _alive_outcome(damage: int = 0) -> dict:
    return {"survived_h": True, "damage_h": damage, "died_to": None}


def _dead_outcome(cause: str = "zombie", damage: int = 9) -> dict:
    return {"survived_h": False, "damage_h": damage, "died_to": cause}


def test_encode_decode_roundtrip() -> None:
    """Encode then decode reproduces the high-level outcome fields."""
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=3)
    # Need to ensure the candidate concepts exist before decode_outcome can
    # find them. learn_outcome / decode_outcome both ensure as needed.
    m._ensure_concept("zombie")

    vec_alive = m.encode_outcome(_alive_outcome(damage=2))
    decoded_alive = m.decode_outcome(vec_alive)
    assert decoded_alive["survived_h"] is True
    assert decoded_alive["damage_h"] <= 4, decoded_alive
    assert decoded_alive["died_to"] is None

    vec_dead = m.encode_outcome(_dead_outcome("zombie", damage=8))
    decoded_dead = m.decode_outcome(vec_dead)
    assert decoded_dead["survived_h"] is False
    assert decoded_dead["damage_h"] >= 5, decoded_dead
    assert decoded_dead["died_to"] == "zombie"


def test_write_then_predict_roundtrip() -> None:
    """A learned outcome is recovered with confidence above the floor."""
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=5)
    for _ in range(5):
        m.learn_outcome("tree", "do", _alive_outcome(damage=0))
    decoded, conf = m.predict_outcome("tree", "do")
    assert decoded is not None, f"Outcome should be recovered, conf={conf:.3f}"
    assert decoded["survived_h"] is True


def test_unwritten_pair_returns_no_recall() -> None:
    """A (concept, action) pair never trained on returns None at low confidence."""
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=7)
    # train one pair so SDM is not empty
    for _ in range(5):
        m.learn_outcome("tree", "do", _alive_outcome())
    # query an unrelated pair
    decoded, conf = m.predict_outcome("zombie", "do")
    assert decoded is None, (
        f"Untrained (zombie, do) should not retrieve an outcome, conf={conf:.3f}"
    )


def test_outcome_role_does_not_pollute_physics_predict() -> None:
    """Writing outcomes for (tree, do) leaves physics predict() reads unchanged."""
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=11)
    # No physics-effect writes for (tree, do) yet — confidence should be near zero.
    _, conf_before = m.predict("tree", "do")
    for _ in range(10):
        m.learn_outcome("tree", "do", _alive_outcome(damage=3))
    _, conf_after = m.predict("tree", "do")
    # Physics predict should be unaffected by outcome-role writes.
    assert abs(conf_after - conf_before) < 0.15, (
        f"Outcome writes should not contaminate physics predict, "
        f"before={conf_before:.3f} after={conf_after:.3f}"
    )


def test_per_pair_outcomes_are_independent() -> None:
    """Different (concept, action) pairs retain distinct outcomes."""
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=13)
    for _ in range(5):
        m.learn_outcome("tree", "do", _alive_outcome(damage=0))
        m.learn_outcome("zombie", "do", _dead_outcome("zombie", damage=9))

    tree_dec, tree_conf = m.predict_outcome("tree", "do")
    zomb_dec, zomb_conf = m.predict_outcome("zombie", "do")
    assert tree_dec is not None and zomb_dec is not None
    assert tree_dec["survived_h"] is True
    assert zomb_dec["survived_h"] is False
    assert zomb_dec["died_to"] == "zombie"


def test_save_load_roundtrip_preserves_outcome_writes() -> None:
    """Save+load brings back the outcome-role learning along with the rest."""
    m1 = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=17)
    for _ in range(5):
        m1.learn_outcome("zombie", "do", _dead_outcome("zombie", damage=9))
    dec_before, conf_before = m1.predict_outcome("zombie", "do")
    assert dec_before is not None

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "wm.pt"
        m1.save(path)

        m2 = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=17)
        ok = m2.load(path)
        assert ok, "load() should report success when the file exists"

        dec_after, conf_after = m2.predict_outcome("zombie", "do")
        assert dec_after is not None
        assert dec_after["survived_h"] is False
        assert dec_after["died_to"] == "zombie"


def test_load_missing_file_returns_false() -> None:
    m = VectorWorldModel(n_locations=SMOKE_LOC, dim=SMOKE_DIM, seed=19)
    assert m.load("/tmp/does-not-exist-vector-world-model.pt") is False
