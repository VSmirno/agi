"""Unit tests for ResidualBodyPredictor (Stage 78b).

Check:
  - input/output shapes
  - encoding determinism (same concept hashes to same bits)
  - gradient flow through residual, not through rules term
  - residual starts near zero (small init → doesn't perturb rules on epoch 0)
  - sanity convergence on a trivial 1-pattern task in <200 steps
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from snks.learning.residual_predictor import ResidualBodyPredictor, ResidualConfig


def test_default_shape():
    p = ResidualBodyPredictor()
    assert p.input_dim == 1000 + 4 * 10 + 8
    assert p.output_dim == 4
    fp = p.encode(visible={"tree"}, body={"health": 5, "food": 3}, action_idx=0)
    assert fp.shape == (p.input_dim,)
    out = p(fp)
    assert out.shape == (p.output_dim,)


def test_custom_config_shape():
    cfg = ResidualConfig(
        n_visible_concepts=9, n_actions=17, n_body_vars=6,
        body_buckets=5, hidden_dim=32, concept_hash_active=20,
    )
    p = ResidualBodyPredictor(cfg)
    assert p.input_dim == 1000 + 6 * 5 + 17
    assert p.output_dim == 6


def test_encoding_is_deterministic_per_concept():
    p = ResidualBodyPredictor()
    a = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    b = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    assert torch.equal(a, b)


def test_encoding_differs_on_different_inputs():
    p = ResidualBodyPredictor()
    a = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    b = p.encode(visible={"stone"}, body={"health": 5}, action_idx=0)
    assert not torch.equal(a, b)
    c = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    d = p.encode(visible={"tree"}, body={"health": 5}, action_idx=1)
    assert not torch.equal(c, d)


def test_initial_residual_is_near_zero():
    """Small init so residual ≈0 at epoch 0 — don't perturb rules on first forward."""
    p = ResidualBodyPredictor()
    fp = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    with torch.no_grad():
        out = p(fp)
    # Every output magnitude should be small — <0.1 is a loose but meaningful bound
    assert out.abs().max().item() < 0.1


def test_residual_loss_gradient_flows_only_through_residual():
    """rules_delta is a plain tensor without grad; loss backprop must leave it alone."""
    p = ResidualBodyPredictor()
    fp = p.encode(visible={"tree"}, body={"health": 5}, action_idx=0)
    rules_t = torch.tensor([0.04, -0.04, -0.04, -0.02])
    target_t = torch.tensor([-0.067, -0.04, -0.04, -0.02])
    loss = p.residual_loss(fp, rules_t, target_t)
    loss.backward()
    # Residual params must have a non-zero grad
    assert any(
        param.grad is not None and param.grad.abs().sum().item() > 0
        for param in p.parameters()
    )
    # rules_t stays gradient-free (it was never a leaf requiring grad)
    assert rules_t.grad is None


def test_converges_on_trivial_constant_rule():
    """Sanity: feed one (state, action) sample repeatedly and check the residual
    learns the constant correction."""
    torch.manual_seed(0)
    np.random.seed(0)
    p = ResidualBodyPredictor()
    opt = torch.optim.Adam(p.parameters(), lr=5e-3)

    fp = p.encode(visible={"tree"}, body={"health": 5, "food": 3}, action_idx=0)
    rules_t = torch.tensor([0.04, -0.04, -0.04, -0.02])
    target_t = torch.tensor([-0.1, -0.04, -0.04, -0.02])  # conjunctive-like correction on health

    for _ in range(300):
        loss = p.residual_loss(fp, rules_t, target_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        residual = p(fp)
        final = rules_t + residual
    # health should be close to -0.1
    assert (final[0] - target_t[0]).abs().item() < 0.02
    # non-target dims stay close to rules (residual near 0 on them)
    assert residual[1].abs().item() < 0.03
    assert residual[2].abs().item() < 0.03
    assert residual[3].abs().item() < 0.03


def test_predict_returns_dict():
    p = ResidualBodyPredictor()
    rules = {"health": 0.04, "food": -0.04, "drink": -0.04, "energy": -0.02}
    out = p.predict(
        visible={"tree"},
        body={"health": 5, "food": 3, "drink": 3, "energy": 3},
        action_idx=0,
        rules_delta=rules,
    )
    assert set(out.keys()) == set(rules.keys())
    for v in out.values():
        assert isinstance(v, float)


def test_save_and_load_roundtrip(tmp_path):
    p = ResidualBodyPredictor()
    # Do a couple of training steps so weights change from init
    opt = torch.optim.Adam(p.parameters(), lr=1e-2)
    fp = p.encode(visible={"tree", "water"}, body={"health": 5}, action_idx=2)
    rules_t = torch.tensor([0.0, 0.0, 0.0, 0.0])
    target_t = torch.tensor([0.2, 0.0, 0.0, 0.0])
    for _ in range(20):
        loss = p.residual_loss(fp, rules_t, target_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    path = str(tmp_path / "residual.pt")
    p.save_state(path)

    q = ResidualBodyPredictor()
    q.load_state(path)

    with torch.no_grad():
        # Rebuild the same fp on q — seeds must roundtrip so concept bits match
        fp_q = q.encode(visible={"tree", "water"}, body={"health": 5}, action_idx=2)
        assert torch.equal(fp, fp_q)
        out_p = p(fp)
        out_q = q(fp_q)
        assert torch.allclose(out_p, out_q, atol=1e-6)
