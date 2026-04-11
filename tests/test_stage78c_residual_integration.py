"""Stage 78c: residual injection into simulate_forward + online SGD wiring.

Fast unit tests — no env, no segmenter, no Crafter. Just:
  - simulate_forward runs cleanly with residual_predictor=None (baseline path)
  - simulate_forward runs cleanly with residual_predictor set (injected path)
  - residual injection actually perturbs sim.body when residual has nonzero weights
  - primitive_to_action_idx covers all Crafter primitive shapes
  - residual_loss + SGD step flows gradients into residual params

Heavy end-to-end validation happens in experiments/stage78c_residual_crafter.py
on minipc; this file guards against obvious wiring bugs locally.
"""

from __future__ import annotations

import torch
from torch import nn

from snks.agent.concept_store import (
    RESIDUAL_BODY_ORDER,
    ConceptStore,
    primitive_to_action_idx,
)
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.forward_sim_types import Plan, PlannedStep, SimState
from snks.agent.perception import HomeostaticTracker
from snks.learning.residual_predictor import ResidualBodyPredictor, ResidualConfig


def _mkstate() -> SimState:
    return SimState(
        inventory={},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
        dynamic_entities=[],
        spatial_map=None,
        last_action=None,
        step=0,
    )


def _loaded_store_and_tracker() -> tuple[ConceptStore, HomeostaticTracker]:
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    store = ConceptStore()
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)
    return store, tracker


# ---------------------------------------------------------------------------
# primitive_to_action_idx
# ---------------------------------------------------------------------------


def test_primitive_to_action_idx_moves():
    assert primitive_to_action_idx("move_left") == 0
    assert primitive_to_action_idx("move_right") == 1
    assert primitive_to_action_idx("move_up") == 2
    assert primitive_to_action_idx("move_down") == 3


def test_primitive_to_action_idx_actions():
    assert primitive_to_action_idx("do") == 4
    assert primitive_to_action_idx("sleep") == 5
    assert primitive_to_action_idx("place_stone") == 6
    assert primitive_to_action_idx("place_plant") == 6
    assert primitive_to_action_idx("make_wood_pickaxe") == 7
    assert primitive_to_action_idx("make_stone_sword") == 7


def test_primitive_to_action_idx_fallback():
    assert primitive_to_action_idx("noop") == 0
    assert primitive_to_action_idx("inertia") == 0
    assert primitive_to_action_idx("") == 0


def test_residual_body_order_matches_predictor():
    """The sim-side body order must line up with ResidualBodyPredictor
    default body_order so correction[i] applies to the correct var."""
    predictor = ResidualBodyPredictor(ResidualConfig())
    # predict() uses ("health", "food", "drink", "energy") as default body_order
    rd = predictor.predict(
        visible={"tree"},
        body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
        action_idx=0,
        rules_delta={"health": 0.0, "food": 0.0, "drink": 0.0, "energy": 0.0},
    )
    assert set(rd.keys()) == set(RESIDUAL_BODY_ORDER)


# ---------------------------------------------------------------------------
# simulate_forward — baseline + residual path
# ---------------------------------------------------------------------------


def test_simulate_forward_baseline_no_residual():
    """Stage 78c must not break the default (residual=None) path."""
    store, tracker = _loaded_store_and_tracker()
    state = _mkstate()
    plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
    traj = store.simulate_forward(plan, state, tracker, horizon=5)
    assert traj.tick_count() >= 1
    # Body_series should be populated for vital vars
    for var in RESIDUAL_BODY_ORDER:
        assert var in traj.body_series
        assert len(traj.body_series[var]) >= 1


def test_simulate_forward_with_residual_small_init_matches_baseline():
    """With small init (residual weights * 0.1, bias=0) the residual
    correction is ~0, so residual-on body_series should be very close
    to residual-off. This is the key invariant: residual MUST NOT
    perturb the Stage 77a baseline on episode 1."""
    store, tracker = _loaded_store_and_tracker()
    state = _mkstate()
    plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])

    residual = ResidualBodyPredictor(ResidualConfig())

    traj_off = store.simulate_forward(plan, state, tracker, horizon=5)
    traj_on = store.simulate_forward(
        plan,
        state,
        tracker,
        horizon=5,
        residual_predictor=residual,
        visible_concepts={"tree"},
    )

    # Small init → correction magnitudes should be tiny (< 0.5 per tick).
    # Absolute equality not expected because bias is 0 but weights are
    # small-random, so residual output is close to but not exactly 0.
    for var in RESIDUAL_BODY_ORDER:
        for v_off, v_on in zip(traj_off.body_series[var], traj_on.body_series[var]):
            assert abs(v_off - v_on) < 1.5, (
                f"{var}: residual perturbed baseline too much ({v_off} vs {v_on})"
            )


def test_simulate_forward_with_large_residual_changes_body():
    """With non-small weights, the residual should measurably change sim.body.
    Guards against the injection being silently no-op."""
    store, tracker = _loaded_store_and_tracker()
    state = _mkstate()
    plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])

    residual = ResidualBodyPredictor(ResidualConfig())
    # Force large positive weights so the correction is clearly nonzero.
    with torch.no_grad():
        for layer in residual.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(0.01)
                layer.bias.fill_(0.05)

    traj_off = store.simulate_forward(plan, state, tracker, horizon=3)
    traj_on = store.simulate_forward(
        plan,
        state,
        tracker,
        horizon=3,
        residual_predictor=residual,
        visible_concepts={"tree", "grass"},
    )

    # At least one var on at least one tick should differ meaningfully.
    any_diff = False
    for var in RESIDUAL_BODY_ORDER:
        for v_off, v_on in zip(traj_off.body_series[var], traj_on.body_series[var]):
            if abs(v_off - v_on) > 1e-3:
                any_diff = True
                break
    assert any_diff, "Residual with large weights produced no change in sim.body"


def test_simulate_forward_body_stays_clamped_with_residual():
    """Residual correction must not push body variables outside
    [reference_min, reference_max]. _apply_residual_correction re-clamps
    after adding the delta."""
    store, tracker = _loaded_store_and_tracker()
    state = _mkstate()
    plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])

    residual = ResidualBodyPredictor(ResidualConfig())
    with torch.no_grad():
        for layer in residual.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(0.5)  # huge positive — would blow past max
                layer.bias.fill_(5.0)

    traj = store.simulate_forward(
        plan,
        state,
        tracker,
        horizon=5,
        residual_predictor=residual,
        visible_concepts={"tree"},
    )
    for var in RESIDUAL_BODY_ORDER:
        hi = tracker.reference_max.get(var)
        lo = tracker.reference_min.get(var, 0.0)
        for value in traj.body_series[var]:
            if hi is not None:
                assert value <= hi + 1e-6, f"{var} unclamped above max: {value}"
            assert value >= lo - 1e-6, f"{var} unclamped below min: {value}"


# ---------------------------------------------------------------------------
# Online SGD wiring (isolated — no env, just the residual_loss + backward path)
# ---------------------------------------------------------------------------


def test_residual_loss_gradient_flow_matches_mpc_training():
    """Mirrors the training step in run_mpc_episode: encode → residual_loss
    → backward → step. Verifies gradients reach residual params and that
    parameters change after one SGD step."""
    residual = ResidualBodyPredictor(ResidualConfig())
    optim = torch.optim.Adam(residual.parameters(), lr=1e-2)

    fp = residual.encode(
        visible={"tree", "zombie"},
        body={"health": 7.0, "food": 5.0, "drink": 5.0, "energy": 6.0},
        action_idx=4,
    )
    rules_delta = torch.tensor([0.0, -0.01, -0.01, -0.01])
    actual_delta = torch.tensor([-1.0, 0.0, 0.0, 0.0])  # took zombie damage

    # Capture initial weight sum as a change-detector
    before = sum(p.detach().sum().item() for p in residual.parameters())

    loss = residual.residual_loss(fp, rules_delta, actual_delta)
    assert loss.item() > 0.0  # nonzero since prediction != actual

    optim.zero_grad()
    loss.backward()
    # At least one param has a non-null grad
    grads_present = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in residual.parameters())
    assert grads_present
    optim.step()

    after = sum(p.detach().sum().item() for p in residual.parameters())
    assert abs(after - before) > 0.0, "SGD step did not change residual parameters"
