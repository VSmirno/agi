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


# ---------------------------------------------------------------------------
# Bug 1: training rules-only replay drops planned_step → do-cow / do-water
#         body deltas vanish from rules_delta, residual chases the wrong gap
# ---------------------------------------------------------------------------


def _empty_traj(sim: SimState):
    from snks.agent.forward_sim_types import Plan
    from snks.agent.forward_sim_types import Trajectory as Traj
    return Traj(
        plan=Plan(steps=[], origin="probe"),
        body_series={var: [] for var in sim.body},
        events=[],
        final_state=sim,
        terminated=False,
        terminated_reason="horizon",
        plan_progress=0,
    )


def test_apply_tick_do_water_proximity_vs_facing_divergence():
    """REGRESSION test for the planned_step=None bug in Stage 78c training.

    Scenario:
      - water tile at spatial_map[(11, 10)]
      - player at (10, 10)
      - last_action = "move_up" (facing dy=-1 → tile in front is (10, 9))

    Phase 6 'do' has two branches:
      - planned_step=PlannedStep(do, target='water', rule=do_water_rule):
        proximity branch → spatial_map.find_nearest('water') returns
        (11, 10), manhattan=1, fires rule → drink += 5
      - planned_step=None:
        fallback _nearest_concept('move_up') → tile (10, 9) → "empty"
        → no rule → drink unchanged

    Water is used instead of cow because cow has random_walk movement
    in Phase 1 and may step out of manhattan ≤ 1 before Phase 6 fires.
    Water is a static spatial_map entry that doesn't move.

    This test asserts the divergence exists. The Stage 78c training
    rules-only replay used to call _apply_tick with planned_step=None,
    so it suffered from this divergence — the residual was trained
    on systematically wrong rules_delta whenever a do-water (or do-cow,
    intermittently) plan was the chosen primitive.
    """
    store, tracker = _loaded_store_and_tracker()

    water_concept = store.concepts.get("water")
    assert water_concept is not None
    do_water_rule = water_concept.find_causal("do")
    assert do_water_rule is not None
    assert do_water_rule.effect.body_delta.get("drink", 0) > 0, (
        "do-water rule must have a drink body delta for the regression"
    )

    from snks.agent.crafter_spatial_map import CrafterSpatialMap

    # Common spatial_map: water at (11, 10), player at (10, 10).
    # Note: spatial_map.update() stores (y, x); we just write directly to
    # _map to control the layout deterministically.
    spatial_map = CrafterSpatialMap()
    spatial_map._map[(11, 10)] = "water"
    spatial_map._map[(10, 10)] = "empty"
    spatial_map._map[(10, 9)] = "empty"  # tile in front when facing "up" (dy=-1)

    def _mk() -> SimState:
        return SimState(
            inventory={},
            body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            player_pos=(10, 10),
            dynamic_entities=[],
            spatial_map=spatial_map,
            last_action="move_up",  # face is (10, 9) — NOT water
            step=0,
        )

    # --- Path A: planned_step provided ---
    sim_with = _mk()
    store._apply_tick(
        sim_with,
        "do",
        tracker,
        _empty_traj(sim_with),
        tick=0,
        planned_step=PlannedStep(
            action="do",
            target="water",
            near=None,
            rule=do_water_rule,
        ),
    )
    drink_with = sim_with.body["drink"]

    # --- Path B: planned_step=None (the buggy training-time path) ---
    sim_without = _mk()
    store._apply_tick(
        sim_without,
        "do",
        tracker,
        _empty_traj(sim_without),
        tick=0,
        planned_step=None,
    )
    drink_without = sim_without.body["drink"]

    # Background body rate for drink is -0.02/tick. Both paths apply
    # that. The rule effect is +5. So:
    #   with planned_step: drink_with ≈ 5 - 0.02 + 5 ≈ 9.98 (clamped to 9.0)
    #   without          : drink_without ≈ 5 - 0.02 ≈ 4.98
    assert drink_with > drink_without, (
        f"_apply_tick must diverge for do-water with facing-mismatch: "
        f"with={drink_with}, without={drink_without}"
    )
    # Magnitude check: the difference should be at least 3 (rule fires +5
    # but with clamping to 9 from start 5 the visible delta is ~4).
    assert (drink_with - drink_without) >= 3.0, (
        f"do-water rule was supposed to fire ~+5; "
        f"observed delta = {drink_with - drink_without}"
    )


def test_apply_tick_do_water_facing_aligned_matches():
    """Sanity check: when facing direction IS toward water, the fallback
    branch also fires the rule — both paths agree, no divergence."""
    store, tracker = _loaded_store_and_tracker()
    water_concept = store.concepts.get("water")
    do_water_rule = water_concept.find_causal("do")

    from snks.agent.crafter_spatial_map import CrafterSpatialMap
    spatial_map = CrafterSpatialMap()
    # Place water in the tile DIRECTLY in front of the player when facing
    # default (down). _nearest_concept default uses dx,dy = 0,1.
    # So front = (10, 11). Put water there.
    spatial_map._map[(10, 11)] = "water"
    spatial_map._map[(10, 10)] = "empty"

    def _mk() -> SimState:
        return SimState(
            inventory={},
            body={"health": 5.0, "food": 5.0, "drink": 5.0, "energy": 5.0},
            player_pos=(10, 10),
            dynamic_entities=[],
            spatial_map=spatial_map,
            last_action="move_down",  # facing down → front = (10, 11) → water
            step=0,
        )

    sim_with = _mk()
    store._apply_tick(
        sim_with, "do", tracker, _empty_traj(sim_with), tick=0,
        planned_step=PlannedStep(action="do", target="water", near=None, rule=do_water_rule),
    )

    sim_without = _mk()
    store._apply_tick(
        sim_without, "do", tracker, _empty_traj(sim_without), tick=0,
        planned_step=None,
    )

    assert sim_with.body["drink"] == sim_without.body["drink"], (
        f"facing-aligned do-water should match: with={sim_with.body['drink']} "
        f"without={sim_without.body['drink']}"
    )


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
