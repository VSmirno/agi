"""Stage 77a Commit 5: Tests for MPC agent loop.

Unit tests for score_trajectory, extract_failures, generate_candidate_plans,
DynamicEntityTracker, plus an integration smoke test with a mock env.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_spatial_map import CrafterSpatialMap
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.forward_sim_types import (
    DynamicEntity,
    Failure,
    Plan,
    PlannedStep,
    SimEvent,
    SimState,
    Trajectory,
)
from snks.agent.mpc_agent import (
    DynamicEntityTracker,
    build_sim_state,
    extract_failures,
    generate_candidate_plans,
    outcome_to_verify,
    run_mpc_episode,
    score_trajectory,
    update_spatial_map_from_viewport,
)
from snks.agent.perception import HomeostaticTracker, VisualField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_store():
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    store = ConceptStore()
    tb.load_into(store)
    return store


@pytest.fixture
def tracker(loaded_store):
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    t = HomeostaticTracker()
    t.init_from_textbook(tb.body_block, loaded_store.passive_rules)
    return t


def _mkstate(
    inventory=None,
    body=None,
    player_pos=(10, 10),
    entities=None,
):
    return SimState(
        inventory=dict(inventory or {}),
        body=dict(body or {"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0}),
        player_pos=player_pos,
        dynamic_entities=list(entities or []),
        spatial_map=None,
        last_action=None,
        step=0,
    )


def _mktraj_alive(ticks=20, min_body=4.0, final_body=3.8):
    """Build a synthetic alive trajectory for scoring tests."""
    body_series = {"health": [9.0] * ticks, "food": [8.0] * ticks}
    # Inject the min/final values via the last few entries
    if ticks >= 2:
        body_series["health"][0] = 9.0
        body_series["food"][-1] = 1.0
    return Trajectory(
        plan=Plan(steps=[]),
        body_series=body_series,
        events=[],
        final_state=_mkstate(),
        terminated=False,
        terminated_reason="horizon",
        plan_progress=0,
    )


def _mktraj_dead(death_tick=10):
    body_series = {"health": [], "food": []}
    for i in range(death_tick + 1):
        body_series["health"].append(max(9.0 - i, 0.0))
        body_series["food"].append(9.0 - i * 0.1)
    return Trajectory(
        plan=Plan(steps=[]),
        body_series=body_series,
        events=[],
        final_state=_mkstate(body={"health": 0.0, "food": 8.0, "drink": 9.0, "energy": 9.0}),
        terminated=True,
        terminated_reason="body_dead",
        plan_progress=0,
    )


# ---------------------------------------------------------------------------
# score_trajectory
# ---------------------------------------------------------------------------


class TestScoreTrajectory:
    def test_alive_beats_dead(self, tracker):
        alive = _mktraj_alive()
        dead = _mktraj_dead(death_tick=15)
        alive_score = score_trajectory(alive, tracker)
        dead_score = score_trajectory(dead, tracker)
        assert alive_score > dead_score
        # Primary key: alive=1 > dead=0
        assert alive_score[0] == 1
        assert dead_score[0] == 0

    def test_among_dead_longer_wins(self, tracker):
        """For dead trajectories, longer survival is primary signal."""
        dead_long = _mktraj_dead(death_tick=15)
        dead_short = _mktraj_dead(death_tick=5)
        long_score = score_trajectory(dead_long, tracker)
        short_score = score_trajectory(dead_short, tracker)
        assert long_score > short_score
        # Dead bucket shape: (0, n_ticks, min_body, final_body)
        # Index 1 is n_ticks — longer should be higher
        assert long_score[1] > short_score[1]

    def test_normalization_by_reference_max(self, tracker):
        """Score normalized by reference_max (=9 in Crafter) per var.

        Stage 80 (Bug 3 fix): the alive tuple is now
        (1, has_gain, min_body, n_ticks, final_body), so min_body is at
        index 2 (was index 1). With no inv_gain events, has_gain=0.
        """
        body_series = {"health": [9.0], "food": [9.0], "drink": [9.0], "energy": [9.0]}
        traj = Trajectory(
            plan=Plan(steps=[]),
            body_series=body_series,
            events=[],
            final_state=_mkstate(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        score = score_trajectory(traj, tracker)
        # alive bucket: (1, has_gain, min_body, n_ticks, final_body)
        assert score[0] == 1
        assert score[1] == 0  # no inv_gain events
        # min_body = 4 vars × (9/9) = 4.0
        assert score[2] == pytest.approx(4.0)

    def test_tuple_order_lexicographic(self, tracker):
        """Python tuple comparison gives correct lexicographic ordering.

        Stage 80: alive tuple is (1, has_gain, min_body, n_ticks, final_body).
        """
        # Same has_gain, different min_body
        a = (1, 0, 3.5, 20, 3.5)
        b = (1, 0, 3.4, 20, 3.6)
        assert a > b  # higher min_body wins
        # Different has_gain — gain wins regardless of min_body
        gain_low = (1, 1, 0.1, 20, 0.1)
        nogain_high = (1, 0, 3.9, 20, 3.9)
        assert gain_low > nogain_high
        # Alive vs dead — alive always wins
        c = (0, 99, 99.0, 99.0)
        d = (1, 0, 0.1, 1, 0.1)
        assert d > c


# ---------------------------------------------------------------------------
# extract_failures
# ---------------------------------------------------------------------------


class TestExtractFailures:
    def _mktraj(self, body_series=None, events=None):
        return Trajectory(
            plan=Plan(steps=[]),
            body_series=body_series or {},
            events=events or [],
            final_state=_mkstate(),
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )

    def test_var_depleted_detected(self):
        traj = self._mktraj(body_series={"health": [9.0, 5.0, 0.0, -1.0]})
        failures = extract_failures(traj)
        var_dep = [f for f in failures if f.kind == "var_depleted"]
        assert len(var_dep) == 1
        assert var_dep[0].var == "health"
        assert var_dep[0].step == 2

    def test_attributed_damage_detected(self):
        events = [
            SimEvent(step=3, kind="body_delta", var="health", amount=-2.0, source="zombie"),
            SimEvent(step=5, kind="body_delta", var="health", amount=-2.0, source="zombie"),
        ]
        traj = self._mktraj(events=events)
        failures = extract_failures(traj)
        zombie_failures = [f for f in failures if f.cause == "zombie"]
        assert len(zombie_failures) == 1
        assert zombie_failures[0].step == 3  # first damage tick

    def test_background_events_not_failures(self):
        events = [
            SimEvent(step=1, kind="body_delta", var="food", amount=-0.04, source="_background"),
        ]
        traj = self._mktraj(events=events)
        failures = extract_failures(traj)
        assert len(failures) == 0

    def test_stateful_events_not_attributed(self):
        """Stateful rules (like food > 0 → health regen) are not external threats."""
        events = [
            SimEvent(step=1, kind="body_delta", var="health", amount=-0.5, source="stateful:food"),
        ]
        traj = self._mktraj(events=events)
        failures = extract_failures(traj)
        assert len(failures) == 0

    def test_failures_sorted_by_step(self):
        body_series = {"food": [9.0] * 10 + [0.0], "health": [9.0] * 5 + [0.0] * 6}
        events = [
            SimEvent(step=2, kind="body_delta", var="health", amount=-2.0, source="zombie"),
        ]
        traj = self._mktraj(body_series=body_series, events=events)
        failures = extract_failures(traj)
        steps = [f.step for f in failures]
        assert steps == sorted(steps)


# ---------------------------------------------------------------------------
# generate_candidate_plans
# ---------------------------------------------------------------------------


class TestGenerateCandidatePlans:
    def test_baseline_always_included(self, loaded_store, tracker):
        state = _mkstate()
        candidates = generate_candidate_plans(state, loaded_store, tracker, horizon=20)
        baseline = [p for p in candidates if p.origin == "baseline"]
        assert len(baseline) == 1

    def test_zombie_triggers_remedy_plans(self, loaded_store, tracker):
        """When baseline predicts death from zombie, remedy plans appear."""
        state = _mkstate(entities=[DynamicEntity("zombie", (13, 10))])
        candidates = generate_candidate_plans(state, loaded_store, tracker, horizon=40)
        remedies = [p for p in candidates if p.origin == "remedy"]
        # At least one remedy plan should include combat (do zombie final step)
        assert any(
            p.steps and p.steps[-1].action == "do" and p.steps[-1].target == "zombie"
            for p in remedies
        )

    def test_safe_state_gives_baseline_plus_proactive_remedies(self, loaded_store, tracker):
        """Stage 77a Attempt 2: even in a safe state, proactive remedy plans
        are generated for known dangerous concepts (zombie, skeleton from
        spatial rules) and decaying body vars (food/drink/energy from
        innate_rates). Baseline is still one of the candidates."""
        state = _mkstate()
        candidates = generate_candidate_plans(state, loaded_store, tracker, horizon=10)
        # Must contain baseline
        assert any(c.origin == "baseline" for c in candidates)
        # Must contain proactive remedies (combat + consume plans)
        remedies = [c for c in candidates if c.origin == "remedy"]
        assert len(remedies) > 0


# ---------------------------------------------------------------------------
# DynamicEntityTracker
# ---------------------------------------------------------------------------


class TestDynamicEntityTracker:
    def test_register_dynamic_concept(self):
        et = DynamicEntityTracker()
        et.register_dynamic_concept("zombie")
        assert "zombie" in et.dynamic_concepts

    def test_update_extracts_dynamic_entities(self):
        et = DynamicEntityTracker()
        et.register_dynamic_concept("zombie")
        vf = VisualField()
        # Viewport (row=3, col=4) is the center of 7x9. Player at (10, 10).
        # Detection at (row=3, col=5): one tile right of center.
        # Expected world: wx = 10 + (5-4) = 11, wy = 10 + (3-2) = 11.
        vf.detections = [("zombie", 0.9, 3, 5)]
        et.update(vf, (10, 10))
        assert len(et.entities) == 1
        assert et.entities[0].concept_id == "zombie"
        assert et.entities[0].pos == (11, 11)

    def test_update_filters_non_dynamic(self):
        et = DynamicEntityTracker()
        et.register_dynamic_concept("zombie")
        vf = VisualField()
        vf.detections = [
            ("zombie", 0.9, 3, 5),
            ("tree", 0.9, 3, 5),
        ]
        et.update(vf, (10, 10))
        concepts = {e.concept_id for e in et.entities}
        assert "zombie" in concepts
        assert "tree" not in concepts

    def test_update_refreshes_each_call(self):
        et = DynamicEntityTracker()
        et.register_dynamic_concept("zombie")
        vf1 = VisualField()
        vf1.detections = [("zombie", 0.9, 3, 5)]
        et.update(vf1, (10, 10))

        vf2 = VisualField()
        vf2.detections = []  # no zombie visible
        et.update(vf2, (10, 10))
        assert et.entities == []  # dropped


# ---------------------------------------------------------------------------
# update_spatial_map_from_viewport
# ---------------------------------------------------------------------------


class TestSpatialMapUpdate:
    def test_writes_player_tile_with_near_concept(self):
        sm = CrafterSpatialMap()
        vf = VisualField()
        vf.near_concept = "empty"
        vf.detections = []
        update_spatial_map_from_viewport(sm, vf, (10, 10))
        assert sm._map.get((10, 10)) == "empty"

    def test_writes_viewport_detections(self):
        sm = CrafterSpatialMap()
        vf = VisualField()
        vf.detections = [("tree", 0.9, 3, 5)]
        vf.near_concept = "empty"
        update_spatial_map_from_viewport(sm, vf, (10, 10))
        # tree at viewport (3, 5) → world (11, 11) per coord mapping
        assert sm._map.get((11, 11)) == "tree"


# ---------------------------------------------------------------------------
# outcome_to_verify
# ---------------------------------------------------------------------------


class TestOutcomeToVerify:
    def test_do_gather_returns_item(self):
        assert outcome_to_verify("do", {"wood": 0}, {"wood": 1}) == "wood"

    def test_do_restore_food(self):
        assert outcome_to_verify("do", {"food": 5}, {"food": 9}) == "restore_food"

    def test_place_returns_item(self):
        assert outcome_to_verify("place_table", {"wood": 2}, {"wood": 0}) == "table"

    def test_make_returns_crafted(self):
        assert (
            outcome_to_verify("make_wood_sword", {"wood": 1}, {"wood_sword": 1})
            == "wood_sword"
        )

    def test_no_change_returns_none(self):
        assert outcome_to_verify("do", {}, {}) is None


# ---------------------------------------------------------------------------
# Integration: run_mpc_episode with mock env
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal env that returns scripted observations.

    Body vars decay 0 (no natural decay in mock) and agent can never die
    unless we force it. Useful for smoke-testing the MPC loop end-to-end.
    """

    def __init__(self, max_steps: int = 10):
        self.step_count = 0
        self.max_steps = max_steps
        self.inventory = {
            "health": 9, "food": 9, "drink": 9, "energy": 9,
            "wood": 0,
        }
        self.player_pos = (32, 32)

    def reset(self):
        self.step_count = 0
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        info = {"inventory": dict(self.inventory), "player_pos": self.player_pos}
        return pixels, info

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        # Simulate tree-adjacent wood gain
        if action == "do":
            self.inventory["wood"] = self.inventory.get("wood", 0) + 1
        pixels = np.zeros((64, 64, 3), dtype=np.uint8)
        info = {"inventory": dict(self.inventory), "player_pos": self.player_pos}
        return pixels, 0.0, done, info


class MockSegmenter:
    """Segmenter that always returns an empty viewport (no detections)."""
    pass


def _mock_perceive(pixels, segmenter):
    """Return a VisualField with nothing visible — player is in empty field."""
    vf = VisualField()
    vf.near_concept = "empty"
    vf.detections = []
    return vf


class TestIntegrationSmoke:
    def test_mpc_episode_completes_without_exception(self, loaded_store, tracker):
        """Run a 10-step MPC episode and verify it completes cleanly."""
        env = MockEnv(max_steps=10)
        segmenter = MockSegmenter()
        rng = np.random.RandomState(42)

        result = run_mpc_episode(
            env=env,
            segmenter=segmenter,
            store=loaded_store,
            tracker=tracker,
            rng=rng,
            max_steps=10,
            horizon=15,
            perceive_fn=_mock_perceive,
        )

        assert result["length"] > 0
        assert result["length"] <= 10
        assert "final_inv" in result
        assert "action_counts" in result
        assert "action_entropy" in result
