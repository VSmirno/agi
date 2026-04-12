"""Tests for Stage 83: VectorState + simulate_forward."""

from __future__ import annotations

import pytest

from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.vector_sim import (
    VectorState,
    VectorPlan,
    VectorPlanStep,
    simulate_forward,
    score_trajectory,
)


@pytest.fixture
def model():
    return VectorWorldModel(dim=4096, n_locations=2000, seed=42)


@pytest.fixture
def base_state():
    return VectorState(
        inventory={"wood": 0},
        body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0},
        player_pos=(10, 10),
    )


class TestVectorState:
    def test_apply_effect_adds_inventory(self, base_state):
        new = base_state.apply_effect({"wood": 1})
        assert new.inventory["wood"] == 1
        assert base_state.inventory["wood"] == 0  # immutable

    def test_apply_effect_clamps_body(self, base_state):
        new = base_state.apply_effect({"health": -20})
        assert new.body["health"] == 0.0

    def test_apply_effect_body_max_9(self, base_state):
        new = base_state.apply_effect({"health": 5})
        assert new.body["health"] == 9.0  # clamped

    def test_is_dead_when_health_zero(self, base_state):
        dead = base_state.apply_effect({"health": -10})
        assert dead.is_dead(["health"])

    def test_is_dead_false_when_healthy(self, base_state):
        assert not base_state.is_dead(["health"])

    def test_copy_independent(self, base_state):
        copied = base_state.copy()
        copied.inventory["wood"] = 5
        assert base_state.inventory["wood"] == 0

    def test_to_vector_returns_correct_dim(self, base_state, model):
        vec = base_state.to_vector(model)
        assert vec.shape == (4096,)


class TestSimulateForward:
    def test_one_step_prediction(self, model, base_state):
        # Teach model: do tree → wood +1
        for _ in range(10):
            model.learn("tree", "do", {"wood": 1})

        plan = VectorPlan(
            steps=[VectorPlanStep(action="do", target="tree")],
        )
        traj = simulate_forward(model, plan, base_state)
        assert not traj.terminated
        assert len(traj.states) == 2  # initial + 1 step
        final = traj.final_state
        assert final.inventory.get("wood", 0) >= 1

    def test_chain_wood_table_sword(self, model, base_state):
        # Teach multi-step chain
        for _ in range(10):
            model.learn("tree", "do", {"wood": 1})
            model.learn("table", "place", {"table": 1})
            model.learn("wood_sword", "make", {"wood_sword": 1})

        plan = VectorPlan(steps=[
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="place", target="table"),
            VectorPlanStep(action="make", target="wood_sword"),
        ])
        traj = simulate_forward(model, plan, base_state)
        assert not traj.terminated
        # Should have gained at least some inventory items
        assert traj.total_inventory_gain() > 0

    def test_terminates_on_death(self, model, base_state):
        # Teach: proximity zombie → health -9
        for _ in range(10):
            model.learn("zombie", "proximity", {"health": -9})

        plan = VectorPlan(steps=[
            VectorPlanStep(action="proximity", target="zombie"),
            VectorPlanStep(action="do", target="tree"),  # should not reach
        ])
        traj = simulate_forward(model, plan, base_state, vital_vars=["health"])
        assert traj.terminated

    def test_unknown_action_skipped(self, model, base_state):
        # Don't teach anything about "unknown_thing"
        plan = VectorPlan(steps=[
            VectorPlanStep(action="do", target="unknown_thing"),
        ])
        traj = simulate_forward(model, plan, base_state)
        assert not traj.terminated
        # State should be essentially unchanged (no effect applied)
        assert traj.final_state.inventory == base_state.inventory


class TestScoreTrajectory:
    def test_survived_beats_dead(self, model, base_state):
        alive_plan = VectorPlan(steps=[])
        dead_state = base_state.apply_effect({"health": -10})

        alive_traj = simulate_forward(model, alive_plan, base_state)
        dead_traj = simulate_forward(model, alive_plan, dead_state,
                                     vital_vars=["health"])

        s_alive = score_trajectory(alive_traj)
        s_dead = score_trajectory(dead_traj)
        assert s_alive > s_dead

    def test_higher_gain_beats_lower(self, model, base_state):
        # Teach do tree → wood +1
        for _ in range(10):
            model.learn("tree", "do", {"wood": 1})

        short_plan = VectorPlan(steps=[
            VectorPlanStep(action="do", target="tree"),
        ])
        long_plan = VectorPlan(steps=[
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
            VectorPlanStep(action="do", target="tree"),
        ])

        short_traj = simulate_forward(model, short_plan, base_state)
        long_traj = simulate_forward(model, long_plan, base_state)

        s_short = score_trajectory(short_traj)
        s_long = score_trajectory(long_traj)
        # Long chain has more total_gain → scores higher
        assert s_long >= s_short
