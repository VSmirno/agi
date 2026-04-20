"""Tests for Stage 83: VectorState + simulate_forward."""

from __future__ import annotations

import pytest
from pathlib import Path

from snks.agent.vector_bootstrap import load_from_textbook
from snks.agent.vector_world_model import VectorWorldModel
from snks.agent.vector_sim import (
    DynamicEntityState,
    VectorState,
    VectorPlan,
    VectorPlanStep,
    simulate_forward,
    score_trajectory,
)

TEXTBOOK_PATH = Path(__file__).parent.parent / "configs" / "crafter_textbook.yaml"


@pytest.fixture
def model():
    return VectorWorldModel(dim=8192, n_locations=5000, seed=42)


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
        assert vec.shape == (8192,)


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

    def test_arrow_hit_applies_learned_proximity_effect(self, model, base_state):
        for _ in range(10):
            model.learn("arrow", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(9, 10),
                    velocity=(1, 0),
                )
            ],
        )
        plan = VectorPlan(steps=[VectorPlanStep(action="sleep", target="self")])

        traj = simulate_forward(model, plan, state, vital_vars=["health"])
        assert traj.final_state is not None
        assert traj.final_state.body["health"] == 6.0

    def test_move_can_avoid_arrow_hit(self, model, base_state):
        for _ in range(10):
            model.learn("arrow", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(9, 10),
                    velocity=(1, 0),
                )
            ],
        )
        plan = VectorPlan(steps=[VectorPlanStep(action="move_up", target="self")])

        traj = simulate_forward(model, plan, state, vital_vars=["health"])
        assert traj.final_state is not None
        assert traj.final_state.player_pos == (10, 9)
        assert traj.final_state.body["health"] == 9.0

    def test_baseline_plan_advances_dynamic_threats(self, model, base_state):
        for _ in range(10):
            model.learn("arrow", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="arrow",
                    position=(9, 10),
                    velocity=(1, 0),
                )
            ],
        )
        plan = VectorPlan(steps=[])

        traj = simulate_forward(model, plan, state, vital_vars=["health"])
        assert traj.final_state is not None
        assert len(traj.states) > 2
        assert traj.final_state.body["health"] == 6.0

    def test_benign_dynamic_entity_does_not_trigger_passive_rollout(self, model, base_state):
        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="cow",
                    position=(11, 10),
                    velocity=(1, 0),
                )
            ],
        )
        baseline = simulate_forward(model, VectorPlan(steps=[]), state, vital_vars=["health"])
        sleep = simulate_forward(
            model,
            VectorPlan(steps=[VectorPlanStep(action="sleep", target="self")]),
            state,
            vital_vars=["health"],
        )

        assert len(baseline.states) == 1
        assert len(sleep.states) == 2

    def test_adjacent_zombie_applies_proximity_damage(self, model, base_state):
        model.proximity_ranges["zombie"] = 1
        for _ in range(10):
            model.learn("zombie", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="zombie",
                    position=(11, 10),
                    velocity=None,
                )
            ],
        )

        traj = simulate_forward(model, VectorPlan(steps=[]), state, vital_vars=["health"], horizon=1)
        assert traj.final_state is not None
        assert traj.final_state.body["health"] == 6.0

    def test_move_can_avoid_adjacent_zombie_damage(self, model, base_state):
        model.proximity_ranges["zombie"] = 1
        for _ in range(10):
            model.learn("zombie", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="zombie",
                    position=(11, 10),
                    velocity=None,
                )
            ],
        )
        plan = VectorPlan(steps=[VectorPlanStep(action="move_up", target="self")])

        traj = simulate_forward(model, plan, state, vital_vars=["health"], horizon=1)
        assert traj.final_state is not None
        assert traj.final_state.player_pos == (10, 9)
        assert traj.final_state.body["health"] == 9.0

    def test_skeleton_range_damage_applies_within_proximity_range(self, model, base_state):
        model.proximity_ranges["skeleton"] = 5
        for _ in range(10):
            model.learn("skeleton", "proximity", {"health": -3})

        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="skeleton",
                    position=(12, 10),
                    velocity=None,
                )
            ],
        )

        traj = simulate_forward(model, VectorPlan(steps=[]), state, vital_vars=["health"], horizon=1)
        assert traj.final_state is not None
        assert traj.final_state.body["health"] == 6.0

    def test_bootstrap_loads_proximity_range_facts(self):
        model = VectorWorldModel(dim=8192, n_locations=5000, seed=42)
        load_from_textbook(model, TEXTBOOK_PATH)

        assert model.proximity_ranges["zombie"] == 1
        assert model.proximity_ranges["skeleton"] == 5
        assert model.proximity_ranges["arrow"] == 1


class TestScoreTrajectory:
    def test_survived_beats_dead(self, model, base_state):
        from snks.agent.stimuli import HomeostasisStimulus, StimuliLayer, SurvivalAversion

        alive_plan = VectorPlan(steps=[])
        dead_state = base_state.apply_effect({"health": -10})  # health → 0.0

        alive_traj = simulate_forward(model, alive_plan, base_state)
        dead_traj = type(alive_traj)(
            plan=alive_plan,
            states=[dead_state],
            terminated=True,
            terminated_reason="dead",
        )

        # Use StimuliLayer — HomeostasisStimulus sees health=0 in dead_state.
        # (stimuli=None path: empty plan → terminated=False for both → same survived score)
        stimuli = StimuliLayer([SurvivalAversion(), HomeostasisStimulus(["health"])])
        s_alive = score_trajectory(alive_traj, stimuli=stimuli)
        s_dead = score_trajectory(dead_traj, stimuli=stimuli)
        assert s_alive > s_dead

    def test_goal_progress_beats_fewer_steps(self, model, base_state):
        """With gather_wood goal, more wood progress beats a shorter plan."""
        from snks.agent.goal_selector import Goal

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

        goal = Goal("gather_wood")
        short_traj = simulate_forward(model, short_plan, base_state)
        long_traj = simulate_forward(model, long_plan, base_state)

        s_short = score_trajectory(short_traj, goal=goal)
        s_long = score_trajectory(long_traj, goal=goal)
        # Long chain has higher inventory_delta("wood") → goal_prog higher → scores higher
        assert s_long >= s_short

    def test_baseline_beats_sleep_when_only_rollout_length_differs(self, model, base_state):
        state = VectorState(
            inventory=dict(base_state.inventory),
            body=dict(base_state.body),
            player_pos=(10, 10),
            dynamic_entities=[
                DynamicEntityState(
                    concept_id="zombie",
                    position=(30, 30),
                    velocity=None,
                )
            ],
        )
        baseline = score_trajectory(simulate_forward(model, VectorPlan(steps=[]), state))
        sleep = score_trajectory(
            simulate_forward(
                model,
                VectorPlan(steps=[VectorPlanStep(action="sleep", target="self")]),
                state,
            )
        )

        assert baseline > sleep
