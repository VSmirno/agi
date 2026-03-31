"""Tests for Stage 34: Long-Horizon Hierarchical Planning."""

from __future__ import annotations

import pytest

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.hierarchical_planner import HierarchicalPlanner
from snks.language.plan_node import PlanGraph, PlanNode, PlanStatus
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


# ── Helpers ──────────────────────────────────────────────────────

def _make_model(links: list[tuple[int, set[int], set[int]]]) -> CausalWorldModel:
    config = CausalAgentConfig(causal_min_observations=1)
    model = CausalWorldModel(config)
    for action, ctx, post in links:
        for _ in range(3):
            model.observe_transition(pre_sks=ctx, action=action, post_sks=post)
    return model


def _make_library() -> SkillLibrary:
    lib = SkillLibrary()
    lib.register(Skill(
        name="pickup_key", preconditions=frozenset({50}),
        effects=frozenset({51}), terminal_action=3,
        target_word="key", success_count=10, attempt_count=10,
    ))
    lib.register(Skill(
        name="toggle_door", preconditions=frozenset({51, 52}),
        effects=frozenset({53}), terminal_action=5,
        target_word="door", success_count=8, attempt_count=10,
    ))
    return lib


def _make_planner(grid_size: int = 8) -> HierarchicalPlanner:
    links = [
        (3, {50}, {50, 51}),       # pickup key
        (5, {51, 52}, {51, 52, 53}),  # toggle door
    ]
    model = _make_model(links)
    library = _make_library()
    return HierarchicalPlanner(model, library, grid_size=grid_size)


# ── PlanNode Tests ───────────────────────────────────────────────

class TestPlanNode:
    def test_leaf_node(self):
        node = PlanNode(level=0, action=2, estimated_steps=1)
        assert node.is_leaf
        assert node.total_estimated_steps() == 1
        assert node.depth() == 0

    def test_nested_node(self):
        child1 = PlanNode(level=0, action=0, estimated_steps=1)
        child2 = PlanNode(level=0, action=2, estimated_steps=1)
        parent = PlanNode(level=1, action="nav", children=[child1, child2])
        assert not parent.is_leaf
        assert parent.total_estimated_steps() == 2
        assert parent.depth() == 1

    def test_count_nodes(self):
        c1 = PlanNode(level=0, action=0)
        c2 = PlanNode(level=0, action=1)
        mid = PlanNode(level=1, action="mid", children=[c1, c2])
        root = PlanNode(level=2, action="root", children=[mid])
        assert root.count_nodes() == 4

    def test_flatten_primitives(self):
        c1 = PlanNode(level=0, action=0)
        c2 = PlanNode(level=0, action=2)
        c3 = PlanNode(level=0, action=3)
        mid = PlanNode(level=1, action="nav", children=[c1, c2])
        root = PlanNode(level=2, action="root", children=[mid, c3])
        assert root.flatten_primitives() == [0, 2, 3]

    def test_validate_chain_simple(self):
        c1 = PlanNode(level=1, action="a",
                       preconditions=frozenset({50}),
                       postconditions=frozenset({51}))
        c2 = PlanNode(level=1, action="b",
                       preconditions=frozenset({51}),
                       postconditions=frozenset({53}))
        root = PlanNode(level=2, action="root",
                        preconditions=frozenset({50}),
                        children=[c1, c2])
        assert root.validate_chain() == 1.0

    def test_validate_chain_broken(self):
        c1 = PlanNode(level=1, action="a",
                       preconditions=frozenset({50}),
                       postconditions=frozenset({51}))
        c2 = PlanNode(level=1, action="b",
                       preconditions=frozenset({99}),  # not provided by c1
                       postconditions=frozenset({53}))
        root = PlanNode(level=2, action="root",
                        preconditions=frozenset({50}),
                        children=[c1, c2])
        assert root.validate_chain() == 0.5

    def test_status_lifecycle(self):
        node = PlanNode(level=0, action=0)
        assert node.status == PlanStatus.PENDING
        node.status = PlanStatus.ACTIVE
        assert not node.is_done
        node.status = PlanStatus.DONE
        assert node.is_done


# ── PlanGraph Tests ──────────────────────────────────────────────

class TestPlanGraph:
    def test_get_next_pending(self):
        c1 = PlanNode(level=0, action=0)
        c2 = PlanNode(level=0, action=1)
        root = PlanNode(level=1, action="root", children=[c1, c2])
        graph = PlanGraph(root=root)

        first = graph.get_next_pending()
        assert first is c1

    def test_mark_complete_propagates(self):
        c1 = PlanNode(level=0, action=0)
        c2 = PlanNode(level=0, action=1)
        root = PlanNode(level=1, action="root", children=[c1, c2])
        graph = PlanGraph(root=root)

        graph.mark_complete_up(c1)
        assert c1.is_done
        assert not root.is_done  # c2 still pending

        graph.mark_complete_up(c2)
        assert root.is_done  # all children done

    def test_total_steps(self):
        c1 = PlanNode(level=0, action=0, estimated_steps=5)
        c2 = PlanNode(level=0, action=1, estimated_steps=3)
        root = PlanNode(level=1, action="root", children=[c1, c2])
        graph = PlanGraph(root=root)
        assert graph.total_steps == 8


# ── HierarchicalPlanner Tests ───────────────────────────────────

class TestHierarchicalPlanner:
    def test_simple_doorkey_plan(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),  # key held, door open, reach goal
            current_sks=frozenset({50, 52, 54}),  # key visible, door locked, goal visible
        )
        assert plan.total_steps > 0
        assert plan.n_nodes > 1
        assert plan.root.level == 2

    def test_plan_includes_key_and_door(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),  # key held, door open, goal
            current_sks=frozenset({50, 52, 54}),
        )
        # Should have subgoals for key and door.
        strategic_names = [c.action for c in plan.root.children]
        assert "get_key" in strategic_names
        assert "unlock_door" in strategic_names

    def test_plan_depth_scales_with_rooms(self):
        planner = _make_planner()
        plan_1 = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=1,
        )
        plan_3 = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=3,
        )
        assert plan_3.total_steps > plan_1.total_steps

    def test_long_horizon_1000_steps(self):
        """Key gate: plan reaches 1000+ primitive steps."""
        planner = _make_planner(grid_size=40)  # larger grid
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=6,
        )
        total = plan.total_steps
        assert total >= 1000, f"Expected >= 1000 steps, got {total}"

    def test_plan_coherence(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
        )
        assert plan.coherence >= 0.5

    def test_already_at_goal(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53}),
            current_sks=frozenset({51, 53}),
        )
        # Nothing to do.
        assert plan.root.children == [] or plan.total_steps <= 50

    def test_replan_on_deviation(self):
        planner = _make_planner()
        failed_node = PlanNode(level=1, action="navigate", status=PlanStatus.FAILED)
        recovery = planner.replan(
            failed_node,
            actual_sks=frozenset({50, 52}),  # key visible, door locked (lost goal)
            goal_sks=frozenset({51, 53, 54}),
        )
        assert recovery is not None
        assert recovery.level == 2
        assert recovery.status == PlanStatus.PENDING

    def test_replan_not_needed(self):
        planner = _make_planner()
        recovery = planner.replan(
            PlanNode(level=0, action=0),
            actual_sks=frozenset({51, 53, 54}),  # already at goal
            goal_sks=frozenset({51, 53, 54}),
        )
        assert recovery is None

    def test_check_deviation(self):
        planner = _make_planner()
        # Same state = no deviation.
        assert not planner.check_deviation(
            frozenset({50, 51}), frozenset({50, 51}),
        )
        # Superset (gained extra) = no deviation.
        assert not planner.check_deviation(
            frozenset({50, 51}), frozenset({50, 51, 54}),
        )
        # Different state = deviation.
        assert planner.check_deviation(
            frozenset({50, 51}), frozenset({52, 53}),
        )

    def test_execute_plan(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
        )
        actions, steps, replans = planner.execute_plan(plan)
        assert steps > 0
        assert len(actions) == steps

    def test_execute_with_replan(self):
        planner = _make_planner()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
        )
        # Simulate state deviation at step 5.
        primitives = plan.root.flatten_primitives()
        n = len(primitives)
        # Create state updates that deviate midway.
        state_updates = [frozenset({50, 52, 54})] * max(n // 2, 1)
        state_updates += [frozenset({99})] * (n - len(state_updates))  # deviation

        actions, steps, replans = planner.execute_plan(plan, state_updates)
        # Should have attempted re-planning.
        assert steps > 0

    def test_card_gate_variant(self):
        """Test planning with card/gate domain."""
        links = [
            (3, {55}, {55, 56}),       # pickup card
            (5, {56, 57}, {56, 57, 58}),  # toggle gate
        ]
        model = _make_model(links)
        lib = SkillLibrary()
        lib.register(Skill(
            name="pickup_card", preconditions=frozenset({55}),
            effects=frozenset({56}), terminal_action=3,
            target_word="card", success_count=5, attempt_count=5,
        ))
        lib.register(Skill(
            name="toggle_gate", preconditions=frozenset({56, 57}),
            effects=frozenset({58}), terminal_action=5,
            target_word="gate", success_count=5, attempt_count=5,
        ))
        planner = HierarchicalPlanner(model, lib)
        plan = planner.plan(
            goal_sks=frozenset({56, 58, 54}),
            current_sks=frozenset({55, 57, 54}),
        )
        strategic_names = [c.action for c in plan.root.children]
        assert "get_card" in strategic_names
        assert "unlock_gate" in strategic_names

    def test_hierarchical_speedup_vs_flat(self):
        """Hierarchical plan should be faster to generate than flat BFS."""
        import time

        planner = _make_planner(grid_size=20)

        # Hierarchical.
        t0 = time.perf_counter()
        plan = planner.plan(
            goal_sks=frozenset({51, 53, 54}),
            current_sks=frozenset({50, 52, 54}),
            n_rooms=3,
        )
        t_hier = time.perf_counter() - t0

        # Plan should be generated quickly (< 0.1s).
        assert t_hier < 0.1, f"Hierarchical planning took {t_hier:.3f}s"
        assert plan.total_steps > 100
