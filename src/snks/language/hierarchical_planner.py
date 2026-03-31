"""HierarchicalPlanner: 3-level planning for long horizons (Stage 34).

Decomposes high-level goals into strategic → tactical → primitive plans.
Uses CausalWorldModel for forward prediction and SkillLibrary for abstraction.
Supports re-planning when actual state deviates from expected state.

Architecture:
    Level 2 (Strategic): backward chaining from goal → subgoal sequence
    Level 1 (Tactical):  skill matching → skill execution plans
    Level 0 (Primitive):  concrete environment actions
"""

from __future__ import annotations

from snks.agent.causal_model import CausalLink, CausalWorldModel
from snks.daf.types import CausalAgentConfig
from snks.language.plan_node import PlanGraph, PlanNode, PlanStatus
from snks.language.skill import Skill
from snks.language.skill_library import SkillLibrary


# Standard primitive steps per tactical action (estimates).
_NAV_STEPS_PER_CELL = 3    # avg steps to move one cell (turn + forward)
_INTERACT_STEPS = 2         # steps for pickup/toggle

# State predicate SKS IDs (from grid_perception).
_SKS_KEY_PRESENT = 50
_SKS_KEY_HELD = 51
_SKS_DOOR_LOCKED = 52
_SKS_DOOR_OPEN = 53
_SKS_GOAL_PRESENT = 54
_SKS_CARD_PRESENT = 55
_SKS_CARD_HELD = 56
_SKS_GATE_LOCKED = 57
_SKS_GATE_OPEN = 58


class HierarchicalPlanner:
    """3-level hierarchical planner for long-horizon tasks.

    Given a goal (set of target SKS), produces a plan tree:
    - Level 2: strategic subgoals from backward chaining
    - Level 1: tactical skills from SkillLibrary
    - Level 0: primitive action sequences

    Supports re-planning when execution deviates from expectations.
    """

    def __init__(
        self,
        causal_model: CausalWorldModel,
        skill_library: SkillLibrary,
        grid_size: int = 8,
    ) -> None:
        self._causal = causal_model
        self._library = skill_library
        self._grid_size = grid_size

    def plan(
        self,
        goal_sks: frozenset[int],
        current_sks: frozenset[int],
        n_rooms: int = 1,
    ) -> PlanGraph:
        """Create a hierarchical plan to reach goal_sks from current_sks.

        Args:
            goal_sks: target SKS state.
            current_sks: current SKS state.
            n_rooms: number of rooms (affects navigation estimates).

        Returns:
            PlanGraph with 3-level plan tree.
        """
        # Level 2: strategic decomposition.
        strategic_nodes = self._strategic_plan(goal_sks, current_sks, n_rooms)

        root = PlanNode(
            level=2,
            action="achieve_goal",
            preconditions=current_sks,
            postconditions=goal_sks,
            children=strategic_nodes,
            description=f"Achieve goal: {sorted(goal_sks)}",
        )

        graph = PlanGraph(
            root=root,
            goal_sks=goal_sks,
            initial_sks=current_sks,
        )
        return graph

    def _strategic_plan(
        self,
        goal_sks: frozenset[int],
        current_sks: frozenset[int],
        n_rooms: int,
    ) -> list[PlanNode]:
        """Level 2: backward chaining from goal to current state.

        Identifies what SKS are missing and creates subgoals to achieve them.
        """
        missing = goal_sks - current_sks
        if not missing:
            return []

        subgoals: list[PlanNode] = []
        achieved = set(current_sks)

        # Determine required subgoal chain based on missing predicates.
        # Order matters: get key → unlock door → reach goal.
        ordered_targets = self._order_targets(missing, achieved)

        for target_sks, prereqs, desc in ordered_targets:
            # Create strategic node.
            pre = frozenset(achieved)
            post = frozenset(achieved | {target_sks})

            tactical = self._tactical_plan(target_sks, pre, n_rooms)

            node = PlanNode(
                level=2,
                action=desc,
                preconditions=pre,
                postconditions=post,
                children=tactical,
                description=desc,
            )
            subgoals.append(node)
            achieved.add(target_sks)

        # Final navigation to goal (if goal_present already visible).
        if _SKS_GOAL_PRESENT in achieved or _SKS_GOAL_PRESENT in current_sks:
            nav_node = self._navigation_node(
                frozenset(achieved), n_rooms, "navigate_to_goal",
            )
            subgoals.append(nav_node)

        return subgoals

    def _order_targets(
        self,
        missing: frozenset[int],
        achieved: set[int],
    ) -> list[tuple[int, frozenset[int], str]]:
        """Order missing targets by dependency chain.

        Returns list of (target_sks_id, prerequisites, description).
        """
        targets = []

        # Standard DoorKey ordering.
        if _SKS_KEY_HELD in missing and _SKS_KEY_HELD not in achieved:
            targets.append((_SKS_KEY_HELD, frozenset({_SKS_KEY_PRESENT}), "get_key"))

        if _SKS_DOOR_OPEN in missing and _SKS_DOOR_OPEN not in achieved:
            targets.append((_SKS_DOOR_OPEN, frozenset({_SKS_KEY_HELD, _SKS_DOOR_LOCKED}), "unlock_door"))

        # Card/Gate variant.
        if _SKS_CARD_HELD in missing and _SKS_CARD_HELD not in achieved:
            targets.append((_SKS_CARD_HELD, frozenset({_SKS_CARD_PRESENT}), "get_card"))

        if _SKS_GATE_OPEN in missing and _SKS_GATE_OPEN not in achieved:
            targets.append((_SKS_GATE_OPEN, frozenset({_SKS_CARD_HELD, _SKS_GATE_LOCKED}), "unlock_gate"))

        # Goal navigation (always last).
        if _SKS_GOAL_PRESENT in missing:
            targets.append((_SKS_GOAL_PRESENT, frozenset(), "find_goal"))

        return targets

    def _tactical_plan(
        self,
        target_sks: int,
        current_state: frozenset[int],
        n_rooms: int,
    ) -> list[PlanNode]:
        """Level 1: decompose a strategic goal into tactical skill sequence."""
        nodes: list[PlanNode] = []

        # Navigation to target object.
        nav = self._navigation_node(current_state, n_rooms, f"navigate_to_{target_sks}")
        nodes.append(nav)

        # Interaction: find applicable skill.
        skill = self._find_skill_for_target(target_sks, current_state)
        if skill is not None:
            interact = self._skill_to_node(skill, current_state)
            nodes.append(interact)
        else:
            # Fallback: try causal model query.
            causal_node = self._causal_interaction_node(target_sks, current_state)
            if causal_node is not None:
                nodes.append(causal_node)

        return nodes

    def _navigation_node(
        self,
        current_state: frozenset[int],
        n_rooms: int,
        description: str,
    ) -> PlanNode:
        """Create a navigation node with estimated steps based on grid size."""
        # Estimate: avg distance * steps_per_cell * rooms.
        avg_distance = self._grid_size // 2
        est_steps = avg_distance * _NAV_STEPS_PER_CELL * max(n_rooms, 1)

        # Primitive children: sequence of navigation actions.
        primitives = self._generate_nav_primitives(est_steps)

        return PlanNode(
            level=1,
            action="navigate",
            preconditions=current_state,
            postconditions=current_state,  # navigation doesn't change SKS
            children=primitives,
            estimated_steps=est_steps,
            description=description,
        )

    def _generate_nav_primitives(self, n_steps: int) -> list[PlanNode]:
        """Generate a sequence of primitive navigation actions."""
        primitives = []
        for i in range(n_steps):
            # Cycle through: forward(2), turn_left(0), forward(2), turn_right(1)
            action = 2 if i % 3 != 2 else (0 if (i // 3) % 2 == 0 else 1)
            primitives.append(PlanNode(
                level=0,
                action=action,
                estimated_steps=1,
            ))
        return primitives

    def _find_skill_for_target(
        self,
        target_sks: int,
        current_state: frozenset[int],
    ) -> Skill | None:
        """Find a skill whose effects include the target SKS."""
        for skill in self._library.skills:
            if target_sks in skill.effects:
                return skill
        return None

    def _skill_to_node(self, skill: Skill, current_state: frozenset[int]) -> PlanNode:
        """Convert a Skill to a tactical PlanNode with primitive children."""
        primitives = []
        if skill.terminal_action is not None:
            primitives.append(PlanNode(
                level=0,
                action=skill.terminal_action,
                preconditions=skill.preconditions,
                postconditions=skill.effects,
                estimated_steps=1,
                description=f"execute_{skill.name}",
            ))
        elif skill.sub_skills:
            for sub_name in skill.sub_skills:
                sub = self._library.get(sub_name)
                if sub is not None:
                    primitives.append(self._skill_to_node(sub, current_state))

        return PlanNode(
            level=1,
            action=skill.name,
            preconditions=skill.preconditions,
            postconditions=skill.effects,
            children=primitives,
            estimated_steps=max(len(primitives), _INTERACT_STEPS),
            description=f"skill: {skill.name}",
        )

    def _causal_interaction_node(
        self,
        target_sks: int,
        current_state: frozenset[int],
    ) -> PlanNode | None:
        """Create interaction node from causal model query."""
        results = self._causal.query_by_effect(frozenset({target_sks}))
        if not results:
            return None

        action_id, context, confidence = results[0]
        return PlanNode(
            level=1,
            action=f"causal_action_{action_id}",
            preconditions=frozenset(context),
            postconditions=frozenset({target_sks}),
            children=[PlanNode(level=0, action=action_id, estimated_steps=1)],
            estimated_steps=_INTERACT_STEPS,
            description=f"causal: action {action_id} → SKS {target_sks}",
        )

    # ── Re-planning ──────────────────────────────────────────

    def replan(
        self,
        failed_node: PlanNode,
        actual_sks: frozenset[int],
        goal_sks: frozenset[int],
        n_rooms: int = 1,
    ) -> PlanNode | None:
        """Re-plan from a failed node given actual state.

        Creates a new subtree to recover from deviation.
        """
        missing = goal_sks - actual_sks
        if not missing:
            return None  # already at goal

        # Try to create a new plan from actual state.
        new_strategic = self._strategic_plan(goal_sks, actual_sks, n_rooms)
        if not new_strategic:
            return None

        recovery = PlanNode(
            level=2,
            action="recovery_plan",
            preconditions=actual_sks,
            postconditions=goal_sks,
            children=new_strategic,
            status=PlanStatus.PENDING,
            description="recovery after deviation",
        )
        return recovery

    def check_deviation(
        self,
        expected_sks: frozenset[int],
        actual_sks: frozenset[int],
        threshold: float = 0.3,
    ) -> bool:
        """Check if actual state deviates significantly from expected.

        Returns True if deviation exceeds threshold.
        """
        if not expected_sks:
            return False
        expected = set(expected_sks)
        actual = set(actual_sks)
        union = expected | actual
        if not union:
            return False
        jaccard = len(expected & actual) / len(union)
        return (1.0 - jaccard) > threshold

    # ── Execution ────────────────────────────────────────────

    def execute_plan(
        self,
        plan: PlanGraph,
        state_updates: list[frozenset[int]] | None = None,
    ) -> tuple[list[int], int, int]:
        """Simulate plan execution, optionally with state deviations.

        Args:
            plan: the hierarchical plan.
            state_updates: optional list of actual SKS states at each step
                (for testing re-planning).

        Returns:
            (actions, steps_taken, replans_count)
        """
        actions: list[int] = []
        replans = 0
        current_state = set(plan.initial_sks)

        while True:
            node = plan.get_next_pending()
            if node is None:
                break

            if node.level == 0 and isinstance(node.action, int):
                actions.append(node.action)
                plan.mark_complete_up(node)

                # Apply postconditions.
                current_state |= set(node.postconditions)

                # Check for deviation if state updates provided.
                if state_updates and len(actions) <= len(state_updates):
                    actual = state_updates[len(actions) - 1]
                    if self.check_deviation(frozenset(current_state), actual):
                        recovery = self.replan(
                            node, actual, plan.goal_sks,
                        )
                        if recovery is not None:
                            # Insert recovery into plan.
                            plan.root.children.append(recovery)
                            replans += 1
                        current_state = set(actual)
            else:
                # Non-leaf non-primitive: mark as active, will iterate into children.
                node.status = PlanStatus.ACTIVE
                if node.is_leaf:
                    plan.mark_complete_up(node)

        return actions, len(actions), replans
