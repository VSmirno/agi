"""Stage 77a Commit 4: Tests for ConceptStore forward sim API.

Covers: simulate_forward outer loop + all 6 phases of _apply_tick,
plan_toward_rule backward chaining, find_remedies world-model query,
confidence threshold filter, termination conditions, score helpers.
"""

from __future__ import annotations

import pytest

from snks.agent.concept_store import (
    ConceptStore,
    _apply_movement,
    _apply_player_move,
    _direction_primitive,
    _manhattan,
    _nearest_concept,
    _step_toward_pos,
)
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.forward_sim_types import (
    DynamicEntity,
    Failure,
    Plan,
    PlannedStep,
    SimState,
)
from snks.agent.perception import HomeostaticTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def loaded_store():
    """ConceptStore with crafter_textbook.yaml loaded."""
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
    last_action=None,
):
    return SimState(
        inventory=dict(inventory or {}),
        body=dict(body or {"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0}),
        player_pos=player_pos,
        dynamic_entities=list(entities or []),
        spatial_map=None,
        last_action=last_action,
        step=0,
    )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_manhattan(self):
        assert _manhattan((0, 0), (3, 4)) == 7
        assert _manhattan((5, 5), (5, 5)) == 0

    def test_step_toward_pos_horizontal(self):
        assert _step_toward_pos((10, 10), (15, 10)) == (11, 10)
        assert _step_toward_pos((10, 10), (5, 10)) == (9, 10)

    def test_step_toward_pos_vertical(self):
        assert _step_toward_pos((10, 10), (10, 15)) == (10, 11)

    def test_step_toward_pos_same(self):
        assert _step_toward_pos((10, 10), (10, 10)) == (10, 10)

    def test_apply_movement_chase(self):
        # Entity at (12, 10), player at (10, 10). Chase → step toward.
        assert _apply_movement((12, 10), (10, 10), "chase_player", 0) == (11, 10)

    def test_apply_movement_flee(self):
        assert _apply_movement((11, 10), (10, 10), "flee_player", 0) == (12, 10)

    def test_apply_movement_random_deterministic(self):
        # Same args → same result (pseudo-random seed from pos + tick)
        a = _apply_movement((5, 5), (10, 10), "random_walk", 3)
        b = _apply_movement((5, 5), (10, 10), "random_walk", 3)
        assert a == b

    def test_apply_movement_unknown(self):
        assert _apply_movement((5, 5), (10, 10), None, 0) == (5, 5)

    def test_apply_player_move(self):
        assert _apply_player_move((10, 10), "move_left") == (9, 10)
        assert _apply_player_move((10, 10), "move_right") == (11, 10)
        assert _apply_player_move((10, 10), "move_up") == (10, 9)
        assert _apply_player_move((10, 10), "move_down") == (10, 11)
        assert _apply_player_move((10, 10), "do") == (10, 10)

    def test_direction_primitive(self):
        assert _direction_primitive((10, 10), (11, 10)) == "move_right"
        assert _direction_primitive((10, 10), (9, 10)) == "move_left"
        assert _direction_primitive((10, 10), (10, 11)) == "move_down"
        assert _direction_primitive((10, 10), (10, 9)) == "move_up"

    def test_nearest_concept_dynamic_entity_precedence(self):
        sim = _mkstate(entities=[DynamicEntity("zombie", (10, 11))])
        sim.last_action = "move_down"
        # Player at (10,10), facing down → front = (10, 11)
        # Entity at (10, 11) → nearest = zombie
        assert _nearest_concept(sim) == "zombie"


# ---------------------------------------------------------------------------
# find_remedies
# ---------------------------------------------------------------------------


class TestFindRemedies:
    def test_var_depleted_food_finds_cow(self, loaded_store):
        failure = Failure(kind="var_depleted", var="food", cause=None, step=10)
        remedies = loaded_store.find_remedies(failure)
        # Should find "do cow" rule whose effect adds food to body
        cow_rules = [r for r in remedies if r.concept == "cow"]
        assert len(cow_rules) >= 1

    def test_attributed_to_zombie_finds_combat(self, loaded_store):
        failure = Failure(kind="attributed_to", var=None, cause="zombie", step=5)
        remedies = loaded_store.find_remedies(failure)
        combat_rules = [
            r for r in remedies
            if r.effect and r.effect.scene_remove == "zombie"
        ]
        assert len(combat_rules) == 1

    def test_attributed_to_unknown_returns_empty(self, loaded_store):
        failure = Failure(kind="attributed_to", var=None, cause="dragon", step=0)
        assert loaded_store.find_remedies(failure) == []

    def test_var_depleted_no_positive_rule_empty(self, loaded_store):
        failure = Failure(kind="var_depleted", var="nonexistent_var", cause=None, step=0)
        assert loaded_store.find_remedies(failure) == []


# ---------------------------------------------------------------------------
# plan_toward_rule — backward chaining
# ---------------------------------------------------------------------------


class TestPlanTowardRule:
    def test_direct_gather(self, loaded_store):
        # Find the "do tree gives wood" rule, plan toward it with empty inv
        tree = loaded_store.concepts["tree"]
        tree_rule = next(l for l in tree.causal_links if l.effect and l.effect.kind == "gather")
        plan = loaded_store.plan_toward_rule(tree_rule, {})
        assert len(plan) == 1
        assert plan[0].action == "do"
        assert plan[0].target == "tree"

    def test_craft_chain(self, loaded_store):
        """plan_toward_rule for wood_sword should produce: wood, table, sword."""
        table = loaded_store.concepts["table"]
        sword_rule = next(
            l for l in table.causal_links
            if l.action == "make" and l.effect
            and l.effect.inventory_delta.get("wood_sword", 0) > 0
        )
        plan = loaded_store.plan_toward_rule(sword_rule, {})
        actions = [(s.action, s.target) for s in plan]
        # Expected sequence involves gather wood, place table, then make sword
        assert ("do", "tree") in actions  # gather wood
        assert ("place", "empty") in actions  # place table
        assert ("make", "table") in actions  # make sword at table
        # The make_sword step must be last
        assert plan[-1].action == "make"

    def test_combat_chain_from_find_remedies(self, loaded_store):
        """End-to-end: failure → find_remedies → plan_toward_rule → full chain."""
        failure = Failure(kind="attributed_to", var=None, cause="zombie", step=5)
        remedies = loaded_store.find_remedies(failure)
        plan = loaded_store.plan_toward_rule(remedies[0], {})
        actions = [s.action for s in plan]
        assert plan[-1].action == "do"
        assert plan[-1].target == "zombie"
        # Must include gather wood and make sword
        assert "do" in actions  # gather wood
        assert "make" in actions  # craft sword

    def test_already_satisfied(self, loaded_store):
        """If inventory already has sword, plan to kill zombie = just do zombie."""
        zombie = loaded_store.concepts["zombie"]
        kill_rule = next(
            l for l in zombie.causal_links
            if l.effect and l.effect.scene_remove == "zombie"
        )
        plan = loaded_store.plan_toward_rule(kill_rule, {"wood_sword": 1})
        assert len(plan) == 1
        assert plan[0].action == "do"
        assert plan[0].target == "zombie"


# ---------------------------------------------------------------------------
# simulate_forward — outer loop and phases
# ---------------------------------------------------------------------------


class TestSimulateForwardBasic:
    def test_inertia_baseline_no_death(self, loaded_store, tracker):
        """With no enemies, baseline inertia should survive the horizon."""
        state = _mkstate()
        plan = Plan(
            steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)],
            origin="baseline",
        )
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=20)
        assert traj.terminated is False
        assert traj.terminated_reason == "horizon"
        assert traj.tick_count() == 20

    def test_body_clamped_to_reference_max(self, loaded_store, tracker):
        """Stateful regen shouldn't push health above reference_max."""
        state = _mkstate(body={"health": 9.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=20)
        assert traj.final_state.body["health"] <= 9.0

    def test_zombie_causes_death(self, loaded_store, tracker):
        """Zombie 3 tiles away + inertia should kill within ~6 ticks."""
        state = _mkstate(
            player_pos=(10, 10),
            entities=[DynamicEntity("zombie", (13, 10))],
        )
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=20)
        assert traj.terminated is True
        assert traj.terminated_reason == "body_dead"
        assert traj.failure_step("health") is not None

    def test_background_decay_applied(self, loaded_store, tracker):
        state = _mkstate()
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=10)
        # Food should have decayed by background rate over 10 ticks
        assert traj.final_state.body["food"] < 9.0
        assert traj.final_state.body["food"] > 9.0 - 10 * 0.05  # bounded sanity


class TestSimulateForwardPhases:
    """Test individual phases in isolation where possible."""

    def test_phase1_entity_chase(self, loaded_store, tracker):
        state = _mkstate(
            player_pos=(10, 10),
            entities=[DynamicEntity("zombie", (15, 10))],
        )
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=3)
        # Zombie should have moved closer (and/or encountered player)
        final_zombie = [e for e in traj.final_state.dynamic_entities if e.concept_id == "zombie"]
        if final_zombie:
            # Still alive (agent moving away might keep distance)
            assert _manhattan(final_zombie[0].pos, (10, 10)) < 5

    def test_phase4_stateful_regen(self, loaded_store, tracker):
        """Health regen fires while food > 0 AND drink > 0."""
        state = _mkstate(body={"health": 5.0, "food": 9.0, "drink": 9.0, "energy": 9.0})
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=5)
        # Health should have risen from regen (0.1 per rule × 2 rules × 5 ticks = 1.0)
        assert traj.final_state.body["health"] > 5.0

    def test_phase4_stateful_damage_on_starvation(self, loaded_store, tracker):
        """food == 0 damages health."""
        state = _mkstate(body={"health": 9.0, "food": 0.0, "drink": 9.0, "energy": 9.0})
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=5)
        # Food starvation: -0.5/tick. Drink regen: +0.1/tick. Net -0.4. After 5 ticks: -2
        assert traj.final_state.body["health"] < 9.0

    def test_phase5_spatial_damage(self, loaded_store, tracker):
        """Adjacent zombie applies -2 health per tick."""
        state = _mkstate(
            player_pos=(10, 10),
            entities=[DynamicEntity("zombie", (10, 11))],  # adjacent
            last_action="move_down",
        )
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=3)
        # After adjacency damage (-2) + regen (+0.2) per tick for several ticks,
        # health should be dropping
        assert traj.final_state.body["health"] < 9.0


class TestConfidenceThreshold:
    def test_low_confidence_rule_skipped(self, loaded_store, tracker):
        """Rules below 0.1 confidence don't fire in rollout."""
        # Zero out the confidence of the food decay rule
        for rule in loaded_store.passive_rules:
            if rule.kind == "passive_body_rate" and rule.effect.body_rate_variable == "food":
                rule.confidence = 0.0

        state = _mkstate()
        plan = Plan(steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)])
        traj = loaded_store.simulate_forward(plan, state, tracker, horizon=20)

        # Drink should still decay (not zeroed); food should NOT (zeroed rule skipped)
        assert traj.final_state.body["drink"] < 9.0
        # Food would also rise from regen (food > 0 → health +0.1), but baseline
        # decay is now off. Food stays at 9 (only gains from consume rules, none fired).
        assert traj.final_state.body["food"] == 9.0


# ---------------------------------------------------------------------------
# CausalLink.result backward compat via legacy plan()
# ---------------------------------------------------------------------------


class TestLegacyPlanBackwardCompat:
    def test_legacy_plan_string_still_works(self, loaded_store):
        plan = loaded_store.plan("wood")
        assert len(plan) == 1
        assert plan[0].action == "do"
        assert plan[0].target == "tree"
        assert plan[0].expected_gain == "wood"  # legacy field populated
        assert plan[0].rule is not None  # new field populated too

    def test_legacy_plan_nonexistent(self, loaded_store):
        plan = loaded_store.plan("unicorn_horn")
        assert plan == []

    def test_legacy_plan_with_inventory(self, loaded_store):
        # Already have a sword — no need to re-craft
        plan = loaded_store.plan("wood_sword", inventory={"wood_sword": 1})
        assert plan == []
