"""Stage 79 — CRITICAL FALSIFICATION TEST.

Reproduces the Stage 78a synthetic conjunctive task (`sleep + (food=0
OR drink=0) → health -0.067`) at the nursery level. If the nursery
cannot induce this rule from the prediction-error stream within 500
observations, Stage 79 has no chance on real Crafter — the entire
no-LLM induction approach is falsified and we should stop and rethink.

This test:

  1. Builds a `ConceptStore` from the textbook MINUS the conjunctive
     sleep correction. The textbook normally has `sleep → energy +5`
     unconditionally; the real rule (oracle) is `sleep → energy +5
     AND health +0.04` for the satiated case but `sleep → energy +5
     AND health -0.067` when food OR drink is depleted.
  2. Generates a stream of (visible, body, action) samples roughly
     matching what the agent would see in Crafter.
  3. For each sample, computes:
       - oracle: `true_body_delta` (replicates Stage 78a oracle)
       - rules-only prediction: `simulate_forward` 1-tick replay
     and feeds the (predicted_delta, actual_delta) pair to the
     SurpriseAccumulator + RuleNursery pipeline.
  4. Asserts: by the end of the stream, `store.learned_rules` contains
     a rule whose precondition matches `(sleep, food=quartile-0 OR
     drink=quartile-0)` and whose `effect["health"]` is approximately
     -0.067.

Cost: pure Python, no GPU, no env, ~1 second wall.

If this test passes → the no-LLM induction works on the canonical
case → proceed to Crafter integration.
If it fails → the constants (MIN_OBS, MAD_K, VERIFY_TOL) need tuning,
or the design needs rethinking. STOP and revisit.
"""

from __future__ import annotations

import random

from snks.agent.concept_store import ConceptStore
from snks.agent.crafter_textbook import CrafterTextbook
from snks.agent.forward_sim_types import (
    DynamicEntity,
    Plan,
    PlannedStep,
    SimState,
    Trajectory,
)
from snks.agent.perception import HomeostaticTracker
from snks.learning.rule_nursery import RuleNursery
from snks.learning.surprise_accumulator import (
    BODY_ORDER,
    ContextKey,
    SurpriseAccumulator,
    quartile_for,
)


# ---------------------------------------------------------------------------
# Oracle — replicates Stage 78a's true_body_delta for the canonical task
# ---------------------------------------------------------------------------


def _oracle_delta(visible: set[str], body: dict[str, float], action: str) -> dict[str, float]:
    """Stage 78a oracle, restricted to (sleep, do, move) primitives.

    The conjunctive rule is the only thing the textbook does NOT carry:
      sleep + (food=0 OR drink=0) → health -0.067 (replaces +0.04)
    """
    delta = {var: 0.0 for var in BODY_ORDER}

    # Background body rates (textbook also has these — they cancel out
    # in actual - predicted, so include them for realism but they
    # contribute zero to the surprise).
    delta["food"] -= 0.02
    delta["drink"] -= 0.02
    delta["energy"] -= 0.02

    if action == "sleep":
        # Always +5 energy (textbook has this)
        delta["energy"] += 5.0
        if body.get("food", 0) <= 0 or body.get("drink", 0) <= 0:
            # Conjunctive correction (textbook does NOT have this)
            delta["health"] -= 0.067
        else:
            # Non-starving sleep heals (textbook does NOT have +0.04
            # because Stage 78a oracle was different from textbook —
            # the test focuses only on the conjunctive case below).
            delta["health"] += 0.04

    return delta


def _generate_sample(rng: random.Random, force_conjunctive_prob: float = 0.3):
    """Generate a (visible, body, action) sample like Stage 78a's
    `random_state` but biased toward producing conjunctive cases."""
    visible_pool = ["tree", "stone", "cow", "water", "skeleton", "zombie", "empty"]
    n_visible = rng.randint(0, 4)
    visible = set(rng.sample(visible_pool, n_visible))
    body = {
        "health": float(rng.randint(1, 9)),
        "food": float(rng.randint(0, 9)),
        "drink": float(rng.randint(0, 9)),
        "energy": float(rng.randint(0, 9)),
    }
    # Bias action toward sleep so the conjunctive case can accumulate
    actions = ["sleep", "sleep", "sleep", "do", "move_left", "move_right"]
    action = rng.choice(actions)
    # With some probability, force food=0 or drink=0 to seed the
    # conjunctive case
    if action == "sleep" and rng.random() < force_conjunctive_prob:
        if rng.random() < 0.5:
            body["food"] = 0.0
        else:
            body["drink"] = 0.0
    return visible, body, action


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_nursery_induces_conjunctive_sleep_rule_from_500_observations():
    """The headline falsification test for Stage 79."""

    # Set up store + tracker (textbook does NOT include the conjunctive
    # correction — sleep gives +5 energy, no health condition).
    tb = CrafterTextbook("configs/crafter_textbook.yaml")
    store = ConceptStore()
    tb.load_into(store)
    tracker = HomeostaticTracker()
    tracker.init_from_textbook(tb.body_block, store.passive_rules)

    accumulator = SurpriseAccumulator()
    nursery = RuleNursery()

    rng = random.Random(42)
    n_samples = 500

    for i in range(n_samples):
        visible, body, action = _generate_sample(rng)

        # Predicted delta: rules-only 1-tick replay (textbook only)
        predicted_sim = SimState(
            inventory={},
            body=dict(body),
            player_pos=(10, 10),
            dynamic_entities=[],
            spatial_map=None,
            last_action=None,
            step=0,
        )
        predicted_traj = Trajectory(
            plan=Plan(steps=[], origin="probe"),
            body_series={var: [] for var in predicted_sim.body},
            events=[],
            final_state=predicted_sim,
            terminated=False,
            terminated_reason="horizon",
            plan_progress=0,
        )
        store._apply_tick(
            predicted_sim, action, tracker, predicted_traj, tick=0,
            visible_concepts=visible,
        )
        predicted_delta = {
            var: predicted_sim.body.get(var, 0.0) - body.get(var, 0.0)
            for var in BODY_ORDER
        }

        # Actual delta: oracle
        actual_delta = _oracle_delta(visible, body, action)

        # Build context key the same way the runtime would
        context = ContextKey.from_state(visible, body, action)

        accumulator.observe(
            context=context,
            predicted=predicted_delta,
            actual=actual_delta,
            tick_id=i,
        )
        nursery.tick(accumulator, store, current_tick=i)

    # === Assertion: a rule for sleep + (food=0 OR drink=0) was promoted ===

    promoted = [r for r in store.learned_rules]

    # There must be at least one promoted rule
    assert len(promoted) > 0, (
        f"Nursery promoted ZERO rules from {n_samples} observations. "
        f"Stats: nursery={nursery.stats()}, accumulator={accumulator.stats()}"
    )

    # Find a rule whose precondition is sleep + food in quartile 0
    # (i.e. the food-starvation conjunctive case)
    food_starve_rules = [
        r for r in promoted
        if r.precondition.action == "sleep"
        and r.precondition.body_quartiles[1] == 0  # food quartile = 0
    ]
    drink_starve_rules = [
        r for r in promoted
        if r.precondition.action == "sleep"
        and r.precondition.body_quartiles[2] == 0  # drink quartile = 0
    ]

    assert len(food_starve_rules) > 0 or len(drink_starve_rules) > 0, (
        f"Nursery promoted {len(promoted)} rules but none for the "
        f"conjunctive sleep+starvation case. Promoted preconditions: "
        f"{[r.precondition for r in promoted]}"
    )

    # The rule effect on health should be approximately the conjunctive
    # correction (-0.067). Note: the rule also includes other delta
    # contributions from the oracle (e.g. background decay), so the
    # health effect should be roughly in [-0.10, -0.04] range.
    relevant = food_starve_rules + drink_starve_rules
    health_effects = [r.effect.get("health", 0.0) for r in relevant if "health" in r.effect]
    assert len(health_effects) > 0, (
        "Promoted conjunctive rules don't carry a health effect. "
        f"Effects: {[r.effect for r in relevant]}"
    )

    # At least one rule should have health effect close to -0.067
    closest = min(health_effects, key=lambda e: abs(e - (-0.067)))
    assert abs(closest - (-0.067)) < 0.05, (
        f"Closest health effect is {closest}, expected ≈-0.067. "
        f"All effects: {health_effects}"
    )

    print(
        f"\n=== Stage 79 falsification PASSED ===\n"
        f"  observations: {n_samples}\n"
        f"  nursery stats: {nursery.stats()}\n"
        f"  accumulator stats: {accumulator.stats()}\n"
        f"  promoted rules: {len(promoted)}\n"
        f"  food/drink starve sleep rules: {len(food_starve_rules)} / {len(drink_starve_rules)}\n"
        f"  closest health effect: {closest:.4f} (target -0.067)\n"
    )
