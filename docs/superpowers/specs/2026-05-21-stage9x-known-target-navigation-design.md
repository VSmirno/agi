# Stage9X Known-Target Navigation — Design

**Date:** 2026-05-21
**Status:** Design
**Companion docs:** `docs/IDEOLOGY.md`,
`docs/ROADMAP.md`,
`docs/CONCEPT_SUCCESS_CRITERIA.md`,
`docs/ANTI_TUNING_CHECKLIST.md`,
`docs/architecture-report-2026-05-11.md`,
`docs/superpowers/specs/2026-05-21-stage9x-learned-option-arbitration-design.md`

---

## Problem

The `7877946` seed17 ep0 full-profile video exposed a lower-level gap than
learned option arbitration.

The agent does not merely choose the wrong high-level strategy under compound
pressure. It often fails to execute a basic strategy when the target is already
known on the cognitive map.

Trace summary from `seed17_ep0_7877946_gen2.json`:

- `episode_steps=179`, `death_cause=zombie`.
- `baseline` plan origin: `104/179` steps.
- `baseline_motion` strategy option: `155/179` steps.
- `gather_wood`: `70` steps, of which `59` are `baseline`.
- On step 0, `goal=gather_wood`, known `tree` distance is `4`, but the selected
  plan is `baseline`.
- Across the early `gather_wood` phase, known tree distance repeatedly stays
  within `2..6`, while the agent keeps moving in non-monotonic directions.
- Positive inventory/body changes occur only `8` times in `179` steps.
- Unique positions: `61/179`; revisit ratio `0.659`.

Visually, this is aimless local wandering: the agent sees or has mapped useful
targets, but does not convert "known target needed by current goal" into a
stable approach behaviour.

This invalidates a hidden assumption in the learned-option arbitration spec:
the option set is not complete enough yet. The agent cannot learn sensible
conflict resolution if one of the basic options is still represented as
baseline motion noise.

---

## Doctrine Check

### Anti-Tuning Q1: can this be described without current-environment names?

Yes: *goal-directed navigation toward a known target in a partial cognitive
map*. This is not "walk to a Crafter tree"; it is the generic bridge from a
goal target to an approach action.

### Anti-Tuning Q2: correct layer?

- **Facts:** textbook/goals declare what target concept a goal needs.
- **Experience:** the spatial map stores where instances of that concept have
  been observed.
- **Mechanism:** the planner must be able to emit an approach option for a
  known target.
- **Stimuli:** goal progress should reward distance reduction toward the active
  target.

The fix must not add `if goal == gather_wood then move_to_tree` policy code.

### Anti-Tuning Q3: general capability?

The capability is navigation from symbolic goal target to known map location.
It applies to resources, stations, threats, tools, or any target concept in a
neighbouring domain.

### Anti-Tuning Q4: neighbouring domain?

The mechanism transfers to any grid or embodied environment where the agent has
a partial map, known target positions, and goal-conditioned target concepts.

### Anti-Tuning Q5: capability vs score?

Success is not just longer survival. The trace must show less baseline motion,
monotonic distance reduction toward known targets, and earlier target
interaction.

---

## Core Idea

Add a first-class **known-target navigation option**:

```text
active_goal.target_concept
+ spatial_map.find_nearest(target_concept) exists
+ target is not immediately actionable
-> VectorPlan(origin="navigate_known:<target>")
```

The option emits one abstract step:

```text
VectorPlanStep(action="navigate_known", target=<target_concept>)
```

`expand_to_primitive` converts that abstract step into one concrete movement
primitive that reduces distance to the nearest known target, using the same
spatial map and passability constraints already available to the planner.

This fills the gap between:

- `seek_frontier:<target>` for unknown targets, and
- `do/make/place <target>` for adjacent or immediately actionable targets.

---

## Non-Goals

This stage must not:

- tune the baseline fallback penalty;
- add target-specific branches for `tree`, `water`, `cow`, `zombie`, or any
  other named environment entity;
- change emergency-safety policy;
- lower option-outcome confidence thresholds;
- claim learned conflict arbitration success.

The problem is not that option memory is underweighted. The problem is that the
planner currently lacks a clean option for "approach the known thing my current
goal needs".

---

## Plan Generation

In `generate_candidate_plans`, add a generic candidate when all conditions hold:

1. `active_goal` is present.
2. `active_goal.target_concept` is not `None`.
3. `spatial_map.find_nearest(active_goal.target_concept, player_pos)` returns a
   target position, or dynamic-entity tracking can provide one.
4. The target is not already immediately actionable by the current primitive
   selection path.

Candidate:

```text
origin = "navigate_known:<target>"
steps = [VectorPlanStep(action="navigate_known", target=<target>)]
```

The implementation must derive `<target>` from `active_goal.target_concept` or
textbook-derived goal facts, not from a hand-written environment list.

---

## Primitive Expansion

`expand_to_primitive` should support `action="navigate_known"`:

1. Resolve the nearest known target position.
2. Enumerate legal movement primitives.
3. Predict or estimate each move's next position.
4. Prefer moves that reduce Manhattan distance to target.
5. Reject moves that are known blocked.
6. If no reducing move is available, choose the least-bad non-blocked move and
   trace the blockage reason.

This is a one-step local planner, not a full pathfinding rewrite. The existing
MPC loop re-evaluates next tick, which is sufficient for the first stage.

If dynamic entity tracking provides a target position, the same primitive
expansion should work for a moving target, subject to existing threat/safety
arbitration.

---

## Scoring and Progress

Known-target navigation must beat baseline only when it actually advances the
current goal.

Add a goal-progress signal:

```text
progress = distance_before(target) - distance_after(target)
```

For a `navigate_known:<target>` plan:

- positive progress when distance decreases;
- zero progress when unchanged;
- negative or no bonus when blocked/increasing.

This belongs in goal/stimulus scoring, not in an ad-hoc action priority table.
Death and immediate safety barriers must still dominate this signal.

---

## Strategy Option Mapping

`navigate_known:<target>` should map to:

```text
StrategyOption(kind="seek_known", target=<target>)
option_id="seek_known:<target>"
```

This keeps learned option arbitration aligned with the corrected option set.
Later option-outcome learning can then learn whether `seek_known:water`,
`seek_known:tree`, or `engage_target:<threat>` succeeds or fails in compound
contexts.

---

## Trace Requirements

Each selected `navigate_known` plan should record:

- `plan_origin`;
- `strategy_option`;
- `target_concept`;
- `target_pos`;
- `dist_before`;
- `dist_after`;
- `chosen_move`;
- `candidate_moves` with blocked/reducing flags when local trace is enabled.

The trace must make it obvious whether the agent is approaching a known target
or merely wandering under a new name.

---

## Tests

Unit tests:

1. A known active-goal target produces `navigate_known:<target>`.
2. Unknown active-goal target still produces `frontier:<target>` when eligible.
3. Adjacent/immediately actionable target does not produce redundant navigation.
4. `expand_to_primitive(navigate_known)` chooses a move that reduces distance in
   a simple map.
5. Blocked direct axis chooses a non-blocked alternative if one exists.
6. `navigate_known:<target>` maps to `StrategyOption("seek_known", target)`.

Trace tests:

1. `local_trace` contains navigation debug fields when the option is selected.
2. Distance accounting is consistent: `dist_after < dist_before` for successful
   reducing moves.

Regression tests:

1. Existing `frontier_seek` behaviour remains unchanged for unknown targets.
2. Emergency safety can still override an unsafe navigation primitive.

---

## Seed17 Video Gate

Before multiseed validation, record seed17 ep0 full-profile with perception
overlay.

Pass/fail should be judged first by trajectory shape:

- `gather_wood` baseline share drops sharply from `59/70`.
- When a tree is known at distance `2..6`, selected plans include
  `navigate_known:tree`.
- Distance to selected known target decreases more often than it increases.
- Productive `do tree` / wood gain appears early.
- The path no longer spends the first 70 steps oscillating around the same
  local area without target contact.

Survival improvement is useful but secondary for this stage. A longer episode
with the same wandering pattern is not a pass.

---

## Expected Interaction With Learned Option Arbitration

Known-target navigation is a prerequisite for useful option arbitration.

The option memory result at `7877946` was causally inert:

- gen2 had `option_outcome_recall=179`;
- `used_for_scoring=0`;
- all confident recalls decoded as survived;
- gen1 and gen2 trajectories were identical.

That result should not be fixed by weight tuning. Once `seek_known:<target>` is
a real, trace-visible option, option-outcome memory can start learning about
meaningful strategy choices instead of treating most behaviour as
`baseline_motion`.

---

## Success Claim Boundary

If this stage passes, the claim is narrow:

> The planner can convert a known goal target in the cognitive map into stable
> approach behaviour.

It does not prove:

- learned conflict arbitration;
- cross-generation causal improvement;
- full survival-policy success;
- concept validation.

Those remain downstream proof obligations.
