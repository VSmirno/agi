# Stage9X Interaction Completion / Affordance Closure — Design

**Date:** 2026-05-22
**Status:** Design
**Companion docs:** `docs/IDEOLOGY.md`,
`docs/ROADMAP.md`,
`docs/CONCEPT_SUCCESS_CRITERIA.md`,
`docs/ANTI_TUNING_CHECKLIST.md`,
`docs/architecture-report-2026-05-11.md`,
`docs/superpowers/specs/2026-05-21-stage9x-known-target-navigation-design.md`,
`docs/superpowers/specs/2026-05-21-stage9x-learned-option-arbitration-design.md`

---

## Problem

`490b850` added `navigate_known:<target>` and changed the seed17 ep0 trajectory
shape in the intended direction:

- `navigate_known:tree` fired `16` times;
- `13/16` reduced distance to the known tree target;
- `gather_wood` baseline dropped from `59/70` to `13/33`;
- the first steps show clean approach: tree distance `4 -> 3 -> 2 -> 1`.

But the video still fails:

- episode dies at step `175`, `death_cause=zombie`;
- total `baseline` remains high: `108/175`;
- only one `wood` gain happens during `gather_wood`;
- once the agent reaches distance `1`, it often switches target, emits
  baseline/motion-chain plans, or fails to execute the action that would verify
  the expected effect.

Known-target navigation closed the approach gap, but not the interaction gap.

The missing mechanism is:

```text
approached known target
-> satisfy the action's spatial precondition
-> execute the expected action
-> verify the expected effect
-> keep or abandon the interaction intent based on outcome
```

---

## Doctrine Check

### Anti-Tuning Q1: can this be described without current-environment names?

Yes: *complete an intended interaction with a reached target by satisfying the
action's spatial precondition, executing the action, and verifying the expected
effect*.

This is not "if tree then do"; it is the general closure of an affordance.

### Anti-Tuning Q2: correct layer?

- **Facts:** textbook may declare primitive effects and stable action geometry,
  e.g. `do` operates on the facing tile.
- **Experience:** the agent observes which spatial/action contexts produce the
  expected effect.
- **Mechanisms:** planner maintains an interaction intent and emits approach /
  align / act steps.
- **Stimuli:** goal progress rewards verified effect, not merely being near a
  target.

### Anti-Tuning Q3: general capability?

The capability is target-interaction completion. It applies to resources,
stations, enemies, buttons, doors, tools, and neighbouring embodied domains.

### Anti-Tuning Q4: neighbouring domain?

The same mechanism works anywhere actions require spatial relations such as
adjacent, facing, in-hand, on-tile, or line-of-sight.

### Anti-Tuning Q5: capability vs score?

Success must be visible in trace:

- target reached;
- spatial precondition checked;
- action emitted;
- expected effect verified or failure recorded.

Survival alone is not enough.

---

## Core Idea

Introduce a generic **InteractionIntent** for goal-relevant target actions:

```text
InteractionIntent:
  action: <primitive/action family>
  target_concept: <concept>
  target_pos: <known position if available>
  expected_effect: <textbook/world-model expected delta or outcome>
  status: approaching | aligning | acting | verified | failed | interrupted
  started_step
  attempts
```

The intent is created when the active goal needs an expected effect from a
known target/action pair.

Example without hard-coding the current domain:

```text
goal target = T
candidate action A on T predicts/declares desired effect E
target known or adjacent
-> complete_interaction(A, T, E)
```

For the current failure, this becomes:

```text
complete_interaction(action=do, target=tree, expected_effect={wood:+1})
```

But the mechanism must not name `tree` in policy code.

---

## Action Geometry

The agent needs a representation of what spatial relation an action operates
on.

For current Crafter-style `do`, the stable fact is:

```text
do operates on facing_tile
```

This is not a step-by-step behavioural instruction. It is body/environment
semantics: hands act on what the agent is facing.

The preferred representation is textbook/config data, not planner branches:

```yaml
actions:
  do:
    operates_on: facing_tile
    requires_relation: facing
```

If the first implementation needs a conservative bridge before the textbook
schema is extended, the bridge must stay generic:

```text
action_geometry("do") -> facing_tile
```

No target-specific branches are allowed.

---

## Completion Loop

At each planning tick, if an active `InteractionIntent` exists:

1. **Verify:** if expected effect already appeared, close intent as `verified`.
2. **Abort/interrupt:** if target is lost, emergency overrides, or attempts
   exceed a small bound, mark `failed` or `interrupted`.
3. **Approach:** if target distance > 1, emit/reuse `navigate_known:<target>`.
4. **Align:** if target distance == 1 but target is not in the action's required
   relation, emit an alignment primitive.
5. **Act:** if required relation is satisfied, emit the intended action.
6. **Record:** after action, compare actual delta/outcome to expected effect.

For a facing-tile action, alignment can be achieved by a movement primitive that
sets facing toward the target. If the target is blocked and movement does not
displace the agent, that can still be a successful alignment step if the
environment changes facing.

The implementation should trace this explicitly rather than hiding it.

---

## Learning Hook

The agent should learn operational affordances from attempts:

```text
(action, target_family, relation_to_target, facing_state)
-> expected_effect_achieved | failed | interrupted
```

This gives the agent evidence such as:

- adjacent but not facing + `do` -> failure;
- facing target + `do` -> success;
- target lost before action -> interrupted;
- emergency override during intent -> interrupted.

The first implementation may only record this trace and use declared action
geometry for control. The learning substrate can be wired once the trace proves
the labels are meaningful.

---

## Plan / Option Names

Use explicit origins:

- `complete_interaction:<target>:<action>` for the act step;
- `align_interaction:<target>:<action>` for facing/spatial precondition;
- existing `navigate_known:<target>` for approach.

Strategy option mapping:

- approach: `seek_known:<target>`;
- align/act: `complete_interaction:<target>:<action>` or
  `engage_target:<target>` if the action is a combat/removal action.

The exact option id can be refined during implementation, but it must be
trace-visible and stable enough for later option-outcome learning.

---

## Trace Requirements

When interaction completion is active, each local trace row should include:

- `interaction_completion` object:
  - `status`;
  - `action`;
  - `target_concept`;
  - `target_pos`;
  - `expected_effect`;
  - `relation`;
  - `is_adjacent`;
  - `is_facing_target`;
  - `attempts`;
  - `selected_phase`: `approach | align | act | verify | failed`;
  - `reason`;
- actual delta after the step;
- whether expected effect was achieved.

This must make failures inspectable. A trace that only says `do` happened is
not enough.

---

## Tests

Unit tests:

1. A goal-relevant known target with expected effect creates or selects a
   completion intent.
2. Reached but not-facing target emits an alignment primitive, not baseline.
3. Facing target emits the intended action.
4. Expected effect closes the intent as `verified`.
5. Repeated failed attempts mark the intent as `failed` and release control.
6. Emergency override can interrupt without deleting diagnostic trace.
7. No target-specific branches are required for `tree`; tests should use at
   least two target concepts with the same action geometry.

Regression tests:

1. `navigate_known` still reduces distance before completion.
2. Unknown targets still use frontier.
3. Adjacent actionable local survival affordances still work.

---

## Seed17 Video Gate

Before multiseed validation, record seed17 ep0 full-profile with overlay.

Primary pass criteria:

- after `navigate_known:tree` reaches distance `1`, the next 1-3 steps include
  alignment/action attempts instead of baseline wandering;
- `complete_interaction:tree:do` or equivalent trace appears;
- wood gains increase materially above the current one gain in `gather_wood`;
- `gather_wood` phase does not end merely because the agent drifted away or
  switched to unrelated baseline motion;
- trace contains relation/facing/effect verification fields.

Secondary:

- survival may improve, but this stage is not primarily a survival claim.

---

## Success Claim Boundary

If this stage passes, the narrow claim is:

> The agent can complete a goal-relevant interaction after reaching a known
> target, using generic action geometry and verified outcomes rather than
> target-specific policy code.

It does not yet prove:

- learned conflict arbitration;
- full operational-affordance learning;
- cross-generation improvement;
- concept validation.

Those remain downstream obligations.
