# Stage9X Local Survival Affordances — Design Update

## Status

Implemented in `b89e462` on `feature/stage9x-capability-goal-handoff`.

This design update records the rule accepted after inspecting the seed17 video:
when a basic survival resource is locally actionable, the agent should take it
before the corresponding vital becomes critical.

## Ideology

This is not a `water` or `cow` policy rule. The split is:

- **Facts:** textbook rules declare effects such as
  `do water -> body.drink +5` and `do cow -> body.food +5`.
- **Mechanism:** planner detects any local `do <target>` action whose textbook
  effect has a positive body delta for a non-full vital.
- **Experience:** local availability comes from current perception and
  `CrafterSpatialMap`, not from stable world knowledge.
- **Stimuli/arbitration:** immediate hostile emergency still has priority.

The same mechanism would apply to a neighbour domain if the textbook declares
another locally actionable object that restores a basic vital.

## Mechanism

`_positive_body_effect_from_textbook(textbook, action, target)` extracts
positive body deltas from structured textbook rules.

`_opportunistic_survival_plan(...)` builds a one-step
`VectorPlanStep(action="do", target=<target>)` when all conditions hold:

- target is current `near_concept` or in the four adjacent map cells;
- `model.requirements_met(target, "do", inventory)` is true;
- textbook declares a positive body effect for that target;
- the affected vital is below its max reference value (`< 9.0`);
- no hostile is inside its textbook-derived emergency range.

The plan origin is:

```text
opportunistic:<target>:do_survival_buffer
```

Interaction continuation still runs first. This prevents local resource
pickup from interrupting an already active outcome-conditioned combat action.

## Validation

Focused tests on minipc:

```text
8 passed
```

Covered cases:

- textbook effect extraction for water and cow;
- local water is taken before critical drink;
- full vitals do not trigger opportunistic pickup;
- immediate hostile threat suppresses opportunistic pickup;
- prior combat-continuation and dynamic-threat goal regressions still pass.

Seed17 forensic run after `b89e462`:

```text
max_steps=350
episode_steps=253
death_cause=zombie
opportunistic:water:do_survival_buffer on steps 20-21
final_body={health:0, food:0, drink:0, energy:1}
```

The inspected dehydration symptom is no longer reproduced in that window.
The remaining failure is hostile-pressure arbitration under depleted vitals.

Artifact:

```text
/home/yorick/.ductor/workspace/output_to_user/stage9x_interaction/seed17_ep0_b89e462_planner_only_350_20260521.json
```

## Non-Goals

- Do not claim Stage9X PASS from this single forensic run.
- Do not add fixed-count combat rules.
- Do not encode named Crafter resources in planner control flow.
- Do not solve full parent-goal lifecycle or multi-threat tactical combat here.
