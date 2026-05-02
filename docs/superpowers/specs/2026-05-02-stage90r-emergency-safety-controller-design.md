# Stage90R Emergency Safety Controller Design

## Goal

Recover online `mixed_control_rescue` robustness by making emergency behavior a
first-class control layer instead of a patch on top of actor/planner
disagreement.

This stage should:

- introduce an independent emergency safety controller
- trigger intervention from explicit danger state, not mainly from incidental
  `actor_action != planner_action`
- choose emergency actions from a safety-first contract
- attempt a bounded migration of the Crafter facts needed by that controller
  into textbook/config data

This stage should not try to solve the full stimuli-layer redesign.

## Why This Is The Next Step

The current evidence no longer supports "one more actor-target tweak" as the
right layer.

Three inputs point to the same architectural debt:

1. `2026-04-30-stage90r-current-stage-checkpoint.md`
   - `ca12f40` improved the narrow offline actor contract but regressed online
     `mixed_control_rescue`
   - `7544240` repaired one rescue-eligibility hole but did not recover the
     aggregate rescue baseline

2. `stage90r_architecture_review_2026-04-30.md`
   - rescue is not a first-class safety controller
   - offline promotion robustness is not the same as online behavior robustness
   - current control layers are not cleanly separated

3. Hyper rescue traces
   - `ca12f40` episode 0 dies before rescue ever activates
   - `7544240` activates the new path and narrows that hole
   - once rescue does fire, its one-step outcome is usually locally helpful
     rather than systematically harmful

That combination argues for a controller-layer redesign:

- the main problem is not well explained as "rescue consistently chooses bad
  actions"
- the stronger explanation is "rescue is activated under the wrong contract"

## Scope

This design is intentionally narrow.

In scope:

- new emergency activation contract
- new emergency action selector
- runtime logging for emergency activation and override reason
- bounded migration of emergency-relevant Crafter facts into textbook/config
- online evaluation against the current `9083357` rescue baseline

Out of scope:

- full `stage90r_action_utility()` replacement
- full stimulus separation
- global utility retuning
- full repo-wide Crafter literal cleanup
- the separate hyper GPU hang beyond documenting it as a known adjacent issue

## Current Control Problem

Today rescue is structurally downstream of the learner/planner interaction.
That creates two failure modes:

1. The run can enter a dangerous short trajectory and die before the current
   disagreement-driven rescue path becomes eligible.

2. Emergency behavior inherits too much normal-path shape because it is not a
   distinct controller with its own activation and action-selection contract.

`7544240` shows that adding more eligibility conditions can partially repair one
hole, but the aggregate result still stays below `9083357`. That is the signal
to move the problem to the correct layer.

## Design Direction

Introduce a first-class `EmergencySafetyController`.

It should be treated as a sibling of the normal learner/planner path, not as a
minor branch inside it.

Control roles after this change:

- `planner`: parent/advisor for normal progress
- `learner`: normal action selector
- `evaluator`: local outcome predictor over candidate actions
- `emergency safety controller`: independent emergency activation and override
  layer

## Activation Contract

Emergency activation must not depend primarily on actor/planner disagreement.

Recommended contract:

1. Build an emergency feature bundle from current state:
   - body/vitals pressure
   - nearby hostile pressure
   - recent damage / recent no-progress pattern
   - evaluator candidate outcomes relevant to immediate danger
   - threat attributes from textbook/config

2. Compute an explicit emergency activation score from those features.

3. If the score stays below threshold, remain on the normal path.

4. If the score exceeds threshold, hand action choice to the emergency safety
   controller.

`actor_action != planner_action` may remain an input feature, but it is no
longer the main control contract.

## Action Selection Contract

When emergency mode is active, action choice should be driven by a safety-first
objective.

Recommended objective shape:

- avoid immediate damage and trap states first
- prefer actions that increase separation or reduce direct hostile exposure
- consider local survival mobility and escape viability before normal progress
- allow planner/evaluator guidance as inputs, but do not let normal progress
  heuristics dominate the emergency contract

This does not require a full repository-wide stimuli redesign. It only requires
that emergency selection has its own contract and does not inherit normal action
ranking by accident.

## Candidate Inputs

The emergency controller should read:

- current local observation package
- current vitals / inventory pressure
- candidate actions already produced by the runtime
- evaluator outcome predictions for those candidates
- planner suggestion for the same candidates
- textbook threat facts such as hostile category, danger behavior, and
  interaction-relevant range attributes

The first implementation can stay symbolic and local. It does not need a new
world-model subsystem.

## Bounded Textbook Migration

This stage should attempt textbook migration, but only where it directly serves
the emergency controller.

Move out of mechanism code:

- hostile category membership used by emergency logic
- resource category membership if needed for emergency-vs-progress arbitration
- threat interaction attributes that affect emergency activation or emergency
  action choice
- repeated post-mortem range/danger constants that are being reused as stable
  world facts

Do not turn this stage into a repo-wide cleanup campaign. If a Crafter literal
does not affect the new controller path, leave it for follow-up work.

## Runtime Logging

This stage needs better rescue-side observability than the current compare
artifacts provide.

At minimum, log for each emergency intervention:

- why emergency activated
- activation feature values
- planner action
- learner action
- emergency-selected action
- whether the override came from planner-aligned safety logic or an independent
  emergency choice
- immediate outcome delta after the step

This stage should also log enough information to distinguish:

- "controller never activated"
- "controller activated but chose weak action"
- "controller acted reasonably but the trajectory was already unrecoverable"

## Evaluation

Primary online gate:

- bounded `mixed_control_rescue` compare against `9083357`

Required success signals:

- aggregate `mixed_control_rescue` is not worse than `9083357`
- early hostile deaths without any rescue intervention shrink relative to
  `ca12f40`
- rescue coverage no longer collapses when actor/planner agreement tightens
- the new controller does not obviously damage non-emergency behavior

Secondary evidence:

- episode-level traces show emergency activation in dangerous-agreement states
- logs clearly distinguish activation reason from action-selection reason
- emergency-relevant world facts are no longer duplicated as raw local
  constants across mechanism code

## Non-Goals

This design does not claim that:

- full stimulus separation is unnecessary
- the current utility contract is ideal
- textbook migration is complete
- the hyper GPU online hang is solved
- the original no-backprop thesis is proven by this stage

Those remain separate questions.

## Risks

### Risk: The new controller becomes another ad hoc patch

Mitigation:

- keep a clean activation contract
- keep its inputs explicit
- move needed world facts out of mechanism literals instead of adding more
  hidden Crafter logic

### Risk: Bounded textbook migration expands into a broad refactor

Mitigation:

- migrate only facts used by the emergency path
- record all remaining literals as follow-up debt instead of widening scope

### Risk: Emergency activation improves coverage but not aggregate survival

Mitigation:

- treat `9083357` as a hard behavioral baseline
- if aggregate rescue still stays below that baseline after this redesign,
  accept that this controller proxy was insufficient and revisit the next layer
  with new evidence

## Fallback Criterion

This stage has an explicit scientific fallback.

If the emergency safety controller is implemented and the aggregate bounded
`mixed_control_rescue` result still remains below `9083357`, then:

- this redesign should be marked partial rather than successful
- the architecture should not keep accreting more small rescue-trigger tweaks
- the next redesign can legitimately escalate to a deeper motivation/stimulus
  layer change with better rescue-side logging already in place

## Definition Of Done

This stage is done when:

- a first-class emergency safety controller exists in the runtime
- emergency activation is no longer mainly disagreement-gated
- emergency action selection is explicitly safety-first
- emergency-relevant Crafter facts have been attempted in textbook/config
  migration where they directly affect the controller path
- rescue traces contain explicit activation and override reasons
- bounded online compare is rerun against `9083357`
- results are judged by aggregate rescue behavior, not only by local episode
  anecdotes
