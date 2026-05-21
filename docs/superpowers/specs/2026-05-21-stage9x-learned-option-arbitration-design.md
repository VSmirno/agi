# Stage9X Learned Option Arbitration — Design

**Date:** 2026-05-21
**Status:** Design
**Companion docs:** `docs/IDEOLOGY.md`,
`docs/architecture-report-2026-05-11.md`,
`docs/ANTI_TUNING_CHECKLIST.md`,
`docs/CONCEPT_SUCCESS_CRITERIA.md`,
`docs/superpowers/specs/2026-05-12-outcome-role-design.md`,
`docs/superpowers/specs/2026-05-21-stage9x-local-survival-affordances-design.md`

---

## Problem

By `b89e462`, the agent has the basic strategies it needs:

- goal-conditioned frontier exploration for unknown targets;
- dynamic hostile targetability;
- outcome-conditioned combat continuation;
- craft/gather interleaving for required capabilities;
- local survival affordances for textbook-declared positive body effects.

The remaining seed17 failure is no longer "agent does not know water" or
"agent cannot hit a hostile". The inspected forensic run reaches step 253,
uses `opportunistic:water:do_survival_buffer`, and dies under compound pressure:
low vitals, multiple threats, projectile pressure, and an active combat intent.

This is exactly the class of situation the agent should learn from experience.
A parent can teach a child primitives and values; it should not give a
step-by-step table for "hungry, thirsty, and a brick is flying at you".

Therefore the next stage must not add a hand-written arbitration controller.
It must make conflict resolution learnable over already-existing strategy
options.

---

## Doctrine Check

### Anti-Tuning Q1: can this be described without Crafter names?

Yes: *learned option-level conflict resolution under simultaneous survival,
homeostatic, capability, and threat pressures*.

### Anti-Tuning Q2: correct layer?

- **Facts:** textbook still declares primitive effects and threat facts.
- **Mechanisms:** existing planner produces candidate strategies/options.
- **Experience:** the agent records contexts, selected options, and outcomes.
- **Stimuli:** learned option outcome contributes risk/benefit at selection time.

The design must not add `if zombie and drink == 0 then ...` policy code.

### Anti-Tuning Q3: general capability?

The added capability is learning which strategy to invoke or interrupt in a
compound context. This is broader than the current environment.

### Anti-Tuning Q4: neighbouring domain?

The same mechanism applies to any domain with options, needs, hazards,
capabilities, and outcomes: navigation, grid survival, robotics, or game tasks.

### Anti-Tuning Q5: capability vs score?

A metric gain is insufficient. The trace must show that a later episode chooses
a different option in a similar conflict context because persisted option
outcome memory changed the option ranking.

---

## Core Idea

Introduce an **option-level outcome role** in the existing `VectorWorldModel`
substrate:

```text
bind(bind(context_key, option_key), role__OPTION_OUTCOME_H__) -> outcome_vec
```

This extends the successful outcome-role pattern, but lifts the address from
primitive `(concept, action)` pairs to strategy-level `(context, option)` pairs.

It is still one HDC/SDM substrate with multiple roles. No parallel episodic SDM.
No bundled 11-role decision vector that collapses all candidates into uniform
recall.

---

## Strategy Options

`StrategyOption` is a trace-visible abstraction over existing mechanisms. It is
not a new action space for the environment.

Initial option ids:

- `continue_interaction:<target>` — continue a textbook outcome-conditioned
  interaction, such as `do <entity> -> remove_entity`.
- `engage_target:<target>` — start an interaction with a targetable dynamic
  entity or resource.
- `take_local_survival:<target>` — consume/use a locally actionable target whose
  textbook rule restores a non-full vital.
- `seek_frontier:<target>` — explore toward an unknown goal target.
- `seek_known:<target>` — navigate toward a known target in the spatial map or
  dynamic tracker.
- `craft_capability:<item>` — produce a missing capability requested by the
  current goal.
- `recover_self:<mode>` — self-recovery such as sleep when the textbook/action
  model supports it.
- `baseline_motion` — fallback motion when no structured option is available.

The option list is derived from plans already generated today. The stage should
not invent new environment-specific behaviours.

---

## Context Key

The context must be compact enough for SDM contrast, but rich enough to
distinguish conflict states.

Use bucketed symbolic roles:

- `health_bucket`, `food_bucket`, `drink_bucket`, `energy_bucket`:
  `critical`, `low`, `ok`.
- `threat_pressure`: `none`, `near`, `contact`, `projectile`, `multi`.
- `local_restore`: `none`, `food`, `drink`, `health`, `energy`, `multi`.
- `capability_state`: current capability flags such as `armed_melee`,
  `can_restore_food`, `can_restore_drink`.
- `intent_state`: `none`, `continuing_interaction`, `seeking_resource`,
  `crafting_capability`.
- `progress_state`: `normal`, `blocked`, `stuck`.
- `goal_family`: `fight`, `find`, `craft`, `gather`, `recover`, `explore`.

The exact encoded values should come from existing state and textbook-derived
facts, not from named environment branches.

The first implementation can start with the smallest subset needed to separate
the `b89e462` failure class:

```text
vital buckets + threat_pressure + local_restore + intent_state + goal_family
```

`progress_state` can be added immediately if the trace still shows repeated
blocked frontier moves.

---

## Outcome Vector

Reuse the outcome-role structure where possible:

- `survived_h`: alive/dead after horizon `H`.
- `damage_h`: clipped damage over horizon.
- `vital_delta_h`: bucketed change in min vital or per-vital deltas.
- `goal_progress_h`: whether the option advanced the active goal family.
- `interrupted_by`: none / emergency / death / blocked / target_lost.
- `death_cause`: concept or none.

Writes happen after a short horizon, e.g. `H=5` or `H=8`, and flush on death.

Survival-positive outcomes should not become a blind "repeat this forever"
bonus. The stimulus should be conservative: strong negative recall for known
bad option/context pairs, small positive evidence for successful alternatives.

---

## Read Path

At decision time:

1. Build candidate plans as today.
2. Group or annotate plans with `StrategyOption`.
3. Encode the current compact context key.
4. Query `predict_option_outcome(context, option)` for each option.
5. Convert decoded outcome into an `OptionOutcomeStimulus`.
6. Add it to the existing scoring/arbitration path as evidence, not as a hard
   override.

The planner may still choose a risky option if no better option exists. The
learned signal changes ranking; it does not install a rule table.

---

## Write Path

For each selected option, record:

- encoded context at selection time;
- selected `StrategyOption`;
- active goal family;
- primitive action actually executed;
- state snapshot needed to compute horizon outcome.

After `H` steps, or at episode end/death, write:

```text
learn_option_outcome(context_key, option_key, outcome)
```

The write should land in the existing `VectorWorldModel.memory`, using a new
role such as `role__OPTION_OUTCOME_H__`.

Save/load uses the existing world-model persistence. This is cross-episode
learning, not within-episode bookkeeping.

---

## Trace Requirements

Every step must expose enough information to prove or disprove causality:

- `context_key` decoded to readable buckets;
- candidate `strategy_options`;
- per-option predicted outcome/confidence;
- selected option;
- selected primitive;
- whether `OptionOutcomeStimulus` changed the top-ranked option;
- write events after horizon flush.

Without these fields, any survival improvement is not interpretable.

---

## Validation

### Unit Tests

- Context encoder buckets vitals and threat pressure without named
  environment branches.
- Option derivation maps existing plan origins to stable option ids.
- Option outcome role is orthogonal to physics outcome role and primitive
  outcome role.
- Save/load preserves option outcome memory.
- Stimulus differentiates two options in the same context when their stored
  outcomes differ.

### Local Smoke

Before HyperPC, run a short local smoke with `enable_option_outcome=True` and a
minimal/fake trace to catch import/signature mistakes.

### Seed17 Forensic

Run gen1 then gen2 on seed17 ep0 full-profile:

- gen2 must show at least one conflict-context option choice changed by
  option outcome recall;
- the change must be visible in trace, not inferred from final score;
- the agent must not regress into "never fight" or "always flee" collapse.

### Generalisation Check

The first acceptance test should include at least two conflict families:

- threat + depleted drink/food;
- threat + active combat continuation.

If it only helps one named hostile/resources pattern, the result is tactical.

### Stage Claim Boundary

This stage can claim:

> The agent learned option-level conflict preferences from persisted outcome
> experience.

It cannot claim:

> Stage9X is solved.

or:

> The full parent-goal lifecycle is implemented.

---

## Non-Goals

- No hand-written crisis policy.
- No fixed-count combat rule.
- No named Crafter resource/hostile branches.
- No second episodic substrate.
- No Kuramoto/phase-coupling work yet.
- No TextbookPromoter activation in this stage.

---

## Risks

### Uniform Recall

If the context key is too broad or bundled incorrectly, all options may retrieve
the same outcome. This is the failure mode of the reverted episodic-substrate
attempt. Test per-option differentiation before any long run.

### Over-Suppression

If negative recall is too strong, the agent may refuse to fight or explore after
one bad sample. Start with conservative weighting and require confidence before
large penalties.

### Tactical Rebranding

If option ids or context buckets smuggle in environment-specific names, the
stage becomes a hand-coded policy under a learned wrapper. Keep named concepts
only as textbook facts or decoded trace labels, not as control branches.

### Trace Ambiguity

If we cannot show that recall changed option ranking, we cannot claim learning.
Trace instrumentation is part of the mechanism, not optional diagnostics.

---

## Recommended Implementation Slices

1. Add `StrategyOption` derivation and trace fields only. No behaviour change.
2. Add context encoder and unit tests.
3. Add option outcome role to `VectorWorldModel` and save/load tests.
4. Add writer lifecycle for selected option outcomes.
5. Add `OptionOutcomeStimulus` in read path, initially low weight.
6. Run local smoke.
7. Run seed17 gen1/gen2 recording with mandatory video+trace report.
8. Only then consider multiseed.

This ordering keeps the first risky behaviour change late, after the trace can
explain it.

