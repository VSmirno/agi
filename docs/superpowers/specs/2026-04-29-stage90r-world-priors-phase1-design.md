# Stage90R World Priors Phase 1 Design

## Goal

Start the next Stage90R phase with a minimal world-prior vertical slice that
improves shared state representation without adding direct action rules.

This phase should reduce actor/ranking drift by teaching the model two general
properties of the world:

- `danger_pressure`
- `resource_opportunity`

## Why This Phase Exists

The narrow split and teacher-target fixes removed several bookkeeping and
supervision artifacts, but they did not fix the canonical endpoint. The current
evidence says the remaining issue is no longer primarily in split logic or raw
planner-label noise. The next step should therefore improve the shared world
representation rather than add more action-level patches.

This fits the intended ideology:

- planner/parent guidance should describe the world
- the learner should still decide actions
- we should not hard-code rules like "if danger then run"

## Scope

Phase 1 adds exactly two auxiliary state-level priors:

### 1. `danger_pressure`

Meaning:
- how strongly the local situation is becoming dangerous

Candidate signal sources:
- hostile contact / hostile near regimes
- nearest-threat proximity
- `threat_trend_h`
- relevant belief-state features

Intended effect:
- help the encoder distinguish "world is getting riskier" from neutral drift

### 2. `resource_opportunity`

Meaning:
- how attractive the current state is for low-cost buffer building

Candidate signal sources:
- `resource_gain_h`
- `affordance_persistence_h`
- `local_resource_facing`
- relevant belief-state features

Intended effect:
- help the encoder represent "cheap chance to build запас" without turning it
  into a hard action rule

## Non-Goals

This phase does not:

- add `escape_affordance`
- add `projectile_threat`
- change offline gate thresholds
- add direct action heuristics
- replace ranking or actor supervision

## Architecture

Keep the current structure but enrich the shared representation:

- ranking head stays outcome-derived
- actor head keeps soft signature-level planner targets
- shared encoder gains two new auxiliary prediction heads:
  - `danger_pressure`
  - `resource_opportunity`

The new heads should shape the latent state, not override the action heads.

## Data / Supervision Shape

Use state-level targets, not direct action labels.

Recommended first implementation:

- derive scalar target for `danger_pressure`
- derive scalar target for `resource_opportunity`
- train both as auxiliary regressions or bounded scores

The exact target formulas should stay simple and reuse existing labels before
adding new dataset-generation machinery.

## Evaluation Plan

Use the same narrow verification path as the recent debugging work:

- `mixed_control`
- `_docs/stage90r_slice_d_mixed_control_dataset.json`
- `--seed 7`
- `epochs=1` smoke
- `epochs=3` canonical

Primary success signals:

- actor/ranking drift is weaker on the same valid states
- canonical endpoint is less reflexive
- improvement comes without new threshold tuning

Failure interpretation:

- if only early epochs improve, but canonical endpoint stays collapsed, the
  problem likely sits deeper in training dynamics or shared-objective coupling
- if both early and late behavior improve, continue the world-prior path before
  adding more priors

## Risks

### Risk: Priors collapse into disguised action rules

Mitigation:
- keep targets state-level and descriptive
- do not encode "best action"

### Risk: Auxiliary heads add noise without helping representation

Mitigation:
- start with only two priors
- compare against current `seed 7` baseline before expanding scope

### Risk: Current labels are too weak for good priors

Mitigation:
- start from existing signals
- only add richer target construction if Phase 1 shows partial but real gains

## Definition of Done

Phase 1 is done when:

- two auxiliary world-prior heads are implemented
- the same narrow seed-7 verification is rerun
- we can clearly say whether representation improved enough to justify Phase 2

