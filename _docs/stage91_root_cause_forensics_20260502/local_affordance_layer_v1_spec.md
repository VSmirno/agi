# Stage91 Local Affordance Layer V1 Spec

Date: 2026-05-08

## Goal

Introduce a small, explicit, generalizable local affordance layer that exposes
truth already visible in the observed `7x9` viewport, without turning rescue or
control into a hand-written tactical policy.

This layer is intended to be architectural, not game-specific:

- perception provides local scene truth
- affordance extraction converts that truth into action-conditioned local facts
- learned decision layers still choose what to do

## Why This Exists

Stage91 weak-seed forensics established:

- missing rescue activation is not the main failure
- blocked/occupied movement mismatch was real and fixing it helped
- direct feasibility labels helped further
- broad controller-ranking tweaks did not hold up
- a later `do`-facing tweak regressed and is rejected as an active direction

The best current validated baseline is the feasibility-label fix on top of the
movement and blocked-move fixes:

- weak-seed mean `avg_survival=172.875`

The next step should preserve the same paradigm while reducing residual
false-safe terminal judgments.

## Design Principles

1. The neural network still plays the game.
2. The system may explicitly use facts that are honestly observable in the
   local `7x9` viewport.
3. We extract local scene truth, not escape policy rules.
4. We do not tune the environment.
5. We do not add a hand-coded tactical planner.
6. The same local truth should be available to runtime rescue, diagnostics, and
   later learned consumers through one shared contract.

## Non-Goals

V1 does not attempt:

- a full tactical planner
- long-horizon danger heuristics
- dense retraining changes
- global map redesign
- game-specific escape scripts

## Proposed Layer

### A. Debuggable Core

This is the small, human-readable contract used first by runtime rescue and
diagnostics.

#### Scene-level fields

- `facing_concept`
- `facing_blocked`
- `nearest_hostile_distance`
- `nearest_hostile_direction`

#### Per-move affordance fields

For each of:

- `move_left`
- `move_right`
- `move_up`
- `move_down`

record:

- `would_move`
- `blocked_static`
- `blocked_occupied`
- `adjacent_hostile_after`
- `contact_after`
- `effective_displacement`

#### `do` affordance fields

- `do_target_concept`
- `do_affordance_present`
- `do_under_contact_pressure`

These are local truth statements, not recommendations.

### B. Dense Local Channels

This is the later learned-layer consumer form. Not required for V1 runtime
delivery.

Candidate channels:

- occupancy mask
- hostile mask
- blocked/passable mask
- interactable/affordance mask
- facing-direction channel

This part is intentionally deferred until the compact core is stable.

## Code Boundaries

### New module

Create:

- `src/snks/agent/stage90r_local_affordances.py`

Responsibility:

- derive compact local affordance facts from current local scene truth
- remain policy-free
- remain environment-tuning-free

It should not rank actions or choose actions.

### Runtime integration

Primary consumer:

- `src/snks/agent/vector_mpc_agent.py`

Responsibilities:

- build the local affordance snapshot each step
- pass compact affordance facts into rescue counterfactual evaluation
- record affordance facts into traces/diagnostics when enabled

### Rescue consumer

- `src/snks/agent/stage90r_emergency_controller.py`

Responsibilities:

- read affordance facts
- score/rank actions using those facts

The controller should never infer local scene truth on its own.

### Training/data contract

- `src/snks/agent/stage90r_local_policy.py`
- later, if needed: `src/snks/agent/stage90r_local_model.py`

Responsibilities:

- carry the same compact affordance contract into dataset labels/records
- keep runtime and training semantics aligned

## V1 Scope

V1 should implement only the compact debuggable core.

Specifically:

1. Introduce a shared local affordance schema.
2. Extract only immediate local action-feasibility truth.
3. Thread that truth into runtime rescue.
4. Thread the same truth into dataset/record labels.
5. Validate only on weak seeds `7` and `17`.

V1 should not add dense channels yet.

## Suggested V1 Data Shape

The exact Python structure is flexible, but the contract should read roughly as:

```python
{
  "scene": {
    "facing_concept": str | None,
    "facing_blocked": bool,
    "nearest_hostile_distance": int | None,
    "nearest_hostile_direction": str | None,
  },
  "actions": {
    "move_left": {
      "would_move": bool,
      "blocked_static": bool,
      "blocked_occupied": bool,
      "adjacent_hostile_after": bool,
      "contact_after": bool,
      "effective_displacement": int,
    },
    "...": {},
    "do": {
      "do_target_concept": str | None,
      "do_affordance_present": bool,
      "do_under_contact_pressure": bool,
    },
  },
}
```

## First Implementation Experiment

V1 should not try to solve everything at once.

The first clean experiment is:

1. add the compact core
2. use it in runtime rescue only
3. preserve the current best feasibility-label baseline
4. rerun bounded HyperPC validation on weak seeds `7` and `17`

## Validation Criteria

Success is not "more structure exists". Success is measured by:

- weak-seed `avg_survival`
- fewer false-safe terminal rows
- fewer terminal `do` nonsense rows
- improved `nearest_hostile_after`
- no major regression versus the current best feasibility-label baseline

Primary comparison baseline:

- `_docs/hyper_stage91_feasibility_label_fix_validation_20260507T181856Z`

## Risks

1. Over-encoding policy:
   Mitigation: expose only local truth, never "best move".

2. Runtime/training divergence:
   Mitigation: one shared compact contract.

3. Interface bloat:
   Mitigation: keep V1 to the compact core only.

4. Architectural drift into game-specific hacks:
   Mitigation: phrase and implement the layer as a generic
   action-conditioned local affordance interface.

## Current Recommendation

Proceed with V1 as a compact, debuggable local affordance layer.

Do not:

- re-open broad controller heuristic tuning
- tune the environment
- introduce a hand-authored escape policy
- jump straight to dense channel retraining

The correct next move is to make the local action-feasibility interface a
first-class architectural layer and validate it incrementally.
