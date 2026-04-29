# Stage90R Shared Guidance Bottleneck Design

## Goal

Replace the failed evaluator-only world-prior slice with a real state-level
guidance architecture:

- one shared semantic state latent `z_state`
- a first-class guidance bottleneck supervised by parent/world factors
- actor and evaluator both consume that same guidance-shaped state

This redesign keeps the learner in charge of primitive actions while moving
planner/parent supervision up from contradictory `move_*` imitation to richer
state meaning.

## Why The Previous Slice Failed

The earlier `danger_pressure` / `resource_opportunity` slice failed for
architectural reasons, not because world priors are the wrong idea.

- The new prior heads sat on the action-conditioned evaluator branch.
- The actor used a separate state-policy branch.
- Ranking and utility logic did not consume the priors.
- Planner supervision still decomposed too early into contradictory primitive
  actions.

So the model got extra labels, but not a new shared abstraction layer.

## Scope

This redesign introduces four state-level guidance targets and routes them
through one shared state bottleneck.

### Guidance Targets

1. `threat_urgency`
How urgent the local danger state is.

2. `opportunity_availability`
How available a low-cost local opportunity is, especially nearby resources or
durable affordances.

3. `vitality_pressure`
How much the agent's body state is under pressure.

4. `progress_viability`
How viable productive local progress is without stalling.

These are descriptive state factors, not direct action commands.

## Architecture

### 1. Shared State Encoder

Build one shared semantic latent `z_state` from:

- viewport tiles + confidences
- body vector
- inventory vector
- belief state vector

This replaces the old split where actor and evaluator diverged too early.

### 2. Guidance Bottleneck

Predict the four guidance factors from `z_state`.

The predicted guidance vector becomes a first-class control signal:

- actor consumes `z_state + guidance`
- evaluator consumes `z_state + action + guidance`

This makes guidance part of decision formation rather than an auxiliary leaf.

### 3. Action Evaluator

The evaluator becomes explicitly action-conditioned on top of the shared state:

- `z_state`
- candidate action embedding
- guidance vector

Outcome heads remain, but they now read from a guidance-shaped action branch.

### 4. Actor

The actor head reads the same `z_state + guidance`.

This should reduce actor/ranking drift because both heads now rely on the same
semantic state bottleneck.

## Supervision

### State-Level Guidance Supervision

Guidance targets are computed from state-level information already present in
the dataset contract:

- `state_signature`
- `belief_state_signature`
- `nearest_threat_distances`
- regime labels
- body buckets / inventory presence

They should not vary per candidate action for the same state signature.

### Actor Supervision

Actor supervision moves one step away from raw primitive imitation.

Planner records still contribute soft signature-level action distributions, but
they are blended with state-level advisory distributions from outcome-derived
ranking winners when available.

This is not full planner replacement. It is advisory distillation:

- planner says what it preferred
- ranking says what tended to win on that state
- actor learns from the blended state-level advisory signal

## Non-Goals

This redesign does not yet:

- add hard-coded primitive rules
- retune gate thresholds
- rewrite offline gate semantics
- replace the ranking objective with a new pairwise/listwise trainer
- redesign checkpoint selection beyond existing logic

Those remain follow-up work if the shared bottleneck is still insufficient.

## Evaluation

Use the same narrow deterministic path:

- dataset: `_docs/stage90r_slice_d_mixed_control_dataset.json`
- mode: `mixed_control`
- seed: `7`
- `epochs=1` smoke
- `epochs=3` canonical

Primary success signals:

- actor no longer collapses immediately to `move_up`
- ranking/actor drift weakens on the same valid states
- canonical endpoint is less reflexive than the post-split baselines

Secondary signals:

- guidance heads train stably
- explanatory examples show more coherent action-conditioned scoring

## Risks

### Risk: Guidance targets are still just relabeled old outcomes

Mitigation:
- derive them from state signature and belief signature first
- use outcome-derived labels only as a secondary advisory source

### Risk: Actor still overfits planner mode despite the shared bottleneck

Mitigation:
- blend planner and ranking advisory targets by signature
- keep consistency weighting on ambiguous planner signatures

### Risk: Guidance enters both branches but is too weak to matter

Mitigation:
- feed predicted guidance directly into actor and evaluator inputs
- verify state-level guidance outputs in diagnostics

## Definition Of Done

This redesign slice is done when:

- the new shared `z_state` + guidance bottleneck is implemented
- actor and evaluator both consume the guidance vector
- teacher targets use advisory blending when ranking support exists
- focused tests pass
- the narrow seed-7 verification is rerun and compared against the current
  soft-teacher and failed-prior baselines
