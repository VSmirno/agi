# Stage90R Memory Belief State Design

Date: 2026-04-24

## Goal

Add a new Stage90R memory mechanism that strengthens the local world-model
substrate without turning into hidden policy memory.

This stage must improve the model's understanding of local world dynamics:

- what in the world persists
- what in the world is being depleted
- whether real progress is happening
- whether threat is rising or falling

It must not reintroduce direct action-history leakage, canary hacks, or
planner-bypassing policy shortcuts.

## Baseline

Current baseline remains `fullfix`.

Temporal-branch experiments are paused because:

- action-history leakage produced self-referential bias
- trimmed temporal export avoided direct leakage but still failed on a fresh
  clean rerun
- the fresh clean rerun collapsed offline into `sleep` strongly enough to fail
  anti-collapse gate before checkpoint promotion

This new stage is therefore a redesign, not a continuation of the previous
temporal feature patching strategy.

## Design Contract

### What the memory stage is

A compact belief state over **local world dynamics**.

It is:

- a world-state summarizer
- a short-horizon dynamics carrier
- an input to the local evaluator/advisory substrate

It is not:

- a policy head
- a direct action selector
- a planner replacement
- a hidden tape of recent primitive preferences

### What the memory stage may represent

The belief state must make these four dimensions recoverable:

1. **Persistence**
   - whether local affordances remain available and stable
2. **Depletion**
   - whether a local pocket/affordance is exhausting or losing value
3. **Progress**
   - whether the agent is producing real state change, not just safe stalling
4. **Threat Trend**
   - whether danger is rising, stable, or fading

### What the memory stage must not represent

The belief state must not directly export:

- previous action identity one-hots
- recent action histograms
- explicit action streak features
- explicit stationary-via-action-history shortcuts
- state signatures partitioned mainly by recent policy identity

If the mechanism needs these to work, it is not a world-memory mechanism.

## Architectural Shape

### 1. BeliefStateEncoder

Introduce a compact encoder that rolls forward a latent belief state from
transition evidence.

Inputs per step:

- current local observation
- previous belief state
- weak causal action context
- post-step deltas

Outputs:

- updated belief state
- optional inspectable auxiliary predictions

The action input is allowed only as weak causal context for transition updates,
not as dominant representational content.

### 2. MemoryAwareLocalEvaluator

Replace the current stateless local evaluator interface with:

`rank(candidate_action | observation, belief_state)`

The evaluator remains advisory/predictive. It still does not own control.

### 3. Explicit Dynamics Heads

Instead of relying on one scalar shortcut utility, the evaluator should predict
separable aspects of local consequence:

- `pred_damage_risk`
- `pred_survival`
- `pred_resource_flow`
- `pred_progress_delta`
- `pred_stall_risk`
- `pred_affordance_persistence`
- `pred_threat_trend`

Any scalar ranking utility should be derived from these heads, not substituted
for them.

## Data Design

### Transition Evidence, Not Action Memory

The new dataset path should emphasize local world evolution between steps:

- observation change
- body delta
- inventory delta
- displacement delta
- nearby threat persistence/motion
- affordance persistence/depletion

This stage should not add exported targets whose main job is to memorize recent
primitive preferences.

### New Targets

At minimum, add targets for:

- `progress_delta`
- `stall_risk`
- `affordance_persistence`
- `threat_trend`

These targets should be inspectable in summaries and evaluator reports.

## Allowed vs Forbidden Inputs

### Allowed

- observation geometry
- local entity persistence
- local resource continuity
- local displacement/progress evidence
- body and inventory deltas
- post-step world change indicators

### Forbidden as direct exported features

- `prev_action_*`
- `recent_hist_*`
- `action_streak_norm`
- `stationary_streak_norm`
- any direct action-frequency or action-identity shortcut features

## Success Criteria

This stage is successful only if a clean rerun shows:

1. no offline anti-collapse failure
2. no `do` attractor collapse
3. no `sleep` attractor collapse
4. planner-advisory performance not worse than `fullfix`
5. inspectable heads that show meaningful variation rather than hidden collapse

If the memory stage reduces collapse only by becoming a disguised policy memory,
it fails.

## Minimal Implementation Slice

The first implementation slice should stay narrow:

1. add `BeliefStateEncoder`
2. add memory-aware evaluator inputs
3. add only the new world-dynamics heads:
   - progress
   - stall
   - affordance persistence
   - threat trend
4. keep planner architecture unchanged
5. keep canary purely diagnostic

Do not:

- rewrite planner control
- add canary behavior rules
- introduce direct recurrent policy outputs
- bundle unrelated refactors

## Experimental Policy

The next reruns should happen only after:

- the new memory representation is implemented
- offline diagnostics explicitly expose the new heads
- the evaluator can be inspected without relying on a single scalar score

The branch should remain experimental until it beats or at least matches
`fullfix` without introducing a new attractor collapse.

## Hard Rule

Memory must be **state-evolution-centric**, not **action-history-centric**.

That is the central contract for this stage.
