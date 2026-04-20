# Stage 90R World-Model Guardrails Design

**Date:** 2026-04-20  
**Status:** Draft  
**Scope:** Architectural guardrails for `Stage 90R` so the stage strengthens
the world-model stack rather than drifting into tactical policy tuning.

## Goal

Define a hard boundary for `Stage 90R`:

- what kinds of local-learning components are **allowed**
- what kinds of local-learning components are **not allowed**
- what evidence is required to claim that a `Stage 90R` change improved
  world understanding rather than merely tuned action strategy

This document does not prescribe every implementation detail. It sets the
architectural constraints that every `Stage 90R` experiment, PR, and report
must satisfy.

## Background

`Stage 90R` was introduced because the current Crafter agent appears to suffer
from a deeper local-coherence problem:

- immediate threats are often handled incoherently
- obvious nearby affordances are often missed
- action choice appears too dependent on stale intermediate state such as
  `spatial_map` or collapsed local labels

The honest research question is:

**can the system build better immediate behavior from better local world
understanding, without replacing the architecture with a tactical local policy?**

This distinction matters because the repository ideology is explicit:

- teacher-provided prior knowledge belongs in `facts`
- generic inference procedures belong in `mechanisms`
- episode/runtime discoveries belong in `experience`
- motivation belongs in `stimuli`

The stage fails ideologically if it introduces a hidden fifth category:
`tactical policy patches disguised as learned local intelligence`.

## Architectural Principle

`Stage 90R` is allowed to add a **local predictive substrate**.

`Stage 90R` is not allowed to add a **local policy oracle**.

In practical terms:

- allowed: a model that predicts local consequences of actions from the current
  viewport-centered state
- not allowed: a model whose practical role is to directly encode
  “when you see this, press this action”

The distinction is whether the component improves **understanding of local
dynamics** or merely improves **action selection by shortcut**.

## Required Category Discipline

Every `Stage 90R` change must answer the following classification question:

1. Is this a new `fact` about the world?
2. Is this a new `mechanism` for reading/applying facts?
3. Is this `experience` gathered from the current or previous episodes?
4. Is this a `stimulus` for evaluating predicted states?
5. Or is this actually a hidden policy rule?

If the answer is “hidden policy rule”, the change is outside the allowed
design space for `Stage 90R`.

## Allowed Patterns

The following patterns are in-bounds.

### 1. Local Action-Conditioned Prediction

Allowed form:

`P(local_outcome | viewport, body, inventory, action)`

Examples of acceptable predicted quantities:

- short-horizon damage
- threat arrival / contact risk
- resource delta
- escape viability delta
- local survival probability over a fixed horizon

These outputs are acceptable because they describe **what is expected to
happen**, not **what the agent should want**.

### 2. Planner/Stimuli Consumption of Predictions

Allowed integration:

- planner queries the local predictor while ranking candidate actions or plans
- stimuli evaluate predicted future states produced with help from the local
  predictor
- the local component acts as an auxiliary model of short-horizon local
  dynamics

The key requirement is that planning and motivation remain explicit layers.

### 3. Inspectable Local Causal Signals

Allowed outputs should be inspectable and explainable at the level of local
scene dynamics:

- “`move_up` reduces predicted hostile contact risk”
- “`do` is predicted to yield resource gain from the current facing geometry”
- “`move_left` increases separation from the nearest hostile in the viewport”

If a local model cannot provide this kind of explanation, it is likely too
close to a hidden policy.

### 4. Local-Only Canary as Falsification Tool

A `local-only` mode is allowed only as a **diagnostic or falsification**
instrument:

- to test whether the learned local model carries any useful control signal
- to reveal collapse, bias, or missing semantics
- to compare local-model signal against the current stack in a controlled way

It is not, by itself, evidence that the architectural direction is correct.

## Forbidden Patterns

The following patterns are out-of-bounds.

### 1. Direct Action Oracle

Forbidden form:

`policy(action | viewport, body, inventory)` used as the primary intended
solution to the stage.

This is especially forbidden if the representation is not explicitly tied to
predictive local semantics and is evaluated mainly by aggregate survival.

### 2. Hidden Reward Shaping as Policy Tuning

The stage must not smuggle Crafter tactics into:

- hard-coded utility weights
- dataset curation that implicitly favors one movement direction or tactic
- rule-based action priors disguised as “local utility”
- benchmark-tuned losses with weak causal interpretation

If a change works only because the weights happen to favor a particular tactic,
it does not count as world-model progress.

### 3. Chosen-Action-Only Training Used as Counterfactual Selector

This is a specific anti-pattern and should be treated as a red-flag failure
mode.

Forbidden mismatch:

- training data records only the action chosen by the previous policy
- the model learns outcomes for those observed `(state, action)` pairs
- the model is then used to rank **all** candidate actions online

This creates a distribution shift from “observed action outcome model” to
“counterfactual action selector” and strongly risks directional or tactical
collapse.

### 4. Uninspectable Survival Gain

Any gain that cannot be traced to better local prediction is out-of-bounds.

The following claims are insufficient on their own:

- `avg_survival` improved
- the learned local mode beats the baseline on one run
- one utility formulation “feels better”

Without inspectable prediction improvement, such gains count as probable
tactical drift.

## Data Guardrails

Any `Stage 90R` dataset design should satisfy the following rules.

### Rule 1 — State-Centered Framing

The dataset should be centered around local states, not merely around whatever
action the old policy happened to take.

Preferred form:

- one local state
- several candidate primitive actions
- short-horizon outcomes for those actions, observed or approximated explicitly

This is preferable to a flat stream of chosen actions because the research
question is action comparison under local context.

### Rule 2 — Explicit Regime Coverage

The dataset should report coverage over at least these local regimes:

- nearby hostile contact or approach
- visible local resource opportunity
- neutral exploration / wandering
- low-vitals recovery situations

If the dataset is dominated by generic movement with sparse threat/resource
cases, the stage will tend to learn movement priors rather than local causal
competence.

### Rule 3 — Action Diversity Checks

Every dataset artifact should include:

- action distribution
- per-regime action distribution
- class balance for threat-present vs threat-absent samples
- count of samples with valid escape-style labels

The stage should treat strong action imbalance as a first-class warning.

## Training Guardrails

Any learned local component should satisfy the following rules.

### Rule 1 — Prediction First

The primary supervised targets should be predictive local consequences, not
direct action desirability.

### Rule 2 — Ranking Evidence Required

If the model is later used for action comparison, offline evaluation must show
that it ranks actions meaningfully within the same local state.

Examples of acceptable evidence:

- pairwise preference accuracy
- top-1 agreement with explicit short-horizon rollouts
- threat-slice ranking metrics
- resource-opportunity slice ranking metrics

A good `survival_acc` alone is not sufficient.

### Rule 3 — Anti-Collapse Offline Gates

Before online canary evaluation, the checkpoint should pass explicit
anti-collapse diagnostics:

- top predicted action entropy
- no single primitive dominates almost all local states
- threat-present slices produce materially different rankings from neutral slices
- resource-facing states do not collapse onto a generic movement action

If these checks fail, online evaluation should be treated as expected to be
misleading.

## Integration Guardrails

### 1. Local Predictor Feeds Decision Layers

The intended long-term form is:

`local predictor -> planner/world-model/stimuli -> action choice`

not:

`local predictor -> argmax action -> done`

### 2. Local-Only Mode Is Diagnostic, Not Canonical

`local-only` is a stress test and falsification tool.

It may be useful for:

- exposing degenerate action priors
- detecting representation mismatch
- checking whether the learned component contains any usable control signal

It is not the canonical target architecture for this stage.

### 3. Promotion Path Must Stay Conceptual

If `Stage 90R` produces knowledge worth keeping, that knowledge should be
expressible later as:

- refined local dynamics knowledge
- verified threat semantics
- reusable local affordance relations

It should not produce opaque action tables or benchmark-specific reflexes.

## Evaluation Rules

Every `Stage 90R` report or experiment review must answer these questions.

1. What concrete aspect of local world understanding improved?
2. How is that improvement measured independently of aggregate survival?
3. Can the model distinguish consequences of two different actions in the same
   local state?
4. Does the online behavior improve because those predictions improved?
5. Is the learned knowledge inspectable enough to be reused or promoted?

If these questions cannot be answered, the stage result cannot be called a
world-model improvement.

## Decision Standard

### PASS

`Stage 90R` passes only if all of the following are true:

- the local component improves prediction quality on short-horizon local
  dynamics
- action comparison quality improves in a measurable and inspectable way
- online behavior improves in a way consistent with those better predictions
- the result is explainable without tactical Crafter-specific patches

### PARTIAL

`Stage 90R` is partial if:

- there is evidence of useful local signal
- but the signal is mixed with policy bias, collapse risk, or unclear causal
  attribution

### FAIL

`Stage 90R` fails if:

- the learned component behaves mainly as a tactical local controller
- gains appear without inspectable prediction improvement
- collapse or shortcut behavior dominates the observed signal

## Review Checklist

Use this checklist for any new `Stage 90R` PR, experiment, or report.

- Does the change improve prediction of local consequences rather than only
  action preference?
- Can the new component explain why one action is safer or more useful than
  another in the same viewport?
- Is the dataset framed around state-conditioned action comparison, not just
  chosen actions from the old policy?
- Are action-distribution and anti-collapse diagnostics reported?
- Is the online gain traced back to better local world understanding?
- Could the learned output later become reusable knowledge, not just a tactic?
- If Crafter-specific constants were added, are they world facts or just
  policy tuning?

If any of these answers is “no”, the burden of proof is on the change author.

## Non-Goals

This document does not require `Stage 90R` to:

- solve Crafter end-to-end
- replace the full planner with a learned controller
- prove inter-generation transfer
- settle the whole architecture question alone

It only requires the stage to stay intellectually honest about what kind of
progress it is making.

## Immediate Consequence For Current Stage 90R Work

Under this specification, the safest interpretation of the current code is:

- local observation packaging and short-horizon consequence labeling are
  in-bounds
- local-only argmax control is allowed only as a canary
- any future change that treats local-only argmax as the intended architecture
  is out-of-bounds
- future iterations should prefer stronger state-conditioned action-comparison
  evidence over larger end-to-end local-policy gains

