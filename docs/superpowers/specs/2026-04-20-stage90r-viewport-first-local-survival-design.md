# Stage 90R Viewport-First Local Survival Design

## Goal

Test a stricter Phase I hypothesis:

the agent should first learn to act coherently from the **current viewport**
before more memory, more planner depth, or more symbolic local shortcuts are
added.

Stage 90R is a reset-stage inside Phase I. It does not assume that the current
`planner + spatial_map + near_concept` stack is the right substrate for the
next survival gain.

## Why This Stage Exists

Stage 90 cause-finding removed several real mechanism bugs:

- passive proximity damage for hostile contact now reaches the simulator
- diagnostics preserve zero predicted health correctly
- forward simulation no longer grants remote `do target` effects before the
  target is reached

Those fixes were real, but they did not break the main survival wall.

Video review and death traces now point to a deeper problem:

- the agent often fails to exploit obviously useful local opportunities
- the agent often fails to react coherently to immediate local threat
- behavior is still shaped too much by stale or overloaded intermediate state
  such as `spatial_map` and `near_concept`

The next honest question is therefore not "how do we tune the planner more?",
but:

**can the system learn better local behavior directly from the viewport?**

## Position In The Roadmap

Stage 90R remains inside `Phase I — Dynamic World Model`.

It should be treated as a reset/bridge stage between:

- Stage 90 cause establishment
- Stage 91 dynamic planning validation

Stage 91 should not proceed until Stage 90R answers whether current failure is
primarily caused by the local policy interface rather than by one more missing
dynamic rule.

## Design Constraints

The stage is governed by the following hard constraints:

- `viewport-first`
  current viewport is the primary truth for action selection
- `no-near-concept-policy`
  `near_concept` may remain for debugging/compatibility only, not as a primary
  policy signal
- `no-global-memory-first-policy`
  information outside the current viewport must not dominate action choice
- `geometry-preserved`
  local scene should stay spatial, not collapse into one selected label
- `learn-local-behavior`
  threat response, local resource value, and immediate affordance should be
  learned from observation rather than hand-coded
- `no-crafter-specific-reflexes`
  no `if zombie nearby then ...` or similar policy patches in generic code
- `memory-is-secondary`
  `spatial_map` and longer-horizon memory may exist, but only as auxiliary
  signals or ablation baselines

## Non-Goals

Stage 90R does not attempt to:

- solve the whole Crafter game
- fully replace all planning with end-to-end RL
- introduce a new global memory architecture
- optimize crafting chains directly
- claim inter-generation knowledge gain

## Approaches Considered

### 1. Continue planner-first repair

Keep current stack and continue improving threat ranking, candidate generation,
and symbolic local signals.

Pros:
- cheapest incremental path
- reuses most of current machinery

Cons:
- directly conflicts with current diagnosis
- risks another loop of fixing symptoms in overloaded intermediate state

### 2. Viewport-first learned local action evaluator

Learn a short-horizon local action model from viewport-centered data and use it
to score primitive actions under immediate threat/opportunity.

Pros:
- directly tests the new hypothesis
- preserves local geometry
- avoids adding new symbolic shortcuts
- still small enough to validate inside current stack

Cons:
- requires new data path and model head
- may only solve local coherence, not full long-horizon behavior

### 3. Full end-to-end policy replacement

Replace planner path with a learned policy over viewport and body state.

Pros:
- clean conceptual break

Cons:
- too large for the current stage
- would blur whether the gain came from local-view grounding or wholesale
  architecture replacement

### Recommendation

Choose **Approach 2**.

It is the smallest stage that honestly tests the new hypothesis without
collapsing back into planner tuning.

## Proposed Architecture

### 1. Local Observation Package

Define one compact local policy input:

- current viewport tensor / local semantic scene
- current vitals
- compact inventory state

No selected `near_concept` label should be required to use the input.

### 2. Local-Only Policy Path

Add a mode where immediate action choice is produced from the local observation
package with **no primary dependency** on:

- `spatial_map.find_nearest(...)`
- long-range symbolic target search
- `near_concept` fallback shortcuts

This mode is the main experimental object of the stage.

### 3. Short-Horizon Supervision

Build training examples from real rollouts:

- local observation at time `t`
- primitive action at time `t`
- short-horizon outcome over `H` steps

The label space should reflect local utility, not benchmark score directly:

- immediate damage / survival consequence
- local resource acquisition consequence
- whether action improved or worsened local escape viability

### 4. Learned Local Action Evaluator

Train a small model that estimates short-horizon value for primitive actions
from the local observation package.

This is not yet a full world model replacement. It is a learned local decision
surface for immediate behavior.

### 5. Controlled Integration

Integrate the learned local evaluator in a way that allows clean ablations:

- advisory mode for offline analysis
- local-only action-selection mode
- optional hybrid comparison against the current planner path

Stage conclusions must come from these explicit comparisons, not from one mixed
configuration whose source of gain is unclear.

## Data And Evaluation

### Required artifacts

- local-decision dataset from real rollouts
- trained local action evaluator checkpoint
- offline evaluation of short-horizon local prediction quality
- online evaluation on `minipc`
- stage report with ablation comparisons

### Primary evaluation questions

1. Does the local evaluator choose safer actions under immediate hostile
   contact than the current stack?
2. Does it exploit nearby useful resources more coherently than the current
   stack?
3. Does it reduce wandering when the viewport already contains a good local
   opportunity?

### Suggested metrics

- `avg_survival`
- share of deaths in immediate hostile contact
- local escape success rate under short-horizon threat
- local useful-resource capture rate
- wandering rate when a useful local action was available
- offline action-ranking agreement against realized short-horizon outcomes

## Deliverables

### Scripts / entry points

- local dataset collection script
- local evaluator training script
- local-only / hybrid evaluation script

### Artifacts

- local dataset artifact under `_docs/`
- local evaluator eval artifact under `_docs/`
- stage report under `docs/reports/`

### Documentation

- this design spec
- matching implementation plan
- `docs/ASSUMPTIONS.md` update describing the reset constraints

## Success Criteria

Stage 90R is successful only if all of the following hold:

- the system can run in a viewport-first local policy mode
- the learned local evaluator beats the current stack on at least one clear
  local behavior measure
- online behavior shows less local incoherence under immediate threat or
  nearby opportunity
- any gain can be explained without introducing new Crafter-specific policy
  branches
- the result is strong enough to justify moving toward Stage 91 validation

If the result is only "the number moved slightly" without a local-behavior
explanation, mark it `PARTIAL` at best.

## Anti-Tuning Requirements

This stage must pass `docs/STAGE_REVIEW_CRITERIA.md` and
`docs/ANTI_TUNING_CHECKLIST.md`.

In particular:

- gains must be explained as better local observation-to-action grounding
- no entity-specific hand-authored reflexes may be introduced
- memory ablations must show whether the gain really came from the viewport
  path rather than a hidden planner dependency

Good explanation:

- local action selection is now grounded in current spatial evidence
- the agent better separates immediate danger from locally useful opportunity

Bad explanation:

- the agent now knows what to do when a zombie is one tile away

## Risks

- local-only mode may improve coherence but lose too much long-horizon value
- training labels may leak current-policy bias instead of true short-horizon
  utility
- the evaluator may become a disguised heuristic scorer unless ablations stay
  strict
- the stage may show that local behavior is not the real bottleneck after all

## Exit Conditions

The valid exits are:

1. `viewport-first local gain confirmed`
   local learned behavior improves immediate decision quality enough to justify
   Stage 91 validation
2. `viewport-first hypothesis falsified`
   local-only policy path does not beat the current stack, so the bottleneck
   likely sits elsewhere

Both outcomes are acceptable. The purpose of Stage 90R is to answer the
hypothesis cleanly, not to force a PASS.
