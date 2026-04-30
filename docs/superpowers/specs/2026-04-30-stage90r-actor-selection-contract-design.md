# Stage90R Actor-Selection Contract Redesign

## Goal

Keep the proven `9083357` promotion path, but remove the remaining seed-fragile
actor dynamics by changing what the actor is trained to do.

The actor should learn:

- to choose among the candidate actions available in a state
- from state-level parent guidance plus outcome-side advisory
- on the same candidate set that the ranker/evaluator scores

It should stop learning primarily from global primitive-action imitation that
ignores which candidate set is actually present in the state.

## Why The Current Baseline Is Still Fragile

The shared guidance bottleneck plus direct agreement fixed the old terminal
`move_up` collapse and made checkpoint promotion robust on the tiny
`mixed_control` slice. But hyper multiseed verification still showed transient
seed-dependent actor reflexes:

- some seeds still pass through `move_up 4/4`
- others swing through `move_left 4/4` or `move_right 4/4`
- the ranker is more stable than the actor, but the actor still takes a longer
  route to align

The root issue is that the actor is still supervised as a global primitive
classifier. Even after advisory blending, its direct supervised target is
expressed in the full action space, not in the candidate set of the current
state.

That leaves a mismatch:

- ranker/evaluator learns "which candidate wins in this state"
- actor still learns "which primitive button should be high in general"

The agreement pass helps later, but the actor starts from the wrong contract.

## Recommended Direction

Replace flat row-wise actor imitation with state-level actor-selection training.

Recommended contract:

1. Build actor targets per state signature, not per flat teacher row.
2. Filter those targets to the candidate actions actually available in the
   current state.
3. Renormalize over that candidate set.
4. Train the actor on those candidate-normalized targets.
5. Keep evaluator agreement on the same candidate set as a secondary alignment
   term.

This preserves the ideology:

- planner remains parent guidance
- ranking remains outcome-based evidence
- learner still picks the primitive action
- but the learner is trained at the correct abstraction layer: local action
  selection under a concrete state

## Architecture Changes

### 1. State-Level Actor Targets

For each train state sample, build one actor target distribution over its
candidate actions by combining:

- planner soft signature target
- state-level advisory target from ranking winners
- existing consistency/downweight logic for contradictory planner signatures

The target should be defined in the full action space first, then projected onto
the candidate set of the state and renormalized.

### 2. Candidate-Set Actor Loss

Replace the primary mixed-control actor training loss with a candidate-set loss:

- run the actor once on the state
- take the logits only for that state's candidate actions
- compare against the candidate-normalized target distribution

This directly trains "what should I pick here?" instead of "which primitive is
best in the abstract?"

### 3. Candidate-Conditioned Actor Scorer

Do not evaluate the hard-gated actor through a global primitive argmax alone.

For `mixed_control` / `rescue`, the actor should expose a candidate-conditioned
selection score on the same action-conditioned branch used for per-action
evaluation. The actor report and runtime candidate ranking should read that
candidate-conditioned score over the available actions in the current state.

Otherwise an absent action can still dominate the global logits even if the
candidate-set teacher loss is correct.

### 4. Agreement Becomes Secondary, Not Compensatory

Keep the current actor-ranker agreement pass, but it should reinforce an already
correct actor-selection target, not rescue a mismatched primitive classifier.

This means:

- teacher/advisory selection loss is the primary actor signal
- evaluator agreement is the stabilizer

### 5. Bootstrap Modes Stay Simple

For `planner_bootstrap` and `teacher_prep`, the existing flatter teacher path is
still acceptable as advisory pretrain behavior.

The state-level actor-selection contract is most important for real hard-gated
stages such as:

- `mixed_control`
- `rescue`

## Training Changes

### Mixed-Control / Rescue

Use a new state-sample actor training pass:

- input: `train_state_samples`
- target: candidate-normalized state-level actor target distribution
- optional weight: signature consistency / support weight

Then run the existing actor-ranker agreement pass.

### Bootstrap / Advisory Modes

Keep the current soft teacher path unless later evidence says otherwise.

## Non-Goals

This redesign does not:

- add new world-prior heads
- retune gate thresholds
- add hard-coded action rules
- change checkpoint promotion criteria again
- replace the evaluator with a new listwise ranker

It only fixes the actor's training contract.

## Evaluation

Primary narrow evaluation:

- same tiny `mixed_control` slice
- hyper verification
- seeds: `7, 11, 17, 23, 31`

Success criteria:

- no seed ends in the old terminal `move_up 4/4` collapse
- transient actor reflexes shrink versus the current `9083357` baseline
- actor trajectories become closer to ranker trajectories earlier in training
- promotion robustness is preserved

Secondary coverage step after that:

- run the same redesign on a neighboring hard-gated regime such as `rescue`

## Risks

### Risk: Candidate-projected targets become too sparse

Mitigation:

- only train on states with at least two candidates
- keep renormalization stable
- keep flat teacher path for bootstrap modes

### Risk: Planner and advisory targets still conflict after projection

Mitigation:

- preserve consistency weighting
- log projected candidate distributions for diagnostics

### Risk: Actor overfits ranker and loses useful planner bias

Mitigation:

- keep planner contribution in the blended state-level target
- keep agreement as a secondary term, not the only target

## Definition Of Done

This redesign slice is done when:

- mixed-control actor training no longer depends primarily on flat primitive
  teacher rows
- actor supervision is built from state-level candidate-aware targets
- focused tests cover target projection and candidate-normalized loss plumbing
- hyper multiseed verify is rerun
- the new path is compared directly against `9083357`
