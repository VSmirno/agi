# Stage 90 Cause-Finding Design

## Goal

Establish the dominant current cause of death in the post-Stage-89 stack before any new survival fix is designed.

Stage 90 is intentionally defined as a **cause-finding stage first**:

- collect a fast diagnostic slice of death episodes
- convert those deaths into a small failure taxonomy
- identify one dominant repeated failure mode
- validate that failure mode on a larger baseline-sized run

No architectural fix should be discussed as part of Stage 90 execution until the cause is confirmed on full validation.

## Why This Stage Exists

Stage 89 showed that projectile perception, tracking, and imminent arrow dodge are no longer the main bottleneck. The remaining wall appears to sit higher in the stack, around broad hostile survival behavior against `zombie` and `skeleton`, but that is still a hypothesis rather than a proven root cause.

The project has already paid several times for acting on misleading telemetry. Stage 90 therefore starts by proving the cause of death before changing planner or world-model logic again.

## Scope

Stage 90 covers only:

- death forensics on the current Stage 89 agent stack
- bundle-level capture of the last short horizon before death
- aggregation into a small, explicit failure taxonomy
- selection of one dominant failure mode
- confirmation that the same cause still dominates on a full run

Stage 90 does **not** cover:

- implementing the survival fix
- claiming Phase I completion
- advancing to causal-learning work
- broad planner redesign beyond instrumentation needed to explain deaths

## Stage Order

### 1. Quick Slice

Run a fast diagnostic collection over roughly `30-50` death episodes.

Purpose:

- get a rapid picture of the current death landscape
- detect repeated failure patterns quickly
- avoid spending a full baseline run on an unstructured investigation

Output:

- raw death bundles
- quick-slice summary with failure-mode counts
- shortlist of traces worth manual inspection

### 2. Cause Analysis

Analyze the quick slice to identify one dominant repeated failure mode.

This step combines:

- aggregate counts
- repeated pattern detection across seeds
- manual reading of the clearest traces

A failure mode is only a valid candidate if it can be described as one consistent architectural failure, not a collection of unrelated bad outcomes.

### 3. Full Validation

Run the same diagnosis pipeline on a larger baseline-sized evaluation set.

Purpose:

- confirm that the chosen dominant failure mode was not a quick-slice artifact
- measure how much of the total death mass it explains
- decide whether the cause is established strongly enough to justify fix design

If full validation does not confirm the diagnosis, the stage returns to diagnosis rather than proceeding to implementation.

## Failure Taxonomy

Each death must be assigned one primary bucket:

- `missed_imminent_threat`
- `bad_tradeoff_under_threat`
- `no_escape_plan_generated`
- `state_desync_or_execution_failure`
- `resource_vitals_commitment_error`
- `unknown`

The taxonomy is deliberately small. The goal is not to label every nuance, but to separate the major architectural failure classes cleanly enough to support a single next fix.

## Death Trace Bundle

Each collected death episode should produce one compact bundle with at least:

- `seed`
- `episode_id`
- `death_cause`
- `t_final`
- recent hostile geometry before death
- recent body and vital state
- top-k candidate plans
- score and rank for each candidate
- chosen plan
- predicted danger / predicted viability on the short horizon
- actual short-horizon outcome after plan selection
- derived error label:
  - `prediction`
  - `ranking`
  - `generation`
  - `execution`
  - `unknown`

The bundle should be rich enough to answer:
"Did the agent die because it failed to predict danger, failed to generate a viable response, ranked the wrong response highest, or acted on stale/incorrect state?"

## Cause Selection Rule

Stage 90 may name a dominant cause only if all of the following hold:

- it explains a meaningful share of deaths in the quick slice
- it repeats across multiple seeds rather than one lucky or unlucky trace
- it maps to one coherent architectural failure
- it has at least `2-3` readable representative traces
- it survives full-validation re-checking

Until those conditions hold, the correct stage outcome is "cause not established".

## Deliverables

### Scripts

- `experiments/stage90_quick_slice.py`
- `experiments/analyze_stage90_deaths.py`
- `experiments/stage90_full_validation.py`

### Artifacts

- `_docs/stage90_quick_slice.json`
- `_docs/stage90_quick_slice_summary.json`
- `_docs/stage90_full_validation.json`

### Report

- `docs/reports/stage-90-cause-report.md`

The report should include:

- quick-slice setup
- failure taxonomy summary
- representative traces
- chosen dominant failure mode
- rejected alternative explanations
- full-validation confirmation or rejection

## Success Criteria

Stage 90 cause-finding is successful only if:

- the quick slice produces a readable failure breakdown
- `unknown` does not dominate the analysis
- one dominant failure mode is identified
- the same mode remains dominant on full validation
- the report can state the cause in architectural terms rather than Crafter-specific reflex language

If the output is only "deaths are mixed" or "hostiles are still hard", the stage has not succeeded.

## Anti-Tuning Constraints

This stage must follow `docs/STAGE_REVIEW_CRITERIA.md` and `docs/ANTI_TUNING_CHECKLIST.md`.

In particular:

- the diagnosis must be expressible without relying on a hidden policy patch
- telemetry categories must map to the correct layer: `facts`, `mechanisms`, `experience`, or `stimuli`
- the stage must not smuggle in a fix while pretending to only instrument behavior

Good diagnosis language:

- short-horizon contact danger is predicted too late
- candidate generation lacks escape trajectories under local threat
- plan ranking overvalues resource continuation under imminent lethal contact

Bad diagnosis language:

- agent still dies to zombies sometimes
- make it flee better in Crafter

## Risks

- the quick slice may under-sample an important but less frequent death mode
- `unknown` may remain too large if bundles are not rich enough
- one visible pattern may reflect stale telemetry rather than a true planner failure
- the apparent dominant cause may disappear on full validation

## Exit Condition

The only valid exit from this design is one of:

1. `cause established`
   a dominant failure mode is proven on quick slice and full validation
2. `cause not established`
   instrumentation must be improved and diagnosis repeated

Design of the survival fix starts only after exit condition `cause established`.
