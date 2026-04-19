# Stage 90 Implementation Plan — Cause Finding

**Spec:** `docs/superpowers/specs/2026-04-19-stage90-cause-finding-design.md`  
**Date:** 2026-04-19

## Goal

Establish the dominant current cause of death in the post-Stage-89 stack through a fast diagnostic slice, explicit failure taxonomy, and full-run validation.

This plan does **not** include implementation of the eventual survival fix. It ends at one of two outcomes:

- `cause established`
- `cause not established`

## Success Criteria

Stage 90 cause-finding is complete only if all of the following hold:

- a quick-slice run produces readable death bundles for `30-50` deaths
- the analysis produces one explicit dominant failure-mode candidate
- the candidate is supported by `2-3` representative traces
- the same candidate remains dominant on full validation
- `unknown` does not dominate either quick-slice or full-validation breakdown
- the final report states the cause in architectural terms, not Crafter-specific policy language

## Task 1 — Freeze the Stage 89 baseline bundle

**Files:**
- `experiments/stage89_eval.py`
- existing `_docs/` Stage 89 outputs

Steps:
1. Record the current Stage 89 comparison point used for cause-finding.
2. Freeze the baseline metrics that must accompany every Stage 90 artifact:
   - `avg_survival`
   - `death_cause_breakdown`
   - `unknown_death_share`
3. Save or reference one stable baseline artifact path for later comparisons.

Verification:
- one baseline JSON path is named in the eventual report
- later Stage 90 scripts reuse the same comparison basis

## Task 2 — Define the death-bundle schema

**Files:**
- new helper module under `src/snks/agent/` if needed
- `experiments/stage90_quick_slice.py`

Steps:
1. Define the minimal `Death Trace Bundle` schema from the spec.
2. Decide exactly how many last steps before death to keep.
3. Define a machine-readable error-label field:
   - `prediction`
   - `ranking`
   - `generation`
   - `execution`
   - `unknown`
4. Keep the schema compact enough for repeated runs and manual reading.

Done criteria:
- one concrete JSON schema is used consistently by all Stage 90 scripts

## Task 3 — Instrument the current Stage 89 agent for death capture

**Files:**
- `src/snks/agent/vector_mpc_agent.py`
- optionally a new diagnostic helper module

Steps:
1. Capture the last short horizon before death without changing planner behavior.
2. Record:
   - hostile geometry
   - body/vitals
   - top-k candidate plans
   - scores / ranks
   - chosen plan
   - predicted danger / viability
   - actual short-horizon outcome
3. Ensure the instrumentation is passive and does not alter plan ranking.

Verification:
- one short local run produces at least one valid death bundle
- planner outputs before/after instrumentation remain behaviorally unchanged

## Task 4 — Implement `experiments/stage90_quick_slice.py`

**Files:**
- `experiments/stage90_quick_slice.py`

Steps:
1. Build the current Stage 89 stack with the same model/textbook path as current eval.
2. Run until roughly `30-50` death episodes are collected.
3. Write raw bundles to `_docs/stage90_quick_slice.json`.
4. Save a small run summary:
   - episodes run
   - deaths captured
   - death causes
   - unknown bundle failures if any

Gate:
- quick-slice artifact exists and contains enough bundles for analysis

## Task 5 — Implement the failure taxonomy pass

**Files:**
- `experiments/analyze_stage90_deaths.py`

Steps:
1. Convert raw bundles into the Stage 90 primary buckets:
   - `missed_imminent_threat`
   - `bad_tradeoff_under_threat`
   - `no_escape_plan_generated`
   - `state_desync_or_execution_failure`
   - `resource_vitals_commitment_error`
   - `unknown`
2. Emit `_docs/stage90_quick_slice_summary.json`.
3. Include:
   - count by bucket
   - share by bucket
   - seed coverage by bucket
   - shortlist of representative episode ids

Verification:
- summary JSON is readable without opening raw bundles
- at least one bucket candidate is visible from the aggregate data

## Task 6 — Manual trace review and cause selection

**Files:**
- `_docs/stage90_quick_slice.json`
- `_docs/stage90_quick_slice_summary.json`
- working notes or final report draft

Steps:
1. Read the top `2-3` traces for the leading candidate bucket.
2. Confirm that they represent one coherent architectural failure.
3. Identify one secondary candidate in case the first fails on full validation.
4. Write the explicit provisional diagnosis in one sentence.

Done criteria:
- one dominant candidate and one backup candidate are named
- the diagnosis is stated in architecture language rather than environment slang

## Task 7 — Add a stop-gate before full validation

Before any larger run, answer explicitly:
1. Is there one dominant candidate rather than many tied buckets?
2. Do the representative traces actually match the bucket label?
3. Is `unknown` low enough that the analysis is trustworthy?

If any answer is `no`, stop and improve instrumentation or taxonomy before proceeding.

## Task 8 — Implement `experiments/stage90_full_validation.py`

**Files:**
- `experiments/stage90_full_validation.py`

Steps:
1. Re-run the same bundle capture on a baseline-sized evaluation set.
2. Re-apply the same taxonomy logic without changing bucket definitions.
3. Write `_docs/stage90_full_validation.json`.
4. Track:
   - dominant bucket share
   - seed coverage
   - `unknown` share
   - agreement with quick-slice diagnosis

Verification:
- full-validation artifact exists
- quick-slice and full-validation use the same schema and taxonomy

## Task 9 — Decide `cause established` vs `cause not established`

**Files:**
- `_docs/stage90_quick_slice_summary.json`
- `_docs/stage90_full_validation.json`

Decision rule:
- `cause established` only if the same dominant bucket survives full validation
- `cause not established` if the dominant bucket collapses, ties, or is drowned by `unknown`

Output:
- one explicit decision block for the report
- if rejected, state whether the next iteration should improve:
  - bundle richness
  - taxonomy logic
  - seed coverage

## Task 10 — Write `docs/reports/stage-90-cause-report.md`

**Files:**
- `docs/reports/stage-90-cause-report.md`
- `docs/ASSUMPTIONS.md`

Steps:
1. Record:
   - baseline used
   - quick-slice method
   - taxonomy summary
   - representative traces
   - dominant cause decision
   - rejected alternative explanations
2. Add Stage Review template answers:
   - ideological debt addressed
   - layer changed
   - evidence of improvement
   - why the result is architectural, not tactical
   - remaining assumptions / walls
   - decision
3. Update `docs/ASSUMPTIONS.md` with any new diagnostic blind spots or telemetry limits found during Stage 90.

Done criteria:
- report is sufficient to justify either moving to fix design or repeating diagnosis

## Task 11 — Final Stage Gate

Before any fix design begins, answer explicitly:
1. What is the dominant cause of death?
2. What share of deaths does it explain?
3. Why is this an architectural failure rather than a Crafter-local anecdote?
4. What evidence would falsify this diagnosis?

If these answers are weak or ambiguous, Stage 90 remains in cause-finding mode.

## Dependency Order

```text
Task 1 ──> Task 2 ──> Task 3 ──> Task 4 ──> Task 5 ──> Task 6 ──> Task 7
                                                                │
                                                                └──> Task 8 ──> Task 9 ──> Task 10 ──> Task 11
```

## Risks

- passive instrumentation may accidentally perturb planner behavior
- bundles may omit one variable needed to distinguish `prediction` from `ranking` failure
- quick-slice distribution may overstate a noisy local pattern
- `unknown` may stay too large until taxonomy logic is sharpened
- full validation may invalidate the quick-slice diagnosis, forcing another diagnostic iteration
