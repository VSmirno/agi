# Stage 90R Implementation Plan — Viewport-First Local Survival

**Spec:** `docs/superpowers/specs/2026-04-20-stage90r-viewport-first-local-survival-design.md`  
**Date:** 2026-04-20

## Goal

Execute a reset-stage that tests whether the agent can improve immediate local
behavior from the current viewport, with reduced dependence on `spatial_map`
and `near_concept`.

## Success Criteria

Stage 90R is complete only if:

- a reproducible viewport-first local-policy path exists
- its inputs and labels are explicit and inspectable
- online `minipc` runs compare it against the current stack
- at least one local-behavior metric improves clearly
- the final report explains whether the viewport-first hypothesis is confirmed
  or falsified

## Task 1 — Freeze the comparison baseline

**Files:**
- existing Stage 89 / Stage 90 artifacts under `_docs/`
- evaluation runner scripts

Steps:
1. Name the baseline artifacts that Stage 90R will compare against.
2. Freeze the comparison metrics:
   - `avg_survival`
   - death-cause breakdown
   - local-contact death share if derivable
3. Reuse the same baseline reference in every Stage 90R artifact.

Verification:
- one baseline reference block is reused consistently in report and eval output

## Task 2 — Define the viewport-first input package

**Files:**
- new helper module under `src/snks/agent/` if needed
- collector / training scripts

Steps:
1. Define the local observation package:
   - viewport tensor or spatial local scene
   - vitals
   - compact inventory
2. Explicitly exclude `near_concept` as a required policy input.
3. Explicitly mark `spatial_map` as auxiliary only.

Done criteria:
- one concrete schema is used by data collection, training, and eval

## Task 3 — Build passive local-decision dataset collection

**Files:**
- new experiment script under `experiments/`
- optional helper in `src/snks/agent/`

Steps:
1. Collect per-step local observations from real rollouts.
2. Record:
   - local observation package
   - primitive action taken
   - short-horizon outcomes over `H`
   - immediate damage change
   - local resource gain
   - contact/escape change if measurable
3. Keep the collector passive: no behavior change during data collection.

Verification:
- a real dataset artifact is produced on `minipc`
- samples are readable and schema-consistent

## Task 4 — Define short-horizon local labels

**Files:**
- dataset helper module
- training script

Steps:
1. Convert rollout outcomes into local utility targets.
2. Keep labels about local consequence, not benchmark score:
   - threat cost
   - local utility gain
   - escape viability delta
3. Document label semantics clearly enough for later report review.

Gate:
- label definitions are inspectable and not disguised Crafter-specific reward
  shaping

## Task 5 — Train a learned local action evaluator

**Files:**
- new training script under `experiments/`
- model/helper module under `src/snks/agent/` or `src/snks/learning/`

Steps:
1. Train a compact model on the viewport-first input package.
2. Predict short-horizon value for primitive actions.
3. Save checkpoint and offline metrics.

Verification:
- checkpoint artifact exists
- offline evaluation shows the model learned non-trivial action ranking

## Task 6 — Add a local-only policy mode

**Files:**
- current agent entry point
- optional policy adapter module

Steps:
1. Add a mode where immediate action selection is driven by the learned local
   evaluator.
2. Ensure this mode does not depend primarily on:
   - `near_concept`
   - `spatial_map.find_nearest`
   - long symbolic target search
3. Preserve an ablation switch for current-stack comparison.

Verification:
- the agent can run end-to-end in local-only mode
- runtime logs show the mode is actually active

## Task 7 — Add controlled comparisons

**Files:**
- evaluation script under `experiments/`

Steps:
1. Compare at least:
   - current stack
   - viewport-first local-only mode
   - optional hybrid mode if useful
2. Run all real evaluations on `minipc`.
3. Save artifacts under `_docs/`.

Verification:
- comparison artifacts are directly comparable
- config differences are explicit

## Task 8 — Measure local behavior, not only survival

**Files:**
- evaluation / analysis scripts

Steps:
1. Compute local-behavior metrics:
   - local escape success under nearby hostile threat
   - local useful-resource capture rate
   - wandering despite local opportunity
2. Pair them with `avg_survival` and death-cause metrics.
3. Reject any result that only improves one aggregate number without local
   behavioral evidence.

Done criteria:
- report-ready tables exist for both aggregate and local metrics

## Task 9 — Stage review and decision

**Files:**
- `docs/reports/stage-90r-viewport-first-local-survival-report.md`
- `docs/ASSUMPTIONS.md`

Steps:
1. Write the stage report using `docs/STAGE_REVIEW_CRITERIA.md`.
2. Record:
   - baseline used
   - dataset setup
   - model design
   - ablations
   - local metrics
   - online metrics
   - decision
3. Update `docs/ASSUMPTIONS.md` with any new limits discovered.

Decision rule:
- `PASS` if viewport-first local behavior clearly improves and survives review
- `PARTIAL` if signal exists but remains mixed or tactically shaped
- `FAIL` if the viewport-first hypothesis is falsified

## Dependency Order

```text
Task 1 ──> Task 2 ──> Task 3 ──> Task 4 ──> Task 5 ──> Task 6 ──> Task 7 ──> Task 8 ──> Task 9
```

## Risks

- local dataset may be too biased by current policy behavior
- short-horizon labels may be too noisy
- local-only mode may regress survival sharply before learning becomes useful
- hybrid mode may hide where the gain really came from
- the stage may prove that local policy was not the dominant wall
