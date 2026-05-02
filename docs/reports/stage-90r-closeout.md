# Stage 90R Closeout

**Date:** 2026-05-02  
**Baseline implementation commit:** `71d1e29`  
**Baseline closeout commit:** `ea68e65`

## Purpose

This document closes the long-running `Stage 90 / 90R` line as one
consolidated baseline so later work can reference a single finished stage
instead of a chain of partially overlapping slices.

It does **not** rewrite the historical documents. Those remain the record of how
the project moved through diagnosis, reset, contract redesign, and emergency
control.

## What Stage 90 / 90R Covered

This line ended up containing four distinct kinds of work:

1. `Stage 90` cause-finding:
   - identify the dominant near-term death mechanisms
   - stop patching behavior before diagnosis

2. `Stage 90R` viewport-first reset:
   - test whether coherent local behavior had to be rebuilt from viewport-first
     inputs before more planner/memory strengthening

3. `Stage 90R` contract repairs:
   - shared guidance bottleneck
   - actor-selection contract redesign
   - rescue eligibility repairs

4. `Stage 90R` emergency-controller closeout:
   - first-class emergency safety controller
   - bounded textbook/config migration for emergency-relevant Crafter facts
   - rescue-side telemetry upgrade
   - bounded HyperPC compare above the frozen `9083357` rescue baseline

## Consolidated Outcome

The consolidated `Stage 90 / 90R` answer is:

- cause-finding was necessary
- viewport-first/local-behavior reset was necessary
- actor/ranker/rescue contracts were real structural debt
- rescue had to become a first-class controller rather than a disagreement patch

The final closeout baseline for this line is:

- implementation commit: `71d1e29`
- report: `docs/reports/stage-90r-emergency-controller-report.md`
- bounded compare result:
  - candidate `avg_survival = 190.0`
  - frozen `9083357 avg_survival = 179.25`
  - `early_hostile_deaths_without_rescue = 0`

## What Counts As Closed

Closed:

- the `Stage 90 / 90R` question of whether rescue should stay a
  disagreement-gated patch
- the specific emergency-controller implementation slice
- the bounded proof that this controller improves the frozen rescue baseline

Not closed:

- broader multi-seed rescue robustness
- CPU-only vs GPU online parity
- full stimuli-layer redesign
- repo-wide Crafter literal cleanup

Those are no longer `Stage 90 / 90R` work. They belong to subsequent stages.

## Canonical Baseline

When later documents say "current Stage 90R baseline", they should mean:

- code baseline: `71d1e29`
- closeout report: `docs/reports/stage-90r-emergency-controller-report.md`
- umbrella closeout: this document

This avoids using old midpoint artifacts like:

- cause-finding only
- viewport reset only
- actor-selection contract only
- rescue eligibility only

as if any of them were the final baseline by themselves.

## Decision

`Stage 90 / 90R` is **closed**.

Future work should treat it as a completed baseline and open a new stage with a
single, narrower success criterion.
