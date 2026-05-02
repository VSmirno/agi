# Stage 90R Implementation Plan — Emergency Safety Controller

**Spec:** `docs/superpowers/specs/2026-05-02-stage90r-emergency-safety-controller-design.md`  
**Date:** 2026-05-02

## Goal

Implement a first-class emergency safety controller for `mixed_control_rescue`
 and validate it against the current `9083357` online rescue baseline while
attempting a bounded migration of emergency-relevant Crafter facts into
textbook/config.

## Success Criteria

This stage is complete only if all of the following hold:

- emergency activation no longer depends mainly on `actor != planner`
- a first-class emergency action-selection path exists in the runtime
- the controller reads emergency-relevant world facts from textbook/config where
  practical
- rescue traces explain activation and override reasons explicitly
- bounded online `mixed_control_rescue` compare is rerun against `9083357`
- aggregate rescue behavior is judged against the frozen baseline, not only
  anecdotal episode wins

## Task 1 — Freeze the rescue baseline and acceptance gate

**Files:**
- existing `_docs/hyper_stage90r_mixed_control_rescue_compare*/`
- evaluation runner scripts under `experiments/`

Steps:
1. Freeze `9083357` as the hard rescue baseline for this stage.
2. Record the comparison metrics that every new artifact must report:
   - `avg_survival`
   - `death_cause_breakdown`
   - `controller_distribution`
   - `planner_dependence`
   - `learner_control_fraction`
   - `rescue_rate`
   - early hostile deaths with zero rescue events
3. Name one stable reference path for baseline compare reuse.

Verification:
- one reference block is reused in plan outputs, eval outputs, and final report

## Task 2 — Inventory current rescue contracts and hardcoded fact leaks

**Files:**
- `src/snks/agent/vector_mpc_agent.py`
- `src/snks/agent/stage90r_local_policy.py`
- post-mortem / diagnostic modules that reuse threat or range constants
- textbook/config files for Crafter facts

Steps:
1. Identify the current rescue trigger and rescue action-selection entry points.
2. Enumerate the emergency-relevant hardcoded facts now living in mechanism
   code.
3. Separate them into:
   - must move now for the controller path
   - safe to defer
4. Record any controller-path constants that still lack a config/textbook home.

Done criteria:
- one explicit "move now vs defer" inventory exists before runtime edits begin

## Task 3 — Define the bounded textbook/config schema

**Files:**
- Crafter textbook/config source files
- optional helper module for fact lookup

Steps:
1. Add emergency-relevant concept attributes needed by the new controller.
2. Cover only the facts needed by activation or emergency action choice.
3. Provide one lookup path that runtime controller code can consume without
   embedding raw local literals.
4. Keep the schema narrow enough that follow-up migrations remain possible
   without redesigning this stage.

Verification:
- controller-path threat/resource facts are readable from textbook/config
- runtime emergency logic no longer needs the moved raw literals

## Task 4 — Implement emergency activation as a first-class contract

**Files:**
- `src/snks/agent/vector_mpc_agent.py`
- optional new helper module under `src/snks/agent/`

Steps:
1. Build an emergency feature bundle from current runtime state:
   - vitals pressure
   - nearby hostile pressure
   - recent damage
   - repeated non-progress signal
   - evaluator danger-side candidate evidence
   - textbook threat facts
2. Compute one explicit activation score or activation decision from those
   features.
3. Make activation independent of the current primary disagreement gate.
4. Preserve any legacy trigger signal only as an input feature, not the main
   contract.

Verification:
- runtime can activate emergency mode even in dangerous-agreement states
- activation logic is testable outside the full episode loop

## Task 5 — Implement emergency action selection

**Files:**
- `src/snks/agent/vector_mpc_agent.py`
- optional helper module for emergency action ranking

Steps:
1. Add a distinct emergency action-selection path over current candidate
   actions.
2. Use a safety-first contract:
   - avoid immediate damage/traps first
   - prefer separation / exposure reduction
   - use planner/evaluator signals as inputs, not as the whole decision
3. Keep the normal learner/planner path unchanged when emergency mode is not
   active.
4. Distinguish planner-aligned overrides from independent emergency choices.

Done criteria:
- one explicit emergency selector exists in code
- normal path and emergency path have separate entry points

## Task 6 — Upgrade rescue-side telemetry

**Files:**
- runtime logging / eval output code
- `experiments/stage90r_eval_local_policy.py`

Steps:
1. Extend rescue trace output with:
   - activation reason
   - activation feature values
   - planner action
   - learner action
   - emergency-selected action
   - override source
   - immediate outcome delta
2. Ensure logs can distinguish:
   - never activated
   - activated, weak action
   - activated, reasonable action, unrecoverable trajectory
3. Keep output compact enough for bounded compare artifacts.

Verification:
- one bounded run produces readable episode-level emergency traces

## Task 7 — Add focused tests and local validation

**Files:**
- targeted tests under `tests/`
- optional small fixtures under `_docs/` or test assets

Steps:
1. Add unit or focused integration tests for:
   - textbook/config fact lookup
   - emergency activation in dangerous-agreement states
   - emergency selector behavior on representative candidate bundles
2. Add at least one regression test proving the new path can activate without
   disagreement.
3. Run local validation before wider online compare.

Verification:
- focused tests pass
- new controller path is exercised by tests rather than only by full runs

## Task 8 — Run bounded online compare

**Files:**
- `experiments/stage90r_eval_local_policy.py`
- new `_docs/` compare artifacts for this stage

Steps:
1. Reuse the existing bounded `mixed_control_rescue` compare shape.
2. Compare the new stage against frozen `9083357` metrics.
3. Capture:
   - aggregate metrics
   - episode examples
   - early-death-without-rescue counts
   - controller activation evidence
4. Keep GPU-path issues documented separately if the compare remains CPU-only.

Done criteria:
- one directly comparable artifact bundle exists
- rescue result can be judged against the frozen baseline without ambiguity

## Task 9 — Stage decision and closeout

**Files:**
- new stage report under `docs/reports/` or stage checkpoint docs
- `docs/ASSUMPTIONS.md` if needed

Steps:
1. Record whether the stage is:
   - `PASS`
   - `PARTIAL`
   - `FAIL`
2. Use the explicit fallback criterion from the spec:
   - if aggregate bounded rescue remains below `9083357`, do not keep stacking
     small trigger tweaks
3. Record:
   - what moved to textbook/config
   - what remained deferred
   - what the new traces now reveal
4. State the next layer clearly if this stage remains partial.

Decision rule:
- `PASS` only if aggregate rescue meets or beats `9083357` and activation no
  longer collapses in dangerous-agreement states
- `PARTIAL` if activation improves but aggregate rescue still misses baseline
- `FAIL` if the new controller does not materially improve the rescue contract

## Dependency Order

```text
Task 1 ──> Task 2 ──> Task 3 ──> Task 4 ──> Task 5 ──> Task 6 ──> Task 7 ──> Task 8 ──> Task 9
```

## Critical Path

The critical path is:

1. freeze the baseline
2. define the bounded fact migration
3. implement activation
4. implement emergency selection
5. upgrade telemetry
6. run bounded compare
7. make the stage decision

## Risks

- bounded textbook migration may uncover more controller-path literals than
  expected
- emergency activation may improve coverage but still miss the aggregate rescue
  baseline
- the new selector may become too entangled with normal-path ranking if
  boundaries are not kept explicit
- telemetry may still be too weak to explain a partial result if activation
  features are not logged cleanly
- the separate GPU online hang may still constrain how quickly rescue-side
  verification can iterate
