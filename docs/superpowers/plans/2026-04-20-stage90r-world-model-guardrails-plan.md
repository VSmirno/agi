# Stage 90R World-Model Guardrails Implementation Plan

**Date:** 2026-04-20  
**Spec:** `docs/superpowers/specs/2026-04-20-stage90r-world-model-guardrails-design.md`  
**Status:** Draft  
**Goal:** Execute the next `Stage 90R` iterations in a way that strengthens
local world understanding and avoids tactical local-policy drift.

## Goal

Reshape the current `Stage 90R` pipeline so that:

- local data supports **state-conditioned action comparison**
- training measures **predictive and ranking quality**
- online evaluation distinguishes **world-model gain** from **policy shortcut**
- `local-only` mode remains a falsification tool, not the target architecture

## Non-Goals

This plan does not attempt to:

- replace the planner with a learned local controller
- prove inter-generation transfer
- optimize aggregate Crafter survival by any means necessary
- treat `local-only argmax` as the canonical architecture

## Milestones

| # | Milestone | Target Outcome |
|---|-----------|----------------|
| 1 | Guardrail Baseline Frozen | Current collector/train/eval artifacts and failure modes are fixed as reference |
| 2 | Dataset Reframed | Local dataset supports action comparison rather than chosen-action-only outcomes |
| 3 | Offline Gates Added | Ranking and anti-collapse metrics exist and gate online runs |
| 4 | Predictor Integration Hardened | Local model feeds decision layers without becoming the architecture |
| 5 | Reviewable Stage Result | Report can state PASS/PARTIAL/FAIL against guardrail spec |

## Phase 1 — Freeze Current Failure Mode

### Task 1.1 — Freeze current baseline artifacts

**Files:**
- `_docs/stage90r_local_dataset*.json`
- `_docs/stage90r_local_evaluator_eval.json`
- `_docs/stage90r_local_only_eval.json`
- baseline reference helper / report files

**Work:**
- Capture current smoke and full dataset artifacts as explicit baseline.
- Record the current collapse signature:
  - chosen-action-only dataset
  - good-looking offline survival metric
  - degenerate online action behavior

**Done criteria:**
- one baseline block is reusable in later reports
- collapse is written down as a known failure mode, not an anecdote

### Task 1.2 — Measure current action-bias regime

**Files:**
- `experiments/stage90r_collect_local_dataset.py`
- new or extended analysis helper under `experiments/`

**Work:**
- Report action distribution overall and by regime:
  - threat visible
  - resource opportunity visible
  - neutral wandering
  - low-vitals
- Report how often `escape_delta_h` is valid vs masked.

**Done criteria:**
- current data imbalance is quantified
- dataset reports already show whether collapse risk is predictable offline

## Phase 2 — Reframe the Dataset Around Action Comparison

### Task 2.1 — Introduce state-centered sample format

**Files:**
- `src/snks/agent/stage90r_local_policy.py`
- `experiments/stage90r_collect_local_dataset.py`

**Work:**
- Replace or extend flat chosen-action records with a state-centered structure:
  - local state snapshot
  - candidate primitive action set
  - observed or estimated short-horizon outcomes per candidate
- Keep the current chosen-action stream only as auxiliary data, not the primary
  training target.

**Done criteria:**
- one local state can be inspected together with multiple candidate outcomes
- online ranking use matches dataset semantics

### Task 2.2 — Add controlled counterfactual supervision

**Files:**
- `experiments/stage90r_collect_local_dataset.py`
- optional helper under `src/snks/agent/`

**Work:**
- For a bounded primitive set (`move_left/right/up/down/do/sleep`), collect
  short-horizon outcomes beyond the action chosen by the old policy.
- Prefer explicit short rollouts or similarly inspectable approximations.

**Guardrail:**
- no hidden reward shaping
- no Crafter-specific reflex labels

**Done criteria:**
- dataset meaningfully supports in-state action comparison
- counterfactual coverage exists for at least the core primitive set

### Task 2.3 — Make regime coverage explicit

**Files:**
- dataset summary/report helpers

**Work:**
- Tag each sample or state with local regime labels:
  - hostile-contact / hostile-near
  - local-resource-facing
  - neutral
  - low-vitals
- Publish counts per regime.

**Done criteria:**
- later training and eval can be sliced by local situation, not only aggregate

## Phase 3 — Redesign Offline Training Gates

### Task 3.1 — Preserve predictive heads, add ranking evaluation

**Files:**
- `src/snks/agent/stage90r_local_model.py`
- `experiments/stage90r_train_local_evaluator.py`

**Work:**
- Keep predictive heads for:
  - damage
  - resource gain
  - survival
  - escape delta
- Add offline action-ranking metrics such as:
  - pairwise preference accuracy
  - top-1 agreement against counterfactual short rollouts
  - per-regime ranking accuracy

**Done criteria:**
- checkpoint quality is judged by ranking quality, not survival accuracy alone

### Task 3.2 — Add anti-collapse offline gates

**Files:**
- `experiments/stage90r_train_local_evaluator.py`
- optional new analysis script under `experiments/`

**Work:**
- Add explicit diagnostics before online canary:
  - action entropy
  - dominant-action share
  - threat-slice action diversity
  - resource-slice action diversity

**Decision rule:**
- If anti-collapse gate fails, do not treat online run as evidence of
  architectural success.

**Done criteria:**
- “looks trainable offline” is no longer enough to justify canary optimism

### Task 3.3 — Reweight metrics toward causal slices

**Files:**
- offline training/eval scripts

**Work:**
- Give higher interpretive importance to slices where action consequence matters:
  - visible threat
  - visible affordance
  - escape/no-escape choice points
- Keep aggregate metrics, but subordinate them to causal slices.

**Done criteria:**
- report can explain whether the model learned real local distinctions

## Phase 4 — Constrain Online Integration

### Task 4.1 — Keep `local-only` as a canary only

**Files:**
- `experiments/stage90r_eval_local_policy.py`
- report/spec references

**Work:**
- Explicitly document `local-only` mode as diagnostic only.
- Prevent it from being interpreted as the intended end-state architecture.

**Done criteria:**
- all docs and logs treat `local-only` as a stress test / falsification tool

### Task 4.2 — Add planner-facing advisory mode

**Files:**
- planner entry point(s)
- optional adapter under `src/snks/agent/`

**Work:**
- Introduce a controlled mode where the local predictor informs planner choice
  without replacing planner/stimuli.
- Keep logs showing:
  - predictor outputs
  - planner choice before/after advisory signal

**Done criteria:**
- local predictor improves decision context while decision layers remain explicit

### Task 4.3 — Require inspectable explanations in eval logs

**Files:**
- evaluation scripts / logging helpers

**Work:**
- Log top candidate actions with predicted local consequences.
- Make it possible to read a failure and say:
  - which action was preferred
  - what damage / gain / escape forecast drove that preference

**Done criteria:**
- online runs are readable as local causal reasoning, not opaque scores

## Phase 5 — Review and Decision

### Task 5.1 — Run controlled comparisons

**Compare at least:**
- current Stage 89 / Stage 90 baseline
- current `local-only` canary baseline
- improved local predictor with offline gates
- planner-advisory mode using the improved local predictor

**Done criteria:**
- every compared mode has explicit config and artifact path

### Task 5.2 — Write guardrail-aware report

**Files:**
- `docs/reports/stage-90r-guardrails-report.md`
- `docs/ASSUMPTIONS.md`

**Report must answer:**
- what local world understanding improved
- how that improvement was measured offline
- how it affected online choice quality
- whether any observed gain was world-model gain or tactic drift

**Done criteria:**
- stage result can honestly be labeled `PASS`, `PARTIAL`, or `FAIL`

## Dependency Order

```text
Freeze failure mode
    ↓
Measure action bias
    ↓
Reframe dataset around action comparison
    ↓
Add ranking + anti-collapse offline gates
    ↓
Constrain online integration
    ↓
Run controlled comparisons
    ↓
Write guardrail-aware report
```

## Stop Conditions

Stop and reassess before continuing if any of the following happens:

- dataset still depends primarily on chosen-action-only supervision
- offline ranking remains weak even after wider data collection
- model still collapses onto a dominant action despite passing aggregate metrics
- any improvement appears only in `local-only` mode and not in planner-facing use
- changes become dominated by utility-weight tuning or ad hoc Crafter tactics

## Immediate Next Step

The most important next move is **not** to retune the current utility or make
the MLP larger.

The next move is:

**rebuild the dataset/eval semantics around state-conditioned action
comparison**, because this is the clearest architectural mismatch in the
current pipeline and the most likely root cause of tactical collapse.

