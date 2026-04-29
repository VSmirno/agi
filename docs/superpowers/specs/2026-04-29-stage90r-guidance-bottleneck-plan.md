## Project: Stage90R Shared Guidance Bottleneck

**Goal**: Replace the evaluator-only prior slice with a shared semantic state
encoder, a first-class guidance bottleneck, and blended state-level advisory
actor supervision.

**Owner**: Codex

**Constraints**:
- No hard-coded primitive action rules
- No gate-threshold tuning in this pass
- Keep the current dataset contract usable
- Stay narrow enough to validate on the existing `seed 7` path

---

## Milestones

| # | Milestone | Success Criteria |
|---|-----------|------------------|
| 1 | Redesign spec written | Shared `z_state` + guidance bottleneck is documented |
| 2 | Data contract updated | Guidance targets and advisory teacher targets are built from current dataset rows |
| 3 | Model refactor complete | Actor and evaluator both consume the shared guidance state |
| 4 | Focused verification complete | Narrow tests pass and seed-7 verification artifacts are produced |

---

## Phase 1: State Guidance Targets

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Define target formulas for `threat_urgency` | S | - | Formula uses state signature / belief signature only |
| Define target formulas for `opportunity_availability` | S | - | Formula uses resource geometry / affordance state only |
| Define target formulas for `vitality_pressure` | S | - | Formula uses body buckets / low-vitals state only |
| Define target formulas for `progress_viability` | S | - | Formula uses belief progress/stall semantics only |
| Expose all four targets in batch collation | S | Formula choice | Batches contain state-level guidance tensors |

---

## Phase 2: Advisory Teacher Targets

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Preserve state signature + threat context on teacher records | XS | - | Teacher records can compute state guidance locally |
| Build ranking winner distributions per state signature | S | Train state samples | Outcome-side advisory target exists by state |
| Blend planner soft targets with advisory distributions | S | Ranking winner distributions | Actor teacher targets are no longer raw planner-only imitation |
| Keep consistency weighting on ambiguous planner signatures | XS | Blend logic | Conflicted planner signatures remain downweighted |

---

## Phase 3: Shared Bottleneck Model Refactor

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Replace split actor/evaluator trunks with shared `z_state` encoder | M | Phase 1 | One shared semantic state path exists |
| Add guidance heads and guidance vector | S | Shared encoder | Four guidance predictions are emitted from `z_state` |
| Condition actor on `z_state + guidance` | S | Guidance vector | Actor no longer reads a separate independent trunk |
| Condition evaluator on `z_state + action + guidance` | S | Guidance vector | Outcome heads use guidance-shaped action branch |

---

## Phase 4: Training And Diagnostics

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Add guidance losses to evaluator training | S | Phase 3 | Guidance losses train numerically |
| Keep actor training on blended advisory targets | S | Phase 2, Phase 3 | Teacher epoch uses the new targets |
| Extend focused tests for guidance tensors and forward outputs | S | Phase 3 | Tests cover the new contract |
| Run narrow local tests | XS | Test updates | Focused tests pass |

---

## Phase 5: Narrow Verification

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Run `epochs=1` smoke on seed 7 | S | Phase 4 | Early actor/ranking behavior captured |
| Run `epochs=3` canonical on seed 7 | S | Smoke | Endpoint behavior captured |
| Summarize deltas vs soft-teacher and failed-prior baselines | XS | Canonical | One verification note states what improved or failed |

---

## Decision Rules

| Outcome | Next Move |
|---------|-----------|
| Actor/ranking drift weakens and endpoint improves | Continue guidance-bottleneck path |
| Early behavior improves but endpoint still collapses | Rework training schedule / checkpoint criteria next |
| No meaningful gain | Escalate to stronger planner decomposition redesign, not more auxiliary heads |

---

## Critical Path

`guidance target formulas` -> `advisory teacher blending` -> `shared z_state refactor` -> `guidance-conditioned actor/evaluator` -> `focused tests` -> `seed 7 verify`
