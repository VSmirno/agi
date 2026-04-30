## Project: Stage90R Actor-Selection Contract

**Goal**: Stabilize the post-`9083357` actor dynamics by replacing flat
primitive imitation with state-level candidate-aware actor supervision.

**Owner**: Codex

**Constraints**:
- Keep `9083357` promotion behavior intact unless a better path is verified
- Do not add hard-coded action rules
- Keep bootstrap/advisory modes usable
- Validate first on the existing tiny `mixed_control` slice, then expand

---

## Milestones

| # | Milestone | Success Criteria |
|---|-----------|------------------|
| 1 | Contract spec written | State-level candidate-aware actor training is documented |
| 2 | Target builder complete | Train states can emit candidate-normalized actor targets |
| 3 | Actor training refactor complete | Mixed-control actor no longer depends on flat teacher rows as its primary signal |
| 4 | Focused verification complete | Hyper multiseed summary shows reduced transient actor reflexes |

---

## Phase 1: Target Construction

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Reuse planner soft targets by signature | XS | Existing aggregation | Full-action soft target available per signature |
| Reuse advisory targets by signature | XS | Existing advisory builder | Full-action advisory target available per signature |
| Blend them into one actor-selection target | S | Both target sources | One full-action target per signature |
| Project target onto each state's candidate set | S | Blended target | Candidate-normalized target exists per state sample |
| Preserve support / consistency weight | XS | Existing teacher weighting | Candidate target carries usable training weight |

---

## Phase 2: Actor Training Refactor

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Add state-sample actor selection epoch | M | Phase 1 | Actor can train directly on candidate-aware state targets |
| Add candidate-conditioned actor selection score | M | Phase 1 | Hard-gated actor can be evaluated over the available candidate set |
| Branch training by gate mode | S | Actor epoch | Mixed-control/rescue use new contract; bootstrap modes keep simple path |
| Keep actor-ranker agreement pass | XS | Actor epoch | Agreement remains a secondary pass |
| Extend diagnostics/history | XS | Actor epoch | History records actor-selection metrics clearly |

---

## Phase 3: Tests

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Test candidate target projection | XS | Phase 1 | Projection over candidate actions is correct |
| Test renormalization and weighting | XS | Phase 1 | Sparse candidate sets remain numerically stable |
| Test branch selection by mode | XS | Phase 2 | Mixed-control path uses state-level actor contract |
| Run focused test file | XS | Test updates | `tests/test_stage90r_local_model.py` passes |

---

## Phase 4: Verification

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Local smoke probe on known fragile seeds | S | Phase 2 | Gross regressions caught early |
| Hyper multiseed verify on `7,11,17,23,31` | M | Smoke probes | New robustness summary produced |
| Compare against `9083357` | XS | Hyper summary | Delta is explicit in one note |
| Expand to neighboring hard-gated regime if improved | M | Positive mixed-control result | Adjacent coverage result exists |

---

## Decision Rules

| Outcome | Next Move |
|---------|-----------|
| Promotion stays robust and transient actor reflexes shrink | Keep the new actor-selection contract and expand coverage |
| Promotion stays robust but behavior is still equally fragile | Rework actor-evaluator objective coupling next, not schedule |
| Promotion regresses | Revert to `9083357` and redesign actor objective more deeply |

---

## Critical Path

`state-level blended actor target` -> `candidate projection` -> `mixed-control actor epoch refactor` -> `focused tests` -> `fragile-seed smoke probes` -> `hyper multiseed verify`
