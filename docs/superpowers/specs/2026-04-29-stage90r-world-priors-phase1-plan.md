## Project: Stage90R World Priors Phase 1

**Goal**: Add two general world-prior auxiliary targets, `danger_pressure` and
`resource_opportunity`, to improve shared representation and reduce
actor/ranking drift without introducing direct action rules.

**Timeline**: 1 short implementation cycle, then narrow seed-7 verification

**Owner**: Codex

**Constraints**:
- No gate-threshold tuning
- No direct action heuristics
- Keep current soft signature-level actor supervision
- Reuse existing signals before adding new dataset machinery

---

## Milestones

| # | Milestone | Success Criteria |
|---|-----------|------------------|
| 1 | Prior targets defined | Exact formulas for `danger_pressure` and `resource_opportunity` are written down and map to existing labels/signals |
| 2 | Model path implemented | Two auxiliary heads are wired into the shared representation and training loop |
| 3 | Narrow verification complete | `seed 7` smoke/canonical rerun compares current baseline vs world-prior slice |
| 4 | Decision point | We can say whether this path reduced drift enough to justify Phase 2 |

---

## Phase 1: Target Definition

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Audit existing signals for `danger_pressure` | S | - | Candidate formula uses current regime / threat / belief signals only |
| Audit existing signals for `resource_opportunity` | S | - | Candidate formula uses current affordance/resource signals only |
| Choose simple scalar target formulas | S | Signal audits | Two formulas are frozen for Phase 1 |
| Record formulas in code comments or nearby docs | XS | Formula choice | Future implementation can proceed without re-deciding semantics |

---

## Phase 2: Minimal Model Integration

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Extend batch collation for new targets | S | Phase 1 | Training batches expose both targets cleanly |
| Add `danger_pressure` auxiliary head | S | Phase 1 | Model returns predictions for the new target |
| Add `resource_opportunity` auxiliary head | S | Phase 1 | Model returns predictions for the new target |
| Add losses with conservative weighting | S | New heads | Training loop logs both losses without disturbing existing paths |
| Preserve current actor/ranking supervision | XS | New heads | Existing split and soft teacher logic remain intact |

---

## Phase 3: Narrow Verification

| Task | Effort | Depends On | Done Criteria |
|------|--------|------------|---------------|
| Add focused regression tests for new targets | S | Phase 2 | New target plumbing is covered by small tests |
| Run local focused tests | XS | Regression tests | New tests pass with existing narrow tests |
| Run `epochs=1` smoke on `seed 7` | S | Phase 2 | Artifact captured, early actor/ranking behavior compared to baseline |
| Run `epochs=3` canonical on `seed 7` | S | Smoke | Canonical endpoint compared to baseline |
| Summarize delta vs current baseline | XS | Canonical | One note says what improved, what stayed the same, and why |

---

## Decision Rules

| Outcome | Next Move |
|---------|-----------|
| Early and late drift both improve | Continue world-prior path before adding more priors |
| Early behavior improves, endpoint still collapses | Investigate deeper training dynamics / objective coupling |
| No meaningful change | Re-evaluate target formulas or move to stronger representation redesign |

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Priors become disguised action labels | High | Keep targets scalar and descriptive, never direct best-action targets |
| Losses destabilize current training | Medium | Use conservative weights and compare to current seed-7 baseline |
| Existing labels are too weak | Medium | Keep Phase 1 simple; only escalate to richer target construction if partial gains appear |

---

## Critical Path

`target formulas` -> `model heads + losses` -> `focused tests` -> `seed 7 smoke` -> `seed 7 canonical` -> `decision`

