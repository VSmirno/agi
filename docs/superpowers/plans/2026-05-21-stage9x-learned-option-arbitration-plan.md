# Stage9X Learned Option Arbitration — Implementation Plan

**Date:** 2026-05-21
**Design:** `docs/superpowers/specs/2026-05-21-stage9x-learned-option-arbitration-design.md`
**Branch:** `main` at `586a2f3`

---

## Goal

Teach the agent to choose between existing strategy options in compound
contexts by persisting option-level outcomes in the existing `VectorWorldModel`
SDM. Do not add a hand-written crisis policy.

Done means:

- each step traces compact conflict context and selected strategy option;
- option outcomes are written to and loaded from the same world-model SDM;
- candidate options can read learned option outcomes during scoring;
- seed17 gen2 shows at least one conflict-context option choice changed by
  persisted recall, with trace evidence;
- no new fixed-count or named-environment arbitration rule is introduced.

---

## Constraints

- One HDC substrate only: extend `VectorWorldModel.memory`, no parallel SDM.
- No named Crafter policy branches in arbitration.
- First behaviour-changing patch happens only after trace-only instrumentation
  is test-covered.
- Every Stage9X recording must be delivered with the mp4 in the report.
- HyperPC validation must use git checkout of an exact commit.

---

## Milestones

| # | Milestone | Success Criteria |
|---|---|---|
| 1 | Trace-only option/context instrumentation | Local trace exposes selected option and context buckets; no behaviour change |
| 2 | Option outcome SDM role | Unit tests prove write/read/save/load and role separation |
| 3 | Recorder lifecycle | Selected option outcomes are written after horizon flush |
| 4 | Stimulus read path | Candidate options show learned outcome/confidence in trace |
| 5 | Seed17 gen1/gen2 proof | Gen2 option choice changes in a conflict context due to persisted recall |

---

## Phase 1 — Trace-Only StrategyOption and Context (4-6h)

| Task | Effort | Done Criteria |
|---|---:|---|
| Add `StrategyOption` dataclass/helper | 1h | Maps current plan origins/actions to stable option ids |
| Add compact context encoder | 1.5h | Buckets vitals, threat pressure, local restore, intent, goal family |
| Wire selected option/context into `local_trace` | 1h | Trace includes `strategy_option` and decoded `option_context` |
| Unit tests for option derivation/context | 1.5h | Tests cover baseline, frontier, local survival, combat continuation |

Notes:

- This phase must not change selected actions.
- Keep helper code small; likely home is `vector_mpc_agent.py` first, extract
  later only if it grows.

---

## Phase 2 — Option Outcome Role in VectorWorldModel (4-6h)

| Task | Effort | Done Criteria |
|---|---:|---|
| Add option context/option key encoding | 1.5h | Deterministic HDC keys for context+option |
| Add `learn_option_outcome` / `predict_option_outcome` | 2h | Roundtrip unit test passes |
| Extend save/load coverage | 1h | Saved model preserves option outcome memory |
| Orthogonality tests | 1h | Option outcome writes do not affect physics/outcome-role reads |

Notes:

- Follow existing `learn_outcome` / `predict_outcome` pattern.
- Avoid bundled high-dimensional context that collapses all options into the
  same recall. Test two options in same context and same option in two contexts.

---

## Phase 3 — Lifecycle Writer (3-5h)

| Task | Effort | Done Criteria |
|---|---:|---|
| Add `_OptionOutcomeRecorder` | 1.5h | Pending ring writes after horizon `H` and flushes on death |
| Integrate into `run_vector_mpc_episode` behind flag | 1.5h | `enable_option_outcome_learning` writes selected options only |
| Unit/integration tests | 1.5h | Recorder writes alive, damage, death, interrupted/status fields |

Notes:

- Reuse `_OutcomeRecorder` structure where possible.
- Write selected option outcome even if primitive was emergency-overridden, but
  include `interrupted_by` so read-side can distinguish it.

---

## Phase 4 — Stimulus Reader and Trace (4-7h)

| Task | Effort | Done Criteria |
|---|---:|---|
| Add `OptionOutcomeStimulus` | 2h | Converts predicted option outcome into conservative score signal |
| Annotate/scored candidates with option recall | 1.5h | Trace shows per-option confidence/outcome for candidates |
| Add read path flag/weight | 1h | Disabled by default unless flag set |
| Tests for differentiation | 1.5h | Same context, two options, stored bad outcome changes ranking contribution |

Notes:

- Start with strong negative recall and small/zero positive bonus.
- The stimulus must not become "never fight after one bad sample"; require
  confidence threshold or small weight for early runs.

---

## Phase 5 — Local Smoke (1-2h)

| Task | Effort | Done Criteria |
|---|---:|---|
| Run focused pytest | 30m | New tests pass plus nearby outcome/vector MPC tests |
| Run minimal local smoke | 30m | Import/signature paths work with option outcome enabled |
| Inspect trace fields | 30m | `strategy_option`, `option_context`, read/write events visible |

Nearby tests:

```bash
PYTHONPATH=src:experiments pytest \
  tests/test_vector_mpc.py \
  tests/agent/test_outcome_stimulus.py \
  tests/agent/test_outcome_recorder.py \
  tests/agent/test_world_model_outcome_role.py -q
```

---

## Phase 6 — Seed17 Gen1/Gen2 Recording (HyperPC, 30-60m wall)

| Task | Effort | Done Criteria |
|---|---:|---|
| Commit and push exact implementation commit | 10m | HyperPC checks out the commit by hash |
| Record gen1 seed17 ep0 full-profile | 5-10m | mp4/json copied to `output_to_user` |
| Record gen2 with saved world model | 5-10m | Same flags, preloaded option outcome memory |
| Analyze conflict-context choices | 1h | Report identifies changed option choice caused by recall |
| Deliver video artifact | 5m | Report includes `<file:/...mp4>` |

Pass criteria:

- `option_recall_confidence > 0` for at least one conflict candidate in gen2;
- at least one selected option differs from gen1 in a comparable context;
- trace shows recall changed candidate ranking or score contribution;
- no collapse into always-flee / never-fight / baseline-only.

---

## Phase 7 — Broader Check (optional after seed17 proof)

| Task | Effort | Done Criteria |
|---|---:|---|
| Run 5-seed gen1/gen2 | ~30m wall | Summarize survival, option changes, death causes |
| Determinism check | ~10m wall | Same preloaded model gives byte-identical traces |
| Update docs | 30m | Architecture report records PASS/PARTIAL/FAIL honestly |

---

## Dependency Map

```text
Phase 1 trace-only
  -> Phase 2 SDM role
      -> Phase 3 writer
          -> Phase 4 reader/stimulus
              -> Phase 5 local smoke
                  -> Phase 6 seed17 proof
                      -> Phase 7 multiseed/determinism
```

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Uniform recall across options | High | Test same-context different-option recall before integration |
| Hidden tactical policy | High | Reject named environment branches in option/context logic |
| Over-penalizing useful risky options | Medium | Conservative weight, confidence threshold, trace ranking deltas |
| Trace too sparse to prove causality | High | Phase 1 trace-only first; do not change behaviour until visible |
| HyperPC roundtrip wasted on import bug | Medium | Local smoke with option feature enabled before remote run |
| Existing untracked artifacts accidentally committed | Low | Stage only explicit files |

---

## First Implementation Slice

Start with Phase 1 only:

1. Add `StrategyOption` derivation helper.
2. Add option context encoder.
3. Add selected option/context to `local_trace`.
4. Add focused tests.

This gives immediate visibility and creates a stable contract for the learned
role without changing behaviour.

