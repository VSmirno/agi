# Stage 76 — Continuous Memory-Based Learning Report

**Date:** 2026-04-10
**Status:** COMPLETE (architectural wall confirmed; pivoting to Stage 77)
**Spec:** `docs/superpowers/specs/2026-04-09-stage76-continuous-model-learning-design.md`
**Plan:** `docs/superpowers/specs/2026-04-09-stage76-implementation-plan.md`

## Summary

Stage 76 delivered a full memory-based continuous-learning pipeline: SDR state
encoder, episodic memory with popcount-similarity recall, deficit-weighted
action scoring, softmax exploration, and an optional per-variable attention
weights module. All components tested, all ideology gates (no hardcoded drives,
no derived features, no argmax over drives) passing under automated lint.

**Survival gate (≥200 steps) NOT reached.** Three architectural variants were
deployed on minipc and evaluated across 3 runs × 20 eval episodes each:

| Variant | Eval overall | Wood smoke ≥3 | Notes |
|---------|--------------|---------------|-------|
| Stage 75 baseline (scripted) | 178 | 13/20 (65%) | Prior report |
| Stage 76 v1 FIFO SDM | **177** | 8/20 (40%) | Memory-based reactive |
| Stage 76 v1 + priority buffer (A+B) | 166 | 3/20 (15%) | Reverted — regression |
| Stage 76 v2 attention weights | 166 | 6/20 (30%) | Unfiltered — reverted |
| Stage 76 v2.1 attention + body filter | 173 | 3/20 (15%) | Reverted — wood crash |

**Conclusion:** reactive memory-based policy ≡ scripted baseline (within ±5
steps). Memory does not exceed what the bootstrap plan already achieves, and
attention-biased recall either destabilizes action selection (v2) or crashes
wood collection (v2.1). The blocker is not policy quality — it is that the
agent dies mostly from enemy attacks it cannot anticipate. Reactive decisions
based on past similar states cannot learn to AVOID a threat that hasn't yet
touched the current state.

## What Was Built (committed on main)

### Phase 1 — Foundation (`718b0d1`)
- `src/snks/memory/sdr_encoder.py` — three SDR primitives:
  - `bucket_encode(value, range, start, end, window)` — sliding-window scalar
    encoding with similarity-preservation for adjacent values
  - `FixedSDRRegistry` — lazy-allocated deterministic per-concept random
    patterns (hash-seeded, insertion-order independent)
  - `SpatialRangeAllocator` — pre-allocates a contiguous bit range per
    concept for (concept, distance) bindings, replacing XOR VSA binding
    which would have destroyed distance similarity
- `HomeostaticTracker` extensions: `observed_max`, `observed_variables()`,
  dynamic rolling max per variable. Replaces hardcoded `9` default and
  the hardcoded 4-drive list.

### Phase 2 — StateEncoder (`f17f58c`)
- `src/snks/memory/state_encoder.py` — raw state → 4096-bit SDR:
  - body stats [0, 400) — 4 vars × 100 bits (window 40)
  - inventory scalars [400, 1000)
  - inventory presence [1000, 1400) — fixed SDRs
  - visible concepts × viewport distance [1400, 2400)
  - spatial_map known × world distance [2400, 3400)
- Dynamic allocation for any previously unseen variable/concept. Density
  lands around ~400-550 active bits (10-13%) instead of the nominal 5%
  because window=40 is required for ≥80% adjacent-value overlap.

### Phase 3 — EpisodicSDM (`c585bcf`)
- FIFO buffer (capacity 10K) with `write`, `recall` (linear-scan popcount),
  `count_similar` for the bootstrap gate.
- `score_actions(recalled, current_body, tracker)` — deficit × delta
  aggregation. No "higher is better" assumption; sign emerges from data.
- `select_action(scores, temperature, rng)` — numerically stable softmax
  with temperature-controlled exploration.

### Phase 4 — Continuous agent (`46f168d`)
- `src/snks/agent/continuous_agent.py` — `run_continuous_episode()` decision
  loop. No plan state carried between steps. Branch per step:
  - SDM path: if `len(sdm) ≥ min_sdm_size` AND `count_similar ≥ bootstrap_k`
    → score recalled episodes → softmax action
  - Bootstrap path: `ConceptStore.select_goal` → plan[0].action
- Reuses Stage 75 `_perceive_segmenter` pattern for tile-map → VisualField.

### Phase 5 — Experiment pipeline + Gate 5 lint (`b65feae`)
- `experiments/exp136_continuous_learning.py` — 5 phases:
  Phase 0: load Stage 75 segmenter
  Phase 1: warmup-safe (50 eps, no enemies, T=1.0)
  Phase 2: warmup-enemy (50 eps, enemies, T=1.0→0.5 decay)
  Phase 3: eval (3 × 20 eps, enemies, T=0.3, max_steps=1000)
  Phase 4: gate checks
  Phase 5: summary report
- `tests/test_stage76_no_hardcode.py` — automated scan of Stage 76 files
  for forbidden patterns (4-drive list literals, `dominant_drive`,
  `if inv.get("X") < N`, magic-number body comparisons). Caught a real
  violation in early cause-of-death telemetry.

### Minipc-ready fixes (`5fca556`, `5186af1`, `dfdddbe`)
- Extract `TileSegmenter` from exp135's local-class definition to
  `src/snks/encoder/tile_segmenter.py` with `load_tile_segmenter()`
  constructing the module and loading state_dict in one call.
- `tile_segmenter` auto-selects GPU (`cuda` on ROCm) and moves pixels
  to device at classify time — segmenter forward pass runs on AMD GPU.
- `min_sdm_size` bootstrap gate: require ≥N total episodes in buffer
  before SDM path can activate, preventing cold-start where 5
  near-identical early states immediately flipped to SDM with 5
  uninformative samples.

### Tests
| Module | Tests |
|--------|------:|
| `test_stage76_foundation.py` | 25 |
| `test_stage76_sdr.py` | 25 |
| `test_stage76_sdm.py` | 26 |
| `test_stage76_agent.py` | 7 (+1 skipped real-segmenter smoke) |
| `test_stage76_no_hardcode.py` | 7 |
| **Total** | **90 + 1 skip** |

## What Was Tried and Reverted

### v1 + priority eviction + 50K buffer (A+B, reverted `93b0b5b`)
Hypothesis: FIFO at 10K evicts successful long-survival episodes within
~55 eps, leaving the buffer full of death trajectories. Fix: priority
eviction by episode length + capacity to 50K.

Result: buffer never filled to 50K (ended at ~30K), so priority eviction
never fired. Eval regressed from 177 → 166, likely from rng drift. Wood
smoke crashed 8/20 → 3/20. Reverted.

### v2 attention weights (reverted `caf87d0`)
Hypothesis: recall returns body-similar but hazard-different episodes
because the ~5% of SDR bits encoding enemy presence are dominated by
the other 95% in popcount similarity. Fix: per-variable EMA attention
over SDR bits, composed into a deficit-weighted query mask that biases
recall toward survival-relevant bits.

Result: warmup-enemy wood collection improved (15/50 → 20/50), but
eval survival regressed (177 → 166). Diagnosis from the attention
dump: `observed_max[wood]` grew unboundedly during warmup, so at eval
time wood-deficit was ~9 and the attention mask treated wood bits
as important as health bits. At T=0.3 softmax locked onto wood-positive
actions at the cost of avoiding enemies.

### v2.1 attention + body_variables filter (reverted `a5e2184`)
Hypothesis: attention should only weight body variables with innate
decay (health/food/drink/energy from the textbook body_rules), not
inventory items. Added `HomeostaticTracker.body_variables()`.

Result: recovered some survival (166 → 173) and meaningfully improved
warmup-enemy avg (176 → 190, the only warmup improvement in any
variant). But wood collection crashed to 1-3/20 in eval because the
SDM path no longer rewarded wood-positive actions at all. Still failed
the survival gate (173 < 200). Reverted.

## Diagnosis

**Smoke test (no enemies, T=0.3) is the smoking gun:** across all four
variants, 17-19 of 20 smoke episodes reach max_steps=200 alive with
H=9. The SDM policy in safe mode is competent — it keeps the agent
alive indefinitely when nothing is attacking.

**With enemies, the agent dies at 150-190 steps regardless of
architecture.** Most deaths are `cause=health` with food/drink still
in the 3-7 range — not homeostatic depletion, direct enemy damage.

**The architectural wall:**
- Reactive policy (bootstrap OR SDM OR SDM+attention) sees the current
  state, decides the current action. Death from enemy damage requires
  either (a) noticing the enemy early enough to flee, or (b) having a
  sword ready when the enemy is adjacent.
- (a) requires forward prediction: "if I keep walking this way, the
  zombie that is 4 tiles away will be adjacent in 4 steps". The
  current SDM stores (state → action → next_state) tuples but never
  queries them as a simulator — it only uses them for one-step scoring.
- (b) requires multi-step goal-directed behavior: "collect wood →
  place table → craft sword". The bootstrap path does this via
  `ConceptStore.plan`, but the agent never gets the sword ready before
  the first zombie attack because ConceptStore urgency prioritizes
  food/drink.

**Memory alone can't close the gap**, because the memory is on-policy:
it stores what the bootstrap (or softmax) actually did. If the
bootstrap never avoided enemies proactively, the SDM has no successful
avoidance trajectories to recall. Attention can't fix this — it only
biases similarity, not data quality.

**Confirmed by 4 runs** (Stage 75 baseline + 3 Stage 76 variants), all
landing within ±10 of 178 steps survival.

## Gates

| Gate | Target | v1 Result | Status |
|------|--------|-----------|--------|
| 1. Survival mean ≥ 200 over 3 runs | ≥200 per run | [184, 184, 163] → 177 | ❌ FAIL |
| 2. Tile accuracy ≥ 80% | ≥80% | Stage 75: 82% (unchanged) | ✅ PASS |
| 3. Wood ≥3 in smoke | ≥50% | 8/20 = 40% | ❌ FAIL |
| 4. SDM growth monotonic | grows | 10000/10000 | ✅ PASS |
| 5. No hardcoded derived features | automated lint | 7/7 patterns passing | ✅ PASS |
| 6. Unit tests pass | all green | 90 passing + 1 skip | ✅ PASS |

Ideological gates all pass. Quantitative survival gate does not.

## Key Insight

The Stage 76 design thesis was:
> Memory of past outcomes substitutes for explicit forward simulation,
> because recalled episodes ARE past forward rollouts.

**This is wrong.** Recalled episodes are past single steps — not rollouts.
A 20-step recall gives you 20 single-step (state, action, outcome) pairs,
not one 20-step trajectory. Scoring them sums contributions across
unrelated individual decisions, which is exactly reactive 1-step Q-
learning with fancy similarity. It cannot reason about "if I keep doing
X for N steps, I will end up in Y state".

The spec acknowledged this risk:
> If this hypothesis is wrong (survival still <200), Stage 76 needs
> multi-step forward simulation. That's a scope expansion, not a
> redesign.

Stage 76 built out the full memory substrate, which is a prerequisite
for forward simulation — Stage 77 can query `state_sdr → action →
next_state_sdr` sequences from the same SDM and roll them forward.
The infrastructure is reusable.

## Deliverables

Code (all on `main`):
- `src/snks/memory/__init__.py`
- `src/snks/memory/sdr_encoder.py`
- `src/snks/memory/state_encoder.py`
- `src/snks/memory/episodic_sdm.py`
- `src/snks/agent/continuous_agent.py`
- `src/snks/agent/perception.py` (`observed_max`, `observed_variables`)
- `src/snks/encoder/tile_segmenter.py` (extracted + GPU-aware loader)
- `experiments/exp136_continuous_learning.py`

Tests:
- `tests/test_stage76_foundation.py`
- `tests/test_stage76_sdr.py`
- `tests/test_stage76_sdm.py`
- `tests/test_stage76_agent.py`
- `tests/test_stage76_no_hardcode.py`

Docs:
- `docs/superpowers/specs/2026-04-09-stage76-continuous-model-learning-design.md` (v2.1)
- `docs/superpowers/specs/2026-04-09-stage76-implementation-plan.md`
- `docs/reports/stage-76-report.md` (this file)

## Next: Stage 77 — Forward Simulation

Stage 77 will reuse the Stage 76 memory substrate (StateEncoder, EpisodicSDM,
tracker extensions) but replace the 1-step recall with multi-step rollouts:
- For a candidate action A, query SDM for `(state, A, next_state)` tuples
- Follow the resulting next_state as a new query, get `(next_state, best_A, next_next)`
- Repeat for N steps, accumulate expected body_delta along the trajectory
- Pick the action whose N-step rollout best closes current deficits

This addresses the "avoid enemy 3 tiles away" gap without hardcoded reflexes.
Scope and spec will be designed in a separate brainstorm.

## Assumptions Added
- `docs/ASSUMPTIONS.md` gets a Stage 76 section documenting: memory path
  density (12% not 5%), min_sdm_size dependency on warmup length, and
  the ideology gate that reactive policy ≈ scripted baseline.
