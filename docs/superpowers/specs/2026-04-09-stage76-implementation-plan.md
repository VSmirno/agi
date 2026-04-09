# Stage 76: Implementation Plan

**Spec:** `2026-04-09-stage76-continuous-model-learning-design.md` (v2.1)
**Status:** Draft plan, ready for execution
**Created:** 2026-04-09

## Project Summary

**Goal:** Agent survives ≥200 steps avg in Crafter with enemies via memory-based
continuous learning (replaces Stage 75 linear-plan execution).

**Constraints:**
- No backprop in main loop (ideology)
- No hardcoded thresholds / derived features (ideology)
- Single developer, GPU optional (tracker + SDM are CPU)
- Must reuse Stage 75 segmenter checkpoint (no retraining)

**Definition of done:**
- All gates from spec pass (survival ≥200 over 3 runs, tile_acc ≥80%,
  wood ≥50%, automated no-hardcode lint passes)
- Tests ≥90% coverage on new modules
- Documentation: Stage 76 report written

---

## Milestones

| # | Milestone | Phase | Success Criteria |
|---|-----------|-------|------------------|
| M1 | Foundation ready | 1 | Tracker extended, helpers tested, fixed SDR seeding works |
| M2 | StateEncoder passes tests | 2 | Encoder produces deterministic, similarity-preserving SDRs |
| M3 | EpisodicSDM functional | 3 | SDM writes/recalls/scores/selects actions correctly |
| M4 | End-to-end smoke | 4 | Agent runs 1 episode, SDM grows, no crashes |
| M5 | Warmup complete | 5 | 100 warmup episodes run, SDM size stable |
| M6 | Evaluation PASS | 6 | All spec gates pass on 3 runs × 20 eps |

---

## Phase 1: Foundation

Prepare the building blocks: tracker extensions, bucket encoding utilities,
fixed SDR pattern generation. Nothing that requires other new components.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 1.1 Extend `HomeostaticTracker` | `observed_max[var]`, `observed_variables()` methods in `perception.py` | Tests: max updates monotonically from observations; observed_variables returns dynamically-grown set; no hardcoded list | — |
| 1.2 Create `memory/` package | `src/snks/memory/__init__.py` empty package | Import works | — |
| 1.3 Bucket encoding helper | `bucket_encode(name, value, range, window, start, end)` in `sdr_encoder.py` | Tests: consecutive values share ~80% bits; distant values share 0 bits; deterministic output | — |
| 1.4 Fixed SDR registry | Seeded-random 40-bit SDR patterns per concept id, lazy allocation | Tests: same concept id → same bits; different ids → negligible overlap; unknown concept auto-allocates | — |
| 1.5 Spatial range allocator | Compute bit ranges for concept×distance bindings, 100 bits per concept | Tests: each concept gets distinct range; ranges don't overlap; allocation persistent across calls | 1.4 |
| 1.6 Unit tests 1.1-1.5 | `tests/test_stage76_foundation.py` | All tests pass | All above |

**Risks Phase 1:** Low. Standalone utilities, easy to test.

---

## Phase 2: StateEncoder

Build the raw-state → SDR projection. This is the main new abstraction
layer. Must be deterministic and similarity-preserving.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 2.1 `StateEncoder` class skeleton | `src/snks/memory/sdr_encoder.py` with `encode(inv, vf, spatial_map, player_pos) → np.ndarray[bool]` | Class instantiates, encode returns correct shape (4096 bool) | 1.3, 1.4, 1.5 |
| 2.2 Body stats encoding | Bucket-encoded HP, food, drink, energy (100 bits each, window 40) | HP=5 and HP=6 share ≥80% bits in body range; HP=1 and HP=9 share 0% | 2.1 |
| 2.3 Inventory encoding | Scalar counts (wood, stone, coal, iron) bucket-encoded; binary items (sword, pickaxe, table) via fixed SDR | wood=2 close to wood=3 in bits; sword=0 vs sword=1 distinguishable | 2.1 |
| 2.4 Visible concepts encoding | Spatial range allocation per concept × distance | `see_zombie@dist=2` and `see_zombie@dist=3` share ≥75% bits within zombie range; different concepts share 0% | 2.1, 1.5 |
| 2.5 Spatial map encoding | Known-class nearest distance via spatial range allocation (30-bit range per class) | `know_tree@dist=5` distinguishable from `know_water@dist=5`; close distances cluster | 2.1, 1.5 |
| 2.6 Similarity property test | Test: encode(state1) vs encode(state2) has similarity correlated with "semantic distance" | Manually-constructed similar states yield ≥60% overlap; distinct states ≤20% | 2.2-2.5 |
| 2.7 Density check | Target ~200 active bits out of 4096 | Mean active bits in 50 random states is 150-250 | 2.2-2.5 |
| 2.8 Unit tests 2.1-2.7 | `tests/test_stage76_sdr.py` | All tests pass | All above |

**Risks Phase 2:**
- Similarity tuning may need iteration (window sizes, range sizes)
- Density may be too high/low → adjust bit ranges
- Mitigation: start with 50 random states, measure, tune

---

## Phase 3: EpisodicSDM

Build the episodic memory system: buffer, recall, action scoring, selection.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 3.1 `Episode` dataclass | In `episodic_sdm.py` with state_sdr, action, next_state_sdr, body_delta, step | Creates cleanly; serializable | — |
| 3.2 `EpisodicSDM` class + FIFO buffer | Circular buffer size 10K, `write(episode)`, `len()` | Buffer grows, wraps at max, size monotonic | 3.1 |
| 3.3 `recall(query_sdr, top_k)` | Linear scan with popcount overlap similarity | Returns top_k episodes sorted by similarity; empty buffer returns [] | 3.2 |
| 3.4 `score_actions(recalled, body, tracker)` | Deficit-weighted aggregation using tracker.observed_max | Mean action score computed per action; deficit × delta emergent sign; tests with synthetic episodes | 3.3, 1.1 |
| 3.5 `select_action(scores, temperature)` | Softmax selection with numerical stability | Higher score = higher probability; temperature=∞ → uniform; temperature→0 → argmax; handles empty dict | 3.4 |
| 3.6 Bootstrap similarity threshold | "≥5 similar episodes" where similar = popcount overlap ≥ 50% of query popcount | Returns count of sufficiently-similar recalled episodes | 3.3 |
| 3.7 Performance check | Benchmark: 10K buffer query latency | Recall < 50ms; if slower, document in Open Questions | 3.3 |
| 3.8 Unit tests 3.1-3.7 | `tests/test_stage76_sdm.py` | All tests pass | All above |

**Risks Phase 3:**
- Linear scan too slow → benchmark determines. Fallback: reduce buffer or add hash bucketing.
- Score outlier sensitivity → may need median variant (Risk #7 in spec).

---

## Phase 4: Continuous Agent

Integrate all components into a single decision loop. This replaces Stage 75
phase6_survival policy.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 4.1 `continuous_agent.py` skeleton | `run_continuous_episode(env, segmenter, encoder, sdm, store, tracker, rng, max_steps, temperature)` function | Function signature matches spec pseudocode; imports resolve | Phase 1, 2, 3 |
| 4.2 Perception + spatial_map update | Reuse Stage 75 `_perceive_segmenter` pattern, update spatial_map correctly | Tree detected → spatial_map gets entry | 4.1 |
| 4.3 Main decision branch | If len(recalled) ≥ bootstrap_k → SDM action; else ConceptStore plan fallback | Both branches tested with synthetic SDM states | 4.1, 3.6 |
| 4.4 Bootstrap fallback integration | Call `select_goal` + plan execution similar to Stage 75 phase6 but only as fallback | Empty SDM → uses ConceptStore path; plan action composed correctly for make/place | 4.3 |
| 4.5 Episode write integration | After each env.step, write (state, action, next_state, body_delta) to SDM | SDM grows monotonically during episode; body_delta uses tracker.observed_variables() | 4.3 |
| 4.6 ConceptStore confidence update | Keep existing `outcome_to_verify` + `verify_outcome` path | Confidence updates in bootstrap phase | 4.5 |
| 4.7 Episode-level metrics | Return dict: length, sdm_size, final_inv, bootstrap_ratio, action_entropy | All metrics computed correctly | 4.5 |
| 4.8 Smoke test 1 episode | Single episode on known seed completes without crash | Runs to done or max_steps; SDM has entries; no exceptions | 4.7 |
| 4.9 Unit tests 4.1-4.8 | `tests/test_stage76_agent.py` | All tests pass | All above |

**Risks Phase 4:**
- Bootstrap→SDM transition may misfire (too early / too late). Log extensively.
- make/place action name composition — Stage 75 had bugs here. Reuse same logic.

---

## Phase 5: Experiment Pipeline

Build `exp136` to run warmup + evaluation phases.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 5.1 `exp136_continuous_learning.py` skeleton | Phase 0: load segmenter; Phase 1: warmup no-enemies; Phase 2: warmup with-enemies; Phase 3: eval with-enemies; Phase 4: summary | Phases callable, segmenter loads | Phase 4 |
| 5.2 Phase 1: Warmup no-enemies | 50 episodes, max_steps=500, agent learns basic navigation | SDM accumulates ≥5K episodes; wood collection works (smoke-test-equivalent) | 5.1 |
| 5.3 Phase 2: Warmup with-enemies | 50 episodes with enemies enabled, temperature still high | SDM has diverse experiences (deaths recorded); temperature starts decaying | 5.2 |
| 5.4 Phase 3: Evaluation | 3 runs × 20 episodes, max_steps=1000, temperature=0.3 | Metrics logged per episode: length, cause of death, SDM size, bootstrap ratio | 5.3 |
| 5.5 Gate checks | Tile acc re-run, wood collection smoke, Gate 5 linter test | All gates produce pass/fail output | 5.4 |
| 5.6 Summary reporting | Print table: survival per run, overall avg, variance, gate status | Output matches Stage 75 report style | 5.5 |

**Risks Phase 5:**
- Warmup may be insufficient → adjust count/difficulty
- 3 × 20 × 1000 steps = 60K total. On CPU ~ several hours. Plan for overnight run on minipc.

---

## Phase 6: Evaluation + Iteration

Run the full pipeline on minipc, analyze results, iterate if gates fail.

| Task | Deliverable | Done Criteria | Dependencies |
|------|------------|---------------|--------------|
| 6.1 Deploy to minipc | git push + pull, tmux session via `scripts/minipc-run.sh` | exp136 runs on minipc without setup errors | Phase 5 |
| 6.2 First eval run | Full exp136 pipeline | Completes without crash, metrics available | 6.1 |
| 6.3 Analysis: gate status | Check Gate 1-6 results | Pass/fail per gate documented | 6.2 |
| 6.4 IF gates pass → Stage 76 report | `docs/reports/stage-76-report.md` with metrics, architecture, comparison to Stage 75 | Report written and committed | 6.3 |
| 6.5 IF gates fail → diagnose | Identify which component fails: SDM recall noisy? Bootstrap transition wrong? Exploration collapsed? | Concrete hypothesis formed, max 1 round of fixes before re-evaluation | 6.3 |
| 6.6 IF still fails after 1 fix round | Invoke Stage 76 v2 attention mechanism OR escalate to Stage 77 forward sim | Decision made with user | 6.5 |

**Risks Phase 6:**
- Gates may not pass on first run. Plan for 1 targeted fix + re-eval.
- After 2 failed rounds → architectural re-evaluation required (per systematic debug rule).

---

## Dependencies Map

```
Phase 1 (Foundation)
    │
    ├──> Phase 2 (StateEncoder)
    │       │
    │       └──> Phase 3 (EpisodicSDM)
    │               │
    │               └──> Phase 4 (Continuous Agent)
    │                       │
    │                       └──> Phase 5 (Experiment Pipeline)
    │                               │
    │                               └──> Phase 6 (Evaluation)
```

**Critical path:** 1 → 2 → 3 → 4 → 5 → 6 (sequential, no parallelism).

**Parallelizable tasks within phase:**
- Phase 1: tasks 1.1, 1.3, 1.4 (independent helpers)
- Phase 2: tasks 2.2, 2.3, 2.4, 2.5 (independent encoding strategies)
- Phase 3: tasks 3.1, 3.2 then 3.3-3.5 sequential

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| SDR similarity tuning iterative | Medium | Start with generous windows, measure on synthetic states, iterate |
| SDM linear scan too slow | Medium | Benchmark at Phase 3.7, fallback: smaller buffer or LSH |
| Bootstrap transition misfires | High | Extensive logging of bootstrap/SDM ratio; parameter tuning in Phase 6 |
| Exploration collapse | High | Monitor action entropy each episode; if drops, inject temperature |
| Survival < 200 after first eval | High | Plan for 1 targeted fix round. After 2 rounds → question architecture. |
| SDM buffer catastrophic forgetting | Medium | Monitor buffer wrap; reservoir sampling as backup |
| Hidden hardcode creep | Medium | Gate 5 linter runs every commit; code review all new files |
| minipc downtime | Low | Cached state allows local dev; checkpoints persist |

---

## Scope Control — NOT in this Stage

Explicitly excluded (Stage 77+ or never):
- AttentionWeights v2 (only if v1 gates fail with diagnostic pointing at noise)
- Multi-step forward simulation (Stage 77 if v1 fails)
- Neural network components
- Replay buffer prioritization beyond FIFO
- Transfer to non-Crafter environments (Stage 78+)
- ConceptStore consolidation from SDM (Stage 78+)

---

## Success Metrics (from spec)

Gates (must pass):
1. Survival mean ≥ 200 over 3 runs × 20 episodes, each run individually ≥ 200
2. tile_acc ≥ 80%
3. Wood collection ≥ 50% of smoke episodes reach 3 wood
4. SDM growth monotonic, no empty states after wrap
5. Automated no-hardcode lint passes on all new files
6. Unit tests pass

Should demonstrate:
1. Bootstrap transition: bootstrap ratio decreases over episodes
2. Generalization: novel states recall helpful memories
3. Improvement trend: survival trend positive over warmup + eval
4. Action entropy stable above collapse threshold

---

## Execution Order Reminder

Per systematic debugging and brainstorming skills:
1. Implement Phase 1-4 (foundation + components + agent)
2. Run local smoke test (Phase 4.8)
3. Deploy to minipc (Phase 6.1)
4. Run full evaluation (Phase 6.2)
5. Analyze (Phase 6.3)
6. If fails: ONE targeted fix. If still fails: architectural review, not another patch.

**No thrashing. No multiple speculative fixes at once.** The Stage 75 trap
was 11 patches in one day. Stage 76 commits to systematic iteration.
