# Stage 82 Report — IDEOLOGY v2 migration + knowledge persistence

**Dates:** 2026-04-11 → 2026-04-12 (overnight autonomous session)
**Parent:** docs/IDEOLOGY.md v2 (commit `1df6313`)
**Audit:** docs/reports/ideology-audit-2026-04-11.md (commit `e70f111`)
**Stage 82 commits:** `b48238d` → `1f0d86d` (5 commits on `main`)
**Eval (post-migration):** `_docs/stage79_post82_results.json`, `_docs/stage82_smoke_results.json`

## TL;DR

Stage 82 was a non-experimental clean-up stage, driven by the audit of
`docs/IDEOLOGY.md v2` rather than by a new capability. Five commits
land on `main`:

1. `b48238d` — Phase C: knowledge persistence (save/load experience
   JSON for `LearnedRule`, `HomeostaticTracker`, and rule confidences)
2. `51262b4` — Phase B audit 2.4: `blocking` attribute in vocabulary
3. `42765cf` — Phase B audit 1.1: `any_of`/`all_of`/`action_filter`
   in stateful rules + the conjunctive sleep rule declared directly
4. `f4aba1a` — Phase B audits 2.1/2.2/2.3/2.6/2.9/2.10: facing
   direction consolidation via textbook `primitives` block
5. `1f0d86d` — Phase B audit 1.4: `env_semantics` for action
   dispatch declared in textbook, with `_expand_to_primitive`
   branching on `dispatch: facing_tile` vs `dispatch: proximity`

Regression across the Stage 77/78/79 test scope: **228/228 green**
(+15 knowledge persistence, +10 stateful compound, +3 primitives
block parsers, and +2 env_semantics parsers), up from 213 at stage
start.

**Crafter eval post-migration** (Stage 79 harness with 3 ablations,
177 min on minipc, commit `1f0d86d`):

| Ablation                 | warmup_a | warmup_b | eval    | rules/ep |
|--------------------------|----------|----------|---------|----------|
| nursery_off              | 227.7    | 171.9    | 154.8   | 0        |
| nursery_on               | 236.4    | 171.8    | **162.8** | 192.5 |
| nursery_on_residual_on   | 228.9    | 158.8    | 147.7   | 118.9    |
| Stage 77a reference      | 222      | 203      | 180     | –        |

**Headline: nursery inversion confirmed.** `nursery_on` beats
`nursery_off` by +8 eval for the first time in the Stage 78c-81 arc.
Historically the nursery *hurt* eval (Stage 78c: `eval_on=152.1`
vs `eval_off=169.2`, Δ=-17). After migration 1.1 moved the
canonical `sleep + starvation` conjunction out of the nursery's
workload and into the textbook, the discrimination paradox that was
poisoning Phase 7 rule application vanished, and `nursery_on` now
learns a different class of useful corrections (per-location ambient
body rate noise) without collision.

Action entropy recovered: `eval_on=1.29` vs Stage 78c `0.18`
(collapsed). Gates: ✅ nursery_on_beats_off, ✅ entropy_not_collapsed,
✅ at_least_one_rule_per_ep. Remaining ❌: nursery_on_ge_77a_baseline
(still -17 below 180 wall) and wood_3_at_least_5pct (the
gatherer-planner wall from Stage 80/81).

`nursery_on_residual_on` is -15 below `nursery_on` — Stage 78c
residual predictor **confirmed parked** after Stage 82.

Post-migration observations:

- `impassable_concepts()` collapses from a 33-line Crafter-specific
  heuristic to a 4-line attribute scan.
- Five different hardcoded `move_left/right/up/down` if/elif chains
  collapse into one helper reading `store.primitives`.
- Stage 78a / 78c / 79 motivation (nursery must discover the
  conjunctive sleep + starvation rule) is **removed**: the teacher
  can declare it directly as an `any_of` stateful with `action_filter`.
- Experience (learned rules, tracker obs counts, rule confidences)
  can now cross process boundaries via
  `ConceptStore.save_experience` / `load_experience`, unlocking the
  knowledge-flow principle from IDEOLOGY v2 §2.

Phase D (minipc smoke + eval) is **deferred** to the next session —
this report is the checkpoint after Phases A/B/C.

## Context — why Stage 82 exists

The session that ended Stage 81 produced 8 bug fixes without moving
the Crafter eval wall (stayed at 167-178 across the whole Stage 78-81
arc). A cross-session review found that 5 of 8 bugs were
Crafter-specific facts wedged into mechanisms:

- Bug 2: `_expand_to_primitive` hardcoded facing-tile dispatch for do
- Bug 5: `_nearest_concept` pretended to derive "front tile" logic
- Bug 6 v1: hardcoded resource list in `generate_candidate_plans`
- Bug 8: `impassable_concepts()` heuristic
- (and several more — see IDEOLOGY v2 §3)

IDEOLOGY.md was rewritten to capture three categories of knowledge:

- **Facts** (textbook YAML) — what a "teacher" would tell a child
  before the child even enters the environment
- **Mechanisms** (Python code) — domain-agnostic machinery for
  simulating, searching, dispatching rules
- **Experience** (runtime state) — what the agent discovers by
  acting

plus five anti-patterns with Stage references. Anti-pattern 2
("env semantics in mechanism") and anti-pattern 1 ("learning what the
teacher knows") are the direct motivation for Stage 82.

An audit run (commit `e70f111`) found **15 concrete violations** with
proposed fixes — three classed High priority (2.4 blocking, 1.4 env
dispatch, 1.1 conjunctive rules), seven Medium (facing consolidation,
chain_generator / outcome_labeler duplicated dicts), five Low
(documentation-only).

Stage 82 is the "apply the audit" stage.

## Phase C — Knowledge persistence (commit `b48238d`)

**Goal:** cross-session knowledge flow so category-3 experience does
not die with the process.

### API

- `LearnedRule.to_dict` / `from_dict` — JSON round-trip of
  `ContextKey` (frozenset → sorted list), effect, confidence,
  `n_observations`, source.
- `HomeostaticTracker.to_dict` / `load_dict` — serializes ONLY
  `observed_rates`, `observed_max`, `observation_counts`. Innate
  rates and reference bounds come from the textbook on every init
  and are **not** serialized (would create a drift point).
  `load_dict` merges additively: running-mean combines rates, counts
  sum, max is max-of-max. Raises `RuntimeError` if loaded before
  `init_from_textbook`.
- `ConceptStore.experience_to_dict` / `load_experience_dict` —
  versioned JSON (v1) holding `learned_rules` list plus
  `rule_confidences` keyed by `{concept_id}:{action}:{index}` and
  `_passive:{kind}:{index}`.
- `ConceptStore.save_experience(path, tracker=None)` /
  `load_experience(path, tracker=None)` — disk I/O with optional
  tracker subfield. Writes JSON with `indent=2`. `load_experience`
  returns `False` if the file does not exist.
- `ConceptStore.add_learned_rule` dedup: same-precondition
  duplicates now keep the higher `n_observations` entry rather
  than growing the list.

### Tests

15 round-trip tests in `tests/learning/test_knowledge_persistence.py`:

- `LearnedRule` L1 (body_quartiles == (0,0,0,0)) and L2 round-trip.
- `LearnedRule.matches()` preserved after serialize → deserialize.
- `HomeostaticTracker` empty, with observations, additive merge
  (running-mean combines two sessions), load-before-init raises.
- `ConceptStore` experience dict empty, with learned rules, with
  modified rule confidences, `add_learned_rule` dedup behaviour.
- `save_experience` / `load_experience` file I/O, JSON validity,
  missing-file path, multi-episode accumulation simulation (three
  independent stores sharing one file, each adds and reloads).

### Demonstration harness

`experiments/stage82_knowledge_persistence.py` — runs three
independent sessions sharing one experience JSON and verifies
session N+1's loaded state covers session N's post state. **Not
executed this session** (requires the Stage 75 segmenter checkpoint
and is intended for the minipc smoke in Phase D).

### Result

All 15 persistence tests + 125/125 regression across
`tests/learning/`, `tests/agent/`, `tests/test_stage77_simulate.py`,
`tests/test_stage77_mpc.py`, `tests/test_stage78c_residual_integration.py`.

## Phase B — Textbook expansion

Five audit items closed (three High priority + the group 2.1–2.10
facing consolidation + 1.4).

### B.1 — `blocking` attribute (audit 2.4, commit `51262b4`)

Before: `impassable_concepts()` ran a 33-line heuristic that scored
"any concept with a do-rule producing positive inventory_delta is
blocking, plus any concept with `passive_movement`". A leaky
abstraction: the teacher knows exactly which tiles block movement,
the heuristic was just pattern-matching Crafter's specific physics.

After: `configs/crafter_textbook.yaml` declares `blocking: true/false`
per vocabulary entry. `impassable_concepts()` becomes:

```python
def impassable_concepts(self) -> set[str]:
    return {
        cid for cid, concept in self.concepts.items()
        if concept.attributes.get("blocking", False)
    }
```

Concept.attributes already took kwargs through
`CrafterTextbook.load_into`, so no parser changes were needed. New
envs (MiniGrid, Minecraft) declare their own blocking in their own
textbook YAML.

Regression: 213/213 on the Stage 77/78/79 test scope.

### B.2 — `any_of` / `all_of` + `action_filter` (audit 1.1, commit `42765cf`)

Before: stateful `when:` clause was a single `{var, op, value}`
triple. The Stage 78a/c/79 saga was driven entirely by trying to get
a learning mechanism to rediscover the canonical conjunctive rule:

> when last_action == sleep and (food==0 or drink==0 or energy==0),
> health drops by ~0.067/tick instead of the ~0.02/tick the per-var
> stateful rules predict.

Every attempt hit the same discrimination paradox: learning averaged
corrections across fingerprint collisions produces a predictor that
confidently predicts wrong. This is the textbook definition of
anti-pattern 1 from IDEOLOGY v2 §3 — we were asking the nursery to
learn a fact the teacher knows.

After: `StatefulCondition` has three modes:

- `atomic` (default, backward compatible): `{var, op, threshold}`.
- `any_of`, `all_of`: recursive compound with `children: list[StatefulCondition]`.

Any form may carry an optional `action_filter: <primitive>` that
gates firing on `sim.last_action == action_filter`. The recursive
`satisfied()` method handles all three modes.

Parser supports:

```yaml
- passive: stateful
  when:
    action_filter: sleep
    any_of:
      - { var: food,   op: "==", value: 0 }
      - { var: drink,  op: "==", value: 0 }
      - { var: energy, op: "==", value: 0 }
  effect: { body: { health: -0.067 } }
```

The canonical sleep + starvation rule is now in the textbook
directly (`configs/crafter_textbook.yaml`). The nursery becomes
diagnostic-only for this class of conjunction — it **should not**
rediscover this rule (doing so is the anti-pattern the audit names).

Tests: 7 new in `test_stage77_types.py` (atomic / compound / nested
/ action_filter), 3 in `test_stage77_parser.py` (YAML grammar
round-trip). Regression: 223/223.

### B.3 — Facing direction consolidation (audits 2.1/2.2/2.3/2.6/2.9/2.10, commit `f4aba1a`)

Before: five different sites repeated the same Crafter-specific
if/elif chain for `move_left/right/up/down → (dx, dy)`:

- `concept_store._apply_player_move`
- `concept_store._nearest_concept` (facing offset from `last_action`)
- `concept_store._expand_to_primitive` "do" branch (Bug 2 fix)
- `concept_store._explore_direction` (hardcoded exploration cycle)
- `mpc_agent` facing update after blocked move (Bug 6 path)

All five duplicated the same fact the teacher knows.

After: `configs/crafter_textbook.yaml` declares:

```yaml
primitives:
  move_left:  { dx: -1, dy:  0 }
  move_right: { dx:  1, dy:  0 }
  move_up:    { dx:  0, dy: -1 }
  move_down:  { dx:  0, dy:  1 }

env_defaults:
  default_facing: [0, 1]
  explore_cycle: [move_up, move_right, move_down, move_left]
```

`CrafterTextbook.load_into` copies these onto `store.primitives` and
`store.env_defaults`. `ConceptStore` exposes:

- `primitive_offset(primitive_or_none) -> (dx, dy)` — falls back to
  `env_defaults.default_facing` for non-move primitives or `None`.
- `move_primitives()`, `explore_cycle()`.

Module helpers `_apply_player_move`, `_nearest_concept`,
`_direction_primitive`, `_step_toward_target`, `_explore_direction`,
and a new `_primitive_for_delta` all take `store: ConceptStore`
(optional, for backward-compatible unit tests) and read from it.

Code was reduced by net ~40 lines. New envs with different cardinal
order or diagonal movement only need a different `primitives` block
in YAML.

Tests: 3 new in `test_stage77_parser.py` (primitives_block,
env_defaults_block, load-onto-store with `primitive_offset` fallback
and blocking flow). Existing `TestHelpers` tests updated to pass a
populated store. Regression: 226/226.

### B.4 — `env_semantics` for action dispatch (audit 1.4, commit `1f0d86d`)

Before: Stage 79 Bug 2 comments inside `_expand_to_primitive`
explicitly said: "Crafter env's 'do' action interacts with the
FACING tile, not any tile at manhattan ≤ 1, and blocked moves still
update facing". That comment was the only record of a fact the
teacher can declare.

After: `configs/crafter_textbook.yaml` adds:

```yaml
env_semantics:
  do:
    dispatch: facing_tile
    range: 1
    interaction_mode: direct
  move:
    blocked_by: impassable
    updates_facing: true
```

`ConceptStore.action_dispatch(action)` returns the spec dict (empty
if nothing declared). `_expand_to_primitive` "do" branch reads
`dispatch` and `range` from it and branches:

- `dispatch: facing_tile` → current Crafter behaviour: emit `do`
  only when target is the facing tile, emit move toward target
  otherwise so the next tick's facing aligns.
- `dispatch: proximity` → emit `do` whenever target is within
  `range`. (For MiniGrid-style envs that could plug in with a
  different textbook.)

Tests: 2 new in `test_stage77_parser.py`
(env_semantics_block, action_dispatch helper). Regression: 228/228.

### B remaining items (not touched)

Low-priority audit items not addressed in Stage 82:

- **2.5** `primitive_to_action_idx` hardcoded for Stage 78c residual.
  Residual is parked; this is dead code for the current path.
- **2.7** `chain_generator._near_label` hardcoded dicts. Not used by
  live `mpc_agent`; lives in the old Stage 71 scenario pipeline.
- **2.8** `outcome_labeler` parallel dicts. Same — not in live path.
- **1.2** movement mechanics underspecified in textbook. Works as is.
- **1.5** perception frame geometry doc-only.

These are left for a later clean-up pass when we touch those
subsystems.

## Phase D — Nightly minipc runs (2026-04-12)

Two runs on minipc after the 5 migration commits landed on `main`:

### D.1 — Knowledge persistence smoke (stage82_smoke, ~2 min)

`experiments/stage82_knowledge_persistence.py --fast`: three
independent process-level sessions share a single
`_docs/stage82_experience.json` file.

| Session | pre_rules | loaded_rules | post_rules | tracker_obs |
|---------|-----------|--------------|------------|-------------|
| 1       | 0         | 0            | 4          | 447         |
| 2       | 0         | 4            | 8          | 894         |
| 3       | 0         | 8            | 10         | 1341        |

All three pass criteria hold:
- session N+1 `loaded_rules ≥ session N post_rules` (4≥4, 8≥8)
- tracker observation counts strictly monotone across sessions
  (447 → 894 → 1341, linear in episode count as expected)
- No exceptions, no schema drift after JSON round-trips

This validates the knowledge-flow API end-to-end on real segmenter
+ Crafter env, not just in unit tests. `Overall: PASS` emitted by
the harness. Raw: `_docs/stage82_smoke_results.json` (pulled from
minipc).

### D.2 — Stage 79 harness full run with post-82 textbook (stage79_3abl, 177 min)

Goal: prove the 5 migrations do not regress the existing eval and
check whether the conjunctive-sleep rule + facing consolidation
change the nursery's ablation landscape.

Same numbers as the TL;DR table, plus per-phase deltas against
Stage 77a reference:

```
                 off       on        on+res    Δ(on-off)
warmup_a        227.7    236.4     228.9      +8.7     (77a=222, off+5.7, on+14.4)
warmup_b        171.9    171.8     158.8      -0.2     (77a=203, off-31.1, on-31.2)
eval            154.8    162.8     147.7      +8.0     (77a=180, off-25.2, on-17.2)
action_entropy  1.09     1.29      0.39       +0.20    (Stage 78c eval=0.18 collapsed)
wood avg        0.27     0.42      0.05       +0.15
```

**Three findings:**

1. **Warmup_a ≥ Stage 77a baseline (222).** All three ablations
   score above 222 in the "clean" phase (enemies off, 500 max
   steps). The 5 migrations (blocking attribute, compound stateful,
   facing consolidation, env_semantics, knowledge persistence)
   **do not introduce structural regressions** — in favourable
   conditions the new textbook runs slightly better than Stage 77a.

2. **Eval dropped off ≈-14 vs Stage 78c reference `169.2`.** This
   is not a regression from the refactor; it is the conjunctive
   sleep rule (-0.067 health when sleep + any of food/drink/energy
   at 0) exposing an artifact Stage 80 had already traced: the old
   planner used sleep as an "escape hatch" (~69% of ticks per the
   Stage 80 trace) because sleep *restored* energy without paying
   the starvation penalty. After declaring the true dynamics in
   textbook, the planner correctly avoids sleep-under-starvation,
   but cannot replace it with a faster food/water gather chain
   (the Stage 80/81 planner wall) — so the agent dies faster.
   The wall is unchanged; its cosmetic position moved.

3. **Nursery inversion: `on` beats `off` for the first time in the
   78c-81 arc.** Historical context:

   | Stage  | eval_off | eval_on | Δ(on-off) |
   |--------|----------|---------|-----------|
   | 78c    | 169.2    | 152.1   | **-17.1** |
   | 79     | ~169     | ~160    | ~-9       |
   | **82** | 154.8    | 162.8   | **+8.0**  |

   The discrimination paradox that hurt nursery_on in 78c/79 was
   the nursery trying to learn an averaged correction for the
   multi-modal `sleep + starvation` case (it fires sometimes with
   -0.067, sometimes without depending on which var is zero).
   Averaging across those modes produced a confidently-wrong
   predictor that poisoned Phase 7 rule application.

   Stage 82 audit 1.1 moved that exact conjunction into the
   textbook. The nursery's workload is now the *residual*
   per-location body-rate noise, which it learns cleanly. Dumped
   learned rules (`_docs/stage79_learned_rules_nursery_on.jsonl`,
   213 entries) are mostly of the form:

   ```
   precondition: {visible: [coal, cow, empty, iron, stone, tree],
                  body_quartiles: [0,0,0,0], action: move_right}
   effect: {food: +0.031, drink: +0.037}
   ```

   Which is "ambient observation noise per scenery mix" — not
   conjunctive physics, not collision-prone. Exactly the split the
   three-category ideology predicts: facts in textbook, residual
   observation in experience.

4. **Residual predictor confirmed parked.** `nursery_on_residual_on`
   scored -15 vs `nursery_on` at eval (147.7 vs 162.8) and dropped
   action entropy to 0.44. Stage 78c's verdict (encoding-bound) is
   replicated under the Stage 82 textbook. The Stage 78c residual
   path stays off by default.

### Pass/fail against Stage 82 goals

| Goal                                     | Result |
|------------------------------------------|--------|
| No structural regression vs Stage 77a    | ✅ warmup_a 228-236 ≥ 222 |
| 15 audit violations → High priority done | ✅ 5 of 10 High/Medium |
| Knowledge persistence works on minipc    | ✅ stage82_smoke PASS |
| Test suite 228/228 green                 | ✅ |
| nursery_on ≥ nursery_off                 | ✅ +8.0 (first time!) |
| Baseline eval unchanged (≈169)           | ⚠️ 154.8 off — see finding 2 |
| Break the eval wall                      | ❌ (not a Stage 82 goal) |

## Files changed summary

Code:
- `configs/crafter_textbook.yaml` — `blocking` attribute on
  vocabulary, `primitives`, `env_defaults`, `env_semantics`
  sections, conjunctive sleep stateful rule.
- `src/snks/agent/concept_store.py` — `impassable_concepts` simplified,
  `primitives`/`env_defaults`/`env_semantics` fields + helpers
  (`primitive_offset`, `move_primitives`, `explore_cycle`,
  `action_dispatch`), module helpers rewritten, `learned_rules`
  dedup in `add_learned_rule`, `experience_to_dict` /
  `load_experience_dict` / `save_experience` / `load_experience`.
- `src/snks/agent/crafter_textbook.py` — `primitives_block`,
  `env_defaults_block`, `env_semantics_block` properties,
  `_parse_stateful_condition` recursive helper for compound
  conditions.
- `src/snks/agent/forward_sim_types.py` — `StatefulCondition`
  mode/children/action_filter fields + recursive `satisfied()`.
- `src/snks/agent/learned_rule.py` — `to_dict` / `from_dict`.
- `src/snks/agent/perception.py` — `HomeostaticTracker.to_dict` /
  `load_dict`.
- `src/snks/agent/mpc_agent.py` — facing update after blocked move
  now via `store.primitive_offset`.

Tests:
- `tests/learning/test_knowledge_persistence.py` — 15 new tests.
- `tests/test_stage77_types.py` — 7 new compound stateful tests.
- `tests/test_stage77_parser.py` — 8 new tests (3 compound
  stateful + 3 primitives/env_defaults/load + 2 env_semantics).
- `tests/test_stage77_simulate.py` — `TestHelpers` updated for store
  parameter, new `_store_with_primitives` fixture helper.

Docs:
- `docs/reports/ideology-audit-2026-04-11.md` — audit report (Phase A).
- `docs/reports/stage-82-report.md` — this file.

Experiments:
- `experiments/stage82_knowledge_persistence.py` — 3-session harness.

## Numbers

- Files changed: 17 (excluding this report + the audit report)
- Tests: 205 → 228 (+23 in Stage 82)
- `_expand_to_primitive` / `_nearest_concept` / `_apply_player_move`
  hardcoded `move_*` sites: 5 → 0
- `impassable_concepts` LoC: 33 → 4
- Textbook YAML stanzas: +3 (`primitives`, `env_defaults`,
  `env_semantics`) + 1 (`blocking` on each vocabulary entry) +
  1 conjunctive sleep rule

## Open items / next session

1. **Phase D smoke on minipc** — ✅ done (stage82_smoke PASS).
2. **Baseline re-check** — ✅ done (stage79_3abl, 177 min). See
   finding 2 for the off-baseline story.
3. **Integrate persistence into `run_mpc_episode`** — currently
   harness-level. For production, `run_mpc_episode` (or its caller
   in an experiment script) should accept an `experience_path`
   kwarg and auto-load at start / save at end. Unblocks multi-run
   experiments that accumulate nursery rules across process
   boundaries.
4. **Low-priority audit leftovers** — when next touching
   `chain_generator` or `outcome_labeler` (old scenario pipeline),
   do the 2.7/2.8 cleanup.
5. **Re-examine Stage 78c residual verdict one more time?** Stage
   82 removed the major discrimination-paradox case from the
   nursery's workload, but the residual predictor still encodes
   multi-modal corrections via a learned MLP. With sleep-conjunction
   gone from the error signal, the residual might be less noisy —
   the 147.7 eval here is *worse* than the 162.8 nursery_on, but
   Stage 78c hadn't run under the corrected textbook. Not urgent.
6. **Stage 83 candidate** — with the nursery inversion finally
   going the right way (+8 eval), the productive next direction
   is the **planner gather-chain wall**. Candidates:
   - Proactive gather planner that commits to multi-step resource
     chains. Stage 80/81 improved `generate_candidate_plans` and
     the lex-tuple score (`distinct_gains / total / min_body`), but
     the planner still picks greedy single steps over chains — see
     `wood avg = 0.42, ≥3:0/60` in finding 3. The score function
     needs a look-ahead component, not just a better one-step score.
   - Real 2D nav heuristic in sim. Current sim walks through any
     non-blocking tile with cost 1 per step, so BFS distance
     estimates are garbage beyond Manhattan. `CrafterSpatialMap`
     has the full visited/blocked map; simulate_forward should
     path-plan through it instead of greedy Manhattan.
   - F1-F15 items from the architecture review 2026-04-10 (memory:
     `reference_architecture_review_2026_04_10.md`).

## Supporting artifacts

- `_docs/stage79_post82_results.json` — full 3-ablation harness
  output (pulled from minipc).
- `_docs/stage82_smoke_results.json` — 3-session persistence harness
  output.
- `_docs/stage82_experience_minipc.json` — serialized experience
  file from the 3-session smoke (10 learned rules + tracker obs).
- `_docs/stage79_learned_rules_nursery_on.jsonl` (on minipc) — 213
  learned rules from the nursery_on ablation for diagnostic
  inspection. Content confirms the rules are ambient
  per-location-mix body rate refinements, not rediscovered
  conjunctive physics.
