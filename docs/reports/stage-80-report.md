# Stage 80 — Diagnostic Bug Hunt + Planner / Spatial Map Fixes Report

**Date:** 2026-04-11
**Status:** **COMPLETE — partial PASS** (4 new bugs found and fixed; warmup_a baseline lifted +34 over Stage 79; eval baseline unchanged at ~178; honest negative on the wall)
**Prior stages:** [Stage 78c](stage-78c-report.md), [Stage 79](stage-79-report.md)

## Headline

Stage 80 began as Option A from the Stage 79 decision points
(encoding refinement) but pivoted within the first hour into a
**diagnostic bug hunt** after a 5-episode trace dump showed the agent
spending 69.5 % of its actions on `sleep` and gathering ZERO
resources across 5 episodes. That single observation invalidated
the framing of Stage 78c / 79 as "the residual / nursery doesn't
help" — the underlying baseline agent was a sleeping agent that
never gathered, and ALL prior fixes had been fine-tuning corrections
to a fundamentally idle policy.

By the end of Stage 80, four more bugs were found and fixed
(numbered Bug 3 through Bug 6, continuing from Stage 78c's Bug 1
and Stage 79's Bug 2). With all 6 bugs fixed:

- **warmup_a** (no enemies) lifted from Stage 79 v3's 215.9 to
  **249.6** in 60-episode eval (+33.7, +16 %).
- **wood collection** lifted from 0.18 to ~0.4 (also stage-77a
  reference baseline never gathered any wood).
- **Action entropy** is no longer collapsing under the nursery
  (Stage 78c had off=0.86 → on=0.18; Stage 80 final has
  off=1.17 → on=1.37).
- **eval avg_len** stayed at **177.6** — the wall did not break.

The eval wall is **architectural**, not bug-driven. The score
function picks the shortest gathering plan that scores has_gain=1
even when longer chains (gather → place table → make pickaxe →
gather harder resources) would yield more cumulative progress.
The agent now gathers wood at every chance but **never crafts
tools, never defeats enemies, and dies of natural body decay or
enemy damage at ~178 steps regardless**.

This report documents the 4 new bugs (Bug 3, 4, 5, 6), the fixes,
the numerical impact, the env-specificity audit (the user asked
"did you accidentally tune everything to env?" — answer: yes for
Bug 6 originally, now refactored to read from textbook), and the
architectural finding for the next decision.

## Run timeline (this session)

| Run | warmup_a off | warmup_b off | eval off | eval on | wood off | bugs |
|---|---:|---:|---:|---:|---:|---|
| Stage 78c v1 (Bug 1+2 latent) | n/a | n/a | 169.2 | 152.1 | 0 | 0 |
| Stage 78c v2 (Bug 2 fixed) | n/a | n/a | 180.4 | 154.5 | 0 | 1, 2 |
| Stage 79 v1 (Bug 2 latent) | 215.9 | 190.3 | 170.8 | 155.5 | 0.10 | nursery |
| Stage 79 v2 (Bug 2 fixed) | 223.1 | 190.3 | 184.6 | 161.4 | 0.15 | nursery |
| Stage 79 v3 (Bug 2 + cap) | 215.9 | 184.2 | 177.1 | 177.4 | 0.18 | nursery |
| Stage 80 full (Bug 3) | 223.3 | 184.2 | 177.4 | 166.6 | 0.15 | 1, 2, 3 |
| **Stage 80 full2 (Bugs 1-6)** | **249.6** | **199.1** | **177.6** | **172.1** | **~0.4** | **all** |

## Bug 3 — score function and gather plan generation

### Symptom

Stage 80 5-episode diagnostic showed:

```
Action distribution (5 eps, 814 actions total):
  sleep:  566 (69.5%)
  move:   201 (24.7%)
  do:      47 ( 5.8%)
  place:    0 ( 0.0%)
  make:     0 ( 0.0%)
Resources: wood=0, stone=0, coal=0, iron=0
```

### Root cause

Two coupled issues in `mpc_agent.py`:

1. **`generate_candidate_plans` only emitted plans as remedies for
   failures.** A failure is a body var that depletes or an entity
   that damages a vital var. Wood / stone / coal / iron are not body
   vars and never appear as failures, so the planner never even
   generated a "gather wood" plan unless it bubbled up via a tool
   prerequisite chain (which never triggered because the chain itself
   never started).

2. **`score_trajectory` alive lex tuple was `(1, min_body, n_ticks,
   final_body)`.** Sleep plans always slightly raise min_body via
   energy clamping + stateful health regen, so they out-scored every
   alternative on the safety axis. The diff was tiny but lex
   ordering is strict.

### Fix

1. `generate_candidate_plans` now adds **proactive gather plans** on
   every step. For each `action_triggered` rule whose effect has
   positive `inventory_delta`, emit a `plan_toward_rule` plan with
   origin="gather". This adds wood/stone/coal/iron gathering and
   table/tool crafting plans to the candidate set on every step
   regardless of whether a body failure triggered them.

2. `score_trajectory`'s alive tuple becomes
   `(1, has_gain, min_body, n_ticks, final_body)`, where `has_gain`
   is `1` if the rollout contains any positive `inv_gain` event,
   else `0`. Alive plans that gather beat alive plans that don't,
   regardless of min_body.

### Effect

- **5-episode diagnostic improved survival 162.8 → 213.2** (+50,
  but turned out to be a small-sample artifact).
- **60-episode full eval baseline UNCHANGED**: 177.1 → 177.4.
- Wood collection improved on the nursery side (0.05 → 0.30).
- Action entropy on nursery_on improved 0.75 → 0.96.

The 5-ep diag was misleading because of the wide variance (one
ep4 lasted 424 steps and dragged the mean up). The full 60-ep
eval showed the wall held.

## Bug 4 — `expand_to_primitive` `dist=0` oscillation in sim

### Symptom

A step-by-step diag of ep0 (seed=1000) showed the agent gathering
wood at step 3 (✅), then steps 4-7 ALL gather plans had has_gain=0
even though tree was visible 94.6 % of the time and was within
manhattan 1 in spatial_map. By step 8 has_gain=1 returned.

### Root cause

In `_expand_to_primitive`'s "do" branch (Stage 79 Bug 2 fix), when
`sim.player_pos == target_pos` (dist=0), the facing check
`(facing_dx, facing_dy) == (dx, dy)` becomes `(facing_dx, facing_dy)
== (0, 0)`, which is impossible because facing is always non-zero.
The code fell through to the "wrong facing" branch and returned a
navigation step.

After that nav step, sim.player_pos moved off the target → adjacent
(dist=1) → wrong facing (last_action just set to the move that came
FROM the target) → return another nav step → back onto the target
→ loop. Sim oscillates indefinitely between target_pos and
target_pos±1 in `_expand_to_primitive` calls during a rollout. The
plan rule never fires → has_gain=0 → score ties with baseline.

This is a sim-only artifact (env blocks walking onto trees, so
dist=0 never happens in real env), but it broke gather plan
rollouts for any sequence after the first successful gather.

### Fix

When `dist == 0` in `_expand_to_primitive`'s "do" branch, return
`"do"` immediately. The planner's `_apply_tick` Phase 6 'do' branch
with `planned_step` uses a proximity check (manhattan ≤ 1), so the
rule fires when player is at exact target. Only triggered in sim
rollouts, never in env primitives.

### Effect

Combined with Bug 5 fix below — see end of section.

## Bug 5 — `find_nearest` returns stale player-tile entry

### Symptom

Step 5 of the same step-by-step diag, after Bug 4 fix:

```
player_pos: (28, 32)  near: empty
spatial_map nearby resources (manhattan ≤ 5):
  (28, 32) = tree (d=0)   ← STALE — env removed this tree at step 3
  (28, 31) = tree (d=1)
  ...
```

The agent walked onto a chopped tree's tile. Perception's
`vf.near_concept` correctly reported "empty" for the player tile,
but `update_spatial_map_from_viewport`'s **detections loop**
overwrote the entry with "tree" — Stage 75 segmenter mis-classified
the player sprite (or its underlayer pixels) as a resource. The
spatial_map then reported `(28, 32) = tree` and `find_nearest`
returned that position with `d=0`. Combined with Bug 4 fix, the
agent emitted "do" facing some other direction → no rule fired in
env → infinite loop.

### Fix

In `crafter_spatial_map.find_nearest`, **skip entries at the
player's own position**. Crafter blocks walking onto impassable
resources, so the player can never actually be on a resource tile
in env — any spatial_map entry there is necessarily a stale
perception artifact.

This is a workaround for a perception bug, not a fix at the source
(the segmenter would need retraining), but it's a 14-line patch
and unblocks the planner.

### Effect

Combined with Bug 4 + Bug 6, see end of section.

## Bug 6 — `spatial_map` not cleared after successful gather

### Symptom

Even with Bug 5 fix (skip player's own tile), `find_nearest`
returned the **next** stale tree (e.g. `(28, 31)`) which was real
at the time but was about to be chopped. After the agent chopped
it, the perception detections loop wrote "tree" back to that
position → another stale entry → planner picked it again →
oscillation between two adjacent trees.

The root cause is **the spatial_map keeps reporting chopped
resources because the segmenter detections override the player's
near_concept**. Bug 5 patches one half (player's own tile); the
other half (the tile the agent JUST chopped, which is now in front
of them) needs its own patch.

### Fix

After `env.step` completes successfully and an inventory item
increased, the MPC loop now explicitly marks the **facing tile** as
`empty` in spatial_map. The facing tile is computed from the
previous `last_action`. If the gather succeeded, the env removed
the resource at that tile, so we know it's empty regardless of what
the segmenter reports next perception cycle.

The list of "what counts as a successful gather" is **read from
the textbook** (`store.gatherable_items()`), not hardcoded — the
user's "you accidentally tuned everything to env?" question
prompted this refactor. The textbook uses `_item` suffixes for
extracted resources (`stone_item`, `coal_item`, `iron_item`); env
inventory uses bare names (`stone`, `coal`, `iron`). The helper
returns BOTH forms so the check works against either naming
convention without hardcoding either.

### Effect (Bugs 3-6 combined)

| | Stage 79 v3 (Bug 1, 2) | Stage 80 full (Bug 1-3) | **Stage 80 full2 (Bug 1-6)** |
|---|---:|---:|---:|
| warmup_a (off) | 215.9 | 223.3 | **249.6** |
| warmup_a (on) | 215.9 | 228.4 | 236.3 |
| warmup_b (off) | 184.2 | 184.2 | 199.1 |
| warmup_b (on) | 159.8 | 175.1 | 193.9 |
| **eval (off)** | **177.1** | **177.4** | **177.6** |
| **eval (on)** | 177.4 | 166.6 | 172.1 |
| eval entropy (off) | 0.97 | 0.98 | 1.17 |
| eval entropy (on) | 0.75 | 0.96 | 1.37 |
| wood mean (off, eval) | 0.20 | 0.15 | ~0.4 |

**The eval baseline is unchanged at ~177-178 across all 6 bug fixes.**
warmup_a improved by +34, wood collection by ~2× to ~3×, action
entropy normalized — but eval (which includes 1000 max_steps and
enemies) stayed flat.

## The unbroken wall — architectural finding

After Bugs 3-6, the agent gathers wood reliably. But it never
crafts tools and never defeats enemies. Why? **The score function
can't distinguish "1-step do tree" from "5-step do tree → place
table → make wood_pickaxe → do stone".**

```
Plan A: do(tree)                                       (1 step)
Plan B: do(tree)+do(tree)+place(table)+make(pickaxe)+do(stone)  (5 steps)

Both have has_gain=1 (Plan A: wood +1; Plan B: wood +2 + table +
                      pickaxe + stone)
Plan A min_body: high (1 navigation + 1 do, body decays ~0.06 over 2 ticks)
Plan B min_body: lower (4 navigations + 1 do + 1 place + 1 make,
                        body decays ~0.4 over 7 ticks)

Lex tuple (1, has_gain, min_body, ...) ranks Plan A > Plan B
→ planner ALWAYS picks short gather, never the chain.
```

The agent infinite-loops on `(navigate, do, navigate, do, ...)`,
gathering 1 wood at a time, never accumulating enough for crafting.
The wall ~178 is **the wall of the agent that never crafts a
sword**. Without sword, every skeleton arrow is unanswered, and
the agent dies at the same moment regardless of how much wood it
gathered.

This is **the architectural bottleneck**, distinct from the 6 bugs.
The fix is one of:

- **Score function redesign**: replace `has_gain` (binary) with a
  cumulative score (sum of inv_gains, weighted by item rarity, or
  by tool-readiness). Plans that build toward sword score higher.
- **Hierarchical goals**: planner emits subgoal "make wood_sword"
  which is decomposed into the chain. Each step rewards toward
  the subgoal, not just immediate inv_gain.
- **Action priors**: explicitly bias the planner toward make/place
  actions when the prerequisite items are accumulated.

These are Stage 81+ design decisions, not single-line bug fixes.

## Env-specificity audit

The user asked: "did you accidentally tune everything to env?"
Honest answer: **partially**.

| Bug | Fix file | Env-specific? |
|---|---|---|
| 1 (Stage 78c) | mpc_agent.py training | **No** — universal: training signal must match planning signal |
| 2 (Stage 79) | concept_store expand_to_primitive | **Partial** — relies on Crafter "do uses facing tile" semantics. Other envs may differ. |
| 3 | mpc_agent.py | **No** — proactive gather plans + has_gain priority work for any env with inventory items, no hardcoded names |
| 4 | concept_store expand_to_primitive | **Partial** — relies on "env blocks walking onto impassable tiles". Other envs may not block. Sim/env semantic mismatch. |
| 5 | crafter_spatial_map.find_nearest | **Partial** — same root as Bug 4 (env blocking) + perception artifact specific to Stage 75 segmenter |
| 6 (original) | mpc_agent.py | **🚨 Yes — hardcoded Crafter item names** |
| 6 (refactored) | mpc_agent.py + concept_store.gatherable_items | **No** — reads item names from textbook |

**Refactored** during this report-writing pass: Bug 6 originally
hardcoded `("wood", "stone", "coal", "iron", "diamond", "sapling")`
in `mpc_agent.py`. Now `mpc_agent` calls `store.gatherable_items()`,
which iterates the loaded textbook and returns any item appearing
as a positive `inventory_delta` in a `do` rule. Strips `_item`
suffix to handle the textbook/env naming convention gap. No
Crafter-specific strings remain in the planner.

Bugs 2, 4, 5 still rely on the implicit assumption that the env
follows Crafter's "do uses facing + blocks impassable" semantics.
The right place for these to live is a future **env_model** layer
that declares env semantics formally. Stage 81 candidate.

## Strategy v5 novelty accounting after Stage 80

| # | Novelty | Status |
|---|---|---|
| 1 | DAF substrate as residual learner | parked Branch C |
| 2 | No-LLM rule induction from surprise | synthetic PASS, Crafter neutral. **Stage 80 confirms the nursery infrastructure is sound** — the wall is in the planner, not the learner. |
| 3 | Three-category ideology | retained, correctly implemented through Phase 7. The Bug 6 refactor restored ideology compliance after the original hardcoded list violated it. |

## Decision points

The wall is **architectural**, not bug-driven. Three options for
Stage 81:

### Option B1 — Score function redesign (cheapest structural fix)

Replace `has_gain` (binary) with a cumulative score. Change the
alive tuple to weigh long chains higher when they yield more
total inv_gain. Risk: tuning. Cost: 1-2 days. Could break Stage
80 results if the new score over-rewards risky plans.

### Option B2 — Hierarchical planning

Planner emits high-level goal ("achieve wood_sword"), low-level
controller handles the chain. Architecturally larger change.
Cost: 1-2 weeks. Risk: high — the existing planner would need
restructuring.

### Option B3 — Stop and reconsider

Stage 80's diagnostic phase was the most informative work of this
multi-stage session. We now know:

- The discrimination paradox from Stage 78a/c/79 is **not** the
  primary blocker
- Stage 78c residual / Stage 79 nursery were both **fine** as
  mechanisms; their failures were measured against a sleeping
  baseline that never gathered
- The real bottleneck is the planner's inability to commit to
  multi-step crafting chains
- Bug fixes can lift the no-enemy baseline by ~34 steps but the
  enemy-on eval is bounded by the no-craft-no-defense gap

A consolidation step here would mean: write a "Stage 80 closing
report" (this doc), update memory with the architectural finding,
revisit the broader Strategy v5 framing in light of "the bottleneck
is in scoring chains, not in learning corrections", and present a
clear next-stage decision to the user.

**Default recommendation: B3 (consolidate), then B1 (score redesign)
as the next stage of work.** B1 is the cheapest test of the
chain-scoring hypothesis. If B1 succeeds (eval > 200), the
architectural finding is confirmed and we have Crafter Gate 1 path.
If B1 fails (eval still ~178), the bottleneck is deeper and B2
becomes more attractive.

## Files

### Code (MODIFIED)

- `src/snks/agent/mpc_agent.py`:
  - `generate_candidate_plans` adds proactive gather plans
  - `score_trajectory` adds `has_gain` to alive tuple
  - `run_mpc_episode` clears facing tile from spatial_map after
    successful gather (uses `store.gatherable_items()`)
- `src/snks/agent/concept_store.py`:
  - `expand_to_primitive` `do` branch handles `dist=0` (Bug 4)
  - `gatherable_items()` helper (env-agnostic)
- `src/snks/agent/crafter_spatial_map.py`:
  - `find_nearest` skips player's own position (Bug 5)

### Code (NEW diagnostic scripts)

- `experiments/diag_stage80_baseline.py` (5-ep trace dump)
- `experiments/diag_one_step_scores.py` (10-step ALL-candidates trace)

### Tests

- `tests/test_stage77_mpc.py`:
  `test_normalization_by_reference_max` and
  `test_tuple_order_lexicographic` updated for new alive tuple shape
- 110/110 tests passing across all changes

### Reports / data

- `_docs/stage80_full2_results.txt` — final 60-episode eval (Bugs 1-6)
- `_docs/stage80_diag_results.txt` — 5-ep baseline diag (no fix)
- `_docs/stage80_diag10v4_results.txt` — 10-step trace with Bug 5
- `docs/reports/stage-80-report.md` — this report

## Commit chain (this session)

- `2997061` — Stage 78c initial implementation
- `b871309` — Stage 78c launcher fix
- `faf3ca3` — **Bug 1 fix** (training planned_step)
- `6347824` — Stage 78a methodological gaps softening
- `99e79da` — Stage 78c v1 results
- `fa60add` — Stage 78c v1 report
- `15cc2d1` — Stage 79 design + 78c paradox connection
- `16a4a97` — Stage 79 modules (accumulator + nursery + learned_rule)
- `b652554` — Stage 79 mpc integration
- `582e5a0` — Stage 79 v2 results (Bug 2 fix)
- `db73536` — **Bug 2 fix** (expand_to_primitive facing)
- `e0fedd6` — Stage 79 v3 MAX_ABS_EFFECT cap
- `c472fac` — Stage 79 v3 + Stage 78c v2 results
- `e5b14e9` — Stage 79 final report
- `3b16de8` — Stage 80 baseline diag script
- `a359984` — **Bug 3 fix** (gather plans + has_gain)
- `a6b3b7a` — One-step diag script
- `249be17` — 10-step diag expansion
- `e0ddaa3` — **Bug 4 fix** (expand dist=0)
- `ba4ab4a` — Spatial map dump in diag
- `5a64fbd` — **Bug 5 fix** (find_nearest skip player tile)
- `7cd2e3e` — **Bug 6 fix** (clear chopped tile, original hardcoded)
- `3a2bab0` — Stage 80 data files
- (next) — Bug 6 refactor + Stage 80 report
