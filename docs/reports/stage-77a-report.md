# Stage 77a — ConceptStore Forward Simulation + MPC Report

**Date:** 2026-04-10
**Status:** COMPLETE (partial PASS — ideology validated, survival wall persistent)
**Spec:** `docs/superpowers/specs/2026-04-10-stage77a-conceptstore-forward-sim-design.md`
**Plan:** `docs/superpowers/specs/2026-04-10-stage77a-implementation-plan.md`
**Architecture review driving the pivot:** `docs/reports/architecture-review-2026-04-10.md`

## Summary

Stage 77a replaced the Stage 76 memory-based reactive substrate with a
Model-Predictive-Control (MPC) loop that **simulates forward through the
ConceptStore causal rules**. Every tick the agent generates a small pool
of candidate plans, rolls each out through `simulate_forward`, scores the
resulting trajectories against the homeostatic tracker's vital bounds,
picks the best, and executes only the first primitive action — then
re-plans. No string-dispatched pseudo-concepts, no reactive recall, no
hardcoded drive categories.

**Gate 1 (survival ≥200) — FAIL at overall 180.** Same wall as Stage 76
(178). **But the architecture change succeeded:** the forward-sim MPC
path runs end-to-end on real Crafter rollouts without any hardcoded
reflexes or derived features, the Bayesian tracker refines innate priors
from observation, blocked tiles are learned from failed moves, and all
140 Stage 77 tests are green.

| Variant | Eval overall | Warmup A (safe) | Warmup B (enemies) | Wood smoke ≥3 |
|---------|--------------|-----------------|--------------------|---------------|
| Stage 75 scripted baseline | 178 | — | — | 13/20 (65%) |
| Stage 76 v1 FIFO SDM | 177 | — | — | 8/20 (40%) |
| Stage 77a run8 (final) | **180** | **222** | **203** | 0/20 (0%) |

Interpretation: the agent now survives **as long** as the scripted
baseline in eval but trades off wood gathering for survival-focused
actions. Warmup without enemies hits 222 — it's only the enemy-driven
health loss that pins eval at the wall. The fix is not more tuning of
the current architecture; it is the runtime rule induction deferred to
Stage 77b+.

## Partial PASS Criteria Met

1. **Ideology — PASS.** No `if-else` reflexes in policy code, no
   hardcoded drive categories, no derived features. The entire
   decision loop reads `store.simulate_forward(plan, state, tracker)`
   and scores the resulting `Trajectory` against `tracker.vital_mins`.
2. **Architecture — PASS.** Three-category knowledge taxonomy
   (facts / mechanisms / experience) implemented and enforced by tests.
   Facts live in YAML, mechanisms in `simulate_forward` dispatch,
   experience in tracker observed_rates and spatial_map blocked set.
3. **Forward sim correctness — PASS.** 6-phase tick dispatch
   (body_rate → stateful → action-triggered → clamp → spatial →
   movement → step). 140/140 stage77 tests green including `is_dead`
   behaviour, `plan_toward_rule` backward chaining, `find_remedies`
   query, and end-to-end MPC integration.
4. **Observation-based learning — PASS.** Blocked tiles are marked
   from `(prev_action startswith "move_") && prev_pos == pos`, not
   from a hardcoded wall-avoidance rule. Homeostatic rates update via
   Bayesian combination `w·innate + (1-w)·observed` where
   `w = prior_strength / (prior_strength + n)`.
5. **Tests — PASS.** `pytest tests/test_stage77_*.py` → 140 passed.
   Full-suite stage77 portion green; preexisting failures elsewhere
   (encoder/replay/stage15/47/66) are unrelated.

## Gate Failures (accepted)

1. **Gate 1 — survival ≥200**: FAIL at overall 180 (per-run 193/171/175).
2. **Gate 3 — wood ≥3 in smoke**: FAIL at 0/20.

Both failures trace back to the same root cause: the textbook teaches
`rough directional priors` (skeleton spatial range 5, zombie adjacent
damage, body decay -0.02/tick) and the agent's forward-sim picks the
survival plan — but it has no mechanism yet to **induce** sharper
conditional rules from observed surprises. When a plan predicts health=9
and the real outcome is health=3 after a skeleton arrow, the observation
is logged by the tracker but no rule is added that would change the next
plan's estimate. This is the Stage 77b deliverable.

## What Was Built (committed on main)

All commits on main are at or after `a5cf447` (Stage 76 close-out).

### Commit 1 — `forward_sim_types.py` (new dataclasses)
- `RuleEffect` — structured replacement for `CausalLink.result: str`.
  10 valid `kind` values (`gather`, `craft`, `place`, `remove`, `movement`,
  `spatial`, `stateful`, `body_rate`, `consume`, `self`).
- `StatefulCondition`, `SimState`, `DynamicEntity`, `SimEvent`,
  `Trajectory`, `Failure`, `Plan`, `PlannedStep`. All pure data, no
  behaviour beyond trivial helpers.

### Commit 2 — Textbook YAML parser + grammar
- `_parse_rule_dict` in `crafter_textbook.py` handles structured YAML
  (action-triggered × 4 + passive × 4) and produces
  `(concept_id, CausalLink)` tuples.
- `configs/crafter_textbook.yaml` rewritten to structured rules with a
  `body` block containing `prior_strength`, per-variable reference bounds,
  and rough directional rates.

### Commit 3 — HomeostaticTracker refactor
- Innate / observed split: `innate_rates` (from textbook) + running-mean
  `observed_rates`. `get_rate(var)` returns Bayesian combination.
- `vital_mins` — only variables flagged `vital: true` in the textbook
  trigger `is_dead`; non-vital depletion triggers stateful damage.
- `init_from_textbook(body_block, passive_rules)` is idempotent —
  calling twice does not overwrite in-progress observation counts.

### Commit 4 — ConceptStore forward sim methods
- `simulate_forward(plan, state, tracker, horizon)` runs a multi-tick
  rollout applying the 6-phase tick to a deep-copied SimState. Produces
  a `Trajectory` with per-tick body series, events, and termination.
- `_apply_tick(state, tracker, planned_step)` — the 6-phase dispatcher.
- `plan_toward_rule(rule, state, store)` — backward chain a target
  causal rule into a sequence of `PlannedStep`s, resolving prerequisites
  via `_find_rule_producing_item` / `_find_rule_producing_world_place`.
- `find_remedies(failure)` — query the world model for rules that
  counteract an observed `Failure`.
- `verify(prediction, outcome)` now dispatches via
  `_effect_matches_outcome(effect, outcome)` — inventory delta,
  body restore, scene_remove, world_place.
- Confidence threshold filter: rules below `CONFIDENCE_THRESHOLD = 0.1`
  are skipped by simulate_forward.

### Commit 5 — MPC loop (`mpc_agent.py`)
- `run_mpc_episode(env, segmenter, store, tracker, max_steps, horizon)`
  is the single entry point. Each tick:
  1. Observe world (pixels → `perceive_tile_field` → VisualField).
  2. Update `DynamicEntityTracker` and homeostatic tracker.
  3. `generate_candidate_plans` — baseline (inertia) + remedy plans
     for observed failures + proactive resource-gather injections.
  4. `score_trajectory` — lexicographic tuple
     `(survived, neg_time_to_death, resources_gained, exploration)`.
  5. Pick best, `expand_to_primitive`, execute first step only.
  6. Re-plan next tick.
- Observation-based blocked-tile learning:
  `prev_action.startswith("move_") && prev_pos == pos → mark blocked`.

### Commit 6 — `exp137_mpc_forward_sim.py`
- Five phases: segmenter load, warmup A (safe), warmup B (enemies),
  evaluation (3×20 eps, max_steps=1000, enemies on), gate checks.
- Warmup A feeds the tracker enemy-free decay observations; Warmup B
  conditions the spatial/stateful damage. Eval is the unchanged gate.

### Commit 7 — Stage 76 substrate removal
- Deleted `src/snks/memory/` (sdr_encoder, state_encoder, episodic_sdm,
  bit_attention), `continuous_agent.py`, all `tests/test_stage76_*.py`
  (5 files), `experiments/exp136_continuous_learning.py`, diag scripts.

### Commit 8 — Dead perception + legacy compat cleanup
- Removed `CausalLink.result` (string dispatch);
- Removed `ConceptStore.plan()` / `_plan_recursive` (legacy forward
  planner); `save/load` stubbed to NotImplementedError pending
  RuleEffect serializer.
- Removed `_parse_rule_legacy`, `_parse_rule` polymorphic dispatcher,
  `_derive_legacy_result`, `body_rules` synthesis from
  `crafter_textbook.py`. Structured-YAML only.
- Collapsed `perception.py` to the live surface: `HomeostaticTracker`,
  `VisualField`, `perceive_tile_field`, `verify_outcome`,
  `outcome_to_verify`. Removed `HOMEOSTATIC_VARS`, `select_goal`,
  `compute_drive`, `compute_curiosity`, `get_drive_strengths`,
  `_STAT_GAIN_TO_NEAR`, `perceive_field`, `ground_*`,
  `on_action_outcome`, `retrain_features` (all Stage 72-74 dead code).
- Deleted `tests/test_stage71.py`, `test_stage72.py`, `test_stage73.py`.
- Followup: `fix: perceive_tile_field — wrap numpy pixels into torch
  tensor` — the numpy→tensor conversion previously lived in
  `continuous_agent.perceive_field`; restored inside
  `perceive_tile_field` itself.

## Iteration Log

Five runs on minipc during the development cycle. Each run reset to a
clean venv and pulled the latest main.

| Run | Rates model | Warmup A | Warmup B | Eval overall | Notes |
|-----|-------------|----------|----------|--------------|-------|
| 1   | legacy string dispatch | 167 | 164 | 167 | Baseline pre-refactor |
| 5   | structured effect, rough rates | 174 | 176 | 172 | Plan-count bug fixed |
| 6   | precise rates from Crafter source | 190 | 195 | 183 | Best absolute numbers |
| 7   | rough textbook prior + tracker-learned (ideology-first) | 252 | 204 | 178 | First per-run ≥200 in run0 |
| 8   | same + Commit 8 cleanup | **222** | **203** | **180** | Architecture validated |

The jump from Run 1 → Run 8 (167 → 180) is modest in eval average but
decisive in structure: Run 1 had a string-based dispatch with hardcoded
plan-completion checks; Run 8 dispatches through a single
`RuleEffect.kind` discriminator and uses the same `simulate_forward`
primitive for every rollout.

## Why We Stopped Here

The user's guidance after Run 7: *"я за идеологию. Мне важны не
абсолютные числа, а подтверждение работоспособности идеологии, пусть
даже метрики хуже и дольше обучается."* Run 8 confirms the ideology
holds under cleanup. The per-run variance (393 max in run0, 42 min in
run1) shows the architecture has the **range** to exceed 200 — it just
can't find the right plan reliably without a way to sharpen its world
model from observed surprises. That is the Stage 77b scope.

## Deferred to Stage 77b / later

1. **Surprise→rule induction.** When a `simulate_forward` prediction
   disagrees with the actual `tracker.update` observation by more than
   some threshold, emit a candidate rule (with `confidence ≈ 0.3`)
   describing the observed delta. Verify next time the state recurs.
2. **when-clause conjunction grammar.** Current passive_stateful rules
   support only a single predicate on one body variable. Needed:
   AND/OR chains over body + inventory + visible concepts.
3. **Conditional rate learning.** `tracker.observed_rates` is global
   per-variable. It should be conditioned on visible_concepts /
   inventory presence so that `food rate while zombie visible` can
   diverge from `food rate in open field`.
4. **Enemy spawn modelling.** Zombie/skeleton spawn timing is currently
   invisible to the sim — only their current positions matter. A
   spawn-rate passive rule would let MPC account for "no enemies now
   but one will arrive in ≤5 ticks with p≈0.3".
5. **Lava / water / drowning.** Not in the current textbook. Skipped
   after Run 6 showed the real cause of early deaths was skeleton
   arrows at range 5, not environment hazards.
6. **ConceptStore save/load.** Re-introduce after RuleEffect gets a
   dedicated JSON serializer. Not urgent since textbook always loads
   from YAML and experience lives in tracker/spatial_map.

## Files Touched

Source additions:
- `src/snks/agent/forward_sim_types.py` (new)

Source rewrites:
- `src/snks/agent/concept_store.py` — simulate_forward, plan_toward_rule,
  find_remedies, verify, _apply_tick, confidence threshold
- `src/snks/agent/crafter_textbook.py` — structured YAML parser only
- `src/snks/agent/perception.py` — tracker refactor, dead-code removal
- `src/snks/agent/mpc_agent.py` — run_mpc_episode, generate_candidate_plans,
  score_trajectory, blocked-tile observation
- `src/snks/agent/crafter_spatial_map.py` — `_blocked` set,
  `mark_blocked`, `is_blocked`, `unvisited_neighbors` filter

Configs:
- `configs/crafter_textbook.yaml` — structured rules, body block, rough
  directional priors

Experiments:
- `experiments/exp137_mpc_forward_sim.py` (new)

Tests (Stage 77a):
- `tests/test_stage77_types.py` (140 items total across all 5 files)
- `tests/test_stage77_parser.py`
- `tests/test_stage77_tracker.py`
- `tests/test_stage77_simulate.py`
- `tests/test_stage77_mpc.py`

Deleted:
- `src/snks/memory/` (entire package)
- `src/snks/agent/continuous_agent.py`
- `tests/test_stage71.py`, `test_stage72.py`, `test_stage73.py`,
  `test_stage76_*.py` (5 files)
- `experiments/exp136_continuous_learning.py`, diag scripts

## Status

**Stage 77a: accepted as partial PASS. Architecture proven,
survival wall persistent. Next:** Stage 77b runtime rule induction
from surprise, or Stage 78 if the user pivots.
