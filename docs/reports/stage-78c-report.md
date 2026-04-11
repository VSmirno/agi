# Stage 78c — MLP Residual Crafter Integration + Online SGD + Eval Report

**Date:** 2026-04-11
**Status:** COMPLETE — **PARTIAL FAIL on the survival gate**, with informative substructure
**Spec:** ad-hoc (no formal spec doc — implementation-driven from Stage 78b status revision)
**Prior stage:** [Stage 78b unit-PASS / Crafter-pending report](stage-78b-report.md)
**Bug fix that gates this report:** see "Bug 1: planned_step in training rules-only replay"

## Summary

Stage 78c integrated `ResidualBodyPredictor` (Stage 78b artifact) into
the live MPC loop:

- `ConceptStore.simulate_forward` now accepts an optional
  `residual_predictor`. When provided, after each `_apply_tick` the
  residual encodes `(visible_concepts, body_pre_tick, primitive)` and
  adds a 4-dim correction to `sim.body`, with re-clamping. Small init
  guarantees `≈0` correction at episode 1.
- `run_mpc_episode` accepts `residual_predictor`, `residual_optimizer`,
  `residual_train`. After each `env.step` it runs an online SGD step:
  computes a 1-tick rules-only replay on the current state, takes the
  observed `actual_delta` from `env.step`, and minimises
  `MSE(rules_delta + residual(fp), actual_delta)`.
- A new harness `experiments/stage78c_residual_crafter.py` runs both
  ablations (residual_off, residual_on) on identical seeds, with
  fresh `ConceptStore` + `HomeostaticTracker` per ablation, the
  same warmup A / warmup B / eval phasing as Stage 77a's `exp137`.

The result is a partial fail on the headline gate (residual_on does
not beat residual_off in eval) but with two structurally important
findings: residual_on **does** beat residual_off in warmup_a (no
enemies) by +43.6 steps, and residual_on suffers a clear
**action-entropy collapse** in enemies-on phases. Stage 78b was first
reopened from "COMPLETE PASS" to "unit-PASS / Crafter-pending"
under the Crafter-every-stage rule; Stage 78c now closes that
Crafter eval, so 78b's reopening is resolved as well.

## Setup

- **Phases per ablation:** warmup A (50 eps, enemies off, max_steps
  500), warmup B (50 eps, enemies on, max_steps 500), eval (3 runs ×
  20 eps, enemies on, max_steps 1000).
- **Seeds:** identical across ablations; `ep × 11 + seed_offset` per
  Stage 77a `exp137` convention. Per phase: 300, 500, 1000+i×100.
- **Fresh state:** each ablation gets its own `ConceptStore`,
  `HomeostaticTracker`. Segmenter is shared (read-only Stage 75
  checkpoint).
- **Residual:** `ResidualConfig()` defaults — 1048→64→4 MLP, 67396
  params, Adam lr=1e-3, small init (weights × 0.1, bias zero).
- **Residual training:** ON during all three phases for residual_on.
- **Hardware:** minipc GPU (AMD Radeon, ROCm 7.2). Total run time
  78.7 minutes.

## Results — headline ablation table

| Phase | residual_off | residual_on | Δ (on − off) | Stage 77a baseline | off Δ vs base | on Δ vs base |
|---|---:|---:|---:|---:|---:|---:|
| warmup_a (no enemies, 50 ep) | 215.9 | **259.5** | **+43.6** | 222 | −6.1 | **+37.5** |
| warmup_b (enemies on, 50 ep) | 182.3 | 159.8 | −22.5 | 203 | −20.7 | −43.2 |
| **eval** (60 ep, enemies on, max 1000) | **169.2** | **152.1** | **−17.1** | 180 | −10.8 | **−27.9** |

| Phase | wood ≥ 3 (off) | wood ≥ 3 (on) |
|---|---:|---:|
| warmup_a | 0/50 | 0/50 |
| warmup_b | 0/50 | 0/50 |
| eval | 0/60 | 0/60 |

Wood gathering wall is unchanged from Stage 77a — neither ablation
breaks through the wood ≥ 3 threshold (consistent with Stage 77a
Run 8 result of 0/20 wood gates).

## The headline number

`residual_on_beats_off` — **FAIL**. Eval avg_len with residual is 152.1,
17.1 steps below residual_off and 27.9 steps below the cached Stage 77a
Run 8 baseline of 180.

But the diagnostic gate `residual_on_meaningful_delta` is **TRUE** —
the residual is making a real difference, just not in the direction
we wanted on the eval phase.

## Action-entropy collapse — the smoking gun

| Phase | residual_off entropy | residual_on entropy | Δ |
|---|---:|---:|---:|
| warmup_a | 1.14 | 0.73 | −0.41 |
| warmup_b | 1.25 | 0.23 | **−1.02** |
| eval combined | **0.86** | **0.18** | **−0.68** |

`avg_entropy` is the Shannon entropy of the agent's action distribution
across an episode. residual_on collapses to a much narrower action
distribution as soon as enemies are introduced (warmup_b → eval). In
eval the residual_on agent is acting **5× more deterministically**
than residual_off.

This is the actual mechanism by which residual_on hurts: the residual
is making a small subset of plans look much more attractive than
others to the planner's `score_trajectory`, the planner deterministically
picks them, and the agent fails to use the diverse-action tactics that
were keeping residual_off alive in enemies-on contexts.

residual_off entropy stays in 0.86–1.25 range across phases — the
baseline planner explores diverse tactics. residual_on entropy starts
at 0.73 in warmup_a (already noticeably narrower) and crashes to
0.18–0.23 once enemies are on the scene.

## Residual training is converging — that's not the problem

| Phase | residual_loss_mean |
|---|---:|
| warmup_a | 0.7429 |
| warmup_b | 0.2031 |
| eval | 0.2158 |

The MLP residual is learning. Loss drops 4× from warmup_a to warmup_b
and stays in a narrow band through eval. The trained residual is
not diverging or oscillating. **The training signal is now correct
(after the planned_step bug fix).**

So this is not a "training broke" failure mode. The residual is
fitting *something*, and that *something* is not useful for survival
in enemies-on Crafter.

## Bug 1: `planned_step` in training rules-only replay (root cause + fix)

This deserves its own subsection because the first version of Stage 78c
had a wiring bug that produced very different (worse) numbers, and the
debugging session that found it materially shaped the rest of the
analysis.

### Symptom

Initial smoke run on minipc (3 ep × 3 phases × 2 ablations, 18 ep total)
showed `residual_on` slightly worse than `residual_off` everywhere, and
`residual_loss_mean` going *up* across phases (warmup_a 0.33 →
warmup_b 0.24 → eval 0.62) instead of converging.

### Investigation (systematic-debugging skill)

**Phase 1 — Root Cause.** Read `configs/crafter_textbook.yaml` looking
for any `do <target>` rule with a body delta (the only kind that would
distinguish the planner-side proximity branch from the trainer-side
fallback). Found two:

```yaml
- action: do
  target: cow
  effect: { body: { food: +5 } }

- action: do
  target: water
  effect: { body: { drink: +5 } }
```

Re-read `_apply_tick` Phase 6 in `concept_store.py:725`. The `do`
branch has two paths:

1. **`if planned_step and ...`** — proximity-based: if the rule's
   target is at `manhattan ≤ 1` from the player (via dynamic_entities
   or spatial_map), fire the rule.
2. **`if not fired:`** — facing-based fallback: `_nearest_concept(sim)`
   returns the tile in front of the player (per `last_action`
   direction); look up a `do` rule for that concept.

The Stage 77a comment in `concept_store.py:646–651` explicitly notes
that the facing-based fallback is broken in sim contexts:

> Stage 77a Attempt 2: planned_step is optionally passed so Phase 6
> "do" can fire a rule by target proximity (manhattan ≤1) instead
> of relying on facing-direction. **In the sim, navigation walks
> through target tiles, making facing-based rules never fire** —
> proximity check works around that.

The first version of `run_mpc_episode`'s training block called
`store._apply_tick(rules_sim, primitive, tracker, rules_traj, tick=0)`
**without** a `planned_step` argument (default `None`). That dropped
the proximity branch and fell into the broken fallback. For any chosen
plan whose first step was `do cow` or `do water`, the training-side
rules_delta became 0 instead of the planner-side prediction of +5,
even though the planner had used the proximity branch when computing
`best_traj`.

### Confirmation via failing test

`tests/test_stage78c_residual_integration.py::test_apply_tick_do_water_proximity_vs_facing_divergence`
constructs a SimState with water at (11, 10), player at (10, 10), and
`last_action=move_up` (facing direction is (10, 9), NOT water). Calls
`_apply_tick` twice — once with `planned_step=PlannedStep(do, water,
do_water_rule)`, once with `planned_step=None` — and asserts the
divergence:

- **with planned_step**: `drink ≈ 9.0` (rule fires +5, clamped at
  ref_max=9)
- **without planned_step**: `drink ≈ 4.98` (just background -0.02
  decay, fallback misses)

Test passes (asserting the divergence exists), confirming the bug.
Cow was tried first but cow has `random_walk` movement in Phase 1 of
`_apply_tick`, so by the time Phase 6 fires the cow may have moved out
of `manhattan ≤ 1` — the test is unstable for cow. Water is in the
spatial_map (no movement) and gives a deterministic test.

### Effect on the residual

When training rules_delta is systematically 0 for do-cow / do-water
but the env actually applies +5 when facing aligns:

- `target = actual - rules_delta = +5 - 0 = +5`
- residual learns to predict +5 for fingerprints involving cow/water
  visible
- in next planning iteration: planner's first tick rules_delta = +5
  (correct via proximity branch), then residual adds another +5 → +10
- planner over-predicts food/drink restoration by 2×
- agent under-eats / under-drinks because the plan looks already
  optimistic → starvation

### Fix

One change in `mpc_agent.py`:

```python
chosen_planned_step = best_plan.steps[0] if best_plan.steps else None
store._apply_tick(
    rules_sim,
    primitive,
    tracker,
    rules_traj,
    tick=0,
    planned_step=chosen_planned_step,  # propagated, not None
)
```

After the fix, training rules_delta matches what the planner saw on
the chosen plan's first tick. The residual now learns the gap between
that planner-side prediction and the actual env outcome, which is the
correct supervision signal.

### What the fix actually changed in the numbers

| | Buggy smoke (3 ep) | Fixed full run (60 ep eval) |
|---|---:|---:|
| residual_on warmup_a avg_len | 41.0 | **259.5** |
| residual_loss eval | 0.6251 | 0.2158 |
| residual_on eval avg_len | 171.0 | 152.1 |
| residual_off eval avg_len | 184.3 | 169.2 |

The fix dramatically improved residual learning quality (loss 0.62 →
0.22, warmup_a survival 41 → 259) and **proves the wiring is now
correct in the no-enemy phase**. But on enemies-on phases the fixed
residual is *still* worse than residual_off, by a different mechanism
(action-entropy collapse, see above).

So the bug fix was real and load-bearing — without it, all conclusions
about residual learnability would have been wrong — but it did not
change the headline survival gate for the enemies-on case.

## Interpretation — why does residual_on hurt enemies-on?

The fix-then-fail pattern decomposes into two separable findings:

### Finding A — residual learns useful corrections in the no-enemy regime

warmup_a shows residual_on +43.6 steps over residual_off, +37.5 over
the cached Stage 77a baseline. The residual is doing real work: the
training signal is correct, the learned correction reduces planner
prediction error, the planner picks better plans, the agent survives
longer. This is the strongest empirical signal we have that the
Stage 78c design is *capable* of helping.

### Finding B — residual encoding is too coarse for the enemy-conditional structure

The encoding fed to the residual is `(visible_concepts, body_buckets,
action_idx)`:

- 1000 hashed concept bits (multi-hot from `vf.visible_concepts()`)
- 40 body bucket bits (4 vars × 10 buckets, 1-hot per var)
- 8 action one-hot bits

What it does **not** include:

- Last action / facing direction
- Entity positions (cow at right vs cow at left vs cow at top)
- Spatial map state (which tiles are known impassable)
- Inventory beyond the 4 body vars
- Skeleton/zombie distance to player

So multiple env-distinct states that need different corrections map to
the same residual fingerprint. The residual learns the *average*
correction over collisions. In a regime where the gap between planner
prediction and reality is **homogeneous within a fingerprint** (e.g.
warmup_a, where rules predictions are mostly correct and the
corrections are small + consistent), this works. In a regime where the
gap is **bimodal or multimodal within a fingerprint** (e.g. enemies-on
where one fingerprint covers both "skeleton at range 5 → take damage"
and "skeleton at range 6 → no damage"), the average correction is
wrong for both modes.

The action-entropy collapse is a secondary symptom of this: the
residual amplifies the relative score of a small subset of plans
deterministically, the planner's `score_trajectory` then ranks those
plans top consistently, and the agent's action distribution narrows.
That's catastrophic in enemies-on contexts where varied tactics
(retreat, attack with sword, place stone barrier, sleep to recover)
are how the baseline planner survives.

### Finding C — the warmup_b → eval distribution shift makes it worse

residual_on warmup_b avg_len 159.8 is 22.5 below residual_off 182.3.
But residual_off warmup_b 182.3 is itself 20.7 below the Stage 77a
baseline 203 — there is a baseline degradation in warmup_b vs the
cached number that we have not fully explained (might be run-to-run
variance, might be a difference in tracker convergence speed). The
residual_on degradation in warmup_b is on top of that baseline drop.

By the time we get to eval, both residual_off (169.2) and residual_on
(152.1) are below the cached Stage 77a 180, and the gap between them
is 17.1.

## Honest scoping vs Stage 77a Run 8

The cached "Stage 77a baseline 180" comes from Run 8 (the final
production run). The Stage 77a report itself shows multiple runs
varying 167–180 across configurations. The current 78c residual_off
eval = 169.2 is firmly inside that historical 167–180 band (and
nearly identical to Stage 77a Run 1's 167). So residual_off is not
broken — it's reproducing the lower end of Stage 77a's natural run-to-run
variance.

The 78c contribution is therefore the *delta* between the two
ablations on the same seeds, not the absolute number. Δ_eval = −17.1
on identical seeds is a real signal that residual_on hurts.

## What this rules out and what it leaves open

**Ruled out:**

- The Stage 78c wiring is broken or training loop is wrong (after Bug
  1 fix, 72/72 unit tests pass and the warmup_a result confirms the
  pipeline is functional).
- The MLP residual cannot learn anything useful from real Crafter
  rollouts (warmup_a +43.6 contradicts that).
- The Stage 77a baseline is unreproducible (residual_off matches the
  historical band).

**Not ruled out:**

- A residual with a richer encoding (facing direction, entity
  positions, spatial map state) might handle enemies-on phases
  without entropy collapse.
- A residual with a smaller-magnitude correction (e.g. tighter init,
  lower lr, or output clipping) might preserve action diversity and
  still capture some gap signal.
- A residual trained only on warmup_a-like episodes and then frozen
  might transfer better than continuous online training across phases.
- The Stage 79 surprise accumulator + rule nursery may be the right
  level of abstraction for this problem — it adds new *symbolic
  rules* with their own context conditions, rather than averaging an
  MLP correction over collision-prone fingerprints.

## Connection to Stage 78a methodological gaps

The Stage 78c retro debugging session prompted a re-read of Stage 78a
under the same systematic-debugging lens. Four coverage gaps were
documented in the Stage 78a report addendum: spikerate readout was
never tested as primary, motor → output propagation was never
measured, voltage readout reflects steady-state not dynamics, and
SKS DBSCAN used a single parameter setting. None of those are bugs
in the same way Stage 78c's `planned_step` was — Stage 78a code is
self-contained and correct for what it tested — but together they
mean the "DAF residual dropped" verdict should be read as "parked
on Branch C, re-test cost ~3 GPU-hours, reopens if a passing regime
is found".

This is relevant here because Stage 78c's failure to beat Stage 77a
on enemies-on suggests **the MLP residual at 67K params + this
encoding is not a strong enough learner** to handle the
multi-modal-correction regime that enemies-on Crafter presents. If
Stage 78c had succeeded, the MLP residual would have been the winner
and DAF would have stayed parked. Since it didn't, the DAF re-test
becomes a candidate "Plan B" — if a richer substrate can carry
state×action interactions that the MLP encoding cannot, that might
unlock the eval delta we wanted here.

## Strategy v5 novelty accounting

| Novelty | Status before 78c | Status after 78c |
|---|---|---|
| #1 DAF substrate as residual learner | parked (Branch C) | still parked, **possibly worth re-test** in light of MLP failure |
| #2 No-LLM rule induction from surprise | retained, Stage 79 next | **retained, more important** — the Stage 78c failure suggests symbolic rules with explicit conditions are the right abstraction |
| #3 Three-category ideology (facts / mechanisms / experience) | retained | retained — the 78c residual lives in *experience* without rewriting *facts* or *mechanisms*, which the design correctly implemented |

The Stage 78b unit-PASS / Crafter-pending status is now resolved:
the residual *does* learn (Stage 78b synthetic pass), the wiring is
correct (Stage 78c warmup_a +43.6), but the **encoding-driven
generalisation fails** in enemies-on phases.

## Decision points

Three credible next moves; user input needed.

### Option 1 — Stage 79 surprise accumulator + rule nursery (most aligned with Strategy v5)

Skip residual encoding improvements, accept the 78c finding as
informative-negative, and proceed to Stage 79 as planned. The Stage 79
sketch is already written
(`docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`).
Surprise rule induction adds new *symbolic rules with explicit
preconditions* to `ConceptStore.learned_rules`, which sidesteps the
encoding collision problem by representing "skeleton at range 5 →
damage" as its own discrete rule, not as an averaged correction.

Cost: 2–3 weeks of design + implementation + Crafter eval per the
Strategy v5 plan.

Pro: this is the central novelty of Strategy v5; the 78c failure
*supports* the case for symbolic rule induction over averaged
corrections.

Con: Stage 79 has its own implementation risk and is 2–3 weeks of
work before we know if it closes the wall.

### Option 2 — Stage 78d residual encoding improvement

Add facing direction, last_action, and skeleton distance to the
residual fingerprint. Re-train on the same phases and see if the
warmup_a gain transfers to enemies-on. ~1 week of work, narrow scope.

Pro: directly attacks the diagnosed encoding collision. Cheap and
fast.

Con: still an averaged-correction approach, may just push the
collision boundary instead of removing it. Doesn't address Stage
79's central novelty.

### Option 3 — Stage 78a re-test with spikerate readout (Branch C reopen)

Run R2/R4/R6 with `readout="spikerate"` instead of `voltage`. ~3
GPU-hours on minipc. If any regime beats baseline, DAF residual
reopens as Strategy v5 novelty #1, and the residual approach gets
a substrate that can encode state × action interactions natively
without MLP encoding gaps.

Pro: closes the largest 78a methodological gap; gives DAF residual
a fair shot before final retirement.

Con: branch C research bet, not on the Crafter Gate 1 critical path.
Even if it passes, integrating the substrate into the MPC loop is
its own engineering job.

**Default recommendation:** Option 1. The Stage 78c failure is
*evidence for* the Stage 79 hypothesis (symbolic rules >> averaged
MLP corrections for multi-modal contexts). Stage 79 was already the
next planned stage, and 78c gives it a stronger empirical case.
Option 3 (78a re-test) is worth running in parallel as a low-cost
backup if minipc has spare cycles.

## Files

- Code: `experiments/stage78c_residual_crafter.py` (560 lines)
- Code: `src/snks/agent/concept_store.py` (residual injection +
  `_apply_residual_correction` + module-level helpers)
- Code: `src/snks/agent/mpc_agent.py` (`run_mpc_episode` residual
  params + online SGD with `planned_step` fix)
- Tests: `tests/test_stage78c_residual_integration.py` (11 tests:
  primitive_to_action_idx, simulate_forward residual paths, body
  clamp invariance, planned_step divergence regression, training
  gradient flow)
- Results JSON: `_docs/stage78c_results.json`
- Raw log: `_docs/stage78c_results.txt`
- Checkpoints: `demos/checkpoints/exp138/residual_residual_on_after_*.pt`
  (after_warmup_a / after_warmup_b / after_eval)
- Stage 78b status revision: `docs/reports/stage-78b-report.md`
- Stage 78a methodological gaps: `docs/reports/stage-78a-report.md`
- Crafter-every-stage rule: `feedback_crafter_every_stage.md` in
  auto-memory

## Commit chain

- `2997061` Stage 78c initial implementation (residual injection +
  online SGD + harness + tests + 78b status revision + rule)
- `b871309` Stage 78c launcher fix (`main_fast` / `main_full`
  entrypoints to avoid heredoc quoting issues)
- `faf3ca3` **Bug 1 fix:** `planned_step` propagation in training
  rules-only replay
- `6347824` Stage 78a verdict softening with methodological gaps
- `99e79da` Stage 78c full results data files (from minipc rebase)
- (this commit) Stage 78c report
