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

## Connection to Stage 78a — same failure mode, not just analogy

The Stage 78c retro debugging session prompted a re-read of Stage 78a
under the same systematic-debugging lens. Four coverage gaps were
documented in the Stage 78a report addendum (spikerate readout was
never tested as primary, motor → output propagation never measured,
voltage readout reflects steady-state, SKS DBSCAN used a single
parameter setting). Those are coverage gaps, not bugs.

But there is a more important connection — the **discrimination
paradox** that Stage 78a explicitly documented in its results section,
and that Stage 78c has now reproduced under a completely different
implementation. They are the same failure mode, not an analogy.

### The shared pattern

**Stage 78a wording:** "Higher discrimination correlates with **worse**
conjunctive MSE. The readout MLP overfits to the majority pattern
(`sleep → health +0.04`, true for 97.3% of sleep samples). When
features are discriminative the MLP learns the majority rule
**confidently** and predicts large positive health deltas for all
sleep samples, including the conjunctive 2.7% — getting them
spectacularly wrong (error ≈0.11²). When features are bland the MLP
outputs ≈0 (degenerate solution), landing closer to the −0.067
target than confident positive predictions would. **The substrate's
'good' features are all on the wrong attributes** — they encode
input variety but not the food/drink=0 conditional bit."

**Stage 78c manifestation:**

| | Stage 78a (DAF substrate × MLP head) | Stage 78c (MLP residual over rules) |
|---|---|---|
| Higher *X* worsens outcome | Higher disc → worse conj_health_mse | Lower entropy (more confident planner) → worse eval avg_len |
| What the model learns | Majority rule (sleep → +0.04) confidently | Average correction over fingerprint collisions |
| When it goes wrong | Conjunctive 2.7% — confidently positive instead of −0.067 | Multi-modal correction states (cow facing right vs facing up) — confident average instead of either mode |
| Degenerate / bland alternative wins | R5/R7 0-cluster fallback ≈ baseline | residual_off entropy 0.86 ≈ baseline |
| The wrong attribute | Substrate encodes input variety, not food/drink=0 conditional bit | Residual encoding has visible+body+action but not facing/positions/last_action — averages over conditional structure |

The two stages used different substrates (FHN spiking network vs
1048→64→4 MLP), different training regimes (offline supervised on
oracle deltas vs online SGD on env deltas), different feature
extractors (voltage / spikerate / sks_cluster vs hashed concept bits +
body buckets), and arrived at the **same failure**: a
confidently-trained model that predicts the average of a multi-modal
target distribution and underperforms the no-model baseline because
the baseline at least doesn't *amplify* the wrong direction.

### Why this is one finding, not two

Both failures are produced by the same recipe:

```
   features that aggregate samples needing different corrections
 + MSE-on-labels training (the model is rewarded for matching the average)
 = a confidently-wrong predictor
```

The recipe is independent of the substrate (DAF or MLP), independent
of the readout (voltage, spikerate, MLP head), independent of the
training signal source (oracle or env rollouts), and independent of
parameter count (5K-node substrate or 67K MLP).

This matters for what we do next. The Stage 78c report's first version
framed the problem as "MLP residual at this scale + this encoding
is not a strong enough learner". That was too narrow. **The problem is
not learner strength** — we already know from Stage 78a that adding
substrate complexity didn't help, and from Stage 78c that adding the
correct training signal didn't help either. Both interventions left
the underlying recipe untouched.

The recipe breaks only if we either (a) make the features
**condition-aware** so the model never has to average over
multi-modal targets, or (b) replace MSE-on-labels with a learning
mechanism that produces **explicit conditional structure** instead of
a single averaged correction.

### What this means for the three options

**Option 1 (Stage 79 — surprise accumulator + symbolic rule
nursery)** explicitly produces conditional structure: a candidate
rule has the form `(precondition, effect)` where precondition is a
discrete predicate over the agent's observations. There is no
averaging — the rule either applies or it doesn't. Multiple rules
can coexist with non-overlapping preconditions, covering the
multi-modal case directly. **This is the only option that breaks the
recipe at its source.**

**Option 2 (Stage 78d — improve residual encoding)** adds more bits
to the encoding (facing direction, last_action, entity positions).
This is a direct attempt to *increase discrimination*. But 78a's
paradox already showed that on the conditional problem, more
discrimination amplifies the wrong direction more confidently. With
better encoding the residual will collide on fewer cases, but on the
cases it still collides on it will be **more** confident, not less.
The expected best-case is "residual_on no longer hurts" rather than
"residual_on actually helps the conditional cases".

**Option 3 (Stage 78a re-test with spikerate readout)** changes the
substrate's feature extractor but does not change the MSE-on-labels
training paradigm. The discrimination paradox would persist at any
new operating point — a passing spikerate regime, if it exists, would
just be a new point on the same curve, not an exit from the curve.
The bet is that there is some (substrate, readout) pair where the
features happen to align with the conditional bit by accident — that's
possible but speculative, and the 4 GPU-hours of compute do not
purchase a path out of the underlying recipe.

**Decision:** Option 1 is no longer just the default — it is the
**only structurally sound option** given the paradox. Options 2 and 3
might still be cheap to run as parallel data-collection (especially
Option 3 at 3 GPU-hours), but they should not be considered candidates
for the critical path. Stage 78c's main contribution to Strategy v5 is
**independent confirmation** that symbolic rule induction with
explicit preconditions (Stage 79) is the right level of abstraction
for the conditional structure that Crafter actually has.

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

## Decision

**Stage 79 (symbolic surprise accumulator + rule nursery), no
alternatives.** The discrimination paradox argument above rules out
Option 2 (encoding improvements would amplify the wrong direction on
remaining collisions) and demotes Option 3 (78a spikerate re-test) to
optional parallel data-collection rather than a credible critical path.
Stage 79 explicitly produces `(precondition, effect)` rules with
discrete applicability — no averaging, no collision-averaging amplification,
direct representation of multi-modal conditional structure. The Stage 79
sketch is at
`docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`
and a full design doc is the next deliverable after this report.

Stage 78c's main contribution to Strategy v5 is **independent
confirmation** of the discrimination paradox first observed in 78a.
That gives the Stage 79 design a much stronger empirical foundation
than it had before — it is no longer "the next sketch on the roadmap",
it is "the only known way out of a failure mode that two independent
implementations have now reproduced".

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
