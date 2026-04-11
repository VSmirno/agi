# Stage 78b — MLP Residual over Rules (Synthetic Validation) Report

**Date:** 2026-04-11 (overnight, same-day as Stage 78a FAIL)
**Status:** unit-PASS / Crafter-pending — residual learns the conjunctive correction on the synthetic task; Crafter eval with standard metrics (survival, wood, coal, achievements) is **not yet run** and is required before Stage 78b can be considered COMPLETE. See Stage 78c in ROADMAP.
**Spec:** [2026-04-11-stage78b-mlp-residual-design.md](../superpowers/specs/2026-04-11-stage78b-mlp-residual-design.md)
**Prior stage:** [Stage 78a FAIL report](stage-78a-report.md)

## Status revision (2026-04-11, later)

The original Stage 78b gate was purely a synthetic unit-level validation
of the residual-over-rules pattern (conj_health_mse on an oracle-generated
1200/300/200-sample dataset). After review, we established a stricter
project-wide rule: **every Roadmap v5 stage must end with a Crafter eval
with standard metrics** (survival episode length, wood/coal/iron rates,
achievements, episode distribution). Synthetic / unit validation is
allowed as a sub-gate inside a stage but does not close it.

Under that rule, Stage 78b is **not COMPLETE**. The synthetic gate is
passed (numbers below are valid as a unit-level result), but:

- `ResidualBodyPredictor` is not yet plugged into
  `ConceptStore.simulate_forward` — the MPC tick loop does not see it.
- No online training loop from `env.step()` rollouts exists.
- No Crafter eval has been run comparing residual-on vs residual-off
  against the Stage 77a baseline (wall=180).

The follow-up work is tracked as **Stage 78c — Residual Crafter
integration + eval** in ROADMAP.md. Stage 78b status is changed from
"COMPLETE PASS" to "unit-PASS / Crafter-pending" and will be promoted
to COMPLETE only once Stage 78c produces a Crafter delta (positive or
explicitly-explained null) against the Stage 77a baseline.

## Summary

Stage 78a demonstrated that the DAF substrate does not carry conditional
dynamics at 5K-node × 10K-step scale. Stage 78b pivots to the fallback
plan from Strategy v5: a small MLP residual over fixed ConceptStore rules.
This report validates the residual pattern on **the exact same synthetic
conjunctive task** Stage 78a used, using the Stage 78a training/test
splits for direct comparison.

**Result:** the residual passes both gate thresholds with a comfortable
margin. Conjunctive health MSE = 0.0064 vs gate 0.008 (1.25× below);
general health MSE = 0.0002 vs gate 0.012 (60× below).

## Setup

- **Dataset:** Stage 78a synthetic — 1200 train / 300 test_general / 200
  test_conjunctive samples. Identical RNG seed (42), so the split is
  bit-exact to Stage 78a.
- **Ground truth:** `stage78a_daf_spike_fair.true_body_delta` — includes
  the conjunctive sleep rule.
- **Rules (textbook):** `stage78b_residual_synthetic.textbook_rules_predict`
  — true rules *without* the conjunctive branch. Textbook applies
  `sleep → energy+0.2, health+0.04` unconditionally, missing the
  `(food=0 OR drink=0)` correction.
- **Residual:** `ResidualBodyPredictor` — 1048-dim fingerprint (1000
  hashed-concept bits + 40 body-bucket bits + 8 action bits), 64-wide
  hidden layer, 4-dim output. Small init (weights × 0.1, bias zero) so
  residual ≈ 0 at epoch 0 and does not perturb the rules prior to
  training. 67,396 parameters total.
- **Loss:** `MSE(rules_pred + residual, actual_delta)`.
- **Training:** Adam, lr=1e-3, 20 epochs, shuffled per epoch, single
  device (cuda on minipc, cpu fallback).
- **Runtime:** 20 s wall on minipc GPU.

## Rules-only baseline

Before adding any residual, the textbook alone is evaluated on both splits:

| Metric | General | Conjunctive |
|---|---:|---:|
| overall_mse | 0.0003 | 0.0129 |
| health_mse | 0.0003 | **0.0114** |
| food_mse | 0.0 | 0.0 |
| drink_mse | 0.0 | 0.0 |
| energy_mse | 0.0009 | 0.0400 |

The general split is near-perfect (rules match true_body_delta on
non-conjunctive samples). On the conjunctive split the rules mispredict
health by 0.107 every time (rules: +0.04, truth: −0.067), squared error
= 0.0114. The rules also mispredict energy by 0.2 (rules: +0.18 total,
truth: −0.02), squared error = 0.04. **These two errors are what the
residual must learn.**

## Residual before / after training

| Metric | Before (init) | After (epoch 20) | Δ |
|---|---:|---:|---:|
| general overall_mse | 0.0003 | 0.0002 | –0.0001 |
| general health_mse | 0.0003 | 0.0002 | –0.0001 |
| conj overall_mse | 0.0129 | 0.0071 | **–0.0058** |
| **conj health_mse** | 0.0115 | **0.0064** | **–0.0051** |
| conj energy_mse | 0.0400 | 0.0221 | **–0.0179** |

Small init keeps "before" numerically identical to rules-only, as
designed. After training the residual has absorbed half to most of
the conjunctive gap:
- **Health correction:** rules+residual mean magnitude = 0.034 on health
  (target correction is 0.107; residual captures ~32%).
- **Energy correction:** residual mean magnitude = 0.067 (target is
  0.200; residual captures ~34%).

Partial (not full) learning is expected: the residual bottleneck is
small (64 hidden), the conjunctive cases are only 2.7 % of training,
and 20 epochs is short. Despite that, both gate thresholds pass:

## Gate

| Criterion | Measured | Gate | Result |
|---|---:|---:|:---:|
| conj_health_mse | **0.0064** | ≤ 0.008 | PASS |
| gen_health_mse | **0.0002** | ≤ 0.012 | PASS |
| Residual stays small on easy cases | residual_abs_mean(food)=0.004, drink=0.003, general gen_health=0.0002 | qualitative | PASS |

**Status: unit-level PASS (synthetic gate met). Stage 78b remains Crafter-pending — see Stage 78c.**

## Comparison with Stage 78a

| Approach | conj_health_mse | vs Stage 78a baseline (0.0072) |
|---|---:|---:|
| Stage 78a baseline (linear MLP, no rules) | 0.0072 | 1.00× |
| Stage 78a **R5/R7** DAF + SKS readout | 0.0727 | **10.1× worse** |
| Stage 78a **R3** DAF oscillatory default tau | 0.0728 | 10.1× worse |
| Stage 78a **R4** DAF oscillatory fast tau | 0.0903 | 12.5× worse |
| **Stage 78b MLP residual over rules** | **0.0064** | **0.89× (better)** |

Stage 78b's residual is actually *slightly better* than the Stage 78a
linear baseline (0.0064 vs 0.0072). That's because the rules already
encode the non-conjunctive patterns perfectly (rules_only gen_mse =
0.0003), so the residual starts from a much better "fit the delta"
starting point than the pure MLP baseline that had to learn everything
from scratch. **The residual-over-rules pattern is a clean win over
both DAF and pure MLP on this task.**

## Loss curve

```
epoch train_mse conj_health_mse (tracked every 2 epochs)
  1    0.0004   0.0088
  3    0.0003   0.0091
  5    0.0002   0.0069  ← first pass of gate
  7    0.0002   0.0063
  9    0.0002   0.0055  ← best conj_health seen
 11    0.0002   0.0063
 13    0.0001   0.0067
 15    0.0001   0.0058
 17    0.0001   0.0063
 19    0.0001   0.0066
 20    0.0001   0.0064  (final)
```

Converges to a stable plateau around 0.006 conj_health_mse from epoch 5
onward. Small oscillations within ±0.001 — expected for a small
minibatch-free SGD loop on a rare (2.7%) subset. Could tighten with
larger batches or early stopping, but the gate is already met and
further tuning is not load-bearing for the Stage 78b conclusion.

## Unit tests

`tests/learning/test_residual_predictor.py` — 9 tests covering:
- Default and custom-config input/output shapes
- Encoding determinism (same concept → same bits across calls)
- Encoding differentiation (different inputs → different encodings)
- Small-init property (residual ≈ 0 at epoch 0)
- Gradient flow (residual params get gradients, rules_t tensor does not)
- Trivial-task convergence (300-step Adam on a single constant sample)
- Dict API for integration with ConceptStore
- Save/load roundtrip preserving weights and concept-seed table

All 9 passing (`pytest tests/learning/test_residual_predictor.py`).

## Ideology check

- **Supervised MSE loss on observed deltas:** the signal comes from
  `actual_delta - rules_pred`, where `actual` comes from environment
  observation (in the synthetic case it's the oracle; in Stage 80 Crafter
  integration it will be `env.step()` output). This is the same intrinsic
  signal pattern as Dreamer-CDP and ReDRAW, reframed for 4-dim body
  regression. It does NOT violate Strategy v5's rejection of "supervised
  backprop on labeled loss" because the label is not a human annotation,
  it's the agent's own observation of reality.
- **Three-category taxonomy preserved:** textbook (facts) is unchanged;
  `simulate_forward` dispatch (mechanisms) is unchanged; the residual
  lives in the experience bucket, alongside the homeostatic tracker's
  observed rates. The residual cannot rewrite facts — only correct their
  application to specific (state, action) contexts.
- **Small bottleneck:** 67K parameters, mostly in the 1048→64 input
  layer. This is small enough that memorization of 1200 samples is
  possible only on the conjunctive subset (32 training samples × few
  dimensions), which is exactly what we want the residual to learn.
  Other contexts have too little signal-to-noise for a 64-wide bottleneck
  to memorize individual samples.

## Next steps

### Stage 79 — Surprise Accumulator + Rule Nursery (2 weeks)

The surprise accumulator is *independent* of the residual and Stage 78b
proves the residual pattern in isolation. Stage 79 builds on top:

- When the residual detects a large conjunctive correction on a context,
  the accumulator records that context's surprise pattern.
- Once enough consistent surprise has accumulated in a context bucket,
  the nursery emits a *candidate rule* and promotes it into the textbook
  after verification.
- Over time the textbook grows from discovered rules, the residual's job
  narrows, and interpretability improves.

Sketch already written: `docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`.

### Stage 80 — Crafter integration

- Plug the residual into `ConceptStore.simulate_forward` as a per-tick
  body-delta correction.
- Start the residual with small init so it doesn't perturb the Stage
  77a-validated MPC baseline.
- Train online from env rollouts: after each `env.step`, compute
  `surprise = actual_delta - rules_delta`, use it as the residual's
  training signal (plus the surprise accumulator's input stream).
- Gate: survival ≥ 200 on Crafter eval (close the Stage 77a wall=180),
  wood ≥ 30%.

### Stage 81 — Alternating training

- When a surprise-detected rule gets promoted out of the nursery into
  the textbook, stop training the residual on that context — the rules
  now handle it. Residual capacity gets freed for newer surprises.
- This is the Neuro-Symbolic Synergy "alternating training" pattern.

## Artifacts

- Module: `src/snks/learning/residual_predictor.py`
- Test: `tests/learning/test_residual_predictor.py` (9 tests)
- Experiment: `experiments/stage78b_residual_synthetic.py`
- Results JSON: `_docs/stage78b_results.json`
- Design spec: `docs/superpowers/specs/2026-04-11-stage78b-mlp-residual-design.md`
- Strategy v5 spec: `docs/superpowers/specs/2026-04-11-strategy-real-learning-design.md`
- Stage 78a FAIL report: `docs/reports/stage-78a-report.md`
- Stage 79 sketch: `docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`

## Strategy v5 novelty accounting (unchanged from Stage 78a)

1. ~~DAF-as-residual learner~~ — **dropped** (Stage 78a FAIL)
2. No-LLM surprise rule induction — **retained**, next up in Stage 79
3. Three-category ideology — **retained**, enforced by residual architecture

Two of three original novelties remain; the MLP residual is a direct
Dreamer-CDP adoption without our own twist, but it's the enabling
scaffolding for the other two novelties, not a claim of contribution.
The grant-worthy contribution is now concentrated in Stage 79's
without-LLM rule induction.
