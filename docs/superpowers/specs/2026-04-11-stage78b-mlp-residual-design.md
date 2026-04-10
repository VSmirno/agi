# Stage 78b — MLP Residual over ConceptStore Rules (design)

**Date:** 2026-04-11
**Context:** Stage 78a FAIL — see `docs/reports/stage-78a-report.md`. DAF-as-residual novelty dropped. Branch A (primary) is the MLP-residual-over-symbolic-rules pattern from Dreamer-CDP 2026-03 (residual loss), ReDRAW 2025-04 (residual-on-frozen-base), and Neuro-Symbolic Synergy 2026-02 (alternating training). Stage 78b builds the residual learner and validates it on the same synthetic conjunctive task that Stage 78a used as a probe.

## Goal

Build a small neural residual that corrects the error of `ConceptStore.simulate_forward` on per-tick body deltas. The residual sees `(state_embedding, action)` and outputs a 4-dim body delta correction. The rules stay authoritative; the residual is a *correction*, not a replacement.

## Scope vs out-of-scope

**In scope for 78b:**
- `ResidualBodyPredictor` module (state → body_delta correction)
- Synthetic validation test (mirror of Stage 78a): does residual learn the conjunctive rule when the rules don't know it?
- MSE loss on `actual_delta - rules_delta`
- Simple training loop (Adam, shuffle, early stop)
- Checkpointing
- Per-variable MSE breakdown

**Out of scope for 78b (deferred to 79 / 80):**
- Integration with `simulate_forward` in Crafter — that's Stage 80
- Surprise accumulator + rule nursery — that's Stage 79
- Alternating training (rules bump confidence only on rule-matched traces) — Stage 81
- Online training from live rollouts — Stage 80

## Gate

**Stage 78b unit gate:** on the Stage 78a synthetic dataset (1200 train, 200 conjunctive test), with `rules_prediction` fixed to the textbook (`sleep → +0.04 health`, no conjunctive knowledge), the residual must:

1. **Match baseline on conjunctive health:** conj_health_mse ≤ 0.0080 (the Stage 78a linear baseline floor is 0.0072; allow 10% slack for rules vs raw regression framing).
2. **No regression on general health:** gen_health_mse ≤ 0.0120 (baseline 0.0106).
3. **No residual blow-up on easy cases:** for samples where `actual == rules`, residual magnitude should be ≤ 10% of body variance (otherwise residual is memorizing, not correcting).

Gate 1 passing means the *MLP residual over rules* pattern is functional in the simplest possible setting. Gate 2 (Crafter integration) is Stage 80.

## Architecture

```
┌─────────────────────────┐
│  StateActionEmbedding   │ — coarse state fingerprint
│  1-hot visible (9)      │
│  body buckets (40)       │
│  1-hot action (8)       │  → 57-dim vector
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  ResidualMLP            │
│  Linear(57 → 64)        │
│  ReLU                   │
│  Linear(64 → 4)         │  → body delta correction (H, F, D, E)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Combined prediction    │
│  rules_delta            │  ← from ConceptStore (fixed)
│  + residual             │  ← learned
│  = final_delta          │
└─────────────────────────┘
```

**Why small (57→64→4):** the residual should only carry the correction, not re-learn the whole world. Matches ReDRAW's bottleneck principle. Larger residuals risk memorizing the rules the textbook already encodes, which defeats the three-category ideology (facts live in textbook, corrections live in residual).

**Why no action embedding layer:** action space is 8 discrete actions; 1-hot is cheaper and equally expressive than an embedding at this scale.

## Loss

```
rules_pred = rules_predict(state, action)   # 4-dim, no gradient
residual   = ResidualMLP(embed(state, action))   # 4-dim, with gradient
final_pred = rules_pred + residual
target     = actual_body_delta  # 4-dim
loss       = MSE(final_pred, target)
```

This is equivalent to training the residual directly on `actual - rules`:

```
loss = MSE(residual, actual - rules_pred)
```

**Why MSE and not neg-cosine-similarity:** Dreamer-CDP uses neg-cos-sim on 8192-dim latents, where direction is meaningful and magnitude is irrelevant after normalization. For 4-dim body deltas, direction and magnitude are both load-bearing (e.g. `-0.067` vs `-0.04` for health matters), so MSE is the right objective.

**Stop-gradient on rules:** `rules_pred` is a fixed lookup, so stop-gradient happens trivially — no torch.no_grad needed, but no gradients flow back through `rules_predict` regardless.

## Synthetic `rules_predict` for the 78b test

Mirror the Stage 78a synthetic task (same `true_body_delta` oracle for ground truth). For the `rules_predict` function, use the *textbook without conjunctive knowledge*:

```python
def rules_predict(visible: set[str], body: dict[str, float], action: str) -> dict[str, float]:
    """The 'textbook' — all rules the baseline knows, without the conjunctive one."""
    delta = {v: 0.0 for v in BODY_VARS}
    delta["food"] -= 0.04
    delta["drink"] -= 0.04
    delta["energy"] -= 0.02
    if action == "sleep":
        # TEXTBOOK ONLY KNOWS: sleep → recovery. Does NOT know the conjunctive rule.
        delta["energy"] += 0.2
        delta["health"] += 0.04
    if "skeleton" in visible:
        delta["health"] -= 0.4
    if "zombie" in visible:
        delta["health"] -= 0.5
    if action == "do_cow" and "cow" in visible:
        delta["food"] += 5.0
    if action == "do_water" and "water" in visible:
        delta["drink"] += 5.0
    return delta
```

This is identical to `true_body_delta` minus the conjunctive branch. The residual must learn:

- On non-conjunctive samples: `residual ≈ 0` (rules already correct)
- On conjunctive samples: `residual["health"] ≈ -0.107` (= -0.067 actual - +0.04 rules)

That's a sparse, context-dependent correction — exactly the hard case for a linear model to learn from pattern and exactly what a small MLP with a bottleneck should handle (it has enough capacity for the AND gate but not enough to memorize all 1200 samples).

## File plan

1. **New module:** `src/snks/learning/residual_predictor.py`
   - `StateActionEmbedding` class
   - `ResidualBodyPredictor` class (nn.Module)
   - Pure and torch-only, no ConceptStore imports — keeps it testable in isolation.

2. **New experiment:** `experiments/stage78b_residual_synthetic.py`
   - Imports from `stage78a_daf_spike_fair` (dataset generation, `true_body_delta`, `generate_dataset`, `conjunctive_dataset`)
   - Defines a `textbook_rules_predict` that omits the conjunctive rule
   - Trains the residual on `actual - rules_pred`
   - Evaluates on test_general and test_conj
   - Reports gate pass/fail
   - Runs locally on CPU (it's tiny — 1200 samples, 64-hidden MLP) — **exception to the "never run locally" rule because it's a few seconds of CPU compute with no agent loop**, explicitly a unit test of the residual, not an experiment.
   - *Or* runs on minipc via `scripts/minipc-run.sh stage78b "from stage78b_residual_synthetic import main; main()"` — safer default.

3. **New test file:** `tests/learning/test_residual_predictor.py`
   - Shape tests (input/output dims)
   - Gradient flow test (loss backprops to residual only, not rules)
   - Regression test: residual converges on a trivial synthetic task in <100 steps

4. **Stage 78b report:** `docs/reports/stage-78b-report.md` (written after run)

## Why this passes where Stage 78a failed

Stage 78a tested *whether the DAF substrate carries conditional information*.
It didn't — the substrate was the bottleneck.

Stage 78b tests *whether a small MLP on the same static encoding, operating
as a residual over fixed rules, can learn the conditional correction*.
The Stage 78a linear baseline already did this (conj_health_mse=0.0072)
on the encoding alone. The residual variant is strictly easier — it
only has to learn the delta from `+0.04` to `-0.067` on the conjunctive
subset, and `0` on the rest. A 64-hidden MLP trivially has the capacity.

The remaining question for Stage 80 is whether the pattern survives
integration with the actual `simulate_forward` and live Crafter rollouts,
where the distribution shifts and the conjunctive case is rarer. That's
where Stage 79 (surprise accumulator) earns its keep: it notices
conjunctive-like patterns via prediction error *before* the residual
fully converges on them and emits candidate rules to make the residual's
job easier.

## Ideology check

- **Supervised backprop on MSE labels** — this is the one ideology
  wrinkle. The Stage 78a report notes that the supervised MLP readout
  was a diagnostic probe; now we're *actually using* an MLP in the
  agent's prediction loop. Is this a violation of Strategy v5's
  rejection of supervised backprop?
- **Resolution (proposed):** Strategy v5 rejects supervised backprop
  **on labeled loss with ground-truth oracle**. The residual in Stage
  78b is trained on `actual - rules_pred`, where `actual` is observed
  from the environment (not an oracle). This is the *same* intrinsic
  signal as Dreamer-CDP and ReDRAW, just at a lower dimensionality
  (4 body vars instead of 8192 latent). Stage 79's surprise accumulator
  consumes the *same* signal, just with different downstream processing.
  The ideology constraint is: the training signal must come from the
  agent's own observations, not from a human-provided label. Stage 78b
  satisfies this because `actual_body_delta` in the live Crafter loop
  is env.step output, not oracle ground truth.
- For the synthetic test, the "actual" comes from `true_body_delta` —
  which *is* an oracle. This is OK for the unit test: we're validating
  that the mechanism works in a controlled setting. In Stage 80
  (Crafter integration) the oracle is replaced by env observation.

## Effort estimate

- **Residual module:** 100–150 lines
- **Synthetic test harness:** ~200 lines (lots of reuse from 78a spike)
- **Unit tests:** 100–150 lines
- **Report:** writing after run
- **Wall clock:** 2–4 hours focused work (can overlap with Stage 79 work)

## Dependencies

- `torch` (already available)
- No changes to existing modules
- Does not touch `ConceptStore`, `simulate_forward`, `CrafterPixelEnv`
  until Stage 80

## Open questions (for the user when they wake)

1. **Residual hidden dim:** 64 is guessed. Should run a small ablation
   {32, 64, 128, 256} to pick the smallest that passes the gate.
2. **Action encoding:** 1-hot (8-dim) vs embedding (16-dim)? 1-hot is
   simpler and likely sufficient at this scale — start there.
3. **Should Stage 78b gate include a "residual stays small on easy cases"
   metric?** Currently proposed (item 3 of gate) but not universally
   adopted for residual learners. Can drop if it complicates training.
4. **Should we test multi-regime (e.g. different residual capacities)
   like Stage 78a did, or commit to a single config?** Single config
   is faster; multi-regime is more defensible for a report.
   Recommendation: single config (64 hidden), with ablation deferred
   to Stage 80 if Crafter integration wobbles.
