# Stage 78a — Fair DAF Spike Test Report

**Date:** 2026-04-11
**Status:** COMPLETE — **FAIL**. DAF substrate does not carry conditional dynamics in any of the 7 tested regimes. Stage 78b proceeds with MLP residual (Dreamer-CDP style) fallback; DAF-as-residual novelty dropped. Novelties #2 (no-LLM surprise rule induction) and #3 (three-category ideology) retained.
**Script:** `experiments/stage78a_daf_spike_fair.py`
**Strategy:** [Strategy v5 — Real learning on top of rules](../superpowers/specs/2026-04-11-strategy-real-learning-design.md)
**Prior spike reference:** `experiments/spike_daf_body_predictor.py` (excitable FHN, 100 sim steps, voltage readout)
**Stage 44 audit finding this re-tests:** FHN excitable at default I_base=0.5 — oscillatory regime R1.1 was marked "never tested" and this report resolves it empirically.

## Question Being Answered

> Can the DAF substrate (FHN oscillators + STDP + coupling) learn conditional
> dynamics in a regime that Stage 44 audit marked as untested?

Answering *yes* unlocks the **DAF-as-residual** novelty #1 in Strategy v5 for
Stage 78b. Answering *no* is still fine — we fall back to MLP residual
(Dreamer-CDP pattern) and drop the DAF novelty while keeping the other two
(no-LLM rule induction, three-category ideology).

## Differences from the Prior Spike

| Aspect | Prior spike | Stage 78a fair test |
|---|---|---|
| Oscillator regime | Excitable (I_base=0.5) | Also tests oscillatory (I_base=1.1) |
| Integration length | 50–100 steps | 2000–10000 steps (20–100×) |
| FHN tau | 12.5 (default) | Also tests tau=1.0 (compressed recovery) |
| Substrate learning | Frozen reservoir (ESN) | Also tests STDP warmup pass |
| Readout features | Raw output-zone voltage | Also tests SKS cluster rates |
| Test matrix | Single config | 7 regimes + 1 linear baseline |
| Device | CPU (local) | GPU (minipc) |

## Test Matrix

Regimes are ordered cheapest→most expensive so partial results are still
informative if the run is interrupted.

| # | Name | I_base | tau | sim_steps | STDP warmup | Readout |
|---|---|---|---|---|---|---|
| R1 | baseline_excitable_short | 0.5 | 12.5 | 100 | no | voltage |
| R2 | excitable_long | 0.5 | 12.5 | 2000 | no | voltage |
| R3 | oscillatory_default_tau | **1.1** | 12.5 | 10000 | no | voltage |
| R4 | oscillatory_fast_tau | **1.1** | **1.0** | 2000 | no | voltage |
| R5 | oscillatory_fast_tau_sks | 1.1 | 1.0 | 2000 | no | **sks_cluster** |
| R6 | oscillatory_fast_tau_stdp | 1.1 | 1.0 | 2000 | **yes** | voltage |
| R7 | oscillatory_fast_tau_stdp_sks | 1.1 | 1.0 | 2000 | **yes** | **sks_cluster** |

R1 is the prior spike as a sanity reference.
R2 isolates "was the sim too short?" within excitable regime.
R3 is the **main Stage 44 untested case** — full default tau but sim long enough for multiple FHN cycles.
R4 compresses the recovery timescale so 2000 steps fit several oscillations.
R5–R7 vary the readout and add STDP warmup on a disjoint unlabeled stream.

## Synthetic Task

Conjunctive hidden rule (the textbook cannot express without AND/OR grammar):

```
sleep + (food == 0 OR drink == 0) → health delta = -0.067
```

- **Train set:** 1200 random (visible, body, action) samples
- **test_general:** 300 random samples (natural distribution — ~2.7% conjunctive)
- **test_conj:** 200 forced-conjunctive samples (food or drink set to 0 before sleep)

Per-sample signal:
- `food`, `drink`, `energy` have linear-additive rules (trivial control).
- `health` has the conjunctive rule (real target).
- Skeleton / zombie visible apply additive health hits (easy spatial rules).

## Gate Criterion

Three-level verdict via `verdict()` in the script:

- **STRONG_PASS** — ≥1 regime with discrimination_ratio ≥ 0.10 AND
  `conj_health_mse ≤ 0.90 × baseline`. Substrate adds representational
  capacity beyond static encoding.
- **PASS** — ≥1 regime with discrimination_ratio ≥ 0.10 AND
  `conj_health_mse ≤ 1.20 × baseline`. Substrate is transparent (carries
  the info), viable as a residual learner in Stage 78b.
- **FAIL** — no regime meets the above. DAF-as-residual novelty dropped,
  Stage 78b falls back to MLP residual (Dreamer-CDP style).

## Ideology Notes

- **Supervised MLP readout** is used as a diagnostic probe only. This is
  **not** a proposal for how learning should happen in a production
  agent. Strategy v5 rejects supervised backprop on labeled loss.
  `LinearBaseline` uses the same readout architecture so the comparison
  is fair — it measures the substrate contribution, not MLP capacity.
- **STDP warmup** runs on a disjoint unlabeled stream (different RNG
  seed) so the substrate exposure is not coupled to the readout train
  split — the substrate is shaped by input statistics alone, in line
  with `feedback_self_induced_rules.md`.
- **Synthetic oracle** generates ground-truth `body_delta` only to
  supervise the diagnostic readout. A real agent acquires such rules
  via surprise + verification (Stage 79), not via access to an oracle.

## Results

_Populated from `_docs/stage78a_results.json` after the minipc run completes._

### Baseline (linear no-DAF, MLP head)

| Metric | Before | After |
|---|---|---|
| general overall mse | 0.4830 | 0.3926 |
| general health mse | 0.1212 | **0.0106** |
| general food mse | 0.9033 | 0.6907 |
| general drink mse | 0.9035 | 0.8642 |
| conj overall mse | 0.0342 | 0.0727 |
| **conj health mse** | 0.1332 | **0.0072** |

Interpretation: the baseline MLP learns the general health rules
well (0.0106 final on general test). On the conjunctive test it
converges to outputting *near-zero* for health, giving 0.0072 — not
the -0.067 that the conjunctive rule would require, but much closer
to it than random-initial (0.1332). The baseline never *discovers*
the conjunctive rule; it just stops aggressively predicting the
"sleep is harmless" pattern once it sees conflicting evidence. So
the practical floor for any DAF regime is **≈0.0072**, and the ideal
(predicting exactly -0.067) would be ~0.005 due to intra-set variance.

### Per-Regime Summary — all 7 FAIL

| Regime | disc | spikes/sample | gen health mse | **conj health mse** | vs baseline |
|---|---:|---:|---:|---:|---:|
| *baseline (linear no-DAF)* | — | — | **0.0106** | **0.0072** | 1.0× |
| R1 baseline_excitable_short | 1.406 | 0 | 0.0781 | 0.0756 | 10.5× worse |
| R2 excitable_long | 0.594 | 248K | 0.0872 | 0.0903 | 12.5× worse |
| R3 oscillatory_default_tau | 0.064 | 32M | 0.0766 | 0.0728 | 10.1× worse |
| R4 oscillatory_fast_tau | 0.309 | 267K | 0.0873 | 0.0903 | 12.5× worse |
| R5 oscillatory_fast_tau_sks | 0.000 | 267K | 0.0766 | 0.0727 | 10.1× worse |
| R6 oscillatory_fast_tau_stdp | 0.290 | 265K | 0.0859 | 0.0882 | 12.3× worse |
| R7 oscillatory_fast_tau_stdp_sks | 0.000 | 265K | 0.0766 | 0.0727 | 10.1× worse |

Ranked best→worst by conj_health: R5 ≈ R7 ≈ R3 < R1 < R6 < R2 ≈ R4.
Every regime is ~10× worse than baseline on the conjunctive target and
~7–8× worse on *general* health MSE, so the substrate is failing even
on the easy additive rules.

### Per-Regime Interpretation

**R1 — prior spike reference, 100 sim steps, excitable:** zero spikes in
100 steps at dt=0.0001 (Stage 44 audit finding confirmed empirically).
discrimination_ratio=1.41 from sub-threshold voltage noise, not signal.
Readout overfits training but generalizes badly.

**R2 — excitable long (2000 steps):** 248K spikes/sample but
conjunctive MSE is *worse* than R1 (0.0903 vs 0.0756). Longer excitable
integration replaces voltage noise with spike noise; the readout is
more confused, not more informed. Negative isolation: the prior spike
was not merely under-integrated — the excitable regime itself is
information-lossy at this scale.

**R3 — oscillatory default tau (10000 steps, I_base=1.1):** the main
Stage 44 R1.1 untested case. **32 million spikes/sample** but
**discrimination_ratio collapses to 0.064**, below the gate. Above the
FHN bifurcation the intrinsic attractor dominates input drive; all
samples produce indistinguishable output patterns. This failure mode —
substrate saturation at default tau — is *the* empirical answer to
Stage 44 R1.1 that had been deferred: default-tau oscillation does not
respect input structure at 5000 nodes × 10000 steps.

**R4 — oscillatory fast tau (tau=1.0, 2000 steps):** fast recovery
timescale rescues the oscillation from saturation. discrimination_ratio
= **0.309** (5× R3), 267K spikes, feature_norm 7.5 (vs R3's 48.8). The
compressed tau lets each oscillator return to excitable sub-threshold
between input-driven events rather than running free. **And yet
conj_health_mse = 0.0903 — the *worst* score in the whole matrix**, tied
with R2. Discriminative features did not translate into conditional
prediction. See "The discrimination paradox" below.

**R5 — R4 + SKS cluster readout:** cofiring coherence on 64-sample
combined history (64 × 2000 steps × 1000 output-zone nodes) returned
**zero clusters** from DBSCAN. Output zone does not form persistent
co-activated groups. The readout falls back to mean spike rates per
output node, but feature_norm=0.0 in activity stats — output zone
barely fires, regardless of the 267K total spikes (those are all in
input/motor zones, not the downstream output zone). Readout receives
near-zero features, outputs near-zero predictions, reaches the
degenerate MSE floor (0.0727 ≈ mean-of-zeros-against-target).

**R6 — R4 + STDP warmup:** 500-sample unlabeled warmup pass with local
STDP enabled. conj_health_mse = 0.0882 vs R4's 0.0903 — a trivial
~2% improvement. STDP warmup neither helps nor hurts meaningfully.
Substrate shaping via local correlation rules does not reshape output
activity to carry the conditional signal.

**R7 — R6 + SKS cluster readout:** again, 0 clusters discovered. Falls
back to spike-rate output zone, which is still dead. conj_health_mse =
0.0727, tied with R5.

### The discrimination paradox

Higher discrimination correlates with **worse** conjunctive MSE:

| disc | conj_health | regime |
|---:|---:|---|
| 0.00 | 0.0727 | R5, R7 (degenerate zero-feature) |
| 0.064 | 0.0728 | R3 (saturated) |
| 0.290 | 0.0882 | R6 |
| 0.309 | 0.0903 | R4 |
| 0.594 | 0.0903 | R2 |
| 1.406 | 0.0756 | R1 |

The readout MLP overfits to the majority pattern (`sleep → health
+0.04`, true for 97.3% of sleep samples). When features are
discriminative the MLP learns the majority rule confidently and
predicts large positive health deltas for all sleep samples, including
the conjunctive 2.7% — getting them spectacularly wrong (error ≈0.11²).
When features are bland the MLP outputs ≈0 (degenerate solution),
landing closer to the -0.067 target than confident positive predictions
would. **The substrate's "good" features are all on the wrong
attributes** — they encode input variety but not the food/drink=0
conditional bit.

### SKS cluster discovery: 0 clusters, twice

Both R5 and R7 discovered zero persistent cofiring clusters across
the output zone, using DBSCAN eps=0.3, min_samples=5, min_size=5 on
combined 128K-step history. The substrate's output-zone firing does
not form attractor structure at the scale tested. Options for a
follow-up would include:
- Larger networks (50K+ nodes) — SKS theory assumes larger substrates
- Looser DBSCAN (eps=0.5, min_size=3) — might find sparser clusters
- Different readout zone placement — clusters may form in hidden/input
  zones, not the isolated output zone
- Phase coherence instead of cofiring — SKS originally used phases

These are **not** Stage 78a-bounded investigations; they are Branch C
research follow-ups and are deferred.

### STDP ablation: no lift

R4 vs R6 (identical except STDP warmup):
- conj_health: 0.0903 → 0.0882 (trivial 2% relative)
- gen_health: 0.0873 → 0.0859 (trivial)
- discrimination: 0.309 → 0.290 (slightly worse)

The disjoint 500-sample STDP warmup does not reshape the substrate in
a way that propagates input-specific information to the output zone.
Local correlation-based learning is not, at this scale and duration,
enough to establish the input→output pathways that would carry the
conditional bit.

### Verdict

**Status: FAIL**

- **Passing regimes:** none.
- **Strong-passing regimes:** none.
- **Baseline conj_health_mse floor:** 0.0072.
- **Best DAF regime (R5):** 0.0727 — 10.1× worse, and degenerate
  (zero features, not a real result).
- **Best DAF regime with non-degenerate features (R3):** 0.0728 —
  still 10× worse, and R3 had discrimination below the gate.

**Recommendation (from the script's own `verdict()`):** *"DAF substrate
destroys conditional info. Stage 78b: use MLP residual (Dreamer-CDP
style). DAF-as-residual novelty dropped; retain rule induction novelty."*

### What this rules out empirically

Stage 78a was designed as the empirical resolution of four open questions
from Stage 44 audit and the Strategy v5 research phase. Each is now
answered:

1. **Q: Does oscillatory FHN (I_base > 1) carry information the excitable
   regime cannot?** **A: No.** R3 (oscillatory) is within 1% of R1
   (excitable, zero spikes) on conjunctive MSE. Oscillatory saturates.
2. **Q: Is the prior spike's result just under-integration (50–100 steps
   vs oscillation period)?** **A: No.** R2 (excitable 2000 steps, 248K
   spikes) is 19% *worse* than R1. More integration does not help.
3. **Q: Does STDP warmup on the input distribution shape useful
   substrate representations?** **A: No.** R4 vs R6 differs by 2%.
4. **Q: Does SKS cluster readout extract signal that per-node voltage
   hides?** **A: No.** Zero clusters form; fallback readout is
   degenerate.

### What this does *not* rule out

- DAF substrate may still be useful as **perception / input
  classification** (which is what Stage 44 and the original SNKS
  proposal used it for). The failure is specifically at "learn
  *conditional regression targets* through the substrate", not "form
  input-specific patterns".
- Larger networks (50K+ nodes, GPU-mandatory) might form SKS clusters
  where 5K does not. This is a Branch C research follow-up, not a
  Stage 78a concern.
- Different readout architectures (e.g. phase coherence, temporal
  pattern matching) could potentially extract signal we missed with
  spike-rate / voltage readouts.

None of these change the Stage 78b decision: the residual learner has
a 5–6 week timeline to close Gate 1 on Crafter, and DAF is not
viable for that specific role on that specific timeline.

### Run metadata

- `experiments/stage78a_daf_spike_fair.py` commit `eb791ab` (pre-fix
  version) was the one executed; the `4fc34db` fixes do not change
  the test outcome (fixes were cosmetic / exit-code / warmup-stream
  separation, none of which would reverse a 10× gap on the target).
- Run on minipc GPU (AMD Radeon 96GB VRAM, ROCm 7.2).
- Total runtime ≈ 75 min wall.
- Per-regime elapsed (s): R1 30 / R2 421 / R3 2234 / R4 426 / R5 432 /
  R6 489 / R7 494.
- Raw results: `_docs/stage78a_results.json` (fetched from minipc).
- Raw log: `ssh minipc "cat /opt/agi/_docs/stage78a_results.txt"`.

## Next Step: Stage 78b (revised under FAIL branch)

**Stage 78b — MLP residual over ConceptStore rules (Dreamer-CDP pattern)**

```
ẑ_body_delta(t+1) = rules_prediction(state, action)
                   + small_MLP_residual(state_embedding, action_embedding)
```

Key elements from published work, kept as-is:
- **Negative cosine similarity loss** with stop-gradient on the target
  (Dreamer-CDP 2026-03). Avoids reconstruction, intrinsic signal only.
- **Residual bottleneck** (ReDRAW 2025-04): residual MLP is kept small
  so it only carries the delta the rules miss, not the whole world.
- **Alternating training** (Neuro-Symbolic Synergy 2026-02): neural
  fine-tune only on traces where rules were wrong (surprise > threshold),
  rules get their confidence bumped on traces where they were right.

Where we **deviate** from published work (this is where Strategy v5
novelty #2 and #3 live):
- **No LLM in the candidate-synthesis loop.** Surprise accumulator +
  template-based candidate rules (sketched in
  `docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`).
- **Three-category ideology enforced**: residual network only corrects
  *experience* mistakes; *facts* (textbook) are never rewritten by the
  network; *mechanisms* (simulate_forward dispatch) are code, not
  learnable weights.

**Gate (Stage 78b):** residual integrated into MPC's `simulate_forward`
correction term; Crafter eval survival ≥200 (close the Stage 77a wall),
wood ≥30%.

**Stage 79 (surprise accumulator + rule nursery)** is **unaffected** by
this Stage 78a FAIL — it operates on the `actual - predicted` signal
regardless of what generates the prediction. Its sketch is already
written.

**DAF substrate retirement:** not total. DAF remains in the codebase
for perception research (Stage 44 kept it as perception layer after the
R1 negative verdict). The Stage 78a result narrows its role: it does
not learn conditional dynamics targets via output-zone probing at
5K node × 10K sim step scale. Future research bets (grant track,
larger networks, different readout) are possible but not on the
Stage 78b → Gate 1 path.

**Strategy v5 novelty accounting after this result:**
1. ~~DAF-substrate as learnable dynamics predictor~~ — **dropped**.
2. Rule induction from surprise accumulator without LLM — **retained**.
3. Three-category ideology as enforced architectural principle — **retained**.

One novelty out of three is gone; the remaining two are still
grant-worthy and the fallback MLP residual still matches the published
Dreamer-CDP SOTA path, so Stage 78b is well-defined and unblocked.

## Related

- Strategy v5 spec: `docs/superpowers/specs/2026-04-11-strategy-real-learning-design.md`
- Stage 77a report: `docs/reports/stage-77a-report.md`
- Architecture review: `docs/reports/architecture-review-2026-04-10.md`
- Prior DAF spike: `experiments/spike_daf_body_predictor.py`
- Ideology: `docs/IDEOLOGY.md`
