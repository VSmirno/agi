# Stage 79 — Surprise Accumulator + Rule Nursery Report

**Date:** 2026-04-11
**Status:** **COMPLETE — partial PASS** (infrastructure validated, Bug 2 fixed in baseline, nursery+cap reaches neutral on Crafter, no wall break)
**Spec:** [`docs/superpowers/specs/2026-04-11-stage79-rule-nursery-design.md`](../superpowers/specs/2026-04-11-stage79-rule-nursery-design.md)
**Prior stages:** [Stage 78c](stage-78c-report.md), [Stage 78a](stage-78a-report.md)

## Headline

The Stage 79 nursery infrastructure works (synthetic falsification test
passes; integration into MPC loop is wired and stable; Bug 2 in
`expand_to_primitive` was found and fixed). On real Crafter the nursery
in its initial form **degraded** baseline by ~23 steps in eval (same
failure mode as Stage 78c residual). Adding a `MAX_ABS_EFFECT=0.5`
heuristic that drops "textbook-cancellation" rules brought the agent
back to **neutral** vs the baseline (+0.4 in eval). The wall (Stage 77a
~180 → target ~200) was not broken.

The session also surfaced two real bugs that materially shifted the
baseline interpretation:

- **Bug 1** (Stage 78c training rules-only replay dropped `planned_step`)
  was fixed earlier in the session. Already in `stage-78c-report.md`.
- **Bug 2** (`expand_to_primitive` 'do' branch ignored facing direction)
  was discovered during Stage 79 analysis. The fix lifted the residual_off
  / nursery_off baseline from ~169 to ~180 in eval, **matching the
  original Stage 77a Run 8 number rather than the cached 78c result**.

## Run timeline (this session)

| Run | nursery_off / residual_off eval | nursery_on / residual_on eval | Δ on-off | Notes |
|---|---:|---:|---:|---|
| Stage 77a Run 8 (cached, May 2026) | — | 180 | — | original baseline (eventually shown to be Bug-2-corrupted) |
| Stage 78c v1 (Bug 1+2 latent) | 169.2 | 152.1 | -17.1 | Bug 1 found post-hoc via systematic-debugging |
| Stage 78c v2 (Bug 2 fixed) | **180.4** | 154.5 | -26.0 | residual_off lifted, residual_on unchanged → broken approach |
| Stage 79 v1 (Bug 2 latent) | 170.8 | 155.5 | -15.4 | nursery wired correctly, but inherits Bug 2 |
| Stage 79 v2 (Bug 2 fixed) | **184.6** | 161.4 | -23.2 | nursery_off lifted, nursery_on still hurts; 151 rules promoted |
| **Stage 79 v3 (Bug 2 + cap)** | 177.1 | **177.4** | **+0.4** | **first run where nursery_on ≥ nursery_off**; 137 rules |

The "off" baseline in v2 / v3 sits in the band ~177-184, consistent with
Stage 77a's historical run-to-run variance (167-180). The honest absolute
number for the rules-only baseline after Bug 2 is **~180 ± 4 in eval**,
matching Stage 77a Run 8.

## What worked

1. **Synthetic conjunctive falsification test** — the nursery induced the
   Stage 78a conjunctive sleep+starvation rule from 500 surprise
   observations with no LLM and no labelled training. Strategy v5 novelty
   #2 has its first empirical proof. Test:
   `tests/learning/test_nursery_synthetic_conjunctive.py`.
2. **Wiring correctness** — `SurpriseAccumulator`, `RuleNursery`, and
   `LearnedRule` are integrated into `ConceptStore.simulate_forward`
   (Phase 7) and `run_mpc_episode`. 110/110 unit tests pass.
3. **Bug 2 discovery and fix** — ~+11 steps to baseline in eval, matching
   Stage 77a Run 8 cleanly. This **invalidates the Stage 78c v1 partial
   FAIL conclusion** (residual_off was 169 because of Bug 2; with the
   fix it's 180).
4. **MAX_ABS_EFFECT=0.5 cap** — eliminated the entropy collapse failure
   mode by dropping the ~10 highest-magnitude rules per ablation. These
   were textbook-cancellation noise (drink: -5 for misaligned do-water,
   energy: -1 for clamping when sleep target was at 9, etc).

## What didn't work

1. **Nursery doesn't help eval survival.** Even with the cap, nursery_on
   eval is +0.4 vs nursery_off — statistical noise. The wall (Stage 77a
   ~180) was not broken.
2. **Action entropy still partially collapses** with the nursery on
   (eval: off 0.97, on 0.75). Less severe than v2 (0.60) or Stage 78c
   (0.39 / 0.18), but still a measurable bias toward determinism.
3. **MLP residual approach (Stage 78c)** is broken in both v1 and v2 —
   Bug 2 fix did NOT lift residual_on, only residual_off. This is the
   stronger of the two failure modes (Δ = -26 vs nursery's -23 → +0.4
   with cap).
4. **Wood collection unchanged** at 0/60 ≥3 wood across all variants —
   no learning approach we tried touches the gathering wall.

## The discrimination paradox, reproduced three times

Stage 78a documented the **discrimination paradox**:

> Higher discrimination correlates with **worse** conjunctive MSE. The
> readout MLP overfits to the majority pattern and predicts large
> positive deltas confidently for all sleep samples, including the
> conjunctive 2.7% — getting them spectacularly wrong. When features
> are bland the MLP outputs ≈0, landing closer to the target than
> confident positive predictions would.

Stage 78c reproduced this with an MLP residual + online SGD:

| | Stage 78a | Stage 78c |
|---|---|---|
| Higher *X* worsens outcome | Higher disc → worse conj_health_mse | Lower entropy (more confident planner) → worse eval avg_len |
| What it learns | Majority rule (sleep → +0.04) confidently | Average correction over fingerprint collisions |
| Why it fails | "Good" features encode input variety not the conditional bit | Encoding collisions → average correction wrong for both modes |

Stage 79 reproduced it AGAIN with a completely different mechanism
(symbolic rule induction with explicit preconditions):

| | Stage 79 v1/v2 |
|---|---|
| Higher *X* worsens outcome | More learned rules → entropy collapse → worse eval |
| What it learns | Many small noise corrections + ~10 large textbook cancellations |
| Why it fails | Phase 7 cumulative effect of 100+ small rules biases planner trajectories toward suboptimal action distributions |

The recipe that produces this paradox:

```
features that aggregate samples needing different corrections
+ a learning mechanism that minimises mean error at promotion / training
= a confidently-wrong predictor that biases the MPC planner
```

The recipe is **invariant to**:

- Substrate (FHN spiking network, 67K MLP, symbolic discrete rules)
- Training paradigm (oracle MSE, online SGD, surprise statistics)
- Supervision source (synthetic labels, env rollouts)
- Parameter count (5K nodes, 67K params, 100+ symbolic rules)

The Stage 79 v3 cap **doesn't break the recipe**, it just rejects the
top of the magnitude distribution where the harmful textbook
cancellations live. The remaining low-magnitude noise corrections
aren't strong enough to bias the planner significantly. That's a
band-aid, not a structural solution.

## Bug 2 — proximity vs facing in `expand_to_primitive`

### Symptom

Stage 79 v1 nursery promoted ~10 rules with magnitudes ≥ 1.0:

```json
{"action": "do", "body_quartiles": [0,0,0,0],
 "visible": ["coal","cow","empty","stone","tree"],
 "effect": {"drink": -5.0}}
```

Reading `drink: -5.0` as "the nursery learned that do action in this
context loses 5 drink" — but Crafter's "do" doesn't actively REMOVE
drink. The rule was the nursery cancelling the textbook's positive
prediction (do-water → drink +5) when the env didn't deliver.

### Root cause

The original Stage 77a `expand_to_primitive` for "do" used proximity
(`manhattan ≤ 1`) to decide whether to emit "do" or a navigation step.
But Crafter's "do" action interacts with the **facing tile** (determined
by `last_action`), not any tile within manhattan ≤ 1.

Scenario:

- Player at (10, 10), water at (11, 10), `last_action = move_up` → facing UP
- Planner Phase 6 'do' branch fires with proximity check → predicts drink +5
- `expand_to_primitive` returns "do" (manhattan=1 ≤ 1)
- `env.step("do")` interacts with (10, 9), not (11, 10) → no drink
- Surprise: predicted +5, actual 0 → gap −5
- Nursery learns "this context, do → drink −5"
- Subsequent planning: rules +5 + learned −5 = 0 → planner abandons
  do-water → agent dehydrates

The Stage 77a comment in `concept_store.py:646-651` explicitly notes
the proximity-vs-facing mismatch and uses proximity as a workaround
"because facing-based fallback never fires in sim". That workaround
made sim agree with itself but disagree with env.

### Fix

In `expand_to_primitive`'s "do" branch: only return "do" when the
target tile **is** the facing tile. When adjacent but not facing,
return a navigation step toward the target. Crafter blocks movement
into impassable tiles (water/stone/tree) but still updates
`last_action`, so the next iteration's expand sees the correct facing
and emits "do".

`_apply_tick` was NOT changed — the planner's rollout still predicts
"do works at manhattan ≤ 1", which is correct for long-term plan
selection (the planner picks "drink water" plans for their long-term
value). `expand_to_primitive` handles the multi-tick face-then-do
execution implicitly.

### Effect

- Stage 78c v2 residual_off lifted from 169.2 to 180.4 (+11.2 in eval)
- Stage 79 v2 nursery_off lifted from 170.8 to 184.6 (+13.8 in eval)
- Both **match** Stage 77a Run 8 cached baseline of 180

This means the cached Stage 77a Run 8 number was the **right** baseline
all along (eval 180 ≈ historical band 167-180), and Stage 78c v1 was
the OUTLIER because Bug 2's effect on the Stage 78c training cycle
was different from its effect on the original Stage 77a run.

## MAX_ABS_EFFECT=0.5 — pragmatic mitigation

After Bug 2 was fixed, Stage 79 v2 still showed nursery_on -23 below
nursery_off in eval. Inspecting the 151 promoted rules:

- 168 effects total (some rules have multiple vars)
- median magnitude 0.027
- 10 effects ≥ 1.0 — all textbook cancellations:
  - `drink: -5.0` for do (Bug 2 surfaced but not fully fixed in some
    contexts where spatial_map said water was nearby but perception
    missed it)
  - `food: -5.0` for do (analogous, do-cow case)
  - `energy: -1.0` for sleep (clamping when energy was at max=9)
  - `health: ±0.5+` for sleep / do in misc contexts

These large-effect rules act as confident *negative* facts: "in this
context, this action does NOT deliver what the textbook says". When
applied during planning (Phase 7), they cancel valid plans the agent
needs for survival.

The cap at 0.5 drops these from promotion. Legitimate per-tick body
deltas (background rates ≤0.05; spatial damage ≤0.5) survive.
Synthetic conjunctive test still passes (effect ≈0.026, well below cap).

This is a **heuristic, not a structural fix**. The deeper bug is that
the encoding (visible+body+action) loses load-bearing distinctions
(facing direction, entity positions, last_action context). A learned
rule with a "matches almost any sleep observation" precondition will
fire too broadly during planning. The cap papers over this.

## Files

### Code (NEW)

- `src/snks/learning/surprise_accumulator.py` — `ContextKey`,
  `SurpriseRecord`, `SurpriseAccumulator` with sliding-window L1/L2
  bucketing. ~270 lines.
- `src/snks/learning/rule_nursery.py` — `CandidateRule`, `RuleNursery`
  with emit/verify/promote pipeline. Constants: MIN_OBS_L1=5,
  MIN_OBS_L2=10, MAD_K=2.0, VERIFY_N=10, VERIFY_TOL=0.02,
  SIGNIFICANCE_FLOOR=0.01, MAX_ABS_EFFECT=0.5. ~290 lines.
- `src/snks/agent/learned_rule.py` — `LearnedRule` dataclass with
  `matches()` predicate. Subset semantics on visible, exact on action,
  body quartile check skipped for L1 rules. ~95 lines.
- `experiments/stage79_nursery_crafter.py` — Crafter ablation harness
  (nursery_off, nursery_on, optional nursery_on_residual_on). ~470 lines.

### Code (MODIFIED)

- `src/snks/agent/concept_store.py`:
  - `learned_rules: list` field
  - `add_learned_rule()` helper
  - Phase 7 in `_apply_tick` (after Phase 6, before final clamp)
  - `_apply_tick` accepts `visible_concepts` parameter, threaded from
    `simulate_forward`
  - `expand_to_primitive` 'do' branch now checks facing (Bug 2 fix)
- `src/snks/agent/mpc_agent.py`:
  - `run_mpc_episode` accepts `surprise_accumulator`, `rule_nursery`,
    `nursery_tick_every`
  - After `env.step`: 1-tick rules-only replay (with planned_step
    propagation, the Stage 78c Bug 1 fix), feed surprise to accumulator,
    optionally tick nursery
  - Lazy import keeps mpc_agent independent of `snks.learning` when
    nursery isn't used

### Tests (NEW, 44 tests, all passing)

- `tests/learning/test_surprise_accumulator.py` (14 tests)
- `tests/learning/test_rule_nursery.py` (13 tests)
- `tests/learning/test_nursery_synthetic_conjunctive.py` (1 critical
  falsification test, **PASSED**)
- `tests/agent/test_learned_rule.py` (10 tests)

Plus the Stage 78c regression suite (11 tests) was preserved through
all changes — no Stage 77/78/79 regressions across 110 tests.

### Results / data

- `_docs/stage79_results.json` — v3 final results (with cap)
- `_docs/stage79_results.txt` — v3 raw log
- `_docs/stage79_v2_results.txt` — v2 raw log (without cap)
- `_docs/stage79_v3_results.txt` — v3 raw log (with cap)
- `_docs/stage79_learned_rules_nursery_off.jsonl` — empty (off doesn't learn)
- `_docs/stage79_learned_rules_nursery_on.jsonl` — 137 promoted rules (v3)
- `_docs/stage78c_v2_results.txt` — Stage 78c re-run with Bug 2 fix

### Specs / docs

- `docs/superpowers/specs/2026-04-11-stage79-rule-nursery-design.md`
  — full design doc (promoted from sketch)
- `docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md`
  — original sketch (unchanged)

## Strategy v5 novelty accounting after Stage 79

| # | Novelty | Status before 79 | Status after 79 |
|---|---|---|---|
| 1 | DAF substrate as residual learner | parked Branch C | still parked. Discrimination paradox arguments suggest re-test won't break the recipe. |
| 2 | No-LLM rule induction from surprise | retained, Stage 79 next | **mechanism validated on synthetic, infrastructure complete, but doesn't beat baseline on Crafter at current encoding granularity**. The synthetic test pass is publishable in itself; the Crafter neutral result is honest negative. |
| 3 | Three-category ideology | retained | retained — facts (textbook) untouched, mechanisms (Phase 7) is code, experience (`learned_rules`) is the runtime addition. The cap is a property of the experience layer's promotion gate, not a violation of the taxonomy. |

## Decision points (need user input)

The Stage 79 outcome is partial PASS / informative negative. The agent
infrastructure is solid; what's missing is the **right encoding** for
prediction-error driven learning to overcome the discrimination paradox.

### Option A — Stage 80 encoding refinement

Add facing direction, last_action, entity positions (relative or
binned) to `ContextKey`. Re-train both nursery and (optionally) the
Stage 78c residual on the richer encoding. Expected outcome: fewer
fingerprint collisions → less averaging noise → real signal can
emerge.

Cost: ~1 week. Risk: still might not break the paradox if the
encoding loses other load-bearing dimensions we haven't identified
yet.

This is the **most promising structural fix** for the paradox.

### Option B — Strategic pivot

Abandon the "learn corrections via prediction error in MPC loop"
paradigm. Possible alternatives:

- **Hierarchical planning**: planner emits subgoals ("get drink") and
  a low-level controller chooses primitives. The low-level can
  experiment with different action sequences when the current one
  fails, without the planner needing to learn body-delta corrections.
- **Imagination-augmented agent**: rollout in a learned world model
  but use NON-additive composition (e.g. probabilistic effects,
  attention-weighted integration). Avoids the "100 small rules
  cumulatively bias trajectory" failure.
- **Goal-conditioned exploration**: when the planner's preferred plan
  fails (surprise > threshold), trigger a brief exploration burst with
  diverse actions to find what actually works.

Cost: 2-4 weeks of design + implementation. Risk: highest, could be
the right move or could be another local minimum.

### Option C — Stage 78a re-test with spikerate readout

Run R2/R4/R6 with `readout="spikerate"` instead of `voltage`. ~3
GPU-hours.

Discrimination paradox arguments suggest this is unlikely to break
the recipe — it just changes the substrate's feature extractor, not
the learn-from-error paradigm. But it's cheap and would close the
last open Stage 78a methodological gap. **Useful if running in
parallel with Option A**, useless as a primary path.

### Option D — Consolidate and stop

Stage 79 has shipped: synthetic conjunctive induction works without
an LLM (publishable result). Bug 2 fix lifted the baseline to its
honest position. The agent doesn't beat Stage 77a wall but nothing
in Strategy v5 was guaranteed to. Write a comprehensive synthesis
doc, update memory, mark Strategy v5 as "informative negative" for
the wall-breaking question, and step back to reconsider the broader
research direction (e.g. is "close Crafter wall via learning
corrections" the right framing at all? What about "close it via
hierarchical planning"?).

This is the lowest-cost outcome and gives time to think.

**Default recommendation:** Option A (encoding refinement). It's the
cheapest structural attack on the paradox, and the synthetic test
already shows the nursery mechanism works when the encoding is
informative enough. If Option A doesn't help, Option D becomes more
attractive. Option C can run in parallel with Option A as a free
data point.

## Strategy v5 honest assessment

After three Stage 78 / 79 attempts, we have:

- **Bug fixes**: 2 real bugs found and fixed, each lifted the baseline
- **Infrastructure**: synthetic falsification + wired Crafter
  integration for both residual and nursery approaches
- **Negative results**: discrimination paradox is robust to substrate,
  training paradigm, and parameter count
- **Wall**: not broken — Stage 77a baseline ≈180 still holds in eval
- **Wood**: untouched — 0 ≥3 across all variants

The Strategy v5 thesis ("neural-symbolic learning corrections within
MPC") has produced clean infrastructure but no Crafter delta. The
roadmap from here either doubles down on encoding (Option A) or
pivots away from MPC corrections (Option B).
