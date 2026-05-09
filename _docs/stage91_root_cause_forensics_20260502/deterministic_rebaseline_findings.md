# Stage 91 Deterministic Re-Baseline Findings

Date: 2026-05-09 (run on HyperPC, RTX 3090, torch 2.5.1+cu121)
Local HEAD: `09dd6ec` + uncommitted Stage91 working-tree files

## Goal

Now that strict torch determinism is achievable (kthvalue CPU offload + crafter
set-iteration patch, both committed in `09dd6ec`), re-measure the canonical
multiseed eval (seeds 7, 17, 27, 37, 47) at two baselines, and re-validate the
rejected directions from the prior single-trial forensics. Spread is 0 by
construction, so 1 trial per seed is sufficient signal.

## Run Provenance

- Wrapper: `/opt/cuda/agi-stage91-strict-determinism-kthvalue-cpu-20260509T145726Z/_docs/.../commands/deterministic_wrapper.py`
- Eval: `experiments/stage90r_eval_local_policy.py --mode mixed_control_rescue --n-episodes 4 --max-steps 220 --perception-mode symbolic --enable-planner-rescue --smoke-lite`
- Local-evaluator: `/opt/cuda/agi-stage91-verify-71d1e29-20260502T113506Z/_docs/stage90r_seed7_actor_selection_probe3.pt`
- All checkouts cloned via `cp -al` (hardlink, no extra disk).
- Phase A artifacts: `/opt/cuda/agi-stage91-determ-rebase-A-20260509T180000Z/_docs/deterministic_rebaseline/`
- Phase B artifacts: `/opt/cuda/agi-stage91-determ-rebase-B-20260509T180000Z/_docs/deterministic_rebaseline/`
- Phase C artifacts: `/opt/cuda/agi-stage91-determ-rebase-C-{threat_ranking,do_facing,extractor}-20260509T180000Z/_docs/deterministic_rebaseline/`

## Phase A — HEAD baseline (cumulative Stage91 fixes + determinism patches)

Source: clone of `agi-stage91-feasibility-label-fix-...` overlaid with HEAD's
`vector_world_model.py`, `crafter_pixel_env.py`, plus uncommitted local
working-tree files (`vector_mpc_agent.py`, `vector_sim.py`, `vector_bootstrap.py`,
`stage90r_emergency_controller.py`, `stage90r_local_policy.py`,
`stage90r_local_affordances.py`, `experiments/stage90r_eval_local_policy.py`).

This is the "diagnostics-only rollback" runtime per
`local_affordance_diagnostics_only_validation_findings.md`: feasibility-label
+ blocked-move + movement + extractor module loaded (read-only, not wired).

| seed | avg_survival | rescue_rate | death breakdown |
|------|------:|------:|---|
| 7    | 155.50 | 0.495 | unknown:1, zombie:2, skeleton:1 |
| 17   | 141.50 | 0.394 | zombie:2, dehydration:1, unknown:1 |
| 27   | 208.25 | 0.269 | zombie:2, arrow:1, alive:1 |
| 37   | 201.25 | 0.199 | zombie:3, dehydration:1 |
| 47   | 194.75 | 0.261 | zombie:4 |

- Weak-seed mean (7+17): **148.50**
- Strong-seed mean (27+37+47): **201.42**
- Overall mean (5 seeds): **180.25**

Seeds 7, 17, 27 byte-match the values cited in commit `09dd6ec` message
(155.50 / 141.50 / 208.25). Confirms strict determinism holds and the Phase A
build is faithful.

## Phase B — pre-Stage91 baseline (pure 71d1e29 + only the two determinism patches)

Source: clone of `agi-stage91-verify-71d1e29-20260502T113506Z` with HEAD's
`vector_world_model.py` and `crafter_pixel_env.py` overlaid. No movement,
no blocked-move, no feasibility-label, no diagnostics module.

| seed | avg_survival | rescue_rate | death breakdown |
|------|------:|------:|---|
| 7    | 112.00 | 0.404 | unknown:1, zombie:2, arrow:1 |
| 17   | 150.50 | 0.442 | skeleton:1, dehydration:1, zombie:1, unknown:1 |
| 27   | 162.25 | 0.182 | zombie:2, arrow:1, alive:1 |
| 37   | 200.25 | 0.226 | zombie:3, dehydration:1 |
| 47   | 160.00 | 0.342 | zombie:3, arrow:1 |

- Weak-seed mean (7+17): **131.25**
- Strong-seed mean (27+37+47): **174.17**
- Overall mean (5 seeds): **157.00**

## Phase A vs Phase B — Stage91 fixes contribution

| seed | Phase A | Phase B | Δ (A − B) |
|------|------:|------:|------:|
| 7    | 155.50 | 112.00 | **+43.50** |
| 17   | 141.50 | 150.50 | **−9.00** |
| 27   | 208.25 | 162.25 | **+46.00** |
| 37   | 201.25 | 200.25 | +1.00 |
| 47   | 194.75 | 160.00 | **+34.75** |
| weak (7+17) | 148.50 | 131.25 | **+17.25** |
| strong (27+37+47) | 201.42 | 174.17 | **+27.25** |
| overall | 180.25 | 157.00 | **+23.25** |

**Interpretation.** The "Stage91 fixes contributed nothing" null hypothesis is
**rejected**. The cumulative Stage91 fixes (movement + blocked-move +
feasibility-label + diagnostics-only extractor module) deliver a measurable
overall +23.25 avg-survival lift, with positive deltas on 4/5 seeds. The
weakest-seed gain is +43.5 on seed 7 — exactly the seed the fixes targeted.

The single negative seed is 17 (−9.0). This is small in magnitude, but is now
the canonical "weakness" remaining: feasibility-label + blocked-move + movement
combined slightly hurt seed 17 even as they help seeds 7, 27, 47. Seed 17's
loss is not noise — it is reproducible under strict determinism.

## Phase C — re-measurement of rejected directions

Each direction was rebuilt from `verify-71d1e29` plus that direction's
`synced_runtime_diff.patch`, plus the two HEAD determinism patches. Run for
the canonical weak seeds 7 and 17 only (compute budget; weak-mean is the
canonical decision metric).

### Recovered directions

| direction | seed 7 | seed 17 | weak mean | Δ vs Phase A weak (148.50) | Δ vs Phase B weak (131.25) | prior single-trial claim |
|---|---:|---:|---:|---:|---:|---|
| **threat_ranking** | 104.00 | 119.75 | **111.88** | **−36.62** | −19.38 | −28 to −31 (vs movement-fix) |
| **do_facing** | 116.25 | 126.75 | **121.50** | **−27.00** | −9.75 | −40.3 (vs feasibility-label) |
| **local-affordance extractor v1** | 155.50 | 141.50 | **148.50** | **0.00** | +17.25 | −10.4 (vs feasibility-label) |

**threat_ranking**: large regression confirmed under strict determinism. The
algorithmic change in `stage90r_emergency_controller.py` (replacing
`sleep_threat_penalty` with `freeze_threat_penalty` + `movement_escape_bonus`
+ context-dependent `survival_term`) costs **−36.62** weak-mean vs Phase A
and is also worse than the pre-Stage91 baseline (−19.38 vs Phase B). Prior
rejection holds; the regression is real, not noise.

**do_facing**: regression confirmed but **smaller** than originally claimed
(−27.0 vs claimed −40.3). Still clearly negative vs Phase A. Worse than
Phase A but only marginally worse than pre-Stage91 (−9.75 vs Phase B).
Prior rejection holds.

**local-affordance extractor v1**: re-measurement under determinism shows
**zero** regression vs Phase A (148.50 = 148.50 on the weak-seed pair).
The vector_mpc_agent.py file md5-differs from Phase A (it has the active
wiring `affordance_snapshot=local_affordance_snapshot` passed into
`_build_local_counterfactual_outcomes`, while Phase A has the
diagnostics-only rollback that does not pass it). Yet on these seeds the
counterfactual labels resulting from the affordance scene happen to not
shift the controller's action choice — output is bit-identical to Phase A
on the weak-seed pair. **The original "−10.4" rejection was inside the
prior 25–55 noise spread and was not signal.** The wiring is neutral on the
weak seeds; whether it would change strong seeds is untested in this Phase C.

### Lost directions

| direction | reason |
|---|---|
| **diagnostics-only local-affordance** | Not lost in the recovery sense — the local working tree at HEAD already _is_ the diagnostics-only state per `local_affordance_diagnostics_only_validation_findings.md`. Phase A measures it directly. The original "−21.5" claim was made against a baseline that no longer exists as a checkout (feasibility-label + extractor wiring removed minus the diagnostics module). Not retestable as a distinct point. |

The original isolated experiment checkouts (`agi-stage91-do-facing-fix-...`,
`agi-stage91-local-affordance-extractor-...`,
`agi-stage91-local-affordance-diagnostics-only-...`,
`agi-stage91-threat-ranking-fix-...`) were all deleted from `/opt/cuda` before
this re-baseline, but each direction's `synced_runtime_diff.patch` survived
inside `verify-71d1e29/_docs/hyper_stage91_*/env/`, allowing reconstruction
for three of four directions.

## Bottom line

Which prior conclusions hold under strict determinism, which were noise:

| prior conclusion | status under deterministic re-measurement |
|---|---|
| Stage91 fixes (cumulative: movement + blocked-move + feasibility-label) lift weak-seed survival | **HOLDS.** +43.50 on seed 7, +17.25 weak-mean, +23.25 overall. Real signal. |
| Seed 17 is a remaining weakness | **HOLDS, sharpened.** Stage91 fixes cost −9.0 on seed 17. Reproducible, not noise. Worth a focused investigation. |
| do_facing regresses | **HOLDS** (−27.0 weak-mean), magnitude smaller than claimed (was −40.3). |
| threat_ranking regresses | **HOLDS** (−36.6 weak-mean), magnitude consistent / slightly larger than claimed. |
| local-affordance extractor v1 regresses | **OVERTURNED.** Δ = 0.00 weak-mean. The −10.4 was noise. The wiring is bit-equivalent to Phase A on the weak seeds. The earlier rejection was based on a single-trial reading inside the noise band. Strong-seed effect untested. |
| diagnostics-only local-affordance regresses | **OVERTURNED in spirit.** Phase A _is_ the diagnostics-only runtime. It is +17.25 vs the pre-Stage91 baseline; the original "−21.5" claim was vs a no-longer-extant intermediate baseline and is not a meaningful comparison under the current state. |

The single most actionable revision is on **local-affordance extractor v1**:
that direction was rejected on noise. It can be revived if the strong-seed
effect is also non-negative. (Re-measurement on seeds 27/37/47 was outside
this Phase C compute budget but is cheap to add — ~5 min.)
