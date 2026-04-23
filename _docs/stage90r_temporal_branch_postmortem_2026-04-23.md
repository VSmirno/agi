# Stage90R Temporal Branch Postmortem

Date: 2026-04-23

## Decision

Return `main` to the proven `fullfix` Stage90R baseline and pause the temporal
branch.

Baseline restored to the code state at commit `3b6c1a9`, which is the last
Stage90R point before temporal-belief work began.

## Why The Temporal Branch Was Paused

The temporal branch went through several rounds of debugging:

1. Initial temporal-belief run exposed a simple runtime bug (`inv_after`
   undefined) that was fixed.
2. Clean temporal runs then showed worse online survival than `fullfix` even
   when action collapse was reduced.
3. A policy-history leak was found in the temporal export:
   - `prev_action_*`
   - `recent_hist_*`
   - `action_streak_norm`
   - `stationary_streak_norm`
4. Removing those features improved the branch, but did not beat `fullfix`.
5. The old validation split overstated collapse; switching to a stable episode
   split corrected that evaluation bug.
6. After split correction, the branch still underperformed and developed a
   `sleep`-heavy online policy due to checkpoint-selection miscalibration.
7. The final decisive test was a fresh full rerun under the trimmed 5-feature
   temporal schema. That run failed offline gate completely before checkpoint
   promotion.

## Decisive Failure

Fresh clean worktree on `minipc`:

- `/opt/agi-stage90r-temporalfullclean-122b3a2`

Fresh full collection summary:

- `samples_collected = 1488`
- `n_state_samples = 940`
- `n_comparison_ready_states = 832`

Training/eval log:

- `/opt/agi-stage90r-temporalfullclean-122b3a2/_docs/stage90r_temporalfullclean_20260423_122b3a2_continue.log`

Fresh train report:

- `/opt/agi-stage90r-temporalfullclean-122b3a2/_docs/stage90r_local_evaluator_eval.json`

Observed behavior:

- All 20 epochs failed anti-collapse gate.
- Dominant predicted action was `sleep` on every epoch.
- Dominant share stayed `0.9211`.
- Normalized entropy stayed `0.1541`.

Failing anti-collapse checks on every epoch:

- `dominant_action_share`: `0.9211` vs required `<= 0.70`
- `predicted_top1_entropy`: `0.1541` vs required `>= 0.45`
- `threat_slice_diversity`: `n_states=57`, `unique_top1_actions=1`

This means the fresh clean temporal branch collapsed immediately under the
trimmed temporal schema. The problem is no longer explainable by:

- bad split integrity
- checkpoint ordering alone
- stale checkpoint reuse

## Comparative Outcome

Reference runs:

- `fullfix` remained the best working baseline.
- `temporalfix` reduced some collapse but worsened survival.
- `temporaltrim` reuse runs improved on `temporalfix`, but still failed to beat
  `fullfix`.
- fresh full-clean temporal rerun failed before online eval because no epoch
  passed offline gate.

## Interpretation

The trimmed 5-feature temporal representation is not strong enough on a fresh
clean recollection. Earlier reuse runs were useful diagnostically, but they did
not prove that the trimmed temporal schema could stand on its own.

## Policy Going Forward

- Keep `fullfix` as the Stage90R baseline.
- Treat the temporal branch as paused.
- Do not run more heavy temporal reruns without a new representation design.
- If temporal work is revisited, start with offline-first representation
  redesign, not canary hacks or more reruns of the same schema.
