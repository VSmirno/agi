# Stage 89c Post-Mortem Plan — Local Threat Gain vs Global Survival Loss

**Roadmap:** `docs/ROADMAP.md`  
**Phase:** `Phase I — Dynamic World Model`  
**Date:** 2026-04-18

## Goal

Explain why Stage 89/89b improved arrow-specific competence but reduced overall survival.

Current evidence:
- fair baseline: `avg_survival=191.25`, `arrow_death_pct=25.0`, `defensive_action_rate=0.0`
- current Stage 89b: `avg_survival=163.6`, `arrow_death_pct=20.0`, `defensive_action_rate=0.031`

This means the project has a **real local improvement** in dynamic threat handling, but that improvement is not yet integrated into a better whole-agent survival policy.

## Success Criteria

Stage 89c is complete only if all of the following hold:
- we can name the dominant failure mode behind `survival_delta < 0`
- we can separate `improved`, `neutral`, and `regressed` episodes by explicit criteria
- we can point to at least one concrete class of states where local dodge behavior harms longer-horizon viability
- the final diagnosis identifies the next architectural fix without relying on a new Crafter-specific reflex

## Task 1 — Freeze the Comparison Pair

**Artifacts:**
- `_docs/stage89_baseline.json`
- `_docs/stage89_eval.json`

Steps:
1. Treat the new baseline as the only valid pre-Phase-I reference for Stage 89 analysis.
2. Record the current comparison bundle:
   - `avg_survival`
   - `arrow_death_pct`
   - `danger_prediction_error`
   - `defensive_action_rate`
   - death-cause breakdown
3. Explicitly mark the old bootstrap baseline as obsolete for Stage 89 conclusions.

Done criteria:
- all later Stage 89c notes and reports compare only against the fair baseline

## Task 2 — Failure Bucket Analysis

**Goal:** partition episodes into `improved`, `neutral`, `regressed`.

For each episode pair on matching seed:
1. Compare `episode_steps` between baseline and current.
2. Label:
   - `improved`: current survives materially longer
   - `neutral`: survival difference small / outcome equivalent
   - `regressed`: current dies materially earlier
3. Record:
   - final death cause
   - first step where `defensive_action_steps` increments
   - first visible `arrow` threat step
   - vitals at `t`, `t+10`, `t+20`, `t+30` after first defensive action

Done criteria:
- every seed belongs to one bucket
- at least one `regressed` bucket gets concrete manual explanation

## Task 3 — Post-Defense Trajectory Audit

**Goal:** inspect what current policy buys and what it sacrifices after defensive motion.

For each episode with at least one defensive action:
1. Log the chosen primitive and originating plan.
2. Log the predicted immediate loss vs baseline-plan predicted loss.
3. Log the post-maneuver state:
   - `health`, `food`, `drink`, `energy`
   - nearest water / cow / tree / hostile distances
   - whether the agent moved into a worse threat geometry
4. Measure opportunity cost:
   - delayed access to water
   - delayed access to food
   - delayed gather/craft progress
   - movement into zombie/skeleton pressure

Done criteria:
- we can describe whether the agent is overpaying for dodge with lost tempo, bad positioning, or resource starvation risk

## Task 4 — Counterfactual State Comparison

**Goal:** compare `baseline-best` vs `current-best` on identical states.

For selected states from regressed episodes:
1. Save a diagnostic snapshot before the first relevant defensive choice.
2. Re-evaluate:
   - baseline mode
   - current mode
3. Compare:
   - chosen top plan
   - predicted health loss
   - predicted state after short horizon
   - access to survival resources after the horizon

Done criteria:
- at least one counterfactual clearly shows `safer now, worse later`
  or `still unsafe, but now slower`

## Task 5 — Integration Metrics Prototype

**Goal:** add diagnostics for the gap between local danger avoidance and long-horizon viability.

Add temporary metrics, not final policy logic:
- `first_arrow_threat_step`
- `first_defensive_action_step`
- `post_defense_survival_10`
- `post_defense_survival_20`
- `post_defense_health_delta_20`
- `post_defense_food_delta_20`
- `post_defense_drink_delta_20`
- `resource_access_loss`
- `threat_distance_after_defense`
- `threat_distance_delta_10`

Interpretation target:
- did the maneuver create a safer corridor or just defer damage?
- did the maneuver preserve viability or destroy access to essentials?

Done criteria:
- these metrics are present in diagnostic outputs and can explain at least one regressed episode

## Task 6 — New Diagnostic Outputs

Add a dedicated experiment or analysis script that writes:
- `_docs/stage89c_episode_buckets.json`
- `_docs/stage89c_regression_cases.json`
- `_docs/stage89c_counterfactuals.json`

Each regression case should include:
- `seed`
- `baseline_steps`
- `current_steps`
- `baseline_death_cause`
- `current_death_cause`
- `first_arrow_threat_step`
- `first_defensive_action_step`
- `predicted_best_loss`
- `predicted_baseline_loss`
- `post_defense_vitals`
- `post_defense_resource_access`
- `notes`

## Task 7 — Architectural Verdict

After the diagnostics, answer explicitly:
1. Is current policy overreacting to arrow threat and sacrificing viability?
2. Is current policy reacting too late to matter?
3. Is the remaining problem mostly:
   - candidate horizon
   - multi-threat integration
   - resource-vs-danger tradeoff
   - another layer entirely?
4. What is the smallest next fix that addresses the identified layer?

If this section cannot be answered concretely, Stage 89c is not done.

## Exact Logs to Add

Per-episode:
- `mode`
- `seed`
- `episode_bucket`
- `first_arrow_threat_step`
- `first_defensive_action_step`
- `n_defensive_sequences`
- `death_cause`
- `episode_steps`

Per-defensive event:
- `step`
- `primitive`
- `plan_origin`
- `predicted_best_loss`
- `predicted_baseline_loss`
- `health`
- `food`
- `drink`
- `energy`
- `nearest_zombie_dist`
- `nearest_skeleton_dist`
- `nearest_arrow_dist`
- `nearest_tree_dist`
- `nearest_water_dist`
- `nearest_cow_dist`

Windowed follow-up after each defensive event:
- `health_delta_10`
- `food_delta_10`
- `drink_delta_10`
- `health_delta_20`
- `food_delta_20`
- `drink_delta_20`
- `survived_10`
- `survived_20`

## Exact Questions the Data Must Answer

1. In regressed runs, does the first defensive action happen before or after the outcome is already effectively determined?
2. Does the action reduce immediate predicted damage but worsen `food/drink/resource access` enough to lower total survival?
3. Are we seeing a narrow arrow win that converts into zombie pressure later?
4. Is current policy paying for motion without obtaining a durable positional advantage?

## Dependency Order

```text
Task 1 ──> Task 2 ──> Task 3 ──> Task 4 ──> Task 5 ──> Task 6 ──> Task 7
```

## Risks

- matching baseline/current by seed may still hide environment stochasticity if other sources of nondeterminism leak in
- too many logs may obscure the dominant failure mode; prefer a compact regression bundle
- there is a risk of inventing a new tactical metric that silently becomes a policy hack; keep 89c diagnostic-first

## Exit Rule

Do not move to the next Stage 90-style fix until Stage 89c can state, in one sentence:

**“Stage 89 currently loses overall survival because local threat avoidance fails to integrate with global viability in this specific way: `<diagnosed mechanism>`.”**
