# Stage 89 Report — Arrow Trajectory Modeling

**Dates:** 2026-04-17 → 2026-04-19  
**Status:** **PARTIAL**  
**Parent phase:** `Phase I — Dynamic World Model`  
**Related docs:** [docs/ROADMAP.md](../ROADMAP.md), [docs/ASSUMPTIONS.md](../ASSUMPTIONS.md), [docs/STAGE_REVIEW_CRITERIA.md](../STAGE_REVIEW_CRITERIA.md)

## TL;DR

Stage 89 did produce a real dynamic-threat capability, but the first readings were partly distorted by lower-layer bugs and one telemetry mistake.

What is now confirmed:
- `exp137` perception retrain made CNN perception nearly identical to `semantic` on held-out agreement checks.
- `arrow` was restored as a proper concept in the textbook vocabulary, so live tracking and velocity inference work again.
- `arrow -> proximity` was seeded as a stable world fact, so the simulator can finally predict projectile damage instead of treating arrows as harmless visible objects.
- the original `defensive_action_rate` looked artificially bad because `arrow_threat_steps` counted **all visible arrows**, not only states where baseline predicted imminent damage.

After the telemetry fix, a targeted seed-44 diagnostic showed:
- `arrow_visible_steps = 66`
- `imminent_arrow_steps = 13`
- `defensive_actions_on_imminent_steps = 13`

This means the planner was not failing on every arrow-visible step; the old metric was overstating the failure. A fresh smoke run on `exp137` gave:
- `avg_survival = 156.5`
- `arrow_death_pct = 0.0`
- `arrow_visible_steps = 87`
- `arrow_threat_steps = 3`
- `defensive_action_steps = 3`
- `defensive_action_rate = 0.25`
- death causes: `zombie=3`, `skeleton=1`

So Stage 89 succeeded at **local projectile modeling and imminent dodge selection**, but it did **not** solve overall survival. The remaining wall is now higher in the stack: broader survival policy against `zombie/skeleton`, not projectile perception or tracking.

## What Changed

Core Stage 89 work:
- arrow state added to runtime tracking with inferred velocity and short persistence
- dynamic entities propagated into `VectorState`
- `simulate_forward()` extended with short-horizon projectile movement
- threat-aware scoring via `VitalDeltaStimulus`
- motion plans and motion chains added for generic repositioning

Critical fixes discovered during debugging:
- `arrow` added to textbook vocabulary, so bootstrap creates the concept and tracker can register it
- `arrow -> proximity` added as textbook passive fact, so the world model predicts projectile damage
- Stage 89 telemetry changed to count **imminent** arrow threat, not just projectile visibility
- viewport→world mapping fixed for `spatial_map` / `DynamicEntityTracker` (off-by-one on Y)
- perception now emits off-center `empty`, so stale resource labels are cleared instead of lingering as ghost trees

## Evidence

Perception / lower-layer evidence:
- held-out `exp136`: `near_match_rate = 0.800`, `mean_jaccard = 0.487`
- held-out `exp137`: `near_match_rate = 1.000`, `mean_jaccard = 0.999`

Planner / dynamic-threat evidence:
- pre-fix smoke on `exp137` showed `arrow_visible_steps > 0` but `arrow_threat_steps = 0`
- root cause 1: `arrow` missing from textbook vocabulary
- root cause 2: `arrow -> proximity` missing from textbook passive rules
- root cause 3: `arrow_threat_steps` counted visibility rather than imminent simulated damage
- separate resource trace on `seed=44` disproved the earlier "tree semantics are noisy" hypothesis:
  - Crafter source says `tree -> wood`, while `grass -> sapling`
  - old trace showed `facing_label_before = tree`, but `env_material_before = grass`
  - after mapping + stale-map fixes, the same short trace produced `n_frustrated_tree_do = 0`
    and `n_successful_tree_do = 3`

Targeted diagnostic:
- on seed `44`, after the above fixes, baseline predicted non-zero projectile damage on only a subset of visible-arrow steps
- on those imminent states, movement plans beat baseline and were selected consistently
- on the same seed, resource interaction no longer hit `grass` while believing it was `tree`;
  `tree:do` yielded real `wood` again

## Stage Review

**Ideological debt addressed:** projectile dynamics were visible in the environment but absent or misplaced across facts, runtime tracking, and short-horizon simulation.

**Layer changed:** `facts`, `mechanisms`, `experience`, `stimuli`

**What changed:** arrow knowledge was restored to textbook facts, runtime experience now tracks projectile motion, simulator models short-horizon projectile collision, threat telemetry now measures imminent danger instead of raw visibility, and the perception→map layer now writes/clears world tiles with correct geometry.

**Evidence of improvement:** `exp137` agreement fix; live arrow tracking with velocity; seed-44 diagnostic showing imminent-threat dodge selection; fresh smoke with `arrow_death_pct = 0.0` and `defensive_action_steps = arrow_threat_steps = 3`; fixed seed-44 resource trace with `3/3` successful `tree:do` interactions and no more `tree -> grass` mismatch.

**Why this is architectural, not tactical:** fixes corrected missing world facts, generic runtime tracking, generic short-horizon simulation, measurement semantics, and a lower-layer world-coordinate bug in the perception→map path. No Crafter-specific reflex like `if arrow then sidestep` was introduced.

**Knowledge flow outcome:** stable projectile knowledge now lives in textbook facts (`arrow` concept and `arrow:proximity` damage). Runtime projectile trajectories live in episode-local dynamic entity state. This knowledge now survives process start correctly because bootstrap creates the concept from textbook.

**Remaining assumptions / walls:** Stage 89 still does not raise overall survival; the remaining bottleneck is broad survival policy and multi-threat integration against `zombie/skeleton`. Dynamic-threat telemetry is now trustworthy, and adjacent resource interaction is no longer corrupted by stale ghost trees, but Phase I is not complete.

**Decision:** `PARTIAL`

## Conclusion

Stage 89 should not be described as a failed arrow-defense stage anymore. It did establish the missing projectile capability, but earlier telemetry mixed together:
- visible arrows,
- missing textbook facts,
- and missing projectile damage rules.

Once those were corrected, the narrow projectile story became coherent. The next work item should no longer be “make the agent finally notice arrows”; it should be “improve overall survival policy now that projectile perception and imminent-threat modeling are no longer the dominant failure.”
