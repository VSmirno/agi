# Stage 74 Overnight Tuning Summary

**Date:** 2026-04-08

## What was done

7 hours of autonomous tuning with ~15 iterations on minipc (exp131, 200 episodes/phase).

### Architecture changes (from brainstorming session):
1. **HomeostaticTracker** — tracks body rate of change + conditional rates (STDP-like)
2. **Homeostatic drives** — urgency = 1/steps_until_zero (no hardcoded weights)
3. **Curiosity drive** — model incompleteness as biological drive
4. **Preparation drive** — proactive craft when known threats can't be handled
5. **Strategy 2** — trace cause of health drop → plan to remove cause (zombie→sword)
6. **Body rules in textbook** — innate rates (food depletes, zombie hurts)

### Tuning fixes discovered overnight:
1. **Background rates from body rules** — `_background` concept → unconditional rates
2. **Relative matching** — margin ≥0.1 between best and second-best match (fixes inter-class confusion)
3. **Plan stability** — don't replan during active plan execution
4. **Skip dangerous plan targets** — don't navigate TO zombie, let reactive handle
5. **Prereq-aware plan advance** — don't advance to "place table" until enough wood
6. **Immediate craft after place** — place table → immediately make sword (same location)

### Perception quality diagnostic:
- Intra-class similarity: 0.99 (excellent — tree vs tree)
- Inter-class similarity: 0.55-0.85 (problematic — tree vs water: 0.55, stone vs water: 0.82)
- 256-dim per-position features not trained for cosine matching
- Relative matching (margin) partially compensates

## Results

| Gate | Start of night | End of night | Status |
|------|----------------|--------------|--------|
| Tree nav ≥50% | 42.5% | **58.5%** | **PASS** |
| Grounding ≥5 | 6 | **5-7** | **PASS** |
| Stone ≥20% | 0% | 0% | FAIL |
| Survival ≥200 | 96 | **138** | FAIL |
| Verification ≥3 | 5 | **3-5** | **PASS** |

**3/5 gates PASS consistently.**

## Key achievement

**Sword emergence from homeostatic drives**: Agent crafted wood_sword in episode 74 of survival phase:
- HomeostaticTracker knew: zombie causes health -2.0 (from body rules)
- Agent had no sword → preparation drive activated
- Plan: kill_zombie → wood_sword → table → wood (backward chain)
- Collected 3 wood → placed table → IMMEDIATELY crafted sword

This happened WITHOUT ANY hardcoded "craft sword" logic. Purely from:
- Body: "zombie hurts" (innate)
- World model: "sword kills zombie" (textbook) 
- Planning: backward chain (ConceptStore)
- Drives: urgency from body rates

## Remaining bottleneck

256-dim spatial features are insufficiently discriminative for reliable object recognition. Agent needs ~80 steps to find 3 trees (babble + navigation), but dies in ~100-140 steps from zombie/starvation. Sword crafting succeeded once in 200 episodes.

## exp132: CNN 512 channels

Retrained CNN with 512 channels instead of 256. Results:
- Tree nav: 55.5% PASS
- Survival: **150** steps (best ever, +9% vs 256-ch)
- Sword: still 0/200
- Last 50 episodes trend: 156 steps

512 channels improve discrimination but sword crafting still blocked by timing — agent needs ~80 steps to collect 3 wood, zombie arrives at ~50-100.

## Recommendations for next session

1. **Sword timing** — agent needs wood faster. Consider: initial inventory boost, or faster babble convergence
2. **Forward simulation** — agent should mentally simulate "if I craft sword first vs gather food first"
3. **Episode learning** — carry map/context between episodes for faster startup
4. **Contrastive loss** — fine-tune features specifically for cosine matching (not just classification)
