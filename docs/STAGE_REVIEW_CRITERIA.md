# Stage Review Criteria

This document defines the minimum evaluation criteria for new strategies, stage implementations, and reported results. Apply it before declaring a stage successful, closing ideological debt, or promoting learned behavior into long-lived knowledge.

For repository-level proof that the overall concept works, apply `docs/CONCEPT_SUCCESS_CRITERIA.md` in addition to this stage checklist.
Also apply `docs/ANTI_TUNING_CHECKLIST.md` whenever a result looks strong enough to tempt a claim of architectural progress; this is the anti-self-deception guard against environment-specific tuning.

## 1. Correct Layer Placement
Every new rule, heuristic, or behavior must be assigned to the correct layer:
- `facts` for stable world knowledge
- `mechanisms` for generic algorithms
- `experience` for episode-local state
- `stimuli` for action valuation and motivation

If a change cannot be classified cleanly, the design is probably wrong.

## 2. No Tactical Eval Hacks
Reject changes that improve a benchmark by inserting Crafter-specific policy logic, direct reward proxies, or entity-specific `if/else` branches in generic planning code. A stage only counts as progress if the improvement comes from the intended architectural layer.

## 3. Metric vs Motivation Separation
External metrics such as survival, success rate, or resource count are evaluation tools, not internal agent goals. Check that planning and scoring are driven by stimuli and world knowledge, not by benchmark-shaped shortcuts.

## 4. Knowledge Flow Quality
Any claimed learning result must answer:
- What was learned?
- Where does that knowledge live now?
- Does it persist across runs or generations?
- Is it a causal rule or only a correlation?

Do not promote unstable correlations into textbook facts.

## 5. Mechanism Generality
Generic components such as planners, simulators, trackers, and world models must stay generic. If a mechanism only works because it encodes one environment’s quirks, it is misplaced knowledge, not architecture.

## 6. Result Explainability
A passing gate is insufficient on its own. Each reported improvement must include a clear explanation of:
- which layer changed,
- why behavior improved,
- why the gain is not just seed variance, map bias, or threshold tuning.

## 7. Assumptions and Failure Modes Logged
Every stage must update `docs/ASSUMPTIONS.md` with new simplifications, limitations, false hypotheses, calibration mistakes, or known walls. Unlogged debt will be rediscovered later as if it were new.

## 8. Next-Generation Benefit
The strongest test is whether the system starts the next run or generation with a better world model, not merely whether one runtime performed better. If the gain dies with the process, knowledge flow is still incomplete.

## Required Review Questions
Before marking a stage complete, explicitly answer:
1. What ideological debt did this stage remove?
2. Which layer changed?
3. What evidence shows the result is architectural rather than tactical?
4. What new assumptions or walls remain?

## Stage Review Template
Copy this block into stage reports, design closeouts, or evaluation summaries:

```md
## Stage Review

**Ideological debt addressed:** <what architectural problem this stage was meant to remove>

**Layer changed:** `<facts | mechanisms | experience | stimuli>`

**What changed:** <brief description of the implemented change>

**Evidence of improvement:** <tests, eval numbers, diagnostics, comparisons>

**Why this is architectural, not tactical:** <why the gain did not come from env-specific hacks or benchmark-shaped shortcuts>

**Knowledge flow outcome:** <what knowledge was created/updated, where it now lives, whether it persists>

**Remaining assumptions / walls:** <what is still simplified, fragile, unexplained, or blocked>

**Decision:** `<PASS | PARTIAL | FAIL>`
```

## Usage Rules
- Use the template when closing a stage, writing an eval summary, or claiming a meaningful strategy improvement.
- `Decision: PASS` is valid only if the gain survives the checks above, not only because a gate number turned green.
- If the change is useful but still tactically shaped, mark it `PARTIAL` and record the debt explicitly in `docs/ASSUMPTIONS.md`.
- Before upgrading a result from “useful” to “architectural”, run the anti-tuning questions from `docs/ANTI_TUNING_CHECKLIST.md` explicitly.
