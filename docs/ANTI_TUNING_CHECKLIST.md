# Anti-Tuning Checklist

Use this checklist before declaring a new strategy, stage result, or architectural change meaningful. Its purpose is to catch the point where AGI-oriented research quietly degrades into environment-specific benchmark tuning.

This checklist is not a replacement for `docs/STAGE_REVIEW_CRITERIA.md` or `docs/CONCEPT_SUCCESS_CRITERIA.md`. It is a guardrail that should be applied alongside them.

## 1. Can the change be described without naming the current environment?

Good:
- short-horizon dynamic threat forecasting
- local safety vs long-horizon viability tradeoff
- inter-generation knowledge transfer

Bad:
- dodge skeleton arrows better in Crafter
- survive zombies at low drink

If the change cannot be explained without naming `Crafter`, `arrow`, `zombie`, or another environment-specific entity, it is likely too narrow.

## 2. Does the change live in the correct layer?

Every improvement must still fit one of:
- `facts`
- `mechanisms`
- `experience`
- `stimuli`

If the real answer is “we inserted a small special case into planner/policy code”, this is a warning sign even if the benchmark improved.

## 3. Is the gain explained by a general capability, not a case-specific trick?

Good:
- the world model now represents moving hazards
- the planner now distinguishes immediate safety from delayed viability

Bad:
- the agent now knows the right move for this one enemy pattern

If the explanation depends on one environment’s named mechanic instead of a broader capability class, the gain is probably tactical.

## 4. Would the idea still make sense in a neighboring domain?

The question is not whether it would work instantly somewhere else. The question is whether the mechanism would still be a meaningful architectural idea outside the current task.

If the answer is no, the change is probably environment tuning.

## 5. Are we proving a capability or only moving a score?

A metric increase alone is not enough.

Acceptable evidence should show:
- what capability was added,
- where it lives in the architecture,
- why the score changed as a consequence.

If the only defensible claim is “the number went up”, treat the result as suspicious until proven otherwise.

## Required Final Question

Before calling a stage or strategy successful, answer explicitly:

**Did we just add a general cognitive mechanism, or only another way to do better on the current environment?**

If the answer leans toward the second, mark the result as at least partially tactical and record the debt explicitly.
