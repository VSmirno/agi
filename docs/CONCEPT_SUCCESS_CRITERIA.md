# Concept Success Criteria

This document defines the conditions under which the repository can honestly claim that the core concept works. It is stricter than a stage gate: a stage may pass while the concept still remains unproven.

## 1. Next Generation Must Be Better
The strongest signal is cross-generation improvement.

Required evidence:
- `genN+1` outperforms `genN` on the repository’s primary task metric
- the gain repeats across independent runs, not one lucky seed
- the inherited knowledge responsible for the gain is identified and inspectable

If knowledge flow does not make later generations measurably better, the concept is not yet proven.

## 2. Improvement Must Come From the Right Layer
A positive metric change only counts if it is explained by an architectural layer:
- better `facts`
- better `mechanisms`
- better `experience` carried forward correctly
- better `stimuli`

Threshold tuning, scoring tweaks, and Crafter-specific patches do not count as concept validation.

## 3. The System Must Learn Causally Useful Knowledge
The system must show at least one clean case of:
`observation -> hypothesis -> verification -> retained knowledge -> better behavior`

That retained knowledge must be causally useful, not just a correlation discovered after the fact. Promoting patterns like “zombie + low drink” is not enough if low drink is merely a consequence of combat rather than a cause of failure.

## 4. The Architecture Must Generalize to a Neighboring Case
The concept is not validated if every new threat, object, or dynamic requires a new hand-built mechanism. At least one new neighboring case should be absorbed by the same architecture with local textbook/model changes rather than planner rewrites or special policy branches.

## 5. Better Planning Must Follow From Better World Understanding
The cleanest proof is:
- the world model is wrong about a concrete dynamic,
- learning reduces that prediction error,
- planning improves specifically because the model became more accurate,
- the improvement survives into the next run or generation.

If behavior improves without a demonstrated improvement in world understanding, the concept is still unproven.

## Minimal Claim Threshold
The repository may claim that “the concept works” only when all of the following are true:
1. Cross-generation benefit is demonstrated.
2. The benefit comes from the correct architectural layer.
3. At least one promoted or persisted piece of knowledge is causally useful.
4. The same architecture handles at least one neighboring case without a bespoke control patch.
5. Planning improves because the world model improved.

## Anti-Claims
The repository should explicitly avoid stronger claims when the evidence only shows:
- one-run adaptation without persistence
- correlation capture without causal validation
- benchmark gains from tactical heuristics
- improvements that disappear in the next generation
- progress confined to one fragile environment-specific loop
