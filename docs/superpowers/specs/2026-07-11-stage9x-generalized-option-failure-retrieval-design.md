# Stage9X-B3 — Generalized Option-Failure Retrieval

Date: 2026-07-11  
Status: Design  
Scope: Stage9X learned option arbitration, not a Stage/AGI PASS claim.

## 1. Problem

Stage9X-B proved several mechanics but failed the causal behavior gate.

Validated mechanics:

- option outcomes are stored in the same `VectorWorldModel.memory` SDM;
- sparse negative option failures survive many positive writes via the `__OPTION_FAILURE_H__` role;
- `OptionOutcomeStimulus` can produce negative score;
- interaction-completion is no longer immune to ranking after `4436bcc`.

Remaining failure:

- exact `(OptionContext, StrategyOption)` recall is too local;
- when gen2 changes trajectory, it often reaches a different failure path before matching gen1's exact failure contexts;
- therefore persisted negative memory exists, but does not cause a robust next-run policy change.

This is an architectural wall, not a weight/horizon bug.

## 2. Evidence

Key CUDA artifacts:

- `f2a3381`: gen2 recalled outcomes but `used_for_scoring=0`; writer produced no useful negative recall.
- `44ce31f`: sparse failure role fixed rare-negative washout in unit tests.
- `736e603`: gen2 had `used_for_scoring=186`, but no divergence because interaction completion bypassed ranking.
- `0953241`: candidate-score diagnostic proved selected completion path was absent from ranked candidates.
- `4436bcc`: score-gated completion worked mechanically, but gen2 diverged into a different terminal path before matching gen1 failure contexts.

Conclusion:

```text
exact context recall works only when the future repeats too precisely.
AGI-relevant learning needs controlled generalization of failure memory.
```

## 3. Doctrine check

From `docs/IDEOLOGY.md`:

- Facts: textbook/env facts remain declarative.
- Mechanisms: retrieval/generalization procedure belongs in code.
- Experience: observed option failures are runtime experience.
- Stimuli: negative recall is a death-warning stimulus, not a direct policy.

From `docs/CONCEPT_SUCCESS_CRITERIA.md`:

- A result only matters if retained knowledge changes later planning.
- Correlation is insufficient; the retained failure memory must be causally useful.

From `docs/ANTI_TUNING_CHECKLIST.md`:

- The mechanism must be describable without naming Crafter entities.
- It must not become a hand-written crisis policy.

Therefore B3 must not add rules like "if thirsty, choose water" or "if zombie, flee".
It should add a general memory retrieval mechanism: exact and abstract failure recall over option contexts.

## 4. Design

### 4.1 Context abstraction levels

For every `OptionContext`, derive a small hierarchy of abstraction keys:

1. `full`
   - all current fields;
2. `drop_progress`
   - omit `progress_state`;
3. `drop_intent`
   - omit `intent_state` and `progress_state`;
4. `need_threat_capability`
   - keep only vital buckets, `threat_pressure`, `local_restore`, `capability_state`;
5. `need_threat`
   - keep only vital buckets and `threat_pressure`;
6. `vitals_only`
   - keep only health/food/drink/energy buckets.

The hierarchy is generic. It does not mention tree, water, zombie, skeleton, or any Crafter entity.

### 4.2 Option abstraction levels

For every `StrategyOption`, derive option keys:

1. `exact`
   - current `option_id`, e.g. `seek_known:tree`;
2. `kind_target_class`
   - option kind + target class if textbook/entity metadata supports a class;
3. `kind`
   - option kind only, e.g. `seek_known`;
4. `any`
   - context-only failure warning.

Initial B3 can implement `exact` and `kind` only. Target classes are deferred until textbook vocabulary flags are clean enough.

### 4.3 Storage

On negative option outcome, write to the same SDM memory under role-isolated addresses:

```text
bind(bind(context_abstraction_vec, option_abstraction_vec), role__OPTION_FAILURE_H__:<level>)
```

This is still one substrate. It is not a second episodic SDM.

Positive outcomes remain in the aggregate option-outcome role. B3 should not boost positive recall.

### 4.4 Retrieval

`predict_option_outcome(context, option_id)` should retrieve in priority order:

1. exact failure;
2. less-specific failure levels;
3. aggregate exact outcome.

Return trace metadata:

```json
{
  "decoded": {...},
  "confidence": 0.73,
  "retrieval_level": "drop_intent/kind",
  "failure_role": "__OPTION_FAILURE_H__",
  "abstraction_penalty": 0.6
}
```

The stimulus score should be scaled by abstraction specificity:

- exact: `1.0`
- drop_progress: `0.85`
- drop_intent: `0.7`
- need_threat_capability: `0.55`
- need_threat: `0.4`
- vitals_only: `0.25`

These are not policy weights for Crafter behavior. They are confidence penalties for less-specific memory retrieval.

### 4.5 Causal precursor credit

When an episode fails, B3 should not only label the final pending option.

For the last `N` selected options before a real terminal/critical failure:

- write negative outcome to exact context/option;
- write abstracted failure keys with a horizon distance field;
- decay confidence by distance from failure.

Initial value:

- `N = option_outcome_horizon * 2`
- distance decay: `1.0` at terminal, `0.5` at farthest precursor.

This is not backpropagation through a neural net. It is a trace-visible credit assignment policy for episodic failure memory.

### 4.6 Trace requirements

Every option recall should report:

- exact/abstract retrieval level;
- option abstraction level;
- confidence;
- decoded outcome;
- whether it affected scoring;
- final scalar contribution.

Candidate debug should expose the same fields for top candidates.

Without this trace, B3 cannot be claimed as causal.

## 5. Non-goals

- No new Crafter-specific crisis rule.
- No positive recall boosting.
- No direct override such as "choose water".
- No new parallel SDM.
- No Stage/AGI PASS claim from one seed.
- No tuning `option_outcome_weight` as a substitute for causal retrieval.

## 6. Validation ladder

### Unit gates

1. Exact failure still wins over aggregate survived writes.
2. Abstract failure is returned when exact context misses.
3. Exact safe/no-failure context is not penalized by unrelated abstract failure above its confidence threshold.
4. Retrieval trace reports the abstraction level.
5. Primitive outcome and physics roles remain unpolluted.

### Smoke gate

Construct two nearby contexts:

- gen1 writes failure in one context;
- gen2 queries a neighboring context that differs only in `intent_state` or `goal_family`;
- `OptionOutcomeStimulus` returns a scaled negative score with trace level `drop_intent` or similar.

### Behavioral gate

Use the existing seed17 paired probe:

- gen1 writes failure memory;
- gen2 shows negative scoring before the terminal range;
- first divergence is explainable by candidate-score debug;
- divergence is not caused by random trace length or unrelated ranking noise.

### Ablation

Run gen2 with generalized failure retrieval disabled:

- if behavior is identical to B3 enabled, B3 fails;
- if only B3 enabled changes candidate ranking, B3 passes the local causal gate.

## 7. Expected risk

Generalized failure memory can over-penalize broad contexts. The safety valve is not a hand-coded exception; it is trace-visible confidence:

- exact failures dominate;
- abstract failures are weaker;
- repeated contradictory exact survival can override broad warnings;
- every penalty must show its retrieval level.

If broad warnings suppress exploration too much, the issue should be solved by better abstraction/confidence, not by Crafter-specific policy exceptions.

## 8. Claim boundary

B3 can only claim:

> The agent can retrieve learned negative option outcomes from a neighboring context and use them as a planning stimulus.

B3 cannot claim:

- the concept works;
- Stage9X is closed;
- Crafter is solved;
- AGI/world-model proof is achieved.

