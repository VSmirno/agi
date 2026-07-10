# Stage9X AGI-Relevance Proof Plan

**Date:** 2026-07-10
**Status:** Plan
**Scope:** Stage9X methodology, proof gates, and near-term validation order.
**Companion docs:** `docs/IDEOLOGY.md`,
`docs/ANTI_TUNING_CHECKLIST.md`,
`docs/CONCEPT_SUCCESS_CRITERIA.md`,
`docs/STAGE_REVIEW_CRITERIA.md`,
`docs/ASSUMPTIONS.md`,
`docs/superpowers/specs/2026-05-21-stage9x-learned-option-arbitration-design.md`,
`docs/superpowers/specs/2026-05-22-stage9x-combat-affordance-closure-design.md`.

---

## 1. Purpose

This plan prevents Stage9X from being misread as "solve Crafter" work.

Crafter remains the MVP environment, but a local Crafter pass is not enough to
claim progress toward AGI or a general world model. Stage9X is AGI-relevant
only if it demonstrates inspectable causal knowledge flow:

```text
prediction/outcome error
  -> retained knowledge
  -> changed planning
  -> better later behavior
```

Affordance execution is necessary for that proof, but it is not the proof.

---

## 2. Stage Split

### Stage9X-A: Affordance Executability

Goal: make existing strategy options physically executable.

This includes:

- `seek_known:<target>` reducing distance to a known mapped target;
- `seek_frontier:<target>` exploring toward unknown goal targets;
- `complete_interaction:<target>:<action>` performing approach -> align -> act
  -> verify;
- `craft_capability:<item>` decomposing into gather/place/make work;
- `engage_target:<entity>` respecting action geometry before attack.

Claim allowed: "The option interface is executable enough to test learning."

Claim not allowed: "The AGI/world-model concept is proven."

### Stage9X-B: Learned Option Arbitration

Goal: show that persisted option-outcome memory changes strategy selection in
compound contexts.

The relevant context class is generic: simultaneous homeostatic pressure,
threat pressure, capability constraints, and active intent. The implementation
must not add a hand-written crisis controller.

Claim allowed only if the trace shows:

- the same or similar `OptionContext` occurred before;
- a selected option produced a bad or interrupted outcome;
- the later run recalls that outcome;
- the recall changes candidate ranking before the terminal window.

### Stage9X-C: Causal World-Model Proof

Goal: show that planning improves because the model of the world became more
accurate or more causally useful.

The minimum proof chain is:

```text
wrong prediction or bad outcome
  -> hypothesis/update/write
  -> persisted causal knowledge
  -> different rollout/ranking
  -> better later behavior
```

This is the first level that can support a concept-level AGI/world-model claim.

---

## 3. Mandatory Gates

### Gate 1: Exact-Commit Behavioral Evidence

Before claiming any Stage9X behavior change:

- validate the exact commit under full-profile settings;
- record seed 17 episode 0 MP4 plus JSON/local trace;
- inspect the video and trace before reporting;
- deliver the MP4 in the same report as the verification result.

Smoke-lite runs are not valid Stage9X baselines.

### Gate 2: No Crafter-Policy Gate

The change must be describable without naming the current environment or its
entities. Acceptable descriptions:

- option-level conflict arbitration under simultaneous needs and threats;
- action-geometry closure for target interactions;
- retained outcome memory changing plan ranking.

Suspicious descriptions:

- survive zombies better;
- drink water earlier;
- dodge skeleton arrows.

Named environment facts may live in textbook/config. They must not become new
planner policy branches.

### Gate 3: Causal Trace Gate

A metric gain is insufficient. The trace must identify:

- the relevant context;
- candidate options;
- recalled outcome or promoted knowledge;
- scoring/ranking difference;
- selected primitive;
- resulting behavior difference.

If the only defensible result is "survival improved", the result is tactical.

### Gate 4: Ablation Gate

For learned arbitration or world-model claims, compare:

- same code and seed with persisted memory disabled or removed;
- same code and seed with persisted memory enabled;
- same validation profile.

The behavior difference must appear before the terminal failure window and be
trace-explainable.

### Gate 5: Cross-Generation Gate

`genN+1` must improve over `genN` in both:

- task outcome, such as survival or death cause;
- trace evidence, such as fewer repeated bad options in similar contexts.

The inherited knowledge responsible for the change must be inspectable.

### Gate 6: Neighboring-Case Transfer Gate

At least one neighboring case must be handled by changing facts/adapters, not
planner policy code. Examples:

- a new hostile type;
- a new resource or restoration affordance;
- a new station/action-geometry fact;
- a synthetic mini-environment with the same option abstractions and different
  names/facts.

If every new case requires editing `vector_mpc_agent.py` policy branches, the
stage remains local engineering, not AGI-relevant architecture.

---

## 4. Work Plan

### Phase 0: Validate Current `cf076af` Slice

Owner: implementation agent.

Tasks:

1. Fix or reconcile the local test mismatch around `expected_effect: {}`.
2. Run focused combat-affordance tests.
3. Push the exact validation commit through the normal git-flow path.
4. Run seed 17 episode 0 full-profile recording on HyperPC.
5. Inspect MP4 and JSON trace.
6. Report only the affordance-executability claim.

Done criteria:

- focused tests pass;
- exact-commit full-profile video and trace exist;
- adjacent armed hostile interactions show align-before-attack behavior;
- hostile removal verification limitations are recorded if still present.

Estimated effort: 0.5-1 day.

### Phase 1: Close Executable Option Contracts

Owner: implementation agent.

Tasks:

1. Audit all strategy options for lifecycle trace fields.
2. Ensure each option has success and failure reasons.
3. Verify that baseline motion no longer hides missing option generation.
4. Record seed 17 episode 0 after any behavior change.

Done criteria:

- each option can be classified as success, failed, interrupted, or abandoned;
- repeated baseline-heavy behavior has a trace-visible reason;
- no new environment-specific policy branches are introduced.

Estimated effort: 1-2 days after Phase 0.

### Phase 2: Prove Learned Option Arbitration

Owner: implementation agent plus reviewer.

Tasks:

1. Define the minimal `OptionContext` buckets needed for compound conflict.
2. Write bad/interrupted outcomes, not only death outcomes.
3. Ensure read-side option recall can affect ranking.
4. Run paired gen1/gen2 seed 17 episode 0 validation.
5. Run ablation with persisted option memory disabled.

Done criteria:

- gen2 selects a different option in a similar context;
- trace shows the recall that changed ranking;
- the ranking change happens before the terminal window;
- disabling persisted memory removes the behavior difference.

Estimated effort: 2-4 days.

### Phase 3: Prove Causal World-Model Improvement

Owner: implementation agent plus reviewer.

Tasks:

1. Select one concrete wrong prediction or bad model assumption.
2. Show before/after prediction or rollout error.
3. Persist the learned correction or causal outcome.
4. Show that planning changes because rollout/ranking changed.
5. Run cross-generation validation.

Done criteria:

- retained knowledge is inspectable;
- planning changes because model recall/prediction changed;
- gen2 improves over gen1;
- the result is not explained by threshold tuning.

Estimated effort: 3-5 days.

### Phase 4: Neighboring-Case Transfer

Owner: implementation agent plus reviewer.

Tasks:

1. Choose a minimal neighboring case.
2. Add the case through textbook/config or env adapter.
3. Reuse the same option lifecycle and arbitration path.
4. Validate behavior and trace without new planner policy branches.

Done criteria:

- the new case works through the same mechanism;
- no new target-specific policy branch is needed;
- traces show the same option and learning abstractions.

Estimated effort: 2-4 days for the first minimal case.

---

## 5. Kill Gates

Stop local patching and reframe if any of these repeats for 2-3 iterations:

- each new failure requires another planner special case;
- option recall is present but never changes ranking;
- gen2 and gen1 remain behaviorally identical;
- survival changes without trace-causal explanation;
- neighboring cases require new policy code;
- improvements depend on smoke-lite or non-reproducible runs.

---

## 6. Claim Rules

Allowed after Stage9X-A:

- "The agent's option interface is more executable."
- "The current system is ready to test learned arbitration."

Allowed after Stage9X-B:

- "Persisted option outcomes can change strategy selection in a repeated
  compound context."

Allowed after Stage9X-C plus transfer:

- "The MVP shows a minimal causal world-model loop: experience changes retained
  knowledge, retained knowledge changes planning, and the change transfers to a
  neighboring case."

Not allowed without all mandatory gates:

- "The AGI concept works."
- "Stage9X proves general intelligence."
- "Crafter success validates the architecture."

