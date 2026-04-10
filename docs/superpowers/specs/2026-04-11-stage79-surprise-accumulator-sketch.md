# Stage 79 — Surprise Accumulator + Rule Nursery (design sketch)

**Date:** 2026-04-11
**Status:** Sketch — waiting on Stage 78a verdict to pick substrate (DAF residual vs MLP residual vs rules-only). Stage 79 itself is **independent** of that choice because surprise flows from the `simulate_forward` prediction vs actual outcome, regardless of what generates the prediction.

## Goal

Let the agent **discover rules the textbook does not contain** by watching its own prediction errors, without an LLM in the loop.

> Per Strategy v5 novelty #2: "Rule induction from surprise accumulator without LLM" — all published neurosymbolic WMs (Neuro-Symbolic Synergy, OneLife, IDEA) use an LLM to synthesize candidate laws. Our project has no LLM access and ideologically rejects it in the live loop. Template-based + surprise-clustering induction is the alternative.

## The conjunctive test case

From Stage 78a's synthetic rule:
```
sleep + (food == 0 OR drink == 0) → health -0.067
```
The textbook does not carry this. An agent running MPC will predict
`health ≈ +0.04` (the non-starving sleep rule) and observe actual
`health = -0.067`. That's a **prediction error of -0.107** on the
sleep action, specifically when food or drink is zero. The accumulator
must notice that this error repeats in the same context and emit a
candidate rule.

## Pipeline

```
  simulate_forward(plan)         ┐
         ↓                       │
  predicted_body_delta            │
                                  ├──→  surprise = actual - predicted
  env.step(primitive)             │
         ↓                        │
  actual_body_delta              ┘
         ↓
  SurpriseAccumulator.observe(context, primitive, surprise)
         ↓
  bucket = (context_key(state), primitive)
         ↓
  bucket.append(surprise)
         ↓
  if bucket.n >= MIN_OBS and bucket.consistent():
         ↓
      RuleNursery.emit_candidate(precondition, effect_delta)
         ↓
      verify over next N observations
         ↓
  promote → committed rule in ConceptStore
```

## Data structures

```python
@dataclass
class ContextKey:
    """Hashable precondition fingerprint — 'what the world looked like'."""
    visible: frozenset[str]          # visible concept IDs (no body values)
    body_quartiles: tuple[int, ...]  # (h_q, f_q, d_q, e_q) each 0..3
    action: str                      # primitive
    # (Design choice: exclude body raw values to keep the fingerprint
    # coarse enough to aggregate. Quartiles give 256 possible body
    # regions instead of ~10000.)

@dataclass
class SurpriseRecord:
    context: ContextKey
    predicted: dict[str, float]
    actual: dict[str, float]
    delta: dict[str, float]         # = actual - predicted
    tick_id: int

@dataclass
class CandidateRule:
    """Emitted by the nursery once a bucket shows consistent error.

    Two promotion gates:
      1. MIN_OBS: bucket has at least N records.
      2. CONSISTENT: the mean error has low relative variance AND
         |mean| > 2 * MAD (one-sided outlier rejection).
      3. DIFFERENT: mean error deviates from 0 by more than the
         verify threshold (±0.01 per body var).
    """
    context: ContextKey
    mean_effect: dict[str, float]
    n_obs: int
    mad: dict[str, float]
    status: str  # "pending" | "verifying" | "promoted" | "rejected"
    verify_records: list[SurpriseRecord]
```

## Context keying — two levels

**Level 1 (coarse, always used):** `(visible_concepts, action)` — 9 concepts × 17 actions = 153 possible buckets, very trackable. Good for disjunctive effects like skeleton_in_view → health down.

**Level 2 (refined, triggered when L1 bucket hits MIN_OBS):** split L1 bucket by `body_quartiles` — captures body-state-dependent rules like the conjunctive sleep+starvation case. Only refined when L1 mean has high variance (>2 MAD), otherwise the refinement is not informative.

This two-level scheme avoids the combinatorial explosion of keying by full body state from the start, while still catching conjunctive rules when the evidence demands finer resolution.

## Template-based candidate synthesis

No LLM needed because the "candidate" is simply the empirical mean error of the accumulated records:

```python
def emit_candidate(bucket: list[SurpriseRecord]) -> CandidateRule | None:
    if len(bucket) < MIN_OBS:
        return None
    mean_effect = {}
    for var in BODY_VARS:
        errs = [r.delta[var] for r in bucket]
        mean_effect[var] = np.mean(errs)
    # MAD-based consistency check
    mad = {var: np.median(np.abs(np.array([r.delta[var] for r in bucket]) - mean_effect[var]))
           for var in BODY_VARS}
    significant = {var: abs(mean_effect[var]) > max(2 * mad[var], 0.01)
                   for var in BODY_VARS}
    if not any(significant.values()):
        return None
    return CandidateRule(
        context=bucket[0].context,
        mean_effect={v: mean_effect[v] for v in BODY_VARS if significant[v]},
        n_obs=len(bucket),
        mad=mad,
        status="pending",
        verify_records=[],
    )
```

This is **template** in the sense that the rule structure is fixed:
*"when context X observed, body delta is Y for vars Z"*. The content
(X, Y, Z) is learned, the shape is programmed.

## Verification loop

```python
def verify(candidate: CandidateRule, new_record: SurpriseRecord) -> bool:
    """Incrementally confirm a candidate as more evidence arrives."""
    if new_record.context != candidate.context:
        return False
    candidate.verify_records.append(new_record)
    if len(candidate.verify_records) < VERIFY_N:
        return False
    # Mean-of-deltas matches the proposed effect (within tolerance)
    ok = True
    for var, effect in candidate.mean_effect.items():
        observed_mean = np.mean([r.delta[var] for r in candidate.verify_records])
        if abs(observed_mean - effect) > VERIFY_TOL:
            ok = False
            break
    candidate.status = "promoted" if ok else "rejected"
    return ok
```

## Promotion to ConceptStore

```python
def promote(candidate: CandidateRule, store: ConceptStore) -> None:
    store.add_learned_rule(
        precondition=candidate.context,   # fingerprint → ConceptStore.Rule predicate
        effect=candidate.mean_effect,
        source="runtime_nursery",
        confidence=0.5,                    # start at neutral, grows with use
    )
```

The added rule then participates in `simulate_forward` on future ticks,
which reduces prediction error on that context and naturally suppresses
further surprise records there — the bucket dries up once the rule is
committed. Confidence grows as the rule continues to match observations
(existing `verify_outcome` machinery).

## Ideology check

- **No supervised backprop on labels** — the nursery only takes means
  of observed errors; no gradient descent.
- **No LLM in loop** — candidate structure is template-fixed; only
  the numeric content is learned from data.
- **Textbook unchanged** — new rules go into a `learned_rules` list,
  orthogonal to the YAML textbook. Facts (textbook) and experience
  (nursery) are kept separate per three-category taxonomy.
- **Self-induced** — the whole pipeline runs on `actual - predicted`
  with no oracle teacher; the agent's own prediction error IS the
  training signal.

## Dependencies

- `ConceptStore.simulate_forward` → already in place from Stage 77a
- `ConceptStore.verify_outcome` → already in place, handles confidence
- `HomeostaticTracker` body_rates → already in place
- `CrafterPixelEnv` body_delta per step → need to confirm it's exposed

## Integration hooks

- Add `SurpriseAccumulator` to the MPC tick loop: after each `env.step`,
  compute `surprise = actual_delta - predicted_delta` and feed to the
  accumulator.
- Add `RuleNursery.tick()` to the same loop for periodic candidate
  emission and verification.
- Extend `ConceptStore` with `learned_rules: list[Rule]` and include
  them in `simulate_forward`'s rule application phase.

## Gates (proposed)

- **Unit tests:** emit candidate from synthetic bucket; verify promotion
  after VERIFY_N consistent records; verify rejection after inconsistent.
- **Synthetic integration:** inject surprise for the conjunctive sleep
  rule; nursery must emit, verify, and promote within 200 episodes.
- **Crafter eval:** wood ≥50%, survival ≥220, with nursery promoting
  at least one non-trivial learned rule per episode avg.

## Out of scope for Stage 79

- Conjunction grammar (AND/OR preconditions richer than context fingerprint)
- Negation / absence rules ("when X is NOT visible")
- Composition of multiple learned rules
- Pruning of obsolete learned rules
- Rule generalization (merging two similar buckets)

These are deferred to Stage 80+ if Stage 79's flat context-key approach
proves insufficient.

## Open questions

1. **Quartile boundaries:** fixed thresholds (0,3,6,9) vs quantile-derived? Fixed is simpler and interpretable; quantile adapts to runtime distribution.
2. **Bucket staleness:** should old records age out? Argument for yes: distribution shift (new area, new enemies). Argument for no: stable rules should stay. Compromise: sliding window of last 100 obs per bucket.
3. **Ownership of tick_id:** who allocates? Simplest: MPC loop increments a monotone counter.
4. **Threshold for "consistent":** 2×MAD is a heuristic; no theoretical justification. May need empirical tuning.

---

This sketch is **not ready to implement** — it needs the Stage 78a
verdict to confirm whether the substrate also needs plumbing into the
nursery (if DAF residual works, the residual's own error signal can
also drive surprise buckets, giving two streams of evidence). When
Stage 78a closes, promote this sketch to a full implementation plan.
