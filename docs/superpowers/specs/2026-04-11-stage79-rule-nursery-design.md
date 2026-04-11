# Stage 79 — Surprise Accumulator + Rule Nursery (full design)

**Date:** 2026-04-11
**Status:** Design — promoted from sketch (`2026-04-11-stage79-surprise-accumulator-sketch.md`) after Stage 78c partial-FAIL closed the substrate question.
**Strategy:** Strategy v5 novelty #2 — no-LLM symbolic rule induction.
**Replaces sketch decisions:** none (sketch is mostly preserved; this doc adds module structure, constants, integration points, tests, and the residual co-existence policy).

## Why Stage 79 is now the only option on the table

Stage 78a (DAF substrate × MLP head, 7 regimes, synthetic conjunctive
task) and Stage 78c (MLP residual over `simulate_forward` + online SGD,
real Crafter rollouts) **independently reproduced the same failure
mode** — the discrimination paradox documented in `stage-78a-report.md`.
Both training paradigms minimise an MSE-on-labels loss against a
multi-modal target distribution, and both produce a confidently-wrong
predictor that **amplifies** the wrong direction on the conditional
minority cases. The recipe is independent of substrate (FHN spiking
network or 67K MLP), feature extraction (voltage / spikerate /
sks_cluster or hashed concept bits + body buckets), and training
signal source (oracle deltas or env rollouts).

The recipe breaks only if learning produces **explicit conditional
structure** — discrete `(precondition, effect)` rules whose preconditions
non-overlappingly partition the state space — instead of a single
averaged correction. Stage 79 builds exactly that:

```
   surprise = actual_body_delta - predicted_body_delta
       ↓
   bucket by (visible_concepts, body_quartiles, action)
       ↓
   when bucket is large enough AND mean error is consistent
       ↓
   emit CandidateRule(precondition=bucket_key, effect=mean_error)
       ↓
   verify over next N observations
       ↓
   promote to ConceptStore.learned_rules
       ↓
   simulate_forward picks it up on the next plan, error vanishes from
   that bucket, the bucket dries up
```

No averaging across heterogeneous contexts. No gradient on labels.
No LLM. The agent's own prediction error IS the training signal, and
the learned object is a discrete predicate-effect pair, not a weight
vector.

## What's already in place

| Component | Where | Status |
|---|---|---|
| `ConceptStore.simulate_forward` | `src/snks/agent/concept_store.py` | Stage 77a; takes optional residual + visible_concepts |
| `simulate_forward` Phase 4 (stateful) | `concept_store.py:_apply_tick` | matches `stateful_condition` against sim — **this is the hook** for `learned_rules` to fire |
| `verify_outcome` confidence updates | `concept_store.py` | grows confidence on rule firings that match observations |
| `HomeostaticTracker.observed_rates` | `src/snks/agent/perception.py` | already accumulates per-var rates from observations |
| `run_mpc_episode` surprise measurement | `src/snks/agent/mpc_agent.py:560` | already computes `surprise = actual - predicted_next_body` per step (currently only when trace is enabled) |
| Body order convention | `RESIDUAL_BODY_ORDER` in `concept_store.py` | reused for nursery's body delta vector |

## What Stage 79 adds

Three new modules + an integration in two existing modules:

1. **`src/snks/learning/surprise_accumulator.py`** (NEW, ~250 lines)
   - `ContextKey` (frozen dataclass): the bucket fingerprint
   - `SurpriseRecord`: one observation
   - `SurpriseAccumulator`: per-context bucket store; `observe(...)`,
     `bucket_size(key)`, `bucket_mean_effect(key)`, `bucket_mad(key)`,
     `iter_buckets()`
   - `MIN_OBS_L1=5`, `MIN_OBS_L2=10`, `MAX_BUCKET_SIZE=100` (sliding
     window — answers Open Question #2 from sketch)

2. **`src/snks/learning/rule_nursery.py`** (NEW, ~200 lines)
   - `CandidateRule` (mutable dataclass): pending → verifying → promoted
   - `RuleNursery`: holds candidates; `tick(accumulator, store)` runs
     emission + verification + promotion in one pass per env step
   - `VERIFY_N=10`, `VERIFY_TOL=0.02`, `MAD_K=2.0` (consistency threshold)
   - `promote(...)` writes to `ConceptStore.learned_rules`

3. **`src/snks/agent/learned_rule.py`** (NEW, ~80 lines)
   - `LearnedRule` dataclass: `precondition: ContextKey`, `effect:
     dict[str, float]`, `confidence: float`, `n_observations: int`,
     `source: str = "runtime_nursery"`
   - `LearnedRule.matches(visible, body, action) -> bool`: predicate
     evaluation against current sim state in `simulate_forward`
   - `LearnedRule.apply_to_sim(sim) -> list[SimEvent]`: mutates
     `sim.body` and emits events for `traj.events`

4. **`src/snks/agent/concept_store.py`** (MODIFY, small):
   - Add `self.learned_rules: list[LearnedRule] = []` to
     `ConceptStore.__init__`
   - Add `add_learned_rule(...)` helper
   - In `_apply_tick` add **Phase 7** (after Phase 6 action effects,
     before clamp): iterate `learned_rules`, fire matching ones
   - Phase 7 is positioned AFTER the existing rules so that learned
     rules see the post-rule body and add corrections on top — they
     are *additive* to the textbook, not replacements

5. **`src/snks/agent/mpc_agent.py`** (MODIFY, small):
   - `run_mpc_episode` accepts `surprise_accumulator` and `rule_nursery`
     (both optional, default None)
   - After `env.step` and existing surprise computation: feed
     `(context, surprise)` to accumulator
   - Periodically (every N env steps, not every step) call
     `nursery.tick(accumulator, store)`
   - At episode end: log nursery state (n candidates, n promoted,
     n rejected) into return dict for harness reporting

The residual code from Stage 78c **stays in place** but defaults off
during Stage 79 evaluation. We do not delete it because:
- it does pass on warmup_a (no-enemy regime), so it has a use case
- the wiring tests it added validate the simulate_forward signature
- a follow-up Stage 81 (alternating training, Neuro-Symbolic Synergy
  pattern) might re-enable it gated on rule coverage

## Module API specifications

### `ContextKey`

```python
@dataclass(frozen=True)
class ContextKey:
    """Bucket fingerprint for surprise accumulation. Hashable."""
    visible: frozenset[str]
    body_quartiles: tuple[int, int, int, int]  # (h, f, d, e), each 0..3
    action: str  # primitive ('do', 'sleep', 'move_left', 'place_stone', ...)

    @classmethod
    def from_state(
        cls,
        visible: set[str],
        body: dict[str, float],
        action: str,
        body_order: tuple[str, ...] = RESIDUAL_BODY_ORDER,
    ) -> "ContextKey":
        quartiles = tuple(
            min(3, max(0, int(body.get(var, 0.0) // 2.5)))
            for var in body_order
        )
        return cls(
            visible=frozenset(visible),
            body_quartiles=quartiles,
            action=action,
        )

    def coarsen(self) -> "ContextKey":
        """L2 → L1 by zeroing body_quartiles. Used to also feed L1 buckets."""
        return ContextKey(
            visible=self.visible,
            body_quartiles=(0, 0, 0, 0),
            action=self.action,
        )
```

**Quartile boundaries (resolves Open Question #1 from sketch):** fixed
thresholds at body values 2.5, 5.0, 7.5. This puts food=0 (the
canonical conjunctive case from 78a) cleanly in quartile 0, food=9
(full) in quartile 3, with two intermediate buckets. Fixed beats
quantile-derived because it stays interpretable across episodes and
makes tests deterministic.

### `SurpriseRecord` and `SurpriseAccumulator`

```python
@dataclass
class SurpriseRecord:
    context: ContextKey
    predicted: dict[str, float]
    actual: dict[str, float]
    delta: dict[str, float]  # actual - predicted, per body var
    tick_id: int


class SurpriseAccumulator:
    """Per-context buckets of (predicted, actual, delta) records.

    Two-level keying: every observation feeds BOTH the L2 bucket (full
    ContextKey) and the L1 bucket (key.coarsen()). The nursery emits
    candidates from L2 buckets that meet MIN_OBS_L2 first, falling back
    to L1 only if no L2 bucket is significant — this captures
    body-state-dependent rules like the conjunctive sleep+starvation
    case while still detecting body-independent rules like
    skeleton_in_view → health drop.
    """

    def __init__(self, max_bucket_size: int = 100) -> None:
        self._buckets: dict[ContextKey, deque[SurpriseRecord]] = defaultdict(
            lambda: deque(maxlen=max_bucket_size)
        )

    def observe(
        self,
        context: ContextKey,
        predicted: dict[str, float],
        actual: dict[str, float],
        tick_id: int,
    ) -> None:
        delta = {
            var: actual.get(var, 0.0) - predicted.get(var, 0.0)
            for var in RESIDUAL_BODY_ORDER
        }
        record = SurpriseRecord(context, dict(predicted), dict(actual), delta, tick_id)
        self._buckets[context].append(record)
        # Also feed L1
        l1_key = context.coarsen()
        if l1_key != context:
            self._buckets[l1_key].append(record)

    def bucket_size(self, key: ContextKey) -> int:
        return len(self._buckets.get(key, ()))

    def bucket_records(self, key: ContextKey) -> list[SurpriseRecord]:
        return list(self._buckets.get(key, ()))

    def iter_buckets(self) -> Iterator[tuple[ContextKey, list[SurpriseRecord]]]:
        for key, dq in self._buckets.items():
            yield key, list(dq)

    def stats(self) -> dict[str, Any]:
        return {
            "n_buckets": len(self._buckets),
            "total_records": sum(len(b) for b in self._buckets.values()),
            "max_bucket_size": max((len(b) for b in self._buckets.values()), default=0),
        }
```

### `CandidateRule` and `RuleNursery`

```python
@dataclass
class CandidateRule:
    context: ContextKey
    mean_effect: dict[str, float]  # var → mean delta, only significant vars
    n_obs: int
    mad: dict[str, float]
    status: str  # "pending" | "verifying" | "promoted" | "rejected"
    verify_records: list[SurpriseRecord] = field(default_factory=list)
    emitted_at_tick: int = 0


class RuleNursery:
    MIN_OBS_L1: int = 5
    MIN_OBS_L2: int = 10
    VERIFY_N: int = 10
    VERIFY_TOL: float = 0.02
    MAD_K: float = 2.0
    SIGNIFICANCE_FLOOR: float = 0.01

    def __init__(self) -> None:
        # context → CandidateRule (one in flight per context at a time)
        self._candidates: dict[ContextKey, CandidateRule] = {}
        self._stats = {"emitted": 0, "promoted": 0, "rejected": 0}

    def tick(
        self,
        accumulator: SurpriseAccumulator,
        store: ConceptStore,
        current_tick: int,
    ) -> None:
        """Per-step: emit new candidates from saturated buckets, verify
        existing candidates, promote/reject as gates fire."""
        for key, records in accumulator.iter_buckets():
            if key in self._candidates:
                continue  # already tracking
            min_obs = self.MIN_OBS_L1 if key.body_quartiles == (0, 0, 0, 0) else self.MIN_OBS_L2
            if len(records) < min_obs:
                continue
            candidate = self._try_emit(records, key, current_tick)
            if candidate is not None:
                self._candidates[key] = candidate
                self._stats["emitted"] += 1

        # Verify existing
        to_resolve: list[ContextKey] = []
        for key, candidate in self._candidates.items():
            if candidate.status not in ("pending", "verifying"):
                continue
            new_records = accumulator.bucket_records(key)[len(candidate.verify_records):]
            for r in new_records:
                self._add_verify(candidate, r)
                if len(candidate.verify_records) >= self.VERIFY_N:
                    self._resolve(candidate, store)
                    to_resolve.append(key)
                    break

        # House-keeping
        for key in to_resolve:
            cand = self._candidates[key]
            if cand.status == "rejected":
                del self._candidates[key]  # allow re-emission later

    def _try_emit(
        self,
        records: list[SurpriseRecord],
        key: ContextKey,
        current_tick: int,
    ) -> CandidateRule | None:
        deltas = {var: np.array([r.delta[var] for r in records]) for var in RESIDUAL_BODY_ORDER}
        mean = {var: float(np.mean(deltas[var])) for var in RESIDUAL_BODY_ORDER}
        mad = {var: float(np.median(np.abs(deltas[var] - mean[var]))) for var in RESIDUAL_BODY_ORDER}
        significant = {
            var: abs(mean[var]) > max(self.MAD_K * mad[var], self.SIGNIFICANCE_FLOOR)
            for var in RESIDUAL_BODY_ORDER
        }
        if not any(significant.values()):
            return None
        return CandidateRule(
            context=key,
            mean_effect={v: mean[v] for v in RESIDUAL_BODY_ORDER if significant[v]},
            n_obs=len(records),
            mad=dict(mad),
            status="verifying",
            verify_records=[],
            emitted_at_tick=current_tick,
        )

    def _add_verify(self, candidate: CandidateRule, record: SurpriseRecord) -> None:
        candidate.verify_records.append(record)

    def _resolve(self, candidate: CandidateRule, store: ConceptStore) -> None:
        ok = True
        for var, expected in candidate.mean_effect.items():
            obs_mean = float(np.mean([r.delta[var] for r in candidate.verify_records]))
            if abs(obs_mean - expected) > self.VERIFY_TOL:
                ok = False
                break
        if ok:
            candidate.status = "promoted"
            self._stats["promoted"] += 1
            store.add_learned_rule(
                LearnedRule(
                    precondition=candidate.context,
                    effect=dict(candidate.mean_effect),
                    confidence=0.5,
                    n_observations=candidate.n_obs + len(candidate.verify_records),
                )
            )
        else:
            candidate.status = "rejected"
            self._stats["rejected"] += 1

    def stats(self) -> dict[str, Any]:
        return {
            **dict(self._stats),
            "in_flight": sum(1 for c in self._candidates.values() if c.status == "verifying"),
        }
```

### `LearnedRule` and Phase 7 in `_apply_tick`

```python
@dataclass
class LearnedRule:
    precondition: ContextKey
    effect: dict[str, float]
    confidence: float
    n_observations: int
    source: str = "runtime_nursery"

    def matches(self, visible: set[str], body: dict[str, float], primitive: str) -> bool:
        """Predicate evaluation. Visible match is subset (rule's visible
        must all be present in current visible); body_quartiles and
        action must equal exactly."""
        if not self.precondition.visible.issubset(visible):
            return False
        if self.precondition.action != primitive:
            return False
        # Body quartiles: skip if rule used L1 (all zeros) — i.e. body
        # state was not load-bearing for the rule.
        if self.precondition.body_quartiles != (0, 0, 0, 0):
            body_q = ContextKey.from_state(visible, body, primitive).body_quartiles
            if body_q != self.precondition.body_quartiles:
                return False
        return True
```

In `concept_store.py:_apply_tick` after Phase 6 and BEFORE the
existing residual injection / final clamp:

```python
# === Phase 7: Learned rules (Stage 79 nursery output) ===
visible_for_phase7 = visible_concepts or set()
for lr in self.learned_rules:
    if lr.confidence < self.CONFIDENCE_THRESHOLD:
        continue
    if not lr.matches(visible_for_phase7, sim.body, primitive):
        continue
    for var, delta in lr.effect.items():
        sim.body[var] = sim.body.get(var, 0.0) + delta
        traj.events.append(SimEvent(
            step=tick, kind="body_delta", var=var,
            amount=delta, source=f"learned:{lr.source}",
        ))

# (Existing Phase 78c residual injection / clamp follows here)
```

Phase 7 is BEFORE the clamp so the same clamping logic catches both
rule and learned-rule overshoot. Phase 7 is BEFORE the residual so
that the residual (when enabled) sees post-learned-rule state and can
in principle correct any remaining gap — they compose additively
without one cannibalising the other.

### `run_mpc_episode` integration

```python
def run_mpc_episode(
    env, segmenter, store, tracker, rng,
    max_steps=500, horizon=20, perceive_fn=None, verbose=False,
    trace_path=None,
    residual_predictor=None, residual_optimizer=None, residual_train=False,
    surprise_accumulator: SurpriseAccumulator | None = None,
    rule_nursery: RuleNursery | None = None,
    nursery_tick_every: int = 1,
) -> dict:
    ...
    for step in range(max_steps):
        ...
        # (existing) plan, execute, env.step, get inv_after
        ...
        # === Stage 79: surprise → accumulator → nursery ===
        if surprise_accumulator is not None:
            predicted = {var: float(state.body.get(var, 0.0)) for var in RESIDUAL_BODY_ORDER}
            # The planner's first-tick body prediction; use the same
            # 1-tick rules-only replay as Stage 78c training (with
            # planned_step propagation) so the predicted_delta matches
            # what the planner actually committed to.
            rules_sim = state.copy()
            chosen_planned_step = best_plan.steps[0] if best_plan.steps else None
            store._apply_tick(
                rules_sim, primitive, tracker, _empty_traj(rules_sim),
                tick=0, planned_step=chosen_planned_step,
            )
            predicted_after = {var: float(rules_sim.body.get(var, 0.0)) for var in RESIDUAL_BODY_ORDER}
            actual_after = {var: float(inv_after.get(var, 0)) for var in RESIDUAL_BODY_ORDER}
            context = ContextKey.from_state(visible_concepts, state.body, primitive)
            surprise_accumulator.observe(
                context=context,
                predicted={var: predicted_after[var] - predicted[var] for var in RESIDUAL_BODY_ORDER},
                actual={var: actual_after[var] - predicted[var] for var in RESIDUAL_BODY_ORDER},
                tick_id=step,
            )
            if rule_nursery is not None and step % nursery_tick_every == 0:
                rule_nursery.tick(surprise_accumulator, store, current_tick=step)
    ...
    if rule_nursery is not None:
        result["nursery_stats"] = rule_nursery.stats()
        result["accumulator_stats"] = surprise_accumulator.stats()
```

The 1-tick rules-only replay reuses the Stage 78c bug-fixed pattern
(with `planned_step` propagation) — same code, same correctness
guarantees. This is intentional: surprise must be measured against
the planner's actual prediction, not against a degenerate fallback
prediction, otherwise the nursery learns rules to compensate for an
imaginary bug.

## Constants — first proposed values, all overridable per experiment

| Constant | Value | Rationale |
|---|---:|---|
| `MIN_OBS_L1` | 5 | Coarse bucket — needs few observations to surface obvious patterns like "skeleton visible → health drop" |
| `MIN_OBS_L2` | 10 | Refined bucket — needs more obs because there are 256× as many possible L2 buckets, so each L1 collision needs more support before warranting an L2 split |
| `MAX_BUCKET_SIZE` | 100 | Sliding window for distribution shift; old observations age out |
| `MAD_K` | 2.0 | Consistency threshold; matches sketch and is the standard "outlier rejection" cutoff |
| `SIGNIFICANCE_FLOOR` | 0.01 | Minimum mean magnitude — anything smaller is noise vs the typical body delta scale (per-tick rates are ±0.02–0.05) |
| `VERIFY_N` | 10 | Number of held-out observations needed before promotion |
| `VERIFY_TOL` | 0.02 | How close the verifying mean must be to the candidate's emitted mean |
| `nursery_tick_every` | 1 | Run the nursery on every env step initially — cheap, no batching overhead |

These are all `class` attributes on `RuleNursery` and overridable by
subclassing or by mutating the class attribute in tests. The harness
will report them in the results JSON for reproducibility.

## Persistence

Learned rules are kept in `ConceptStore.learned_rules` for the
lifetime of the store. Per Stage 78c convention each ablation has
a fresh store, so learned rules from a previous run do not leak.
Within an episode, rules persist across phases (warmup_a, warmup_b,
eval) — the agent learns continually.

For diagnostics: at episode end the harness writes
`learned_rules.jsonl` to the run output dir, one rule per line, with
`{precondition, effect, confidence, n_observations, source}`. This
gives a human-readable record of what was discovered.

## Compatibility with Stage 78c residual

Both can run simultaneously. Phase 7 (learned rules) and the residual
correction are independent additive contributions to `sim.body`. Phase
order is: rules → learned rules → residual → clamp.

Default for Stage 79 evaluation: **residual_off, nursery_on**. This
isolates the nursery's contribution. A second ablation arm can run
**residual_on, nursery_on** to see if they compose, but that is not
the headline test.

## Tests

### Unit tests (`tests/learning/test_surprise_accumulator.py`)

1. `ContextKey.from_state` produces correct quartiles for body values
   {0, 2, 5, 7, 9}
2. `ContextKey.from_state` is hashable and equality-comparable
3. `ContextKey.coarsen()` zeros quartiles, preserves visible+action
4. `SurpriseAccumulator.observe` adds to both L2 and L1 buckets
5. `SurpriseAccumulator` sliding window evicts oldest at MAX_BUCKET_SIZE
6. `SurpriseAccumulator.stats` reports correct counts

### Unit tests (`tests/learning/test_rule_nursery.py`)

1. `RuleNursery._try_emit` returns None below MIN_OBS
2. `RuleNursery._try_emit` returns None when MAD too high (inconsistent)
3. `RuleNursery._try_emit` returns None when mean below SIGNIFICANCE_FLOOR
4. `RuleNursery._try_emit` returns CandidateRule when MIN_OBS + consistent
5. `RuleNursery.tick` promotes a candidate after VERIFY_N matching records
6. `RuleNursery.tick` rejects a candidate when verify mean drifts >VERIFY_TOL
7. Promoted candidates write a `LearnedRule` to `store.learned_rules`
8. Rejected candidates are removed from in-flight tracking

### Synthetic integration test (`tests/learning/test_nursery_synthetic_conjunctive.py`)

Reproduces Stage 78a's conjunctive task at the nursery level:

1. Construct an in-memory `ConceptStore` with the textbook minus the
   conjunctive sleep correction
2. Loop 500 times: pick a random `(visible, body, action)` sample,
   compute the oracle `true_body_delta`, compute the rules-only
   prediction via `_apply_tick`, feed the surprise to accumulator
3. Run nursery tick after each observation
4. Assert: by the end, `store.learned_rules` contains a rule whose
   precondition matches `(sleep, food=0 or drink=0)` and whose
   effect approximately matches `health: -0.067`

This test mirrors the Stage 78a dataset and Stage 78b/c synthetic
gate, but uses the **nursery** instead of an MLP residual. If the
nursery cannot pass this test, Stage 79 has no chance on real
Crafter — it is the cheapest possible falsification.

### Crafter integration test (`experiments/stage79_nursery_crafter.py`)

Mirrors the Stage 78c harness structure (warmup A / warmup B / 3 ×
20 eval, identical seeds, fresh store per ablation, checkpoints per
phase). Three ablations:

1. `nursery_off` (baseline = Stage 78c residual_off)
2. `nursery_on` (residual off, nursery on)
3. `nursery_on_residual_on` (both — composition test)

Headline metrics: eval avg_len, wood avg, wood ≥3 rate, action
entropy, **n promoted rules per episode**, **n rejected per episode**,
**learned rule preconditions distribution** (which contexts the
nursery actually fired in).

## Gates

| Gate | Threshold | Why |
|---|---|---|
| Synthetic conjunctive nursery test | passes within 500 obs | proves the no-LLM induction works on the canonical conjunctive case |
| Crafter eval `nursery_on` ≥ `nursery_off` | mean Δ ≥ 0 | symbolic learning must not hurt — minimum bar |
| Crafter eval `nursery_on` ≥ Stage 77a Run 8 baseline (180) | survival ≥ 180 | meet or exceed the historical wall |
| At least 1 promoted rule per episode (mean) | nursery is actually doing work in real rollouts | guards against "nursery is silent because thresholds are too tight" |
| Action entropy `nursery_on` ≥ `nursery_off` − 0.1 | does not collapse like residual_on did | the entropy collapse was Stage 78c's killer; we should not reproduce it |
| Wood ≥ 3 rate ≥ 5/60 | very modest improvement over Stage 78c's 0/60 | demonstrates *some* gathering progress |

A "PARTIAL PASS" outcome that's still informative: nursery passes
the synthetic test and emits learned rules in Crafter, but Crafter
survival is unchanged. That tells us the nursery infrastructure works
but the rules it learns are not the right ones for Crafter survival
— at which point Stage 80 becomes "tune the context key / quartile
boundaries / verification thresholds" rather than "design a new
mechanism".

## Risks and what to watch for

1. **L2 bucket count explosion.** 9 visible concepts (frozenset, ~512
   subsets in practice ~50 in episodes) × 256 quartile combinations ×
   17 actions ≈ 200K possible L2 buckets. Each bucket holds up to 100
   records → 20M records max. In practice most buckets will be empty.
   Watch `accumulator.stats()["n_buckets"]` per episode; if it grows
   linearly with steps and never plateaus, the keying is too fine.

2. **False positives.** The MAD-based consistency check can promote
   spurious correlations (e.g. agent happens to be near water every
   time food drops, leading to a rule "near water → food drops").
   Mitigation: VERIFY_N=10 with VERIFY_TOL=0.02 should reject most
   spurious patterns; track `_stats["rejected"]` and inspect rejected
   candidates if rejections are very rare.

3. **Slow learning vs MPC's impatience.** The nursery needs MIN_OBS=5+
   per bucket to emit. In a 500-step episode the agent may not visit
   any single bucket 5 times. Mitigation: L1 buckets aggregate across
   body states, so they accumulate faster; the conjunctive case still
   needs L2 but the test will reveal if that's a real problem.

4. **Stale rules after distribution shift.** Promoted rules persist
   indefinitely. If the agent's behaviour changes (e.g. starts going
   to caves), old surface rules become irrelevant. Out of scope for
   Stage 79; flagged for Stage 80+.

5. **Phase 7 ordering bug.** If a learned rule depends on body values
   AFTER Phase 3 (background rates) but the matching uses body values
   BEFORE the tick, conditions may be evaluated against the wrong
   body. Fixed by passing `body=sim.body` (post-Phase-6, pre-Phase-7)
   to `LearnedRule.matches`.

## Open questions deferred to implementation review

- Should the nursery emit candidates for L1 buckets when L2 has
  enough data but L1 hasn't? The current design says "L2 first, fall
  back to L1" — but what if L2 finds a rule and we never check L1?
  Decision: emit for both independently; deduplication happens in
  `_apply_tick` because two matching rules will both fire and
  compose, which is fine for body deltas.
- How to surface promoted rules in the agent's "introspection" UI
  (if/when we add one)? Out of scope, but the JSONL export is a
  starting point.
- Does Phase 7 need a SimEvent kind of its own (`learned_body_delta`)
  separate from the existing `body_delta`? Probably yes for diagnostics
  — add `learned_rule:<context_summary>` as the source, parseable by
  trace analysis tools.

## Implementation order

1. `surprise_accumulator.py` + unit tests (~1 day)
2. `rule_nursery.py` + unit tests (~1 day)
3. `learned_rule.py` + Phase 7 integration in `concept_store.py`
   + unit tests (~1 day)
4. `run_mpc_episode` integration + smoke test (~0.5 day)
5. Synthetic conjunctive integration test (`tests/learning/`) (~0.5 day)
6. `experiments/stage79_nursery_crafter.py` harness (~0.5 day)
7. Smoke run on minipc, full eval, Stage 79 report (~1 day)

Estimate: ~5 working days. The largest risk is the synthetic
conjunctive test failing (which would mean the no-LLM induction
heuristic needs rethinking), so it should be the first deliverable
after the unit tests. If it fails, stop and revisit the design; if
it passes, the Crafter integration is mostly mechanical.

## Stage 79 → 80 → 81 narrative

- **Stage 79** (this doc): nursery infrastructure + Crafter eval
- **Stage 80** (if 79 passes the synthetic test but not Crafter
  gates): tune quartile boundaries, MIN_OBS, VERIFY_N empirically;
  consider adding negation rules ("when X is NOT visible") and
  conjunction grammar (AND of multiple visible predicates)
- **Stage 81** (if 79 + 80 pass): alternating training — when a
  learned rule is promoted, the residual stops training on that
  context. Per Strategy v5 plan, this is the Neuro-Symbolic Synergy
  pattern. Re-enable the Stage 78c residual gated on rule coverage.

## Files that will exist after Stage 79 lands

```
src/snks/learning/surprise_accumulator.py     (NEW)
src/snks/learning/rule_nursery.py              (NEW)
src/snks/agent/learned_rule.py                 (NEW)
src/snks/agent/concept_store.py                (MODIFY: add learned_rules + Phase 7)
src/snks/agent/mpc_agent.py                    (MODIFY: thread accumulator + nursery)
tests/learning/test_surprise_accumulator.py    (NEW)
tests/learning/test_rule_nursery.py            (NEW)
tests/learning/test_nursery_synthetic_conjunctive.py  (NEW)
experiments/stage79_nursery_crafter.py         (NEW)
docs/reports/stage-79-report.md                (NEW after run)
```

No existing tests should break. The Phase 7 addition is gated on
`store.learned_rules` being non-empty — when it's empty (everywhere
in existing tests and Stage 78c regression), the loop is a no-op
that costs ~50 ns of attribute access.

## Strategy v5 novelty position after Stage 79

| # | Novelty | Status |
|---|---|---|
| 1 | DAF residual learner | parked Branch C, possibly worth re-test |
| 2 | **No-LLM rule induction from surprise** | **active — Stage 79 implementation** |
| 3 | Three-category ideology (facts / mechanisms / experience) | retained — facts (textbook) untouched, mechanisms (Phase 7 in `_apply_tick`) is code, experience (`learned_rules`) is the runtime addition |

Stage 79 is the focal contribution of the rest of the Strategy v5
arc. If it passes the synthetic test, that alone is a publishable
result regardless of Crafter outcome — no published neurosymbolic
WM has demonstrated rule induction without an LLM in the loop. If
it also closes the Crafter wall, the path to Stage 86 (Minecraft)
opens.
