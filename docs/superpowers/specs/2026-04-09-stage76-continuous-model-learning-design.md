# Stage 76: Continuous Model Learning — Design

**Date:** 2026-04-09
**Status:** Draft v2 (post-brainstorming + spec review)
**Depends on:** Stage 75 perception (82% tile accuracy) + homeostatic bugs fixed

## Problem

Stage 75 reached an architectural ceiling at ~180 survival steps (target ≥200)
despite 11 code-level fixes. Root cause: plan execution is LINEAR and BLIND.
Agent commits to multi-step plans (collect 3 wood → place table → make sword →
kill zombie) and cannot evaluate "will I survive executing this plan?" during
execution. Zombies attack, HP drops, agent dies mid-plan.

Symbolic STRIPS-style planning is fundamentally brittle in stochastic,
adversarial environments. The missing capabilities are:
1. **Episodic memory** — "I tried this before, what happened?"
2. **Forward simulation** — "if I do X now, what state will I be in?"
3. **Learned relevance** — "what features matter for this decision?"
4. **Continuous adaptation** — learning from each interaction, no batch phases

## Ideology Constraints

Per `docs/IDEOLOGY.md` and lessons from Stage 75:

**MUST:**
- Continuous learning. Each step updates the model. No batch training.
- Emergent from experience. No hardcoded thresholds, reflexes, magic numbers.
- Top-down: goal = homeostasis, strategy derives from drives + model + experience.
- No reward shaping. No policy gradients. No RL traps.
- No derived features. All "concepts" must emerge from raw sensory data via
  statistical learning. Hand-coded `in_danger`, `critical_HP`, `can_craft`
  are forbidden (see `feedback_no_feature_engineering.md`).
- No hardcoded reflexes. "If zombie: flee" is forbidden
  (see `feedback_no_hardcoded_reflexes.md`).
- CNN = V1 (perception only). World model lives in ConceptStore + new memory
  components. CNN is a replaceable sensor module.

**CAN USE:**
- SDR / VSA representations (memory-based, no gradients)
- Sparse Distributed Memory (SDM) for episodic storage
- Hebbian-style statistical learning (correlation counters)
- ConceptStore as bootstrap instincts (early episodes before memory fills)
- Tracker-style EMA for slow continuous updates
- One-shot confidence updates for causal rules

## Architecture Overview

Four memory systems working in concert:

1. **Working memory** (existing): `VisualField` + `inventory` — what I see NOW.
2. **Semantic memory** (existing, expanded role): `ConceptStore` — innate
   instincts from textbook + slow confidence updates. Used for bootstrap only.
3. **Spatial memory** (existing): `CrafterSpatialMap` — where I saw what.
4. **Body memory** (existing): `HomeostaticTracker` — how my stats change.

Three NEW memory systems:

5. **Episodic memory**: `EpisodicSDM` — stores specific (state, action,
   next_state, body_delta) tuples. Queryable by similarity.
6. **Attention weights**: `AttentionWeights[drive, bit]` — learned correlation
   between state bits and drive-outcome changes. Acts as dynamic relevance
   filter.
7. **State encoder**: `StateEncoder` — deterministic projection of raw
   perception into sparse binary vector (SDR). No learning; just encoding.

### Key design principle — no drive argmax

First draft included a "dominant drive" argmax selector (take the most urgent
body variable and query attention weighted by that drive). This was itself a
hardcoded priority heuristic — it presupposes agent needs to "think about one
thing at a time".

**Revised**: body state is encoded directly into SDR as raw scalar fields.
Attention weights are per-body-variable (not per-drive). Action selection
uses the full SDR query (no masking) and scores candidate actions by
**expected cumulative deficit reduction across ALL body variables**, weighted
by current deficit.

There is no "which drive wins" step. The agent reacts to the whole state
simultaneously. Deficit weighting makes urgent variables naturally dominate
action scoring without an explicit priority loop.

### Data Flow

```
Raw perception (inv, vf, spatial_map, player_pos, tracker)
         │
         ▼
┌──────────────────────┐
│ StateEncoder         │
│ → state_sdr          │  (sparse binary vector, ~200 of 4000 bits active)
│ includes all body    │  (HP, food, drink, energy as bucket-encoded scalars)
│ state as bucket bits │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ EpisodicSDM.recall   │  query with full state_sdr (no masking v1)
│ → top-K past episodes│  sorted by SDR similarity (popcount overlap)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Action Aggregator    │  per action A in recalled:
│                      │    expected_delta[var] = mean(body_delta[var])
│                      │  score[A] = Σ_var deficit[var] × expected_delta[var]
│                      │  choose A via softmax (temperature > 0 for exploration)
└──────────┬───────────┘
           │
           ▼ (env.step)
┌──────────────────────┐
│ EpisodicSDM.write    │  (state, action, next_state, body_delta)
│ ConceptStore.verify  │  confidence update for causal links
│ tracker.update       │  body rate observation (unchanged)
└──────────────────────┘
```

**Note on AttentionWeights**: deferred to v2 within Stage 76. v1 ships
without attention mask, relying on raw SDR similarity. If SDM recall proves
too noisy (retrieves irrelevant episodes), v2 adds attention. This keeps
scope minimal and testable.

### Bootstrap Phase (cold start)

For the first N episodes (~50-100), EpisodicSDM is nearly empty. During this
phase:
- Action selection falls back to `ConceptStore.plan(goal, inventory)` —
  the existing Stage 75 backward chaining from textbook rules.
- EpisodicSDM still accumulates experience.
- AttentionWeights still update.
- Agent uses instincts while learning.

Transition criterion: once SDM recall returns ≥5 sufficiently-similar
episodes for the current state, switch to memory-based action selection.
Otherwise use ConceptStore plan.

This is analogous to biological development: infants rely on innate reflexes
while building episodic memory, then shift to experience-based decisions.

## Component Design

### StateEncoder

Purpose: deterministically project raw perception into a sparse binary vector
(SDR) suitable for similarity-based memory.

Shape: `np.bool_[4096]`. Target ~200 active bits (5% density).

Input:
- `inventory: dict[str, int]`
- `vf: VisualField` (from perceive_tile_field)
- `spatial_map: CrafterSpatialMap`
- `player_pos: tuple[int, int]`

Encoding strategies (per field type):

**1. Scalar body stats (HP, food, drink, energy)** — bucket encoding.
```python
# 0..9 → sliding window in bit range
HP=5: bits[500:540] = 1
HP=6: bits[504:544] = 1  # overlap with HP=5 → similar states match
```
Each stat gets 100 bits, window width 40. Similar values share ~80% bits.

**2. Inventory counts (wood, stone_item, coal_item, ...)** — bucket encoding,
smaller range (0..5+).
```python
wood=2: bits[1000:1040] = 1
wood=3: bits[1010:1050] = 1
```

**3. Inventory presence (wood_sword, wood_pickaxe, table)** — fixed random
SDR per item, 40 bits each, drawn from a known seed. Present → bits on.

**4. Visible concepts with distance** — spatial bit allocation (not VSA XOR).

Rejected approach: `vsa_bind(concept, dist)` via XOR. XOR destroys similarity
— `bind(zombie, dist=2)` and `bind(zombie, dist=3)` share ~0% bits, breaking
the "similar values → similar patterns" principle for distances.

Used approach: pre-allocate a dedicated bit range per concept, bucket-encode
distance within that range.
```python
# bit ranges per concept, pre-allocated
SEE_ZOMBIE_RANGE = (2000, 2100)   # 100 bits
SEE_TREE_RANGE = (2100, 2200)
SEE_WATER_RANGE = (2200, 2300)
# ... one range per known class

def encode_visible(bits, concept, distance):
    start, end = CONCEPT_RANGES[concept]
    # Bucket encode distance 0..9 in 100-bit range, window=40
    bucket = bucket_encode_range(distance, 9, start, end, window=40)
    bits[bucket] = True
```

Property: `see_zombie@dist=2` and `see_zombie@dist=3` share ~80% bits within
the SEE_ZOMBIE_RANGE. `see_tree@dist=2` shares 0% bits with `see_zombie@dist=2`
because they occupy different ranges. Similarity-preserving by construction.

Multiple instances: use max-distance or min-distance per concept (closest
instance typically most relevant). v1 uses min-distance only.

**5. Spatial map distances** — same spatial allocation strategy, different
ranges and wider distance scale.
```python
KNOW_TREE_RANGE = (3000, 3100)
KNOW_WATER_RANGE = (3100, 3200)
# ...

for concept in KNOWN_CLASSES:
    nearest = spatial_map.find_nearest(concept, player_pos)
    if nearest:
        dist = min(manhattan(nearest, player_pos), 30)
        bucket = bucket_encode_range(dist, 30, *KNOW_RANGES[concept], window=30)
        bits[bucket] = True
```

**VSA note**: we are not using circular convolution or XOR binding in v1.
The spatial allocation scheme above gives us similarity-preserving structured
encoding with simpler implementation and better interpretability. If future
stages need role-filler binding (e.g., "agent at position X, zombie at Y,
relative angle Z"), proper VSA can be added then.

No derived features. No booleans like `can_craft` or `in_danger`. Raw sensor
values only, encoded so that similar values produce similar patterns.

Fixed SDR patterns (for concept bindings and presence) generated from a
deterministic random seed and stored once. Same seed across all runs for
consistency.

### Primitives

Only two operations on SDRs:

**Bundle (OR)**: `C = A | B`
- Element-wise OR of binary SDRs. Accumulates features.

**Similarity**: `sim(A, B) = popcount(A & B)`
- Bit overlap count. Linear in sparse density.

No bind/unbind operations needed in v1 thanks to spatial allocation scheme
above. Each concept×value combination maps directly to a pre-allocated bit
region.

### EpisodicSDM

Purpose: store experience tuples and retrieve similar past episodes.

Storage schema:
```python
@dataclass
class Episode:
    state_sdr: np.ndarray      # shape (4096,) bool
    action: str                # from ACTION_NAMES
    next_state_sdr: np.ndarray
    body_delta: dict[str, int] # {"health": -2, "food": -1, ...}
    step: int                  # for replay ordering / decay
```

**Write**: append to a circular buffer (size ~10,000 recent episodes).
Old episodes evicted FIFO. No indexing — just a list.

**Read (recall)**:
```python
def recall(query_sdr, top_k=20) -> list[Episode]:
    scores = [(popcount(query_sdr & ep.state_sdr), ep) for ep in buffer]
    scores.sort(reverse=True)
    return [ep for _, ep in scores[:top_k]]
```

Simple linear scan over buffer. At 10K episodes × 4096 bits, each scan is
~40M bitwise ops, ~10ms in numpy. Acceptable.

Optimization (future): hash buckets, LSH, or proper SDM hard locations. Start
with brute force.

**Per-action aggregation**:
```python
def score_actions(
    recalled: list[Episode],
    current_body: dict[str, int],
    tracker: HomeostaticTracker,
) -> dict[str, float]:
    """
    Score each action by expected improvement of body state.

    Deficit-weighted aggregation:
    - For each body variable V (from tracker.rates keys, not hardcoded):
      deficit[V] = max(0, tracker.observed_max[V] - current_body[V])
    - For each recalled episode with action A:
      score[A] += Σ_V deficit[V] × body_delta_V_in_episode
    - Average over episodes of same action

    Sign is emergent: if body variable 'health' is observed to go up in
    good episodes and down in bad ones, deficit × delta naturally scores
    "restoring" actions higher. No hardcoded "higher is better" assumption.
    Works for any variable tracker observes (food, drink, energy, HP,
    and any future variables).
    """
    scores = defaultdict(float)
    counts = defaultdict(int)
    for ep in recalled:
        for var, delta in ep.body_delta.items():
            if var not in tracker.observed_max:
                continue
            current = current_body.get(var, tracker.observed_max[var])
            deficit = max(0, tracker.observed_max[var] - current)
            scores[ep.action] += deficit * delta
        counts[ep.action] += 1
    return {a: scores[a] / counts[a] for a in scores if counts[a] > 0}


def select_action(action_scores: dict[str, float], temperature: float = 1.0) -> str:
    """
    Softmax-based action selection for exploration.
    Higher temperature → more exploration, lower → more exploitation.
    """
    if not action_scores:
        return None
    actions = list(action_scores.keys())
    values = np.array([action_scores[a] for a in actions])
    # Subtract max for numerical stability
    exp_values = np.exp((values - values.max()) / temperature)
    probs = exp_values / exp_values.sum()
    return np.random.choice(actions, p=probs)
```

Notes:
- `tracker.observed_max[var]` is a new tracker feature: rolling max observed
  across episodes. Replaces hardcoded `9`. Initialized from first observation.
- `body_delta` set is whatever tracker has seen, not a hardcoded list.
  Adding new body variables (e.g., mana, fatigue) requires only tracker
  updates, not this function.
- Softmax with temperature > 0 gives stochastic action choice — prevents
  collapse onto early strategies (exploration mechanism). Temperature
  schedule: start high (1.0), decay over episodes.

### AttentionWeights — deferred to v2 within Stage 76

The initial brainstorming proposed per-drive attention weights `[drive × bit]`.
Spec review identified load-bearing issues: (a) "drive" selection is itself
a hardcoded argmax, (b) unbounded Hebbian updates drift, (c) zero-init plus
positive threshold breaks bootstrap.

**v1 decision**: ship Stage 76 without AttentionWeights. Query SDM with the
full state_sdr. Rely on bucket-encoding and spatial allocation to produce
similar patterns for similar states naturally.

**v2 trigger**: if SDM recall is consistently noisy (top-K matches include
many irrelevant episodes → poor action scoring → survival doesn't improve),
add attention as a per-variable relevance weight:

```python
# v2 (not in scope for initial ship)
weights: np.float32[n_body_vars, n_bits]  # per-variable, not per-drive
# EMA update with clipping, normalized initialization
# Soft weighting (no hard threshold), used as query mask
```

Rationale: avoid scope creep. v1 is simpler and testable. v2 justified only
if v1 evidence shows noise-in-recall is the bottleneck. Matches the
"systematic debugging" principle: no speculative complexity before measuring.

### Interaction with Existing Components

**HomeostaticTracker**: unchanged. Provides dominant drive selection for
attention query.

**ConceptStore**: unchanged internals. Used during bootstrap phase via
`select_goal` + `plan()` when SDM is empty. After bootstrap, confidence still
updates via `verify()`, but plans are not executed as primary strategy.

**CrafterSpatialMap**: unchanged. Used in StateEncoder to provide "what do I
remember about world positions".

**Stage 75 phase6_survival loop**: replaced. New loop in `phase6_continuous`
uses the components above.

## Components Summary

### v1 (this Stage 76 ship)

New files:
- `src/snks/memory/__init__.py` — new package
- `src/snks/memory/sdr_encoder.py` — StateEncoder class + bucket encoding
- `src/snks/memory/episodic_sdm.py` — EpisodicSDM class with FIFO buffer,
  recall, best_action scoring, softmax selection
- `src/snks/agent/continuous_agent.py` — decision loop using SDR + SDM +
  ConceptStore bootstrap

Modified:
- `src/snks/agent/perception.py` — add `HomeostaticTracker.observed_max`,
  `observed_variables()` methods (drop hardcoded 4-drive list)
- `experiments/exp136_continuous_learning.py` — new experiment pipeline
  (warmup + evaluation phases)
- `tests/test_stage76_sdr.py` — StateEncoder tests
- `tests/test_stage76_sdm.py` — EpisodicSDM tests
- `tests/test_stage76_agent.py` — end-to-end decision loop smoke test
- `tests/test_stage76_no_hardcode.py` — automated check for forbidden
  patterns (Gate 5)

### v2 (deferred, only if v1 insufficient)

- `src/snks/memory/attention.py` — AttentionWeights class with per-variable
  relevance, EMA updates, clipping. Only added if v1 survival < 200 AND
  diagnostic shows SDM recall noise is the cause.

### Unchanged

- Perception pipeline (reuses `exp135/segmenter_9x9.pt`)
- `snks/agent/concept_store.py` (except confidence update which already exists)
- `snks/agent/crafter_spatial_map.py`
- `snks/agent/crafter_pixel_env.py`
- `snks/agent/crafter_textbook.py`

## Decision Loop Pseudocode

```python
def continuous_decision_loop(env, segmenter, encoder, sdm, store, tracker,
                              rng, max_steps, temperature=1.0, bootstrap_k=5):
    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    prev_inv = None

    for step in range(max_steps):
        inv = dict(info["inventory"])
        player_pos = info["player_pos"]

        # Perception
        vf = perceive_tile_field(torch.from_numpy(pixels), segmenter)
        spatial_map.update(player_pos, vf.near_concept)
        for cid, conf, gy, gx in vf.detections:
            wx = player_pos[0] + (gx - center_c)
            wy = player_pos[1] + (gy - (center_r - 1))
            spatial_map.update((wx, wy), cid)

        # Track body changes (observes rates, updates observed_max)
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Encode raw state → SDR (includes body stats as bucket-encoded)
        state_sdr = encoder.encode(inv, vf, spatial_map, player_pos)

        # Query episodic memory with full SDR
        recalled = sdm.recall(state_sdr, top_k=20)

        if len(recalled) >= bootstrap_k:
            # Memory-based decision
            action_scores = score_actions(recalled, inv, tracker)
            action = select_action(action_scores, temperature=temperature)
        else:
            # Bootstrap: ConceptStore plan from textbook instincts
            goal, plan = select_goal(inv, store, tracker, vf, spatial_map)
            if plan:
                action = plan[0].action
                # For make/place, compose env action name
                if plan[0].action in ("make", "place"):
                    action = f"{plan[0].action}_{plan[0].expected_gain}"
            else:
                # No plan — random move for data gathering
                action = str(rng.choice(MOVE_ACTIONS))

        # Execute
        inv_before = inv
        pixels, _, done, info = env.step(action)
        next_inv = dict(info["inventory"])
        next_vf = perceive_tile_field(torch.from_numpy(pixels), segmenter)
        next_state_sdr = encoder.encode(next_inv, next_vf, spatial_map,
                                         info["player_pos"])

        # Body delta — variables come from tracker, not hardcoded
        body_delta = {
            var: next_inv.get(var, inv.get(var, 0)) - inv.get(var, 0)
            for var in tracker.observed_variables()
        }

        # Write experience (continuous, every step)
        sdm.write(Episode(
            state_sdr=state_sdr,
            action=action,
            next_state_sdr=next_state_sdr,
            body_delta=body_delta,
            step=step,
        ))

        # ConceptStore confidence update (existing path, for bootstrap quality)
        outcome = outcome_to_verify(action, inv_before, next_inv)
        if outcome:
            verify_outcome(vf.near_concept, action, outcome, store)

        prev_inv = inv
        if done:
            break

    return {
        "length": step + 1,
        "sdm_size": len(sdm.buffer),
        "final_inv": next_inv,
    }
```

**Key differences from Stage 75 phase6**:
1. No `select_goal` in the main path — only as bootstrap fallback
2. No plan execution state machine (no plan_step_idx, no nav_steps timer)
3. No hardcoded "drive" argmax
4. Full state encoded every step, memory queried every step
5. Softmax action selection (exploration mechanism)
6. Single source of truth for action choice: SDM recall + deficit-weighted
   aggregation + softmax. No if-else chains.

## Why 1-step reactive with memory should beat Stage 75 planning

The Stage 75 survival gap wasn't "can't plan far enough" — the agent had a
4-step kill_zombie plan. The gap was "plan commits too early, can't adapt
when conditions change". Specifically:

- Stage 75 agent sees zombie → plans kill_zombie → navigates to tree →
  during navigation zombie attacks → HP drops → plan doesn't re-evaluate →
  dies mid-plan.

Stage 76 addresses this WITHOUT forward simulation by making every step a
fresh decision based on similar past experiences:

- Stage 76 agent sees zombie → queries SDM with "zombie visible + HP=X +
  sword=0 + ..." → recalls past episodes where similar state led to death →
  scoring pushes away from that action class → picks different action (flee,
  or if past episodes with attack succeeded, attack).

The key claim: **memory of past outcomes substitutes for explicit forward
simulation**, because recalled episodes ARE past forward rollouts. Each
episode is a 1-step look at "what happens when I do X in state S".

What this approach CANNOT do:
- Plan novel multi-step sequences the agent has never executed
- Reason about "if I collect 2 wood now, I'll have enough for table later"
- Discover new strategies without first trying them randomly

These are Stage 77+ concerns (compositional forward simulation). Stage 76
tests whether memory-based reaction alone beats Stage 75's 178 steps.

Hypothesis: YES, because Stage 75's failure mode is architectural commitment,
not insufficient planning depth. A reactive agent that remembers past deaths
can avoid repeating them. A planning agent that commits to a 4-step plan
executes it until death regardless.

If this hypothesis is wrong (survival still <200), Stage 76 needs multi-step
forward simulation. That's a scope expansion, not a redesign.

## Success Criteria

### Must pass (gates)

1. **Survival mean ≥ 200 steps** over 3 independent runs × 20 episodes each
   (60 total), with enemies enabled, max_steps=1000. Each individual run
   must also show mean ≥ 200 (no lucky-run cheats).
2. **Perception maintained**: tile_acc ≥ 80% on fresh eval (no regression
   from Stage 75).
3. **Wood collection maintained**: ≥ 50% of smoke episodes (no enemies,
   max_steps=200) reach 3 wood.
4. **Memory growth monotonic**: SDM size grows each episode until buffer
   wraps. After wrap, buffer stays at size=max_buffer (no empty state).
5. **Zero hardcoded derived features**: automated check — a linter-style
   test that scans `sdr_encoder.py` for forbidden patterns:
   - No `if` statements computing booleans from inventory/vf
   - No magic number thresholds (e.g., `HP < 3`)
   - No hardcoded list of "drive variables" in encoder
   Only allowed: bucket_encode, fixed SDR lookup, spatial range assignment.
6. **No `most_urgent_drive` or similar priority argmax** in decision loop
   (code review + grep).
7. **Tests pass**: unit tests for StateEncoder (deterministic, similarity
   property), EpisodicSDM (write, recall, wrap), action scoring
   (deficit-weighted, sign-emergent), softmax selection.

### Should demonstrate

1. **Bootstrap transition**: early episodes use ConceptStore, later episodes
   use SDM. Log proportion per episode.
2. **Drive-dependent attention**: weight patterns differ across drives.
   Print top-10 bits per drive after training.
3. **Successful generalization**: agent handles situations it hasn't exactly
   seen before (similar but not identical states recall helpful memories).
4. **Improvement over training**: survival trend should rise across
   episodes as memory accumulates.

### Nice to have

1. **Cross-world portability**: run same architecture on a different
   environment (future stage) without code changes beyond perception.
2. **SDM compression**: older episodes consolidate into ConceptStore rules
   automatically (future stage).

## Risks

1. **Slow warm-up**. First 200-500 steps will be nearly random. If episodes
   are too short, agent dies before accumulating useful memory. Mitigation:
   seed SDM with a warmup phase in enemy-free env (50 episodes) before
   enabling enemies for evaluation.

2. **SDM scan cost**. Linear scan over 10K episodes per step = ~40M bitwise
   ops per query. At 10ms per query, 500-step episodes cost 5s perception-free.
   Acceptable for evaluation but may slow experimentation. Mitigation:
   start with smaller buffer (2K), add LSH or hash buckets if bottleneck.

3. **Catastrophic forgetting via FIFO buffer**. Once buffer fills (~10K steps
   = ~50 episodes), early exploration-phase experiences get evicted. Those
   experiences might be important (they cover state space more broadly than
   later experiences from a converged policy). Mitigation: implement
   reservoir sampling or priority eviction (keep episodes with high
   prediction error). v1 uses simple FIFO; monitor forgetting empirically.

4. **Exploration collapse**. Without sufficient temperature, softmax collapses
   onto "whatever worked first" — the first successful strategy dominates
   and the agent never discovers better ones. Mitigation: (a) initial high
   temperature (1.0+), (b) temperature decay schedule, (c) monitor action
   distribution entropy across episodes; if entropy drops below threshold,
   inject temperature. Alternative: Thompson sampling over action scores.

5. **Bootstrap transition threshold hand-tuned**. "≥5 similar episodes" is
   an arbitrary number, and "similar" needs a definition. Mitigation:
   define similarity concretely (popcount overlap ≥ 50% of query popcount),
   tune bootstrap_k by warmup experiments. Log bootstrap/SDM split per
   episode for diagnostic visibility.

6. **Hidden hardcoding creep**. Easy to smuggle in "critical threshold" in
   action scoring, or "danger bit" in encoder. Mitigation: (a) automated
   linter in gate 5, (b) code review must grep for common violations
   (`if inv["health"] < N`, `MAX_X = 9`, hardcoded variable lists).

7. **Deficit scoring may favor risky actions**. High deficit × positive delta
   scores well; but a single positive delta outlier may dominate the mean.
   E.g., one episode where `do(zombie)` accidentally killed a zombie → large
   positive delta → agent tries `do(zombie)` again despite many deaths.
   Mitigation: use median instead of mean, or penalize high-variance actions.

8. **Memory-reaction vs. multi-step tradeoffs**. 1-step reactive can't
   discover novel multi-step strategies. If surviving 200 requires crafting
   a sword (a 4-step sequence), agent must execute those steps by accident
   first to have memory of them. Mitigation: ConceptStore bootstrap provides
   multi-step plans as initial seed; SDM learns from their executions even
   if imperfect. If insufficient, Stage 77 adds forward simulation.

## Non-Goals

- Beating Crafter benchmark (not a goal; survival ≥200 is enough).
- DreamerV3 parity (we deliberately avoid batch training).
- Multi-environment (Stage 77+).
- True MCTS rollouts (horizon limited to 1 step initially; multi-step later).
- Neural network components (no backprop in main loop).

## Open Questions (to resolve in plan phase)

1. **SDR bit layout**: exact allocation of bit ranges per field. Needs
   experimentation. Suggested: 100 bits per body stat (window 40), 100 bits
   per inventory item, 200 bits per VSA-bound concept, remainder for padding.
2. **Bootstrap → SDM transition threshold**: 5 similar episodes is a starting
   value. May need tuning.
3. **Attention learning rate**: 0.01 initial. May need schedule (faster early,
   slower later).
4. **SDM buffer size**: 10K is a guess. Smaller = less memory, faster scan.
   Larger = more coverage. Tune empirically.
5. **Forward simulation depth**: starts at 1 (reactive). Stage 76 doesn't yet
   implement multi-step forward simulation — that's an extension once
   memory-based 1-step decisions work reliably.

## Relationship to Prior Stages

- **Stage 72 (Perception Pivot)**: established ConceptStore + homeostatic
  drives. Stage 76 builds on this, adds episodic + attention layers.
- **Stage 73 (Autonomous Craft)**: introduced drives-from-world-model
  principle. Stage 76 extends with learned relevance (attention).
- **Stage 74 (Homeostatic Agent)**: declared "no hardcoded reflexes". Stage 76
  enforces same principle for feature engineering (no hardcoded derived features).
- **Stage 75 (Per-Tile Perception)**: delivered 82% tile accuracy, which makes
  rich SDR encoding possible. Stage 76 reuses the segmenter checkpoint.
