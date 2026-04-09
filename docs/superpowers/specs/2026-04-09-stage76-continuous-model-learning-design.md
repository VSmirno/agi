# Stage 76: Continuous Model Learning — Design

**Date:** 2026-04-09
**Status:** Draft (post-brainstorming, ready for plan)
**Supersedes:** `2026-04-09-stage76-continuous-model-learning-design.md` (initial draft)
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

### Data Flow

```
Raw perception (inv, vf, spatial_map, player_pos, tracker)
         │
         ▼
┌──────────────────────┐
│ StateEncoder         │
│ → state_sdr          │  (sparse binary vector, ~200 of 4000 bits active)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dominant drive D     │  (most urgent body variable, from tracker)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ AttentionWeights     │  attended_sdr = state_sdr ∧ weights[D]
│ → attended_sdr       │  (gate out bits irrelevant to current drive)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ EpisodicSDM.recall   │  → list of (past_state, past_action, past_outcome)
│ → past experiences   │     sorted by SDR similarity
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Action aggregation   │  per action: expected body_delta for drive D
│ → best action        │  pick action maximizing D reduction
└──────────┬───────────┘
           │
           ▼ (env.step)
┌──────────────────────┐
│ EpisodicSDM.write    │  (state, action, next_state, body_delta)
│ AttentionWeights     │  Hebbian update for drive × active bits
│ .update              │
│ ConceptStore.verify  │  confidence updates for causal links
└──────────────────────┘
```

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

**4. Visible concepts with distance** — VSA binding.
```python
# zombie detected at tile (gy, gx), distance to player tile center
dist_sdr = bucket_encode("dist", manhattan_to_center, range=9)
bits |= vsa_bind(FIXED_SDR["see_zombie"], dist_sdr)
```
Each detected concept in vf produces one binding term. Multiple instances
bundle via OR.

**5. Spatial map distances** — VSA binding with larger range.
```python
# nearest known tree from spatial_map
for concept in KNOWN_CLASSES:
    nearest = spatial_map.find_nearest(concept, player_pos)
    if nearest:
        dist = manhattan(nearest, player_pos)
        bits |= vsa_bind(FIXED_SDR["know_" + concept], bucket_encode("world_dist", min(dist, 30), range=30))
```

No derived features. No booleans like `can_craft` or `in_danger`. Raw sensor
values only, encoded so that similar values produce similar patterns.

Fixed SDR patterns (for concept bindings and presence) generated from a
deterministic random seed and stored once. Same seed across all runs for
consistency.

### VSA Primitives

Minimal subset needed:

**Bind (circular convolution or XOR-based)**: `C = bind(A, B)`
- Implementation: element-wise XOR is simplest for binary SDRs (and its own
  inverse). More sophisticated: circular convolution, but requires denser vectors.
- Start with XOR: `C = A XOR B`, unbind: `A = C XOR B`.

**Bundle (OR)**: `C = A + B`
- Simple element-wise OR of binary SDRs.
- Accumulates bits; capacity limited by density. Fine at 5% density.

**Similarity**: `sim(A, B) = popcount(A AND B)`
- Bit overlap count. Simple and fast.

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
def best_action(recalled: list[Episode], drive: str) -> str:
    scores = {}  # action → sum of drive_delta
    counts = {}
    for ep in recalled:
        scores[ep.action] = scores.get(ep.action, 0) + ep.body_delta.get(drive, 0)
        counts[ep.action] = counts.get(ep.action, 0) + 1
    # For drive = "health", positive delta is good (health increased)
    # For drive = "food" (depleting), positive delta is good
    avg_scores = {a: scores[a] / counts[a] for a in scores}
    return max(avg_scores, key=avg_scores.get)
```

### AttentionWeights

Purpose: learn relevance of each SDR bit to each body drive via outcome
correlation.

Shape: `np.float32[n_drives, n_bits]` = 4 × 4096 = 64KB. Dense.

Initial values: small positive (e.g., 0.01) or uniform 0.

**Update rule (Hebbian / outcome correlation)**:
```python
def update(self, state_sdr, body_delta_per_drive, lr=0.01):
    # For each drive, positive delta = good outcome, negative = bad
    for drive, delta in body_delta_per_drive.items():
        if drive not in self.drive_idx:
            continue
        d_idx = self.drive_idx[drive]
        # Amplify weights of active bits proportional to delta
        # Positive delta → reinforce; negative → weaken
        self.weights[d_idx] += lr * delta * state_sdr.astype(np.float32)
```

Weights grow for bits that co-occur with positive outcomes, shrink for bits
that co-occur with negative outcomes.

**Mask query**:
```python
def attend(self, state_sdr, drive):
    d_idx = self.drive_idx[drive]
    # Keep bits where weight is positive (relevant); drop where weight is
    # near zero or negative (irrelevant or anti-correlated)
    threshold = 0.0
    mask = self.weights[d_idx] > threshold
    return state_sdr & mask
```

Simple threshold mask. Refinements possible later (soft weighting).

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

New files:
- `src/snks/memory/sdr_encoder.py` — StateEncoder class, bucket encoding utilities
- `src/snks/memory/vsa.py` — VSA primitives (bind, bundle, similarity)
- `src/snks/memory/episodic_sdm.py` — EpisodicSDM class
- `src/snks/memory/attention.py` — AttentionWeights class
- `src/snks/agent/continuous_agent.py` — decision loop using all memory systems

Modified:
- `experiments/exp136_continuous_learning.py` — new experiment pipeline
- `tests/test_stage76_*.py` — unit tests per component

Unchanged:
- Perception (reuses exp135 segmenter checkpoint)
- ConceptStore, HomeostaticTracker, CrafterSpatialMap
- CrafterPixelEnv

## Decision Loop Pseudocode

```python
def continuous_decision_loop(env, encoder, sdm, attention, store, tracker, max_steps):
    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    episode_history = []

    for step in range(max_steps):
        inv = info["inventory"]
        player_pos = info["player_pos"]

        # Perception
        vf = perceive_tile_field(torch.from_numpy(pixels), segmenter)
        spatial_map.update(player_pos, vf.near_concept)
        for cid, conf, gy, gx in vf.detections:
            spatial_map.update(world_pos_of(cid, player_pos, gy, gx), cid)

        # Track body changes
        if episode_history:
            prev_inv = episode_history[-1]['inv']
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Encode raw state
        state_sdr = encoder.encode(inv, vf, spatial_map, player_pos)

        # Decide action
        drive = tracker.most_urgent_drive(inv)
        attended = attention.attend(state_sdr, drive)
        recalled = sdm.recall(attended, top_k=20)

        if len(recalled) >= 5:
            # Memory-based decision
            action = sdm.best_action(recalled, drive)
        else:
            # Bootstrap: ConceptStore plan
            goal, plan = select_goal(inv, store, tracker, vf, spatial_map)
            action = plan[0].action if plan else explore_action(rng, store, inv)

        # Execute
        pixels, _, done, info = env.step(action)
        next_inv = info["inventory"]
        next_vf = perceive_tile_field(torch.from_numpy(pixels), segmenter)
        next_state_sdr = encoder.encode(next_inv, next_vf, spatial_map, info["player_pos"])

        # Compute body delta per drive
        body_delta = {
            var: next_inv.get(var, 9) - inv.get(var, 9)
            for var in ("health", "food", "drink", "energy")
        }

        # Write experience
        sdm.write(Episode(
            state_sdr=state_sdr,
            action=action,
            next_state_sdr=next_state_sdr,
            body_delta=body_delta,
            step=step,
        ))

        # Update attention from this step's outcome
        attention.update(state_sdr, body_delta, lr=0.01)

        # ConceptStore confidence update (existing path)
        outcome = outcome_to_verify(action, inv, next_inv)
        if outcome:
            verify_outcome(vf.near_concept, action, outcome, store)

        episode_history.append({'inv': inv, 'action': action, 'state': state_sdr})

        if done:
            break

    return {
        "length": step + 1,
        "sdm_size": len(sdm.buffer),
        "attention_norms": attention.diagnostic_norms(),
    }
```

## Success Criteria

### Must pass (gates)

1. **Survival ≥ 200 steps avg** with enemies (20 episodes × 1000 max_steps).
   This is the Stage 75 unmet gate.
2. **Perception maintained**: tile_acc ≥ 80% (no regression from Stage 75).
3. **Wood collection maintained**: ≥ 50% of episodes reach 3 wood.
4. **Memory growth**: SDM size grows monotonically during training
   (sanity check that writes work).
5. **Attention convergence**: weights stabilize (change < 5% over 50 episodes
   near end of training).
6. **Zero hardcoded derived features**: code review confirms StateEncoder
   contains only raw sensor fields and VSA bindings.
7. **Tests pass**: unit tests for VSA operations, StateEncoder, SDM recall,
   AttentionWeights updates.

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
   start with easier environment (no enemies) to seed SDM, then enable enemies.
2. **SDM scan cost**. Linear scan over 10K episodes per step could be slow.
   Mitigation: start with smaller buffer (~2K), optimize with hash buckets if
   needed.
3. **Attention weight drift**. Non-stationary behavior (agent improves) causes
   weight distribution shift. Mitigation: EMA with tunable decay.
4. **Bit collision in VSA**. XOR bind has limited capacity. Multiple bindings
   start interfering. Mitigation: larger SDR size, or switch to circular
   convolution if XOR insufficient.
5. **No clear transition criterion**. "When to trust SDM vs ConceptStore" is
   hand-crafted threshold (≥5 similar episodes). Could fail to switch or
   switch prematurely. Mitigation: tune by experiment; consider confidence
   score instead.
6. **Hidden hardcoding creep**. Easy to smuggle in "critical threshold" in
   attention update, or "danger bit" in encoder. Vigilance required in code
   review.

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
