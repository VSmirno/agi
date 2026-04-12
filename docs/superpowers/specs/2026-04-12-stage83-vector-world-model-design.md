# Stage 83 — VectorWorldModel: Embedding-Based World Model

**Date:** 2026-04-12
**Status:** Design approved, pending implementation (post-review v2)
**Parent:** IDEOLOGY v2 (Principle 5: Knowledge Flow), Stage 82 (nursery inversion)
**Approach:** B (clean break — replace ConceptStore AND forward_sim_types)

## Motivation

The symbolic ConceptStore (flat rule list, YAML textbook, string-keyed lookups)
has three unsolved problems:

1. **No generalization.** New concept = new rules from scratch. `do oak` requires
   a separate rule from `do tree`, even though both yield wood.
2. **No surprise → causal rule induction.** Agent cannot discover "skeleton at
   range 5 → health damage" from observation. Only textbook-declared facts fire.
3. **No knowledge flow.** Experience dies with the process. Principle 5 of
   IDEOLOGY v2 is unimplemented.

All three stem from the same root: knowledge is stored as discrete symbolic
entries, not as geometry in a shared vector space.

## Core Idea

Replace symbolic rules with **vector associations** in a single binary
hyperdimensional space. Concepts, actions, and effects are all BitVectors.
Causal knowledge = superposition of `bind(bind(concept, action), effect)` in
one associative memory vector. Prediction = unbind. Learning = XOR into memory.
Generalization = similar vectors → similar predictions, for free.

## Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector type | Binary XOR | Minimal GPU cost, popcount similarity |
| Dimension | 65536 (configurable) | ~256 binding capacity, 8KB/vector, fits 14GB RAM |
| Effect encoding | Single vector per effect | bind(role, scalar) bundled. Scales to new effect types without structural changes |
| MPC loop | Keep as-is | Domain-agnostic mechanism (category 2). Only simulate internals change |
| Textbook | YAML seed → vector associations | Principle: teacher provides rough priors, experience refines |
| ConceptStore | Replaced, not deleted | Stays in codebase until VectorWorldModel passes eval gate |

## Architecture

### 1. VectorWorldModel (core)

```python
class VectorWorldModel:
    dim: int = 65536

    # Embeddings — evolve through experience
    concepts: dict[str, np.ndarray]    # concept_id → binary vector
    actions: dict[str, np.ndarray]     # action_id → binary vector
    roles: dict[str, np.ndarray]       # variable name → role vector for encode/decode

    # Associative memory — causal knowledge via SDM
    memory: SDMMemory                  # Sparse Distributed Memory (write/read, not XOR toggle)
                                       # Stores bind(concept, action) → effect associations
                                       # Write accumulates, does not erase on repeat

    # --- Core API ---
    def predict(concept_id: str, action: str) -> np.ndarray:
        """Predict effect of action on concept.
        address = bind(v_concept, v_action)
        return memory.read(address) → effect vector"""

    def learn(concept_id: str, action: str, observed_effect: dict) -> float:
        """Learn from observation. Returns surprise (0..1).
        1. Encode observed_effect as effect vector
        2. address = bind(v_concept, v_action)
        3. memory.write(address, effect_vector)  # SDM accumulates, no erase
        4. Update v_concept embedding via weighted bundle with context
        5. Return 1.0 - hamming_sim(predicted, observed)"""

    def query_similar(vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Nearest neighbor search over concept embeddings."""

    def encode_effect(deltas: dict) -> np.ndarray:
        """Encode {wood: +1, health: -3} as single binary vector.
        bundle([bind(v_wood, encode_scalar(+1)),
                bind(v_health, encode_scalar(-3))])"""

    def decode_effect(effect_vector: np.ndarray) -> dict:
        """Unbind each known role, decode scalar, return dict."""

    def encode_scalar(value: int, max_val: int = 10) -> np.ndarray:
        """Thermometer encoding for small integers (0..max_val).
        Value K → first K * (dim // max_val) bits set to 1, rest 0.
        Invertible via popcount. Crafter values are 0-9, fits exactly."""

    # --- Bootstrap ---
    def load_from_textbook(yaml_path: str):
        """Parse YAML, create seed associations in memory."""

    # --- Persistence (knowledge flow) ---
    def save(path: str):
        """Save concepts + actions + roles + memory as binary."""

    def load(path: str):
        """Load and merge: memory = XOR(current, loaded).
        Concept embeddings merged via bundle."""
```

### 2. VectorState (replaces SimState)

```python
@dataclass
class VectorState:
    # Structured source of truth — simulation operates on this
    inventory: dict[str, int]
    body: dict[str, float]
    player_pos: tuple[int, int]       # proprioception
    step: int
    last_action: str | None
    spatial_map: CrafterSpatialMap | None

    def to_vector(model: VectorWorldModel) -> np.ndarray:
        """Encode current state as binary vector for similarity queries.
        bundle([bind(v_wood, encode_scalar(inv[wood])), ...])
        Used for SDM address computation, not for simulation."""

    def apply_effect(decoded_effect: dict) -> "VectorState":
        """Apply decoded effect dict to inventory/body.
        Operates on structured data, not vectors."""

    def is_dead() -> bool:
        """Check body vitals <= 0. Direct dict access."""

    def copy() -> "VectorState": ...
```

**Design note (from spec review B3):** Bundle is lossy — cannot selectively
unbind one component. State is stored as structured dict (inventory + body),
with `to_vector()` for encoding when needed (SDM addressing, similarity
queries). Simulation applies decoded effects to the dict directly. This
is honest: vectors are for **knowledge** (predict, learn, generalize),
structured data is for **accounting** (track exact inventory counts).

### 3. simulate_forward (in vector space)

```python
def simulate_forward(model, plan, state, horizon):
    trajectory = []
    for step in plan.steps[:horizon]:
        predicted_effect = model.predict(step.target, step.action)
        decoded_effect = model.decode_effect(predicted_effect)
        state = state.apply_effect(decoded_effect)
        trajectory.append(state)
        if state.is_dead():
            break
    return VectorTrajectory(states=trajectory, plan=plan)
```

Predict returns vector, decode converts to dict, apply works on structured
state. Vectors are the knowledge layer; dicts are the accounting layer.

### 4. Surprise-Driven Rule Induction

Every env.step:

```
1. predicted = model.predict(near_concept, action)
2. execute action in env
3. observed = model.encode_effect(actual_deltas)
4. surprise = 1.0 - hamming_sim(predicted, observed)
5. if surprise > high_threshold:
     model.learn(near_concept, action, actual_deltas)
     # Concept embedding updates with higher weight
6. if surprise < low_threshold:
     # Correct prediction — reinforce with lower weight
```

**Entity-correlated surprise:**
When unexpected health loss occurs and entities are visible, create/strengthen
association between visible entity and the damage effect:
```
for entity in visible_entities:
    model.learn(entity.concept_id, "proximity", {"health": delta})
```

Agent discovers "skeleton nearby → damage" without textbook declaration.

### 5. generate_candidate_plans (forward imagination)

Replaces backward chaining with **forward imagination**:

```python
def generate_candidate_plans(model, state, spatial_map, visible, depth=3):
    candidates = []

    for concept_id in visible | spatial_map.known_objects:
        for action in model.actions:
            predicted = model.predict(concept_id, action)
            decoded = model.decode_effect(predicted)
            if has_positive_effect(decoded, state):
                plan = build_nav_plan(concept_id, action, spatial_map)
                candidates.append(plan)

    # Multi-step chains via recursive prediction
    candidates += generate_chains(model, state, spatial_map, depth)

    return candidates
```

**generate_chains:** recursive forward search. "If I get wood, what can I do
with wood?" → predict for each action with updated state. Depth 3 default.
Discovers wood → table → sword chain through 3 predicts, not symbolic rules.

**This solves the planner wall:** cumulative `total_gain` in score_trajectory
naturally prefers the sword chain over single wood gather.

### 6. score_trajectory

```python
def score_trajectory(trajectory, model):
    final = trajectory.states[-1].decode(model)
    survived = not trajectory.terminated
    total_gain = sum(all positive inventory deltas across trajectory)
    min_vital = min(final[v] for v in vital_variables)
    steps = len(trajectory.states)
    return (int(survived), total_gain, min_vital, -steps)
```

Lex-tuple with `total_gain` (cumulative) instead of binary `has_gain`.
Long chains with more total gain beat short greedy gathers.

### 7. Perception Interface

**Unchanged:** TileSegmenter (CNN V1), perceive_tile_field(), VisualField.

**New grounding flow:**
- CNN classifies tile as "class_7"
- If "class_7" not in model.concepts → initialize random BitVector
- Agent does `do` near class_7, observes wood +1
- model.learn("class_7", "do", {wood: +1})
- class_7 embedding drifts toward other "gives resource" concepts
- "tree" is a human alias in textbook seed, not a necessity

**Kept as-is:**
- CrafterSpatialMap (with confidence tracking from F10 fix)
- DynamicEntityTracker (entity positions)
- HomeostaticTracker (body rates — category 3 experience)

### 8. Knowledge Flow (Principle 5)

```python
# After episode
model.save("experience/gen_N.bin")
# Saves: SDM locations + counters, concept embeddings, action/role vectors

# Next generation
model = VectorWorldModel(dim=65536)
model.load_from_textbook("configs/crafter_textbook.yaml")  # seed
model.load("experience/gen_N.bin")  # inherited knowledge
# SDM: loaded locations merged into memory (additive counters)
# Concepts: loaded embeddings bundled with seed (majority vote)
# New concepts from experience not in seed: added directly
```

Binary file. No YAML for experience. Vectors ARE knowledge.

Optional promotion: when concept embedding stabilizes (low variance over N
episodes) and association is verified 10+ times → export to YAML for human
review. Not required for operation.

## File Layout

**New files:**
```
src/snks/agent/vector_world_model.py   # VectorWorldModel, BitVector ops, encode/decode
src/snks/agent/vector_sim.py           # VectorState, simulate_forward, apply_effect
src/snks/agent/vector_bootstrap.py     # load_from_textbook (YAML → seed associations)
```

**Modified files:**
```
src/snks/agent/mpc_agent.py            # VectorWorldModel instead of ConceptStore
                                        # generate_candidate_plans → forward imagination
                                        # score_trajectory → total_gain
src/snks/agent/forward_sim_types.py    # Plan, PlanStep kept. SimState/RuleEffect deprecated.
```

**Untouched:**
```
src/snks/agent/perception.py           # HomeostaticTracker, perceive_tile_field
src/snks/agent/crafter_spatial_map.py  # cognitive map + confidence (F10)
src/snks/agent/crafter_textbook.py     # YAML parser (used by vector_bootstrap)
src/snks/agent/tile_segmenter.py       # CNN V1
```

**ConceptStore:** not deleted. Stays in codebase, not imported by mpc_agent.
Removed after VectorWorldModel passes eval gate.

## Testing Strategy

**Unit tests:**
```
tests/test_vector_world_model.py
    - bind/unbind roundtrip
    - encode/decode scalar
    - effect encode/decode (wood: +1 → vector → decode → wood: +1)
    - predict after learn (learn association → predict → correct)
    - surprise high on novel event
    - surprise low after learning
    - concept embedding drift (similar interactions → similar vectors)
    - generalization (tree learned, oak similar → oak predicts wood)

tests/test_vector_sim.py
    - simulate one step
    - simulate chain (wood → table → sword)
    - is_dead detection
    - apply_effect roundtrip

tests/test_vector_bootstrap.py
    - load_textbook creates associations
    - seed predict matches textbook rules

tests/test_vector_mpc.py
    - generate_chains depth 3
    - score total_gain prefers long chain
    - forward imagination finds sword plan
```

**Eval gate (minipc):**
- survival ≥ 155 (current baseline post-Stage 82)
- wood ≥3 in ≥10% episodes (currently 0%)
- surprise correlates with novel events (entity damage, new resource)
- knowledge flow: gen2 warmup > gen1 warmup

## Novelty

1. **No-LLM causal rule induction from surprise in HDC space.** All neurosymbolic
   2025-2026 papers use LLM for rule synthesis. Our surprise → vector association
   → verify → promote is unpublished.

2. **Binary HDC as world model substrate for embodied agent.** HDC literature
   focuses on classification/NLP. Using it as the primary world model for
   planning in a survival game is novel.

3. **Knowledge flow through vector inheritance.** Save/load concept embeddings
   across generations. No YAML serialization of rules — vectors compress
   experience into transferable geometry.

## Addressed Review Findings

**B1 (scalar encoding):** Replaced fractional power encoding with thermometer
encoding. Value K → first K·(dim/max_val) bits = 1. Invertible via popcount.
Crafter values 0-9 fit exactly.

**B2 (XOR toggle):** Replaced single XOR-accumulated vector with SDMMemory
(write accumulates via ±1 counters, read via popcount top-k). Reuses existing
`vsa_world_model.SDMMemory` architecture. No more erase-on-repeat.

**B3 (lossy bundle):** VectorState stores structured dict (inventory, body),
not a bundled vector. `to_vector()` available for similarity/SDM addressing.
Simulation operates on dicts, prediction/learning on vectors.

**W1 (capacity):** SDM with N=10000 locations stores thousands of associations.
Not limited by single-vector bundle capacity.

**W2 (decode_scalar):** Thermometer decode = popcount / (dim / max_val).
O(1) via hardware popcount.

**W3 (combinatorial explosion):** generate_chains uses beam search with
beam_width=5 at each depth level. Max evaluations: 5^3 * |actions| = 1000.
Budget configurable.

**W4 (spurious entity associations):** Negative evidence: when entity visible
but no surprise, weaken association via SDM write with inverted effect vector
(small magnitude). Confidence = read magnitude — low magnitude = unreliable.

**W5 (merge semantics):** Loaded concepts not in seed → added directly.
Loaded concepts in seed → bundled (majority vote, 50/50 weight). SDM merge
is additive (counters sum). Documented in load() contract.

**W6 (no rollback):** SDM counters naturally decay via write saturation.
Periodic re-calibration of SDM radius prevents capacity overflow. Corrupted
single observations are noise — SDM's distributed storage is robust to
individual errors (same property that makes SDR robust).

**N1:** Use torch.Tensor, not np.ndarray. GPU-aware per project convention.

**N3:** Anti-patterns 4 (sleeping baselines) addressed by built-in surprise
diagnostic — sleeping agent has zero surprise, immediately visible.
Anti-pattern 5 (local fixes without architectural movement) — this IS the
architectural movement.

**N5:** Eval gate refined: "mean surprise on first encounter with entity-type
> 2x mean surprise after 5+ encounters with same entity-type."

## Risks

1. **Decode accuracy at dim=65536 with many bundled effects.** Mitigation:
   typical effects are 1-3 deltas, well within capacity. Test with synthetic
   stress (10+ simultaneous deltas).

2. **Forward imagination depth 3 may be too shallow / too expensive.**
   Mitigation: depth is configurable. Profile predict() cost on AMD GPU.

3. **Scalar encoding precision.** Fractional power encoding on binary vectors
   is lossy. Mitigation: Crafter values are small integers (0-9), test
   roundtrip accuracy for this range.

4. **No backward chaining.** Symbolic backward chain had guarantees (if rule
   exists, plan reaches it). Forward imagination may miss valid plans.
   Mitigation: depth 3 + broad action set covers Crafter's craft chains.
   Monitor "plans attempted vs plans that reach goal."

## Connection to IDEOLOGY v2

- **Category 1 (Facts):** YAML textbook → seed vector associations. Teacher
  provides rough priors as before.
- **Category 2 (Mechanisms):** MPC loop, simulate_forward, score_trajectory,
  perception pipeline. Domain-agnostic, unchanged in structure.
- **Category 3 (Experience):** Concept embeddings + associative memory. Updated
  continuously. Persisted via save/load. This IS the knowledge flow.
- **Anti-pattern 1 (learn what teacher knows):** Textbook seeds go straight into
  memory. No learning pipeline to rediscover them.
- **Anti-pattern 2 (env semantics in mechanism):** env_semantics, blocking,
  primitives stay in textbook YAML. VectorWorldModel reads them through
  vector_bootstrap, not hardcoded.
- **Anti-pattern 3 (not watching agent):** Surprise metric is built-in
  diagnostic. Every step produces a measurable prediction error.

## Open Questions (deferred)

- **Q1 (scale):** When concept count exceeds ~1000, linear scan in
  query_similar becomes slow. Solution: LSH index (already exists in
  dcam/lsh.py). Deferred — Crafter has ~20 concepts.
- **Q2 (hierarchy):** H-JEPA suggests multi-level prediction. Current design
  is flat. Add levels when single-level hits a wall.
- **Q3 (latent z):** JEPA's stochastic z for multiple futures. Binary bundle
  of multiple predictions = natural multi-future. Deferred — test single
  prediction first.
