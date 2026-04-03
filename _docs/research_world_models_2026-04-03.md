# World Models for AI Agents: Research Report

**Date:** 2026-04-03
**Context:** SNKS project, Stage 62 BossLevel. Binary VSA (512-bit, XOR bind, majority bundle) + per-rule SDM architecture. Current system has 7 causal rule SDMs, works at small scale, fails at ~6000 items in single SDM.

---

## Table of Contents

1. [LeCun's JEPA and Energy-Based World Models](#1-lecuns-jepa)
2. [World Models in RL](#2-world-models-in-rl)
3. [SDM as World Model: Capacity Analysis and Fixes](#3-sdm-as-world-model)
4. [Bio-Inspired World Models](#4-bio-inspired-world-models)
5. [VSA for World Models](#5-vsa-for-world-models)
6. [Architectural Recommendations for SNKS](#6-recommendations)

---

## 1. LeCun's JEPA

### Core Ideas (LeCun 2022, "A Path Towards Autonomous Machine Intelligence")

JEPA = Joint Embedding Predictive Architecture. The key departure from generative world models: **predict in representation space, not pixel space**. A JEPA world model has:

- **Encoder** x -> s_x (maps observation to representation)
- **Predictor** (s_x, a) -> s_y_hat (predicts next representation given action)
- **Energy function** E(s_y_hat, s_y) = distance in representation space

The predictor never reconstructs raw observations. This avoids the "curse of pixel prediction" that plagues VAE-based world models (Ha & Schmidhuber 2018 spend all capacity modeling irrelevant visual details).

### Hierarchical Planning via JEPA Stack

LeCun proposes a hierarchy of JEPA modules at different time scales:
- **Level 0**: 100ms, low-level motor predictions
- **Level 1**: seconds, object interaction predictions  
- **Level 2**: minutes-hours, abstract plan chunks

Each level's predictor operates on increasingly abstract representations. Higher levels predict coarser, longer-term state changes. This maps to the **dorsal/ventral hierarchy** in neuroscience.

### What is Relevant to SNKS

Your VSA encodings ARE representations, not raw observations. Your CausalWorldModel already predicts in "representation space" (VSA vectors). In JEPA terms:
- VSAEncoder = encoder (obs -> s)
- SDM read = predictor (s + action -> s')
- Hamming similarity = energy function

**The gap**: JEPA assumes learned representations via contrastive/non-contrastive training. Your representations are hand-designed (role-filler bindings). This is actually an advantage for interpretability and one-shot learning, but limits generalization to novel features.

### I-JEPA and V-JEPA (2023-2024)

Meta's I-JEPA (image) and V-JEPA (video) validated the core idea. V-JEPA learns video representations without pixel reconstruction. However:
- Both use gradient descent heavily (ViT backbone)
- Pre-training requires millions of examples
- Not directly applicable to one-shot causal learning

### What to Take from JEPA

1. **Predict in representation space** -- you already do this
2. **Hierarchical predictors at multiple timescales** -- this is the key architectural insight for scaling. Your 7 rule types are a flat set; JEPA suggests organizing them hierarchically (motor actions < object interactions < room-level plans)
3. **Energy-based selection** -- instead of hard matching, use soft similarity with energy-based ranking. You partially do this with SDM confidence scores.

---

## 2. World Models in RL

### Ha & Schmidhuber "World Models" (2018)

Architecture: VAE (visual encoder) + MDN-RNN (transition model) + Controller.
- VAE compresses 64x64 frames to z (latent, 32-dim)
- MDN-RNN: P(z_{t+1} | z_t, a_t, h_t) -- recurrent, predicts next latent
- Controller: linear policy from (z, h) -> action

Key result: agent can learn entirely "in its dream" (inside the world model). Car racing task solved with ~10K environment interactions.

**Relevance**: The MDN-RNN is a learned transition function. Your SDM is a memorized transition function. The advantage of SDM: one-shot writes, no training loop. The disadvantage: capacity limits (their RNN can compress/generalize, your SDM cannot).

### DreamerV3 (Hafner et al., 2023)

State of the art for model-based RL with learned world models. Key innovations:
- **RSSM** (Recurrent State Space Model): deterministic + stochastic state components
- **Symlog predictions**: handles varying reward scales
- **Works across 150+ tasks** without hyperparameter tuning

Architecture: encoder -> RSSM -> decoder + reward + continue predictors. Imagined rollouts (15-step) used for policy training via actor-critic in imagination.

**Key numbers**: DreamerV3 needs ~1M environment steps for Atari, ~10M for DMC. This is 3-4 orders of magnitude more data than your demo-based approach (you use ~200 demos).

**What to take**: The RSSM splits state into deterministic (history summary) and stochastic (current uncertainty). For SDM: you could maintain a deterministic context vector (accumulating VSA bundle of recent transitions) alongside the stochastic SDM readout.

### IRIS (Micheli et al., 2023)

Autoregressive Transformer world model. Tokenizes observations with discrete VAE, then GPT-like model predicts next tokens. Achieves human-level Atari in 100K steps.

**Relevance**: minimal. Fully gradient-based, transformer-heavy.

### MuZero (Schrittwieser et al., 2020)

Plans without knowing the rules. Learned model predicts: reward, value, policy (but NOT observations). Tree search (MCTS) in learned latent space.

Key architectural insight: **the model does not need to predict observations, only quantities useful for planning** (reward, value, valid actions). This aligns with your CausalWorldModel -- you predict preconditions and consequences, not full next states.

**What to take from MuZero**: 
1. Prediction targets should be planning-relevant (preconditions, effects, rewards), not full state reconstructions
2. Tree search in abstract space is powerful -- your backward chaining QA-C is a form of this

### Summary: RL World Models vs Your Approach

| Feature | DreamerV3/MuZero | SNKS CausalWorldModel |
|---------|-----------------|----------------------|
| Learning | Gradient descent, millions of steps | One-shot SDM writes |
| Representation | Learned latent | Hand-designed VSA |
| Prediction target | Full next state / value | Preconditions + effects |
| Planning | Imagination rollout / MCTS | Backward chaining QA |
| Capacity | Unlimited (parameter count) | SDM-limited (~50 items/SDM) |
| Generalization | Via neural compression | Via VSA identity property |
| Data efficiency | 100K-10M steps | 10-200 demos |

Your approach is radically more data-efficient but hits capacity walls. The core challenge is scaling the memory, not the representation.

---

## 3. SDM as World Model: Capacity Analysis

### Kanerva's Original SDM (1988)

Sparse Distributed Memory: N hard locations in D-dimensional binary space. Write: activate locations within Hamming radius r of address, add content. Read: activate same locations, sum content, threshold.

**Theoretical capacity** (Kanerva 1988): For N locations and D dimensions, the number of patterns M that can be reliably stored satisfies:

    M < N / (D * k)

where k depends on the activation fraction. For your setup:
- D = 512, N = 10000, activation ~1-5%
- Theoretical clean capacity: ~200-500 patterns per SDM

With 6000 transitions in one SDM, you are 10-30x over capacity. The failure is expected.

### Why Per-Rule SDMs Work

With 7 rule types and ~10-50 items each, each SDM stores well within capacity. The per-rule factoring is essentially **hash bucketing by rule type** -- you reduce interference by ensuring only semantically similar patterns share an SDM.

### Known Capacity Enhancement Techniques

**1. Increase dimensionality**
- Going from D=512 to D=2048 increases capacity roughly 4x (capacity scales sublinearly with D)
- Cost: 4x memory, 4x compute for Hamming distance
- Practical: your SDM at D=2048, N=10000 could hold ~1000-2000 clean patterns

**2. Increase N (number of hard locations)**
- Linear increase in capacity, linear increase in memory
- N=100000 with D=512: ~2000-5000 patterns
- Cost: 100K * 512 * 4 bytes = 200MB per SDM (manageable)

**3. Sparse activation with learned addresses**
- Instead of random hard locations, use data-dependent addresses
- Jaeckel (1989): content-addressable SDM where hard locations are data points themselves
- This eliminates random interference; only genuinely similar patterns interfere

**4. Partitioned SDM (critical for your case)**
- Instead of one giant SDM, use a routing function to direct queries to sub-SDMs
- Your per-rule approach is a manual version of this
- Automated approach: hash the query address to select 1 of K sub-SDMs
- With K=100 sub-SDMs and 6000 total items: 60 items/SDM on average (within capacity)

**5. SDM with cleanup memory**
- After SDM read, project result onto nearest known clean pattern
- Eliminates accumulated noise from interference
- Requires a separate "item memory" of all stored patterns
- Neuroscience analog: hippocampal pattern completion + neocortical cleanup

### What Does Not Work

- **Simply writing more patterns**: interference grows quadratically with pattern count
- **Adaptive radius only**: helps somewhat but does not solve fundamental capacity
- **Multiple reads and averaging**: noise is systematic (from interfering patterns), not random -- averaging does not help
- **Increasing activation fraction**: more activated locations = more interference, not less

### Concrete Capacity Fix for SNKS

Your current architecture (7 per-rule SDMs, 10-50 items each) is actually well-designed for the current scale. The question is scaling to thousands of rules.

**Recommended approach: Hierarchical Partitioned SDM**

```
Level 1: Rule-type router (7+ categories)
Level 2: Per-rule-type SDM banks (10-20 sub-SDMs per type)
Level 3: Individual SDMs (N=1000, D=512, capacity ~100 patterns each)
```

Total capacity: 7 types * 15 sub-SDMs * 100 patterns = ~10,500 causal rules.
Total memory: 7 * 15 * (1000 * 512 * 4 bytes) = ~210 MB.

The routing at Level 2 uses the VSA address itself: hash a portion of the address bits to select the sub-SDM. This is biologically plausible (hippocampal subfields).

---

## 4. Bio-Inspired World Models

### Hippocampal Replay and Memory Consolidation

The hippocampus is the brain's SDM. Key mechanisms relevant to world models:

**1. Sharp-wave ripples (SWR) and replay**
During rest/sleep, the hippocampus replays experienced sequences in compressed time (20x speedup). This serves two functions:
- **Memory consolidation**: transfer from hippocampus (fast, limited capacity) to neocortex (slow, large capacity)
- **Planning**: "preplay" of novel trajectories by recombining experienced segments

This directly maps to your capacity problem. Your SDM (hippocampus) has limited capacity. You need a "neocortex" for consolidated, compressed knowledge.

**Implication**: After learning rules via SDM, consolidate frequently-accessed rules into a fixed lookup table or compressed format. The SDM handles novel/recent rules; consolidated rules go into a separate structure.

**2. Place cells and cognitive maps**
Hippocampal place cells form a topological map of space. The brain's world model for navigation is not a transition table but a metric map with local dynamics.

Your SpatialMap + FrontierExplorer already implement this. This is correct.

**3. Entorhinal grid cells**
Grid cells provide a coordinate system for the cognitive map. They enable:
- Path integration (dead reckoning)
- Novel shortcut computation
- Multi-scale spatial representation

For SNKS: your pathfinding already handles this via grid coordinates. No immediate action needed.

### Predictive Coding (Rao & Ballard 1999, Friston's Free Energy)

The cortex implements a hierarchical prediction machine:
- Each level generates predictions of the level below
- Only prediction errors propagate upward
- Learning minimizes prediction error (free energy)

**Relevance to world models**: A predictive coding world model only transmits surprises. For your system:
- When the world matches the causal model, no update needed
- When a prediction error occurs (unexpected transition), that is the signal to learn a new rule

**Concrete implementation**: Before writing to SDM, check if the transition is already predicted correctly. If yes, skip the write (saves capacity). If no, write it AND flag it as a novel rule type.

This "write-on-surprise" policy could dramatically reduce SDM load. If 90% of transitions follow known rules, you only store the 10% that are novel.

### Complementary Learning Systems (McClelland et al., 1995, Kumaran et al., 2016)

The CLS theory: two memory systems with complementary properties:

| Property | Hippocampus (fast) | Neocortex (slow) |
|----------|-------------------|-------------------|
| Learning rate | One-shot | Gradual |
| Capacity | Limited | Very large |
| Representation | Pattern-separated | Overlapping/compressed |
| Function | Episode storage | Generalized knowledge |

**Your SDM = hippocampus**: one-shot writes, limited capacity, pattern-separated (per-rule).
**What you are missing = neocortex**: a slow-learning, high-capacity, compressed knowledge store.

**CLS-based architecture for SNKS**:

```
SDM (hippocampal) -- stores recent/novel causal rules, one-shot
    |
    | consolidation (periodic)
    v
Rule Table (neocortical) -- stores verified, compressed rules
    |
    | if SDM query misses, check Rule Table
    v
    Combined answer
```

The Rule Table is simply a Python dict mapping (rule_type, key_features) -> (preconditions, effects). After an SDM rule has been accessed N times and always returns the same answer, promote it to the Rule Table and free SDM capacity.

This is not cheating -- it is exactly what the brain does. The hippocampus is a staging area, not the final store.

### Successor Representation (Dayan 1993, Momennejad 2017)

SR = a representation that encodes expected future state occupancy:
    M(s, s') = E[sum_t gamma^t * I(s_t = s') | s_0 = s]

The SR decomposes value into **transition structure** (M) and **reward** (w):
    V(s) = M(s, :) . w

**Why this matters**: The SR can be stored in SDM-like associative memory (Stachenfeld et al. 2017 showed hippocampal place cells encode SR). It supports:
- Instant revaluation when rewards change (just update w)
- Transfer between tasks with same transition structure
- Efficient planning without full forward simulation

**SR in VSA**: Encode M(s, :) as a VSA vector (bundle of discounted future states). Store in SDM. For planning: retrieve M(s, :), unbind with goal state to estimate distance/reachability.

This is a powerful complement to your backward chaining QA-C.

---

## 5. VSA for World Models

### Holographic Reduced Representations (Plate 1995, 2003)

Tony Plate's HRR = the continuous-valued cousin of your binary VSA. Key operation: circular convolution for binding, superposition for bundling.

Plate showed HRRs can represent:
- **Sequences**: bind(pos_1, item_1) + bind(pos_2, item_2) + ...
- **Stacks**: recursive binding for nested structures
- **Variable bindings**: role-filler pairs (exactly what you use)
- **Analogical retrieval**: given partial structure, retrieve best match

**Relevance**: Your role-filler encoding is textbook HRR/VSA. The gap is in how you USE the stored representations for reasoning.

### VSA for Reasoning (Gayler 2003, Kanerva 2009, Kleyko et al. 2021-2023)

Kleyko et al.'s comprehensive survey (2023) covers VSA applications. Key findings for reasoning:

**1. VSA for graph operations**
- A graph (V, E) can be encoded as: G = sum_e bind(src_e, edge_type, dst_e)
- Querying: unbind with known elements to retrieve unknown
- Works for small graphs (< 100 edges at D=10000)
- **Capacity**: ~D/log(D) items in a single bundle (for D=512: ~55 items)

This matches your observation: ~50 items per SDM works, 6000 does not.

**2. VSA for causal inference**
Emruli et al. (2013, 2014) used VSA for storing and querying causal relationships:
- Causal rule: bind(cause, CAUSES, effect)
- Forward inference: unbind(rule, cause) -> extract effect
- Backward inference: unbind(rule, effect) -> extract cause
- Chain inference: compose multiple rules via sequential unbinding

Their results: works reliably for up to ~30-50 rules at D=1000. Beyond that, retrieval errors climb rapidly.

**3. VSA for planning (Rasanen & Saarinen 2016)**
Used VSA to represent STRIPS-like planning operators:
- Operator = bundle(bind(PRECOND, prec), bind(EFFECT_ADD, add), bind(EFFECT_DEL, del))
- Plan search: match current state against preconditions, apply effects, iterate

Worked for toy problems (Blocks World, 5-10 blocks). Did not scale to complex domains.

**4. Resonator Networks (Frady et al. 2020, Kent et al. 2020)**
Resonator networks are iterative VSA decoders that can factorize a composite VSA vector into its constituents. Think of it as VSA's version of "attention."

Given: x = bind(a, b, c) and codebooks for a, b, c
Find: which specific a, b, c were bound together

This is done via iterative message passing between factor nodes. Convergence in ~10-50 iterations for 3-5 factors.

**Relevance**: If you store a causal rule as bind(precond, action, effect), a resonator network can decompose a query and find the best matching rule from a bundle -- more reliably than single-step SDM readout.

### VSA Capacity: The Hard Numbers

For binary VSA with D=512:
- **Bundle capacity** (items in one superposition): ~25-55 items
- **Clean-up memory capacity** (items distinguishable in codebook): ~2^(D/4) = 2^128 (practically unlimited)
- **Sequence memory** (position-bound items): ~D/4 = 128 positions
- **SDM capacity** (with N=10000): ~200-500 patterns

For D=1024: all numbers roughly double.
For D=2048: roughly 4x.
For D=10000: bundle capacity ~500-1000 items, SDM with N=100K: ~5000 patterns.

### What Actually Works in VSA World Models

Based on the literature and your empirical results:

**Works well**:
- Role-filler encoding of structured states (your VSAEncoder)
- Per-type SDM storage with <50 items (your CausalWorldModel)
- XOR bind for self-inverse property (bind(X,X) = identity)
- Simple QA via unbinding (your QA-A, QA-B)

**Does not work well**:
- Massive flat bundles (>100 items at D=512)
- Single SDM for heterogeneous content (your failed 6000-item experiment)
- Long causal chains via sequential unbinding (noise compounds at each step)
- Continuous variable representation (VSA is inherently discrete/categorical)

---

## 6. Architectural Recommendations for SNKS

Based on all five research areas, here are concrete, implementable recommendations ordered by priority.

### Recommendation 1: Hierarchical Partitioned SDM (Immediate, High Impact)

**Problem**: Scaling beyond 7 rule types with 50 items each.

**Solution**: Two-level routing.

```
CausalWorldModel
  |
  |- RuleTypeRouter (Level 1): hash(rule_type) -> one of K banks
  |    |- "same_color_unlock" -> Bank 0
  |    |- "pickup_requires_adjacent" -> Bank 1
  |    |- ... (current 7 types)
  |    |- "door_color_red_unlock" -> Bank 7  (auto-split when Bank 0 overflows)
  |    |- ...up to 100+ banks
  |
  |- SubBank Router (Level 2): hash(address_bits[0:64]) -> one of M sub-SDMs
       |- sub-SDM 0 (N=1000, D=512)
       |- sub-SDM 1
       |- ...
```

**Implementation**:
- Keep your existing per-rule SDM architecture
- Add automatic splitting: when an SDM's n_writes exceeds a threshold (e.g., 100), split it into 2 sub-SDMs by hashing the first 64 address bits
- Use a dict[str, list[SDMMemory]] instead of dict[str, SDMMemory]
- Read: hash address to find sub-SDM, read from that one only
- Write: same routing

**Estimated capacity**: 1000+ rule types, 100 items each = 100K causal rules.
**Memory cost**: ~500MB at maximum scale, fits in your 14GB RAM.

**Code-level change**: Modify CausalWorldModel to hold banks of SDMs per rule type, with a split/route mechanism. The external API (qa_a, qa_b, qa_c) stays identical.

### Recommendation 2: Complementary Learning System (Medium-term, High Impact)

**Problem**: SDM capacity is fundamentally limited; frequently-used rules waste it.

**Solution**: Two-tier memory inspired by hippocampal-neocortical CLS.

```python
class TwoTierCausalMemory:
    """Fast SDM for novel rules + consolidated dict for verified rules."""
    
    def __init__(self):
        self.sdm_tier = CausalWorldModel(...)   # hippocampal: one-shot, limited
        self.rule_table = {}                     # neocortical: verified, unlimited
        self.access_counts = defaultdict(int)    # track per-rule access
        self.consolidation_threshold = 5         # promote after N consistent reads
    
    def query(self, rule_type, address):
        # Check consolidated rules first (fast, exact)
        key = self._make_key(rule_type, address)
        if key in self.rule_table:
            return self.rule_table[key]
        # Fall back to SDM (slower, approximate)
        return self.sdm_tier.query(rule_type, address)
    
    def consolidate(self):
        """Promote well-established SDM rules to rule_table."""
        # For each rule type, read all stored patterns
        # If a pattern has been accessed N times with consistent result,
        # move to rule_table and free SDM capacity
        ...
```

**Why this is not cheating**: 
- The SDM handles novel situations (one-shot learning, generalization via similarity)
- The rule table handles known situations (exact match, zero noise)
- This is exactly what the brain does: hippocampus for new memories, neocortex for established knowledge
- The SDM remains necessary for generalization to novel combinations

**Estimated impact**: Reduces effective SDM load by 80-90% for stable environments. The SDM only needs to handle the ~10% of queries that involve novel situations.

### Recommendation 3: Write-on-Surprise Policy (Immediate, Easy)

**Problem**: Most observed transitions are predictable; writing them wastes SDM capacity.

**Solution**: Before writing to SDM, check if the transition is already correctly predicted.

```python
def maybe_write(self, state, action, next_state, reward):
    """Only write if the transition is surprising."""
    predicted_next, confidence = self.sdm.read_next(state, action)
    if confidence > 0.3:
        sim = VSACodebook.similarity(predicted_next, next_state)
        if sim > 0.8:
            return  # already known, skip write
    # Novel or incorrectly predicted: write it
    self.sdm.write(state, action, next_state, reward)
```

**Impact**: Reduces write count by 50-90% in environments with regularities. Directly extends effective capacity.

### Recommendation 4: Resonator Network for Multi-Factor QA (Medium-term)

**Problem**: Complex queries (e.g., "what action achieves effect E given precondition P?") require decomposing composite VSA vectors.

**Solution**: Implement a resonator network for iterative factorization.

A resonator network iterates:
```
For each factor i:
    x_i <- cleanup(unbind(target, product(x_j for j != i)))
```

Where cleanup finds the nearest vector in the codebook for factor i.

This enables:
- Querying composite rules: "given I have a red key and face a locked door, what should I do?"
- Decomposing observed states into role-filler pairs
- Analogical reasoning: "this situation is LIKE that other one because they share factors X and Y"

**Implementation complexity**: Moderate (50-100 lines). The cleanup step is a codebook lookup (cosine similarity / Hamming distance to all codebook entries).

### Recommendation 5: Successor Representation in VSA (Longer-term, Research)

**Problem**: Planning requires forward chaining through the world model, which compounds SDM noise.

**Solution**: Pre-compute SR vectors that encode multi-step reachability.

```
SR(s) = bundle([s, gamma * s', gamma^2 * s'', ...])
```

For each state cluster, maintain an SR vector. Planning becomes:
1. Compute similarity(SR(current), goal) for candidate actions
2. Pick the action whose successor state has highest SR-similarity to goal

**Benefits**:
- One-step readout for multi-step planning (avoids compounding noise)
- Instant revaluation when goals change
- Compatible with VSA bundle representation

**Risk**: SR vectors are dense (bundle of many states), so they may exceed bundle capacity for complex environments. Mitigated by using higher D or hierarchical SRs.

### Recommendation 6: Increase VSA Dimensionality to 2048 (Easy Win)

**Problem**: D=512 gives bundle capacity of ~50 items.

**Solution**: Increase to D=2048.

**Impact**:
- Bundle capacity: ~200 items
- SDM capacity (N=10000): ~1000-2000 patterns
- Memory cost: 4x current (still small)
- Compute cost: 4x for Hamming distance (still fast on CPU)

This is the simplest way to double or triple your headroom without architectural changes. Can be combined with all other recommendations.

### Recommendation 7: Hierarchical Temporal Abstraction (JEPA-inspired)

**Problem**: All rules are at the same temporal granularity (one-step transitions). Planning long action sequences requires many SDM reads.

**Solution**: Store rules at multiple temporal abstractions.

```
Level 0: primitive actions (toggle, pickup, move)
Level 1: tactical sequences (go_to_key, unlock_door)  <- your current subgoals
Level 2: strategic plans (clear_room, solve_mission)
```

Each level has its own SDM bank. Level N rules reference Level N-1 rule outputs.

**This maps to your existing architecture**:
- Level 0: CausalWorldModel rules
- Level 1: MissionModel subgoals  
- Level 2: BossLevelAgent episode planning

The recommendation is to formalize this hierarchy so that Level 2 rules can be stored and queried via VSA+SDM, not just hardcoded in the agent class. This enables learning new strategic patterns from demos.

---

## Summary: Priority Matrix

| # | Recommendation | Effort | Impact | Prerequisites |
|---|---------------|--------|--------|---------------|
| 1 | Hierarchical Partitioned SDM | Medium | High | None |
| 2 | Complementary Learning System | Medium | High | None |
| 3 | Write-on-Surprise | Low | Medium | None |
| 4 | Resonator Network | Medium | Medium | Rec 6 helps |
| 5 | Successor Representation | High | High | Research needed |
| 6 | Increase D to 2048 | Low | Medium | None |
| 7 | Hierarchical Temporal Abstraction | High | High | Recs 1-2 |

**Recommended execution order**: 3 -> 6 -> 1 -> 2 -> 4 -> 7 -> 5

Start with write-on-surprise (immediate capacity relief, 10 lines of code), then bump dimensionality (config change), then implement partitioned SDM (structural change), then CLS consolidation (new component), then the research-heavy items.

---

## Key References

1. LeCun, Y. (2022). "A Path Towards Autonomous Machine Intelligence." OpenReview.
2. Ha, D. & Schmidhuber, J. (2018). "World Models." arXiv:1803.10122.
3. Hafner, D. et al. (2023). "Mastering Diverse Domains through World Models." (DreamerV3) arXiv:2301.04104.
4. Schrittwieser, J. et al. (2020). "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model." (MuZero) Nature.
5. Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
6. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." Cognitive Computation.
7. Kleyko, D. et al. (2023). "A Survey on Hyperdimensional Computing." ACM Computing Surveys.
8. Plate, T. (2003). "Holographic Reduced Representations." CSLI.
9. Frady, E.P. et al. (2020). "Resonator Networks: Robust and Efficient Factorization." Neural Computation.
10. McClelland, J.L. et al. (1995). "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex." Psychological Review.
11. Kumaran, D. et al. (2016). "What Learning Systems Do Intelligent Agents Need?" Neuron.
12. Dayan, P. (1993). "Improving Generalization for Temporal Difference Learning: The Successor Representation." Neural Computation.
13. Stachenfeld, K.L. et al. (2017). "The Hippocampus as a Predictive Map." Nature Neuroscience.
14. Emruli, B. et al. (2014). "Analogical Mapping and Inference with Binary Spatter Codes and Sparse Distributed Memory." IJCNN.
15. Rao, R.P. & Ballard, D.H. (1999). "Predictive Coding in the Visual Cortex." Nature Neuroscience.
16. Micheli, V. et al. (2023). "Transformers are Sample-Efficient World Models." (IRIS) ICLR.
