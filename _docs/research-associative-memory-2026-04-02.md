# Associative Memory Systems for AGI World Models

**Date:** 2026-04-02
**Context:** SNKS system -- spiking FHN oscillator network (50K nodes), SDR representations, no backpropagation. Need a universal world model supporting prediction, causality, planning, and arbitrary queries via associative memory.

---

## 1. Classical Associative Memory

### 1.1 Hopfield Networks (1982)

**Mechanism:** Binary attractor network with symmetric weights. Patterns are stored as fixed points; retrieval is gradient descent on an energy function. Learning rule is outer-product (Hebbian): `W = sum(x_i * x_i^T)`.

**Capacity:** ~0.14N patterns for N neurons (Amit et al. 1985). For 50K nodes, that is ~7,000 patterns. Retrieval is content-addressable: present a partial/noisy pattern, network converges to the nearest stored attractor.

**Pros:**
- Fully local learning (outer product rule)
- Content-addressable retrieval
- Well-understood mathematically
- Naturally maps to oscillator/spiking networks

**Cons:**
- Catastrophic interference at capacity
- No temporal/sequential storage (static attractors only)
- Correlated patterns degrade capacity severely
- Retrieval is slow (iterative relaxation)
- No variable binding or structured queries

**Relevance to SNKS:** Direct conceptual ancestor. FHN oscillator SKS (Stable Kinetic States) already function as attractor states. The question is how to go beyond basic pattern completion.

### 1.2 Sparse Distributed Memory (Kanerva, 1988)

**Mechanism:** High-dimensional binary address space (typically 1000 bits). "Hard locations" (physical memory rows) are randomly distributed in this space. A write activates all hard locations within Hamming radius of the address; a read sums the contents of activated locations. Naturally produces SDRs.

**Capacity:** Scales well -- critical radius allows clean retrieval with O(1M) hard locations in 1000-bit space. Noise-tolerant due to distributed storage.

**Pros:**
- Designed for SDR -- perfect match for SNKS's existing SDR pipeline
- One-shot write, no iterative learning needed
- Naturally handles noise and partial cues
- Can store heteroassociations (address -> content mapping)
- Sequence storage via chaining: store (pattern_t -> pattern_{t+1})
- Online/continual learning by construction

**Cons:**
- Raw capacity depends on number of hard locations (memory rows)
- No native variable binding (but composable with VSA)
- Retrieval quality degrades with many writes to overlapping addresses
- Not natively GPU-optimized (sparse access patterns)

**Relevance to SNKS:** Strong candidate. SDR-native, local write rule, supports heteroassociation for prediction. 50K hard locations in 1000-bit space is feasible. The question is GPU efficiency of the sparse access pattern.

### 1.3 Bidirectional Associative Memory (Kosko, 1988)

**Mechanism:** Two-layer network with bidirectional weights. Stores associations between pattern pairs (X, Y). Retrieval: present X, get Y, feed back, iterate to convergence.

**Capacity:** Limited -- roughly min(M, N) patterns for layers of size M and N. Less practical than Hopfield or SDM for large-scale use.

**Relevance to SNKS:** Minimal. Subsumed by modern approaches. Mentioned for completeness.

---

## 2. Modern Approaches (2020-2025)

### 2.1 Modern Hopfield Networks / Dense Associative Memory

**Key paper:** Ramsauer et al. "Hopfield Networks is All You Need" (ICLR 2021)

**Mechanism:** Replaces quadratic energy function with exponential interaction function. This yields exponential storage capacity (~exp(N/2)) and one-step retrieval equivalent to softmax attention. The update rule becomes: `new_state = softmax(beta * X^T * xi) * X` -- identical to transformer attention.

**Capacity:** Exponential in dimension -- vastly exceeds classical Hopfield. For 50K-dimensional patterns, capacity is astronomically large.

**Pros:**
- Massive storage capacity
- One-step retrieval (no iteration)
- Mathematically elegant connection to attention
- Well-suited to GPU (matrix multiplies)

**Cons:**
- The update rule IS effectively a form of gradient computation (softmax over all stored patterns)
- Storing new patterns requires appending to memory matrix -- not strictly local
- The exponential energy function has no obvious biological/spiking implementation
- Retrieval requires computing similarity against ALL stored patterns (O(N*M) for M memories)
- Not compatible with local Hebbian learning in its modern form

**Relevance to SNKS:** Theoretically interesting but architecturally incompatible. The softmax/exponential mechanism does not map to spiking dynamics. However, the CONCEPT of increasing interaction order (polynomial energy rather than quadratic) could inspire modifications to SNKS coupling.

### 2.2 Kanerva Machine (Wu et al., 2018)

**Mechanism:** Variational memory architecture combining SDM with a generative model. Uses learned read/write operations with a fixed set of memory slots.

**Relevance to SNKS:** Uses backprop for the variational component. Not directly applicable. However, the SDM-inspired addressing scheme is relevant.

### 2.3 Hyperdimensional Computing / Vector Symbolic Architectures (VSA)

**Key authors:** Kanerva (2009), Plate (1995 -- HRR), Gayler (2003 -- MAP), Rachkovskij (sparse binary)

This is a family of approaches, not a single architecture. All share the principle: represent structured information as high-dimensional vectors using three operations:

1. **Bundling (addition/OR):** Combine concepts into a set. `A + B + C` = "all of these"
2. **Binding (element-wise multiply/XOR/circular convolution):** Create associations. `role * filler` = "role is bound to filler"
3. **Permutation (shift):** Encode sequence/order. `p(A)` = "A in position 1"

**Variants relevant to SNKS:**

| Variant | Vector type | Binding op | Bundling op | Notes |
|---------|------------|------------|-------------|-------|
| **BSC** (Binary Spatter Code) | Binary {0,1} | XOR | Majority vote | Simplest, SDR-compatible |
| **MAP** (Multiply-Add-Permute) | Bipolar {-1,+1} | Element-wise multiply | Addition + threshold | Fast, GPU-friendly |
| **HRR** (Holographic Reduced Repr.) | Real-valued | Circular convolution | Addition | Elegant math, FFT-based |
| **MBAT** (Sparse binary) | Sparse binary | Shift+AND | OR | Most SDR-compatible |

**Capacity:** For N-dimensional vectors, can store O(N / log N) bound pairs in a single bundle. For N=1000, roughly 100 bindings. For N=10000, roughly 1000.

**Key capability -- variable binding:**
```
scene = (color * blue) + (shape * triangle) + (size * large)
query: what is the color?
answer: unbind(scene, color) = scene (*) color^{-1} ≈ blue
```

This is EXACTLY the "what color was the key?" query type needed.

**Sequential/temporal encoding:**
```
sequence = p^0(A) + p^1(B) + p^2(C)   # A then B then C
query: what comes after A?
answer: unbind(sequence, p^0) then find next...
```

Or more naturally via chaining: `AB = bind(A, p(B))`, `BC = bind(B, p(C))`.

**Pros:**
- Variable binding is a FIRST-CLASS operation
- Arbitrary queries without retraining
- All operations are local (no backprop)
- Composable: can represent arbitrarily complex structures
- Naturally maps to SDR (especially BSC and MBAT variants)
- GPU-friendly (element-wise ops, optionally FFT)
- O(N) compute per operation
- Continual learning: just add new bindings to memory

**Cons:**
- Capacity limited by dimensionality (need high-D vectors, 1000-10000)
- Retrieval is approximate (similarity-based, not exact)
- Deep nesting degrades signal (binding chains lose fidelity)
- No built-in learning of WHAT to bind -- needs an external controller
- Causality and prediction require additional architecture on top

**Relevance to SNKS:** EXTREMELY HIGH. This is the strongest candidate for the structured representation layer. SDR-compatible, local operations, supports variable binding and queries. The main question is how to interface VSA operations with FHN oscillator dynamics.

### 2.4 Differentiable Neural Dictionary (DND)

**Mechanism:** Key-value memory with differentiable lookup. Used in NEC (Neural Episodic Control). Stores (key, value) pairs; retrieval is weighted sum by kernel similarity.

**Relevance to SNKS:** Requires backprop for the embedding network. Not directly applicable. But the key-value lookup concept maps well to SDM.

---

## 3. Biological/Cognitive Inspiration

### 3.1 Complementary Learning Systems (CLS) -- McClelland et al. (1995), O'Reilly (2014)

**Core idea:** Two memory systems with complementary properties:

1. **Hippocampus:** Fast, one-shot, sparse, pattern-separated. Stores specific episodes. High learning rate.
2. **Neocortex:** Slow, gradual, distributed, overlapping. Stores statistical regularities. Low learning rate.

Consolidation: hippocampal memories are "replayed" during sleep to gradually train neocortex.

**Relevance to SNKS:** This is the architecture template. Map to SNKS:
- **Hippocampal component:** SDM or sparse associative memory for one-shot episodic storage
- **Neocortical component:** Hebbian-learned distributed representations in the oscillator network
- **Consolidation:** Replay of stored episodes during idle periods (the "sleep" phase from the roadmap)

### 3.2 Hippocampal Indexing Theory (Teyler & DiScenna, 1986)

Hippocampus stores sparse INDEX patterns that, when reactivated, reinstate the full cortical pattern. This is essentially SDM: the hippocampal index is the address, the cortical pattern is the content.

### 3.3 STDP-Based Associative Learning

**Spike-Timing-Dependent Plasticity** naturally implements temporal association: if neuron A fires before neuron B, strengthen A->B. This creates directed associations (A predicts B), which is exactly what a predictive world model needs.

**Key property:** STDP automatically builds a temporal transition model: the connection weights encode P(B fires | A just fired). This IS a world model in compressed form.

**Relevance to SNKS:** Already partially implemented via eligibility traces. The question is how to layer structured (VSA-style) representations on top of STDP-learned temporal associations.

### 3.4 Temporal Context Model (Howard & Kahana, 2002)

Items are associated not directly to each other but to a slowly drifting temporal context vector. Retrieval of an item reinstates its temporal context, which then cues nearby items. This naturally produces temporal clustering in free recall.

**Relevance to SNKS:** Could be implemented as a slowly-varying "context SDR" that is bound (via VSA) to each experience. Provides temporal grounding without explicit sequence storage.

---

## 4. World Models from Associative Memory

### 4.1 Graph-Based World Models

Represent world state as a graph: nodes = entities/concepts, edges = relations. Transitions are graph edits. Planning = graph search.

**Examples:** DYNA-like architectures with tabular models, relational world models.

**Pros:** Explicit, inspectable, supports planning via search.
**Cons:** Combinatorial explosion, hard to learn graph structure from raw observations.

**Connection to VSA:** A VSA can represent graphs. Each node is a high-D vector. Edges are bindings: `edge = bind(node_A, bind(relation, node_B))`. The entire graph is a bundle of edges. This is sometimes called a "holographic graph."

### 4.2 Schema Theory / Frames (Minsky, 1975; Bartlett, 1932)

Schemas are structured memory templates with slots and default values. A "kitchen schema" has slots for stove, fridge, table with typical fillers.

**Connection to VSA:** Schemas map naturally to VSA bundles:
```
kitchen = bind(has_stove, yes) + bind(has_fridge, yes) + bind(floor, tile) + ...
```

Slot-filling = binding, default values = initial bundle content, instantiation = adding specific bindings.

### 4.3 Predictive Processing + Associative Memory

The brain as a prediction machine (Friston, Clark). Each level of hierarchy predicts the activity of the level below. Prediction errors drive learning.

**Connection to SNKS:** The oscillator network's coupling already implements a form of prediction (coupled oscillators entrain to predicted patterns). Adding an explicit associative memory for storing and retrieving predictions could implement a predictive processing hierarchy.

### 4.4 Proposed Architecture: VSA-SDM World Model

No single existing system does everything needed. But a COMBINATION of VSA + SDM + STDP-based temporal learning covers all requirements:

1. **SDM** provides the storage substrate (fast write, content-addressable read)
2. **VSA** provides the representational language (variable binding, structured queries)
3. **STDP** provides temporal/causal learning (A predicts B)
4. **CLS dual-system** provides the architecture (fast episodic + slow semantic)

---

## 5. Comparison Table

| Feature | Hopfield | SDM | Modern Hopfield | VSA/HRR | VSA+SDM |
|---------|---------|-----|----------------|---------|---------|
| **Capacity (50K nodes)** | ~7K patterns | ~50K addr, ~100K writes | Exponential | O(N/log N) per bundle | Combines both |
| **Variable binding** | No | No (but address=key) | No | YES -- first class | YES |
| **Temporal sequences** | No | Via chaining | No | Via permutation | YES |
| **Arbitrary queries** | Partial cue only | Address lookup | Partial cue | YES -- unbinding | YES |
| **Continual learning** | Degrades | YES -- additive | Append to matrix | YES -- additive | YES |
| **No backprop** | YES | YES | Debatable | YES | YES |
| **GPU-friendly** | Matrix ops | Sparse access | Matrix ops | Element-wise/FFT | Mixed |
| **Spiking-compatible** | YES (attractor) | Needs adaptation | NO | YES (BSC variant) | YES |
| **Prediction** | Attractor completion | Heteroassociation | One-step retrieval | Needs STDP layer | YES |
| **Causality** | No | Directed pairs | No | Directed bindings | YES (via STDP) |
| **Planning** | No | Chain search | No | Sequence manipulation | Graph search |
| **Biological basis** | Strong | Strong (hippocampal) | Weak | Moderate (fly brain) | Strong (CLS) |

---

## 6. Recommendations for SNKS

### 6.1 Primary Recommendation: VSA + SDM Hybrid

**Why:** This combination covers all five requirements (prediction, causality, planning, queries, continual learning) while remaining compatible with spiking dynamics and local learning rules.

**Concrete architecture:**

```
Layer 1: Sensory SDR (existing)
    |
    v
Layer 2: VSA Encoding
    - Each observation becomes a structured VSA vector
    - e.g., obs = bind(agent_pos, p3_2) + bind(has_key, yes) + bind(door, locked)
    - Use Binary Spatter Code (BSC) for maximum SDR compatibility
    - Dimensionality: 2048-4096 bits (sparse binary)
    |
    v
Layer 3: SDM Storage (Episodic Memory / Hippocampus)
    - 50K hard locations, 4096-bit addresses and contents
    - Write: store (current_state_VSA -> next_state_VSA) pairs
    - Read: given current state, retrieve predicted next state
    - One-shot, no training needed
    |
    v
Layer 4: STDP Transition Model (Semantic Memory / Neocortex)
    - Oscillator network learns statistical regularities via STDP
    - Slow consolidation from episodic memory (replay)
    - Encodes P(next | current, action)
    |
    v
Layer 5: Query Interface
    - Prediction: read SDM with (state + action) address
    - Causality: trace STDP weight chains backward
    - Planning: iterative SDM reads (state -> next -> next -> ...)
    - Fact queries: VSA unbinding on retrieved patterns
```

### 6.2 Implementation Priority

**Phase 1 (integrate with current Stage 44):**
- Implement BSC-VSA encoding for MiniGrid observations (structured SDR)
- Test that VSA operations (bind, bundle, unbind) work with existing kWTA/SDR pipeline
- Estimated effort: 1-2 stages

**Phase 2 (add SDM):**
- Implement SDM with 50K hard locations
- Store (state, action) -> next_state transitions
- Test prediction accuracy on DoorKey-5x5 trajectories
- Estimated effort: 2-3 stages

**Phase 3 (add query interface):**
- Implement VSA unbinding for fact queries
- Implement forward chaining for planning
- Test: "what color is the key?", "how do I reach the goal?"
- Estimated effort: 2-3 stages

### 6.3 Dimensionality Budget for 50K Nodes

```
Sensory encoding:     10K nodes (SDR from visual encoder)
VSA working memory:    4K nodes (current structured state)
SDM hard locations:   30K nodes (episodic memory, 30K locations x 4K bit content)
STDP semantic layer:   5K nodes (slow-learned regularities)
Control/routing:       1K nodes (query routing, action selection)
Total:                50K nodes
```

This fits within the existing 50K node budget on the AMD 96GB GPU.

### 6.4 GPU Implementation Notes

- BSC-VSA operations (XOR, majority vote) are trivially parallel -- single CUDA/ROCm kernel
- SDM access pattern is sparse but regular: find all hard locations within Hamming radius, then sum. Can be implemented as a batched distance computation (50K x 4K binary matrix) followed by threshold and sum. This is ~25MB of memory and a single matrix op.
- STDP weight updates are already implemented in SNKS
- Total additional GPU memory: ~100-200MB for SDM storage at 50K x 4K

### 6.5 What NOT to Use

- **Modern Hopfield / Dense Associative Memory:** Elegant but requires softmax over all memories, no spiking implementation, effectively requires gradient-like computation.
- **Kanerva Machine:** Requires backprop for the variational part.
- **Transformers/attention-based memory:** Incompatible with SNKS philosophy.
- **DND / Neural Episodic Control:** Requires learned embeddings via backprop.

---

## 7. Key Papers to Read

### Must-Read (directly applicable):

1. **Kanerva, P. (1988).** *Sparse Distributed Memory.* MIT Press. -- The foundational SDM reference.
2. **Plate, T. (1995).** "Holographic Reduced Representations." *IEEE Trans. Neural Networks.* -- Foundational VSA paper.
3. **Kanerva, P. (2009).** "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation.* -- Best overview of VSA/HDC.
4. **Frady et al. (2018).** "A Theory of Sequence Indexing and Working Memory in Recurrent Neural Networks." *Neural Computation.* -- VSA for temporal sequences.
5. **Kleyko et al. (2022).** "Vector Symbolic Architectures as a Computing Framework for Emerging Hardware." *Proc. IEEE.* -- Comprehensive survey of VSA, 100+ pages, covers all variants.
6. **McClelland, McNaughton, O'Reilly (1995).** "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex." *Psychological Review.* -- CLS theory.

### Highly Relevant:

7. **Ramsauer et al. (2021).** "Hopfield Networks is All You Need." *ICLR.* -- Modern Hopfield, understand the theory even if we don't use softmax.
8. **Kleyko et al. (2023).** "A Survey on Hyperdimensional Computing." *ACM Computing Surveys.* -- Updated survey with hardware implementations.
9. **Neubert et al. (2019).** "An Introduction to Hyperdimensional Computing for Robotics." *KI - Kunstliche Intelligenz.* -- Practical VSA for embodied agents.
10. **Renner et al. (2022).** "Neuromorphic Visual Scene Understanding with Resonator Networks." -- VSA factorization via oscillator dynamics (directly relevant to FHN).
11. **Frady, Kleyko, Sommer (2020).** "Resonator Networks." *Neural Computation.* -- Factorization of VSA representations using coupled oscillators. CRITICAL for SNKS integration.

### Background:

12. **Hopfield, J.J. (1982).** "Neural networks and physical systems with emergent collective computational abilities." *PNAS.*
13. **Gayler, R.V. (2003).** "Vector Symbolic Architectures Answer Jackendoff's Challenges for Cognitive Neuroscience." *ICCS/ASCS Joint Conference.*
14. **Howard & Kahana (2002).** "A Distributed Representation of Temporal Context." *JMLR.* -- Temporal context model.

---

## 8. Concrete Next Steps

1. **Immediate experiment:** Implement BSC-VSA encoding for MiniGrid observations. Take the existing SDR output from the visual encoder, structure it as a VSA vector with role-filler bindings for (agent_position, key_status, door_status, goal_position). Verify that unbinding recovers correct fillers. This is a pure CPU test, no GPU needed.

2. **SDM prototype:** Implement a minimal SDM (10K hard locations, 2048-bit address/content) and test: write 1000 (state, action) -> next_state transitions from DoorKey episodes, then measure prediction accuracy. If retrieval accuracy > 80% for seen states and > 50% for interpolated states, proceed to GPU scale.

3. **Resonator Networks investigation:** Paper #11 above (Frady et al. 2020) shows how to factor VSA representations using coupled oscillators -- this maps almost directly to the FHN oscillator network. This could be the bridge between VSA and SNKS dynamics.

4. **CLS integration plan:** Design the dual-system architecture (fast SDM episodic + slow STDP semantic) and the replay/consolidation mechanism. This maps to Phase 3 of the AGI roadmap ("sleep/consolidation").

---

## 9. Summary

The research strongly points to **Vector Symbolic Architectures (VSA) combined with Sparse Distributed Memory (SDM)** as the optimal foundation for the SNKS world model. This combination:

- Requires NO backpropagation (all operations are local)
- Supports variable binding, sequential association, and arbitrary queries natively
- Is SDR-compatible (especially the Binary Spatter Code variant)
- Scales to 50K nodes on a single GPU
- Has deep connections to biological memory (CLS theory, hippocampal indexing)
- Has a natural bridge to oscillator dynamics via Resonator Networks

The critical insight from this research is that **Resonator Networks** (Frady et al. 2020) provide a direct mathematical connection between VSA factorization and coupled oscillator dynamics. This is the most promising path to implementing VSA operations within the existing FHN oscillator framework, rather than adding VSA as a separate layer.

The recommended implementation order is: (1) VSA encoding of observations, (2) SDM for episodic storage, (3) resonator network integration with FHN oscillators, (4) CLS dual-system architecture with replay/consolidation.
