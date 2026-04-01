# Visual Encoder Research for DAF Oscillator System (SNKS)

**Date:** 2026-04-01
**Context:** Replace Gabor-based encoder in PureDafAgent for MiniGrid DoorKey-5x5
**Current bottleneck:** Agent at 12% success rate; Gabor+Pool+kWTA SDR cannot distinguish key/door/goal

---

## 1. Current Encoder Analysis

The existing pipeline:
```
obs (64x64 grayscale) -> GaborBank (128 filters, 19x19 kernels)
    -> AdaptiveAvgPool2d(4, 8) -> flatten (4096) -> kWTA (k=164) -> SDR
    -> modular hash mapping to 50K FHN oscillator nodes as external currents
```

**Why it fails for object discrimination:**

1. **Gabor filters detect edges/textures, not objects.** In MiniGrid, objects are 4-6 pixel colored squares. Gabor filters at scales sigma=1..4 are tuned for natural image statistics, not discrete pixel grids.

2. **Grayscale conversion destroys color identity.** MiniGrid uses color to distinguish key (yellow), door (yellow/green), goal (green). Converting to grayscale collapses these.

3. **AdaptiveAvgPool(4,8) destroys spatial layout.** The 4x8 pooling grid is too coarse to localize objects in a 5x5 grid world (each cell is ~12 pixels in a 64x64 render).

4. **SDR sparsity (4%) is fixed, not object-aligned.** The 164 active bits represent edge energy, not "there is a key at position (2,3)".

5. **Modular hash (node_i * PRIME % sdr_size) scatters spatial info.** Adjacent oscillator nodes receive unrelated SDR bits.

---

## 2. Approach-by-Approach Analysis

### 2.1 Object-Centric Models: Slot Attention, SAVi, SLATE

**Slot Attention (Locatello et al., 2020; NeurIPS)**

- **Mechanism:** Iterative attention over CNN features; N "slots" compete to bind to different objects via softmax attention. Each slot is a learned vector (64-128 dim) representing one object.
- **Output format:** N_slots x D_slot (e.g., 7 slots x 64 dim = 448 total floats). Each slot can be decoded to object identity + position.
- **Dimensionality:** Very compact. 7 slots x 64 = 448 dims, easily maps to a few hundred oscillator nodes.
- **Spatial/object info:** Excellent. Each slot explicitly binds to one object. Position is encoded in the slot vector (decoder can reconstruct location).
- **Computational cost:** Small CNN backbone + 3 iterations of cross-attention. For 64x64 input: ~2-5ms on GPU. Well within 100ms budget.
- **Grid world applicability:** Tested on CLEVR (synthetic objects), multi-dSprites, and similar. Grid worlds are *simpler* than these benchmarks. Multiple papers have applied Slot Attention to Atari and grid RL environments (e.g., SMORL, SLATE-based world models).
- **SDR mapping:** Each slot can be quantized to a sparse code. Natural "what + where" factorization per slot.

**SAVi (Kipf et al., 2022; ICLR)**

- Extension of Slot Attention for video. Uses temporal conditioning of slots across frames. Relevant for sequential RL but adds complexity.
- **Verdict:** Overkill for single-frame MiniGrid; Slot Attention suffices.

**SLATE (Singh et al., 2022; ICLR)**

- Combines Slot Attention with discrete VAE tokens (dVAE). Slots attend over discrete tokens rather than CNN features.
- **Output:** N_slots x D_slot, similar to Slot Attention.
- **Advantage:** Discrete tokens provide a natural bridge to SDR.
- **Disadvantage:** More complex training (two-stage: dVAE then transformer).

**Assessment for SNKS:**

| Criterion | Score |
|-----------|-------|
| Output dimensionality | Excellent (448 dims for 7 objects) |
| Computational cost | Excellent (<5ms) |
| Object identity preservation | Excellent (slot = object) |
| Position preservation | Good (learned, not explicit) |
| SDR compatibility | Good (needs quantization layer) |
| Ease of integration | Medium (needs training, but small model) |

---

### 2.2 Self-Supervised ViT Features: DINOv2, MAE

**DINOv2 (Oquab et al., 2024; TMLR)**

- **Mechanism:** Self-supervised Vision Transformer pre-trained on 142M images. Produces patch-level features with strong semantic content.
- **Output format:** For ViT-S/14 on 64x64 input: (64/14)^2 = ~16 patch tokens x 384 dims = 6144 floats. ViT-B: 768 dims per patch.
- **Dimensionality:** 6144 is manageable but not compact. CLS token alone is 384 dims (very compact but loses spatial info).
- **Computational cost:** ViT-S forward pass: ~10-20ms on modern GPU. ViT-B: ~30-50ms. Within budget for AMD GPU 96GB.
- **Grid world applicability:** DINOv2 was trained on natural images. Grid world renders (solid-color squares on black) are severely out-of-distribution. Patch features may not be meaningful for 4-pixel objects.
- **Frozen features:** Can be used as drop-in replacement. No training needed. But OOD concern is serious.
- **SDR mapping:** Dense features need thresholding or VQ to produce sparse input.

**MAE (He et al., 2022; CVPR)**

- Masked autoencoder; features are less semantic than DINO, more oriented toward reconstruction. Less relevant for object discrimination.

**Assessment for SNKS:**

| Criterion | Score |
|-----------|-------|
| Output dimensionality | Medium (384-6144) |
| Computational cost | Good (10-50ms) |
| Object identity preservation | Poor for grid worlds (OOD) |
| Position preservation | Good (patch tokens retain position) |
| SDR compatibility | Poor (dense, needs quantization) |
| Ease of integration | Excellent (frozen, no training) |

**Critical issue:** DINOv2 features on MiniGrid renders are likely garbage. The visual domain is too different from pre-training data. Fine-tuning defeats the "frozen" advantage. **Not recommended as primary approach.**

---

### 2.3 Contrastive Learning: SimCLR, BYOL

**SimCLR (Chen et al., 2020; ICML)**

- **Mechanism:** Contrastive learning on augmented image pairs. Learns representations where similar observations are close, different ones are far.
- **Output:** Single vector per image. Typically 128-512 dims after projection head.
- **Training:** Needs a dataset of MiniGrid observations. Can collect from random exploration.
- **Object preservation:** Global representation; does not naturally factor out objects. An image with key at (1,1) and key at (3,3) would have different representations.
- **Spatial preservation:** No explicit spatial structure.

**BYOL (Grill et al., 2020; NeurIPS)**

- Like SimCLR but without negative pairs. More stable training.
- Same limitations regarding object factorization.

**SPR / CURL / DrQ (RL-specific contrastive)**

- **CURL (Laskin et al., 2020):** Contrastive learning for RL observations. Learns compact features (50-100 dim) online during RL training.
- **SPR (Schwarzer et al., 2021):** Self-predictive representations. Learns features that predict future observations in latent space.
- These are more relevant as they are designed for RL, but still produce holistic (non-object-centric) features.

**Assessment for SNKS:**

| Criterion | Score |
|-----------|-------|
| Output dimensionality | Excellent (128-512 dims) |
| Computational cost | Excellent (<5ms inference) |
| Object identity preservation | Poor (holistic, not factorized) |
| Position preservation | Poor (no spatial structure) |
| SDR compatibility | Medium (can threshold, but not naturally sparse) |
| Ease of integration | Medium (needs training on MiniGrid data) |

**Verdict:** Contrastive methods produce good *global* features but cannot tell you "there is a key at (2,3)". This is the wrong inductive bias for SNKS where we need per-object binding to oscillator groups.

---

### 2.4 Simple CNN Baselines: Trained on MiniGrid

**Small Conv Net (e.g., NatureCNN from Stable-Baselines3)**

- **Architecture:** 3 layers: Conv(3,32,8,4) -> Conv(32,64,4,2) -> Conv(64,64,3,1) -> flatten -> Linear(512)
- **Output:** 512-dim dense vector. Or keep spatial feature maps: e.g., (64, 4, 4) = 1024 dims with spatial structure.
- **Training options:**
  - Supervised: Classify objects/positions (needs labels)
  - Autoencoder: Reconstruct observations
  - RL end-to-end: Train as part of PPO/DQN (but we need bio-plausible learning)
- **Object discrimination:** Trained CNNs can easily distinguish key/door/goal in MiniGrid. This is a trivial task for even 3-layer CNNs with supervised training.
- **Computational cost:** <1ms forward pass. Negligible.

**Key insight for MiniGrid specifically:**

MiniGrid provides a structured observation: `env.gen_obs()` returns a 7x7x3 tensor (partial view) where each cell has (object_type, color, state). The image rendering is just a visualization. **We may not even need pixel-level encoding** -- we could encode the structured observation directly.

If we must use pixel observations (for generality), a tiny CNN with 3 input channels (RGB, not grayscale!) trivially solves object discrimination.

**Assessment for SNKS:**

| Criterion | Score |
|-----------|-------|
| Output dimensionality | Excellent (512-1024) |
| Computational cost | Excellent (<1ms) |
| Object identity preservation | Good (with proper training) |
| Position preservation | Good (if using spatial feature maps) |
| SDR compatibility | Medium (needs sparsification) |
| Ease of integration | Easy (replace GaborBank with Conv layers) |

**Issue:** Training with backprop violates SNKS bio-plausibility principles. But a frozen pre-trained CNN as a "retinal preprocessing" layer is architecturally similar to frozen Gabor filters -- just more effective.

---

### 2.5 Neuro-Inspired Modern: Predictive Coding, Capsule Networks

**Predictive Coding Networks (Rao & Ballard 1999; Millidge et al. 2021)**

- **Mechanism:** Hierarchical network where each layer predicts the layer below. Learning minimizes prediction error (top-down vs bottom-up).
- **Output:** Multi-scale representations. Each level provides features at different abstraction.
- **Bio-plausibility:** Highest among all approaches. Uses only local learning rules (prediction error minimization). Natural fit for SNKS philosophy.
- **Implementation:** `predictive-coding-networks` library (PyTorch). Also: Whittington & Bogacz (2017) framework.
- **Object preservation:** Moderate. PC networks learn to predict, not necessarily to separate objects. But prediction error signals are informative.
- **Computational cost:** Iterative inference (multiple passes per image). 5-20 iterations typical. For small images: 10-30ms.

**Key paper:** Millidge, Seth, Buckley (2022) -- "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs." Shows PC networks can match backprop performance with local rules.

**Capsule Networks (Sabour, Hinton & Fross 2017; Hinton et al. 2018)**

- **Mechanism:** Groups of neurons ("capsules") represent entity properties. Routing-by-agreement determines part-whole relationships.
- **Output:** N_capsules x D_capsule (e.g., 32 capsules x 16 dims = 512). Each capsule's activation probability indicates object presence, vector encodes pose/properties.
- **Bio-plausibility:** Moderate. Capsules are loosely inspired by cortical columns.
- **Object preservation:** Good in theory (capsules bind to entities), but in practice capsules struggle beyond MNIST/simple datasets.
- **Computational cost:** Dynamic routing adds overhead. ~5-10ms for small images.
- **Current status:** Capsule research has largely stalled. Most practitioners prefer Slot Attention for object-centric learning.

**Dendritic Predictive Coding (latest work, 2024-2025)**

- Combines predictive coding with dendritic computation. Apical dendrites carry top-down predictions, basal dendrites carry bottom-up input.
- Very relevant to SNKS architecture (oscillators could implement dendritic compartments).
- Still early-stage; no off-the-shelf implementations for vision encoding.

**Assessment for SNKS:**

| Criterion | PC Networks | Capsule Nets |
|-----------|-------------|-------------|
| Output dimensionality | Flexible (matches architecture) | Good (512) |
| Computational cost | Medium (10-30ms) | Good (5-10ms) |
| Object identity | Medium | Medium-Good |
| Position preservation | Good (hierarchical) | Good (pose vectors) |
| SDR compatibility | Good (PE signals are sparse) | Medium |
| Bio-plausibility | Excellent | Medium |
| Ease of integration | Medium | Medium |

---

### 2.6 Vector Quantized Approaches: VQ-VAE, Discrete Tokenization

**VQ-VAE (van den Oord et al., 2017; NeurIPS)**

- **Mechanism:** Encoder produces continuous features, which are quantized to nearest codebook vectors. Decoder reconstructs from discrete codes.
- **Output:** Grid of codebook indices. For 64x64 input with 4x downsampling: 16x16 = 256 discrete tokens, each from a codebook of K entries (K=128-512 typical).
- **Dimensionality:** 256 tokens, each a one-hot over K=256 codebook. As SDR: 256 active bits out of 256*256=65536 total. **This is a natural SDR!**
- **Computational cost:** <5ms forward pass. Codebook lookup is O(1).
- **Object preservation:** VQ-VAE learns to tokenize the image. In grid worlds, different object types typically map to different codebook entries. Position is preserved by the spatial grid of tokens.
- **SDR mapping:** **Best natural fit.** Each codebook index is a discrete symbol. The spatial grid of symbols maps directly to oscillator populations.

**VQ-VAE-2 (Razavi et al., 2019)**

- Hierarchical VQ-VAE with multiple scales. Overkill for 64x64 but provides multi-scale discrete codes.

**dVAE (Ramesh et al., 2021 / DALL-E)**

- Relaxed VQ-VAE using Gumbel-Softmax. Produces soft discrete tokens during training, hard tokens at inference.
- Used by SLATE (Section 2.1) as the tokenizer before Slot Attention.

**FSQ -- Finite Scalar Quantization (Mentzer et al., 2024; ICLR)**

- Replaces VQ codebook with simple scalar quantization per channel. More stable training, no codebook collapse.
- Output: grid of integer vectors. Each position has a small integer tuple, e.g., (3, 1, 4, 2) from FSQ with L=[5,3,5,3].
- **Effective codebook size** = product of levels. L=[8,5,5,5] gives 1000 codewords.

**Assessment for SNKS:**

| Criterion | Score |
|-----------|-------|
| Output dimensionality | Excellent (256-1024 discrete tokens) |
| Computational cost | Excellent (<5ms) |
| Object identity preservation | Good (objects map to distinct codes) |
| Position preservation | Excellent (spatial grid of tokens) |
| SDR compatibility | **Excellent** (discrete tokens = natural SDR) |
| Ease of integration | Good (needs training, but well-understood) |

**This is the most natural bridge to SDR/oscillator input.**

---

## 3. Hybrid Approach: Slot Attention + VQ (Recommended)

### 3.1 Rationale

No single approach is perfect. The ideal encoder for SNKS combines:

1. **Object factorization** (from Slot Attention) -- needed to solve DoorKey
2. **Discrete/sparse output** (from VQ) -- needed for oscillator input
3. **Spatial preservation** (from spatial feature maps) -- needed for navigation
4. **Bio-plausible interface** -- SDR-like output for STDP learning

### 3.2 Proposed Architecture

```
RGB obs (64x64x3)
    |
    v
Small CNN backbone (3 conv layers, ~5K params)
    -> feature map (64, 8, 8)
    |
    +---> [Path A: Object slots]
    |     Slot Attention (N=7 slots, D=64)
    |     -> 7 slots x 64 = 448 dims
    |     -> Binary: top-k per slot -> object SDR (448 bits, ~10% sparse)
    |
    +---> [Path B: Spatial tokens]
          VQ layer (codebook K=64, grid 8x8)
          -> 64 discrete tokens (8x8 spatial grid)
          -> One-hot expand: 64 tokens x 64 codes = 4096 bit SDR
          -> kWTA: keep 64 active = 1.6% sparsity
    |
    v
Concatenate: object_SDR (448) + spatial_SDR (4096) = 4544 bit SDR
    |
    v
Map to 50K oscillator nodes via structured (not random) hash:
  - Nodes 0-10K: spatial zone (maps from spatial_SDR)
  - Nodes 10K-20K: object zone (maps from object_SDR)
  - Nodes 20K-50K: association zone (receives from both via STDP)
```

### 3.3 Implementation Plan

**Phase 1 (Quick win, 1-2 days): Fix color + simple CNN**

The single highest-impact change: stop converting to grayscale. Use RGB input with a 3-layer CNN pre-trained on MiniGrid observations (autoencoder or random crop augmentation).

```python
class MiniGridCNNEncoder(nn.Module):
    """Minimal CNN encoder for MiniGrid RGB observations."""
    def __init__(self, sdr_size=4096, k=164):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 64->32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # 32->16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 8))  # -> (64, 4, 8)
        self.k = k  # number of active bits
        # Output: 64*4*8 = 2048, pad to sdr_size
        self.sdr_size = sdr_size

    def encode(self, images):
        x = self.conv(images)
        x = self.pool(x)
        x = x.flatten(1)  # (B, 2048)
        # Pad to sdr_size if needed
        if x.shape[1] < self.sdr_size:
            x = F.pad(x, (0, self.sdr_size - x.shape[1]))
        return kwta(x, self.k)
```

Training: Self-supervised with simple augmentation (random crop, color jitter) on observations collected from random exploration. No labels needed. The CNN just needs to learn features that separate different observations.

**Phase 2 (3-5 days): VQ-VAE tokenizer**

Train a small VQ-VAE on MiniGrid observations:
- Encoder: 3 conv layers -> 8x8 feature map
- VQ: codebook of 64 entries, each 32-dim
- Decoder: 3 deconv layers -> reconstruct 64x64x3

The 8x8 grid of codebook indices becomes the spatial SDR. Each cell in the MiniGrid maps to roughly one codebook token, providing natural object/position encoding.

**Phase 3 (1-2 weeks): Slot Attention integration**

Add Slot Attention on top of the CNN backbone:
- 5-7 slots (matching max objects in DoorKey: agent, key, door, goal, walls)
- Each slot: 64 dims, sparsified via top-k
- Slots feed the "object zone" of the DAF

### 3.4 But Also Consider: Skip Pixel Encoding Entirely

MiniGrid's `env.gen_obs()` returns a **symbolic observation**: a 7x7x3 integer array where:
- Channel 0: object type (0=empty, 1=wall, 2=floor, ..., 5=key, 4=door, 8=goal)
- Channel 1: color (0-5)
- Channel 2: state (0-2, e.g., door open/closed/locked)

This structured observation can be encoded directly:

```python
def symbolic_to_sdr(obs_grid, sdr_size=4096, n_types=11, n_colors=6):
    """Encode MiniGrid symbolic observation to SDR.

    Each cell (i,j) activates bits for:
    - Object type at position (i,j)
    - Color at position (i,j)
    - State at position (i,j)
    """
    sdr = torch.zeros(sdr_size)
    for i in range(7):
        for j in range(7):
            obj_type = obs_grid[i, j, 0]
            color = obs_grid[i, j, 1]
            state = obs_grid[i, j, 2]
            if obj_type > 0:  # not empty
                # Position-dependent hash
                base = (i * 7 + j) * 64
                sdr[base + obj_type] = 1.0
                sdr[base + n_types + color] = 1.0
                sdr[base + n_types + n_colors + state] = 1.0
    return sdr
```

This gives **perfect object and position information** with zero training and zero computational cost. It is not biologically plausible as a visual encoder, but it cleanly separates the "perception problem" from the "learning/planning problem." If the agent still cannot solve DoorKey with perfect perception, the bottleneck is in the DAF/STDP/credit-assignment, not the encoder.

---

## 4. Comparative Summary

| Approach | Object ID | Position | SDR Fit | Cost | Bio-Plaus. | Training | Recommendation |
|----------|-----------|----------|---------|------|------------|----------|---------------|
| Gabor (current) | Poor | Poor | Native | <1ms | High | None | **Replace** |
| Slot Attention | Excellent | Good | Good | 2-5ms | Low | Medium | Phase 3 |
| DINOv2 frozen | Poor (OOD) | Good | Poor | 20ms | None | None | **Skip** |
| SimCLR/BYOL | Medium | Poor | Medium | <5ms | None | Medium | Skip |
| Small CNN | Good | Good | Medium | <1ms | Low | Easy | **Phase 1** |
| Predictive Coding | Medium | Good | Good | 20ms | Excellent | Hard | Future |
| Capsule Nets | Medium | Good | Medium | 5ms | Medium | Hard | Skip |
| VQ-VAE | Good | Excellent | **Best** | <5ms | Low | Medium | **Phase 2** |
| FSQ | Good | Excellent | Excellent | <5ms | Low | Easy | Phase 2 alt |
| Symbolic (direct) | Perfect | Perfect | Good | 0ms | None | None | **Diagnostic** |

---

## 5. Concrete Recommendation for SNKS

### Immediate (this week): Diagnostic + CNN

**Step 1: Symbolic encoder diagnostic.** Implement direct encoding of MiniGrid's structured observation. If DoorKey success jumps above 50%, the problem was 100% the encoder. If it stays low, the bottleneck is in credit assignment / temporal learning / planning -- and a better visual encoder alone will not fix the agent.

**Step 2: RGB CNN encoder.** Replace Gabor with a 3-layer CNN on RGB input. Pre-train as an autoencoder on 10K random MiniGrid observations (30 min training). This is the minimum viable replacement that actually processes visual input.

### Short-term (2 weeks): VQ-VAE tokenizer

Train a VQ-VAE with codebook size 64 on MiniGrid observations. The 8x8 grid of discrete tokens maps naturally to SDR. Each MiniGrid cell becomes 1-2 tokens, and similar objects (keys in different positions) map to the same codebook entries. This provides exactly the "what + where" factorization needed.

### Medium-term (Phase 2 of roadmap): Slot Attention + VQ

When moving beyond MiniGrid to richer visual environments, add Slot Attention for explicit object binding. The slot representations feed a dedicated "object identity zone" in the DAF, while VQ tokens feed a "spatial layout zone."

### What NOT to do

1. **Do not use DINOv2/MAE for grid worlds.** They are trained on natural images and will produce meaningless features on solid-color squares.
2. **Do not use contrastive learning (SimCLR/BYOL) alone.** They produce holistic features that do not factor out objects.
3. **Do not invest in Capsule Networks.** The research line has largely been superseded by Slot Attention.
4. **Do not build a Predictive Coding encoder now.** It is the most bio-plausible but also the hardest to implement correctly. Save for Phase 2+ of the roadmap when bio-plausibility of the encoder itself becomes a priority.

---

## 6. Key Papers and References

### Object-Centric
- Locatello et al. (2020). "Object-Centric Learning with Slot Attention." NeurIPS 2020.
- Kipf et al. (2022). "Conditional Object-Centric Learning from Video." ICLR 2022. (SAVi)
- Singh et al. (2022). "Illiterate DALL-E Learns to Compose." ICLR 2022. (SLATE)
- Lin et al. (2020). "SPACE: Unsupervised Object-Oriented Scene Representation." ICLR 2020.
- Zadaianchuk et al. (2023). "Object-Centric Learning for Real-World Videos by Predicting Temporal Feature Similarities." NeurIPS 2023.

### RL-Specific Visual Encoding
- Laskin et al. (2020). "CURL: Contrastive Unsupervised Representations for Reinforcement Learning." ICML 2020.
- Schwarzer et al. (2021). "Data-Efficient Reinforcement Learning with Self-Predictive Representations." ICML 2021. (SPR)
- Ye et al. (2021). "Mastering Atari Games with Limited Data." NeurIPS 2021. (EfficientZero)
- Hafner et al. (2023). "Mastering Diverse Domains through World Models." (DreamerV3 -- uses CNN encoder + discrete latent similar to VQ)

### VQ / Discrete Representations
- van den Oord et al. (2017). "Neural Discrete Representation Learning." NeurIPS 2017. (VQ-VAE)
- Razavi et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2." NeurIPS 2019.
- Mentzer et al. (2024). "Finite Scalar Quantization: VQ-VAE Made Simple." ICLR 2024. (FSQ)
- Micheli et al. (2023). "Transformers are Sample-Efficient World Learners." ICLR 2023. (IRIS -- VQ tokens for RL world models)

### Neuro-Inspired
- Millidge, Seth, Buckley (2022). "Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs." NeurIPS 2022.
- Rao & Ballard (1999). "Predictive coding in the visual cortex." Nature Neuroscience.
- Sabour, Frosst, Hinton (2017). "Dynamic Routing Between Capsules." NeurIPS 2017.

### Bridging Deep Learning and Spiking Networks
- Dampfhoffer et al. (2022). "Are SNNs Really More Energy-Efficient Than ANNs?" IJCNN 2022.
- Zenke & Vogels (2021). "The Remarkable Robustness of Surrogate Gradient Learning for Instilling Complex Function in Spiking Neural Networks." Neural Computation.
- Kim et al. (2022). "Rate Coding Or Direct Coding: Which One Is Better for Accurate, Robust, and Energy-Efficient Spiking Neural Networks?" ICASSP 2022.
- Diehl & Cook (2015). "Unsupervised learning of digit recognition using STDP." Frontiers in Computational Neuroscience. (STDP + lateral inhibition for MNIST -- directly relevant to SNKS)

### Object-Centric RL / Planning
- Zadaianchuk et al. (2022). "Self-supervised Visual Reinforcement Learning with Object-Centric Representations." ICLR 2022. (SMORL)
- Yoon et al. (2023). "Investigation of the Role of Object-Centric Representations in Sample-Efficient Reinforcement Learning."
- Stanic et al. (2024). "Towards Self-Assembling Artificial Neural Networks through Neural Developmental Programs." (Relevant to SNKS's self-organizing philosophy)

---

## 7. Implementation Priority Checklist

- [ ] **P0:** Implement symbolic observation encoder (bypass pixels for diagnostic)
- [ ] **P0:** Fix RGB input (stop grayscale conversion in ObsAdapter)
- [ ] **P1:** Train 3-layer CNN autoencoder on 10K MiniGrid observations
- [ ] **P1:** Replace GaborBank with CNN backbone, keep kWTA/SDR pipeline
- [ ] **P2:** Implement VQ-VAE tokenizer (codebook K=64, spatial 8x8)
- [ ] **P2:** Map VQ tokens to oscillator nodes (structured, not random hash)
- [ ] **P3:** Add Slot Attention module for object-centric binding
- [ ] **P3:** Create dedicated oscillator zones: spatial / object / association

---

## 8. Expected Impact

**With symbolic encoder (diagnostic):** If DoorKey jumps to >30%, encoder was the primary bottleneck. If stays at ~12%, credit assignment is the bottleneck and encoder improvements alone are insufficient.

**With CNN + VQ encoder:** Expected DoorKey improvement from 12% to 20-35%, because:
- Color information is preserved (key vs door vs goal distinguishable)
- Spatial layout is preserved at cell resolution
- SDR now carries meaningful object/position information for STDP to learn from

**With full Slot Attention + VQ:** Expected 30-50% on DoorKey, because:
- Each object has a dedicated slot, making object-goal associations learnable by STDP
- "Pick up key" and "open door" become distinct attractor states in the DAF

**Note:** Reaching 50%+ on DoorKey likely requires *both* a better encoder AND better temporal credit assignment (eligibility traces / working memory). The encoder alone is necessary but not sufficient.
