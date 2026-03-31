# Stage 38: Pure DAF Agent — Return to Paradigm

**Date:** 2026-04-01
**Status:** DESIGN
**Author:** Autonomous Development

## Problem Statement

Starting from Stage 19, SNKS introduced scaffolding components that bypass the DAF pipeline:
- **GridPerception** reads MiniGrid grid state directly → hardcoded SKS IDs (50-58)
- **GridNavigator** uses BFS on full grid → optimal action sequences
- **BlockingAnalyzer** scans entire grid for obstacles
- **CausalWorldModel** uses dict-based counting instead of STDP
- **CuriosityModule** uses count-based novelty instead of prediction error

By Stage 37, the scaffolding IS the agent. DAF (oscillators, STDP, coherence) is running but its output is ignored. The 96-100% success rates prove capabilities but not SNKS paradigm.

## Goal

Replace scaffolding with pure DAF computation:
1. **Perception**: observation → VisualEncoder → DAF oscillators → SKS clusters (no grid reading)
2. **Navigation**: world model planning via MentalSimulator (no BFS)
3. **Causal learning**: reward-modulated STDP strengthens action-outcome associations
4. **Action selection**: attractor dynamics + intrinsic motivation (no hardcoded SKS)

Expected performance: 10-30% success on DoorKey-5x5 (vs 100% scaffolded). This is acceptable — real SNKS results.

## Architecture

```
PureDafAgent (extends EmbodiedAgent)
├── Pipeline (existing)
│   ├── VisualEncoder: image → Gabor → SDR → currents
│   ├── DafEngine: 50K FHN oscillators + STDP + homeostasis
│   ├── SKS Detection: coherence → DBSCAN clusters
│   └── HAC Prediction: embedding similarity
├── DafCausalModel (NEW — replaces dict CausalWorldModel)
│   ├── Action-conditioned STDP: action modulates lr
│   ├── Reward-modulated plasticity: success → potentiate recent
│   ├── State representation: HAC embedding similarity
│   └── Causal query: predict_effect via attractor activation
├── AttractorNavigator (NEW — replaces BFS GridNavigator)
│   ├── Goal encoding: inject goal observation as DAF current
│   ├── Action scoring: run mental simulation per action
│   ├── Winner-take-all: highest predicted reward wins
│   └── Fallback: intrinsic motivation (exploration)
└── EnvAdapter (NEW — environment-agnostic interface)
    ├── Protocol: reset() → obs, step(action) → (obs, reward, done, info)
    ├── MiniGridAdapter: wraps gymnasium MiniGrid envs
    └── No env-specific code in agent
```

## Path A: Return to Paradigm

### A1. DafCausalModel

Replaces `CausalWorldModel` (dict counting) with STDP-based causal memory.

**Key insight**: Instead of storing `(context_hash, action) → effect` in a dict, we:
1. Encode action as motor current into DAF (already done by MotorEncoder)
2. Run DAF step → observe which SKS clusters change
3. Reward-modulated STDP: if reward > 0, strengthen recent spike-timing pairs
4. Query: re-inject context + action, read predicted SKS change from attractor

**Implementation**:
```python
class DafCausalModel:
    """Causal learning through reward-modulated STDP."""

    def __init__(self, engine: DafEngine):
        self.engine = engine
        self._reward_trace: list[float] = []  # eligibility trace
        self._pre_weights: Tensor | None = None  # snapshot for modulation

    def before_action(self, action: int):
        """Snapshot STDP weights before action."""
        self._pre_weights = self.engine.graph.edge_attr[:, 0].clone()

    def after_action(self, reward: float):
        """Modulate STDP weight changes by reward signal."""
        if self._pre_weights is None:
            return
        delta_w = self.engine.graph.edge_attr[:, 0] - self._pre_weights
        # Reward-modulated: amplify STDP changes proportional to reward
        modulated = delta_w * (1.0 + reward * self.reward_scale)
        self.engine.graph.edge_attr[:, 0] = self._pre_weights + modulated

    def predict_effect(self, current_embedding: Tensor, action: int) -> Tensor:
        """Predict next state by running DAF with action input."""
        # Mental simulation: inject current state + motor action
        # Run short DAF step, read out predicted embedding
        ...
```

### A2. AttractorNavigator

Replaces `GridNavigator` (BFS) with attractor-based action selection.

**Key insight**: Instead of computing optimal path via BFS, the agent:
1. Holds goal observation in working memory (DAF external currents)
2. For each possible action, runs mental simulation (short DAF integration)
3. Selects action whose predicted state is most similar to goal (HAC cosine similarity)
4. Falls back to exploration when similarity is low

**Implementation**:
```python
class AttractorNavigator:
    """Goal-directed navigation via attractor dynamics."""

    def select_action(self, current_sks, goal_embedding, n_actions) -> int:
        """Select action by mental simulation."""
        best_action = -1
        best_similarity = -1.0

        for action in range(n_actions):
            predicted = self.mental_simulate(current_sks, action)
            sim = cosine_similarity(predicted, goal_embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_action = action

        if best_similarity < self.min_threshold:
            return self.explore(current_sks, n_actions)
        return best_action
```

### A3. PureDafAgent

Orchestrates everything without grid access.

**Episode loop**:
1. `obs = env.reset()` → image observation only
2. `goal_obs = env.render_goal()` → goal state image (or initial observation for DoorKey)
3. Encode goal → DAF → goal_embedding (HAC)
4. Loop:
   a. Encode current obs → DAF → perception_cycle → current SKS + embedding
   b. AttractorNavigator.select_action(current, goal) → action
   c. DafCausalModel.before_action(action)
   d. obs, reward, done = env.step(action)
   e. DafCausalModel.after_action(reward)
   f. If done: break

## Path B: Environment-Agnostic Interface

### B1. EnvAdapter Protocol

```python
class EnvAdapter(Protocol):
    def reset(self) -> np.ndarray:
        """Reset and return initial observation (H, W, 3) uint8."""
        ...

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action, return (obs, reward, terminated, truncated, info)."""
        ...

    @property
    def n_actions(self) -> int:
        """Number of available discrete actions."""
        ...

    @property
    def name(self) -> str:
        """Environment name for logging."""
        ...
```

### B2. MiniGridAdapter

```python
class MiniGridAdapter:
    """Wraps any MiniGrid env into EnvAdapter."""

    def __init__(self, env_name: str, render_mode: str = "rgb_array"):
        self._env = gymnasium.make(env_name, render_mode=render_mode)

    def reset(self) -> np.ndarray:
        obs, _ = self._env.reset()
        return obs["image"]  # (H, W, 3) partial observation

    def step(self, action: int) -> tuple:
        obs, reward, term, trunc, info = self._env.step(action)
        return obs["image"], reward, term, trunc, info

    @property
    def n_actions(self) -> int:
        return self._env.action_space.n
```

### B3. Remove MiniGrid-specific imports from agent

PureDafAgent must NOT import:
- `GridPerception`, `GridNavigator`, `BlockingAnalyzer`
- `BabyAIExecutor`, `GoalAgent`
- Any `grid.get()` calls
- Any `env.unwrapped` access

## Gate Criteria

| Experiment | Metric | Gate | Notes |
|-----------|--------|------|-------|
| exp97a: DoorKey-5x5 | success_rate | >= 0.10 | Pure DAF, no scaffolding |
| exp97b: DoorKey-5x5 | causal_learning | > 0 | STDP weights change correlate with rewards |
| exp97c: Empty-8x8 | navigation | >= 0.30 | Can navigate to visible goal |
| exp97d: env-agnostic | runs_without_error | True | Same agent, different envs |

## File Plan

### New files:
- `src/snks/agent/pure_daf_agent.py` — PureDafAgent
- `src/snks/agent/daf_causal_model.py` — DafCausalModel (reward-modulated STDP)
- `src/snks/agent/attractor_navigator.py` — AttractorNavigator
- `src/snks/env/adapter.py` — EnvAdapter protocol + MiniGridAdapter
- `src/snks/experiments/exp97_pure_daf.py` — experiments
- `tests/test_pure_daf_agent.py` — unit tests
- `demos/stage-38-pure-daf.html` — web demo

### Modified files:
- `ROADMAP.md` — Stage 38 status update
- `docs/reports/stage-38-report.md` — final report

### NOT modified (preserved):
- All existing scaffolded agents (GoalAgent, etc.) — kept for comparison
- All existing experiments (exp93-96) — kept for regression testing

## Risks and Mitigations

1. **DAF SKS clusters are noisy** → Use HAC embedding similarity instead of cluster ID matching
2. **Mental simulation is slow** → Reduce n_steps per simulation (10 instead of 100)
3. **No reward signal in DoorKey** → Use intrinsic reward (prediction error) as STDP modulator
4. **10% may be too optimistic** → Accept any learning signal (> random baseline ~2%)

## Decision Log

- **Chose Path A first**: Core value is proving DAF works, not abstraction layer
- **Keep existing scaffolding**: Side-by-side comparison proves the point
- **Low gates (10%)**: Honest about expected performance drop
- **Reward-modulated STDP**: Simpler than full three-factor learning rule, sufficient for proof
