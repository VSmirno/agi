# Stage 14: Embodied Agent — Design Spec

**Version:** 1.1
**Date:** 2026-03-25
**Status:** Approved

---

## 1. Goal

Create `EmbodiedAgent` — a thin orchestration layer that wires `CausalAgent` (Stage 6) with `StochasticSimulator` (Stage 11) and the Configurator FSM output (Stage 13) into a full agent loop operating in MiniGrid KeyDoor.

**Key insight:** `Pipeline.perception_cycle()` already runs MetaEmbedder, HACPredictionEngine, IntrinsicCostModule, and Configurator internally (runner.py lines 282–313). `CycleResult` carries their outputs. `EmbodiedAgent` reads these outputs and uses them for action selection — it does NOT re-instantiate or re-call these components.

**Success criteria (all three must hold):**
- **Behavioural:** full stack outperforms Stage 6 baseline (ablation Exp 30)
- **Architectural:** all components run error-free in one step() cycle (Exp 29 smoke test)
- **Quantitative:** `coverage >= 0.40 AND success_rate >= 0.30` in MiniGrid KeyDoor 8×8

---

## 2. Environment

**MiniGrid KeyDoor 8×8** — agent finds key → unlocks door → reaches goal (green tile).

| Phase | Agent state | Expected Configurator mode |
|-------|-------------|---------------------------|
| Episode start | high PE, unfamiliar map | EXPLORE |
| After N episodes | map known, PE drops | CONSOLIDATE |
| Key collected, goal in model | directed navigation | GOAL_SEEKING |

Note: `coverage >= 0.40` in KeyDoor implicitly requires unlocking the door (goal is behind it). This is by design — the gate validates that EXPLORE and GOAL_SEEKING modes are both triggered.

**Parameters:**
- Size: 8×8 (Exp 29–30), 16×16 (Exp 31)
- `max_steps = 200` per episode
- `n_episodes = 100` (Exp 29–30), `500` (Exp 31)
- ObsAdapter: existing `src/snks/env/` (64×64 grayscale tensor)

---

## 3. Architecture

### 3.1 EmbodiedAgent (thin orchestration wrapper)

```
EmbodiedAgent
├── CausalAgent           # owns Pipeline (which owns MetaEmbedder, HACPredictor,
│                         #   IntrinsicCostModule, Configurator inside perception_cycle())
├── StochasticSimulator   # Stage 11: N-sample Monte Carlo planning (action selection only)
└── EmbodiedAgentConfig   # ablation flags + StochasticSimulator params
```

`CausalAgent` is **not modified**. Its `pipeline.configurator.enabled` and
`pipeline.cost_module` are configured at construction time via `PipelineConfig`.

### 3.2 EmbodiedAgentConfig

```python
@dataclass
class EmbodiedAgentConfig:
    causal: CausalAgentConfig          # includes PipelineConfig (configurator.enabled,
                                       #   cost_module.enabled set here for ablation)
    use_stochastic_planner: bool = True
    n_plan_samples: int = 8
    max_plan_depth: int = 5
    goal_cost_value: float = 1.0       # passed to icm.set_goal_cost() when GOAL_SEEKING
```

Ablation is controlled by setting `causal.pipeline.configurator.enabled = False` and/or
`causal.pipeline.cost_module.enabled = False` in the config before constructing the agent.

### 3.3 One-line addition to Pipeline (permitted)

`runner.py` gains one line at the end of `perception_cycle()`:

```python
self.last_cycle_result = result   # cached for EmbodiedAgent access
```

`Pipeline.__init__` initializes `self.last_cycle_result: CycleResult | None = None`.
This is non-breaking and does not change any existing behaviour.

### 3.4 step() cycle

```python
def step(self, obs: np.ndarray) -> int:
    # 1. CausalAgent.step() runs ALL Stage 6-13 components via Pipeline.perception_cycle()
    #    and sets _pre_sks / _last_action internally.
    action = self.causal_agent.step(obs)
    result = self.causal_agent.pipeline.last_cycle_result   # CycleResult (cached above)

    sks = set(result.sks_clusters.keys())
    mode = result.configurator_action.mode if result.configurator_action else "neutral"

    # 2. Goal cost update for next cycle's ICM
    if mode == "goal_seeking":
        self.causal_agent.pipeline.cost_module.set_goal_cost(self.config.goal_cost_value)
    else:
        self.causal_agent.pipeline.cost_module.set_goal_cost(0.0)

    # 3. Action selection by mode
    if mode == "goal_seeking" and self.config.use_stochastic_planner and self._goal_sks:
        plan, _ = self.simulator.find_plan_stochastic(
            sks, self._goal_sks,
            n_actions=self.n_actions,
            n_samples=self.config.n_plan_samples,
            max_depth=self.config.max_plan_depth,
        )
        if plan:
            return plan[0]

    if mode == "explore":
        return random.randint(0, self.n_actions - 1)
    return action   # consolidate / neutral / goal_seeking fallback: CausalAgent default
```

### 3.6 Goal representation

Goal SKS (`_goal_sks: set[int] | None`) is obtained as follows:

1. At episode start: `_goal_sks = None`
2. Each step: if env info contains `"goal_visible": True`, run an extra
   `perception_cycle(goal_obs)` on the observation and store resulting SKS as `_goal_sks`
3. Alternative (simpler): pass the goal tile position from `env.unwrapped.goal_pos` through
   a fixed synthetic observation; store SKS cluster IDs.
4. When `_goal_sks is None`, GOAL_SEEKING falls back to greedy action selection.

The concrete method is determined during implementation based on what `env.info` exposes.

### 3.5 observe_result() contract

`EmbodiedAgent` follows the two-phase `CausalAgent` contract:

```python
# In run loop:
action = agent.step(obs)                         # perception + action selection
obs_next, reward, done, _, info = env.step(action)
agent.observe_result(obs_next)                   # causal model update (takes next obs)
```

`CausalAgent.observe_result(obs: np.ndarray)` runs a second perception cycle on `obs_next`
and updates the causal world model. `EmbodiedAgent.observe_result()` delegates directly to
`causal_agent.observe_result(obs_next)` — reward and done are not passed (not part of the
CausalAgent API).

---

## 4. Experiments

### Exp 29 — Integration test

**File:** `src/snks/experiments/exp29_integration.py`

- Full `EmbodiedAgent`, KeyDoor 8×8, 100 episodes
- **Gate:** `coverage >= 0.40 AND success_rate >= 0.30`
- Also logs: Configurator mode history per episode, mean steps to goal

### Exp 30 — Ablation study

**File:** `src/snks/experiments/exp30_ablation.py`

Four variants, each run 100 episodes on the same KeyDoor 8×8 (fixed seed):

| Variant | cost_module.enabled | configurator.enabled | Description |
|---------|--------------------|--------------------|-------------|
| Full stack | ✅ | ✅ | all Stage 10-13 active |
| No Configurator | ✅ | ❌ | ICM active, FSM disabled |
| No ICM | ❌ | ❌ | both disabled |
| Stage 6 baseline | ❌ | ❌ | epsilon-greedy, no planning |

Note: "No ICM" implies no Configurator (Configurator inputs cost_state from ICM).
A variant with `cost_module=❌, configurator=✅` is degenerate (Configurator always sees
zero cost → never leaves NEUTRAL) and is excluded as uninformative.

- **Score:** `0.5 * coverage + 0.5 * success_rate`
- **Gate:** `full_stack_score > no_configurator_score > baseline_score`

### Exp 31 — Scaling (mini-PC)

**File:** `src/snks/experiments/exp31_scaling.py`

- Full `EmbodiedAgent`, KeyDoor 16×16, 500 episodes
- `num_nodes=50_000`, `device="hip"` (AMD ROCm gfx1151, HSA_OVERRIDE_GFX_VERSION=11.0.0)
- **Gate:** `success_rate >= 0.30 AND steps_per_sec >= 100`
- `steps_per_sec`: wall-clock average over all 500 episodes, measured as
  `(n_episodes * max_steps) / total_elapsed_seconds` (includes stochastic planning overhead)
- Runs remotely: `ssh -p 2244 gem@10.253.0.179`, path `/opt/agi`

---

## 5. File structure

**New files:**
```
src/snks/agent/embodied_agent.py       # EmbodiedAgent + EmbodiedAgentConfig
src/snks/experiments/exp29_integration.py
src/snks/experiments/exp30_ablation.py
src/snks/experiments/exp31_scaling.py
tests/test_embodied_agent.py
```

**Unchanged:**
- `src/snks/agent/agent.py` (CausalAgent)
- `src/snks/pipeline/runner.py` (Pipeline)
- All Stage 10–13 modules

---

## 6. Tests

`tests/test_embodied_agent.py`:
- `test_step_observe_result_cycle` — one full step()+observe_result() cycle completes
- `test_ablation_flags` — each config combination instantiates and runs one step without error
- `test_goal_seeking_calls_simulator` — when mode=goal_seeking and goal_sks set, plan is requested

---

## 7. SPEC.md update (after all experiments pass)

- Add Stage 14 row (✅) to stages table
- Add Exp 29–31 rows to experiments table
- Update version to 0.6.0
