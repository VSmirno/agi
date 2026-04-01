# Stage 39: Curriculum Learning + Adaptive Exploration

**Date:** 2026-04-01
**Status:** DESIGN
**Author:** Autonomous Development

## Problem Statement

Stage 38 Pure DAF Agent achieves 12% success on DoorKey-5x5 and 34% on Empty-8x8.
These are genuine DAF results, but far from useful. Key bottlenecks:

1. **Random exploration (70% epsilon)**: agent wastes most steps doing random actions
2. **No curriculum**: trains directly on target env with no progression
3. **Fixed epsilon**: never reduces even when agent has learned something
4. **No experience replay**: successful trajectories don't reinforce learning

## Goal

Improve Pure DAF Agent success rate through:
- Curriculum: progressive difficulty (Empty → DoorKey)
- Prediction-error curiosity: explore states with high prediction error
- Decaying epsilon: reduce randomness as agent learns
- Trajectory replay: replay successful trajectories for STDP reinforcement

Target: **DoorKey-5x5 ≥ 0.25** (from 0.12), **Empty-8x8 ≥ 0.50** (from 0.34)

## Approach Analysis

### Approach A: Curriculum + Epsilon Decay (RECOMMENDED)
- Simple, compositional improvements
- Curriculum: Empty-5x5 (warmup) → Empty-8x8 → DoorKey-5x5
- Epsilon decay: start 0.7, decay by 0.95 per episode to floor 0.1
- Trade-off: doesn't change exploration quality, just quantity
- **Selected**: simplest to implement, most bang for buck

### Approach B: Curiosity-Driven (prediction error)
- Replace epsilon-greedy with prediction-error based action bias
- Actions that produce high PE get bonus probability
- Trade-off: PE may be noisy with small DAF (2K nodes on CPU)
- Partial adoption: add PE bonus to action selection as soft bias

### Approach C: Trajectory Replay
- Store successful trajectories, replay them offline
- Re-inject observations → run perception cycles → modulate STDP
- Trade-off: expensive (2x compute for same learning)
- Partial adoption: store success stats per env for curriculum scheduling

### Decision: A with elements from B

1. **CurriculumScheduler**: manages env progression based on success rate
2. **EpsilonDecay**: exponential decay with floor
3. **PredictionErrorBonus**: soft bias toward novel actions (from B)
4. Keep same PureDafAgent core — enhancements wrap it

## Architecture

```
CurriculumTrainer
├── CurriculumScheduler
│   ├── env_sequence: [(env_name, gate_threshold), ...]
│   ├── current_env_idx: int
│   └── promote(): advance when gate met for N consecutive episodes
├── EpsilonScheduler
│   ├── initial: 0.7, decay: 0.95, floor: 0.1
│   └── step(): called per episode
├── PredictionErrorExplorer
│   ├── action_pe_history: per-action rolling PE average
│   └── bonus(action): softmax over PE means → bias toward high-PE actions
└── PureDafAgent (unchanged core)
    └── AttractorNavigator (epsilon from scheduler, PE bonus)
```

## Gate Criteria

| Experiment | Metric | Gate | Notes |
|-----------|--------|------|-------|
| exp98a: Curriculum DoorKey-5x5 | success_rate | >= 0.25 | After curriculum training |
| exp98b: Curriculum benefit | curriculum_rate - baseline_rate | > 0.05 | Curriculum > direct training |
| exp98c: Empty-8x8 improved | success_rate | >= 0.50 | After warmup on Empty-5x5 |
| exp98d: Epsilon decay | final_epsilon < initial_epsilon | True | Epsilon actually decays |
| exp98e: PE exploration | unique_states_visited | > random | More coverage than random |

## File Plan

### New files:
- `src/snks/agent/curriculum.py` — CurriculumScheduler, EpsilonScheduler, PredictionErrorExplorer
- `src/snks/experiments/exp98_curriculum.py` — experiments
- `tests/test_curriculum.py` — unit tests
- `demos/stage-39-curriculum.html` — web demo

### Modified files:
- `src/snks/agent/pure_daf_agent.py` — add epsilon_override, PE hooks
- `src/snks/agent/attractor_navigator.py` — accept PE bonus in action selection
- `ROADMAP.md` — Stage 39 entry
- `demos/index.html` — Stage 39 card

## Risks

1. **Curriculum may not help**: if DAF can't distinguish Empty from DoorKey → test on 2 stages minimum
2. **PE noise**: FHN with 2K nodes on CPU may produce noisy PE → use rolling average (window=5)
3. **Overfitting to curriculum order**: mitigate with interleaved episodes
