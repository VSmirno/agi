# Stage 76: Continuous Model Learning — Design

**Date:** 2026-04-09
**Status:** Draft (post-Stage 75)
**Depends on:** Stage 75 perception (82% tile_acc) + homeostatic bugs fixed

## Problem

Stage 75 hit an architectural ceiling at ~180 survival steps (target ≥200)
despite 11 code-level fixes. Root cause diagnosis (systematic debugging):

**Current phase6 loop:**
```
1. Perceive (82% accurate tile field)
2. Observe body rates (HomeostaticTracker)
3. select_goal → most urgent drive
4. ConceptStore.plan(goal, inventory) → linear plan
5. Execute step (navigate + verify inventory delta)
6. Repeat
```

**What's missing:**

The agent COMMITS to a multi-step plan and executes it linearly. It cannot
answer "given my current state, will I survive executing this plan?" Every
symptom encountered in debugging (placed table invisible, wood consumed
mid-plan, zombie damage accumulating) is a manifestation of this single
gap: no forward simulation.

Symbolic STRIPS-style planning is fundamentally brittle in stochastic,
adversarial environments. Either:
- Plan too short to be useful (explore-react-die)
- Plan too long to adapt (commit to kill_zombie, die during execution)

## Ideology Constraints

Per `docs/IDEOLOGY.md` (Stage 72/73/74):

MUST:
- **Continuous learning, not batch**. Model updates online from experience.
- **Emergent from world model**. No hardcoded reflexes / magic numbers.
- **Top-down**. Goal = homeostasis, strategy derives from drives + model.
- **No reward shaping**. No RL traps (policy gradients, advantage estimates).
- **Forward simulation**. World model primary, policy secondary.

CAN USE:
- Model-based planning (rollout through ConceptStore + tracker)
- State-value estimation via memory-based approach (SDM, VSA, tabular)
- Tracker-style EMA for slow rate learning
- One-shot confidence updates for causal rules
- Surprise-driven rule discovery

## Proposed Architecture

### Component 1: Transition Model

Learned predictor of next state given (state, action):
```
transition(state, action) → (next_state, confidence)
```

State = (inventory snapshot, tracker rates, visible threats)
Action = from ACTION_NAMES

Initially seeded from textbook rules (deterministic transitions for
`do tree → wood+1`, etc.). Updated continuously from observed
(state, action, next_state) triples.

Open questions:
- Should transition model learn visual features too, or only symbolic state?
- How to represent state — flat vector, SDR, VSA bundle?
- Confidence decay for stale predictions?

### Component 2: Value Estimation

Expected survival steps from a given state. Used to score simulated rollouts.

Candidates:
- **Tabular**: state_hash → (value, visits). Simple but sparse.
- **SDM**: associative memory, recall similar states.
- **VSA bundle**: compositional state representations, graceful degradation.
- **Small MLP**: dense features, but violates "no gradient" principle? (Discuss)

Initial value from body rules: `value = min(inv[health]/rate_health, inv[food]/rate_food, ...)`.
Updated from observed episode outcomes.

### Component 3: Planner (Model-Based)

Replaces current `select_goal` + `ConceptStore.plan` + linear execution.

```python
def decide_action(current_state, world_model, value_fn, horizon=20):
    best_action = None
    best_score = -inf
    for action in available_actions:
        simulated_state = world_model.rollout(current_state, action, horizon)
        score = value_fn(simulated_state)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action
```

Key properties:
- **No plan commitment** — agent re-decides each step
- **Forward simulation** — considers consequences before acting
- **Emergent strategy** — drives + model shape behavior, nothing hardcoded

Horizon tuning: short horizon = reactive, long horizon = strategic.
Should probably be adaptive (longer when safe, shorter under threat).

### Component 4: Continuous Learning Loop

After each action:
1. Observe (state, action, next_state, reward_proxy=body_delta)
2. Update transition_model(state, action) → next_state
3. Update value_fn(state) via bootstrap: V(s) ← V(s) + α*(r + V(s') - V(s))
4. Update tracker rates (already exists)
5. Surprise check: if prediction error > threshold, mark as "anomaly to learn from"

## Alternatives Considered

### A. Keep phase6 loop, add reflexes
Rejected. Hardcoded reflexes violate ideology (documented in
`feedback_no_hardcoded_reflexes.md`). Tried multiple times in Stage 75
debugging, each reverted.

### B. Classic MCTS
Rejected as too heavy. MCTS requires many simulations per decision
(~100+). Slow. Also value backpropagation is bootstrapped from rollout
returns, which requires reward — we want to use body homeostasis as
implicit reward signal without explicit reward shaping.

### C. DreamerV3-style learned world model
Interesting but requires:
- Huge model (500K+ params)
- Batch training with replay buffer
- Not continuous
- Violates "no batch training" principle

### D. Model-based MCTS-light (THIS PROPOSAL)
Middle ground. Forward simulation through a lightweight learned model,
value estimation via memory-based recall, no gradient-based policy.
Continuous updates from experience.

## Success Criteria

### Must pass
- Survival ≥200 avg with enemies (Stage 75 gate that failed)
- Tile perception maintained ≥80% (no regression)
- Wood collection maintained 60%+ (no regression)

### Should demonstrate
- Plan switching: agent changes goal mid-episode when conditions change
- Learned rates: tracker conditional_rates evolve from experience
- Transition model predictions match observations (prediction error falls over time)
- No hardcoded thresholds / magic numbers in policy

### Nice to have
- Survival ≥300 (strong result, beats naive patches)
- Agent crafts sword in >50% of episodes
- Agent kills zombie via sword in some episodes

## Risks

1. **Complexity explosion**. Model-based planning is richer than symbolic
   planning. Need careful scope control.
2. **Value estimation bootstrapping slow**. Initial values may be poor,
   agent flails for many episodes before learning.
3. **Memory requirements**. State-value table / SDM needs to fit in RAM.
4. **Computational cost**. Forward simulation at each step adds latency.
   Must stay fast enough for real-time play.
5. **Hidden hardcoding**. Easy to smuggle reflexes into "value function".
   Need strict review.

## Non-Goals

- Beating Crafter benchmark
- Full DreamerV3 parity
- Multi-agent
- Transfer to other environments (yet)

## Open Questions

1. **State representation**: symbolic (inventory + rates) or hybrid (symbolic
   + visual latent)?
2. **Horizon**: fixed or adaptive?
3. **Exploration vs exploitation**: current curiosity drive enough, or
   need epsilon-greedy?
4. **Tracker rates — current EMA appropriate?**: maybe need per-concept
   learning rate.
5. **Where does "kill zombie" chain come from?**: should agent invent
   this from transition model, or keep ConceptStore backward chain
   as initial plan sketch?

## Next Steps

1. Review with user, refine design
2. Brainstorm specific value function + memory representation
3. Formal spec via brainstorming skill → writing-plans skill
4. Implementation + test
5. Evaluation on exp135 segmenter checkpoint
