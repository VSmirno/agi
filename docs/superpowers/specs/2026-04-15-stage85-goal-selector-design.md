# Stage 85 — Goal Selector Design

**Date:** 2026-04-15
**Status:** Approved
**Ideological debt:** Category 4 (Stimuli) — `total_gain` is Crafter-specific. Stage 85 replaces it with a goal-derived progress signal and curiosity.

---

## Problem

After Stage 84, `score_trajectory` returns:
```python
(base_score, total_gain, -steps)
```

`total_gain` counts all positive inventory deltas. This is Crafter-specific and cannot transfer to other environments. The agent never gathers wood because that drive is disconnected from survival goals.

---

## Design

### 1. Textbook `goals` block

Add to `configs/crafter_textbook.yaml`:

```yaml
goals:
  primary: survive
```

Add to `src/snks/agent/crafter_textbook.py`:

```python
@property
def goals_block(self) -> dict:
    return self._config.get("goals", {})
```

`GoalSelector` reads this key to confirm the primary goal but derives all threat→response mappings from the existing `rules` and `passive` entries — no new per-threat declarations.

Derivation logic:
- `vital: true` in `body.variables` → health=0 means death
- `passive: spatial, entity: zombie, effect: body.health: -0.5` → zombie threatens health
- `rule: do zombie, requires: {wood_sword: 1}` → counter = have wood_sword
- `rule: make wood_sword, requires: {wood: 1}, near: table` → crafting chain

---

### 2. VectorTrajectory — new field and methods

In `src/snks/agent/vector_sim.py`, add to `VectorTrajectory`:

```python
@dataclass
class VectorTrajectory:
    plan: VectorPlan
    states: list[VectorState]
    terminated: bool = False
    terminated_reason: str = ""
    confidences: list[float] = field(default_factory=list)
    # Per-step prediction confidence from model.predict().
    # Populated by simulate_forward. Range [0,1]: 1.0=certain, 0.0=surprised.

    # ... existing final_state, total_inventory_gain ...

    def vital_delta(self, var: str) -> float:
        """Change in body variable from first to last state."""
        if len(self.states) < 2:
            return 0.0
        return self.states[-1].body.get(var, 0.0) - self.states[0].body.get(var, 0.0)

    def inventory_delta(self, item: str) -> float:
        """Change in inventory item count from first to last state."""
        if len(self.states) < 2:
            return 0.0
        return float(
            self.states[-1].inventory.get(item, 0)
            - self.states[0].inventory.get(item, 0)
        )

    def item_gained(self, item: str) -> bool:
        """True if item count went from 0 to >0 during trajectory."""
        if len(self.states) < 2:
            return False
        return (
            self.states[0].inventory.get(item, 0) == 0
            and self.states[-1].inventory.get(item, 0) > 0
        )
```

In `simulate_forward`, `confidence` is already computed at line 174 (`effect_vec, confidence = model.predict(...)`). Add one line to accumulate it into the trajectory. Since `VectorTrajectory` is constructed at return time, accumulate into a local list and pass to constructor:

```python
def simulate_forward(...) -> VectorTrajectory:
    states = [initial_state.copy()]
    state = initial_state.copy()
    confidences: list[float] = []          # NEW

    for step in plan.steps[:horizon]:
        ...
        effect_vec, confidence = model.predict(step.target, step.action)
        confidences.append(confidence)     # NEW — capture before early-continue

        if confidence < 0.2:
            ...
            continue
        ...

    return VectorTrajectory(plan=plan, states=states, confidences=confidences)
    # Also update the early-return (terminated=True) to pass confidences=confidences
```

---

### 3. GoalSelector

New file: `src/snks/agent/goal_selector.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from snks.agent.vector_sim import VectorTrajectory, VectorState
    from snks.agent.crafter_textbook import CrafterTextbook


@dataclass
class Goal:
    id: str
    requirements: dict = field(default_factory=dict)

    def progress(self, trajectory: VectorTrajectory) -> float:
        """How much did this trajectory advance the goal? Returns float ≥ 0."""
        if self.id == "fight_zombie":
            # Proxy: health delta positive → threat reduced.
            # Full scene comparison deferred (VectorState.spatial_map optional).
            return max(0.0, trajectory.vital_delta("health"))
        elif self.id == "fight_skeleton":
            return max(0.0, trajectory.vital_delta("health"))
        elif self.id == "find_cow":
            return max(0.0, trajectory.vital_delta("food"))
        elif self.id == "find_water":
            return max(0.0, trajectory.vital_delta("drink"))
        elif self.id == "sleep":
            return max(0.0, trajectory.vital_delta("energy"))
        elif self.id == "craft_wood_sword":
            return 1.0 if trajectory.item_gained("wood_sword") else 0.0
        elif self.id == "gather_wood":
            return max(0.0, trajectory.inventory_delta("wood"))
        elif self.id == "explore":
            if not trajectory.confidences:
                return 0.0
            return 1.0 - sum(trajectory.confidences) / len(trajectory.confidences)
        return 0.0


@dataclass
class _Threat:
    """Internal: one entry in the priority-ordered threat list."""
    active_fn: callable    # (VectorState) -> bool
    response_fn: callable  # (VectorState) -> Goal


class GoalSelector:
    def __init__(self, textbook: CrafterTextbook):
        self._threats = self._derive_threats(textbook)

    def select(self, state: VectorState) -> Goal:
        """Pure function: current state → active goal. Called every step."""
        for threat in self._threats:
            if threat.active_fn(state):
                return threat.response_fn(state)
        return Goal("explore")

    def _derive_threats(self, textbook: CrafterTextbook) -> list[_Threat]:
        """Build priority-ordered threat list from textbook passive + action rules.

        Priority order:
          1. Physical threats: entity nearby with negative health passive rule
             → response depends on inventory (has sword → fight, else → craft)
          2. Critical vitals: health < 2 → find_cow (recover health via food)
          3. Low vitals: food < 3 → find_cow, drink < 3 → find_water, energy < 3 → sleep
        """
        threats = []

        # 1. Parse passive spatial rules for dangerous entities
        for rule in textbook.rules:
            if rule.get("passive") == "spatial":
                entity = rule.get("entity")
                effect = rule.get("effect", {})
                body_effect = effect.get("body", {})
                if body_effect.get("health", 0) < 0:
                    # This entity damages health → it's a threat
                    def make_entity_threat(ent):
                        # Check if entity appears in spatial_map nearby
                        def active(state, _e=ent):
                            sm = state.spatial_map
                            if sm is None:
                                return False
                            return sm.nearest(_e) is not None and sm.nearest(_e)[1] <= 3

                        def response(state, _e=ent):
                            # Find what item is required to fight this entity
                            for r in textbook.rules:
                                if r.get("action") == "do" and r.get("target") == _e:
                                    req = r.get("requires", {})
                                    weapon = next(iter(req), None)
                                    if weapon and state.inventory.get(weapon, 0) > 0:
                                        return Goal(f"fight_{_e}", {})
                                    elif weapon:
                                        return Goal(f"craft_{weapon}", {"item": weapon})
                            return Goal("explore")

                        return _Threat(active_fn=active, response_fn=response)

                    threats.append(make_entity_threat(entity))

        # 2. Critical health
        threats.append(_Threat(
            active_fn=lambda s: s.body.get("health", 9) < 2,
            response_fn=lambda s: Goal("find_cow"),
        ))

        # 3. Low vitals
        for vital, goal_id in [("food", "find_cow"), ("drink", "find_water"), ("energy", "sleep")]:
            def make_vital_threat(v, g):
                return _Threat(
                    active_fn=lambda s, _v=v: s.body.get(_v, 9) < 3,
                    response_fn=lambda s, _g=g: Goal(_g),
                )
            threats.append(make_vital_threat(vital, goal_id))

        return threats
```

Key property: `select(state)` is a pure function, re-evaluated every step. No push/pop. Zombie appears → zombie response immediately. Zombie threat clears → falls through to next applicable threat or explore.

**Design note — future:** Priority ordering is currently fixed. Future stages should make this learnable: `priority(goal, state) = time_to_death_if_ignored(state)`, derived from world model rollouts (Stage 87+ debt).

---

### 4. CuriosityStimulus

Add to `src/snks/agent/stimuli.py`. Not wired into `StimuliLayer` default — curiosity is handled via `Goal("explore").progress(trajectory)`:

```python
@dataclass
class CuriosityStimulus(Stimulus):
    """Defined for Stage 87 (death-relevant curiosity weighting). Unused in Stage 85."""
    weight: float = 0.1

    def evaluate(self, trajectory: VectorTrajectory) -> float:
        if not trajectory.confidences:
            return 0.0
        avg_surprise = 1.0 - sum(trajectory.confidences) / len(trajectory.confidences)
        return self.weight * avg_surprise
```

---

### 5. `score_trajectory` — new signature

In `src/snks/agent/vector_sim.py`:

```python
def score_trajectory(
    trajectory: VectorTrajectory,
    stimuli: StimuliLayer | None = None,
    goal: Goal | None = None,
) -> tuple:
    """3-tuple: (base_score, goal_prog, -steps).

    base_score: StimuliLayer.evaluate() if provided, else survived (0/1).
    goal_prog:  Goal.progress(trajectory) if goal provided, else 0.
    """
    goal_prog = goal.progress(trajectory) if goal is not None else 0.0
    steps = len(trajectory.states) - 1

    if stimuli is not None:
        base = stimuli.evaluate(trajectory)
    else:
        base = 0 if trajectory.terminated else 1

    return (base, goal_prog, -steps)
```

`total_gain` is removed. `total_inventory_gain()` method stays on `VectorTrajectory` (used in tests).

---

### 6. Integration in `run_vector_mpc_episode`

Changes to `src/snks/agent/vector_mpc_agent.py`:

```python
def run_vector_mpc_episode(
    env,
    segmenter,
    model,
    tracker,
    max_steps: int = 1000,
    stimuli: StimuliLayer | None = None,
    textbook: CrafterTextbook | None = None,   # NEW
    verbose: bool = False,
) -> dict:
    ...
    # Init once per episode:
    from snks.agent.goal_selector import Goal, GoalSelector
    goal_selector = GoalSelector(textbook) if textbook is not None else None

    # Per step, before plan scoring:
    current_goal = goal_selector.select(state) if goal_selector else Goal("explore")

    # Score trajectory (replaces old total_gain logic):
    sim_score = score_trajectory(traj, stimuli=stimuli, goal=current_goal)
    # sim_score = (base_score, goal_prog, -steps)
    score = (sim_score[0], sim_score[1], known, sim_score[2])
    #         stimuli      goal_prog    known  -steps
```

**Self-action suppression:** The existing code zeros `gain` for self-action plans (sleep HDC noise fix from Stage 83). With Stage 85, `goal_prog` replaces `gain` entirely — **remove the self-action zero-out completely**. Reason: `goal_prog` is already self-normalizing. If the goal is `sleep` and action is sleep → `vital_delta("energy")` > 0, which is the correct signal. If the goal is `fight_zombie` and action is sleep → `vital_delta("health")` ≈ 0, which correctly penalizes the wrong action. No suppression needed.

**Import in vector_mpc_agent.py:** Add at function entry (lazy import to avoid circular):
```python
from snks.agent.goal_selector import Goal, GoalSelector
```

`stage85_eval.py` passes `textbook=tb` to `run_vector_mpc_episode`.

---

## Files Changed

| File | Change |
|---|---|
| `configs/crafter_textbook.yaml` | Add `goals: {primary: survive}` |
| `src/snks/agent/crafter_textbook.py` | Add `goals_block` property |
| `src/snks/agent/goal_selector.py` | NEW: `Goal`, `GoalSelector`, `_Threat` |
| `src/snks/agent/stimuli.py` | Add `CuriosityStimulus` (defined, not wired) |
| `src/snks/agent/vector_sim.py` | `VectorTrajectory`: `confidences` field + `vital_delta`, `inventory_delta`, `item_gained` methods; `simulate_forward`: accumulate confidences; `score_trajectory`: add `goal` param, remove `total_gain` |
| `src/snks/agent/vector_mpc_agent.py` | Add `textbook` param, `GoalSelector(textbook)`, thread `current_goal`, fix self-action suppression |
| `tests/test_goal_selector.py` | NEW |
| `experiments/stage85_eval.py` | NEW |

---

## Eval Gates

| Gate | Criterion |
|---|---|
| `survival_ge_155` | `avg_survival ≥ 155` (regression floor) |
| `wood_ge_10pct` | wood ≥ 3 in ≥ 10% of episodes |
| `no_total_gain` | `score_trajectory` has no `total_gain` reference |

Primary signal: `wood ≥ 10%` is evidence that goal-directed planning works where pure stimuli didn't.

**Note on total_gain removal:** `total_gain` counted all positive inventory deltas regardless of current goal. Its signal is preserved in `goal_prog`: when `current_goal.id == "gather_wood"`, `goal_prog = inventory_delta("wood")` — the same signal, but activated only when the goal is active. When goal is `fight_zombie`, wood deltas do not pollute the score. This is the intended behavior change.

---

## What This Is Not

- **Not a behavior tree**: no hardcoded `if zombie → craft_sword` sequence. Response derived from textbook rules + inventory state.
- **Not a new hardcode**: `goals` YAML adds one line. Threat→response mapping is derived.
- **Not replacing curiosity**: curiosity lives in `Goal("explore").progress()`. Coexists with homeostasis stimuli.
