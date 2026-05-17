# Stage 9X Phase 1 — Goal-Conditioned Frontier Exploration

## Problem

Phase 0b audit (`output_to_user/stage9x_phase0b_audit_20260517.md`) showed that
seed 17 ep 0 spends 35 consecutive steps under `goal=find_water` while
`plan_origin=baseline` 25/35 times. The agent dies dehydrated and exposed to a
zombie at step ~185 — root cause: planner has no candidate plan to evaluate
when the goal target has never been observed, falls through baseline, then the
uniform-random move fallback walks in circles instead of toward unexplored
ground.

Same pattern blocks `goal=find_cow`, `goal=fight_<entity>`, and
`gather_<material>` whenever the relevant concept is not yet on the cognitive
map.

## Ideology check (before design)

Run against `docs/IDEOLOGY.md`, `docs/ROADMAP.md`,
`docs/CONCEPT_SUCCESS_CRITERIA.md`, `docs/ANTI_TUNING_CHECKLIST.md`:

| Question | Verdict for this design |
|---|---|
| Anti-Tuning Q1: describe without Crafter names | ✅ "goal-conditioned frontier exploration on a partial cognitive map" — general; Crafter concept names (water, cow) come from textbook rules, not Python. |
| Anti-Tuning Q2: lives in correct layer | ✅ Facts already in textbook (consumption rules). Mechanism is new plan-type in the existing `generate_candidate_plans` + `expand_to_primitive` + scoring stack. Experience layer (`spatial_map._visited`/`_blocked`) consumed unchanged. |
| Anti-Tuning Q3: explained by general capability | ✅ Capability: "act on partial cognitive map by directed frontier search when goal target is not yet known". |
| Anti-Tuning Q4: makes sense in neighbour domain | ✅ Any partially-observable domain with resource-seeking goals (MineRL, gridworld mazes, navigation envs). |
| Anti-Tuning Q5: capability vs only score | ✅ Validation criteria require *capability evidence* (`near_concept=water` reached once, `plan_origin=frontier:water:*` occurs in find_water phase) on top of survival. |
| IDEOLOGY Принцип 2 (top-down): no hidden if-else | ✅ Goal carries `target_concept` derived from textbook; planner uses it generically. No `if goal == "find_water"` branch in mechanism code. |
| IDEOLOGY Распределение: which layer | Facts: textbook consumption rules (existing). Mechanism: planner emits frontier plan + expand walks unvisited. Experience: spatial_map (existing). Stimuli: scoring extension lets `goal.progress` recognise frontier intent. |
| ROADMAP Phase I (Dynamic World Model) | ✅ Closes a residual planning gap exposed by Stage 90R rescue work — fits Phase I exit-gate "improvement claims tied to one named stage". |
| CONCEPT_SUCCESS_CRITERIA #2 (right layer) | ✅ Gain explained by mechanism+facts, not a tuning patch. |

Tactical alternative (inline if-else in `vector_mpc_agent.py:1326` RNG branch)
was explicitly rejected because it would fail Anti-Tuning Q1 (Crafter concept
names appear directly in mechanism code) and Q2 (special case in planner
fallback). The user's standing direction (2026-05-17 memory) is *clean
architecture over tempo at this stage*; this design takes the longer path
accordingly.

## Mechanism (four-layer mapping)

### Layer 1 — Facts (no textbook edit needed)

`configs/crafter_textbook.yaml` already contains the grounding rules:

```yaml
- action: do, target: water,  effect: { body: { drink: +5 } }
- action: do, target: cow,    effect: { body: { food:  +5 } }
- action: do, target: zombie, requires: { wood_sword: 1 }, effect: { remove_entity: zombie }
- action: do, target: tree,   effect: { inventory: { wood: +1 } }
```

The mapping `goal_id → target_concept` is **derived** from these rules at
`GoalSelector.__init__`, not hardcoded anywhere in Python:

- `find_water` → first concept in rules where `do <concept>` increases
  `body.drink`.
- `find_cow` → same for `body.food` (and for `find_<vital>` more generally).
- `fight_<entity>` → concept named after the prefix-stripped goal id; verified
  against `remove_entity` rules.
- `gather_<material>` → concept whose `do` produces inventory `<material>`.

If the textbook is missing the corresponding rule, the Goal's `target_concept`
stays `None` and behaviour falls back to the original RNG (no regression).

### Layer 2 — Goal-class extension

Add `target_concept: str | None = None` to `Goal` (dataclass) and propagate
via `to_trace()`. `GoalSelector` populates it for the goal kinds it emits:

```python
return Goal("find_water", target_concept="water", reason="low_drink")
```

This is *the* place where the textbook-derived fact becomes part of the goal
object. Planner mechanism does not parse goal-id strings.

### Layer 3 — Mechanism (planner)

In `generate_candidate_plans` (signature gains optional `active_goal`):

- If `active_goal is not None` and `active_goal.target_concept` is a known
  textbook concept and `spatial_map.find_nearest(active_goal.target_concept,
  player_pos)` is None, emit a single plan:

```python
VectorPlan(
    steps=[VectorPlanStep(action="frontier_seek",
                          target=active_goal.target_concept)],
    origin=f"frontier:{active_goal.target_concept}",
)
```

`frontier_seek` is a new action understood by `expand_to_primitive` (not part
of the env action set). `simulate_forward` short-circuits frontier plans by
returning a zero-effect trajectory; the plan exists for scoring, not for
sim-physics rollout.

In `expand_to_primitive`, when `plan_step.action == "frontier_seek"`:

1. `unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)`
2. If non-empty, pick the closest cell (manhattan); ties broken by sorted
   iteration order (deterministic under `PYTHONHASHSEED=0`).
3. `_step_toward(player_pos, chosen_cell, model, rng)`.
4. If empty, return a move from `model.actions` chosen by `rng.choice` —
   the same uniform fallback the original code uses when nothing else is
   knowable.

In `_plan_distance`, frontier plans return the manhattan distance to the
chosen frontier cell instead of 9999. This keeps `known=1` (the dist-known
flag in the score tuple) — frontier IS knowable, just not direct.

### Layer 4 — Stimuli (scoring)

`Goal.progress(trajectory)` extension: for `find_<vital>` and `fight_<entity>`
goals, if `trajectory.plan.origin.startswith("frontier:")` and the plan's
target matches `self.target_concept`, return a small positive constant
(`FRONTIER_PROGRESS_EPSILON = 0.05`) so the score tuple beats baseline at
the `goal_prog` position when the goal is active.

This is *not* a tuning constant aimed at one number — it is a tiebreaker
that promotes "directed exploration toward goal target" over "purely random
fallback" when no concrete plan exists. Documented in `Goal.progress` with
the rationale.

When the goal does not match the plan's target, `Goal.progress` returns 0
as today — frontier plans don't spuriously win under `goal=explore`.

## Validation

### Unit tests (local, fast)

- `tests/test_goal_selector.py`: `GoalSelector(textbook)` populates
  `target_concept` correctly for find_water, find_cow, fight_zombie,
  fight_skeleton, gather_wood; falls back to None when textbook rule absent.
- `tests/test_vector_mpc.py`: `generate_candidate_plans` emits a
  `frontier:water` plan when goal=find_water with no water on map; emits
  no frontier plan when water already on map; no frontier when goal is
  explore.
- `tests/test_vector_mpc.py`: `expand_to_primitive` with
  `frontier_seek/water` returns a move directed toward the closest
  unvisited cell on a synthetic map.
- `tests/test_goal_selector.py`: `Goal("find_water", target_concept="water").progress(traj)`
  with `frontier:water` origin returns `FRONTIER_PROGRESS_EPSILON > 0`;
  with `baseline` origin returns 0; with `frontier:cow` origin under
  goal=find_water returns 0.

### Integration (HyperPC, seed 17 ep 0)

- Recording: full-profile, identical env vars to prior baselines (commit-hash
  in filename).
- Pass criteria (capability, not score):
  1. `near_concept=water` appears at least once in the `find_water` phase.
  2. `plan_origin=frontier:water:*` count > 0 during `goal=find_water` steps.
  3. The 35-step blind RNG cluster from afc4132/ff6e72a measurably shrinks
     (target: `baseline` <50% of find_water-phase steps; current 71%).
- Acceptable regressions (these would still pass):
  - Episode length stays similar or improves.
  - `make_wood_sword`, `place_table` counts ≥ ff6e72a baseline (2/1).
  - No new crash modes; determinism preserved (re-record once to confirm
    byte-identity).
- Rejection: if frontier plans fire but agent still dehydrates because the
  unvisited frontier doesn't include water (map topology issue), Phase 1 was
  necessary but not sufficient — escalate to wider exploration radius or
  scoring rework, do not silently tune the epsilon.

## What this fix does NOT solve

- `fight_<entity>` with visible target but no `single:<entity>:do` plan
  because of `non_targetable = {"empty", "self", "zombie", "skeleton"}` in
  `generate_candidate_plans`. Phase 2 dochinit.
- Baseline-RNG choosing `make_wood_sword` redundantly when armed_melee=True
  (no capability-redundancy gate). Phase 3.
- Goal-selector preferring `fight_skeleton` over `fight_zombie` when both
  present. Phase 2 dochinit.
- `controller_distribution` already fixed in commit `10d8bca`.

## Estimate

5-8 hours focused work + 1 HyperPC recording cycle:

| Layer | Files touched | LOC | Tests |
|---|---|---|---|
| Goal class | `goal_selector.py` | ~10 | 1 |
| Textbook derivation | `goal_selector.py` | ~40 (helper) | 2 |
| Planner emit | `vector_mpc_agent.py` (generate_candidate_plans) | ~25 | 2 |
| expand_to_primitive | `vector_mpc_agent.py` | ~20 | 1 |
| _plan_distance | `vector_mpc_agent.py` | ~10 | 1 |
| simulate_forward short-circuit | `vector_sim.py` | ~15 | 1 |
| Goal.progress epsilon | `goal_selector.py` | ~15 | 2 |
| Total | 3 files | ~135 | 10 |
