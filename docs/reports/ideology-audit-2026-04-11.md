# Ideology Audit — 2026-04-11

**Against:** `docs/IDEOLOGY.md` v2 (commit `1df6313`)
**Method:** explore-agent pass through `src/snks/agent/*.py`, `configs/crafter_textbook.yaml`; grep for Crafter-specific string literals; check anti-pattern 1 (learning known facts) and anti-pattern 2 (env in mechanisms).
**Result:** **15 concrete violations**, all with proposed YAML fix.

## Anti-pattern 2 — Env semantics in mechanisms (10 violations)

Each violation below is a Crafter-specific fact currently encoded as Python
code in a mechanism. IDEOLOGY v2 anti-pattern 2 rule: "если patch в mechanism
содержит string literal с именем concept'а из конкретного env, стоп — это
fact в плохом месте".

### 2.1 — Facing direction mapping in `_nearest_concept`

`src/snks/agent/concept_store.py:1175-1202`

```python
dx, dy = 0, 1  # default: facing down
if sim.last_action == "move_left":
    dx, dy = -1, 0
elif sim.last_action == "move_right":
    dx, dy = 1, 0
elif sim.last_action == "move_up":
    dx, dy = 0, -1
elif sim.last_action == "move_down":
    dx, dy = 0, 1
```

Crafter env convention encoded as code. Other envs may use different
axis conventions or directional primitives.

**Proposed textbook:**
```yaml
primitives:
  move_left:  { dx: -1, dy: 0 }
  move_right: { dx:  1, dy: 0 }
  move_up:    { dx:  0, dy: -1 }
  move_down:  { dx:  0, dy:  1 }
env_defaults:
  default_facing: [0, 1]
```

### 2.2 — Facing check in `_expand_to_primitive` "do" branch

`src/snks/agent/concept_store.py:1308-1378` (Bug 2 fix)

Same facing direction mapping as 2.1, duplicated. Pulled into expand
because "do" needs facing alignment check.

**Proposed mechanism:** single helper `_facing_from_last_action(last_action, store)`
that reads `primitives.*.dx/dy` from textbook, used by both `_nearest_concept`
and `_expand_to_primitive`.

### 2.3 — `_apply_player_move` axis convention

`src/snks/agent/concept_store.py:1154-1172`

```python
if primitive == "move_left":
    return (pos[0] - 1, pos[1])
if primitive == "move_right":
    return (pos[0] + 1, pos[1])
# ...
```

Crafter's (X=pos[0], Y=pos[1]) convention hardcoded. Should read from
the `primitives.*.dx/dy` textbook attributes proposed in 2.1.

### 2.4 — `impassable_concepts()` heuristic

`src/snks/agent/concept_store.py:305-337`

Current heuristic: "any concept with a `do <X>` rule producing positive
`inventory_delta` is blocking, plus any mobile entity". This is
**Crafter-specific physics assumption** dressed up as a "generic rule".
Other envs might have non-blocking resources or blocking non-gatherables.

**Proposed textbook:**
```yaml
vocabulary:
  - { id: tree,     category: resource, blocking: true }
  - { id: stone,    category: resource, blocking: true }
  - { id: water,    category: resource, blocking: true }
  - { id: cow,      category: resource, blocking: true }
  - { id: coal,     category: resource, blocking: true }
  - { id: iron,     category: resource, blocking: true }
  - { id: skeleton, category: enemy,    blocking: true }
  - { id: zombie,   category: enemy,    blocking: true }
  - { id: empty,    category: terrain,  blocking: false }
```

**Generic mechanism:**
```python
def impassable_concepts(self) -> set[str]:
    return {cid for cid, c in self.concepts.items()
            if c.attributes.get('blocking', False)}
```

### 2.5 — `primitive_to_action_idx` hardcoded mapping

`src/snks/agent/concept_store.py:42-66`

8 discrete action slots hardcoded for the Stage 78c residual predictor.
Mixes Crafter's action set with the residual's encoding parameters.

**Proposed:** separate `residual_config` in textbook or in a config file:
```yaml
residual_config:
  n_actions: 8
  action_mapping:
    move_left: 0
    move_right: 1
    move_up: 2
    move_down: 3
    do: 4
    sleep: 5
    place_*: 6
    make_*: 7
```

Low priority (Stage 78c residual currently not used in Stage 79+).

### 2.6 — Exploration direction cycle

`src/snks/agent/concept_store.py:1243`

```python
dirs = ["move_up", "move_right", "move_down", "move_left"]
```

Hardcoded cardinal direction cycle. Needs to read from
`primitives.move_cardinal` list in textbook.

### 2.7 — `_near_label` in chain_generator

`src/snks/agent/chain_generator.py:38-46`

```python
if step.action == "do":
    return step.target  # "tree", "stone", "coal", "iron"
if step.action == "place":
    return "empty"
if step.action == "make":
    return "table"
```

Hardcoded assumptions "make needs table", "place needs empty". These
ARE already in textbook (`rule.near` field), but chain_generator doesn't
read them.

**Fix:** chain_generator should iterate textbook rules for the action
and pull `requires_near` from the matching rule.

### 2.8 — Hardcoded dicts in `outcome_labeler.py`

`src/snks/agent/outcome_labeler.py:12-37`

```python
DO_GAIN_TO_NEAR = {"wood": "tree", "stone": "stone", ...}
MAKE_GAIN_TO_NEAR = {"make_wood_pickaxe": "table", ...}
PLACE_ACTION_COST = {"place_table": {"wood": 2}, ...}
```

**All** this data exists in textbook as rule effects. The labeler
should query rules at runtime instead of maintaining parallel dicts.

### 2.9 — Facing update after blocked move (Bug 6 fix)

`src/snks/agent/mpc_agent.py:713-721`

Duplicates the facing mapping from 2.1/2.2 in a third location (the
spatial_map cleanup path). Same fix — shared helper reading textbook.

### 2.10 — Default facing direction

Spread across 2.1, 2.2, 2.9: `dx, dy = 0, 1` meaning "down by default".
Should live as `env_defaults.default_facing: [0, 1]` in textbook.

## Anti-pattern 1 — Learning what the teacher knows (5 violations)

### 1.1 — Conjunctive stateful rules absent from textbook

Stage 78a's synthetic task asked the nursery to discover
`sleep + (food=0 OR drink=0) → health -0.067`. This is **physics the
teacher can specify**. We spent Stages 78a/78c/79 on discovery
attempts. Stage 79 synthetic test proved induction works on this but
it's not necessary — just write it.

**Current textbook gap:** no `any_of` / `all_of` grammar in stateful
conditions. Can only express single-var predicates.

**Proposed textbook grammar extension:**
```yaml
- passive: stateful
  when:
    any_of:
      - { var: food,  op: "==", value: 0 }
      - { var: drink, op: "==", value: 0 }
    action_filter: sleep  # only when sleep is current primitive
  effect: { body: { health: -0.067 } }
```

**Mechanism change:** `_apply_tick` Phase 4 needs to handle
`any_of`/`all_of` in `when` clause. ~20 lines.

### 1.2 — Movement mechanics underspecified

`crafter_textbook.yaml:146-156`:
```yaml
- passive: movement
  entity: zombie
  behavior: chase_player
```

`chase_player` is a magic string; its implementation in
`_apply_movement()` is Crafter-specific. Teacher could specify:

```yaml
- passive: movement
  entity: zombie
  mechanics:
    algorithm: manhattan_greedy
    priority_axis: max_abs
    step_size: 1
    range: infinite
```

### 1.3 — Blocking semantics inferred not declared

Same as 2.4 but framed from the "facts" side. Teacher **knows** which
tiles are blocking in Crafter. Currently we pretend the mechanism
derives it, but it's just hardcoded assumption.

### 1.4 — Action dispatch semantics only in code comments

Stage 79 Bug 2 comments in `concept_store.py:1309-1324` explicitly
note: "Crafter env's 'do' action interacts with the FACING tile, not
any tile in manhattan ≤ 1". This is a **fact** that lives as a
code comment. Should be a textbook record.

**Proposed:**
```yaml
env_semantics:
  do:
    dispatch: facing_tile
    range: 1
    interaction_mode: direct  # not proximity
  move:
    blocked_by: impassable_tiles
    updates_facing: true  # even on blocked move
```

### 1.5 — Perception frame geometry not in textbook

`DynamicEntityTracker.update()` has `viewport_center=(3,4)` as a
parameter default for Crafter's 7×9 viewport. Not documented
anywhere except in code.

**Proposed:**
```yaml
perception:
  viewport: { height: 7, width: 9, center: [3, 4] }
```

## Migration priority

**High priority (addresses current 8 bugs)**:
- 2.4 `blocking` attribute → removes `impassable_concepts()` heuristic
- 1.4 env dispatch semantics → enables generic `expand_to_primitive`
- 1.1 conjunctive rules → removes Stage 78a/78c/79 core motivation

**Medium priority (code cleanliness)**:
- 2.1/2.2/2.3/2.9 facing direction consolidation → one helper reading textbook
- 2.6 exploration direction list
- 2.7 chain_generator near labels
- 2.8 outcome_labeler dicts

**Low priority (rarely used):**
- 2.5 residual action_idx mapping (Stage 78c residual is parked)
- 1.2 movement mechanics (works as is)
- 1.5 perception geometry (doc only)
- 2.10 default facing (absorbed into 2.1)

## Post-migration state

After migrating 2.4, 1.1, 1.4, and the facing consolidation:
- `impassable_concepts()` becomes one-liner reading `blocking` attribute
- `expand_to_primitive` facing check becomes shared helper
- `_apply_tick` Phase 4 supports `any_of`/`all_of` conditional rules
- 8 Crafter-specific hardcoded sites collapse to ~2 shared helpers
- New env (MiniGrid, Minecraft) = new textbook YAML, no code changes

## What this audit does NOT cover

- **Anti-pattern 3** (not observing agent) — process discipline,
  not code. Audit irrelevant.
- **Anti-pattern 4** (sleeping baselines) — historical issue, now
  we have explicit baseline reproducibility in Stage 80/81 reports.
- **Anti-pattern 5** (local fixes vs right layer) — this audit
  itself is the "stepping back" response. Stage 82 candidate is
  "apply these migrations as a batch".

## Stage 82 candidate

Based on this audit, a natural Stage 82 scope is:

1. Extend textbook grammar with `blocking`, `env_semantics`,
   `primitives`, `any_of`/`all_of` in stateful
2. Migrate the 10 high/medium priority violations
3. Preserve `_apply_tick` behaviour (no regression on 110 tests)
4. Add the conjunctive sleep rule directly to textbook → Stage 79
   nursery becomes diagnostic-only (it should NOT rediscover that rule)
5. Crafter eval should show ≥ Stage 80 performance (no regression)

This is ~1 day of focused refactor. Does not break the wall, but
removes 15 IDEOLOGY v2 violations in one pass.
