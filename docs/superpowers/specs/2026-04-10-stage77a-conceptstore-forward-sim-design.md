# Stage 77a — Forward simulation через ConceptStore (design spec)

**Дата:** 2026-04-10
**Статус:** Design — awaiting implementation plan
**Scope:** Phase B Part 1 из трёх (77a wall-breaker, 77b novelty, 77c cleanup polish)
**Основание:** `docs/reports/architecture-review-2026-04-10.md` (Phase A diagnosis)
**Эталон:** `docs/IDEOLOGY.md`

---

## TL;DR

Заменяем Stage 76 EpisodicSDM substrate на **forward simulation через каузальные правила ConceptStore**. Один MPC loop, одна цель (выжить), один world model. Каждый шаг: baseline rollout предсказывает failures → `find_remedies` ищет правила в world model → backward chain генерирует кандидатные планы → `simulate_forward` катит траектории → лексикографический scoring выбирает лучший → выполняется первое примитивное действие → re-plan на следующем шаге.

Textbook расширяется на stateful/movement/spatial правила, причём grammar — **structured YAML** (не regex), что убирает фрагильность и готовит к Minecraft/AGI.

Новизна в системе появляется из трёх источников: (1) forward rollouts через правила видят multi-step causality, (2) confidence-filtered правила дают агенту возможность «разочароваться», (3) baseline-driven candidate generation эмерджит приоритеты из предсказанного будущего, а не из хардкоженных категорий.

**Primary gate:** survival ≥200 — тот самый wall, который Stage 74/75/76 не смогли взять.
**Ideology gate:** 0 hardcoded категорий, 0 магических порогов в policy code, всё эмерджит из world model + body physiology.

Итого: **~800 строк удалено, ~1200 строк добавлено, net -600 строк**. Все 15 findings F1-F15 из Phase A адресуются архитектурно или объявлены deferred в 77b/77c с документированной причиной.

---

## 1. Context

### Почему 77a существует

Phase A (architecture review `docs/reports/architecture-review-2026-04-10.md`) обнаружила, что живой путь Stage 76 содержит два параллельных world model'я (ConceptStore + EpisodicSDM), которые не делят ни state representation, ни learning signal, ни reasoning. Они переключаются через `if/else` в `continuous_agent.py:248-267` и **мешают** друг другу (wood collection в smoke 65% → 40% под Stage 76 v1).

Root cause стены 178 — не «дефицит forward simulation» (как диагностировал Stage 76 report), а **отсутствие источника новизны**:
- Path B (EpisodicSDM) on-policy к Path A (ConceptStore bootstrap)
- Path A не учится из опыта (F5 — `link.confidence` writer-only, не читается планировщиком)
- Body prior тает через EMA за ~60 шагов (F6)
- Strategy 1/2/Preparation в `select_goal` hardcoded (F4)

Stage 77 в оригинальной формулировке («forward sim через SDM rollouts») — четвёртый виток той же петли: добавить слой поверх несогласованной основы. Phase A прогнозировала survival 185-195, gate не пройден.

### Что делает 77a

Реализует forward simulation **через ConceptStore causal rules**, как IDEOLOGY Stage 73 всегда описывала:

> *«Стратегия = forward simulation через мировую модель ... Это не RL. Это model-based planning из каузальных правил. World model уже есть — нужно научить agent'а её использовать для принятия решений.»*

Удаляет EpisodicSDM как отклонение от идеологии. Чинит F3-F6 (feedback loops, priors, hardcoded strategies). Откладывает F15 novelty mechanism (surprise → new rule) на 77b.

---

## 2. Ideology alignment

Каждое архитектурное решение 77a мапится на явный пункт `IDEOLOGY.md`:

| IDEOLOGY тезис | Реализация в 77a |
|---|---|
| «Стратегия = forward simulation через мировую модель» (Stage 73) | `ConceptStore.simulate_forward` — рекурсивное применение causal rules; MPC loop вызывает её каждый шаг |
| «Drives = наблюдение за телом» (Stage 74) | `HomeostaticTracker` с innate/observed split; drives **не** итерируются, они **предсказываются** baseline rollout'ом |
| «Цель → модель → стратегия из опыта» (Stage 73) | Одна singular цель (выжить), world model = ConceptStore, стратегия = победивший кандидат в MPC |
| «НИКОГДА не хардкодить стратегию, приоритеты или рефлексы» (Stage 73) | `generate_candidate_plans` использует baseline → failures → remedies; priority эмерджит из порядка failures в предсказанном будущем |
| «Три скорости обучения» (Stage 72) | Мгновенное: confidence update + (77b) surprise→rule; Постепенное: observed_rates running mean, spatial_map; Фоновое: CNN retrain (не трогаем) |
| «Textbook = родитель объясняет ребёнку» (Stage 71) | Structured YAML textbook — declarative facts, без лжи про restore_health, без stratege leak |
| «CNN = V1, ConceptStore = thought» (Stage 72) | Tile segmenter не трогается; ConceptStore получает forward sim и становится полноценной «мыслью» |

### Taxonomy трёх категорий знания

Из `feedback_textbook_taxonomy.md` memory:

| Категория | Где живёт | Примеры в 77a |
|---|---|---|
| **1. Факты о мире** | `configs/crafter_textbook.yaml` | `do tree gives wood`, `zombie moves toward player`, `food > 0 restores health`, `prior_strength: 20` |
| **2. Механизмы** | Python код | `simulate_forward`, backward chaining, EMA, MPC loop, confidence threshold 0.1 |
| **3. Выученное из опыта** | `ConceptStore` runtime state | Visual grounding, `link.confidence`, `observed_rates`, `spatial_map` |

Стратегии/приоритеты/рефлексы — **НИ в одной из трёх**. Эмерджат из forward sim.

---

## 3. Scope — три подэтапа

### Stage 77a (этот spec) — Wall-breaker foundation

**Цель:** разбить стену 178, реализовав forward sim через ConceptStore rules. Заменить Path A/Path B if-else на единый MPC loop.

**Включает:**
- `ConceptStore.simulate_forward` + `plan_toward_rule` + `find_remedies`
- Textbook grammar расширения (stateful/movement/spatial rules), structured YAML
- `run_mpc_episode` в новом `mpc_agent.py`
- `HomeostaticTracker` innate/observed split
- Удаление `src/snks/memory/` + `HOMEOSTATIC_VARS` + `Strategy 1/2/Preparation`
- Confidence threshold filter (binary, 0.1)

**Gate:** survival ≥200, wood smoke ≥50%, ideology lint clean.

### Stage 77b — Novelty mechanism (deferred)

**Цель:** surprise → new rule candidate (deferred question из `IDEOLOGY.md`).

**Отложено, потому что:** если 77a даёт survival ≥200 через preloaded textbook, novelty не является критическим. Crafter — well-known домен, textbook может быть достаточно полным. Surprise mechanism становится критичным только для будущих доменов (Minecraft extension, AGI tasks).

### Stage 77c — Cleanup polish (deferred)

**Цель:** low-value cleanup — dead code в `src/snks/agent/` (60+ legacy MiniGrid modules), spatial_map с confidence (F10), консолидация `perceive_*` функций.

**Отложено, потому что:** эти изменения не блокируют wall-breaking, но их совмещение с 77a раздувает spec и увеличивает риск.

---

## 4. Architecture overview

### Один control loop

```python
def run_mpc_episode(env, segmenter, store, tracker, rng, max_steps):
    pixels, info = env.reset()
    spatial_map = CrafterSpatialMap()
    entity_tracker = DynamicEntityTracker()
    prev_inv = None

    for step in range(max_steps):
        # Perception
        inv = dict(info["inventory"])
        player_pos = tuple(info["player_pos"])
        vf = perceive_tile_field(pixels, segmenter)
        spatial_map.update_from_viewport(vf, player_pos)
        entity_tracker.update(vf, player_pos)

        # Observe body physiology
        if prev_inv is not None:
            tracker.update(prev_inv, inv, vf.visible_concepts())

        # Build current SimState
        state = SimState(
            inventory=inv,
            body=extract_body(inv, tracker),
            player_pos=player_pos,
            dynamic_entities=entity_tracker.current(),
            spatial_map=spatial_map,
            last_action=prev_action,
            step=step,
        )

        # Generate candidate plans via baseline rollout
        candidates = generate_candidate_plans(state, store, tracker)

        # Simulate and score each candidate
        scored = []
        for plan in candidates:
            traj = store.simulate_forward(plan, state, horizon=20)
            score = score_trajectory(traj, tracker)
            scored.append((score, plan, traj))

        # Pick best, execute first primitive
        scored.sort(key=lambda x: x[0], reverse=True)
        _, best_plan, _ = scored[0]
        primitive = expand_to_primitive(best_plan.steps[0], state)

        # Execute in real env
        pixels, _, done, info = env.step(primitive)

        # Update confidence based on outcome
        outcome = outcome_to_verify(primitive, inv, info["inventory"])
        if outcome:
            verify_outcome(vf.near_concept, primitive, outcome, store)

        prev_inv = inv
        prev_action = primitive

        if done:
            break
```

**Ключевые архитектурные принципы:**

1. **Один path.** Никаких if/else между «bootstrap» и «SDM». Всегда forward sim через rules.
2. **MPC — re-plan каждый шаг.** Никакого commitment между шагами. Это решает Stage 75 диагноз «plan is linear, cannot adapt to threats mid-execution».
3. **Одна цель.** `score_trajectory` — одна функция, не Strategy 1/2/Preparation ladder.
4. **Снапшоты, не shared state.** `SimState` — это копия текущего мира для rollout'а; spatial_map не обновляется в rollout'е (нельзя узнать новое в воображении).

---

## 5. Data types

### `SimState`

```python
@dataclass
class SimState:
    """Snapshot воображаемого состояния мира в одном sim-тике."""
    inventory: dict[str, int]
    body: dict[str, float]                      # float для fractional rates
    player_pos: tuple[int, int]
    dynamic_entities: list[DynamicEntity]
    spatial_map: CrafterSpatialMap              # read-only snapshot
    last_action: str | None                     # для inertia baseline
    step: int

    def copy(self) -> "SimState":
        """Deep-copy для начала rollout'а."""
        ...

    def is_dead(self, tracker: "HomeostaticTracker") -> bool:
        """Любая body var достигла reference_min → смерть."""
        for var, value in self.body.items():
            ref_min = tracker.reference_min.get(var, 0)
            if value <= ref_min:
                return True
        return False

@dataclass
class DynamicEntity:
    """Сущность с позицией, которая может двигаться (враг, моб)."""
    concept_id: str                             # "zombie" | "skeleton" | "cow" | ...
    pos: tuple[int, int]
```

### `RuleEffect` и `StatefulCondition`

```python
@dataclass
class RuleEffect:
    """Структурированный эффект применения правила на SimState.
    Заменяет string-based CausalLink.result. Эффект dispatchable directly."""
    kind: str                                   # "gather" | "craft" | "place" | "remove" | "movement" | "spatial" | "stateful" | "body_rate"
    inventory_delta: dict[str, int] = field(default_factory=dict)
    body_delta: dict[str, float] = field(default_factory=dict)
    scene_remove: str | None = None             # для "remove": concept_id to remove from dynamic_entities
    world_place: tuple[str, str] | None = None  # для "place": (item, where) e.g. ("table", "adjacent_empty")
    movement_behavior: str | None = None        # "chase_player" | "flee_player" | "random_walk"
    spatial_range: int = 1                      # для "spatial": manhattan distance threshold
    stateful_condition: "StatefulCondition | None" = None
    body_rate: float = 0.0                      # для "body_rate": per-tick delta (background decay)
    body_rate_variable: str | None = None

@dataclass
class StatefulCondition:
    """Evaluates to bool at each tick based on SimState."""
    var: str                                    # body var name
    op: str                                     # ">" | "<" | "==" | ">=" | "<="
    threshold: float

    def satisfied(self, sim: SimState) -> bool:
        val = sim.body.get(self.var, 0)
        return {
            ">":  val > self.threshold,
            "<":  val < self.threshold,
            "==": val == self.threshold,
            ">=": val >= self.threshold,
            "<=": val <= self.threshold,
        }[self.op]
```

### `CausalLink` updated

```python
@dataclass
class CausalLink:
    """One causal rule. Dispatchable via `effect`, not via string `result`."""
    kind: str                                   # "action_triggered" | "passive_body_rate" | "passive_movement" | "passive_spatial" | "passive_stateful"
    concept: str | None                         # target concept for action_triggered; entity for passive
    action: str | None                          # "do" | "make" | "place" | "sleep" | None for passive
    effect: RuleEffect
    requires: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.5
    # Legacy field removed: `result: str` — all dispatch via `effect`.
```

### `Trajectory`, `SimEvent`, `Failure`, `Plan`

```python
@dataclass
class SimEvent:
    step: int
    kind: str                                   # "body_delta" | "damage" | "inv_gain" | "rule_applied" | "death"
    var: str | None
    amount: float
    source: str                                 # concept_id | "_background" | "rule:<kind>:<concept>"

@dataclass
class Trajectory:
    plan: "Plan"
    body_series: dict[str, list[float]]         # var → per-tick values
    events: list[SimEvent]
    final_state: SimState
    terminated: bool
    terminated_reason: str                      # "body_dead" | "horizon" | "plan_complete"
    plan_progress: int                          # сколько PlannedSteps завершено

    def failure_step(self, var: str) -> int | None:
        for i, v in enumerate(self.body_series.get(var, [])):
            if v <= 0:
                return i
        return None

@dataclass
class Failure:
    """Наблюдение из baseline trajectory — что именно пошло не так."""
    kind: str                                   # "var_depleted" | "attributed_to"
    var: str | None                             # для var_depleted
    cause: str | None                           # для attributed_to: concept_id
    step: int                                   # when in trajectory
    severity: float

@dataclass
class PlannedStep:
    action: str                                 # "do" | "make" | "place" | "sleep" | "inertia" | "move"
    target: str | None                          # concept_id to interact with / navigate toward
    near: str | None                            # for make/place: what must be nearby
    rule: CausalLink | None                     # ссылка на target rule (для completion check)

@dataclass
class Plan:
    steps: list[PlannedStep]
    origin: str                                 # "baseline" | "remedy" | "explore" — для debug/logging
```

---

## 6. Textbook grammar

### Format: structured YAML, не regex

Каждое правило — YAML dict с `action` (action-triggered) или `passive` (per-tick).

**Полный пример `configs/crafter_textbook.yaml`:**

```yaml
domain: crafter

vocabulary:
  - { id: tree,     category: resource, kind: entity }
  - { id: stone,    category: resource, kind: entity }
  - { id: coal,     category: resource, kind: entity }
  - { id: iron,     category: resource, kind: entity }
  - { id: water,    category: resource, kind: entity }
  - { id: cow,      category: resource, kind: entity }
  - { id: table,    category: crafted,  kind: entity }
  - { id: empty,    category: terrain,  kind: entity }
  - { id: zombie,   category: enemy,    kind: entity }
  - { id: skeleton, category: enemy,    kind: entity }
  - { id: wood,            category: item, kind: inventory }
  - { id: stone_item,      category: item, kind: inventory }
  - { id: coal_item,       category: item, kind: inventory }
  - { id: iron_item,       category: item, kind: inventory }
  - { id: wood_pickaxe,    category: tool, kind: inventory }
  - { id: stone_pickaxe,   category: tool, kind: inventory }
  - { id: wood_sword,      category: weapon, kind: inventory }

body:
  prior_strength: 20
  variables:
    - { name: health, initial: 9, reference_min: 0, reference_max: 9 }
    - { name: food,   initial: 9, reference_min: 0, reference_max: 9 }
    - { name: drink,  initial: 9, reference_min: 0, reference_max: 9 }
    - { name: energy, initial: 9, reference_min: 0, reference_max: 9 }

rules:
  # ========== ACTION-TRIGGERED ==========

  # Gather (inventory effect)
  - action: do
    target: tree
    effect: { inventory: { wood: +1 } }

  - action: do
    target: stone
    requires: { wood_pickaxe: 1 }
    effect: { inventory: { stone_item: +1 } }

  - action: do
    target: coal
    requires: { wood_pickaxe: 1 }
    effect: { inventory: { coal_item: +1 } }

  - action: do
    target: iron
    requires: { stone_pickaxe: 1 }
    effect: { inventory: { iron_item: +1 } }

  # Consume (body effect directly)
  - action: do
    target: cow
    effect: { body: { food: +5 } }

  - action: do
    target: water
    effect: { body: { drink: +5 } }

  # Combat / removal
  - action: do
    target: zombie
    requires: { wood_sword: 1 }
    effect: { remove_entity: zombie }

  - action: do
    target: skeleton
    requires: { wood_sword: 1 }
    effect: { remove_entity: skeleton }

  # Craft
  - action: make
    result: wood_pickaxe
    near: table
    requires: { wood: 1 }
    effect: { inventory: { wood_pickaxe: +1, wood: -1 } }

  - action: make
    result: stone_pickaxe
    near: table
    requires: { wood: 1, stone_item: 1 }
    effect: { inventory: { stone_pickaxe: +1, wood: -1, stone_item: -1 } }

  - action: make
    result: wood_sword
    near: table
    requires: { wood: 1 }
    effect: { inventory: { wood_sword: +1, wood: -1 } }

  # Place
  - action: place
    item: table
    near: empty
    requires: { wood: 2 }
    effect:
      world_place: { item: table, where: adjacent_empty }
      inventory: { wood: -2 }

  # Self-action
  - action: sleep
    effect: { body: { energy: +5 } }

  # ========== PASSIVE (per-tick in simulator) ==========

  # Background body rates (replaces old `body: [{concept: _background, ...}]`)
  - passive: body_rate
    variable: food
    rate: -0.04

  - passive: body_rate
    variable: drink
    rate: -0.04

  - passive: body_rate
    variable: energy
    rate: -0.03

  # Entity movement
  - passive: movement
    entity: zombie
    behavior: chase_player

  - passive: movement
    entity: skeleton
    behavior: chase_player

  - passive: movement
    entity: cow
    behavior: random_walk

  # Spatial damage (adjacent = manhattan ≤ range)
  - passive: spatial
    entity: zombie
    range: 1
    effect: { body: { health: -2 } }

  - passive: spatial
    entity: skeleton
    range: 1
    effect: { body: { health: -1 } }

  # Stateful (condition on body)
  - passive: stateful
    when: { var: food, op: ">", value: 0 }
    effect: { body: { health: +0.1 } }

  - passive: stateful
    when: { var: drink, op: ">", value: 0 }
    effect: { body: { health: +0.1 } }

  - passive: stateful
    when: { var: food, op: "==", value: 0 }
    effect: { body: { health: -0.5 } }

  - passive: stateful
    when: { var: drink, op: "==", value: 0 }
    effect: { body: { health: -0.5 } }
```

### Parser (`crafter_textbook.py`)

Dict dispatch, не regex:

```python
def parse_rule(entry: dict) -> CausalLink:
    if "action" in entry:
        return _parse_action_rule(entry)
    elif "passive" in entry:
        return _parse_passive_rule(entry)
    raise ValueError(f"unknown rule format: {entry}")

def _parse_action_rule(entry: dict) -> CausalLink:
    action = entry["action"]
    effect_dict = entry.get("effect", {})
    effect = _build_effect(effect_dict, entry)
    return CausalLink(
        kind="action_triggered",
        concept=entry.get("target") or entry.get("near"),
        action=action,
        effect=effect,
        requires=entry.get("requires", {}),
        confidence=0.5,
    )

def _parse_passive_rule(entry: dict) -> CausalLink:
    passive_type = entry["passive"]
    if passive_type == "body_rate":
        return CausalLink(
            kind="passive_body_rate",
            concept=None,
            action=None,
            effect=RuleEffect(
                kind="body_rate",
                body_rate=entry["rate"],
                body_rate_variable=entry["variable"],
            ),
            confidence=1.0,  # innate rules start fully trusted
        )
    elif passive_type == "movement":
        return CausalLink(
            kind="passive_movement",
            concept=entry["entity"],
            action=None,
            effect=RuleEffect(
                kind="movement",
                movement_behavior=entry["behavior"],
            ),
            confidence=0.5,
        )
    elif passive_type == "spatial":
        return CausalLink(
            kind="passive_spatial",
            concept=entry["entity"],
            action=None,
            effect=RuleEffect(
                kind="spatial",
                body_delta=entry["effect"]["body"],
                spatial_range=entry.get("range", 1),
            ),
            confidence=0.5,
        )
    elif passive_type == "stateful":
        when = entry["when"]
        return CausalLink(
            kind="passive_stateful",
            concept=None,
            action=None,
            effect=RuleEffect(
                kind="stateful",
                body_delta=entry["effect"]["body"],
                stateful_condition=StatefulCondition(
                    var=when["var"],
                    op=when["op"],
                    threshold=float(when["value"]),
                ),
            ),
            confidence=0.5,
        )
    raise ValueError(f"unknown passive type: {passive_type}")
```

---

## 7. `ConceptStore` new API

### `simulate_forward`

```python
def simulate_forward(
    self,
    plan: Plan,
    initial_state: SimState,
    horizon: int = 20,
) -> Trajectory:
    """Roll out a plan through causal rules. Returns trajectory.

    Deterministic simulation: at each tick, apply all passive rules
    (movement, body rate, spatial, stateful), then agent primitive
    action, then check termination. Continue with inertia after plan
    completes.
    """
    sim = initial_state.copy()
    traj = Trajectory(
        plan=plan,
        body_series={var: [] for var in sim.body},
        events=[],
        final_state=sim,
        terminated=False,
        terminated_reason="horizon",
        plan_progress=0,
    )
    plan_cursor = 0

    for tick in range(horizon):
        if sim.is_dead(self._tracker):
            traj.terminated = True
            traj.terminated_reason = "body_dead"
            break

        if plan_cursor < len(plan.steps):
            current_step = plan.steps[plan_cursor]
        else:
            current_step = PlannedStep(action="inertia", target=None, near=None, rule=None)

        primitive = expand_to_primitive(current_step, sim, self)
        prev_sim = sim.copy()
        self._apply_tick(sim, primitive, current_step, traj, tick)

        for var, value in sim.body.items():
            traj.body_series[var].append(value)

        if plan_cursor < len(plan.steps) and is_plan_step_complete(current_step, sim, prev_sim):
            plan_cursor += 1

    if not traj.terminated:
        traj.terminated_reason = (
            "plan_complete" if plan_cursor >= len(plan.steps) else "horizon"
        )

    traj.final_state = sim
    traj.plan_progress = plan_cursor
    return traj
```

### `plan_toward_rule`

```python
def plan_toward_rule(
    self,
    target_rule: CausalLink,
    state: SimState,
) -> list[PlannedStep]:
    """Backward chain prerequisites for applying target_rule.
    Recursively resolves requires + spatial preconditions.
    Final step = execution of target_rule."""
    steps: list[PlannedStep] = []
    visited: set[tuple[str, str]] = set()
    inv = dict(state.inventory)
    self._plan_for_rule(target_rule, steps, visited, inv)
    return steps

def _plan_for_rule(self, rule, steps, visited, inventory):
    key = (rule.concept or "", rule.action or "_passive")
    if key in visited:
        return
    visited.add(key)

    # Resolve item prerequisites
    for item, count in rule.requires.items():
        if inventory.get(item, 0) >= count:
            continue
        producer = self._find_rule_producing_item(item)
        if producer:
            self._plan_for_rule(producer, steps, visited, inventory)

    # Resolve spatial preconditions (e.g., must be near table for make)
    if rule.action in ("make", "place"):
        near_target = rule.effect.world_place[1] if rule.effect.world_place else rule.concept
        if near_target:
            producer = self._find_rule_producing_adjacent_state(near_target)
            if producer:
                self._plan_for_rule(producer, steps, visited, inventory)

    # Add this rule's execution as the final step
    steps.append(PlannedStep(
        action=rule.action or "do",
        target=rule.concept,
        near=rule.concept if rule.action in ("make", "place") else None,
        rule=rule,
    ))
```

### `find_remedies`

```python
def find_remedies(self, failure: Failure) -> list[CausalLink]:
    """Query world model: which rules counteract this failure?"""
    remedies = []
    for concept in self.concepts.values():
        for link in concept.causal_links:
            if self._rule_prevents(link, failure):
                remedies.append(link)
    return remedies

def _rule_prevents(self, rule: CausalLink, failure: Failure) -> bool:
    effect = rule.effect
    if failure.kind == "var_depleted":
        # Rule increases the depleted variable?
        return effect.body_delta.get(failure.var, 0) > 0 or (
            effect.kind == "stateful"
            and effect.stateful_condition
            and effect.stateful_condition.var == failure.var
            and effect.body_delta.get(failure.var, 0) > 0
        )
    if failure.kind == "attributed_to":
        # Rule removes the causing entity?
        return effect.scene_remove == failure.cause
    return False
```

---

## 8. `HomeostaticTracker` refactor

```python
@dataclass
class HomeostaticTracker:
    # Running observed state
    observed_max: dict[str, int] = field(default_factory=dict)
    observed_rates: dict[str, float] = field(default_factory=dict)
    observation_counts: dict[str, int] = field(default_factory=dict)

    # Innate prior (loaded from textbook, immutable)
    innate_rates: dict[str, float] = field(default_factory=dict)
    reference_min: dict[str, float] = field(default_factory=dict)
    reference_max: dict[str, float] = field(default_factory=dict)
    prior_strength: int = 20

    def init_from_textbook(self, body_block: dict, rules: list[CausalLink]) -> None:
        """Load innate rates and reference bounds from textbook."""
        self.prior_strength = body_block.get("prior_strength", 20)
        for var_def in body_block.get("variables", []):
            name = var_def["name"]
            self.reference_min[name] = float(var_def.get("reference_min", 0))
            self.reference_max[name] = float(var_def.get("reference_max", 9))
            self.observed_max[name] = int(var_def.get("initial", 9))
        for rule in rules:
            if rule.kind == "passive_body_rate":
                self.innate_rates[rule.effect.body_rate_variable] = rule.effect.body_rate

    def update(
        self,
        inv_before: dict[str, int],
        inv_after: dict[str, int],
        visible_concepts: set[str],
    ) -> None:
        """Update observed_rates (running mean, not EMA) and observed_max."""
        for var in set(inv_before) | set(inv_after):
            delta = float(inv_after.get(var, 0) - inv_before.get(var, 0))
            old_rate = self.observed_rates.get(var, 0.0)
            old_count = self.observation_counts.get(var, 0)
            new_count = old_count + 1
            self.observed_rates[var] = (old_rate * old_count + delta) / new_count
            self.observation_counts[var] = new_count

            for inv in (inv_before, inv_after):
                if inv.get(var, 0) > self.observed_max.get(var, 0):
                    self.observed_max[var] = inv.get(var, 0)

    def get_rate(self, var: str) -> float:
        """Bayesian combination of innate and observed rate.
        No `visible_concepts` parameter — conditional effects now handled
        by spatial rules in ConceptStore, not here.
        """
        n = self.observation_counts.get(var, 0)
        w = self.prior_strength / (self.prior_strength + n)
        innate = self.innate_rates.get(var, 0.0)
        observed = self.observed_rates.get(var, 0.0)
        return w * innate + (1 - w) * observed

    def observed_variables(self) -> set[str]:
        return set(self.observation_counts.keys()) | set(self.innate_rates.keys())
```

**Изменения относительно текущего `HomeostaticTracker`:**

- `conditional_rates` field **удалён** — conditional эффекты теперь в ConceptStore как `passive_spatial` rules
- `get_rate(visible_concepts=...)` параметр **удалён** — spatial effects применяются в `_apply_tick` Phase 5
- EMA (`RATE_EMA_ALPHA=0.05`) заменена на **running mean** (bias-free, сходится к true rate)
- `innate_rates` и `observed_rates` теперь **разделены**, combined via Bayesian weighting (F6 closed)
- `prior_strength` загружается из textbook, default 20

---

## 9. MPC supporting functions

### `generate_candidate_plans`

```python
def generate_candidate_plans(
    state: SimState,
    store: ConceptStore,
    tracker: HomeostaticTracker,
) -> list[Plan]:
    """Baseline rollout → extract failures → find remedies → backward chain.
    Drives эмерджат из предсказанного будущего, не из hardcoded list."""
    # 1. Baseline: что будет если агент продолжит inertia?
    baseline = Plan(
        steps=[PlannedStep(action="inertia", target=None, near=None, rule=None)],
        origin="baseline",
    )
    baseline_traj = store.simulate_forward(baseline, state, horizon=40)
    candidates = [baseline]

    # 2. Extract failures from baseline
    failures = extract_failures(baseline_traj)

    # 3. For each failure, find remedies in world model
    seen_rules: set[int] = set()  # dedup by rule object id
    for failure in failures:
        remedies = store.find_remedies(failure)
        for rule in remedies:
            if id(rule) in seen_rules:
                continue
            seen_rules.add(id(rule))
            plan_steps = store.plan_toward_rule(rule, state)
            if plan_steps:
                candidates.append(Plan(steps=plan_steps, origin="remedy"))

    return candidates
```

### `extract_failures`

```python
def extract_failures(traj: Trajectory) -> list[Failure]:
    """Scan trajectory for critical events: var depletion, damage attribution."""
    failures: list[Failure] = []

    # Type 1: var depleted
    for var, series in traj.body_series.items():
        for i, value in enumerate(series):
            if value <= 0:
                failures.append(Failure(
                    kind="var_depleted",
                    var=var,
                    cause=None,
                    step=i,
                    severity=1.0,
                ))
                break  # first depletion only

    # Type 2: attributed to (damage source)
    damage_sources: dict[str, int] = {}  # source → first step of damage
    for event in traj.events:
        if event.kind == "body_delta" and event.amount < 0 and event.source != "_background":
            if event.source not in damage_sources:
                damage_sources[event.source] = event.step
    for source, first_step in damage_sources.items():
        failures.append(Failure(
            kind="attributed_to",
            var=None,
            cause=source,
            step=first_step,
            severity=1.0,
        ))

    # Sort by step ascending (earliest = highest priority)
    failures.sort(key=lambda f: f.step)
    return failures
```

### `score_trajectory`

```python
def score_trajectory(
    traj: Trajectory,
    tracker: HomeostaticTracker,
) -> tuple[int, float, int, float]:
    """Lexicographic tuple: (alive, min_body, survival_ticks, final_body).
    Higher tuple = better plan."""
    n_ticks = len(next(iter(traj.body_series.values()), []))

    catastrophic = traj.terminated and traj.terminated_reason == "body_dead"
    alive_score = 0 if catastrophic else 1

    def normalized_body_sum(step_idx: int) -> float:
        total = 0.0
        for var, series in traj.body_series.items():
            ref_max = max(tracker.reference_max.get(var, 1), 1)
            total += series[step_idx] / ref_max
        return total

    if n_ticks > 0:
        min_body = min(normalized_body_sum(i) for i in range(n_ticks))
        final_body = normalized_body_sum(n_ticks - 1)
    else:
        min_body = 0.0
        final_body = 0.0

    return (alive_score, min_body, n_ticks, final_body)
```

### `expand_to_primitive`

```python
def expand_to_primitive(
    step: PlannedStep,
    sim: SimState,
    store: ConceptStore,
) -> str:
    """Transform a symbolic PlannedStep into a single primitive env action."""
    if step.action == "inertia":
        return sim.last_action or "move_right"

    target = step.target

    if step.action == "do":
        if _nearest_concept(sim, store) == target:
            return "do"
        target_pos = sim.spatial_map.find_nearest(target, sim.player_pos) if target else None
        if target_pos is not None:
            return _step_toward_pos(sim.player_pos, target_pos)
        return "move_right"

    if step.action == "place":
        effect = step.rule.effect if step.rule else None
        item = effect.world_place[0] if effect and effect.world_place else target
        if _nearest_concept(sim, store) == "empty":
            return f"place_{item}"
        return "move_right"

    if step.action == "make":
        near_target = step.near or (step.rule.concept if step.rule else target)
        if _nearest_concept(sim, store) == near_target:
            result_item = _extract_result_item(step.rule)
            return f"make_{result_item}"
        target_pos = sim.spatial_map.find_nearest(near_target, sim.player_pos) if near_target else None
        if target_pos is not None:
            return _step_toward_pos(sim.player_pos, target_pos)
        return "move_right"

    if step.action == "sleep":
        return "sleep"

    return "move_right"
```

---

## 10. `_apply_tick` — 6-phase dispatch

```python
def _apply_tick(
    self,
    sim: SimState,
    primitive: str,
    planned_step: PlannedStep,
    traj: Trajectory,
    tick: int,
) -> None:
    """One tick of simulation in fixed phase order."""

    # === Phase 1: Dynamic entities move ===
    for entity in sim.dynamic_entities:
        move_rule = self._get_movement_rule(entity.concept_id)
        if move_rule and move_rule.confidence >= 0.1:
            entity.pos = _apply_movement(
                entity.pos,
                sim.player_pos,
                move_rule.effect.movement_behavior,
            )

    # === Phase 2: Player moves ===
    if primitive.startswith("move_"):
        sim.player_pos = _apply_player_move(sim.player_pos, primitive)

    # === Phase 3: Background body rates ===
    for rule in self._body_rate_rules():
        if rule.confidence < 0.1:
            continue
        var = rule.effect.body_rate_variable
        rate = rule.effect.body_rate
        sim.body[var] = sim.body.get(var, 0.0) + rate
        traj.events.append(SimEvent(
            step=tick,
            kind="body_delta",
            var=var,
            amount=rate,
            source="_background",
        ))

    # === Phase 4: Stateful rules ===
    for rule in self._stateful_rules():
        if rule.confidence < 0.1:
            continue
        if not rule.effect.stateful_condition.satisfied(sim):
            continue
        for var, delta in rule.effect.body_delta.items():
            sim.body[var] = sim.body.get(var, 0.0) + delta
            traj.events.append(SimEvent(
                step=tick,
                kind="body_delta",
                var=var,
                amount=delta,
                source=f"stateful:{rule.effect.stateful_condition.var}",
            ))

    # === Phase 5: Spatial rules (adjacency) ===
    for entity in sim.dynamic_entities:
        for rule in self._spatial_rules_for(entity.concept_id):
            if rule.confidence < 0.1:
                continue
            if _manhattan(entity.pos, sim.player_pos) > rule.effect.spatial_range:
                continue
            for var, delta in rule.effect.body_delta.items():
                sim.body[var] = sim.body.get(var, 0.0) + delta
                traj.events.append(SimEvent(
                    step=tick,
                    kind="body_delta",
                    var=var,
                    amount=delta,
                    source=entity.concept_id,
                ))

    # === Phase 6: Action-driven effects ===
    if primitive == "do":
        near = _nearest_concept(sim, self)
        rule = self._find_do_rule(near, sim.inventory)
        if rule and rule.confidence >= 0.1:
            _apply_effect_to_sim(sim, rule.effect, self)
            traj.events.append(SimEvent(
                step=tick,
                kind="rule_applied",
                var=None,
                amount=0.0,
                source=f"do:{near}",
            ))
    elif primitive.startswith("place_"):
        item = primitive[len("place_"):]
        rule = self._find_place_rule(item, sim.inventory)
        if rule and rule.confidence >= 0.1:
            _apply_effect_to_sim(sim, rule.effect, self)
            traj.events.append(SimEvent(
                step=tick,
                kind="rule_applied",
                var=None,
                amount=0.0,
                source=f"place:{item}",
            ))
    elif primitive.startswith("make_"):
        item = primitive[len("make_"):]
        rule = self._find_make_rule(item, sim.inventory)
        if rule and rule.confidence >= 0.1:
            _apply_effect_to_sim(sim, rule.effect, self)
            traj.events.append(SimEvent(
                step=tick,
                kind="rule_applied",
                var=None,
                amount=0.0,
                source=f"make:{item}",
            ))
    elif primitive == "sleep":
        rule = self._find_sleep_rule()
        if rule and rule.confidence >= 0.1:
            _apply_effect_to_sim(sim, rule.effect, self)

    sim.last_action = primitive
    sim.step = tick + 1
```

**Порядок фаз — topological**, не вкусовой:

- (1) entities move, (2) player moves → позиции updated **перед** adjacency check
- (3) background rate **перед** (4) stateful → stateful conditions проверяются на обновлённом body
- (5) spatial damage **перед** (6) action → агент получает damage одновременно с убийством врага (не «action first», который был бы cheat)
- (6) action effects последними → trace событий отражает «вот как state был до моего действия, вот как я среагировал»

---

## 11. Confidence handling (MVP)

### Threshold filter, not probabilistic weighting

Все rule applications в `_apply_tick` проверяют `if rule.confidence < 0.1: continue`. Rules с confidence ≥ 0.1 fires детерминистически с полной magnitude.

**Почему не probabilistic:** дробные inventory deltas семантически бессмысленны (половина дерева не имеет смысла), а Monte Carlo multiple rollouts утраивают compute. Binary threshold — pragmatic compromise для MVP.

**Roadmap:** в 77b+ possible replacement:
- Для body_delta: continuous weighting `delta * confidence`
- Для inventory_delta: stochastic application с rng
- Или: multi-rollout averaging

### Confidence update — writer side unchanged

`verify_outcome` → `ConceptStore.verify` → `link.confidence += ±0.15` как раньше. Единственное изменение: теперь confidence **читается** в `_apply_tick` и в `simulate_forward`, значит feedback loop замкнут (F5 closed).

`CONFIRM_DELTA = REFUTE_DELTA = 0.15` остаются как learning rate constants в коде. Документируются как tunable если learning окажется слишком быстрым/медленным.

---

## 12. Deletion plan and staged commits

Каждый commit должен оставлять main в зелёном состоянии (pytest passes + exp135 regression not broken).

### Commit 1: `add: Stage 77a data types`
- Новый файл `src/snks/agent/forward_sim_types.py` (или аналог)
- Dataclass'ы: `SimState`, `Trajectory`, `SimEvent`, `DynamicEntity`, `Failure`, `Plan`, `PlannedStep`, `RuleEffect`, `StatefulCondition`
- Ничего ещё не используется — только определения
- Unit-тесты на конструкторы, equality

### Commit 2: `add: Stage 77a textbook YAML grammar + parser`
- `crafter_textbook.py` переписан под dict dispatch
- `CausalLink` получает `effect: RuleEffect | None` и `kind: str` fields; `result: str` помечен DEPRECATED (сохранён для backwards compat)
- `configs/crafter_textbook.yaml` переписан в structured формат
- Parser распознаёт **оба** формата (новый dict + старый regex fallback с deprecation warning)
- Тесты Stage 71 обновлены под new YAML

### Commit 3: `refactor: HomeostaticTracker — innate/observed split`
- Новая сигнатура tracker'а: `innate_rates`, `observed_rates`, `observation_counts`, `prior_strength`
- `init_from_textbook` вместо `init_from_body_rules`
- `get_rate(var)` без `visible_concepts`
- Удалены `conditional_rates`, `_initialized` flag, `RATE_EMA_ALPHA`
- Callers обновлены

### Commit 4: `add: Stage 77a ConceptStore methods`
- Три новых метода: `simulate_forward`, `plan_toward_rule`, `find_remedies`
- `ConceptStore.plan(goal_id)` **сохранён** как deprecated stub, который внутри вызывает `plan_toward_rule`
- Unit тесты: каждая фаза `_apply_tick` изолированно + integration

### Commit 5: `add: Stage 77a MPC loop`
- Новый файл `src/snks/agent/mpc_agent.py` с `run_mpc_episode`, `generate_candidate_plans`, `extract_failures`, `score_trajectory`, `expand_to_primitive`
- **Не трогаем** старый `continuous_agent.py` — параллельное существование
- Integration тест: короткий smoke на реальном Crafter

### Commit 6: `add: experiments/exp137_mpc_forward_sim.py`
- Новый эксперимент по пайплайну exp136, использует `run_mpc_episode`
- Локальный smoke run проходит
- **Запуск на minipc** отдельно, не часть commit'а

### Commit 7: `remove: Stage 76 EpisodicSDM substrate`
(После подтверждения что exp137 хотя бы работает на minipc, даже если gate ещё не взят)
- Удалить `src/snks/memory/` целиком
- Удалить Stage 76 tests (5 файлов)
- Удалить импорты в `continuous_agent.py`

### Commit 8: `remove: select_goal, HOMEOSTATIC_VARS, dead perception code`
(После взятия gate или после обоснованного решения продолжать без него)
- Удалить `HOMEOSTATIC_VARS` const + все 5 вхождений
- Удалить `select_goal`, `compute_drive`, `compute_curiosity`, `get_drive_strengths`
- Удалить `_STAT_GAIN_TO_NEAR`, `perceive_field`, `perceive` (legacy), `ground_*`, `on_action_outcome`, `should_retrain`, `retrain_features`, `babble_probability`, `explore_action`
- Удалить `ConceptStore.plan(goal_id)` deprecated stub
- Удалить `CausalLink.result` field
- Удалить старый `continuous_agent.run_continuous_episode` + local `perceive_tile_field` копию
- Удалить regex fallback в `crafter_textbook.py`

### Commit 9: `remove: experiments/exp136_continuous_learning.py`
- Старый эксперимент удалён
- Финальный pytest + regression

**Примерный объём:**
- Удалено: ~2000 строк (memory package, dead perception, Stage 76 tests, select_goal branches)
- Добавлено: ~1200 строк (new data types, simulate_forward, MPC loop, parser, tests)
- **Net: -800 строк**

---

## 13. Tests

### Unit tests

| Файл | Что тестируем |
|---|---|
| `tests/test_stage77_types.py` | `RuleEffect`, `SimState`, `Trajectory` constructors, equality, `StatefulCondition.satisfied` для всех операторов |
| `tests/test_stage77_parser.py` | Каждый из типов правил (action_triggered × 4, passive × 4) парсится в корректный `CausalLink` |
| `tests/test_stage77_concept_store.py` | `find_remedies` для `var_depleted` и `attributed_to`; `plan_toward_rule` backward chain для gather/craft/combat |
| `tests/test_stage77_simulate.py` | Каждая фаза `_apply_tick` в изоляции + integration на 20 шагов |
| `tests/test_stage77_tracker.py` | Innate/observed split, `prior_strength` из textbook, running mean, `get_rate` combination |
| `tests/test_stage77_scoring.py` | `score_trajectory` ordering (alive > dead, min body, survival, final); `extract_failures` на синтетических trajectories |
| `tests/test_stage77_mpc.py` | `generate_candidate_plans` — baseline + remedies; `run_mpc_episode` integration smoke на mock env |

### Integration tests

- `tests/test_stage77_integration.py` — короткий эпизод на реальном Crafter (10 eps, max_steps=200), проверка что нет exceptions и final state sensible

### Regression tests

- `tests/test_stage75.py` — без изменений, должен проходить (tile segmenter не трогается)
- `tests/test_stage71.py`, `test_stage72.py`, `test_stage73.py`, `test_stage74.py` — обновлены под `plan_toward_rule` либо удалены
- Stage 76 tests (5 файлов) удалены в Commit 7

### Performance benchmark

- `benchmarks/bench_simulate_forward.py` — 100 iterations горизонта 20, измерение p50/p99 на CPU

### Ideology lint

- `tests/lint_ideology_77a.py` — автоматический scan на forbidden patterns:
  - `HOMEOSTATIC_VARS\s*=`
  - `for\s+\w+\s+in\s+\{["']\w+["'],` (hardcoded entity list iteration in policy files)
  - `if\s+\w+\s*==\s*["']\w+["']` с entity names в decision code
  - `link\.result\s*==\s*["']\w+["']`
  - `if\s+\w+\s*[<>]\s*-?\d+\.\d+` (магические пороги на rate)
  - `_STAT_GAIN_TO_NEAR|preparation_urgency`
- Scope: `perception.py`, `concept_store.py`, `mpc_agent.py`, `crafter_textbook.py`, `forward_sim_types.py`
- Exit code non-zero если найдено хоть одно нарушение

---

## 14. Gates

**Gate 1 (primary wall-breaker):** `survival ≥ 200`
- 3 eval runs × 20 episodes × max_steps=1000 на minipc
- Все три run means ≥200, overall mean ≥200

**Gate 2 (regression protection):** `wood ≥ 50% smoke`
- 20 smoke episodes без врагов, max_steps=200
- ≥10/20 достигают wood ≥3

**Gate 3 (perception unchanged):** `tile_accuracy ≥ 80%`
- Перегон Stage 75 eval
- Защищает от случайных ломок cnn_encoder dependencies

**Gate 4 (ideology lint):** `0 forbidden patterns`
- `lint_ideology_77a.py` возвращает 0
- Критически шире чем Stage 76 Gate 5 (скан всего живого пути, не только memory/)

**Gate 5 (forward sim performance):** `p99 decision latency ≤ 100ms`
- `bench_simulate_forward.py` на minipc CPU
- 100 iterations, p99 измерен

**Gate 6 (feedback loops closed, unit-level):**
- `test_confidence_filter_affects_rollout` passes
- `test_innate_prior_survives_observation` passes (после 1000 observations innate rate всё ещё contributes per formula)
- `test_plan_toward_rule_backward_chains` passes для combat rule
- `test_find_remedies_queries_world_model` passes

**Gate 7 (correctness):** `pytest green`
- Полный suite проходит
- exp135 regression
- exp137 smoke (5 episodes)

---

## 15. Risks and mitigations

**Risk 1: exp137 на minipc не достигает survival ≥200.**

Mitigation (3-attempt rule, из `feedback_techdebt_pipeline.md`):
1. Attempt 1 — анализ exp137 логов, найти где forward sim неверно предсказывает. Fix stateful/movement rules или scoring.
2. Attempt 2 — если fix #1 не помог, добавить/уточнить дополнительные правила в textbook.
3. Attempt 3 — попробовать `horizon 20 → 40`, beam search на candidates, или больше планов кандидатов.
4. **После 3 attempts** — STOP, возврат к brainstorm. Возможно 77b (surprise→rule) нужно раньше.

**Risk 2: Compute simulate_forward слишком медленный.**

Mitigation: бенчмарк в Commit 5 (unit benchmark). Если >100 ms per decision:
- Reduce horizon (20 → 10)
- Beam на candidates (top-3 вместо всех)
- NumPy vectorize phases
- Worst case: переход на Cython для hot path

**Risk 3: Breaking change в `crafter_textbook.yaml` формате сразу ломает всё что его загружает.**

Mitigation: в Commit 2 парсер распознаёт оба формата. Старый regex помечен deprecated. Удаляется в Commit 8 когда все тесты переписаны.

**Risk 4: Тесты Stage 71-75, использующие `ConceptStore.plan(string)`, ломаются в Commit 4.**

Mitigation: deprecated stub `ConceptStore.plan(goal_id)` вызывает `plan_toward_rule` внутри. Stub удаляется в Commit 8.

**Risk 5: `tile_segmenter.py` случайно ломается при рефакторинге dependencies.**

Mitigation: `tile_segmenter.py` явно **не трогается**. Любое изменение в этом файле — red flag в code review.

---

## 16. Generalization notes — что переносится между Crafter / Minecraft / AGI

### Generalizes к другим доменам

- **Three category taxonomy** (facts / mechanisms / experience) — universal
- **One-goal MPC loop** — applies to any homeostatic agent
- **Baseline rollout → failures → remedies** pattern — works in any world model with causal rules
- **Structured YAML grammar** — extensible to any new rule type (add new `passive:` or `action:` discriminator)
- **`RuleEffect` structured dispatch** — works for any domain with inventory/body/entity state
- **Lexicographic tuple scoring** — independent of which body vars exist
- **Innate/observed Bayesian weighting** — универсальный подход к prior integration
- **Confidence threshold filter** — applies everywhere (with TODO for probabilistic)
- **Staged commits рефакторинга** — стандарт

### Crafter-specific, нуждаются в adaptation

- **Movement behaviors enum** (`chase_player`, `flee_player`, `random_walk`) — Minecraft может добавить `patrol`, `stealth`, etc. Расширение тривиально: новый `behavior` string + новая функция в `_apply_movement`
- **Spatial = grid Manhattan** — 2D grid assumption. Minecraft (3D grid) — замена metric на 3D manhattan. AGI с continuous space — отдельное переосмысление
- **Tick-discrete time** — AGI с continuous dynamics требует другого sim paradigm
- **Action enum `do | make | place | sleep`** — Crafter/Minecraft specific. AGI с другим action space — расширение discriminator
- **Textbook path `configs/crafter_textbook.yaml`** — очевидно per-domain

### Где сейчас заложены минимальные extension points

- `reference_min` / `reference_max` per body var (не хардкод 0/9)
- `prior_strength` из textbook (не константа в коде)
- `RuleEffect` — discriminated union, новые типы добавляются как `kind` string
- `_apply_tick` phases — можно вставить новую phase между существующими без ломания порядка

### Где явно **не** генерализировано (YAGNI + documented)

- Probabilistic confidence-weighted rollouts → 77b+
- Per-var `prior_strength` → сейчас global
- Change-point detection на observed rates (для non-stationary env) → отдельный механизм
- Task goals (external objectives сверх homeostasis) → отдельный scoring component
- Empowerment / intrinsic motivation → 77b или дальше
- `direction: grow | shrink` на body vars → сейчас все higher-better
- Multi-agent → не в scope

---

## 17. Out of scope (deferred с обоснованием)

| Что | Причина deferring | Куда |
|---|---|---|
| Surprise → new rule mechanism | Crafter well-known домен; textbook может быть достаточно полным. 77a MVP без novelty источника приоритет тестируется первым | Stage 77b |
| Dead code cleanup (60+ legacy модулей в `src/snks/agent/`) | Не блокирует wall-breaking; увеличит diff и риск коммитов | Stage 77c |
| Spatial map с confidence (F10) | Не критично для wall-breaking; tile segmenter уже даёт приемлемый noise floor | Stage 77c |
| Probabilistic confidence-weighted rollouts | Binary threshold достаточен для MVP; probabilistic требует multi-rollout | 77b+ |
| Full YAML schema validation | YAGNI для 7 rule types; добавляется когда rule count вырастет | 77b+ |
| CNN encoder дообучение из опыта | Out of 77a scope; сегментер frozen после Stage 75 | 77c+ или отдельный stage |
| DAF / SKS / GWS / metacog компоненты | Ideology deferred; не в живом пути | Отдельный research track |
| MiniGrid regression | Stage 63 завершил; 77a не трогает MiniGrid path | — |
| Task goals (external objectives) | Outside «maintain body» scoring; нужен отдельный scoring component | Отдельный stage когда появятся задачи |

---

## 18. Assumptions to document in `docs/ASSUMPTIONS.md`

После реализации 77a добавить section:

**Stage 77a — Forward simulation через ConceptStore**

- Forward simulation через causal rules, tick-discrete time, 20-шаговый horizon
- Navigation в simulator approximates через Manhattan distance (не полный pathfinder)
- Confidence threshold binary 0.1, deterministic rule firing (not probabilistic)
- Spatial range defaults to 1 (adjacent), может быть расширен до range=N в textbook
- Innate body prior стабилен через Bayesian combination с `prior_strength=20` (из textbook)
- `observed_rates` через running mean, не EMA — assumes stationary environment
- One singular goal: maintain body. No task goals, no empowerment reward
- Crafter-specific: rules для skeleton bow range упрощены до adjacent (spatial_range=1), не range-5
- Spatial map snapshot в rollout не обновляется (нельзя «узнать новое» в воображении)
- `DynamicEntity` tracking через `DynamicEntityTracker` — отдельно от `CrafterSpatialMap` (семантика разная: спатиальная карта для статичных объектов, трекер для движущихся)

---

## 19. Связь с findings из Phase A

| Finding | Адресовано в | Механизм |
|---|---|---|
| F1 (two world models) | § 4, 10, 12 | Один MPC loop, удаление `src/snks/memory/` |
| F2 (active interference) | § 4, 10 | Один scoring criterion (tuple), не два конкурирующих |
| F3 (`HOMEOSTATIC_VARS` hardcoded) | § 12 (commit 8) | Удалено, заменено `tracker.observed_variables()` |
| F4 (hardcoded strategies) | § 9, 12 | `select_goal` удалён; `generate_candidate_plans` через baseline+remedies |
| F5 (confidence loop dead) | § 10, 11 | `_apply_tick` читает confidence, threshold filter замыкает loop |
| F6 (body prior decays) | § 8 | Innate/observed split + Bayesian weighting |
| F7 (textbook lies) | § 6 | Stateful rules `food > 0 restores health` вместо прямого `do cow restores health` |
| F8 (nouns/verbs confusion) | § 6, 7 | `RuleEffect.scene_remove` вместо `kill_zombie` pseudo-concept |
| F9 (next_state_sdr dead) | § 12 (commit 7) | Удалено вместе с `src/snks/memory/` |
| F10 (spatial_map noise) | — | **Deferred в 77c** (не критично для wall-breaking) |
| F11 (three perceive functions) | § 12 (commit 8) | Консолидировано в `perception.perceive_tile_field` |
| F12 (dead code perception.py) | § 12 (commit 8) | Удалено |
| F13 (60+ legacy agent modules) | — | **Deferred в 77c** |
| F14 (mis-diagnosis Stage 76) | § 1 | Reformulated в этом документе |
| F15 (Stage 77 wrong forecast) | — | Stage 77a — reformulated approach |

---

## 20. Next step — implementation plan

После approval этого spec'а следующий шаг — создание **implementation plan** через `writing-plans` skill. Implementation plan развернёт каждый commit (1-9) в конкретные tasks, файлы, строки, acceptance criteria. Это отдельный документ в `docs/superpowers/plans/`.

Implementation plan **не** входит в scope этого spec'а.
