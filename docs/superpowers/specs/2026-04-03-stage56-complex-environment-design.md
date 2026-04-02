# Stage 56: Complex Environment — BabyAI PutNext

**Дата:** 2026-04-03
**Milestone:** M4 — Масштаб (третий этап)
**Gate:** ≥50% BabyAI-PutNextS6N3-v0 (200 random seeds), 5+ уникальных типов объектов

---

## Позиция в фазе

**M4 маркеры:**
- [x] Partial observability (Stage 54)
- [x] Exploration strategy (Stage 55)
- [ ] **Complex environment 5+ object types** ← этот этап
- [ ] Long subgoal chains 5+ (Stage 57)
- [ ] SDM scaling ≥1000 (Stage 58)
- [ ] Transfer learning (Stage 59)
- [ ] Integration test (Stage 60)

Этот этап переводит агента от DoorKey/MultiRoom (key + door + goal = 3 типа) к PutNext (ball + box + key × 6 цветов = 18 типов объектов). Масштабирование object vocabulary — ключевой маркер M4.

---

## Анализ среды

### BabyAI-PutNextS6N3-v0
- **Grid:** 11×6, одна комната (без дверей)
- **Объекты:** 6 штук (3 пары), из типов {ball, box, key} × {red, green, blue, purple, yellow, grey}
- **Миссия:** "put the [color1] [type1] next to the [color2] [type2]"
- **Obs:** 7×7 partial observation (agent-centric)
- **Actions:** 7 (left, right, forward, pickup, drop, toggle, done)
- **Success:** source object dropped в клетке, смежной с target object
- **Max steps:** 288

### Ключевые отличия от DoorKey
1. **Multi-object:** 6 объектов vs 3 (key, door, goal)
2. **Object discrimination:** нужно различать по (type, color), не только по type
3. **Pickup + Drop:** две фазы манипуляции (vs pickup only в DoorKey)
4. **No doors:** проще навигация, но сложнее object handling
5. **Larger grid:** 11×6 = 66 клеток vs 5×5 = 25

---

## Архитектура

### Компоненты

```
MissionParser → (source_type, source_color, target_type, target_color)
     ↓
SpatialMap (extended) → tracks all objects by (type_id, color_id, position)
     ↓
PutNextAgent:
  Phase 1: EXPLORE — frontier exploration to find source & target
  Phase 2: GOTO_SOURCE — BFS navigate to adjacent cell of source
  Phase 3: PICKUP — face source + pickup
  Phase 4: GOTO_TARGET — BFS navigate to adjacent cell of target
  Phase 5: DROP — face empty cell next to target + drop
```

### MissionParser
Regex: `put the (\w+) (\w+) next to the (\w+) (\w+)`
→ source = (color1, type1), target = (color2, type2)
→ convert to MiniGrid IDs: type → OBJECT_TO_IDX, color → COLOR_TO_IDX

### SpatialMap extension
Existing SpatialMap already stores (obj_type, color, state) per cell.
New method: `find_all_objects()` → list of (type_id, color_id, row, col).
New method: `find_object_by_type_color(type_id, color_id)` → (row, col) or None.

### PutNextAgent state machine
- `phase`: EXPLORE → GOTO_SOURCE → PICKUP → GOTO_TARGET → DROP
- Transitions:
  - EXPLORE: both source and target found → GOTO_SOURCE
  - GOTO_SOURCE: adjacent to source → PICKUP
  - PICKUP: carrying source → GOTO_TARGET
  - GOTO_TARGET: adjacent to target, empty cell available → DROP
  - DROP: success (env returns reward)

### Navigation
- Reuse `GridPathfinder` + `FrontierExplorer` from Stage 54-55
- Navigate to **adjacent cell** of target object (can't walk onto objects)
- For DROP: find empty cell adjacent to target, navigate there, face target, drop

---

## Implementation plan

1. `MissionParser` class — regex parsing, tested with 20+ mission strings
2. Extend `SpatialMap` — `find_object_by_type_color()`, `find_all_objects()`
3. `PutNextAgent` — state machine with 5 phases
4. `PutNextEnv` wrapper — provides agent position, carrying state, mission
5. Tests: unit + integration
6. Experiment: 200 seeds on PutNextS6N3

---

## Gate criteria

| Metric | Gate | Measurement |
|--------|------|-------------|
| Success rate (PutNextS6N3, 200 seeds) | ≥50% | source dropped adjacent to target |
| Unique object types handled | ≥5 | distinct (type, color) pairs across episodes |
| Mean steps (successful) | ≤200 | steps to complete mission |

---

## Risks

1. **Partial obs on 11×6:** 7×7 view = good coverage, but objects may be outside initial view
2. **Drop placement:** finding empty cell adjacent to target, agent must face correct direction
3. **Object occlusion:** carrying object → not visible in grid, must track in agent state
