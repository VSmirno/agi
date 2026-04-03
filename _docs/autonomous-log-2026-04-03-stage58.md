# Autonomous Development Log — 2026-04-03 (Stage 58)

## Текущая фаза: M4 — Масштаб, прогресс ~71% (5/7 stages)

## Stage 58: SDM Retrofit — Learned Agent for DoorKey

### Architectural Integrity Check — Stage 58
- DAF: не задействован — R1 негативный вердикт
- VSA: **задействован** — AbstractStateEncoder → 256-dim binary vector
- SDM: **задействован** — 1000 locations, subgoal-level transitions
- Planner: **задействован** — SDM subgoal selection
- Learning phase: **есть** — 50 exploration episodes, 1453+ transitions
- Тип агента: **hybrid** (learned subgoal selection + symbolic navigation)
- Drift counter: **RESET → 0** (11 этапов drift broken)
- Статус: OK

### Фаза 1: Спецификация
- Pipeline: obs → SpatialMap → AbstractStateEncoder (VSA) → SDM → subgoal → BFS nav
- Gate: ≥30% DoorKey-5x5 partial obs + ≥1000 SDM transitions
- Решения пользователя: DoorKey only, frontier exploration, gate 30%

### Фаза 2: Реализация
- AbstractStateEncoder: compact features (has_key, door_state, distances)
- SDMDoorKeyAgent: exploration → SDM recording → planning
- Pipeline Compliance: COMPLIANT (VSA + SDM)
- 14 тестов PASS на minipc

### Итерации planning:
- **v1 (raw action SDM)**: 18% vs 31% random = WORSE than random. SDM не может планировать low-level actions.
- **v2 (subgoal SDM)**: 100% — SDM выбирает WHAT (key/door/goal), BFS делает HOW. Работает.

### Фаза 2.5: Learning Budget
- 50 exploration episodes → 1453 SDM writes → 7751 total writes after eval
- Exploration: 100% success (frontier + reflexes эффективны)
- Training time: 0.7s на CPU

### Фаза 3: Эксперименты
- **112a (50ep):** 100% (200/200), SDM 7751 writes, baseline 88.5% — PASS
- **112b (100ep):** 100% (200/200), SDM 8783 writes, baseline 88.5% — PASS
- ROCm GPU segfault → forced CPU, dim=256, locations=1000

### Проблемы обнаруженные
1. ROCm 6.1 ↔ GPU mismatch → segfault → forced CPU
2. Raw action SDM worse than random → pivoted to subgoal-level SDM
3. Heuristic fallback = 88.5% → SDM improvement marginal (+11.5%)

### ROADMAP обновлён
- Stages 47-57 помечены ⚠️ SYMBOLIC BASELINE
- Добавлен Learned Pipeline Retrofit Plan
- Stage 58 = first hybrid/learned stage

### Решения
- Subgoal-level SDM вместо raw action SDM — critical pivot, без этого SDM деградирует performance
- Honest reporting: 88.5% baseline vs 100% learned → SDM adds +11.5%, navigation still symbolic
- ROCm fix отложен в tech debt
