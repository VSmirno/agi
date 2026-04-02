# Stage 56: Complex Environment — BabyAI PutNext

## Результат: PASS

**Ветка:** `stage56-complex-environment`
**Milestone:** M4 — Масштаб (третий этап)

## Что доказано
- Агент решает BabyAI PutNext с 18 типами объектов (3 типа × 6 цветов)
- PutNextLocalS5N3 (5×5): 100% на 200 random seeds, mean 7.8 steps
- PutNextS6N3 (11×6, 6 объектов): 99.5% на 200 seeds, mean 14.8 steps
- 5-фазный state machine: EXPLORE → GOTO_SOURCE → PICKUP → GOTO_TARGET → DROP
- Partial obs (7×7) работает через SpatialMap + FrontierExplorer
- Mission parsing: regex → (source_type, source_color, target_type, target_color)

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 110a | PutNextLocalS5N3 (200 seeds) | 200/200 = 100%, mean 7.8 steps | ≥50% | PASS |
| 110b | PutNextS6N3 (200 seeds) | 199/200 = 99.5%, mean 14.8 steps | ≥50% | PASS |
| 110c | Unique object types | 18 (ball/box/key × 6 colors) | ≥5 | PASS |

## Ключевые решения

1. **Symbolic BFS + Mission Parsing** — расширение proven infrastructure из Stages 54-55. Learning-based подход отложен (Stage 59).
2. **Carried object clears from map** — MiniGrid показывает тип несомого объекта на позиции агента в partial obs. Fix: явно очищаем клетку агента в SpatialMap при carrying.
3. **Drop plan = (stand_cell, drop_cell, face_dir)** — 3-step планирование: найти пустую клетку рядом с целью, найти позицию для агента, повернуться и сбросить. Первоначальный подход "стоять рядом с целью" не работал из-за геометрических ограничений.
4. **SpatialMap extension** — `find_object_by_type_color()` для дискриминации по типу И цвету (vs только по типу в Stage 54).

## Веб-демо
- `demos/stage-56-putnext.html` — Canvas replay 5 эпизодов (PutNextLocal + PutNextS6N3), 5-фазная визуализация

## Файлы изменены
- `src/snks/agent/putnext_agent.py` — NEW: MissionParser, PutNextAgent, PutNextEnv
- `src/snks/agent/spatial_map.py` — UPDATE: find_object_by_type_color, find_all_objects
- `tests/test_stage56_complex_env.py` — NEW: 19 тестов
- `src/snks/experiments/exp110_putnext.py` — NEW: gate experiments
- `demos/stage-56-putnext.html` — NEW: web demo
- `demos/index.html` — UPDATED: карточка Stage 56

## Следующий этап
- **Stage 57: Long Subgoal Chains** — задачи с 5+ subgoals. Gate: ≥40%.
