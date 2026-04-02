# Stage 55: Exploration Strategy

## Результат: PASS

**Ветка:** `stage55-exploration-strategy`
**Milestone:** M4 — Масштаб (второй этап)

## Что доказано
- Агент решает MultiRoom-N3 (25x25) с 7x7 partial obs: 100% на 200 random layouts
- Frontier exploration эффективен: 3.26 новых клеток/шаг, mean 22.1 шагов до цели
- SpatialMap масштабируется: 25x25 grid (625 клеток) vs 5x5 в Stage 54
- Двери корректно обрабатываются: toggle при facing + BFS через open doors
- TD-006 (full obs shortcut) фактически закрыт для DoorKey и MultiRoom

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 109a | Exploration efficiency | 3.26 cells/step, 64.9 mean explored | — | INFO |
| 109b | MultiRoom-N3 partial obs (200 seeds) | 200/200 = 100% | ≥60% | PASS |

## Ключевые решения

1. **Doors as frontiers** — ключевой fix: SpatialMap.frontiers() не исключает двери. Без этого агент застревал в первой комнате (0 frontiers после exploration).
2. **O(n) frontier search** — одна BFS от агента вместо BFS к каждой frontier. Критично для 25x25 grid.
3. **MultiRoomPartialObsAgent** — упрощённый агент без key/door subgoal logic. Только: explore → toggle doors → navigate to goal.
4. **PartialObsMultiRoomEnv** — wrapper без FullyObsWrapper, возвращает agent position для SpatialMap.

## Веб-демо
- `demos/stage-55-exploration.html` — Canvas 25x25 grid с progressive exploration, 2 episode replays

## Файлы изменены
- `src/snks/agent/spatial_map.py` — UPDATE: doors as frontiers, O(n) BFS explorer
- `src/snks/agent/partial_obs_agent.py` — UPDATE: added MultiRoomPartialObsAgent + PartialObsMultiRoomEnv
- `tests/test_stage55_exploration.py` — NEW: 9 тестов
- `src/snks/experiments/exp109_exploration.py` — NEW: gate experiments
- `demos/stage-55-exploration.html` — NEW: web demo
- `demos/index.html` — UPDATED: добавлена карточка Stage 55

## Следующий этап
- **Stage 56: Complex Environment** — BabyAI PutNext, 5+ object types. Gate: ≥50%.
