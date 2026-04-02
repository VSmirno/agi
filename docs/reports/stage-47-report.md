# Stage 47: Wall-aware навигация

## Результат: PASS

**Ветка:** `stage47-wall-aware-nav`
**Merge commit:** pending

## Что доказано

- GridPathfinder (BFS) находит оптимальный путь на 100% из 200 случайных раскладок DoorKey-5x5
- RandomDoorKeyEnv генерирует корректные, решаемые раскладки с рандомизацией wall_row, door, key, agent, goal
- SubgoalNavigator с BFS обходит стены вместо heuristic навигации Stage 46
- **100% success rate на 200 random layouts** (gate ≥80%) — mean 16 steps
- Obs-based planning: агент строит план прямо из наблюдения, без explore phase
- Средняя длина BFS пути: 9.6 шагов (на 5x5 grid)

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 107a | BFS pathfinding (200 layouts) | 100% | 100% | PASS |
| 107b | Random DoorKey-5x5 (200 layouts, 50 eps each) | 100% (200/200 layouts, mean 16 steps) | ≥80% layouts at ≥80% | PASS |

## Ключевые решения

1. **BFS вместо A*** — на 5x5 grid разницы нет, BFS проще и гарантированно оптимален
2. **Навигация = инфраструктура** — не bio-plausible, но позволяет изолированно тестировать когнитивные функции (subgoal extraction, planning)
3. **Obs-based planning** — `build_plan_from_obs` сканирует observation на key/door/goal позиции, строит subgoal chain без explore phase. Это ключевое улучшение: random explore на DoorKey-5x5 имеет ~1% success rate, что ненадёжно на random layouts
4. **Пересчёт BFS на каждом шаге** — вместо кэширования, т.к. BFS на 7x7 <1ms
5. **Rebuild plan every episode** — `build_plan_from_obs` вызывается каждый reset, т.к. RandomDoorKeyEnv генерирует новую раскладку
6. **Epsilon 0.05** — 5% random actions: достаточно для небольшого exploration, не мешает critical actions (pickup, toggle)

## Веб-демо
- `demos/stage-47-wall-aware-nav.html` — Canvas с 8+ раскладками, BFS path overlay, agent trail, subgoal chain

## Файлы изменены
- `src/snks/agent/pathfinding.py` — NEW: GridPathfinder (BFS)
- `src/snks/agent/subgoal_planning.py` — UPDATE: SubgoalNavigator с BFS, build_plan_from_obs
- `src/snks/experiments/exp107_wall_aware_nav.py` — NEW: RandomDoorKeyEnv + experiments
- `tests/test_stage47_wall_nav.py` — NEW: 26 тестов
- `demos/stage-47-wall-aware-nav.html` — NEW: веб-демо
- `docs/superpowers/specs/2026-04-02-stage47-wall-aware-nav-design.md` — спецификация

## Следующий этап

**Stage 48: Random layouts** — Stage 47 уже достигает ≥80% на 200 random layouts (100%!), что формально закрывает M1 Stage 48 gate. Stage 48 может быть объединён или сфокусирован на edge cases: layouts с очень длинными путями, несколько стеновых рядов, или partial observability.

Альтернативно, если 100% результат стабилен, можно перейти к **Stage 49: Multi-room** (≥60% MultiRoom-N3), что закроет M1 полностью.
