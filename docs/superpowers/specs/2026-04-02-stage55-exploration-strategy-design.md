# Stage 55: Exploration Strategy — Design Spec

**Дата:** 2026-04-02
**Milestone:** M4 — Масштаб
**Gate:** ≥60% success на MultiRoom-N3 с 7x7 partial obs
**Зависимости:** Stage 54 (SpatialMap, FrontierExplorer), Stage 49 (MultiRoom navigation)

---

## Позиция в фазе

M4 Stage 2/7. Продвигает маркер "exploration strategy" — агент исследует большие среды (25x25) с partial obs и находит цель через двери.

---

## Проблема

Stage 54 работает на 5x5 DoorKey (7x7 view покрывает почти весь grid). MultiRoom-N3 = 25x25 grid, 3 rooms, 2 doors (closed, not locked). 7x7 view = ~2% grid. Нужна эффективная exploration strategy.

---

## Решение: MultiRoomPartialObsAgent

Переиспользует SpatialMap + FrontierExplorer из Stage 54. Адаптации:
1. Grid size 25x25 (вместо 5x5)
2. Нет ключей — только doors (toggle to open) + goal
3. MultiRoom doors are closed (state=1), not locked (state=2) — toggle без key

### Алгоритм:
1. Update SpatialMap с текущим 7x7 view
2. Если goal видит → BFS к goal
3. Если рядом с закрытой дверью → toggle
4. Иначе → frontier exploration (BFS к ближайшей неизвестной клетке)

### Env wrapper:
```python
class PartialObsMultiRoomEnv:
    """MultiRoom-N3 without FullyObsWrapper."""
    def __init__(self, n_rooms=3, max_room_size=6, max_steps=300)
    def reset(seed) -> (obs_7x7, agent_col, agent_row, agent_dir)
    def step(action) -> (obs_7x7, reward, term, trunc, agent_col, agent_row, agent_dir)
```

---

## Тест-план

- `test_multiroom_env_wrapper` — correct 7x7 obs format
- `test_spatial_map_25x25` — accumulation on large grid
- `test_agent_finds_goal_multiroom` — integration test
- `exp109`: 200 random MultiRoom-N3 с partial obs

---

## Файлы
- `src/snks/agent/partial_obs_agent.py` — UPDATE: add MultiRoomPartialObsAgent
- `tests/test_stage55_exploration.py` — NEW
- `src/snks/experiments/exp109_exploration.py` — NEW
- `demos/stage-55-exploration.html` — NEW
