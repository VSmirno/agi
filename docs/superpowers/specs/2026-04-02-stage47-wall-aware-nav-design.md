# Stage 47: Wall-aware навигация

**Дата:** 2026-04-02
**Milestone:** M1 — Генерализация
**Gate:** ≥80% DoorKey-5x5 с walls в разных позициях (200 random layouts)
**Зависимости:** Stage 46 (SubgoalPlanningAgent, 92.5% на fixed layout)

---

## Проблема

Stage 46 SubgoalNavigator использует heuristic навигацию (`_navigate_toward`): поворот к цели → forward. Это не учитывает стены — агент упирается в стену и застревает. На fixed DoorKey-5x5 layout это работает (92.5%), потому что расположение предсказуемо, но на random layouts с различными конфигурациями стен — сломается.

## Позиция в фазе

**M1 Gate:** ≥80% random DoorKey-5x5, ≥60% MultiRoom-N3

Stage 47 продвигает первый маркер M1: генерализация навигации на произвольные layouts. Без wall-aware navigation невозможно перейти к Stage 48 (random layouts) и Stage 49 (multi-room).

---

## Подходы

### A: BFS pathfinding на observed grid ✓ ВЫБРАН
- Извлечь карту стен из 7x7 observation (obj_type == 2)
- BFS от текущей позиции агента к target позиции
- Конвертировать BFS path в последовательность действий (turn + forward)
- **Trade-off:** простой, гарантированно оптимальный, zero-failure на passable layouts. Не bio-plausible (но навигация — не core cognitive claim СНКС).

### B: A* pathfinding
- То же что BFS, но с Manhattan heuristic
- **Trade-off:** на 5x5 grid разницы нет (BFS уже O(25)). Лишняя сложность без выигрыша.

### C: SDM-based learned navigation
- Использовать SDM transition predictions для implicit wall avoidance
- **Trade-off:** bio-plausible, но SDM prediction quality 0.85 — недостаточно для reliable navigation. Застрянет на стенах.

**Решение: A (BFS).** Навигация — инфраструктура, не когнитивная функция. BFS гарантирует оптимальный путь, что позволяет изолированно тестировать когнитивные компоненты (subgoal extraction, planning) без noise от плохой навигации.

---

## Дизайн

### 1. GridPathfinder (новый класс)

```python
class GridPathfinder:
    """BFS pathfinding on MiniGrid observation grid."""
    
    def extract_walls(self, obs: np.ndarray) -> set[tuple[int, int]]
    def find_path(self, obs: np.ndarray, start: tuple[int, int], 
                  goal: tuple[int, int], allow_door: bool = False) -> list[tuple[int, int]] | None
    def path_to_actions(self, path: list[tuple[int, int]], 
                        current_dir: int) -> list[int]
```

- `extract_walls`: scan obs for obj_type == 2 (wall) и obj_type == 4 with state == 2 (locked door)
- `find_path`: BFS on grid, returns list of (row, col) positions
- `path_to_actions`: converts path to MiniGrid actions (0=left, 1=right, 2=forward)
- `allow_door`: if True, treat door as passable (for after opening)

### 2. RandomDoorKeyEnv (новый класс)

```python
class RandomDoorKeyEnv:
    """DoorKey-5x5 with randomized layout per episode."""
    
    def __init__(self, size: int = 5, seed: int | None = None)
    def reset(self, seed: int | None = None) -> np.ndarray
    def _generate_layout(self) -> None  # randomize: wall_row, door_col, key/agent/goal positions
```

Рандомизация:
- Wall-divider row: random из {1, 2, 3} (не крайние)
- Door position: random column в wall-divider
- Key: random position в upper half (выше стены)
- Agent start: random position в upper half (не на key)
- Goal: random position в lower half (ниже стены)

### 3. Обновление SubgoalNavigator

Заменить `_navigate_toward` на BFS-based навигацию:
- При `select()`: если есть target position, использовать GridPathfinder
- Кэшировать path (пересчитывать при изменении grid state, e.g. door opened)
- Обработать случай "path не найден" (fallback на random)

### 4. Обновление SubgoalPlanningAgent

- `_extract_target_positions` должен работать на random layouts (позиции из trace, не хардкод)
- Уже работает — positions извлекаются из trace steps

---

## Эксперименты

| Exp | Что | Gate | Тип |
|-----|-----|------|-----|
| 107a | BFS pathfinding unit tests (wall avoidance) | 100% paths found | CPU |
| 107b | Random layouts — SubgoalPlanningAgent | ≥80% success на 200 random layouts | CPU |
| 107c | Comparison: BFS nav vs heuristic nav on random | BFS > heuristic | CPU |

---

## Файлы

| Файл | Изменение |
|------|-----------|
| `src/snks/agent/pathfinding.py` | NEW: GridPathfinder |
| `src/snks/agent/subgoal_planning.py` | UPDATE: SubgoalNavigator uses GridPathfinder |
| `src/snks/experiments/exp107_wall_aware_nav.py` | NEW: RandomDoorKeyEnv + experiments |
| `tests/test_stage47_wall_nav.py` | NEW: unit tests |
| `demos/stage-47-wall-aware-nav.html` | NEW: web demo |
