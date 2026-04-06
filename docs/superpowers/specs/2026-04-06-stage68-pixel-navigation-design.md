# Stage 68: Pixel Navigation — когнитивная карта из NearDetector

**Дата:** 2026-04-06  
**Статус:** Design  
**Зависимости:** Stage 67 (NearDetector, `checkpoints/stage67_encoder.pt`)

---

## Цель

Убрать `info["semantic"]` из прототипного поиска и навигации.

Сейчас object-finding в prototype collection и QA — случайный walk + проверка `_detect_near_from_info(info)`, который читает `info["semantic"]`. После Stage 68: та же логика через `CrafterSpatialMap` — агент накапливает карту мира из `NearDetector` по мере исследования, находит нужные объекты по памяти.

`info["player_pos"]` остаётся — это проприоцепция (агент знает где его тело, как человек).  
`info["inventory"]` остаётся — проприоцепция.  
`info["semantic"]` убирается — это внешнее знание о мире, должно приходить из восприятия.

---

## Скоуп

| Что меняем | Что не трогаем |
|---|---|
| `CrafterSpatialMap` — новый модуль | `NearDetector` / `CNNEncoder` |
| `_find_target_with_map()` — навигация к цели | `CLSWorldModel` |
| `exp124_pixel_nav.py` — новый эксперимент | `exp123_pixel_agent.py` (регрессия) |
| Обновить `crafter_pixel_env.py` — убрать TODO | `info["player_pos"]`, `info["inventory"]` |

---

## Архитектура

### До (Stage 67)
```
random_walk() → _detect_near_from_info(info["semantic"]) == target?
```

### После (Stage 68)
```
CrafterSpatialMap.update(player_pos, near_str_from_NearDetector)
    ↓
найти в карте клетку с нужным объектом → navigate(player_pos → target_pos)
    ↓
если не найдено → explore() (случайный walk, обновляем карту)
```

---

## Компоненты

### 1. `CrafterSpatialMap` (`src/snks/agent/crafter_spatial_map.py`)

Когнитивная карта Crafter. Каждая клетка хранит `near_str` — что было рядом когда агент там стоял. Аналог гиппокампальных place cells.

```python
class CrafterSpatialMap:
    """Cognitive map of Crafter world built from NearDetector observations.

    Maps visited (y, x) positions to observed near objects.
    Unknown cells = never visited = no entry in dict.
    """

    def __init__(self, world_size: int = 64):
        self.world_size = world_size
        # (y, x) → near_str (what NearDetector saw when agent was here)
        self._map: dict[tuple[int, int], str] = {}
        # Set of visited positions
        self._visited: set[tuple[int, int]] = set()

    def update(self, player_pos: tuple[int, int], near_str: str) -> None:
        """Record NearDetector output at current position.

        Args:
            player_pos: (y, x) from info["player_pos"].
            near_str: output of NearDetector.detect(pixels).
        """
        y, x = int(player_pos[0]), int(player_pos[1])
        self._map[(y, x)] = near_str
        self._visited.add((y, x))

    def find_nearest(
        self, target: str, player_pos: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Find nearest known position where target was observed.

        Args:
            target: near_str to search for (e.g. "tree").
            player_pos: current (y, x).

        Returns:
            (y, x) of nearest known position, or None if not in map.
        """
        py, px = int(player_pos[0]), int(player_pos[1])
        best_pos = None
        best_dist = float("inf")
        for (y, x), near in self._map.items():
            if near == target:
                d = abs(y - py) + abs(x - px)
                if d < best_dist:
                    best_dist = d
                    best_pos = (y, x)
        return best_pos

    def unvisited_neighbors(
        self, player_pos: tuple[int, int], radius: int = 3
    ) -> list[tuple[int, int]]:
        """Find unvisited positions within radius for exploration."""
        py, px = int(player_pos[0]), int(player_pos[1])
        result = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = py + dy, px + dx
                if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                    if (ny, nx) not in self._visited:
                        result.append((ny, nx))
        return result

    def reset(self) -> None:
        """Clear for new episode."""
        self._map.clear()
        self._visited.clear()

    @property
    def n_visited(self) -> int:
        return len(self._visited)
```

---

### 2. `_find_target_with_map()` — навигация к цели

Заменяет случайный walk + `_detect_near_from_info`. Стратегия:

```
1. Проверить карту: есть ли известная клетка с near_str == target?
   - Да → navigae_to(target_pos) — серия ходов в сторону цели
   - Нет → explore() — случайный ход к непосещённой клетке

2. На каждом шаге:
   - pixels, _, done, info = env.step(action)
   - near_str = detector.detect(pixels)
   - spatial_map.update(info["player_pos"], near_str)
   - if near_str == target → FOUND
```

```python
def _find_target_with_map(
    env: CrafterPixelEnv,
    detector: NearDetector,
    spatial_map: CrafterSpatialMap,
    target: str,
    max_steps: int = 300,
    rng: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, dict, bool]:
    """Navigate to target object using spatial map + NearDetector.

    Returns (pixels, info, found).
    Does NOT use info["semantic"] — only info["player_pos"] (proprioception).
    """
    if rng is None:
        rng = np.random.RandomState()

    MOVE_ACTIONS = ["move_left", "move_right", "move_up", "move_down"]

    pixels, info = env.observe()

    for _ in range(max_steps):
        near_str = detector.detect(torch.from_numpy(pixels))
        player_pos = info["player_pos"]
        spatial_map.update(player_pos, near_str)

        if near_str == target:
            return torch.from_numpy(pixels), info, True

        # Navigate: check map first, then explore
        known_pos = spatial_map.find_nearest(target, player_pos)
        if known_pos is not None:
            action = _step_toward(player_pos, known_pos, rng)
        else:
            unvisited = spatial_map.unvisited_neighbors(player_pos, radius=5)
            if unvisited:
                goal = unvisited[rng.randint(len(unvisited))]
                action = _step_toward(player_pos, goal, rng)
            else:
                action = rng.choice(MOVE_ACTIONS)

        pixels, _, done, info = env.step(action)
        if done:
            pixels, info = env.reset()
            spatial_map.reset()

    return torch.from_numpy(pixels), info, False


def _step_toward(
    current: tuple[int, int], target: tuple[int, int],
    rng: np.random.RandomState
) -> str:
    """One step toward target (greedy, with random tie-breaking)."""
    cy, cx = int(current[0]), int(current[1])
    ty, tx = int(target[0]), int(target[1])
    dy, dx = ty - cy, tx - cx

    moves = []
    if dy > 0:
        moves.append("move_down")
    elif dy < 0:
        moves.append("move_up")
    if dx > 0:
        moves.append("move_right")
    elif dx < 0:
        moves.append("move_left")

    if not moves:
        return rng.choice(["move_left", "move_right", "move_up", "move_down"])
    return rng.choice(moves)
```

---

### 3. `exp124_pixel_nav.py` (новый эксперимент)

**Фазы:**

**Phase 0:** Загрузить encoder Stage 67, создать `NearDetector`.

**Phase 1: Navigation smoke** — сравнить `_find_target_with_map` vs старый случайный walk:
- 10 объектов × 20 seeds
- Метрика: % найденных за 300 шагов, средний шаг до находки
- Gate (smoke): ≥60% success rate (vs ~40% для random walk)

**Phase 2: QA gate** — тот же Crafter QA L1-L4 что в exp123, но:
- Prototype collection: `_find_target_with_map` вместо random walk + `_detect_near_from_info`
- QA test: то же
- Gate: ≥90% QA accuracy

**Phase 3: Regression** — exp123 pipeline (Stage 67), порог ≥90%.

---

## Тестирование

### Unit (`tests/test_crafter_spatial_map.py`)
- `update()` + `find_nearest()`: записать (2,3)→"tree", найти "tree" из (2,5) → (2,3)
- `find_nearest()` несколько кандидатов → ближайший
- `find_nearest()` target не в карте → None
- `unvisited_neighbors()` корректно исключает посещённые
- `reset()` очищает всё
- `_step_toward()` движется в правильном направлении

### Integration (`tests/test_pixel_navigation.py`)
- `_find_target_with_map()` находит "tree" без `info["semantic"]`
- При found=False возвращает found=False, не падает
- `spatial_map.n_visited` растёт после вызова

---

## Файлы

| Действие | Файл |
|---|---|
| Создать | `src/snks/agent/crafter_spatial_map.py` |
| Создать | `experiments/exp124_pixel_nav.py` |
| Создать | `tests/test_crafter_spatial_map.py` |
| Создать | `tests/test_pixel_navigation.py` |
| Обновить | `src/snks/agent/crafter_pixel_env.py` — убрать Stage 68 TODO про навигацию |

---

## Gate-критерий Stage 68

```
Phase 1 nav smoke:  ≥60% target-found rate без info["semantic"]
Phase 2 QA gate:    Crafter QA L1-L4 avg ≥ 90%
Phase 3 regression: exp123 ≥ 90%
```

При прохождении всех трёх: Stage 68 COMPLETE.

---

## Что остаётся после Stage 68

```python
# Осталось в info[]: только проприоцепция
info["player_pos"]   # OK — это тело агента (проприоцепция)
info["inventory"]    # OK — это память агента (проприоцепция)
# info["semantic"]   # УБРАНО в Stage 68
```

**Следующий рубеж:** circular dependency — near_labels для обучения CNN берутся из той же символики. Stage 69.
