# Stage 54: Partial Observability — Design Spec

**Дата:** 2026-04-02
**Milestone:** M4 — Масштаб (первый этап)
**Gate:** ≥80% success на 200 random DoorKey-5x5 с 7x7 partial view
**Зависимости:** Stage 47 (BFS pathfinding), Stage 46 (subgoal planning)

---

## Позиция в фазе

**M4 — Масштаб:** Stage 54 — первый из 7 этапов. Продвигает маркер "partial observability" — убирает FullyObsWrapper shortcut (TD-006). Это фундамент для Stage 55 (exploration strategy для MultiRoom).

---

## Проблема

Stages 47-49 используют FullyObsWrapper — агент видит всю карту. BFS на полной карте тривиален: 100% success. С partial obs (7x7 view) агент видит только 7x7 клеток в направлении взгляда. Нужно:

1. **Накапливать карту** из последовательности частичных наблюдений
2. **Исследовать** неизвестные области (frontier exploration)
3. **Планировать** на основе накопленной (неполной) карты
4. **Переключаться** между exploration и goal-directed navigation

---

## Решение: SpatialMap + FrontierExplorer + PartialObsAgent

### Компонент 1: SpatialMap

Аккумулятор наблюдений в абсолютных координатах.

```python
class SpatialMap:
    """2D grid map accumulated from partial observations."""
    
    def __init__(self, size: int):
        # 3-channel grid: (obj_type, color, state), -1 = unknown
        self.grid: np.ndarray  # shape (size, size, 3), init -1
        self.explored: np.ndarray  # shape (size, size), bool
    
    def update(self, obs_7x7: np.ndarray, agent_pos: tuple[int,int], agent_dir: int):
        """Project 7x7 egocentric view onto absolute map coordinates.
        
        MiniGrid partial obs: 7x7 grid, agent at (6, 3), facing up.
        Need rotation based on agent_dir to convert to absolute coords.
        """
    
    def to_obs(self) -> np.ndarray:
        """Convert to full-grid-like observation for BFS pathfinding.
        Unknown cells treated as empty (optimistic planning).
        """
    
    def find_objects(self) -> dict:
        """Find known positions of key, door, goal, agent."""
    
    def frontiers(self) -> list[tuple[int,int]]:
        """Find frontier cells: explored cells adjacent to unexplored cells."""
    
    def reset(self):
        """Clear map for new episode."""
```

**Координатная трансформация:**
MiniGrid partial obs — 7x7, agent at position (6, 3), facing up (dir=3). Реальная позиция agent в grid известна через `env.unwrapped.agent_pos`. Для каждого dir нужна rotation matrix:
- dir=0 (right): obs[r,c] → map[agent_row + c - 3, agent_col + (6 - r)]
- dir=1 (down): obs[r,c] → map[agent_row + (6 - r), agent_col - (c - 3)]
- dir=2 (left): obs[r,c] → map[agent_row - (c - 3), agent_col - (6 - r)]
- dir=3 (up): obs[r,c] → map[agent_row - (6 - r), agent_col + (c - 3)]

### Компонент 2: FrontierExplorer

```python
class FrontierExplorer:
    """Navigate to nearest frontier (unexplored reachable cell)."""
    
    def select_action(self, spatial_map: SpatialMap, 
                      agent_pos: tuple, agent_dir: int) -> int:
        """BFS to nearest frontier cell, return first action."""
    
    def _nearest_frontier(self, spatial_map: SpatialMap,
                          agent_pos: tuple) -> tuple[int,int] | None:
        """BFS from agent to nearest frontier."""
```

### Компонент 3: PartialObsAgent

```python
class PartialObsAgent:
    """Agent for partial observability with spatial map accumulation.
    
    Strategy per step:
    1. Update spatial map with current 7x7 view
    2. Check if all objects found (key, door, goal)
    3. If yes → subgoal planning (BFS to key → door → goal)
    4. If no → frontier exploration (BFS to nearest unknown)
    """
    
    def __init__(self, grid_size: int = 5, epsilon: float = 0.05):
        self.spatial_map = SpatialMap(grid_size)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.epsilon = epsilon
        self._subgoal_phase: str | None = None  # "pickup_key" | "open_door" | "reach_goal"
    
    def select_action(self, obs: np.ndarray, 
                      agent_pos: tuple, agent_dir: int) -> int:
        """Main action selection loop."""
    
    def reset(self):
        """Reset for new episode."""
```

### Компонент 4: PartialObsDoorKeyEnv

Обёртка для RandomDoorKeyEnv без FullyObsWrapper — стандартный 7x7 partial obs.

```python
class PartialObsDoorKeyEnv:
    """DoorKey-5x5 with standard 7x7 partial observation."""
    
    def __init__(self, size: int = 5, seed: int | None = None):
        # Используем MiniGrid DoorKey напрямую, без FullyObsWrapper
        # see_through_walls=False (default)
    
    def reset(self) -> tuple[np.ndarray, tuple[int,int], int]:
        """Returns (obs_7x7, agent_pos, agent_dir)"""
    
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, tuple[int,int], int]:
        """Returns (obs_7x7, reward, term, trunc, agent_pos, agent_dir)"""
```

---

## Альтернативы (отклонены)

1. **SDM-only (VSA beam search)** — провалился в Stage 45 на detour tasks
2. **SLAM-подобный подход** — избыточен для 5x5 grid
3. **RNN/LSTM memory** — не bio-plausible, нарушает архитектуру СНКС

---

## Тест-план

### CPU тесты (локально):
1. `test_spatial_map_update` — корректная проекция 7x7 view на карту для всех 4 направлений
2. `test_spatial_map_accumulation` — карта накапливается за несколько шагов
3. `test_frontier_detection` — frontiers корректно определяются
4. `test_frontier_explorer_navigation` — агент движется к frontier
5. `test_partial_obs_agent_finds_objects` — после exploration агент находит key/door/goal
6. `test_partial_obs_agent_subgoal_switch` — переключение explore→plan после обнаружения объектов
7. `test_partial_obs_doorkey_env` — обёртка корректно возвращает 7x7 obs

### Эксперименты:
- **exp108a**: SpatialMap accuracy — % правильно накопленных клеток за 20 шагов
- **exp108b**: PartialObsAgent на 200 random DoorKey-5x5 с 7x7 view (PRIMARY GATE)
- **exp108c**: Ablation — exploration-only vs full agent

---

## Файлы

- `src/snks/agent/spatial_map.py` — SpatialMap + FrontierExplorer
- `src/snks/agent/partial_obs_agent.py` — PartialObsAgent + PartialObsDoorKeyEnv
- `tests/test_stage54_partial_obs.py` — unit tests
- `src/snks/experiments/exp108_partial_obs.py` — experiments
- `demos/stage-54-partial-obs.html` — web demo

---

## Риски

1. **Координатная трансформация** — MiniGrid partial obs имеет нетривиальную ориентацию. Нужно тщательно тестировать.
2. **5x5 grid слишком маленький** — 7x7 view может покрывать большую часть 5x5 grid (~49 клеток view vs 25 клеток grid). Агент может видеть почти всё сразу. Это упрощает задачу, но gate всё равно нетривиален (agent_pos влияет на visible area).
3. **Subgoal detection на partial obs** — агент может не видеть key/door/goal одновременно. Нужна logic для частичного плана.
