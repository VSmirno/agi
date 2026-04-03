# Stage 57: Long Subgoal Chains — Prerequisite-Graph Planning

**Дата:** 2026-04-03
**Milestone:** M4 — Масштаб (четвёртый этап)
**Gate:** ≥40% success на задачах с 5+ subgoals (200 random seeds)

---

## Позиция в фазе

**M4 маркеры:**
- [x] Partial observability (Stage 54)
- [x] Exploration strategy (Stage 55)
- [x] Complex environment 5+ object types (Stage 56)
- [ ] **Long subgoal chains 5+** ← этот этап
- [ ] SDM scaling ≥1000 (Stage 58)
- [ ] Transfer learning (Stage 59)
- [ ] Integration test (Stage 60)

Этот этап доказывает: агент может решать задачи, требующие **последовательной цепочки 5+ зависимых subgoals** с prerequisite-отношениями (нельзя открыть дверь без ключа, нельзя достать мяч без открытия двери). Это переход от "4-шаговых" задач (DoorKey, PutNext) к реальному многошаговому планированию.

---

## Анализ сред

### BabyAI-KeyCorridorS4R3-v0 (основная)
- **Grid:** 10×10, коридор + 3×4 комнат
- **Объекты:** 1 key, 1 locked door (matching color), 5-7 unlocked doors, 1 ball (goal)
- **Миссия:** "pick up the ball"
- **Max steps:** 480
- **Partial obs:** 7×7 (agent-centric)
- **Subgoal chain (7 шагов):**
  1. EXPLORE — найти ключ (может быть за unlocked doors)
  2. NAVIGATE_TO_KEY — BFS через unlocked doors
  3. PICKUP_KEY — подобрать ключ
  4. EXPLORE_FOR_LOCKED_DOOR — найти locked door
  5. NAVIGATE_TO_LOCKED_DOOR — через коридор и unlocked doors
  6. OPEN_LOCKED_DOOR — toggle locked door
  7. NAVIGATE_TO_BALL — через открытую дверь
  8. PICKUP_BALL — подобрать мяч

### BabyAI-KeyCorridorS3R3-v0 (дополнительная)
- **Grid:** 7×7, 3×3 комнат
- **Аналогичная структура**, компактнее
- **Subgoal chain:** 6-7 шагов

### BabyAI-BlockedUnlockPickup-v0 (дополнительная)
- **Grid:** 11×6, две комнаты
- **Объекты:** key + ball (блокирует дверь) + locked door + box (goal)
- **Миссия:** "pick up the box"
- **Subgoal chain (5-6):**
  1. Find key → 2. Pickup key → 3. Navigate to door → 4. Open door → 5. Navigate to box → 6. Pickup box
- Ball блокирует дверь — нужно обойти или сдвинуть

---

## Архитектура

### Ключевая идея: Prerequisite-Graph ChainPlanner

Универсальный символический планировщик, строящий граф зависимостей из SpatialMap:

```
mission "pick up the ball"
        ↓
MissionAnalyzer → goal = (PICKUP, ball)
        ↓
PrerequisiteGraph:
  PICKUP_BALL ← ADJACENT_TO_BALL ← ACCESS_BALL_ROOM
  ACCESS_BALL_ROOM ← OPEN_LOCKED_DOOR(color=purple)
  OPEN_LOCKED_DOOR(purple) ← HAS_KEY(purple) ← PICKUP_KEY(purple)
  PICKUP_KEY(purple) ← ADJACENT_TO_KEY(purple) ← EXPLORE
        ↓
ChainPlanner → ordered subgoals:
  [EXPLORE, GOTO_KEY, PICKUP_KEY, GOTO_LOCKED_DOOR, OPEN_DOOR, GOTO_BALL, PICKUP_BALL]
        ↓
SubgoalExecutor → BFS nav + interaction per subgoal
```

### Компоненты

#### 1. ChainPlanner
Строит и исполняет упорядоченную цепочку subgoals.

```python
class Subgoal:
    name: str           # "PICKUP_KEY", "OPEN_DOOR", "PICKUP_BALL"
    target_type: int    # MiniGrid obj type ID
    target_color: int   # MiniGrid color ID  
    action: str         # "pickup", "toggle", "goto"
    prerequisite: str   # what must be true before this subgoal

class ChainPlanner:
    def __init__(self, mission: str, spatial_map: SpatialMap):
        self.subgoals: list[Subgoal] = self._build_chain(mission)
        self.current_idx: int = 0
    
    def _build_chain(self, mission: str) -> list[Subgoal]:
        """Backward chain from goal to first needed action."""
        ...
    
    def current_subgoal(self) -> Subgoal:
        return self.subgoals[self.current_idx]
    
    def advance(self) -> bool:
        """Mark current subgoal achieved, return True if all done."""
        self.current_idx += 1
        return self.current_idx >= len(self.subgoals)
```

#### 2. SubgoalExecutor
Исполняет один subgoal за раз, используя SpatialMap + BFS.

```python
class SubgoalExecutor:
    def execute_step(self, subgoal: Subgoal, obs, agent_pos, agent_dir) -> int:
        """Return action for current subgoal."""
        if subgoal.action == "explore":
            return frontier_explore()
        elif subgoal.action == "goto":
            return bfs_navigate_adjacent(subgoal.target)
        elif subgoal.action == "pickup":
            return face_and_pickup(subgoal.target)
        elif subgoal.action == "toggle":
            return face_and_toggle(subgoal.target)
```

#### 3. KeyCorridorAgent (main agent)
Интегрирует ChainPlanner + SubgoalExecutor + SpatialMap.

```python
class KeyCorridorAgent:
    def __init__(self, grid_w, grid_h, mission):
        self.spatial_map = SpatialMap(grid_w, grid_h)
        self.explorer = FrontierExplorer()
        self.pathfinder = GridPathfinder()
        self.planner = ChainPlanner(mission)
        self._carrying = False
        self._door_states: dict[tuple, str] = {}  # pos → "locked"/"unlocked"/"open"
    
    def select_action(self, obs_7x7, agent_col, agent_row, agent_dir):
        self.spatial_map.update(obs_7x7, agent_col, agent_row, agent_dir)
        self._update_door_states()
        self._check_subgoal_completion()
        
        subgoal = self.planner.current_subgoal()
        return self._execute_subgoal(subgoal, ...)
```

### Ключевые решения

#### Unlocked doors: toggle-on-sight
Unlocked (closed) doors обрабатываются автоматически при навигации:
- Если BFS-путь проходит через closed unlocked door → navigate to adjacent, toggle, continue
- Не требуют отдельного subgoal — это навигационный примитив

#### Locked doors: prerequisite chain
- Locked door обнаруживается через spatial_map (state=2 = locked)
- Ключ нужного цвета → subgoal "PICKUP_KEY(color)"
- После pickup → subgoal "OPEN_DOOR(color)"

#### Incremental planning
Планирование происходит инкрементально по мере исследования:
1. Начальный план: [EXPLORE] (найти хоть что-то)
2. Нашли ball → добавляем GOTO_BALL, PICKUP_BALL
3. Нашли locked door на пути → вставляем FIND_KEY, PICKUP_KEY, OPEN_DOOR перед GOTO_BALL
4. Нашли key → заменяем FIND_KEY на GOTO_KEY

Это **reactive planning**: план обновляется при каждом новом открытии объекта.

#### Door state tracking
SpatialMap уже хранит state в channel 2:
- state=0: open door
- state=1: closed (unlocked)  
- state=2: locked

Расширяем: `find_doors()` → list of (type, color, row, col, state)

---

## Implementation plan

1. **Extend SpatialMap** — `find_doors()` returns doors with lock state and color
2. **MissionAnalyzer** — parse "pick up the X" → goal object type
3. **ChainPlanner** — build/update prerequisite chain from map state
4. **KeyCorridorAgent** — integrate planner + spatial map + BFS
5. **KeyCorridorEnv** — wrapper for BabyAI KeyCorridor with agent info
6. **Tests** — unit + integration for each component
7. **Experiments** — KeyCorridorS4R3, S3R3, BlockedUnlockPickup (200 seeds each)

---

## Gate criteria

| Metric | Gate | Measurement |
|--------|------|-------------|
| Success rate (KeyCorridorS4R3, 200 seeds) | ≥40% | ball picked up |
| Subgoal chain length | ≥5 | verified subgoal count per episode |
| Success rate (KeyCorridorS3R3, 200 seeds) | ≥50% | ball picked up |
| Mean steps (successful, S4R3) | ≤300 | steps to complete |

---

## Risks

1. **Partial obs + large grid:** 10×10 with 7×7 view — key or ball may be far from start, needs extensive exploration
2. **Door navigation:** unlocked doors need toggle → step → continue, must not get stuck in toggle loops
3. **Key-door color matching:** must correctly identify which key opens which locked door
4. **Ball blocking door (BlockedUnlockPickup):** ball in front of locked door — need to push aside or approach from other direction

---

## Dependencies on previous stages

- **Stage 54:** SpatialMap, FrontierExplorer, PartialObsAgent — all reused
- **Stage 55:** MultiRoom exploration strategy — reused for multi-room KeyCorridor
- **Stage 56:** PutNextAgent pattern (state machine + mission parsing) — pattern reused, different specifics
