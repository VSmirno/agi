# Stage 49: Multi-Room Navigation — Design Spec

**Date:** 2026-04-02
**Stage:** 49
**Gate:** ≥60% success rate on random MultiRoom-N3 layouts (200 episodes)
**Milestone:** M1 (Генерализация) — completing this closes M1

## Позиция в фазе

**Фаза M1 — Генерализация**, маркеры:
- [x] ≥80% random DoorKey-5x5 → 100% (Stage 47)
- [ ] ≥60% MultiRoom-N3 → **этот этап**

Если gate пройден — **M1 COMPLETE**, переход к M2 (Языковой контроль).

## Проблема

Текущий `SubgoalPlanningAgent` работает только с DoorKey-5x5 (7x7 grid, 1 ключ, 1 дверь, 1 цель). MultiRoom-N3 — принципиально другая задача:
- 25x25 grid (vs 7x7)
- 3 комнаты, 2 двери
- Двери закрыты (state=1), но НЕ заблокированы — не нужен ключ
- Рандомная генерация layout каждый эпизод
- Нужен FullyObsWrapper для полного наблюдения

## Подходы

### A: Extend SubgoalPlanningAgent (сложно, ненужно)
- Адаптировать build_plan_from_obs для multi-room
- Поддерживать variable subgoal chains
- Много кода, fragile для edge cases
- **Минус:** over-engineering для задачи без ключей

### B: Reactive BFS Navigator (рекомендуемый) ✓
- BFS от агента до цели с `allow_door=True`
- Следовать пути; при подходе к закрытой двери — toggle
- Реактивный подход: re-plan после каждого toggle
- **Плюс:** простой, robust, переиспользует GridPathfinder
- **Плюс:** работает для любого количества комнат/дверей

### C: Hierarchical room-level planner
- Строить граф комнат, планировать на уровне комнат
- Слишком сложно для задачи, которую BFS решает напрямую

**Выбран: B** — BFS через двери + reactive toggle. Минимальный код, максимальная robustness.

## Архитектура

### Новый класс: `MultiRoomNavigator`

```python
class MultiRoomNavigator:
    """Navigate multi-room environments using BFS + door toggling."""
    
    def __init__(self, epsilon: float = 0.05):
        self.pathfinder = GridPathfinder()  # from Stage 47
        self.epsilon = epsilon
    
    def run_episode(self, env, obs, max_steps=500):
        """Run one episode: BFS to goal, toggle doors on the way."""
        for step in range(max_steps):
            agent_pos, agent_dir = find_agent(obs)
            goal_pos = find_goal(obs)
            
            # Check if next cell is closed door → toggle
            if facing_closed_door(obs, agent_pos, agent_dir):
                action = 5  # toggle
            else:
                # BFS path to goal (doors passable)
                path = pathfinder.find_path(obs, agent_pos, goal_pos, allow_door=True)
                action = path_to_actions(path, agent_dir)[0]
            
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                return reward > 0, step + 1, reward
```

### Модификация GridPathfinder

Текущий `extract_walls` уже поддерживает `allow_door=True`. Закрытые двери (state=1) НЕ блокируют BFS — это exactly то, что нужно. Двери пропускаются как passable cells.

### Обработка дверей

MiniGrid двери:
- state=0: open (свободно проходить)
- state=1: closed (нужен toggle, потом можно пройти)
- state=2: locked (нужен ключ + toggle)

В MultiRoom все двери state=1. Алгоритм:
1. BFS считает closed doors проходимыми
2. Агент идёт по пути
3. Когда следующая клетка — закрытая дверь, агент:
   a. Поворачивается к двери (if needed)
   b. Toggle (action=5)
   c. Forward через дверь (action=2)
4. Продолжает к следующей точке пути

### Environment wrapper

```python
class MultiRoomEnvWrapper:
    """Wrap MiniGrid MultiRoom for our agent interface."""
    
    def __init__(self, n_rooms=3, max_room_size=6, max_steps=300):
        base = MultiRoomEnv(minNumRooms=n_rooms, maxNumRooms=n_rooms, 
                           maxRoomSize=max_room_size, max_steps=max_steps)
        self.env = FullyObsWrapper(base)
    
    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return obs['image']  # (25, 25, 3)
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return obs['image'], reward, term, trunc, info
```

## Файлы

| Файл | Действие | Описание |
|------|----------|----------|
| `src/snks/agent/multi_room_nav.py` | NEW | MultiRoomNavigator + env wrapper |
| `src/snks/agent/pathfinding.py` | UPDATE | handle closed doors (state=1) in extract_walls |
| `tests/test_stage49_multi_room.py` | NEW | Unit tests |
| `src/snks/experiments/exp108_multi_room.py` | NEW | Gate experiments |
| `demos/stage-49-multi-room.html` | NEW | Canvas demo |

## Gate-критерии

| Exp | Метрика | Gate |
|-----|---------|------|
| 108a | BFS pathfinding на 50 random MultiRoom-N3 | 100% path found |
| 108b | MultiRoomNavigator на 200 random MultiRoom-N3 | ≥60% success |
| 108c | Average steps to goal | ≤150 steps |

## Риски

1. **Partial observability** — мы используем FullyObsWrapper, что даёт полное наблюдение. Это "cheating" в смысле POMDP, но консистентно с DoorKey-5x5 (Stage 47 тоже видит весь grid).
2. **Door toggle timing** — агент должен точно определить, когда он стоит перед закрытой дверью. Ошибка → лишние шаги или зависание.
3. **Max steps** — MultiRoom-N3 на 25x25 может потребовать >200 шагов. Используем max_steps=500.
