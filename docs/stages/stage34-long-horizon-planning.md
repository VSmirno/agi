# Stage 34: Long-Horizon Planning

**Статус:** IN PROGRESS
**Ветка:** stage34-long-horizon-planning
**Эксперименты:** exp86, exp87, exp88
**Дата:** 2026-03-31

---

## Что доказывает

СНКС может планировать на **1000+ шагов вперёд** через иерархическую декомпозицию:
1. Высокоуровневый план из Skills/SubGoals (5-20 абстрактных шагов)
2. Каждый абстрактный шаг раскладывается в 50-200 примитивных действий
3. Re-planning при отклонении от ожидаемого состояния
4. Общее число шагов > 1000 при сохранении coherent поведения

---

## Проблема

Текущие планировщики:
- `MentalSimulator.find_plan()` — BFS, max_depth=10, экспоненциальная сложность
- `StochasticSimulator.find_plan_stochastic()` — Monte Carlo, max_depth=10
- `GoalAgent` — backward chaining, MAX_CHAIN_DEPTH=3

Для реальных задач (navigate + pickup + unlock + navigate + ...) нужен горизонт 1000+.

---

## Архитектура

### HierarchicalPlanner — 3 уровня

```
Level 2 (Strategic):   [reach_room_A] → [get_key] → [unlock_door] → [reach_goal]
                              ↓              ↓             ↓              ↓
Level 1 (Tactical):    [nav(0,0→3,2)]  [find+pickup]  [nav+toggle]  [nav(5,3→7,7)]
                              ↓              ↓             ↓              ↓
Level 0 (Primitive):   [fwd,fwd,left,  [fwd,right,    [fwd,toggle]  [fwd,fwd,fwd,
                         fwd,fwd,fwd]   fwd,pickup]                   right,fwd...]
```

### PlanNode — единица плана

```python
@dataclass
class PlanNode:
    level: int                      # 0=primitive, 1=tactical, 2=strategic
    action: int | str               # primitive action ID or skill name
    preconditions: frozenset[int]   # required SKS
    postconditions: frozenset[int]  # expected SKS after
    children: list[PlanNode]        # sub-steps (lower level)
    estimated_steps: int            # estimated primitive steps
    status: str                     # "pending" | "active" | "done" | "failed"
```

### HierarchicalPlanner

```python
class HierarchicalPlanner:
    def plan(goal_sks, current_sks, max_horizon) → PlanGraph
    def expand(node: PlanNode) → list[PlanNode]   # decompose to lower level
    def replan(node: PlanNode, actual_sks) → PlanNode | None  # handle deviation
    def execute_step(plan: PlanGraph) → (action, done)
    def total_steps(plan: PlanGraph) → int
```

---

## Подходы (brainstorming)

### Подход A: HTN (Hierarchical Task Network)
- Формальная HTN декомпозиция с method/operator
- **Pro:** проверенный в AI planning, формально корректный
- **Con:** требует полного описания methods, жёсткий

### Подход B: Skill-Based Hierarchical Planning (рекомендуемый) ✓
- Level 2: backward chaining через CausalWorldModel (какие SKS нужны?)
- Level 1: SkillLibrary.find_applicable() → последовательность skills
- Level 0: MentalSimulator или GoalAgent execution
- Re-planning при deviation > threshold
- **Pro:** использует все существующие модули, гибкий, уже есть skills
- **Con:** зависит от качества каузальной модели

### Подход C: MCTS на абстрактных действиях
- Monte Carlo Tree Search по skill-space
- **Pro:** оптимальный для неопределённости
- **Con:** O(branching^depth * simulations), медленный

**Выбран: Подход B** — Skill-Based Hierarchical Planning
- Обоснование: естественное расширение существующей архитектуры. Skills (Stage 27) дают абстракции, CausalWorldModel даёт forward prediction, backward chaining (Stage 25) даёт goal decomposition. Нужно только "склеить" в иерархию и добавить re-planning.

---

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 86 | plan_depth | ≥ 1000 | Общее число примитивных шагов в плане |
| 86 | plan_coherence | ≥ 0.9 | Доля шагов с valid preconditions |
| 87 | replan_success | ≥ 0.8 | Успешность re-planning при deviation |
| 87 | replan_overhead | ≤ 1.5x | Overhead re-planning vs ideal |
| 88 | multi_room_success | ≥ 0.9 | Успешность в multi-room задаче (3+ комнат) |
| 88 | hierarchical_speedup | ≥ 2x | Speedup vs flat BFS planning |

---

## Модули

1. `src/snks/language/plan_node.py` — PlanNode, PlanStatus
2. `src/snks/language/hierarchical_planner.py` — HierarchicalPlanner
3. `tests/test_hierarchical_planner.py` — unit tests
4. `src/snks/experiments/exp86_plan_depth.py`
5. `src/snks/experiments/exp87_replan.py`
6. `src/snks/experiments/exp88_multi_room.py`
