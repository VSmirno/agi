# Stage 24c: BabyAI Execution (e2e)

**Версия:** 1.0
**Дата:** 2026-03-31
**Статус:** IN PROGRESS

---

## Цель

Полный end-to-end цикл: **текстовая инструкция → parse → plan → execute в MiniGrid → verify**.
Это финальный этап языкового блока (Stages 19–24).

### Что доказывает

1. Языковая pipeline (chunker → planner) порождает **исполнимые** action sequences
2. Grounded perception в grid-мире работает — агент **видит** объекты и распознаёт их как концепты
3. Язык и действие **замыкаются** через единую концептную систему (СКС)

### Ограничения scope

- **DAF не обучается на MiniGrid** — используем GridPerception (прямое чтение grid → SKS)
- Нет navigation planner (A*/BFS) — используем random walk + instruction following
- Нет transfer learning между средами — каждый эксперимент самодостаточен

---

## Архитектура

```
TextInstruction           MiniGrid Env
    │                         │
    ▼                         ▼
RuleBasedChunker        GridPerception
    │                    (grid → SKS map)
    ▼                         │
InstructionPlanner ◄──────────┘
    │                    current_sks
    ▼
ActionSequence ──────► env.step(action)
    │                         │
    ▼                         ▼
SuccessCheck ◄───────── reward / state
```

---

## Компоненты

### 1. GridPerception (`language/grid_perception.py`)

Мост MiniGrid grid state → SKS concept IDs. Заменяет DAF visual encoder для grid-мира.

```python
class GridPerception:
    """Extracts SKS concepts directly from MiniGrid grid state.

    Each unique (object_type, color) pair gets a stable SKS ID.
    Agent position and direction are also tracked.
    """

    def __init__(self, grounding_map: GroundingMap):
        self._gmap = grounding_map
        self._type_color_to_sks: dict[tuple[str, str], int] = {}
        self._next_sks_id: int = 100  # start above reserved IDs

    def perceive(self, grid, agent_pos, agent_dir) -> set[int]:
        """Extract active SKS IDs from current grid state.

        Returns set of SKS IDs for:
        - All visible objects (in agent's field of view)
        - Agent's current cell contents
        - Adjacent cell contents
        """
        ...

    def register_object(self, obj_type: str, color: str, word: str) -> int:
        """Register a MiniGrid object as a grounded concept.

        Creates bidirectional mapping:
        - (obj_type, color) → sks_id
        - word → sks_id (via GroundingMap)
        """
        ...
```

### 2. BabyAIExecutor (`language/babyai_executor.py`)

Orchestrates the full e2e loop. Не наследует EmbodiedAgent (слишком тяжёлый), а использует компоненты напрямую.

```python
class BabyAIExecutor:
    """End-to-end executor: instruction → actions → MiniGrid.

    Components:
    - RuleBasedChunker: text → chunks
    - InstructionPlanner: chunks → action_ids (with CausalWorldModel)
    - GridPerception: grid state → current_sks
    - MiniGrid env: execute actions, get reward
    """

    def __init__(self, env, chunker, planner, perception):
        ...

    def execute(self, instruction: str, max_steps: int = 50) -> ExecutionResult:
        """Execute a text instruction in the environment.

        Returns:
            ExecutionResult with success, steps_taken, trajectory.
        """
        ...
```

### 3. CausalWorldModel seeding

Для InstructionPlanner нужна CausalWorldModel с предзагруженными causal links.
Используем `seed_causal_links()` — инжектируем BabyAI-специфичные связи:

```python
def seed_babyai_links(causal_model: CausalWorldModel, perception: GridPerception):
    """Seed causal model with BabyAI domain knowledge.

    Links:
    - pickup(key) requires: agent_at(key_cell)
    - open(door) requires: key_held
    - goto(X) requires: path_exists(X)
    """
    ...
```

### 4. MiniGrid Action Mapping

```python
MINIGRID_ACTIONS = {
    "left": 0,      # turn left
    "right": 1,     # turn right
    "forward": 2,   # move forward
    "pick up": 3,   # pickup object
    "drop": 4,      # drop object
    "toggle": 5,    # toggle/open
    "done": 6,      # declare done
}
```

---

## Эксперименты

### Exp 56: GoTo + Pickup Success Rate

**Контекст:** простые однофразовые инструкции в BabyAI-GoToObj и BabyAI-PickupLoc.

**Протокол:**
- 50 эпизодов BabyAI-GoToObj-v0 (5x5 grid, одна комната)
- 50 эпизодов с pickup инструкциями
- GridPerception для восприятия, RuleBasedChunker для parsing
- max_steps=50 на эпизод
- Навигация: BFS к целевому объекту (shortest path в known grid)

**Gate:**
```
goto_success_rate  >= 0.6   # 30/50 GoTo
pickup_success_rate >= 0.5  # 25/50 Pickup
overall_success_rate >= 0.55
```

**Метрики:**
- success_rate: reward > 0
- avg_steps: среднее число шагов до успеха
- parse_accuracy: % инструкций корректно распознанных chunker'ом

### Exp 57: Novel Combinations (Generalization)

**Контекст:** агент видел "red key" и "blue ball" отдельно. Теперь инструкция "pick up the blue key" — комбинация невиданных ранее атрибутов.

**Протокол:**
- Training phase: 20 эпизодов с "red key", "blue ball", "green box"
- Test phase: 20 эпизодов с novel combinations: "blue key", "red ball", "green key"
- Тот же GridPerception + chunker + planner
- Compositional generalization: ATTR грounding + OBJECT grounding → novel combo

**Gate:**
```
novel_success_rate >= 0.3    # 6/20 на novel combinations
known_success_rate >= 0.5    # baseline на known не деградирует
```

**Метрики:**
- novel_success_rate: успех на невиданных комбинациях
- known_success_rate: контроль на знакомых
- grounding_accuracy: % правильно идентифицированных объектов

---

## Файлы

| Файл | Назначение |
|------|-----------|
| `src/snks/language/grid_perception.py` | GridPerception: grid → SKS |
| `src/snks/language/babyai_executor.py` | BabyAIExecutor: e2e orchestration |
| `tests/test_babyai_executor.py` | Unit tests (mock env) |
| `experiments/exp56_babyai_goto_pickup.py` | Exp 56 runner |
| `experiments/exp57_babyai_novel_combos.py` | Exp 57 runner |

---

## Зависимости

- `minigrid>=2.3.0` (MiniGrid environments)
- Stage 24a: RuleBasedChunker (sequential + spatial patterns)
- Stage 24b: InstructionPlanner + CausalWorldModel
- Stage 19: GroundingMap (word ↔ SKS)

---

## Навигация

BFS pathfinding в known grid (не random walk):

```python
class GridNavigator:
    """BFS shortest-path navigation in MiniGrid grid."""

    def plan_path(self, grid, agent_pos, agent_dir, target_pos) -> list[int]:
        """Returns list of MiniGrid action IDs to reach target_pos."""
        ...
```

Это scaffolding (как RuleBasedChunker) — в будущем заменится на learned navigation.

---

## Риски

1. **MiniGrid API changes** — minigrid v2.x vs v3.x различия. Зафиксировать версию.
2. **Partial observability** — агент видит только 7x7 перед собой. BFS нужен по full grid (cheating, но допустимо для scaffolding).
3. **Action mapping ambiguity** — "open" в BabyAI = toggle action в MiniGrid.
