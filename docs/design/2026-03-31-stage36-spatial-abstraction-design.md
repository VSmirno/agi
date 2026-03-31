# Stage 36: Spatial Abstraction & Scalable Autonomous Agent

## Проблема

Exp 92 показал 0% success на DoorKey 12x12 (100 эпизодов, N=50K).
EmbodiedAgent (Stage 14) используется "голый" — без capabilities stages 25-35.
IntegratedAgent (Stage 35) — facade без episode loop.
GoalAgent (Stage 25) работает на 5x5, но не масштабируется.

### Root Causes
1. **EmbodiedAgent в explore mode = random walk** — бесполезно на 12x12
2. **Нет интеграции capabilities с episode loop на больших средах**
3. **Causal model не накапливает confidence** за 100 эпизодов (нужно 300+)
4. **Planning horizon = 5** — мало для 12x12 (нужно 15+)
5. **Нет curriculum** — агент сразу на сложном уровне без предварительного обучения

## Подход: AutonomousAgent

### Архитектура

```
AutonomousAgent (Stage 36)
├── GoalAgent (Stage 25) — goal decomposition + causal learning
│   ├── GridPerception — SKS extraction
│   ├── CausalWorldModel — state transitions
│   └── GridNavigator — pathfinding
├── CuriosityModule (Stage 29) — directed exploration
├── SkillLibrary (Stage 27) — macro-actions
├── HierarchicalPlanner (Stage 34) — long-horizon plans
├── MetaLearner (Stage 32) — strategy selection
└── CurriculumManager (NEW) — progressive difficulty
```

### Ключевые решения

**Решение 1: CurriculumManager** — прогрессивная сложность
- Начать с 5x5 (обучить каузальную модель)
- Перенести знания на 6x6, 8x8, 12x12
- Gate: success_rate ≥ 0.5 для перехода на следующий уровень
- TransferAgent (Stage 26) переносит causal links между размерами

**Решение 2: GoalAgent как base loop** — не EmbodiedAgent
- GoalAgent уже имеет episode loop с goal decomposition
- Добавить curiosity-directed exploration (заменить random на frontier)
- Добавить skill reuse (Stage 27)
- Расширить plan depth до 20

**Решение 3: Frontier-based exploration**
- Вместо random walk: BFS от текущей позиции к ближайшей неизвестной клетке
- Приоритет: ключевые объекты (key, door) > неисследованные области > случайные
- Curiosity bonus направляет exploration, но с пространственным bias

**Решение 4: Episode budget увеличен**
- 5x5: 50 эпизодов (warmup causal model)
- 6x6: 50 эпизодов (transfer + refinement)
- 8x8: 100 эпизодов
- 12x12: 200 эпизодов (gate test)
- Итого: ~400 эпизодов, ~30 мин на GPU

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 93 | success_5x5 | ≥ 0.8 | GoalAgent + curriculum warmup |
| 93 | success_8x8 | ≥ 0.5 | Transfer + continued learning |
| 93 | success_12x12 | ≥ 0.2 | Scalability proof |
| 94 | exploration_coverage_12x12 | ≥ 0.6 | Frontier exploration reaches 60%+ cells |
| 94 | causal_links_12x12 | ≥ 10 | Agent learns meaningful transitions |
| 95 | curriculum_speedup | ≥ 1.5x | Curriculum vs from-scratch on 8x8 |

## Модули

### CurriculumManager (NEW)
```python
class CurriculumManager:
    def __init__(self, levels: list[int] = [5, 6, 8, 12]):
        self.levels = levels
        self.current_level_idx = 0
        self.success_history: dict[int, list[bool]] = {}

    def should_advance(self) -> bool:
        """Advance if success_rate >= 0.5 over last 20 episodes."""

    def current_grid_size(self) -> int:
        """Current difficulty level."""

    def advance(self) -> int:
        """Move to next level, return new grid_size."""
```

### AutonomousAgent (NEW) — основной класс
```python
class AutonomousAgent:
    """Combines GoalAgent loop with curriculum and all capabilities."""

    def __init__(self, grid_size: int = 5):
        self.goal_agent: GoalAgent  # episode loop
        self.curriculum: CurriculumManager
        self.curiosity: CuriosityModule
        self.skills: SkillLibrary

    def run_curriculum(self, total_episodes: int = 400) -> CurriculumResult:
        """Run full curriculum from 5x5 to target."""

    def run_episode(self, env, mission: str) -> EpisodeResult:
        """One episode with integrated capabilities."""
```

## Файлы

| Файл | Описание |
|------|----------|
| `src/snks/language/curriculum_manager.py` | CurriculumManager |
| `src/snks/language/autonomous_agent.py` | AutonomousAgent facade |
| `tests/test_autonomous_agent.py` | Unit tests |
| `src/snks/experiments/exp93_curriculum.py` | Curriculum learning |
| `src/snks/experiments/exp94_exploration.py` | Frontier exploration |
| `src/snks/experiments/exp95_curriculum_speedup.py` | Speedup vs from-scratch |
| `demos/stage-36-spatial-abstraction.html` | Web demo |
