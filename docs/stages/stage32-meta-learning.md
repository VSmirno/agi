# Stage 32: Meta-Learning — Design Specification

**Дата:** 2026-03-31
**Статус:** IN PROGRESS
**Ветка:** stage32-meta-learning
**Эксперименты:** exp80, exp81, exp82

---

## Цель

Доказать, что СНКС может **выбирать оптимальную стратегию обучения** в зависимости
от характеристик задачи. Meta-learner наблюдает за производительностью агента и
динамически переключает стратегии (curiosity, skill-based, few-shot, analogy).

## Что доказывает

1. Агент **оценивает характеристики задачи** (наличие демо, плотность знаний, покрытие)
2. На основе оценки **выбирает стратегию** из имеющегося арсенала
3. **Адаптирует гиперпараметры** (epsilon, analogy_threshold) по ходу эпизода
4. Meta-learner **улучшает performance** vs. fixed-strategy baseline

## Философская корректность

- Meta-learning = самонаблюдение + адаптация, НЕ gradient-based MAML
- Решения основаны на метриках (prediction error, coverage, skill success rate)
- Нет backpropagation — только rule-based policy с адаптивными порогами
- Согласовано с MetacogConfig (Stage 13) — расширяем FSM новыми сигналами

---

## Подходы (brainstorming)

### Подход A: Strategy Selector (rule-based)
MetaLearner получает task_profile (features задачи) и возвращает strategy config.
Набор правил: if demos_available → few-shot; if coverage < 0.3 → curiosity; etc.

**Trade-off:** Простой, детерминированный, легко тестировать. Но правила хардкожены.

### Подход B: Bandit-based Strategy Selection
Multi-armed bandit (UCB1) выбирает стратегию по accumulated reward.

**Trade-off:** Адаптивный, но требует множество эпизодов для сходимости.

### Подход C: Performance Monitoring + Adaptive Thresholds
MetaLearner мониторит метрики в реальном времени и адаптирует гиперпараметры.
Если skill success rate падает → увеличить exploration. Если coverage стагнирует →
переключить на analogy.

**Trade-off:** Реактивный и гибкий. Сложнее тестировать.

### Выбор: Подход A + элементы C

**Обоснование:**
- Rule-based selector для начального выбора стратегии (предсказуемый, тестируемый)
- Adaptive thresholds для runtime коррекции (epsilon, analogy_threshold)
- Bandit отвергнут: требует слишком много эпизодов для задач СНКС (1-5 эпизодов)

---

## Архитектура

### Новые модули

#### 1. `TaskProfile` (dataclass)
```python
@dataclass
class TaskProfile:
    has_demos: bool = False          # демонстрации доступны
    n_demos: int = 0                 # количество демонстраций
    known_skills: int = 0            # навыков в библиотеке
    causal_links: int = 0            # каузальных связей в модели
    state_coverage: float = 0.0      # доля посещённых состояний
    mean_prediction_error: float = 1.0  # средняя ошибка предсказания
    episodes_completed: int = 0      # завершённых эпизодов
    last_success: bool = False       # последний эпизод успешен
```

#### 2. `StrategyConfig` (dataclass)
```python
@dataclass
class StrategyConfig:
    strategy: str                    # "curiosity" | "skill" | "few_shot" | "explore"
    curiosity_epsilon: float = 0.2
    analogy_threshold: float = 0.7
    exploration_budget: int = 60
    use_analogy: bool = True
    reason: str = ""                 # explanation for logging
```

#### 3. `MetaLearner` (main class)
```python
class MetaLearner:
    def __init__(self, adaptation_rate: float = 0.1):
        ...

    def select_strategy(self, profile: TaskProfile) -> StrategyConfig:
        """Select optimal strategy based on task characteristics."""

    def adapt(self, profile: TaskProfile, result: EpisodeResult) -> StrategyConfig:
        """Adapt strategy after observing episode outcome."""

    def profile_from_agent(self, agent, env) -> TaskProfile:
        """Extract task profile from current agent state."""
```

#### 4. `EpisodeResult` (dataclass)
```python
@dataclass
class EpisodeResult:
    success: bool
    steps: int
    skills_used: int
    new_states_discovered: int
    prediction_error: float
```

### Алгоритм

#### Strategy Selection (initial):
```
IF has_demos AND known_skills == 0:
    → "few_shot" (bootstrap from demos)
ELIF known_skills >= 2 AND causal_links >= 5:
    → "skill" (exploit existing knowledge)
ELIF state_coverage < 0.3:
    → "curiosity" (need more exploration)
ELSE:
    → "explore" (fallback — backward chaining + exploration)
```

#### Adaptive Thresholds (after each episode):
```
IF NOT success AND strategy == "skill":
    epsilon += adaptation_rate  (more exploration needed)
    analogy_threshold -= 0.05   (accept weaker analogies)
IF success AND steps < budget/2:
    epsilon -= adaptation_rate  (exploitation working well)
IF new_states == 0 AND coverage < 0.5:
    switch to "curiosity"       (stagnation detected)
```

---

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 80 | strategy_selection_accuracy | ≥ 0.8 | Правильный выбор стратегии для известных сценариев |
| 80 | profile_extraction | all correct | TaskProfile корректно извлекается из агента |
| 81 | adaptation_improves | True | Адаптация улучшает performance vs. fixed config |
| 81 | meta_vs_fixed_ratio | ≥ 1.2 | MetaLearner в 1.2x+ эффективнее fixed strategy |
| 82 | multi_task_accuracy | ≥ 0.8 | Точность на 3+ разных типах задач |

---

## Файлы

| Файл | Описание |
|------|----------|
| `src/snks/language/meta_learner.py` | MetaLearner, TaskProfile, StrategyConfig, EpisodeResult |
| `tests/test_meta_learner.py` | Unit tests |
| `src/snks/experiments/exp80_strategy_selection.py` | Strategy selection accuracy |
| `src/snks/experiments/exp81_adaptation.py` | Adaptive vs. fixed comparison |
| `src/snks/experiments/exp82_multi_task.py` | Multi-task performance |
| `demos/stage-32-meta-learning.html` | Web demo |

---

## Зависимости

- `CuriosityAgent` (Stage 29) — curiosity strategy
- `SkillLibrary` / `SkillAgent` (Stage 27) — skill strategy
- `FewShotAgent` (Stage 30) — few-shot strategy
- `AnalogicalReasoner` (Stage 28) — analogy support
- `CausalWorldModel` (Stage 6) — causal link metrics
- `CuriosityModule` (Stage 29) — coverage metrics
