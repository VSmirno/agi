# Stage 35: Integration Demo

**Статус:** IN PROGRESS
**Ветка:** stage35-integration-demo
**Эксперименты:** exp89, exp90, exp91
**Дата:** 2026-03-31

---

## Что доказывает

Все capabilities СНКС (Stages 25-34) работают **вместе** в одном когерентном агенте:
1. MetaLearner выбирает стратегию → CuriosityAgent/SkillAgent/FewShotAgent
2. SkillLibrary + AnalogicalReasoner обеспечивают transfer между доменами
3. HierarchicalPlanner генерирует 1000+ step планы
4. AgentCommunicator позволяет обмениваться знаниями с другими агентами
5. AbstractPatternReasoner решает абстрактные задачи
6. Все без backpropagation, через концепты (SKS), не текст

---

## Архитектура

### IntegratedAgent — unified facade

```python
class IntegratedAgent:
    """Объединяет все capabilities Stages 25-34 в одном агенте."""

    # Core knowledge
    causal_model: CausalWorldModel
    skill_library: SkillLibrary

    # Strategy selection
    meta_learner: MetaLearner

    # Agent capabilities
    curiosity: CuriosityModule
    analogical_reasoner: AnalogicalReasoner
    pattern_reasoner: AbstractPatternReasoner

    # Communication
    communicator: AgentCommunicator

    # Planning
    planner: HierarchicalPlanner

    # Methods
    def profile() → TaskProfile
    def select_strategy() → StrategyConfig
    def run_episode(env, instruction, max_steps) → IntegrationResult
    def plan_to_goal(goal_sks, current_sks) → PlanGraph
    def communicate(other_agent) → int
    def transfer_to(target_domain) → TransferResult
    def solve_pattern(matrix) → (Tensor, float)
```

---

## Подходы

### Подход A: Монолитный агент
- Один класс со всеми методами
- **Con:** сложный, хрупкий

### Подход B: Facade + Pipeline (рекомендуемый) ✓
- IntegratedAgent как тонкий facade над существующими компонентами
- Каждая capability остаётся self-contained
- MetaLearner координирует выбор стратегии
- **Pro:** minimal new code, reuses everything

**Выбран: Подход B** — максимальное переиспользование кода.

---

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 89 | capabilities_count | = 10 | Все 10 capabilities доступны через IntegratedAgent |
| 89 | strategy_pipeline | ≥ 0.9 | MetaLearner → correct agent → success |
| 90 | end_to_end_success | ≥ 0.9 | Full pipeline: profile → strategy → execute → learn |
| 90 | cross_capability | ≥ 0.8 | Capabilities комбинируются (planning + skills + transfer) |
| 91 | multi_agent_integration | ≥ 0.9 | 2 IntegratedAgents кооперируются |
| 91 | zero_backprop | = True | Никакого gradient descent нигде |

---

## Модули

1. `src/snks/language/integrated_agent.py` — IntegratedAgent facade
2. `tests/test_integrated_agent.py` — unit tests
3. `src/snks/experiments/exp89_capabilities.py`
4. `src/snks/experiments/exp90_end_to_end.py`
5. `src/snks/experiments/exp91_full_integration.py`
