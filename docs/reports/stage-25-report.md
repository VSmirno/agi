# Stage 25: Autonomous Goal Composition

## Результат: PASS

## Что доказано
- Автономная декомпозиция целей через backward chaining (без hardcoded структуры задач)
- Каузальное обучение за 1 эпизод — агент открывает preconditions через exploration
- Каузальные знания переносятся между layout'ами (5x5 → 6x6) без потерь
- SKS-концепты (KEY_HELD, DOOR_OPEN, GOAL_PRESENT) обеспечивают type-based обобщение

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 58 | Decomposition accuracy | 1.000 (20/20) | >= 0.90 | PASS |
| 59 | Causal learning episodes | 1 | <= 5 | PASS |
| 60 | Multi-trial success rate | 100% from trial 1 | >= 60% by trial 5 | PASS |
| 61 | Transfer 5x5→6x6 | 100% rate, 34% fewer steps | >= 30% success | PASS |

## Ключевые решения
- **Always goal-driven mode** — вместо text-heuristic парсинга "use X to Y", агент всегда работает backward от финальной цели. Обоснование: робастнее и соответствует принципу "reason from goals, not text"
- **causal_min_observations=1** — достаточно одного наблюдения для обучения каузальной связи. Обоснование: DoorKey среда детерминистична, один пример надёжен
- **Exploration fallback** — при отсутствии каузальных знаний агент пробует все объекты и учится. Обоснование: холодный старт без seed-данных

## Компоненты
- `src/snks/language/goal_agent.py` — GoalAgent: backward chaining orchestrator
- `src/snks/language/blocking_analyzer.py` — BlockingAnalyzer: find_blocker + suggest_resolution
- `src/snks/language/causal_learner.py` — CausalLearner: before/after → observe_transition
- `src/snks/language/grid_perception.py` — state SKS IDs (KEY_PRESENT/HELD, DOOR_LOCKED/OPEN)
- `src/snks/language/grid_navigator.py` — PathResult + plan_path_ex with PathStatus enum
- `src/snks/agent/causal_model.py` — query_by_effect() reverse lookup

## Тесты
- 22 новых + 27 существующих unit-тестов PASS

## Следующий этап
- Stage 26: Transfer Learning — каузальные знания переносятся между разными средами
