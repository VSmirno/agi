# Stage 32: Meta-Learning

## Результат: PASS

## Что доказано
- MetaLearner корректно выбирает оптимальную стратегию для 5 типов задач (100% accuracy)
- Адаптивные пороги (epsilon, analogy_threshold) улучшают performance vs fixed baseline
- MetaLearner: 9/10 successes vs лучший fixed (skill): 7/10 — ratio 1.29x
- Task profiling извлекает характеристики задачи: coverage, skills, demos, prediction error
- Rule-based policy + adaptive thresholds, без backpropagation

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 80 | strategy_selection_accuracy | 1.000 | ≥ 0.8 | PASS |
| 80 | profile_extraction | 10/10 | all correct | PASS |
| 81 | adaptation_improves | True | True | PASS |
| 81 | meta_vs_fixed_ratio | 1.29 | ≥ 1.2 | PASS |
| 82 | multi_task_accuracy | 1.000 | ≥ 0.8 | PASS |

## Ключевые решения
- **Rule-based selector** (не bandit, не MAML) — предсказуемый, тестируемый, работает с 1-5 эпизодами. Бандит требует десятки эпизодов для сходимости, что не реалистично для задач СНКС.
- **Adaptive thresholds** — epsilon и analogy_threshold адаптируются по результатам эпизодов. Простые правила: failure → increase exploration, success → decrease.
- **TaskProfile как единая точка наблюдения** — все метрики агента собираются в один dataclass для strategy selection.
- **4 стратегии** соответствуют Stages 25-30: explore (GoalAgent), curiosity (CuriosityAgent), skill (SkillAgent), few_shot (FewShotAgent).

## Веб-демо
- `demos/stage-32-meta-learning.html` — интерактивная симуляция 10 эпизодов с визуализацией стратегий, gauge'ами параметров и сравнением с fixed strategies.

## Файлы изменены
- `src/snks/language/meta_learner.py` — MetaLearner, TaskProfile, StrategyConfig, EpisodeResult
- `tests/test_meta_learner.py` — 18 unit tests
- `src/snks/experiments/exp80_strategy_selection.py`
- `src/snks/experiments/exp81_adaptation.py`
- `src/snks/experiments/exp82_multi_task.py`
- `demos/stage-32-meta-learning.html`
- `docs/stages/stage32-meta-learning.md`

## Следующий этап
- Stage 33: Multi-Agent Communication — агенты обмениваются концептами (не словами). Ключевой вопрос: можно ли передать каузальные знания от одного агента другому через HAC embeddings?
