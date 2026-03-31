# Stage 30: Few-Shot Learning

## Результат: PASS

## Что доказано
- Агент учится решать задачу из 1 демонстрации (one-shot), без собственного опыта
- FewShotLearner извлекает каузальные связи и навыки из наблюдённых траекторий
- Навыки из демонстрации DoorKey автоматически переносятся на CardGate через аналогии
- Работает через существующий CausalWorldModel — никакой новой learning mechanism

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 74 | skill_accuracy | 1.00 | ≥ 0.9 | PASS |
| 74 | composite_found | True | True | PASS |
| 75 | 1_demo_success | 1.000 | ≥ 0.5 | PASS |
| 75 | 3_demo_success | 1.000 | ≥ 0.8 | PASS |
| 76 | cross_env_success | 1.000 | ≥ 0.7 | PASS |

## Ключевые решения
- **Demonstration → Causal Model** (не Imitation Learning) — каузальные связи из наблюдений идентичны тем, что агент учит сам. Философски корректно: наблюдение = та же каузальная evidence.
- **Reuse existing infrastructure** — CausalWorldModel.observe_transition() работает для чужих действий так же, как для своих.
- **FewShotAgent extends CuriosityAgent** — получает полный стек: skills, analogies, curiosity, backward chaining. Few-shot знания bootstrapят каузальную модель.

## Веб-демо
- `demos/stage-30-few-shot.html` — интерактивная визуализация pipeline: наблюдение демо → извлечение навыков → решение нового layout

## Файлы изменены
- `src/snks/language/demonstration.py` — DemoStep, Demonstration, DemonstrationRecorder
- `src/snks/language/few_shot_learner.py` — FewShotLearner: demos → causal + skills
- `src/snks/language/few_shot_agent.py` — FewShotAgent extends CuriosityAgent
- `tests/test_few_shot.py` — 12 unit tests
- `src/snks/experiments/exp74_one_shot_skill.py`
- `src/snks/experiments/exp75_few_shot_goal.py`
- `src/snks/experiments/exp76_few_shot_transfer.py`
- `demos/stage-30-few-shot.html`

## Следующий этап
- Stage 31: Abstract Pattern Reasoning — Raven's-style pattern completion на концептах
