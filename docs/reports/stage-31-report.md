# Stage 31: Abstract Pattern Reasoning

## Результат: PASS

## Что доказано
- СНКС решает Raven's-style pattern completion на концептах (СКС) без backpropagation
- HAC unbind извлекает трансформационные правила из последовательностей концептов
- Правила консистентны по строкам/столбцам (consistency = 1.000)
- HAC bind применяет правило для предсказания недостающего элемента
- Аналогия A:B :: C:? решается алгебраически через unbind + bind
- Работает с двойными правилами (row + column одновременно)

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 77 | rule_found_rate | 1.000 | ≥ 0.8 | PASS |
| 77 | rule_consistency | 1.000 | ≥ 0.7 | PASS |
| 78 | completion_accuracy | 1.000 | ≥ 0.8 | PASS |
| 78 | analogy_accuracy | 1.000 | ≥ 0.8 | PASS |
| 79 | multi_rule_accuracy | 1.000 | ≥ 0.7 | PASS |

## Ключевые решения
- **HAC Transform Discovery** (не символьный перебор) — трансформации = unbind(e_i, e_{i+1}), консистентность проверяется cosine similarity. Это чистая алгебра на концептных векторах, полностью согласована с философией СНКС.
- **Row + Column decomposition** — для 3x3 матриц правила ищутся по обеим осям, предсказания объединяются через bundle. Позволяет решать задачи с двумя независимыми трансформациями.
- **Reuse HACEngine** — bind/unbind/bundle из Stage 9 используются без изменений. Нет новых примитивов — только новая комбинация существующих операций.
- **No training required** — reasoner работает чисто алгебраически, без обучения.

## Веб-демо
- `demos/stage-31-abstract-pattern.html` — интерактивная демо с 3 режимами: 3x3 matrix, analogy A:B::C:?, dual-rule. HAC algebra реализована на JS для визуализации в браузере.

## Файлы изменены
- `src/snks/language/pattern_element.py` — PatternElement, PatternMatrix, TransformRule
- `src/snks/language/abstract_pattern_reasoner.py` — AbstractPatternReasoner
- `tests/test_abstract_pattern.py` — 12 unit tests
- `src/snks/experiments/exp77_pattern_rules.py` — rule discovery
- `src/snks/experiments/exp78_pattern_completion.py` — completion + analogy
- `src/snks/experiments/exp79_multi_rule.py` — dual-rule patterns
- `demos/stage-31-abstract-pattern.html` — web demo
- `docs/stages/stage31-abstract-pattern-reasoning.md` — spec

## Следующий этап
- Stage 32: Meta-Learning — learning to learn, адаптация стратегий обучения. Агент должен научиться выбирать оптимальную стратегию обучения (exploration vs exploitation, few-shot vs curiosity) в зависимости от характеристик задачи.
