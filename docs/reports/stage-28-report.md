# Stage 28: Analogical Reasoning

## Результат: PASS

## Что доказано
- Агент обнаруживает структурное сходство между навыками ключ/дверь и карта/ворота
- Аналогия вычисляется через статический ROLE_REGISTRY (4 роли: instrument/blocker)
- Адаптированные навыки применяются к новой среде без дополнительного обучения
- 100% success rate на CardGateWorld, 3.9 аналогии per episode
- Нет регрессии: DoorKey=1.000, MultiRoom=1.000

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 68 | analogy found | True (3 analogies) | True | PASS |
| 68 | best similarity | 0.750 | ≥ 0.70 | PASS |
| 69 | success rate CardGateWorld | 1.000 | ≥ 0.80 | PASS |
| 69 | avg analogies/ep | 3.9 | ≥ 1.0 | PASS |
| 70 | DoorKey success rate | 1.000 | ≥ 0.90 | PASS |
| 70 | MultiRoom success rate | 1.000 | ≥ 0.80 | PASS |

## Ключевые решения

- **Purple key/door как аналог**: CardGateWorld использует стандартные MiniGrid объекты с другим цветом — минимальный рефактор GridPerception (цветовая ветка)
- **Static ROLE_REGISTRY**: 4 структурные роли захардкожены вместо динамического поиска — детерминированная similarity
- **AnalogicalReasoner как отдельный модуль**: не расширение SkillLibrary, SRP сохранён
- **Similarity = matched_predicates / total**: Jaccard по роли ≈ 0.75 для всех навыков (3/4 предикатов имеют аналог)
- **Analogy step attempt==0**: аналогии пробуются 1 раз как skills, fallback на GoalAgent

## Файлы изменены

- `src/snks/language/role_registry.py` — NEW: ROLE_REGISTRY, SOURCE_TO_TARGET_SKS
- `src/snks/language/analogical_reasoner.py` — NEW: AnalogyMap, AnalogicalReasoner
- `src/snks/env/card_gate_world.py` — NEW: purple key/door env
- `src/snks/language/grid_perception.py` — +SKS 55-58, цветовая ветка, "card"/"gate" aliases
- `src/snks/language/blocking_analyzer.py` — gate blocker + card prerequisite
- `src/snks/language/skill_agent.py` — analogy step (шаг 3)
- `src/snks/experiments/exp68/69/70_*.py` — все PASS

## Следующий этап

Stage 29: Curiosity-Driven Exploration — агент исследует без внешней награды,
мотивируясь новизной состояний. Ключевой вопрос: информационный прирост от
непосещённых ячеек как внутренний сигнал награды.
