# Stage 29: Curiosity-Driven Exploration

## Результат: PASS

## Что доказано
- Внутренняя награда `r_int = 1.0 / (1 + visit_count)` корректно убывает с посещениями
- CuriosityAgent с 5-шаговым lookahead посещает 100% клеток Empty-8x8 за 60 шагов
- CuriosityAgent в 2.48x эффективнее random при исследовании
- CuriosityAgent успешно решает DoorKey: 100% с навыками + curiosity

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 71 | r_int(new) == 1.0 | 1.000 | == 1.0 | PASS |
| 71 | r_int(repeated) == 0.5 | 0.500 | == 0.5 | PASS |
| 71 | distinct states ≥ 10 | 10 | ≥ 10 | PASS |
| 72 | curious/random ratio | 2.48x | ≥ 1.3 | PASS |
| 72 | coverage | 1.000 | ≥ 0.25 | PASS |
| 73 | DoorKey success rate | 1.000 | ≥ 0.90 | PASS |
| 73 | avg skills/ep | 1.0 | ≥ 1.0 | PASS |

## Ключевые решения

- **State key = frozenset(sks) | {pos_token}**: позиция кодируется как 10000 + x*100 + y — позволяет различать клетки с одинаковыми SKS предикатами в пустой комнате
- **5-step lookahead с discount 0.9**: вместо 1-шагового просмотра — позволяет "видеть за углом" и выбирать траектории к новым областям
- **nav_budget = min(max_steps, 60)**: Phase 1 ограничен 60 шагами, Phase 2 (object interaction) всегда получает бюджет
- **Wrapper for normal navigation**: `self._env.step` патчится для автоматического обсёрва любопытства при навигации
- **_curiosity_wrapper_active flag**: предотвращает двойной обсёрв когда wrapper активен

## Файлы изменены

- `src/snks/language/curiosity_module.py` — NEW: count-based curiosity
- `src/snks/language/curiosity_agent.py` — NEW: CuriosityAgent + 5-step lookahead exploration
- `tests/test_curiosity_module.py` — 9 тестов PASS
- `src/snks/experiments/exp71/72/73_*.py` — все PASS

## Следующий этап

Stage 30: Few-Shot Learning — обучение из 1-3 демонстраций.
Ключевой вопрос: может ли агент наблюдать чужое поведение и скопировать навык без повторного исследования?
