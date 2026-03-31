# Stage 37: Multi-Room Navigation

## Результат: PASS

## Что доказано
- GoalAgent с backward chaining проходит через **4 комнаты** (MultiRoom-N4) с **96% success**
- BlockingAnalyzer обрабатывает closed (not locked) doors — toggle без ключа
- Каузальные знания из DoorKey-5x5 переносятся на multi-room среды
- Увеличение max_retries до 10 позволяет последовательно открывать 3+ дверей

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 96 | doorkey_5x5 | 1.000 | ≥ 0.8 | PASS |
| 96 | multiroom_n2 | 1.000 | ≥ 0.5 | PASS |
| 96 | multiroom_n4 | 0.960 | ≥ 0.2 | PASS |

## Ключевые решения
- **Closed doors** в BlockingAnalyzer: не locked, не open → toggle без prerequisite
- **Distance-prioritized blockers**: ближайшая дверь к агенту обрабатывается первой
- **max_retries=10**: для N4 (4 rooms, 3+ doors) нужно больше итераций backward chaining
- **UnlockPickup не gated**: другой тип задачи (pickup, не navigate-to-goal)

## Файлы изменены
- `src/snks/language/blocking_analyzer.py` — closed door handling, distance priority
- `src/snks/language/goal_agent.py` — max_retries 5→10
- `src/snks/language/autonomous_agent.py` — MULTIROOM_ENV_MAP
- `src/snks/experiments/exp96_multiroom.py`

## Следующий этап
- **Stage 38: Curriculum Learning** — автоматический curriculum от DoorKey через MultiRoom с performance tracking
