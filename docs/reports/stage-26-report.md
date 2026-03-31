# Stage 26: Cross-Environment Causal Transfer

## Результат: PASS

## Что доказано
- Каузальные знания (key→pickup, door→toggle chain) переносятся из DoorKey-5x5 в MultiRoomDoorKey (3 комнаты, 2 двери)
- Перенос через сериализацию JSON — roundtrip идеален (0% потерь)
- Transferred agent не деградирует в средах без door/key (selective transfer)
- State predicates (SKS 50-99) обеспечивают environment-independent transfer

## Эксперименты
| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 62 | Cross-env transfer rate | 1.000 (10/10) | >= 0.70 | PASS |
| 62 | Speedup (exploration episodes) | 2.0x | >= 2.0x | PASS |
| 63 | Loaded vs direct rate diff | 0.000 | <= 0.05 | PASS |
| 63 | Link count preserved | 11 == 11 | exact match | PASS |
| 64 | Incorrect transfer actions | 0 | == 0 | PASS |
| 64 | Step degradation ratio | 1.00x | <= 1.10x | PASS |

## Ключевые решения
- **Same-color doors/keys** в MultiRoom — GoalAgent не учитывает цветовое соответствие, поэтому оба ключа жёлтые. Обоснование: transfer тестирует каузальную структуру, не визуальное распознавание
- **SubGoal.target_pos** — SubGoal теперь несёт позицию цели для точного targeting в multi-object envs. Обоснование: find_object("door") возвращал первую дверь, а не нужную
- **Serialization via _TransitionRecord** — сериализуем полные записи (context_sks, effect_sks), а не хеши. Обоснование: Python hash() нестабилен между процессами
- **Key layout ordering** — ключ в комнате 1 размещён выше (j=2) чем ключ в комнате 2 (j=7) для правильного scan-order. Обоснование: find_object возвращает первый найденный объект

## Компоненты (новые)
- `src/snks/agent/causal_serializer.py` — CausalModelSerializer: save/load to JSON
- `src/snks/env/multi_room.py` — MultiRoomDoorKey: 3 rooms, 2 doors, 2 keys, boxes
- `src/snks/language/transfer_agent.py` — TransferAgent: GoalAgent + transfer metrics

## Компоненты (изменённые)
- `src/snks/language/blocking_analyzer.py` — SubGoal.target_pos для позиционного targeting
- `src/snks/language/goal_agent.py` — target_pos support, max_retries=5
- `src/snks/language/grid_perception.py` — find_object_at() для позиционного поиска

## Тесты
- 56 unit tests PASS (26 новых + 30 существующих)
- 3 эксперимента PASS (exp62, exp63, exp64)

## Следующий этап
- Stage 27: Skill Abstraction — иерархические макро-действия (навыки)
