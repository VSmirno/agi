# Stage 57: Long Subgoal Chains — KeyCorridor

## Результат: PASS

**Ветка:** `stage57-long-subgoal-chains`
**Milestone:** M4 — Масштаб (четвёртый этап)

## Что доказано
- Агент решает задачи с **5-6 последовательными subgoals** с prerequisite-зависимостями
- KeyCorridorS4R3 (10×10): 40.0% на 200 random seeds — gate ≥40% **PASS**
- KeyCorridorS3R3 (7×7): 54.0% на 200 random seeds — gate ≥50% **PASS**
- KeyCorridorS5R3 (13×13): 39.5% stretch test
- Prerequisite-graph planning: locked door → requires matching key → backward chaining
- MiniGrid 3.0 quirk: key not consumed on door toggle → DROP_KEY subgoal needed
- 8-phase state machine: EXPLORE → GOTO_KEY → PICKUP_KEY → GOTO_LOCKED_DOOR → OPEN_DOOR → DROP_KEY → GOTO_GOAL → PICKUP_GOAL

## Эксперименты

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 111a | KeyCorridorS4R3 (200 seeds) | 80/200 = 40.0%, mean 39.5 steps | ≥40% | PASS |
| 111b | KeyCorridorS3R3 (200 seeds) | 108/200 = 54.0%, mean 31.3 steps | ≥50% | PASS |
| 111c | KeyCorridorS5R3 (200 seeds) | 79/200 = 39.5%, mean 50.7 steps | stretch | N/A |
| 111d | Min subgoals (successful) | 5-6 per episode | ≥5 | PASS |

## Ключевые решения

1. **Prerequisite-graph ChainPlanner** — backward chaining от goal через locked door к key. Универсальный подход, не привязан к конкретной среде.
2. **MiniGrid 3.0 key persistence** — Door.toggle() не вызывает env.carrying=None. Ключ остаётся у агента. Решение: DROP_KEY subgoal перед PICKUP_GOAL.
3. **Reactive planning** — план перестраивается при каждом обнаружении нового объекта. Нет предварительного знания о карте.
4. **Automatic door handling** — unlocked doors (state=1) toggled автоматически при навигации, не являются subgoals.
5. **BlockedUnlockPickup** — требует 9+ subgoals (drop key → move blocker → pick key → open). Beyond Stage 57 scope, отложен.

## Анализ неудач (60% failures на S4R3)

Основные причины неудач:
- **Exploration timeout:** на 10×10 grid с 7×7 view агент не успевает найти ключ за 480 шагов
- **Key in remote room:** ключ может быть в дальней комнате, требующей прохождения через 4+ unlocked doors
- **BFS через объекты:** мяч/ключ/коробка блокируют BFS-пути, создавая тупики

Улучшения для Stage 58+ (SDM scaling): SDM world model может предсказывать расположение объектов, сокращая exploration.

## Веб-демо
- `demos/stage-57-keycorridor.html` — Canvas replay 4 эпизодов с subgoal chain bar

## Файлы изменены
- `src/snks/agent/keycorridor_agent.py` — NEW: ChainPlanner, KeyCorridorAgent, KeyCorridorEnv
- `tests/test_stage57_long_chains.py` — NEW: 23 теста
- `src/snks/experiments/exp111_keycorridor.py` — NEW: gate experiments
- `demos/stage-57-keycorridor.html` — NEW: web demo
- `demos/index.html` — UPDATED: карточка Stage 57

## Следующий этап
- **Stage 58: SDM Scaling** — SDM capacity ≥1000 unique transitions. Масштабирование памяти для поддержки больших сред.
