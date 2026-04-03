# Stage 59: SDM Learned Color Matching — LockedRoom

**Дата:** 2026-04-03
**Статус:** SPEC DRAFT
**Предыдущий этап:** Stage 58 (SDM Retrofit — DoorKey)
**Milestone:** M4 — Масштаб (6/7)

---

## Мотивация

Stage 58 показал что SDM infrastructure работает, но DoorKey слишком прост — heuristic `key→door→goal` даёт 100%, SDM не может быть лучше оптимального. Нужна среда где heuristic **не знает** правильный ответ, и SDM должен **обучиться** из опыта.

LockedRoom (19×19) — идеальный полигон:
- 6 комнат, 6 дверей разных цветов, 1 locked door
- 1 ключ в рандомной комнате, цвет = цвет locked door
- Каждый seed — новая конфигурация цветов
- Heuristic не знает какой ключ к какой двери → random choice ~16%
- SDM должен обучить паттерн `same_color(key, door) → success`

## Архитектура

```
LockedRoom (19×19, partial obs 7×7)
    │
    ▼
SpatialMap (exploration + object discovery)
    │
    ▼
ColorStateEncoder (VSA)
  - agent_pos, has_key, key_color, visible_doors[], locked_door_color
    │
    ▼
SDM Color-Transition Memory
  - writes: (key_color, door_color, action_result) → reward
  - reads: given key_color → which door_color gives reward?
    │
    ▼
SubgoalPlanner
  - Phase B: mission parser → target key color → find room → pickup → goto locked door → toggle
  - Phase A: exploration → try combinations → SDM recall → pick best
    │
    ▼
BFS Navigation (символическое, как в Stage 58)
```

### Компоненты

**ColorStateEncoder (VSA):**
- Кодирует: agent_pos (quantized), has_key (bool), key_color (6 вариантов), door_colors (set), locked_door_color, exploration_pct
- Output: 256-dim binary VSA vector
- Расширение AbstractStateEncoder из Stage 58, но с color awareness

**SDM Color-Transition Memory:**
- Locations: 1000 (как Stage 58)
- Dimension: 256
- Write: после каждого toggle двери записываем (key_color, door_color, success/fail)
- Read: дано key_color → query SDM → получаем door_color с максимальным reward
- Обобщение: SDM должен вывести паттерн same_color из ~50-100 exploration episodes

**SDMLockedRoomAgent:**
- Новый агент, не расширяет SDMDoorKeyAgent
- Использует SpatialMap, FrontierExplorer, GridPathfinder из Stage 58
- Subgoal chain: EXPLORE → FIND_KEY → GOTO_KEY → PICKUP → FIND_LOCKED_DOOR → GOTO_DOOR → TOGGLE → GOTO_GOAL

**Mission Parser (Phase B only):**
- Regex: `"get the (\w+) key from the (\w+) room"`
- Извлекает target_key_color, target_room_color
- SDM использует как дополнительный signal

## Два этапа Proof

### Phase B — с mission text (первый, проще)

1. **Exploration (training):** 50-100 seeds
   - Парсим mission → знаем target key color
   - Exploration: frontier → находим комнаты → находим ключ нужного цвета → pickup → находим locked door → toggle
   - SDM записывает: (key_color, locked_door_color, mission_text_features) → success
2. **Evaluation:** 200 unseen seeds
   - SDM + parsed mission → направленный план
   - Метрика: success rate
3. **Ablation:**
   - Pure heuristic (random door) → expect ~16%
   - SDM trained + mission → expect ≥70%
   - SDM untrained + mission → expect ~16%

### Phase A — без mission text (второй, сложнее)

1. **Exploration (training):** 100-200 seeds
   - Агент исследует LockedRoom, пробует ключи на дверях
   - Неправильный ключ → SDM записывает (key_color, door_color) → fail
   - Правильный ключ → SDM записывает (key_color, door_color) → success
   - SDM обобщает: same_color → success
2. **Evaluation:** 200 unseen seeds
   - SDM recall без mission → для каждого найденного ключа, query SDM: "какая дверь?"
   - Если SDM обобщил same_color: сразу идёт к locked door того же цвета
3. **Ablation:**
   - Pure heuristic (random door) → expect ~16%
   - SDM trained → expect ≥50%
   - SDM untrained → expect ~16%

## Subgoal Chain

```
EXPLORE_ROOMS → FIND_KEY → GOTO_KEY_ROOM → PICKUP_KEY → FIND_LOCKED_DOOR → GOTO_LOCKED_DOOR → TOGGLE_DOOR → GOTO_GOAL
```

8 subgoals. BFS handles navigation (HOW), SDM handles color selection (WHAT).

## Gate Criteria

| Метрика | Порог | Обоснование |
|---------|-------|-------------|
| SDM trained Phase B (mission) | ≥ 70% unseen seeds | Mission даёт подсказку, SDM должен использовать |
| SDM trained Phase A (no mission) | ≥ 50% unseen seeds | Чистое обучение из опыта |
| SDM trained > heuristic | p < 0.05, 200 seeds | Статистическая значимость |
| SDM writes | ≥ 1000 transitions | Capacity gate |
| Heuristic baseline measured | expect ~16% | Random choice из 6 дверей |

## Файлы

| Файл | Действие | Описание |
|------|----------|----------|
| `src/snks/agent/sdm_lockedroom_agent.py` | NEW | SDMLockedRoomAgent, ColorStateEncoder, MissionParser |
| `src/snks/experiments/exp113_sdm_lockedroom.py` | NEW | Exploration + eval + ablation для Phase B и A |
| `tests/test_stage59_lockedroom.py` | NEW | Unit тесты компонентов |
| Stage 58 код | НЕ ТРОГАЕМ | Изоляция от предыдущего этапа |

## Риски и митигации

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| 19×19 exploration слишком медленная | Средняя | FrontierExplorer уже работает на MultiRoom, добавить room-level planning |
| SDM не обобщает same_color паттерн | Средняя | Увеличить training episodes, проверить VSA encoding качество |
| MiniGrid door: нужно drop неправильный ключ | Высокая | Добавить DROP_KEY subgoal при fail, как в Stage 57 |
| ROCm GPU segfault | Высокая | CPU first (как Stage 58), GPU как бонус |

## Эксперименты на minipc

Все эксперименты запускаются на minipc (evo-x2). Деплой через git push/pull. CPU first.

| Exp | Phase | Config | Seeds | Gate |
|-----|-------|--------|-------|------|
| 113a | B (mission) | 50 train, 200 eval | unseen | ≥70% |
| 113b | B ablation | heuristic vs trained vs untrained | 200 | p<0.05 |
| 113c | A (no mission) | 100 train, 200 eval | unseen | ≥50% |
| 113d | A ablation | heuristic vs trained vs untrained | 200 | p<0.05 |
