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
- Output: 512-dim binary VSA vector (совпадает с VSACodebook default dim=512 из Stage 58)
- Расширение AbstractStateEncoder из Stage 58, но с color awareness

**SDM Color-Transition Memory:**
- Locations: 1000 (как Stage 58)
- Dimension: 512 (совпадает с VSACodebook)
- API mapping на существующий SDMMemory interface:
  - `state` = VSA(key_color) — закодированный цвет ключа
  - `action` = VSA(door_color) — закодированный цвет двери
  - `reward` = 1.0 (success) / -1.0 (fail)
  - Write: `sdm.write(state=VSA(key_color), action=VSA(door_color), reward=±1.0)`
  - Read: `sdm.read_reward(state=VSA(key_color), action=VSA(door_color))` для всех door_colors → argmax
- Обобщение: SDM должен вывести паттерн same_color из ~50-100 exploration episodes

**SDMLockedRoomAgent:**
- Новый агент, не расширяет SDMDoorKeyAgent
- Использует SpatialMap, FrontierExplorer, GridPathfinder из Stage 58
- **Важно:** SpatialMap.find_objects() возвращает только первый объект каждого типа. Для LockedRoom нужен обход всех cells через `spatial_map.grid` для enumerate всех дверей/ключей с цветами.
- Subgoal chain: EXPLORE → FIND_KEY → GOTO_KEY → PICKUP → FIND_LOCKED_DOOR → GOTO_DOOR → TOGGLE → (fail? DROP_KEY → EXPLORE) → GOTO_GOAL
- DROP_KEY subgoal: если toggle не сработал (wrong color), drop ключ и вернуться к exploration
- Max steps per episode: 1000 (19×19 grid с 6 комнатами требует ~500-800 шагов)

**Mission Parser (Phase B only):**
- Actual MiniGrid LockedRoom mission format: `"get the {color} key from the {color} room, unlock the {color} door and go to the goal"`
- Regex: `r"get the (\w+) key from the (\w+) room, unlock the (\w+) door"`
- Извлекает target_key_color, source_room_color, target_door_color
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
   - Heuristic WITHOUT mission (random door choice) → expect ~16%
   - Heuristic WITH mission parsing (парсит цвет, но без SDM memory) → ожидаем высокий baseline, это BFS+regex
   - SDM trained + mission → expect ≥70%, должен быть >= heuristic+mission
   - SDM untrained + mission → expect ≈ heuristic+mission

   **Примечание:** главный proof Phase B — не SDM > heuristic_random, а что SDM **запоминает** transitions и использует их. Реальный proof обучения — Phase A.

### Phase A — без mission text (второй, сложнее — ГЛАВНЫЙ PROOF)

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
EXPLORE_ROOMS → FIND_KEY → GOTO_KEY → PICKUP_KEY → FIND_LOCKED_DOOR → GOTO_DOOR → TOGGLE_DOOR
  ├─ success → GOTO_GOAL → DONE
  └─ fail (wrong color) → DROP_KEY → EXPLORE_ROOMS (loop)
```

8-10 subgoals (с retry loop). BFS handles navigation (HOW), SDM handles color selection (WHAT).

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

**Seed partitioning:** training seeds 0-99, eval seeds 1000-1199. Фазы B и A используют **отдельные** SDM instances (Phase A не наследует память Phase B).

**Ожидаемое время:** ~5-10 мин на эксперимент (1000 steps × 200 seeds = 200K steps, CPU).

| Exp | Phase | Config | Seeds | Gate |
|-----|-------|--------|-------|------|
| 113a | B (mission) | 50 train, 200 eval, max_steps=1000 | train 0-49, eval 1000-1199 | ≥70% |
| 113b | B ablation | heuristic±mission vs trained vs untrained | eval 1000-1199 | report |
| 113c | A (no mission) | 100 train, 200 eval, max_steps=1000 | train 0-99, eval 1000-1199 | ≥50% |
| 113d | A ablation | heuristic vs SDM trained vs untrained | eval 1000-1199 | p<0.05 |
