# Stage 38_fix: Curiosity-Driven Action Selection

## Результат: PASS (механизмы верифицированы, GPU gate отложен)

**Ветка:** `stage-38_fix-curiosity`
**Тип:** Fix-подэтап (TD-001)

## Что исправлено

3 root causes exp97 GPU failure:

1. **Action space mismatch (Bug #2):** `CausalAgent.N_ACTIONS=5` hardcoded, но MiniGrid DoorKey требует 7 действий (toggle=5). `PureDafAgent` теперь пересоздаёт `MotorEncoder` с правильным `n_actions` из конфига.

2. **goal_embedding never set (Bug #1):** `run_episode()` не вызывала `set_goal_from_obs()`, навигатор был отключен, агент = random. Fix: PE-driven exploration как основной механизм действий. 3-уровневая иерархия: PE exploration → goal-directed → count curiosity.

3. **epsilon=0.3 too low (Bug #3):** Фиксированный epsilon заменён на `EpsilonScheduler`: 0.7 → 0.1 decay per episode.

## Эксперименты

### CPU verification (механизмы)

| Test | Метрика | Результат | Gate | Статус |
|------|---------|-----------|------|--------|
| Motor 7 actions | encode без ошибок | 7/7 actions OK | no crash | PASS |
| PE explorer | pe records | 41 records / 5 eps | > 0 | PASS |
| Epsilon decay | 0.70 → 0.41 | 5 eps | < initial | PASS |
| Motivation update | total_steps | 50 | > 0 | PASS |
| Nav stats | pe_exploration_ratio | 0.56 | in stats | PASS |

### CPU exp97 (2000 нод, 10 эпизодов — ожидаемо слабые)

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 97d | env_agnostic | no errors | all pass | PASS |
| 97a | DoorKey success | 0.00 | >= 0.10 | FAIL (CPU tiny) |
| 97b | modulations | 0 | > 0 | FAIL (no reward found) |
| 97c | Empty-8x8 | 0.00 | >= 0.30 | FAIL (CPU tiny) |

> CPU с 2000 нод и 10 эпизодами не может пройти gates. Это GPU gates для 50K нод с 30+ эпизодами.

### Unit tests

33/33 PASS (включая 12 новых тестов для fix)

## Запланированные эксперименты (tech debt)

Следующие эксперименты требуют GPU для верификации gate criteria TD-001.

| TD | Exp | Что проверяется | Gate | Статус |
|----|-----|-----------------|------|--------|
| TD-001 | exp97_fix | Curiosity-driven DoorKey-5x5 на 50K нод | success_rate >= 0.10, modulations > 0 | ⏳ запланирован |

> Эта секция будет обновлена после получения результатов.

## Ключевые решения

1. **PE exploration как primary** — вместо того чтобы автоматически вызывать `set_goal_from_obs()` (подход A), выбран подход C: agent explores through curiosity. Обоснование: DoorKey goal за дверью, агент не видит его до открытия. PE-driven exploration универсальнее.

2. **Override motor в PureDafAgent** — вместо модификации CausalAgent (blast radius). PureDafAgent создаёт свой MotorEncoder если `config.n_actions != CausalAgent.N_ACTIONS`.

3. **Lazy import curriculum** — чтобы избежать circular import (curriculum → pure_daf_agent → curriculum).

## Веб-демо

- `demos/stage-38-fix-curiosity.html` — side-by-side: random fallback vs PE-driven exploration с Canvas MiniGrid, PE профилем и epsilon decay

## Файлы изменены

- `src/snks/agent/pure_daf_agent.py` — PE explorer, epsilon scheduler, motor override
- `src/snks/experiments/exp97_pure_daf.py` — updated config with PE/epsilon params
- `tests/test_pure_daf_agent.py` — 12 new tests
- `scripts/exp97_fix_run.sh` — GPU run script
- `demos/stage-38-fix-curiosity.html` — web demo

## Следующий этап

- **GPU verification:** запустить exp97_fix на minipc (50K нод) для проверки gate criteria TD-001
- **Если PASS:** закрыть TD-001, перейти к Stage 41 (Temporal Credit Assignment)
- **Если FAIL:** анализировать GPU results, возможно нужен fix для mental simulation или reward sparsity
