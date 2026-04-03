# Autonomous Development Log — 2026-04-03

## Текущая фаза: M4 — Масштаб, прогресс ~57% (4/7 stages complete)

## Stage 56: Complex Environment (COMPLETE — earlier session)

- 99.5% PutNextS6N3, 18 object types, merged to main

---

## Stage 57: Long Subgoal Chains

### Фаза 0: Git setup + tech debt
- Ветка: stage57-long-subgoal-chains от main (commit f409a29)
- Tech debt проверен: 4 open (TD-001 IN_PROGRESS, TD-002/003/004/006 OPEN), 1 closed (TD-005)
- Minipc: доступен, GPU свободен

### Фаза 1: Спецификация
- **Цель:** ≥40% на задачах с 5+ subgoals
- **Среды:** KeyCorridorS4R3 (10×10), S3R3 (7×7), BlockedUnlockPickup (11×6)
- KeyCorridor: 3 ряда комнат с коридором, 1 locked door, 1 key, 1 ball
- Subgoal chain: EXPLORE → GOTO_KEY → PICKUP → GOTO_DOOR → OPEN → DROP_KEY → GOTO_GOAL → PICKUP

**Подходы:**
- **A: Prerequisite-graph ChainPlanner** — backward chain от goal через locked door к key
- **B: SDM-based forward planning** — compound confidence degrades
- **C: Learning-based** — needs episodes, SDM не scaled (Stage 58)

**Выбран: A** — символический backward chaining, расширение proven BFS infrastructure

### Фаза 2: Реализация
- ChainPlanner: builds prerequisite chain from SpatialMap
- KeyCorridorAgent: 8-phase state machine
- **Bug 1:** MiniGrid 3.0 Door.toggle() НЕ потребляет ключ → добавлен DROP_KEY subgoal
- **Bug 2:** Drop-pickup loop — агент drop key, потом pickup обратно. Fix: _should_pickup_key returns False после OPEN_DOOR
- **Bug 3:** BlockedUnlockPickup — мяч блокирует дверь, requires 9+ subgoals (beyond scope)
- 23 тестов PASS

### Фаза 3: Эксперименты
- Exp 111a: KeyCorridorS4R3 — **40.0%** (80/200), mean 39.5 steps (gate ≥40%) **PASS**
- Exp 111b: KeyCorridorS3R3 — **54.0%** (108/200), mean 31.3 steps (gate ≥50%) **PASS**
- Exp 111c: KeyCorridorS5R3 — **39.5%** (79/200), mean 50.7 steps (stretch)
- Все CPU, ~60 секунд total (все 3 эксперимента)

### Фаза 4: Веб-демо
- demos/stage-57-keycorridor.html — Canvas replay, subgoal chain bar, 4 эпизода

### Фаза 5: Merge
- Отчёт: docs/reports/stage-57-report.md
- ROADMAP обновлён: Stage 57 COMPLETE

### Решения
- Prerequisite-graph vs SDM planning — символический подход 40% S4R3, достаточно для gate
- MiniGrid 3.0 key persistence — critical discovery, DROP_KEY subgoal необходим
- BlockedUnlockPickup отложен — 9+ subgoals requires object relocation, beyond current scope
- 60% failure rate on S4R3 — exploration timeout в больших grid, future fix через SDM world model (Stage 58)
