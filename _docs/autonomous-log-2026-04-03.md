# Autonomous Development Log — 2026-04-03

## Текущая фаза: M4 — Масштаб, прогресс ~30%

Stages 54-55 COMPLETE (partial obs + exploration). Следующий: Stage 56 (Complex Environment).
M4 gate: новый env 5+ типов объектов, partial observability, subgoal chains 5+.

## Stage 56: Complex Environment

### [00:00] Фаза 0: Git setup + Tech debt
- Ветка: stage56-complex-environment от main (commit e10fb4b)
- minipc: **НЕДОСТУПЕН** (ssh timeout) — tech debt не проверен
- Tech debt status: TD-001 IN_PROGRESS, TD-002/003/004/006 OPEN — все skip
- Записано: minipc недоступен, GPU эксперименты отложены

### [00:05] Фаза 1: Спецификация
- **Цель:** ≥50% BabyAI PutNext, 5+ object types
- **Среда:** BabyAI-PutNextS6N3-v0 (11x6 grid, 6 объектов, 3 пары, без комнат)
- Миссия: "put the [color] [type] next to the [color] [type]"
- Типы объектов: ball(6), box(7), key(5) × 6 цветов = 18 уникальных объектов
- 7x7 partial obs, Discrete(7) actions

**Подходы:**
- **A: Symbolic BFS + Mission Parsing** — расширить PartialObsAgent для multi-object tracking + pickup/drop. Парсить миссию regex. Trade-off: простой, детерминированный, не learning-based.
- **B: Language-guided (Stage 51 pipeline)** — использовать VSA language encoder. Trade-off: VSA pipeline заточен под DoorKey instructions, нужна значительная адаптация.
- **C: Full VSA+SDM** — интеграция мировой модели. Trade-off: слишком сложно для одного этапа, SDM scaling = Stage 58.

**Выбран: A** — прямое расширение proven infrastructure (Stages 54-55). Символический подход уже даёт 100% на DoorKey и MultiRoom. PutNext — следующий уровень сложности (multi-object, pickup/drop), но тот же BFS planning core. Learning-based подход отложен до Stage 59.

### [00:20] Фаза 2: Реализация
- MissionParser: regex для "put the X Y next to the X Y" → (type_id, color_id) пары
- SpatialMap extended: find_object_by_type_color(), find_all_objects()
- PutNextAgent: 5-фазный state machine (EXPLORE→GOTO_SOURCE→PICKUP→GOTO_TARGET→DROP)
- PutNextEnv: wrapper для BabyAI PutNext с agent position + carrying state
- **Bug fix 1:** MiniGrid показывает carried object type на позиции агента → spatial map записывал "мяч" на клетке агента. Fix: очищаем клетку агента при carrying.
- **Bug fix 2:** Drop plan слишком ограничительный (требовал стоять рядом с целью И иметь смежную пустую клетку). Fix: 3-step planning (stand_cell, drop_cell, face_dir).
- 19 тестов PASS

### [00:35] Фаза 3: Эксперименты
- Exp 110a: PutNextLocalS5N3 — 200/200 = **100%**, mean 7.8 steps (gate ≥50%) **PASS**
- Exp 110b: PutNextS6N3 — 199/200 = **99.5%**, mean 14.8 steps (gate ≥50%) **PASS**
- Exp 110c: 18 уникальных типов объектов (gate ≥5) **PASS**
- Все CPU, ~4 секунды total

### [00:40] Фаза 4: Веб-демо
- demos/stage-56-putnext.html — Canvas replay 5 эпизодов, 5-фазная визуализация

### [00:45] Фаза 5: Merge
- Отчёт: docs/reports/stage-56-report.md
- ROADMAP обновлён: Stage 56 COMPLETE

### Решения
- Символический BFS подход вместо learning-based — оправдано 99.5% результатом
- Cleared agent cell from spatial map — критичный fix для MiniGrid partial obs with carrying
- Drop planning через 3-step (stand, drop, face) вместо наивного "стой рядом и бросай"
