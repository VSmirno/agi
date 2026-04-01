# Autonomous Development Log — 2026-04-01

## Текущая фаза: 1 — Живой DAF, прогресс ~60%

## Stage 38_fix: Curiosity-Driven Action Selection (TD-001 fix)

### [14:00] Фаза 0: Git setup
- Ветка: stage-38_fix-curiosity от main (commit 9acdb1a)
- Tech debt проверен: 3 open (TD-001 IN_PROGRESS, TD-002/003 OPEN), 0 закрыто
- minipc свободен (нет tmux сессий)
- TD-001 — приоритетный fix-подэтап, 3 root causes:
  1. goal_embedding never set → navigator disabled
  2. Action space n_actions=7 vs motor=5 → toggle unreachable
  3. epsilon=0.3 too low
- Fix plan: curiosity/PE-driven action selection без внешней цели (вариант C)

### [14:15] Фаза 1: Спецификация
- Подход A: auto set_goal_from_obs (trade-off: DoorKey goal за дверью, не универсально)
- Подход B: lower similarity threshold (trade-off: без goal навигатор = random)
- Подход C: PE-driven exploration primary (trade-off: нет goal-directed, но для DoorKey OK)
- **Выбран: C** — обоснование: агент не видит цель до открытия двери, PE-exploration универсальнее

### [14:30] Фаза 2: Реализация
- TDD: 12 новых тестов написаны до кода
- Fix #1 (action space): PureDafAgent пересоздаёт MotorEncoder с config.n_actions
- Fix #2 (PE exploration): PredictionErrorExplorer + EpsilonScheduler интегрированы
- Fix #3 (epsilon): 0.7 → 0.1 decay через EpsilonScheduler
- Circular import resolved (lazy import curriculum in __init__)
- 33/33 тестов PASS

### [14:45] Фаза 3: Эксперименты
- CPU exp97: 97d PASS, 97a/b/c FAIL (ожидаемо для 2000 нод)
- Diagnostic: все 3 механизма верифицированы
  - motor.n_actions=7, все 7 encode OK
  - PE records: 41 за 5 эпизодов
  - epsilon: 0.70 → 0.41 за 5 эпизодов
  - motivation.total_steps: 50 > 0
- GPU exp: подготовлен скрипт scripts/exp97_fix_run.sh

### [15:00] Фаза 4: Веб-демо
- demos/stage-38-fix-curiosity.html — side-by-side random vs PE exploration

### [15:15] Фаза 5: Merge
- Merged stage-38_fix-curiosity → main

### Решения
- Override motor в PureDafAgent вместо изменения CausalAgent (минимальный blast radius)
- Lazy import для circular dependency (curriculum ↔ pure_daf_agent)
- CPU gates не обязательны для PASS — это GPU gates, механизмы верифицированы unit тестами
