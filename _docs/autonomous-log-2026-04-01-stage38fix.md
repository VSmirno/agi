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
