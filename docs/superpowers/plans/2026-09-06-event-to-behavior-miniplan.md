# Event-to-behavior mini implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development. По запросу пользователя — минимальный диагностический цикл, без серии ревью и новых framework.

**Goal:** Отделить недостаток покрытия от зависимости event-head от контекста и определить исполнимый следующий поведенческий опыт.

**Architecture:** Один checkpoint-only diagnostic поверх exp169 и существующего replay. Не менять веса, порог, planner или representation. Старые source/unseen layouts считать development.

**Tech Stack:** Python, PyTorch, существующие experiment helpers; runtime HyperPC.

**Spec:** `docs/superpowers/specs/2026-09-05-transferable-learning-core-design.md`, разделы 7–9; ревизия exp164–170 в текущем диалоге.

## Global Constraints

- Разработка локально в текущей feature-ветке; тесты и эксперименты только HyperPC.
- Доставка только Git; не SCP/rsync. Не трогать пользовательские untracked.
- Каждый запуск: persistent run.log, progress.jsonl, results.json, manifest с command/commit/status/exit.
- Никакого обучения, новых архитектур, изменения исторических результатов или threshold tuning.
- Один focused test для диагностического расчёта; не повторять успешную проверку без причины.

### Task 1: exp171 — покрытие и matched-context diagnostic

**Files:** создать `experiments/exp171_event_context_audit.py`, `tests/test_exp171_event_context_audit.py`.

- [x] Загрузить frozen exp153 backbone и exp169 event checkpoint; воспроизвести 120 canonical native event probabilities, сравнить с exp169 (abs tolerance 1e-6). Фактический max diff **0**.
- [x] Восстановить исходный replay и его прежний 75/25 episode split. Получить BEFORE pose/action/changed labels; только train используется для coverage. Считать ключ pose как целые относительные смещения и orientation, без float-rounding collisions.
- [x] Для каждой canonical строки вывести train count с тем же pose/action и отдельно same/opposite event labels. Это coverage coarse state, не доказательство равенства полного состояния: стены и наблюдения в ключ не входят.
- [x] Сопоставить source/unseen rows по pose/action/step, сохраняя все пары и явные unmatched. Удерживая recipient pose/action, заменить только hidden, только z, затем z+hidden на matched donor; вывести probability и flips для каждого варианта. Swap доказывает чувствительность в этой интервенции, но сам по себе не доказывает вредный shortcut (гибридные inputs могут быть вне распределения).
- [x] Один focused test проверяет coverage counts, opposite labels и отсутствие ложного совпадения orientation/action. HyperPC: **1 passed, 0.97 s**, полный audit **137.682 s**, exit **0**, exact protocol и unchanged checkpoint. Коммиты `50134c2`/`bb05139`, root просмотрел код и реальные artifacts без дополнительного review round.

### Task 2: возможность поведенческого сравнения (параллельно, read-only)

- [x] Проверить существующие actual-transition planner controls и frozen model rollout. `exp146._late_fork_audit` уже перебирает 125 одинаковых последовательностей с actual/predicted endpoint costs. Его можно расширить до receding-horizon development comparison, сохраняя общий action budget и scorer; текущий audit сам по себе не full closed-loop control.
- [x] Exp169 не может автономно обновлять pose: `_current_pose` вне `LatentState`, `step` его не меняет, `core_planner._stack/_slice` не переносит. Прямое подключение также сталкивается с batch mismatch. Для learned arm нужен observation-only кандидат либо отдельно обоснованный предиктор состояния. Истинный future pose не подставляется; новый subsystem в этом плане не строится.

### Task 3: решение по данным

- [x] Записать exp171 результаты и ограничения в learning-core report, ASSUMPTIONS и кратко ROADMAP; обновить этот checklist.
- [x] Развилка по данным: native trace совпадает; все шесть critical errors покрыты, но лишь 2–16 примерами; swaps как исправляют, так и создают ошибки. Причина не сведена к нулевому coverage или «плохой памяти». Следующий опыт — matched observation-only direct vector/event кандидат в существующем seam, без pose sidecar и без нового subsystem. Сохранить исходный replay, train-only action/change weighting, бюджет и backbone; не смешивать удаление pose с новой collecting policy или sweep гиперпараметров. Ключевой вопрос: можно ли получить полезный автономный rollout, а не очередной privileged PASS.
- [x] Поведенческий этап выбран: H1/H3 и реальный task success при фиксированном planner/score, с original predictor и actual-transition control. Новый sealed layout/ruleset test остаётся закрытым до выбора кандидата. Следующий опыт ещё не выполнен; exp169 напрямую непригоден из-за отсутствия обновления pose. При отсутствии H3/behavior gain завершить серию heads и отдельно рассмотреть representation/learning design.

## Execution notes

План намеренно ограничен расследованием и решением: новый pose predictor или новая representation требуют отдельного обоснования. По пользовательскому запросу работа продолжается в существующей feature-ветке, без дополнительного worktree, ledger framework и повторных review rounds. Состояние исполнения хранится в этом checklist и Git.

Миниплан выполнен: code + focused test + no-training HyperPC audit + проверка
поведенческого seam + решение. Новых обученных моделей и поведенческого выигрыша
в рамках этого плана нет. Артефакты: `/opt/cuda/agi-core-git-eccfb0e/output_to_user/core/exp171-event-context-audit-001/`.
