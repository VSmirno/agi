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

- [ ] Загрузить frozen exp153 backbone и exp169 event checkpoint; воспроизвести 120 canonical native event probabilities, сравнить с exp169 (abs tolerance 1e-6).
- [ ] Восстановить исходный replay и его прежний 75/25 episode split. Получить BEFORE pose/action/changed labels; только train используется для coverage. Считать ключ pose как целые относительные смещения и orientation, без float-rounding collisions.
- [ ] Для каждой canonical строки вывести train count с тем же pose/action и отдельно same/opposite event labels. Это coverage coarse state, не доказательство равенства полного состояния: стены и наблюдения в ключ не входят.
- [ ] Сопоставить source/unseen rows по pose/action/step, сохраняя все пары и явные unmatched. Удерживая recipient pose/action, заменить только hidden, только z, затем z+hidden на matched donor; вывести probability и flips для каждого варианта. Swap доказывает чувствительность в этой интервенции, но сам по себе не доказывает вредный shortcut (гибридные inputs могут быть вне распределения).
- [ ] Один focused test проверяет coverage counts, opposite labels и отсутствие ложного совпадения orientation/action. Прогнать на HyperPC, затем полный audit без обучения; проверить native reproduction и unchanged checkpoint.

### Task 2: возможность поведенческого сравнения (параллельно, read-only)

- [x] Проверить существующие actual-transition planner controls и frozen model rollout. `exp146._late_fork_audit` уже перебирает 125 одинаковых последовательностей с actual/predicted endpoint costs. Его можно расширить до receding-horizon development comparison, сохраняя общий action budget и scorer; текущий audit сам по себе не full closed-loop control.
- [x] Exp169 не может автономно обновлять pose: `_current_pose` вне `LatentState`, `step` его не меняет, `core_planner._stack/_slice` не переносит. Прямое подключение также сталкивается с batch mismatch. Для learned arm нужен observation-only кандидат либо отдельно обоснованный предиктор состояния. Истинный future pose не подставляется; новый subsystem в этом плане не строится.

### Task 3: решение по данным

- [ ] Записать exp171 результаты и ограничения в learning-core report, ASSUMPTIONS и кратко ROADMAP; обновить этот checklist.
- [ ] Выбрать один следующий опыт по причине: coverage gap → один targeted real-experience collection comparison; context sensitivity при покрытых состояниях → matched-input representation diagnostic; несогласованность native trace → исправление evaluator до обучения.
- [ ] Поведенческий этап должен сравнивать реальный success и многошаговые прогнозы с фиксированным planner. Новый sealed layout/ruleset test открывать после выбора исполнимого observation-only кандидата. Отсутствие такого кандидата — явный результат диагностики, не повод объявить transfer PASS.

## Execution notes

План намеренно ограничен расследованием и решением: новый pose predictor или новая representation требуют отдельного обоснования. По пользовательскому запросу работа продолжается в существующей feature-ветке, без дополнительного worktree, ledger framework и повторных review rounds. Состояние исполнения хранится в этом checklist и Git.
