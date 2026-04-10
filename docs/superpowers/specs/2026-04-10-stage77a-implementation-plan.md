# Stage 77a: Implementation Plan

**Spec:** `2026-04-10-stage77a-conceptstore-forward-sim-design.md`
**Status:** Draft plan, ready for execution
**Created:** 2026-04-10

## Project summary

**Goal:** агент достигает survival ≥200 на Crafter с врагами через forward simulation сквозь ConceptStore causal rules. Заменяет Stage 76 EpisodicSDM substrate как отклонение от идеологии.

**Constraints:**
- Один developer, CPU-only разработка (minipc GPU только для финального eval)
- Ideology compliance: 0 hardcoded категорий, 0 магических порогов в policy code
- Сохранить `tile_segmenter.py` / exp135 checkpoint unchanged
- Каждый commit оставляет main в зелёном состоянии (pytest passes + exp135 regression)

**Definition of done:**
- Все 7 gates из спеки пройдены на minipc
- Тесты ≥90% coverage на новых модулях
- Реверсы ≤3 attempts per gate failure; иначе — откат в brainstorm
- Документация: `docs/reports/stage-77a-report.md` написан
- `docs/ASSUMPTIONS.md` обновлён с section про 77a

---

## Milestones

| # | Milestone | Commits | Success Criteria |
|---|-----------|---------|-------------------|
| M1 | Data types defined | 1 | Новые dataclasses импортируются, конструкторы работают, unit tests pass |
| M2 | Textbook YAML + parser | 2 | `crafter_textbook.yaml` загружается в новом формате, возвращает правильные `CausalLink` с `effect`; backward compat regex fallback работает |
| M3 | HomeostaticTracker refactored | 3 | Tracker имеет innate/observed split, `get_rate(var)` без `visible_concepts`, running mean |
| M4 | ConceptStore methods working | 4 | `simulate_forward`, `plan_toward_rule`, `find_remedies` проходят unit тесты на синтетических state |
| M5 | MPC loop functional | 5 | `run_mpc_episode` проходит integration smoke на реальном Crafter без exceptions |
| M6 | exp137 smoke local pass | 6 | 5 episodes без crash; gate checks запускаются |
| M7 | Stage 76 substrate removed | 7 | `src/snks/memory/` удалён, pytest green, exp137 ещё работает |
| M8 | Legacy dead code removed | 8 | `HOMEOSTATIC_VARS`, `select_goal`, dead perception.py functions удалены, ideology lint passes |
| M9 | Final cleanup | 9 | exp136 удалён, финальный pytest + regression green |
| **M10** | **Gate 1 breakthrough** | — | **Survival ≥200 на minipc eval (3×20 eps)** |

M10 — отдельный milestone вне commit chain, проверяется запуском exp137 на minipc после M8.

---

## Phase 1 (Commit 1): Data types

Prepare pure data classes. No behavior, no dependencies on other new code.

**Target file:** `src/snks/agent/forward_sim_types.py` (новый)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 1.1 | `StatefulCondition` dataclass | `var`, `op`, `threshold`, `satisfied(sim)` method | Unit test: все 5 операторов (`<`, `>`, `==`, `<=`, `>=`) корректно работают | — |
| 1.2 | `RuleEffect` dataclass | 7 kind'ов + inventory_delta/body_delta/scene_remove/world_place/movement_behavior/spatial_range/stateful_condition/body_rate/body_rate_variable fields | Конструктор не падает, equality работает | 1.1 |
| 1.3 | `DynamicEntity` dataclass | `concept_id`, `pos` | Конструктор, equality, copy | — |
| 1.4 | `SimState` dataclass + `copy()`, `is_dead()` | fields: inventory, body, player_pos, dynamic_entities, spatial_map, last_action, step | `copy()` deep-copies all nested state; `is_dead()` reads reference_min correctly | 1.3 |
| 1.5 | `SimEvent` dataclass | `step`, `kind`, `var`, `amount`, `source` | Конструктор, equality | — |
| 1.6 | `Trajectory` dataclass | `plan`, `body_series`, `events`, `final_state`, `terminated`, `terminated_reason`, `plan_progress`, `failure_step(var)` method | Empty trajectory constructs; `failure_step` returns `None` on clean series, int on zero-crossing | 1.4, 1.5 |
| 1.7 | `Failure` dataclass | `kind`, `var`, `cause`, `step`, `severity` | Конструктор + equality | — |
| 1.8 | `PlannedStep` и `Plan` dataclasses | PlannedStep: action, target, near, rule; Plan: steps, origin | Конструкторы, equality | 1.2 |
| 1.9 | Обновить `CausalLink` в `concept_store.py` | Добавить `kind: str`, `effect: RuleEffect \| None`; **сохранить** `result: str` с deprecation warning в docstring | Старый код, использующий `.result`, не падает | 1.2 |
| 1.10 | Unit тесты | `tests/test_stage77_types.py` | 15+ тестов на конструкторы, equality, методы | 1.1-1.9 |

**Gate for Commit 1:**
- `pytest tests/test_stage77_types.py -v` — все green
- `pytest tests/` полностью — не сломалось ничего (CausalLink backward compat)

**Risks Phase 1:** низкие. Pure data types.

---

## Phase 2 (Commit 2): Textbook YAML grammar + parser

Rewrite textbook parser to dict dispatch. Support both old regex (deprecated) and new YAML formats during transition.

**Target files:** `src/snks/agent/crafter_textbook.py`, `configs/crafter_textbook.yaml`, `tests/test_stage71.py` (update)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 2.1 | `_build_effect` helper | Converts effect dict from YAML to `RuleEffect` | Handles `inventory`, `body`, `remove_entity`, `world_place` keys; returns correct `kind` | 1.2 |
| 2.2 | `_parse_action_rule` | Parses `{action: X, target: Y, effect: {...}, requires: {...}}` → `CausalLink` | All 4 action types (`do`, `make`, `place`, `sleep`) parse correctly | 2.1 |
| 2.3 | `_parse_passive_rule` | Parses `{passive: X, ...}` for `body_rate`, `movement`, `spatial`, `stateful` | All 4 passive types parse with correct effect | 2.1 |
| 2.4 | `parse_rule` dispatcher | Routes dict to `_parse_action_rule` or `_parse_passive_rule` based on presence of `action`/`passive` key | Raises `ValueError` on unknown format | 2.2, 2.3 |
| 2.5 | Legacy regex fallback | Сохранить текущий regex parser как `_parse_rule_legacy`; wrapper пытается сначала dict, потом regex | Старые YAML файлы с string rules всё ещё загружаются (с warning) | — |
| 2.6 | `CrafterTextbook.load_into` обновлён | Читает body block structure `{prior_strength, variables, ...}`; передаёт rules в parser | `body.variables` корректно извлекаются; `body.prior_strength` доступен через `tb.body_block` property | 2.4, 2.5 |
| 2.7 | Новый `crafter_textbook.yaml` | Переписан в structured формат согласно спеки (§ 6) | File загружается без warning'ов; все 25+ правил парсятся в корректные `CausalLink` | 2.6 |
| 2.8 | Тесты Stage 71 обновлены | `tests/test_stage71.py` — проверить load нового YAML | Все существующие тесты passes после update | 2.7 |
| 2.9 | Новые тесты парсера | `tests/test_stage77_parser.py` — по одному тесту на каждый из 7 типов правил | 7+ тестов green | 2.4 |

**Gate for Commit 2:**
- `pytest tests/test_stage77_parser.py tests/test_stage71.py -v` green
- Загрузка `configs/crafter_textbook.yaml` возвращает ≥25 `CausalLink` с заполненным `effect`
- exp135 regression passes (segmenter не трогали)

**Risks Phase 2:**
- Старые строковые правила в `crafter_textbook.yaml` не все мапятся 1:1 на структурированные — некоторые (например `"zombie nearby without wood_sword means flee"`) удаляются как dead code
- Mitigation: явно документировать mapping в комментариях YAML для аудита

---

## Phase 3 (Commit 3): `HomeostaticTracker` refactor

Innate/observed split, drop `conditional_rates`, drop `visible_concepts` parameter.

**Target file:** `src/snks/agent/perception.py` (targeted refactor)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 3.1 | Split tracker fields | `innate_rates`, `observed_rates`, `observation_counts`, `prior_strength`, `reference_min`, `reference_max` | Все fields инициализируются через `field(default_factory=...)` | — |
| 3.2 | `init_from_textbook(body_block, rules)` | Загружает `prior_strength`, `reference_min/max` из body_block; `innate_rates` из `passive_body_rate` rules | Unit test: tracker с mock body block получает правильные values | 2.6 |
| 3.3 | `update(inv_before, inv_after, visible_concepts)` refactor | Running mean вместо EMA; `observation_counts` инкрементится; `conditional_rates` updates удалены | Unit test: 1000 updates — running mean сходится к true rate; observed_max растёт монотонно | 3.1 |
| 3.4 | `get_rate(var)` без `visible_concepts` | Bayesian combination `w * innate + (1-w) * observed` где `w = prior_strength / (prior_strength + n)` | Unit test: при n=0 → innate; при n=prior_strength → 50/50; при n→∞ → observed | 3.1 |
| 3.5 | Удалить `conditional_rates`, `RATE_EMA_ALPHA`, `_initialized` flag | Clean deletion | Grep по кодобазе не находит references | 3.3 |
| 3.6 | Обновить callers | `continuous_agent.py`, `experiments/exp136_*.py`, тесты, которые вызывают `tracker.get_rate(var, visible_concepts=...)` | Все callers обновлены, pytest passes | 3.4 |
| 3.7 | Новые тесты | `tests/test_stage77_tracker.py` — innate/observed weighting, `init_from_textbook` | 8+ тестов green | 3.1-3.6 |

**Gate for Commit 3:**
- `pytest tests/test_stage77_tracker.py -v` green
- Полный pytest — ничего не сломано
- exp135 regression passes
- Grep `conditional_rates` в `src/snks/agent/` — 0 вхождений (кроме Stage 76 files, которые удалятся в Commit 7)

**Risks Phase 3:**
- `exp136_continuous_learning.py` напрямую использует старый tracker API — обновление не ломает exp136 (оно продолжает работать с новыми полями)
- Mitigation: exp136 остаётся рабочим до Commit 7, его тестируем smoke run'ом

---

## Phase 4 (Commit 4): `ConceptStore` new methods

Three new methods: `simulate_forward`, `plan_toward_rule`, `find_remedies`. Plus helper `_apply_effect_to_sim`, `_find_rule_producing_item`, etc.

**Target file:** `src/snks/agent/concept_store.py` (add methods)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 4.1 | `ConceptStore.find_remedies(failure)` | Returns list of `CausalLink` whose effect prevents failure | Unit test: `find_remedies(var_depleted=food)` → returns cow/water rules; `find_remedies(attributed_to=zombie)` → returns combat rule | 1.2, 1.7, 2.6 |
| 4.2 | `ConceptStore._find_rule_producing_item(item)` | Internal helper: ищет rule чей `effect.inventory_delta[item] > 0` | Unit test: возвращает `do tree gives wood` для `item="wood"` | 4.1 |
| 4.3 | `ConceptStore._find_rule_producing_adjacent_state(concept)` | Ищет rule который делает `concept` adjacent (через `place` action) | Unit test: `adjacent_to_table` → `place table on empty` | 4.2 |
| 4.4 | `ConceptStore.plan_toward_rule(target_rule, state)` | Backward chain, возвращает `list[PlannedStep]` | Unit test: `plan_toward_rule(combat_rule, empty_state)` → `[do tree×3, place table, make sword, do zombie]`; уже имеющийся sword → `[do zombie]` | 4.2, 4.3 |
| 4.5 | Deprecated stub `ConceptStore.plan(goal_id)` | Старая сигнатура сохранена, внутри ищет rule по `goal_id == result` (legacy) или `effect.kind == remove/inventory_delta` (new), вызывает `plan_toward_rule` | Старые тесты Stage 71-75, использующие `plan(string)`, проходят без изменений | 4.4 |
| 4.6 | `_apply_effect_to_sim(sim, effect, store)` helper | Mutates sim state according to effect: inventory_delta, body_delta, scene_remove, world_place | Unit test: каждый kind корректно обновляет sim | 1.2, 1.4 |
| 4.7 | `ConceptStore.simulate_forward(plan, initial_state, horizon)` outer loop | Основная функция rollout'а согласно спеке § 7. Использует `_apply_tick` (следующая задача) | Unit test: sim rollout на 5 тиков с trivial plan не падает, возвращает `Trajectory` с populated `body_series` | 1.4, 1.6, 4.6 |
| 4.8 | `_apply_tick` Phase 1 — movement | Dynamic entities двигаются по movement rules | Unit test: zombie с `chase_player` behavior движется на 1 тайл ближе | 4.7 |
| 4.9 | `_apply_tick` Phase 2 — player move | Player позиция обновляется если primitive `move_*` | Unit test: `move_right` → `player_pos[0] += 1` | 4.7 |
| 4.10 | `_apply_tick` Phase 3 — body rates | Background rates applied через `_body_rate_rules()` | Unit test: food decay -0.04/tick применяется | 4.7 |
| 4.11 | `_apply_tick` Phase 4 — stateful | Stateful rules applied когда condition satisfied | Unit test: `food > 0 restores health 0.1` применяется когда food=5, не применяется когда food=0 | 4.7 |
| 4.12 | `_apply_tick` Phase 5 — spatial | Adjacency damage rules when manhattan ≤ spatial_range | Unit test: zombie adjacent → health -2; zombie на distance 2 → no effect | 4.7 |
| 4.13 | `_apply_tick` Phase 6 — action effects | `do`/`make`/`place`/`sleep` routes to matching rule, applies effect | Unit test: `do` near tree → inventory[wood]+=1 | 4.7 |
| 4.14 | Confidence threshold в каждой phase | `if rule.confidence < 0.1: continue` в phases 1, 4, 5, 6 | Unit test: rule с confidence=0 не fires, с confidence=1 fires | 4.8-4.13 |
| 4.15 | Новые тесты | `tests/test_stage77_simulate.py`, `tests/test_stage77_concept_store.py` | 20+ тестов green | 4.1-4.14 |

**Gate for Commit 4:**
- `pytest tests/test_stage77_simulate.py tests/test_stage77_concept_store.py -v` green
- `pytest tests/` полный — backward compat сохранён (через `plan` stub)
- Нет performance regression в Stage 76 путях (они ещё работают)

**Risks Phase 4:**
- Самая большая фаза по объёму (15 tasks)
- `_apply_tick` 6 фаз — interaction-heavy, легко забыть edge case
- Mitigation: unit test на каждую фазу в изоляции, затем integration test на 5-tick rollout

---

## Phase 5 (Commit 5): MPC loop supporting functions + `run_mpc_episode`

Create new `mpc_agent.py` file with MPC loop and helpers. **Не трогаем** старый `continuous_agent.py` — параллельное существование.

**Target file:** `src/snks/agent/mpc_agent.py` (новый)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 5.1 | `score_trajectory(traj, tracker)` | Lexicographic tuple: `(alive, min_body, survival_ticks, final_body)` | Unit test: alive > dead ordering; equal-alive → higher min_body wins; equal → longer survival wins | 1.6 |
| 5.2 | `extract_failures(traj)` | Scans body_series for zeros + events for negative deltas от non-background sources | Unit test: trajectory с health→0 at step 9 returns `Failure(var_depleted, var=health, step=9)`; events с `source=zombie` → `Failure(attributed_to, cause=zombie)` | 1.6, 1.7 |
| 5.3 | `expand_to_primitive(step, sim, store)` | Converts `PlannedStep` → env primitive string | Unit test: `do tree` when adjacent → `"do"`; when not adjacent → `"move_*"` toward tree | 4.7 |
| 5.4 | `DynamicEntityTracker` class | Tracks dynamic entities (from vf.detections) across steps | `update(vf, player_pos)` добавляет/удаляет entities; `current()` возвращает list | — |
| 5.5 | `generate_candidate_plans(state, store, tracker)` | Baseline rollout → extract failures → find_remedies → plan_toward_rule для каждого | Unit test: baseline с health depletion возвращает baseline + combat plan | 4.1, 4.4, 4.7, 5.2 |
| 5.6 | `run_mpc_episode(env, ...)` outer loop | Main MPC episode loop согласно спеке § 4 | Integration test: 10-step mock episode не падает, возвращает result dict | 4.7, 5.1, 5.3, 5.5 |
| 5.7 | `_nearest_concept(sim, store)` helper | Returns concept in tile immediately in front of player | Unit test: с tree at (player.x, player.y-1) и `last_action="move_up"` → `"tree"` | 4.7 |
| 5.8 | Performance benchmark | `benchmarks/bench_simulate_forward.py` | Measures p50 и p99 latency of 100 MPC decisions on local CPU | 5.6 |
| 5.9 | Integration smoke test | `tests/test_stage77_mpc.py::test_smoke_integration` — 5 eps на mock Crafter env | No exceptions, SimState progression sensible | 5.6 |

**Gate for Commit 5:**
- `pytest tests/test_stage77_mpc.py -v` green
- `benchmarks/bench_simulate_forward.py` показывает p99 < 200ms (мягкий budget для локального CPU; жёсткий 100ms — на minipc)
- Полный pytest green

**Risks Phase 5:**
- `run_mpc_episode` зависит от всех остальных компонентов — любая ошибка в фазах 1-4 проявится здесь
- Mitigation: integration test с mock env перед реальным Crafter'ом

---

## Phase 6 (Commit 6): `experiments/exp137_mpc_forward_sim.py`

New experiment file using `run_mpc_episode`. Pipeline identical to exp136 (same phases, same gate checks).

**Target file:** `experiments/exp137_mpc_forward_sim.py` (новый)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 6.1 | Phase 0: load segmenter | Тот же код что в exp136 | Запускается, не падает | — |
| 6.2 | Phase 1: warmup-safe (no enemies, 50 eps) | `run_mpc_episode(enemies=False)` × 50 | Завершается, summary показывает средний survival | 5.6 |
| 6.3 | Phase 2: warmup-enemy (enemies on, 50 eps) | С decay temperature (не используется в MPC, но для параллелизма с exp136) | Завершается | 6.2 |
| 6.4 | Phase 3: eval (3 × 20 eps, enemies, max_steps=1000) | Три независимых run с разными seeds | Summary показывает per-run + overall mean survival | 6.2 |
| 6.5 | Phase 4: gate checks | Survival ≥200, wood ≥50% smoke, tile_acc ≥80% | Все 4 gate'а считаются и выводятся | 6.4 |
| 6.6 | Phase 5: summary report | Печатает итоги | Всё визуально читаемо | 6.5 |
| 6.7 | Local smoke test | `python experiments/exp137_mpc_forward_sim.py --smoke` — 5 eps короткие | Не падает, smoke gate'ы считаются | 6.6 |

**Gate for Commit 6:**
- Local smoke запускается без exceptions
- Gate checks код работает (могут не проходить, но должны вычисляться)
- **Не запускается на minipc пока** — это отдельный шаг после Commit 8

**Risks Phase 6:**
- Compute на local CPU — короткие eps, OK для smoke
- Первая реальная интеграция с настоящим Crafter env — могут всплыть edge cases

---

## Phase 7 (Commit 7): Remove Stage 76 substrate

После того как Commit 6 прошёл smoke локально, удаляем старый Stage 76 код.

**Critical checkpoint:** перед началом Phase 7 **запустить exp137 smoke на minipc** чтобы убедиться что новая архитектура хотя бы работает. Если там exception — возврат к фазам 4-5 для диагностики.

**Target:** удаление файлов

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 7.1 | Запустить exp137 на minipc smoke | SSH minipc, 5 eps eval | Не падает, возвращает какой-то survival number (даже если <200) | Commit 6 |
| 7.2 | Delete `src/snks/memory/` package | `rm -r src/snks/memory/` (git rm) | Directory gone, git status clean | 7.1 |
| 7.3 | Delete Stage 76 imports | Remove `from snks.memory import ...` из `continuous_agent.py` и `exp136` | `grep memory src/snks/agent/continuous_agent.py` — 0 matches | 7.2 |
| 7.4 | Delete Stage 76 tests | `tests/test_stage76_foundation.py`, `_sdr.py`, `_sdm.py`, `_agent.py`, `_no_hardcode.py` | Файлы удалены | 7.2 |
| 7.5 | Verify pytest | Полный `pytest tests/` | Green (никакие оставшиеся imports не ломают) | 7.3, 7.4 |
| 7.6 | Verify exp135 regression | `python experiments/exp135_eval_only.py` | Passes | 7.5 |
| 7.7 | Verify exp137 smoke still works | Локальный smoke | Passes | 7.5 |

**Gate for Commit 7:**
- `src/snks/memory/` не существует
- Pytest полный green
- exp135 regression green
- exp137 smoke local green

**Risks Phase 7:**
- Оставшиеся hidden imports к `src/snks/memory/` в тестах других stages
- Mitigation: grep `snks.memory` по всей кодобазе перед удалением

---

## Phase 8 (Commit 8): Remove `HOMEOSTATIC_VARS`, `select_goal`, dead perception code

Критическая фаза для закрытия F3, F4, F11, F12. Gate: ideology lint clean.

**Target file:** `src/snks/agent/perception.py` (major deletion), `src/snks/agent/concept_store.py` (deprecated stub removal), `src/snks/agent/crafter_textbook.py` (regex fallback removal)

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 8.1 | Delete `HOMEOSTATIC_VARS` const | Remove line + all 5 usages | Grep `HOMEOSTATIC_VARS` в `src/snks/` — 0 matches | — |
| 8.2 | Delete `select_goal` function | Remove function + tests that call it | Grep `select_goal` в `src/snks/` — 0 matches | 8.1 |
| 8.3 | Delete `compute_drive`, `compute_curiosity`, `get_drive_strengths` | Remove functions + tests | Grep — 0 matches | 8.2 |
| 8.4 | Delete `_STAT_GAIN_TO_NEAR` dict | Remove | 0 matches | — |
| 8.5 | Delete dead grounding/perception functions | `perceive_field`, `perceive` (legacy), `perceive_tile_field` (в perception.py, не в continuous_agent), `ground_empty_on_start`, `ground_zombie_on_damage`, `on_action_outcome`, `should_retrain`, `retrain_features`, `babble_probability`, `explore_action` | Все удалены, pytest green | — |
| 8.6 | Delete old `continuous_agent.py` | Заменяется `mpc_agent.py` | Файл удалён или пустой stub с `from .mpc_agent import *` для compat | 5.6 |
| 8.7 | Delete local `perceive_tile_field` copy | Был в continuous_agent.py — удаляется вместе с ним | — | 8.6 |
| 8.8 | Remove `CausalLink.result` field | From dataclass definition + all references | `grep '\.result'` в concept_store — 0 matches | 8.1 |
| 8.9 | Remove `ConceptStore.plan(goal_id)` deprecated stub | Only `plan_toward_rule` remains | `grep 'def plan(' src/snks/agent/concept_store.py` — 1 match (plan_toward_rule или ничего) | 4.5 |
| 8.10 | Remove regex fallback в `crafter_textbook.py` | Только dict dispatch остаётся | `_parse_rule_legacy` удалён | 2.5 |
| 8.11 | Write `tests/lint_ideology_77a.py` | Automated scan for forbidden patterns согласно спеке § 13 | Script runs, returns 0 if clean | 8.1-8.10 |
| 8.12 | Run ideology lint | `python tests/lint_ideology_77a.py` | Exit code 0 | 8.11 |
| 8.13 | Update tests Stage 71-74 | Те что использовали `plan(string)`, `HOMEOSTATIC_VARS`, `select_goal`, `conditional_rates` | Обновлены под new API или удалены (если тестировали мёртвую функциональность) | 8.1-8.10 |
| 8.14 | Full regression | `pytest tests/` + `python experiments/exp135_eval_only.py` + local exp137 smoke | Всё green | 8.13 |

**Gate for Commit 8:**
- `tests/lint_ideology_77a.py` exit 0
- Полный pytest green
- exp135 regression green
- exp137 local smoke green
- `git grep` по forbidden patterns ничего не находит

**Risks Phase 8:**
- Самый большой diff (~1000 строк удалено)
- Риск сломать что-то непредвиденное
- Mitigation: делать tasks 8.1-8.10 по одной, проверять pytest после каждой
- Если что-то ломается — откатываем отдельный task, не весь commit

---

## Phase 9 (Commit 9): Remove `exp136`

Последний cleanup step. Старый эксперимент больше не нужен.

| # | Task | Deliverable | Done criteria | Deps |
|---|------|-------------|---------------|------|
| 9.1 | Delete `experiments/exp136_continuous_learning.py` | Файл удалён | — | Commit 8 |
| 9.2 | Final pytest | Полный suite | Green | 9.1 |
| 9.3 | Final regression | exp135 + exp137 smoke | Green | 9.2 |

**Gate for Commit 9:** всё green, никаких follow-up.

---

## Phase 10 (не commit — это gate validation): minipc eval для Gate 1

После того как все 9 commits на main, запускаем exp137 на minipc и проверяем **survival ≥200**.

| # | Task | Deliverable | Done criteria |
|---|------|-------------|---------------|
| 10.1 | Push main на origin | `git push origin main` | Push successful |
| 10.2 | SSH minipc, git pull | `ssh minipc 'cd /opt/agi && git pull'` | Pull successful |
| 10.3 | Запуск exp137 в tmux | `ssh minipc 'tmux new-session -d -s 77a "cd /opt/agi && source .venv/bin/activate && python experiments/exp137_mpc_forward_sim.py 2>&1 \| tee logs/exp137.log"'` | tmux session запущена |
| 10.4 | Мониторинг | Periodically `tmux attach -t 77a` или `tail -f logs/exp137.log` | Процесс не упал |
| 10.5 | Читать финальный отчёт | После завершения (~4-8 часов) — grep gates в логе | Видны Gate 1-4 results |
| 10.6 | Если Gate 1 fail (survival <200) | Follow 3-attempt rule per spec § 15 | Max 3 попыток fix, затем rollback to brainstorm |
| 10.7 | Если Gate 1 pass | Write `docs/reports/stage-77a-report.md` + обновить `docs/ASSUMPTIONS.md` | Report committed, memory обновлено |

**Phase 10 не имеет commit'а** — результаты реального прогона.

---

## Risks & 3-attempt rule (summary)

Согласно `feedback_techdebt_pipeline.md` memory:

**Gate 1 (survival ≥200) fail response:**
1. **Attempt 1:** Анализ exp137 логов на minipc. Найти где forward sim неверно предсказывает. Обычно: (a) missing stateful rule, (b) movement rule слишком оптимистичный, (c) scoring weighting криво.
2. **Attempt 2:** Дополнительные правила в textbook (например забытое «arrow damages health» для skeleton). Fix + re-run.
3. **Attempt 3:** Увеличить horizon (20 → 40), beam search на candidates (top-3 вместо всех), или другая архитектурная корректировка.
4. **После 3 attempts** — **STOP**. Возврат к Phase A brainstorm. Возможно нужен 77b (surprise→rule) раньше, или другой course action.

**Gate 5 (forward sim perf) fail response:**
- Optimize: vectorize phases numpy, beam на candidates, reduce horizon
- Worst case: hot path в Cython

**Gate 4 (ideology lint) fail response:**
- **Не допускается.** Rewrite the offending code. Patterns нарушены осознанно либо случайно.

---

## Commit dependencies graph

```
Commit 1 (types) ─┬─→ Commit 2 (parser)  ─┐
                  ├─→ Commit 3 (tracker)  ─┼─→ Commit 4 (concept store methods) ─→ Commit 5 (MPC loop) ─→ Commit 6 (exp137)
                  │                        │                                                              │
                  │                        │                                                              ├─→ Commit 7 (remove memory/)
                  │                        │                                                              │       │
                  │                        │                                                              │       ├─→ Commit 8 (remove HOMEOSTATIC_VARS, select_goal, dead code)
                  │                        │                                                              │       │       │
                  │                        │                                                              │       │       └─→ Commit 9 (remove exp136)
                  │                        │                                                              │       │               │
                  │                        │                                                              │       │               └─→ Phase 10 (minipc eval)
```

**Critical path:** 1 → 2 → 4 → 5 → 6 → 7 → 8 → 9 → 10.
**Parallelizable:** Commits 3 и 2 не зависят друг от друга (после Commit 1). Можно делать в любом порядке.

---

## Estimate

- **Commit 1 (types):** небольшой, данные
- **Commit 2 (parser):** средний
- **Commit 3 (tracker):** небольшой
- **Commit 4 (ConceptStore methods):** самый большой — 15 tasks
- **Commit 5 (MPC loop):** средний
- **Commit 6 (exp137):** небольшой — pipeline стандартный
- **Commit 7 (remove memory/):** небольшой, в основном удаление
- **Commit 8 (remove dead code):** средний, много осторожности
- **Commit 9 (remove exp136):** trivial

Не указываю часов/дней — согласно `feedback_techdebt_pipeline.md`, время оценок не делаем.

---

## Definition of done (complete list)

- [ ] Все 9 commits на main, pytest green в каждом
- [ ] Phase 10 eval на minipc выполнен, результаты залогированы в `docs/reports/stage-77a-report.md`
- [ ] Gates 1-7 статус зафиксирован (pass или fail с диагнозом)
- [ ] `docs/ASSUMPTIONS.md` обновлён Stage 77a section
- [ ] `MEMORY.md` обновлён — `project_stage76_plan.md` → mark 77a as COMPLETE
- [ ] `ROADMAP.md` обновлён при достижении Gate 1
- [ ] Если Gate 1 не взят — decision point документирован: продолжаем attempts или откат в brainstorm

---

## Next step после этого plan'а

Execution через `executing-plans` skill (если есть) или manual execution commit-by-commit.

Каждый commit — отдельная atomic unit of work. После каждого commit — pytest + regression checkpoint.
