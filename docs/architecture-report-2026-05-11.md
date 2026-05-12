# SNKS Architecture Report — состояние на 2026-05-11

Дата: 2026-05-11
Branch: `stage90r-world-model-guardrails` @ commit `998f250`
Автор: пересборка картины после lava-fix сессии

Это **обзорная карта** текущей кодовой архитектуры. Не how-to, не roadmap. Цель — за один заход понять где что живёт и насколько это соответствует тому что обещает идеология (`docs/IDEOLOGY.md` v2).

---

## 0. TL;DR в одном экране

Система = **четыре слоя знания** (по идеологии) + один пайплайн исполнения. Все четыре слоя сегодня реально существуют в коде, но **не равномерно зрелые**.

```
┌────────────────────────────────────────────────────────────────┐
│  FACTS (категория 1)        configs/crafter_textbook.yaml      │  ← человек пишет
│  «do tree → wood +1»                                            │
│  «lava range=0 → health -1.0»                                   │
├────────────────────────────────────────────────────────────────┤
│  MECHANISMS (категория 2)   src/snks/agent/                    │  ← алгоритмы
│  perceive → spatial_map → planner → sim → score → act          │
├────────────────────────────────────────────────────────────────┤
│  EXPERIENCE (категория 3)   runtime (per-episode)              │  ← агент видит
│  SDM memory, spatial_map, death_log, surprise_accumulator      │
├────────────────────────────────────────────────────────────────┤
│  STIMULI (категория 4)      src/snks/agent/stimuli.py          │  ← «зачем»
│  Survival barrier, Homeostasis, Curiosity                      │
└────────────────────────────────────────────────────────────────┘

Сверху вниз: знание стабилизируется и становится «cheaper» (статичнее).
Снизу вверх: experience очищается до facts через promotion (не реализовано).
```

Главный pipeline исполнения каждого шага:

```
env.step → info["semantic"]
   │
   ▼
PERCEPTION  (perceive_semantic_field, _update_spatial_map, _update_spatial_map_hazards)
   │
   ▼
WORLD MODEL  (VectorWorldModel = SDM; vector_sim.simulate_forward для роллаутов)
   │
   ▼
GOAL SELECTOR  (GoalSelector — символический по textbook)
   │
   ▼
PLAN GENERATION  (generate_candidate_plans: motion + chain + single:target:do)
   │
   ▼
SIMULATE + SCORE  (simulate_forward(plan) → score_trajectory(stimuli, goal))
   │
   ▼
RANK + RESCUE  (EmergencySafetyController, learner-actor advisory)
   │
   ▼
ACT  (env.step(primitive))
   │
   ▼
LEARN  (model.learn(target,action,observed_delta) + spatial_map.update)
```

Это **символический MPC** с **HDC/SDM памятью** для prediction. Не end-to-end neural agent. Learned часть — один локальный actor `.pt`, влияющий на ranking advisory'ем. Остальное — declarative + algorithmic.

---

## 1. Слой Facts (категория 1) — текстбук учителя

### Где живёт

- `configs/crafter_textbook.yaml` (≈315 строк) — основной declarative store
- `configs/promoted_hypotheses.yaml` — пустой пока, должен заполняться автоматически

### Что декларирует

YAML содержит **четыре класса** фактов:

1. **Vocabulary** — какие концепты вообще существуют в этом env и их атрибуты:
   ```yaml
   - { id: tree,  category: resource, blocking: true }
   - { id: lava,  category: hazard,   blocking: false }
   - { id: zombie, category: enemy,   dangerous: true, blocking: true }
   ```
2. **Primitives** — таблица движений с (dx, dy). Раньше эти оффсеты были захардкожены в Python во множестве мест. Stage 82 их вытащил наружу.
   ```yaml
   move_left:  { dx: -1, dy: 0 }
   move_up:    { dx: 0, dy: -1 }
   ```
3. **Env semantics** — как именно среда интерпретирует действия:
   ```yaml
   env_semantics:
     do: { dispatch: facing_tile, range: 1, interaction_mode: direct }
     move: { blocked_by: impassable, updates_facing: true }
   ```
4. **Rules** — действие-эффект и пассивные правила:
   ```yaml
   - { action: do, target: tree,  effect: { inventory: { wood: +1 } } }
   - { passive: spatial, entity: zombie, range: 1, effect: { body: { health: -0.5 } } }
   - { passive: body_rate, variable: food, rate: -0.02 }
   ```

### Кто читает

- `crafter_textbook.py` (loader) → отдаёт `CrafterTextbook` объект с уже распарсенными rules
- `vector_bootstrap.load_from_textbook` — **пишет** facts в SDM (CausalSDM) через `model.learn(target, action, effect)` ×5 раз для confidence. Загружает `proximity_ranges`, `movement_behaviors`, `action_requirements`.
- `goal_selector.GoalSelector` — derives priority-ordered threats из passive spatial rules
- `stage90r_emergency_controller.EmergencyWorldFacts` — derives hostile/resource concept lists и activation thresholds

### Идеологическое соответствие

✅ **Хорошо.** Это самый идеологически чистый слой. После Stage 82 migration движения вынесены в YAML, env semantics declarative, blocking-флаг на vocabulary.

⚠️ **Hard-coded остатки в коде:**
- `_HAZARD_CONCEPTS = {"lava"}` в `vector_mpc_agent.py` — список который должен приходить из vocabulary через флаг (`category: hazard`).
- Перечень «blocking concepts» в `vector_sim._move_target_blocked` — `{water, tree, stone, coal, iron, diamond, table, furnace, cow, zombie, skeleton}` — должно браться из `blocking: true` flag, а не хардкод.
- Crafter `_NATURAL_CONCEPTS` set в `_update_spatial_map`.

Это **antipattern 2** («Зашивать env-семантику в mechanisms») из идеологии — мы признаём что должно быть в textbook, но пока не вытащили.

---

## 2. Слой Mechanisms (категория 2) — алгоритмы

### Перцепция

| Файл | Что делает |
|---|---|
| `crafter_pixel_env.py` | Тонкая обёртка над `crafter.Env`. Возвращает `(pixels, info)`. **NB**: контролирует patch `_balance_chunk` для detrminism (set→sorted iteration). |
| `crafter_textbook.py` | YAML → in-memory CrafterTextbook |
| `perception.py` | `perceive_semantic_field(info) → VisualField` (symbolic mode); `perceive_tile_field(pixels, segmenter) → VisualField` (pixel mode) |
| `tile_head_trainer.py:viewport_tile_label` | Низкоуровневая функция: world coord → semantic class index, фильтрует `_TERRAIN` (lava сейчас в этом списке) |
| `crafter_spatial_map.py` | `CrafterSpatialMap`: cognitive map позиция → концепт. Записывается перцепцией, читается планером. |

**Два режима перцепции:**
- **`symbolic`** (мы сейчас используем) — читает `info["semantic"]` напрямую, бесшумная ground truth.
- **`pixel`** — пиксельный CNN segmenter (`tile_segmenter`, `decode_head`). Обученная сетка. Stage 75 классифицирует тайлы из RGB-пикселей.

### Память (mechanism-side)

| Файл | Что делает |
|---|---|
| `vector_world_model.py` | **`VectorWorldModel` + `CausalSDM`** — sparse distributed memory. Конкретные ассоциации `(target_concept, action) → effect_vec` хранятся через write/read по hyperdimensional addresses. **Это главная learned-вне-нейросеток компонента**, но обучение здесь — символическое write (не градиенты). |
| `vector_bootstrap.py` | Загружает textbook YAML в SDM. После 5 write-ов на правило confidence ≈ 1.0. |
| `vector_sim.py` | `VectorState`, `VectorTrajectory`, `simulate_forward` — символический rollout-симулятор. Использует SDM `model.predict(target, action)` для предсказания эффектов. |

**Важная архитектурная деталь:** SDM = "associative cache" над фактами textbook'а + любыми online-learned rules. Через `model.predict(c, a)` агент получает (effect_vec, confidence). Если confidence < 0.2 → эффект не применяется (skip). SDM-crosstalk под smoke-lite (dim=2048) приводил к false-positives, под full profile (16384/50000) шумовой floor ~0.11 — ниже threshold.

### Планирование

| Файл | Что делает |
|---|---|
| `vector_mpc_agent.py` | **Главная орчестрация эпизода** (`run_vector_mpc_episode`). Цикл per-step: perceive → spatial_map update → goal_select → generate plans → simulate × score → emergency override → execute. ~2000 строк. |
| `vector_sim.py` | `simulate_forward(plan, state, horizon)` — производит trajectory. Поддерживает explicit plan steps + passive rollout с advancement of dynamic entities. |
| `goal_selector.py` | `GoalSelector.select(state) → Goal`. Pure function. Reads textbook threats (passive spatial с health<0), vital thresholds, proactive crafting chain. |
| `stage90r_emergency_controller.py` | EmergencySafetyController. Independent threat assessment + action ranking. Может override planner. |
| `subgoal_planning.py`, `tiered_planner.py`, `chain_generator.py` | Plan composition helpers. Multi-step chains (e.g. `chain:water:do+tree:do`). |

**Что планер реально предлагает как candidate plans:**
- `baseline` (empty plan, 0 steps)
- `self:move_left` / `self:move_right` / `self:move_up` / `self:move_down` / `self:sleep`
- `self:motion_chain:X+Y` (2-3-step motion combinations)
- `single:tree:do`, `single:water:do`, `chain:tree:do+water:do`, etc.

**Чего нет в кандидатах:** `place_*`, `make_*`. Линия 1769:
```python
allowed_actions = ["move_left", "move_right", "move_up", "move_down", "do", "sleep"]
```
Crafter поддерживает 17 действий (`noop`, 4×`move_*`, `do`, `sleep`, 4×`place_*`, 6×`make_*`). Планер видит **только 6**.

**Scoring:**
```
score_trajectory(traj, stimuli, goal) → (base, goal_prog, -steps)
final_score = (base, goal_prog, known, -steps)  # lexicographic
```
- `base` — `stimuli.evaluate(traj)`. Сейчас: 0 если terminated, иначе 1. Под full SDM работает корректно.
- `goal_prog` — `goal.progress(traj)`. Для `gather_wood` это `inventory_delta("wood")`.
- `known` — 1 если target есть в spatial_map.
- `-steps` — длина plan'а с отрицательным знаком (короче=лучше при равном остальном).

### Безопасность / rescue

Два независимых controller'а:
- **`EmergencySafetyController`** — если активирован (vital pressure / hostile pressure / no-progress streak / planner-learner disagreement превышают threshold), пересчитывает действие на основе candidate outcomes и может override planner.
- **Planner-rescue** — внутри `run_vector_mpc_episode`, если planner predicted_loss > N, переключается на rescue advisory. Использует actor advisory (см. следующий слой).

### Идеологическое соответствие mechanisms

✅ **Хорошо для основной механики.** `simulate_forward` env-agnostic — он не знает про tree/water, читает их через model.predict.

✅ **Stage 82 cleanup много починил.** Движения, env_semantics, vocabulary вытащены в textbook.

⚠️ **Остаточный antipattern 2:**
- В `vector_sim._move_target_blocked` — hardcoded set blocking-концептов (должны браться из vocabulary `blocking: true`).
- В `vector_mpc_agent` — hardcoded `allowed_actions` (должно быть конфигом или derived из textbook primitives).
- В `_update_spatial_map._NATURAL_CONCEPTS` — hardcoded set «которые перцепция пишет в карту».

⚠️ **Antipattern 5: scoring fragility.** Текущая lex-tuple scoring уязвима к baseline-wins-ties: пустой plan (`baseline`) выигрывает потому что `-steps=0`, и RNG-fallback дальше может выбрать что угодно. Мы залатали через filter alive_concrete, но фундаментально это симптом — scoring должен инкорпорировать exploration/goal-pressure напрямую, а не через лотерею fallback'а.

---

## 3. Слой Experience (категория 3) — runtime

### Что копится в runtime

| Структура | Что хранит | Срок жизни |
|---|---|---|
| `CrafterSpatialMap._map` | (y,x) → (concept, conf, count) | per-episode (fresh каждый эпизод) |
| `CrafterSpatialMap._blocked` | set of (y,x) которые real env отверг как move target | per-episode |
| `VectorWorldModel.memory` (SDM) | (concept_a, action) → effect_vec, online learn'ed | **per-episode** (fresh) |
| `HomeostaticTracker.observed_rates` | per-vital decay rates Bayesian-updated | per-episode |
| `DynamicEntityTracker` | tracked positions of moving entities | per-episode |
| `PostMortemAnalyzer` damage_log | DamageEvent per damage tick | per-episode (используется в attribution) |
| `LocalBeliefTracker` | recent damage pressure bucket, history | per-episode |
| `surprise accumulator` (concept_store.SurpriseAccumulator) | candidate rules from observation drift | per-episode |
| `nursery.tick()` | promotion candidates | per-episode |

### Online learning

В `run_vector_mpc_episode` после каждого env.step:
- `model.learn(target, action, observed_delta)` — обновляет SDM **только** для `do/place/make` действий. Не для `move_*`.
- При `health_delta < 0` рядом с dynamic entity → `model.learn(entity_cid, "proximity", {"health": health_delta})` — entity-correlated damage discovery.
- `spatial_map.update(player_pos, near_concept, conf)` каждый шаг.
- Detection: новые правила могут попадать в `nursery` через surprise, hypothesis с support_rate≥0.5 и n≥5 эмитятся в `promoted_hypotheses.yaml` (Stage 88).

### Идеологическое соответствие experience

✅ **Continuous learning principle работает.** Каждый шаг → updates SDM + spatial_map + tracker. Нет offline-train-then-deploy split (для SDM).

⚠️ **Принцип 6 «Система, не агент» — частично.**
- ✅ `PostMortemAnalyzer.attribute(damage_log, episode_steps)` существует и работает.
- ✅ Stage 88 `TextbookPromoter` существует.
- ❌ Между эпизодами **SDM не персистится** — каждый episode стартует с свежим VectorWorldModel и заново загружает textbook. Learned rules умирают с эпизодом.
- ❌ `configs/promoted_hypotheses.yaml` пустой — ни одна гипотеза реально не была promoted в сессии этой ветки.
- ❌ Cross-episode knowledge accumulation существует архитектурно, но не работает практически.

⚠️ **Antipattern 1 (учить тому что учитель знает) исторически:** многие правила которые мы пытались learn (Stage 78-80) могли быть прямо записаны в textbook. Сейчас этот pattern меньше повторяется, но остаточно живёт в том что bootstrap learn'ит 5 раз на одно правило вместо direct fact assertion.

---

## 4. Слой Stimuli (категория 4) — мотивация

### Где живёт

`src/snks/agent/stimuli.py` — `StimuliLayer` со списком стимулов:
- `SurvivalAversion` — барьер при death prediction
- `HomeostasisStimulus` — штраф за дефицит eds vital
- `CuriosityStimulus` — bonus за expected surprise (Stage 85)

`StimuliLayer.evaluate(trajectory)` → `base_score` для `score_trajectory`.

### Идеологическое соответствие

✅ **Stage 84 cleanup честно отделил stimuli от mechanism.** Раньше scoring был лексикографически захардкожен внутри `score_trajectory`. Теперь stimuli — конфигурируемый объект.

⚠️ **Stimulus параметры жёстко заданы.** Согласно принципу 4 идеологии, веса/пороги стимулов должны уточняться через experience (post-mortem learning). Сейчас они константы.

⚠️ **avg_survival (наш eval-критерий) косвенно затекает.** В нескольких местах текстовые правила писались с расчётом «чтобы агент жил дольше», что implicitly делает «выживание» целью агента. Согласно принципу: агент должен избегать death-barrier и удовлетворять homeostasis, а survival — эмерджентный результат.

---

## 5. Learned (нейросетевая) часть

### Что реально является trainable

| Компонент | Тип | Файлы | Состояние |
|---|---|---|---|
| **CNN tile segmenter** | CNN классификатор тайлов в pixel-mode | `tile_segmenter.py`, `decode_head.py` | Pretrained (`demos/checkpoints/exp137/segmenter_9x9.pt`). В symbolic-mode не используется. |
| **Local actor (`.pt`)** | MLP advisor for action ranking | `stage90r_local_model.py` | Frozen pretrained (`stage90r_seed7_actor_selection_probe3.pt`). Используется с `actor_share=0` — только advisory bonus. |
| **Дочерние головы local_model** (rescue head, threat head, etc.) | Sub-heads в той же сетке | `stage90r_local_model.py` | Trained вместе с actor. |

**Всё остальное** — символическое:
- World model: SDM с binary HDC vectors, write/read, не градиенты.
- Planner: search над rules.
- Emergency controller: rule-based.
- Goal selector: derived from textbook.

### Training pipeline

Стадия 90R установила полноценный collect→train→eval цикл:

```
1. experiments/stage90r_collect_local_dataset.py
   ↓ запускает eval-агента, дампит per-step (observation, action, outcome) в JSON
2. experiments/stage90r_train_local_evaluator.py
   ↓ тренирует MLP head на собранных данных (PyTorch, обычный SGD)
   ↓ выход: новый .pt чекпоинт
3. experiments/stage90r_eval_local_policy.py
   ↓ грузит .pt, прогоняет multiseed eval
   ↓ выход: stage90r_..._eval.json со survival, controller_distribution, rescue events
```

В нашей сессии мы **только запускали eval** — не делали collect и не делали train. Текущий `.pt` обучен на pre-Stage91 наблюдениях. То что мы поменяли (Stage91 fixes, lava avoidance) **не отражено** в актере. С `actor_share=0` это не критично, но если переключать на actor-led control — actor нужно переобучать.

### Идеологическое соответствие

✅ **«Vision ≠ Knowledge» работает.** CNN segmenter — это perception hardware, отдельно от world model.

⚠️ **Continuous learning принцип частично нарушен.** Local actor `.pt` тренируется в **batch-mode offline**, без online updates. По идеологии это допустимое исключение если относиться к нему как к «врождённому» компоненту, но фактически мы каждые несколько недель переобучаем — это **скрытая batch-deploy фаза**.

⚠️ **`actor_share=0` означает что learned actor де-факто не управляет.** Все Stage91 поведенческие изменения проходят через символический planner. Это легитимно по идеологии (symbolic top-down planning), но создаёт **двусмысленность**: зачем мы обучаем actor если его выход почти не используется? Открытый вопрос: либо повышать actor_share, либо перестать его обучать пока planner не стабильный.

---

## 6. Не-используемые ветки кода (legacy/experimental)

В `src/snks/` много модулей из ранних стадий, которые сейчас фактически dead-end или не подключены к Stage 90R+ pipeline:

| Модуль | Что было | Сейчас |
|---|---|---|
| `daf/` (Dynamic Attractor Fields) | DAP-based representation, Stages 0-20 | Parked в Stage 78a (методологические gaps). Не используется в Stage 9X. |
| `dcam/` (DCAM world model) | HAC bind/unbind storage, Stages 30-50 | Не подключён. |
| `gws/` (Global Workspace) | GWT-based broadcast | 2 файла, не используется в production loop. |
| `metacog/` (Metacognition) | STDP policies, monitor | Не подключено к Stage 9X. |
| `sks/` (СКС formation) | Hopfield-like attractors | Не используется. |
| `encoder/` | VisualEncoder (oscillator-based) | Не используется в symbolic mode. |
| `agent/mpc_agent.py` (classic) | Pre-vector MPC loop | Replaced by `vector_mpc_agent.py` для Stage 90R+. |
| `agent/cls_world_model.py`, `vsa_world_model.py`, `causal_world_model.py` | Alternative WM подходы | `VectorWorldModel` победил, остальные dormant. |

Это не «грязный код» — это **honest snapshot научного поиска**. Идеология (раздел «Чего этот документ НЕ говорит») явно разрешает иметь parked-стадии. Но при чтении кодовой базы важно знать что **только ~25% файлов агента живые в текущем pipeline**.

**Живой минимум (что реально работает в Stage 91):**
```
crafter_pixel_env.py           (env wrapper)
crafter_textbook.py            (YAML loader)
crafter_spatial_map.py         (cognitive map)
perception.py                  (semantic→VisualField)
vector_world_model.py          (SDM)
vector_bootstrap.py            (textbook→SDM)
vector_sim.py                  (rollout)
vector_mpc_agent.py            (orchestration)
goal_selector.py               (goal derivation)
stage90r_emergency_controller.py
stage90r_local_model.py        (actor .pt advisory)
stage90r_local_policy.py       (local action selection)
stage90r_local_affordances.py  (diagnostic local affordance scene)
stimuli.py
post_mortem.py
nav_policy.py, pathfinding.py  (utility)
forward_sim_types.py
concept_store.py (legacy support, частично)
textbook_promoter.py           (Stage 88, не очень работающий)
death_hypothesis.py            (post-mortem hypothesis tracking)
```

≈20 файлов из 71.

---

## 7. Соответствие идеологии — scorecard

Сводная таблица: что обещали vs что реально работает.

| Идеологический принцип / категория | Статус | Комментарий |
|---|---|---|
| **Категория 1 (Facts)** живёт в textbook | ✅ | `crafter_textbook.yaml` — главный store. После Stage 82 movements/env_semantics там. |
| **Категория 2 (Mechanisms)** env-agnostic | 🟡 | Большинство `simulate_forward`, `score_trajectory` чисты. Остатки hardcoded sets (blocking concepts, hazard concepts, allowed_actions) в коде. |
| **Категория 3 (Experience)** живёт в runtime | ✅ для within-episode | Spatial map, SDM, tracker, surprise accumulator — все обновляются per-step. |
| **Категория 4 (Stimuli)** отдельный модуль | ✅ | Stage 84 cleanup. Survival/Homeostasis/Curiosity отдельно от scoring. |
| **Принцип 1: четыре категории, не одна** | ✅ | Архитектурно разделены. Иногда antipattern проявляется (см. остатки hardcoded fact в mechanisms). |
| **Принцип 2: Top-down** | ✅ | Goal driver через stimuli + world model, не if-else'ы в planner. |
| **Принцип 3: Continuous learning** | 🟡 | SDM continuous. Local actor `.pt` — нет (batch trained, frozen). CNN segmenter — нет. |
| **Принцип 4: Vision ≠ Knowledge** | ✅ | CNN отделён от world model. Semantic именование происходит через agent's experience. |
| **Принцип 5: Knowledge flow (textbook ← experience)** | ❌ | `TextbookPromoter` существует архитектурно (Stage 88), `promoted_hypotheses.yaml` пустой. Cross-episode persistence не работает. |
| **Принцип 6: Система, не агент** | 🟡 | Post-mortem analyzer есть, но между эпизодами SDM не персистится. Каждый эпизод стартует с textbook→SDM. Death context не аккумулируется. |

### Открытые вопросы идеологии (раздел 4 IDEOLOGY.md)

| Q | Стояние Q | Наше движение |
|---|---|---|
| Q1: где live world model должна жить? | open | YAML + SDM в runtime. Пока хватает. На большом env упрёмся. |
| Q2: параметрическое vs символическое | open | Hybrid через SDM (binary HDC). Generalization через vector similarity, не hierarchy. Не идеально для transfer. |
| Q3: автономия planner до override учителя | open | SDM по умолчанию aggressive — learned writes могут перекрыть textbook. Нет conflict resolution. |
| Q4: Crafter как goal vs data source | implicit goal | Мы implicitly шли goal-fashion. Roadmap'ом считаем что надо переключаться на data-source framing. |
| Q5: когда останавливать tactical iteration | open | В этой сессии мы 5+ раз попадали в trap. Workflow rules в memory смягчили. |

---

## 8. Известные архитектурные пробелы (актуальные)

### Высокий приоритет

1. **Планер не умеет крафтить.**
   `vector_mpc_agent.py:1769`: `allowed_actions` хардкожен в `[move_*, do, sleep]`. `place_*` и `make_*` никогда не попадают в candidate plans. Агент с 9 wood физически не может попробовать `make_wood_pickaxe`.

2. **Tree-seeking слабый.**
   `single:tree:do` план есть, но при tree-distance > 1 ничем не выделяется среди motion-chain плановов. Нет multi-step pathfinding-bias'а к visible-but-distant ресурсам.

3. **SDM не персистится между эпизодами.**
   `run_vector_mpc_episode` создаёт fresh `VectorWorldModel` и грузит textbook каждый эпизод. Online-learned правила умирают.

### Средний приоритет

4. **Hard-coded sets в mechanisms.**
   `_HAZARD_CONCEPTS`, blocking-concept list в sim, `allowed_actions`, `_NATURAL_CONCEPTS` — всё должно браться из textbook vocabulary через `category` и `blocking`/`dangerous` флаги.

5. **`actor_share=0` де-факто отключает learned actor.**
   Local `.pt` загружается, но дает только advisory bonus к scoring. Либо повышать share (требует более качественной обучающей трассы), либо признать что actor пока не работает и не тратить compute на retraining.

6. **Anti-bounce / anti-stuck для planner_bootstrap fallback.**
   Когда baseline wins ranking, RNG среди alive_concrete не имеет bias'а против повторения недавнего действия. Агент болтается в углах water-peninsula пока RNG не выкатит escape direction.

### Низкий приоритет

7. **`promoted_hypotheses.yaml` всегда пустой.**
   Stage 88 promotion mechanism технически работает, но threshold (`support_rate >= 0.5 AND n_observed >= 5`) почти никогда не достигается в реальных трассах. Death hypotheses собираются, но не promote'ятся.

8. **CNN segmenter и local actor — batch-trained, не continuous.**
   Эти исключения из принципа 3 формально допустимы (perception как «hardware»), но фактически мы используем их как если бы они были continuous, не помечая retraining как осознанный architecture change.

9. **Goal selector не учится.**
   `GoalSelector._derive_threats` — pure function over static textbook. Если опыт показывает что zombie не такой страшный как казалось — нет обратной связи на goal selection.

---

## 9. Что я бы предложил как «следующий честный шаг»

(Это не roadmap — это observation про то что я вижу в коде.)

**Минимальная инвестиция, максимальная видимость:**
1. **Расширить planner allowed_actions** (включить place_/make_/sleep/noop) — это unblock'нет crafting. Несколько часов работы.
2. **Cross-episode SDM persistence** (save/load) — позволит обучаемому опыту жить между эпизодами. Stage 5 IDEOLOGY уже описала, infrastructure есть.
3. **Tree-seeking goal_prog bias** — `goal.progress` для `gather_wood` сейчас считает inventory_delta. Если добавить «приближение к ближайшему дереву» как градиент — multi-step pathfinding к ресурсу заработает без полного pathfinder'а.

**Архитектурный рефактор, больше работы:**
1. Вынести все hardcoded sets (blocking, hazard, allowed actions) в textbook + общий loader.
2. Implement `TextbookPromoter` правильно — death hypotheses через post-mortem → promoted_hypotheses → next-gen agent стартует со enriched textbook.
3. Решить Q4 идеологии явно — выбрать framing «Crafter as goal» vs «as data source» и переориентировать метрики.

---

## 10. Один экран — карта файлов для будущего меня

Если возвращаюсь после месяца и не помню где что:

```
configs/crafter_textbook.yaml        ← FACTS (читай первым)
configs/promoted_hypotheses.yaml     ← должен заполняться, сейчас пуст
docs/IDEOLOGY.md                     ← философия (раздел 1: четыре категории)
docs/architecture-report-2026-05-11.md ← этот файл

src/snks/agent/
├── crafter_pixel_env.py             ← env wrapper, determinism patch
├── crafter_textbook.py              ← YAML loader → CrafterTextbook
├── crafter_spatial_map.py           ← cognitive map
├── perception.py                    ← info["semantic"] → VisualField
├── vector_world_model.py            ← SDM (главная learned-without-grad память)
├── vector_bootstrap.py              ← textbook → SDM seeding
├── vector_sim.py                    ← simulate_forward rollout
├── vector_mpc_agent.py              ← main per-step orchestration (~2000 LOC)
├── goal_selector.py                 ← threat priority from textbook
├── stage90r_emergency_controller.py ← safety override layer
├── stage90r_local_model.py          ← learned actor .pt (advisory)
├── stimuli.py                       ← Survival/Homeostasis/Curiosity
└── post_mortem.py                   ← damage attribution

experiments/
├── stage90r_collect_local_dataset.py ← collect for training actor
├── stage90r_train_local_evaluator.py ← train actor .pt
├── stage90r_eval_local_policy.py     ← canonical eval (what we run)
└── record_stage91_seed_video.py      ← video recording with overlay
```

Это вся «живая» поверхность Stage 91. Всё остальное — либо legacy (DAF, GWS, DCAM), либо вспомогательное (curriculum, scenario_runner, viz).

---

## Резюме

Идеологически архитектура **на 70-80% выровнена с IDEOLOGY.md v2**. Категории 1, 2, 4 — в хорошей форме. Категория 3 (experience) работает within-episode, но **не пересекает границу эпизодов** — это главный архитектурный пробел.

Learned часть (CNN + local actor `.pt`) — минимальная, символическая часть планирования держит большую часть веса. С `actor_share=0` мы фактически работаем как **символический MPC агент с HDC associative memory**, что соответствует идеологическому принципу top-down + symbolic reasoning, но создаёт вопрос «что мы делаем с trained `.pt`» — он практически decorative.

Текущая lava-fix сессия добавила хорошие хирургические фиксы (planner_bootstrap fallback теперь не выбирает летальные actions, sim знает про перцепированные blocking-тайлы, lava честно скорится). Но **обнажила** несколько архитектурных дыр которые годами существовали (crafting не работает, anti-stuck не работает, textbook не persistится между эпизодами).

Следующий честный шаг — **расширение allowed_actions** (1-2 часа) даст самый видимый эффект и unblock'нет наблюдение реального крафтинга поведения. После него — **cross-episode SDM persistence**. После того — **textbook promotion**. Это последовательное движение к принципам 5 и 6 идеологии без рефакторинга остальных слоёв.

---

## Дополнение от 2026-05-12 — outcome-role landed (PCCS step 1)

Допустимая правка к разделам 3 (Experience), 7 (scorecard), 8 (gaps).

### Что добавилось в код

`VectorWorldModel` теперь хранит две роли в **одной** `CausalSDM`:

```
bind(concept_vec, action_vec)                       → effect_vec   (физика среды, существовало)
bind(bind(concept_vec, action_vec), role_outcome_h) → outcome_vec  (исход на горизонте H, ново)
```

`outcome_vec` — HDC-бандл из `survived_h ∈ {alive, dead}`, `damage_h` (thermometer 0..10) и `died_to` (concept причины смерти или `none`). Запись происходит через `model.learn_outcome(concept, action, outcome)` после H=5 env-шагов от каждого принятого решения. Чтение — через `model.predict_outcome(concept, action) → (decoded_dict | None, confidence)`, адрес XOR-ортогонален адресу `effect_vec` для той же пары, поэтому outcome-write'ы **не загрязняют** physics predict.

Читатель — новый `OutcomeStimulus` в `src/snks/agent/stimuli.py`. Per-кандидат разрешает `(concept, action)` через `resolve_outcome_pair(plan.steps, vf.near_concept)` (общий helper, который использует и lifecycle-recorder при write'ах, и стимул при read'ах — read/write side согласованы по адресу substrate'а):

- `do/make/place` plan → `(plan.steps[0].target, action)`
- `sleep` plan → `("self", "sleep")`
- motion / motion_chain plan → `(vf.near_concept, plan.steps[0].action)`
- baseline (empty plan) → `(vf.near_concept, "noop")`

Стимул контрибутит **только negative recall** (survived outcomes = 0, чтобы не подавлять exploration never-tried пар). Это решает root-cause провала первой попытки (отдельная EpisodicSubstrate с bundled-context decision_vec давала uniform recall и неконтрастный сигнал между кандидатами).

Persistence — через существующие `VectorWorldModel.save()/load()`. Один `.pt` на seed (`_docs/wm/seed_{N}.pt`) хранит всё: концепты, действия, роли (включая outcome_h), требования, и SDM content. Размер файла сжат с 6.5 GB → ~3.3 GB через хранение `seed` вместо tensor `addresses` (которые регенерируются deterministically из seed при load).

### Результат на эталоне (seed 17 ep 0, full-profile, strict)

Cross-episode эффект подтверждён:
- **gen1** (свежий substrate): 192 steps, death=zombie, 3 crafting events (place 23, place 47, make_wood_sword 48).
- **gen2** (substrate из gen1 загружен): 220 steps (MAX, нет env-смерти), death=dehydration, 3 crafting + 4-й wood_sword. **Смерть от hostile в gen1 → death-recall в gen2 → планировщик увёл агента от опасных контекстов → дошёл до конца эпизода, упёрся в новый failure mode (вода).**

Multiseed (seeds 7/17/27/37/47, gen1 vs gen2 mean): `do +45%, craft +44%`. Tendency не равномерная (две регрессии, три выигрыша) но направление есть.

Determinism: два независимых run'а одного и того же seed с identical pre-loaded substrate → байт-идентичные action+position трейсы 220/220 шагов. Outcome-role lifecycle не ломает Stage 91 determinism guarantee.

### Что обновляется в scorecard (раздел 7)

| Принцип | Было | Стало |
|---|---|---|
| 3. Continuous learning | 🟡 (SDM continuous within episode) | 🟢 — cross-episode continuity активна |
| 6. Система, не агент | 🟡 (death attribution есть, persistence нет) | 🟢 — основной gap (SDM persistence) закрыт |

### Что закрыто в gaps (раздел 8)

- ✅ **#3 SDM не персистится между эпизодами** — закрыт через outcome-role + `model.save/load` lifecycle hooks в `run_vector_mpc_episode`.

### Что остаётся открытым (новое для PCCS step 2+)

- **#5 Принцип «Knowledge flow textbook ← experience»** всё ещё ❌. TextbookPromoter не активирован — outcome-роль хранит знания в substrate, но **не выливает** их в декларативные правила. Следующий честный шаг — pcccs step 2: death hypotheses, повторяющиеся между эпизодами, promote'ятся в `promoted_hypotheses.yaml`.
- **Q4 идеологии** (Crafter as goal vs data source) — не закрыт.
- **Замена lex-tuple scoring на substrate decode** — PCCS step 3.
- **Kuramoto phase coupling вместо XOR-binding** — PCCS step 4.

### Файлы, затронутые outcome-role работой

- `src/snks/agent/vector_world_model.py` — `learn_outcome`/`predict_outcome` методы, новая роль `__OUTCOME_H__`, сжатый save (без addresses).
- `src/snks/agent/stimuli.py` — `OutcomeStimulus` + `resolve_outcome_pair` helper.
- `src/snks/agent/vector_mpc_agent.py` — `_OutcomeRecorder` класс + 4 lifecycle-hooks + 4 новых kwargs (`enable_outcome_learning`, `world_model_path`, `outcome_horizon`, `outcome_stimulus_weight`).
- `experiments/record_stage91_seed_video.py` — `--enable-outcome-learning`, `--world-model-path`, `--outcome-horizon`, `--outcome-weight`.
- `tests/agent/test_world_model_outcome_role.py` (7 тестов), `tests/agent/test_outcome_stimulus.py` (8 тестов), `tests/agent/test_outcome_recorder.py` (5 тестов).

Спека: [`docs/superpowers/specs/2026-05-12-outcome-role-design.md`](superpowers/specs/2026-05-12-outcome-role-design.md).
