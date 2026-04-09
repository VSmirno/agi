# Допущения и ограничения по этапам

Файл фиксирует упрощения, принятые на каждом stage. Обновляется при завершении каждого stage.

---

## Stage 44 — Foundation Audit
**Что сделано:** Аудит DAF-ядра, выявление ограничений FHN-осцилляторов.
**Допущения/ограничения:**
- FHN работает в возбудимом режиме (I_base=0.5), а не осциллирующем — coupling для планирования непригоден.
- SKS формируются через SDR-инъекции, а не через динамику coupling.
- DAF остаётся как perception layer без возможности планирования.

---

## Stage 47–49 — M1: Generalization (DoorKey, MultiRoom)
**Что сделано:** 100% DoorKey + 100% MultiRoom-N3.
**Допущения/ограничения:**
- **Полная наблюдаемость** — BFS работает на полной карте. Реального обобщения нет, задача тривиальна при full obs.
- SDM не используется для навигации — BFS решает всё.

---

## Stage 54 — Partial Observability
**Что сделано:** 100% DoorKey с 7×7 view через SpatialMap.
**Допущения/ограничения:**
- SpatialMap строится из символьных observations (тип/цвет клетки), не из пикселей.
- Агент помнит карту идеально — нет шума, нет forgetting.

---

## Stage 59 — VSA Causal Induction
**Что сделано:** 100% generalization на unseen colors через bind(X,X)=identity.
**Допущения/ограничения:**
- Тестируется только цветовая генерализация, не объектная (key→ball не проверялось).
- Демонстрации синтетические, не из реальной среды.

---

## Stage 60–61 — World Model + Demo-Guided Agent
**Что сделано:** 100% QA L1-L3, 100% DoorKey + LockedRoom.
**Допущения/ограничения:**
- Правила получены из синтетических демонстраций, не из реального взаимодействия со средой.
- Per-rule SDMs ограничены ~50 элементами — не масштабируются на большие домены.

---

## Stage 62 — CLS World Model
**Что сделано:** 100% QA L1-L4, neocortex + hippocampus.
**Допущения/ограничения:**
- Неокортекс = обычный Python dict (exact match). Нет generalization за пределами VSA identity property.
- Navigation policy на SDM сделала навигацию хуже (16% vs 44%) — BFS оставлен без изменений.
- Write-on-surprise: 77% записей пропускается, но hippocampus недообучен на редких ситуациях.

---

## Stage 63 — Abstraction + Crafter
**Что сделано:** 100% Crafter QA, 25 auto-категорий.
**Допущения/ограничения:**
- Абстракции извлекаются из символьных правил, не из пикселей.
- 25 категорий покрывают Crafter, но не тестировались на других доменах.
- Craft-действия (make_*) не покрыты prototype memory — 0 прототипов из-за редкости ситуаций.

---

## Stage 64 — No Synthetic
**Что сделано:** 93% Crafter QA без синтетических демо.
**Допущения/ограничения:**
- Exploration curiosity-driven, но не affordance-based — агент случайно натыкается на правила.
- 7% miss rate остаётся — редкие объекты (diamond, iron) не обнаруживаются за отведённое время.

---

## Stage 65 — Calibrated Uncertainty
**Что сделано:** Brier=0.12, calibration curve близка к идеальной.
**Допущения/ограничения:**
- Калибровка проверена только на known объектах. Поведение на out-of-distribution не тестировалось.
- Confidence threshold подобран эмпирически, не обоснован теоретически.

---

## Stage 66 — Pixels (Prototype Memory)
**Что сделано:** 100% Crafter QA из пикселей, prototype memory k-NN.
**Допущения/ограничения:**
- **Conv2d сегфолтит на AMD ROCm** через MIOpen backend. Исправлено: `torch.backends.cudnn.enabled=False` включает fallback kernel (медленнее, но работает). Обучение теперь на GPU (1.8x speedup vs CPU).
- VQ Patch Codebook отброшен: decode→symbols теряет информацию на каждом шаге.
- Prototype search в Phase 3 использует ground truth near (символьный), не CNN.
- make_* правила по-прежнему не покрыты (0 прототипов для craft actions).

---

## Stage 67 — Symbolic Near → CNN Near
**Что сделано:** NearDetector (CNN→argmax), CrafterPixelEnv без _to_symbolic(), smoke 99%, QA 100%.
**Допущения/ограничения:**
- **Инвентарь** берётся из `info["inventory"]` — проприоцепция (агент помнит что взял). Менять не нужно.
- **Навигация** — случайный walk + `_detect_near_from_info(info["semantic"])`. Stage 68 убирает info["semantic"].
- near_labels для обучения CNN = ground truth из той же символики (circular dependency: убираем символику, обучаясь на ней).
- Prototype collection в Phase 3 использует ground truth near для поиска ситуаций, не NearDetector.
- make_* правила по-прежнему не покрыты.
- Smoke test 99% оптимистичен: большинство кадров — пустое поле (easy "empty" class).

---

## Stage 68 — Pixel Navigation (когнитивная карта)
**Что сделано:** CrafterSpatialMap + find_target_with_map. Nav smoke 72%, QA 100%, regression 100%.
**Допущения/ограничения:**
- `info["player_pos"]` остаётся — проприоцепция.
- `info["inventory"]` остаётся — проприоцепция.
- CrafterSpatialMap: nav map=72% vs random=69% — небольшое преимущество (знакомые позиции).
- coal/iron/diamond: 1/0/0 из 50 seed — редкие объекты не покрыты навигацией.
- make_* прототипов 0 — table не создаётся при random walk (нет wood в инвентаре).
- near_labels для обучения CNN по-прежнему из символики (circular dependency: Stage 69).

---

## Stage 70 — ScenarioCurriculum (2026-04-06)
**Что сделано:** FSM-цепочки сценариев с OutcomeLabeler. 6 классов (empty/tree/stone/coal/iron/table). Smoke=68.8%, QA=100%, regression=100%.
**Компоненты:**
- ScenarioRunner: FSM executor с directional probing (do в 4 направлениях), window labeling W=5.
- CrafterControlledEnv: прямое редактирование мира — reset_near() и reset_with_items().
- _collect_empty_walk_frames(): random walk + semantic GT для "empty" (соответствие test distribution).
- STONE/COAL/IRON_CHAIN: controlled env для редких объектов (100% success vs ~3% natural).

**Допущения/ограничения:**
- **Nav encoder Phase 0** по-прежнему через exp122 (Stage 68 pipeline, символьные траектории). Circular dependency не устранена на уровне nav encoder. Stage 71 устраняет.
- **Stone в smoke=4.5%** из-за domain gap: controlled stone помещается в grassland, а в random walk stone появляется в горах. Smoke проходит за счёт empty=97.5%.
- **use_semantic_nav=True** используется в TREE/STONE/COAL/IRON_CHAIN для навигации — semantic scaffolding. Stage 71 убирает.
- **_balance_chunk monkeypatching**: отключение врагов через lambda. Работает, но хрупко.
- **table smoke=0/0**: таблицы не встречаются в random walk (только player-placed), поэтому GT=0 для table в smoke test.
- **do near coal 6/50, iron 17/50** в QA: coal/iron embedded в камне, навигатор не находит их за 300 шагов. QA проходит за счёт controlled prototype collection в phase4.

---

## Stage 71 — Text-Visual Integration (2026-04-07)
**Что сделано:** Соединение текстового и визуального пайплайнов. ConceptStore как единое омнимодальное хранилище. Каузальные правила из текстового "учебника" (YAML), visual grounding через co-activation, backward chaining planner, reactive zombie handling.
**Компоненты:**
- ConceptStore: unified Concept (visual + text_sdr + attributes + causal_links + confidence).
- CrafterTextbook: YAML с 10 атомарными правилами, regex-парсер, load_into(store).
- ChainGenerator: backward chaining через ConceptStore.plan() → ScenarioStep.
- GroundingSession: co-activation — K=5 visual samples + text SDR per concept.
- ReactiveCheck: zombie nearby → sword? attack : flee.
- ScenarioRunner.run_chain_with_concepts(): reactive layer + prediction error loop.

**Допущения/ограничения:**
- **PrototypeMemory не интегрирован** в ConceptStore — ConceptStore хранит 1 z_real на концепт, PrototypeMemory хранит тысячи экземпляров. Разные задачи.
- **Confidence delta фиксированный** (±0.15), не байесовский update.
- **Surprise только логируется** — неожиданные события не порождают новые правила автоматически. Требует отдельной проработки.
- **Flee = простая эвристика** (3-5 случайных шагов от врага).
- **Один тип врага** (zombie). Skeleton, arrow — будущий stage.
- **Нет стратегии "сначала крафт меча"** — reactive слой не планирует заранее.
- **Нет decay правил** — confidence не падает со временем.
- **Planner не оптимизирует порядок** — может собирать wood дважды.
- **Nav encoder Phase 0 на exp122** — фокус этого stage = text-visual, не nav cleanup.
- **use_semantic_nav=True остаётся** для редких объектов.
- **find_causal disambiguation** — при одинаковых requires выбирается наиболее специфичный match (по количеству requires items). Wood_sword/wood_pickaxe неразличимы по inventory.
- **Zombie + Survival Gate 5 PASS** (exp128d) — zombie_deaths 41→1, episode length 169→446 (2.65x). Survival rules (food/drink/energy) = основной вклад в выживаемость. Zombie flee = дополнительный.

---

## Stage 72 — Perception Pivot (2026-04-07)
**Что сделано:** Замена supervised NearDetector на ConceptStore.query_visual_scored() (cosine sim). Убран GT semantic nav. Автономный цикл perceive→decide→act→learn. Experiential grounding (one-shot + EMA). Drive-based goal selection. Prediction-verification loop.
**Компоненты:**
- perception.py: perceive(), on_action_outcome(), select_goal(), get_drive_strengths().
- ConceptStore.query_visual_scored(): возвращает (concept, similarity) для детекции "unknown".
- agent_loop.py: автономный цикл вместо ScenarioRunner chains.
- engine.py: use_semantic_nav=False, spatial_map в engine state.

**Допущения/ограничения:**
- **CNN encoder frozen** (exp128) — не дообучается в runtime. Фичи могут быть недостаточны для stone vs coal.
- **One-shot grounding noisy** — первый z_real может быть нетипичным. EMA сглаживает.
- **Cosine threshold=0.5** — подобран эмпирически, не адаптивный.
- **Spatial map cold start** — пустая карта в начале эпизода, random walk для заполнения.
- **Drive competition = max()** — нет GWS winner-take-all, простой argmax по drive strengths.
- **Sleep = 3 шага** — фиксированное количество, не адаптивное.
- **Probing = 4 directions** — для "do" пробуем все стороны. Неэффективно, но надёжно.
- **Replan interval=20** — фиксированный, не событийный.
- **DAF/SKS не интегрированы** — ConceptStore.query_visual() заменяет оба. Oscillator perception deferred.
- **info["player_pos"] остаётся** — проприоцепция.
- **info["inventory"] остаётся** — проприоцепция.
- **NearDetector code сохранён** — используется для backward compat (zombie tracking в wrapper).
- **exp130 результаты:** tree nav 60% PASS, stone 0%, coal not grounded, survival 74, verification 0.
- **exp131 результаты (Stage 74, homeostatic):** tree nav 50.5% PASS, 7 concepts grounded (incl stone), survival 138, verification 3 PASS.
  - HomeostaticTracker: body rates from observation, preparation drive (proactive sword craft).
  - Relative matching (margin ≥0.1): fixes 256-dim inter-class confusion.
  - Sword emergence: plan reaches step 2/4 (make sword near table) but table recognition fails.
  - Root cause: 256-dim inter-class similarity too high (stone vs water: 0.82).

---

## Stage 74 — Homeostatic Agent (2026-04-08)
**Что сделано:** Убраны ВСЕ hardcoded drives и ReactiveCheck. Поведение из body rates + world model + curiosity. CNN 256→512 channels.
**Компоненты:**
- HomeostaticTracker: rate of change body variables + conditional rates (STDP-like).
- compute_drive: urgency = 1/steps_until_zero (pure body physics).
- compute_curiosity: model incompleteness (biological drive).
- Preparation drive: trace known threat → plan to remove cause → proactive craft.
- Strategy 2: health drops → cause=zombie → kill_zombie → sword chain.
- Relative matching: margin ≥0.1 between best/second-best (fixes inter-class confusion).
- CNN 512 channels: retrained, +9% survival.
- ReactiveCheck REMOVED: flee wastes steps, drives handle zombie correctly.

**Результаты (exp131, 500 episodes):**
- Tree nav: 54.4% PASS
- Grounding: 6 concepts PASS (tree, table, empty, zombie, water, cow)
- Verification: 4 rules PASS (tree.do→wood 1.00, empty.place→table 1.00, water.do→restore_drink 1.00, cow.do→restore_food 0.80)
- Survival: 173 steps FAIL (gate ≥200). Стабильно, не растёт с обучением.
- Stone: 0% FAIL. Sword: 0/500.

**Допущения/ограничения:**
- **Survival 173 — потолок текущей архитектуры.** 500 эпизодов не улучшают — learning saturated.
- **Sword 0/500** — agent выводит правильный план (kill_zombie→sword→table→wood) но не успевает собрать 3 wood до смерти (~80 шагов на 3 дерева, zombie убивает за ~100-170).
- **512-dim features:** intra-class 0.99, inter-class 0.55-0.82. Relative matching помогает но не решает.
- **Bottleneck = perception speed.** Agent тратит ~80 шагов на поиск 3 деревьев при текущем качестве cosine matching. Нужно либо быстрее perception, либо рicher features.
- **ReactiveCheck убран** — flee хуже чем терпеть удары. Drives правильно выводят "крафти меч".
- **Архитектура чистая:** zero hardcoded strategy. Sword emergence подтверждён (1 craft в exp131 ранних итерациях).
- **SupCon on center features не помог** (exp132 supcon: survival 169 vs 173 без). Classification features ≠ metric features — разный training objective не решается добавлением contrastive loss.
- **500 эпизодов — learning saturated.** Survival стабильно 169-173, не растёт. Bottleneck архитектурный.
- **Лучший результат:** exp131 без ReactiveCheck, 512ch CNN 4×4 grid: survival 173, tree 54%, 6 concepts, 4 rules verified.
- **8×8 grid (256ch, 3 layers) не помог:** survival 164, tree 48%. Больше позиций (64 vs 16) но quality не улучшилась.
- **Sandbox curriculum не помог:** survival 165. Prototypes не стали точнее — CNN features fundamentally не подходят для cosine matching.
- **Diagnostic: 100% stale map entries.** perceive_field возвращает "tree" на траве. 4 тайла в одной ячейке 4×4 → смешанные features. 8×8 (~1 тайл) не решило — проблема в training objective, не разрешении.
- **512ch×8×8 grid: survival 169.** Лучше чем 256ch×8×8 (164) но хуже чем 512ch×4×4 (173). Чистые ячейки (1 tile) не помогают — CNN features всё равно не metric space.
- **Root cause подтверждён окончательно:** classification CNN features (cross-entropy) не образуют metric space для cosine matching. Проблема в training objective, не в resolution/channels/grid. Протестировано: 256ch×4×4, 512ch×4×4, 256ch×8×8, 512ch×8×8, SupCon, sandbox, 500 episodes. Ничего не помогает. Нужен metric learning или near_head для detection.

  - 3 концепта grounded из опыта: tree, water, cow.
  - Motor babbling (15% prob) → action outcome → one-shot grounding → perception bootstrap.
  - Survival +50% (49→74) через grounding cow/water для еды/питья.
  - Stone FAIL = не перцепция, а planning execution (craft chain не реализован).
  - Verification FAIL = predict/verify не подключён в babble path.

## Stage 75 — Per-Tile Visual Field (2026-04-09)
**Что сделано:** Заменил cosine matching на classification CNN с no-stride FCN architecture.
tile_head через Conv1×1 на output feature map. Per-tile labels через semantic map как teacher.
Полный viewport 7×9 tiles (49×63 px из 64×64), исключая inventory bar и черную границу.

**Компоненты:**
- TileSegmenter: 3× Conv3×3 (stride=1) + BN + ReLU → AdaptiveAvgPool(7,9) → Conv1×1(64,12). 57K params.
- viewport_tile_label: корректный coordinate mapping с учётом render transpose + sprite offset +1.
- ConceptStore.plan(goal, inventory): skip prerequisites уже в inventory.
- Textbook restore_health rules: do cow/water restores health (matches Crafter implicit regen).
- Homeostatic bugs fixed: tb.body_rules property access, plan verification before advance,
  cumulative requires for do, probe_dirs rotation, make/place unconditional advance.

**Результаты (exp135):**
- Tile accuracy: **82%** PASS (was 39% in Stage 74)
- Wood collection: **4.7/ep avg, 65% reach ≥3 in 17 steps** PASS
- Survival with enemies: **178 avg** FAIL (gate ≥200). Variance 94-264.
- Per-class acc: water 83%, tree 67%, stone 65-85%, coal 97%, cow 81%, zombie 100% (small n), skeleton 0-63%.

**Допущения/ограничения:**
- **Survival ≥200 — architectural limit.** 11+ кодовых фиксов не дали результата выше 189.
  Pattern: каждый фикс решает симптом, но рождает новый — signal of architectural gap.
- **Root cause:** plan execution linear/blind. Agent commits to kill_zombie (4 steps), нет
  forward simulation "выживу ли я во время этого плана?". Zombies attack during execution, dies.
- **Skeleton detection 0-63%:** training distribution has 168 skeleton tiles of 670K total (0.025%).
  Class-weighted CE помогает частично но не решает. Accepted limit.
- **Placed table detection unreliable:** 70 table samples in training. Agent кладёт table,
  не находит потом для make_wood_sword. Spatial_map manual update — procedural patch, reverted.
- **Coordinate mapping discovered via visual debug.** Crafter canvas.transpose((1,0,2)) + sprite
  offset +1 row. Labels were wrong in Stage 74, per-tile accuracy был ограничен noise в GT,
  не feature quality. See `feedback_visual_debug.md`.
- **Procedural patches rejected per ideology:** hardcoded flee reflex, flee_timer panic,
  stuck detection random, range-based threat check, manual spatial_map updates. Все были tried
  and reverted. See `feedback_no_hardcoded_reflexes.md`.
- **Cumulative requires check correct:** do(tree) stays until sum of requires across all
  subsequent plan steps is met. Prevents place_table at 2 wood when make_sword also needs 1.
- **explore_action babble conflicts with plan:** during plan execution, babble may consume
  resources (make_wood_pickaxe съедает wood для sword). Solution was to use random walk в
  fallback, но это тоже procedural patch и reverted.
- **Best variance observed:** survival ranges 94-264 within single 20-episode run. Stochastic
  due to zombie spawn positions. Increasing sample size would stabilize but not move mean above 200.
- **Next:** Stage 76 — continuous model learning / forward simulation. Model-based планирование
  через ConceptStore + tracker + learned value function. See
  `docs/superpowers/specs/2026-04-09-stage76-continuous-model-learning-design.md`.

