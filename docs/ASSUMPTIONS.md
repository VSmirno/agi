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

