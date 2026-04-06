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
- **Conv2d сегфолтит на AMD ROCm** — CNN encoder принудительно на CPU.
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
**Что сделано:** CrafterSpatialMap + find_target_with_map, 23/23 тестов, exp124 готов к запуску на minipc.
**Допущения/ограничения:**
- `info["player_pos"]` остаётся — проприоцепция (агент знает где его тело).
- `info["inventory"]` остаётся — проприоцепция (агент помнит что взял).
- `CrafterSpatialMap` хранит near_str на позицию (one label per cell, не полная карта).
- Greedy step-toward навигация, не оптимальный путь — может ходить кружными путями.
- near_labels для обучения CNN по-прежнему из символики (circular dependency: Stage 69).
