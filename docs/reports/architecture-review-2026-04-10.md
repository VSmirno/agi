# Architecture Review — живой путь Stage 76

**Дата:** 2026-04-10
**Scope:** диагностика когерентности и разрывов между `docs/IDEOLOGY.md` и кодом живого Crafter-пайплайна Stage 76 (exp136)
**Статус:** Phase A (диагностика) — Phase B (проектное предложение) по запросу после ревью этого документа
**Повод:** запрос после подряд случившихся реверсов Stage 76 v1/v2/v2.1 — ощущение хаотичного перебора гипотез, потребность остановиться и пересмотреть основание

---

## TL;DR

Живой путь Stage 76 содержит **два параллельных world model'я**, которые не делят ни state representation, ни learning signal, ни reasoning, и переключаются через `if/else` каждый шаг. Все четыре прогона Stage 75/76 дали survival в пределах ±10 от 178 шагов не случайно — это **структурная эквивалентность**. Wood collection в smoke-тесте упал с 65% до 40% не от шума, а из-за **активной интерференции** между goal-directed planner'ом (Path A) и utility-maximizing memory (Path B).

`link.confidence` — главный заявленный цикл обучения — **не читается** нигде, кроме `compute_curiosity`. Planner не различает правило с confidence=0 и с confidence=1. Body rules prior тает через EMA за ~60 шагов. Textbook содержит преднамеренную ложь как workaround для отсутствующей поддержки непрямых причинных цепочек.

Stage 77 в текущей формулировке («forward sim через SDM rollouts, reuse substrate») — **четвёртый виток одной и той же петли** (74→75→76→77). Он улучшит Q-estimate, но не добавит в систему новизны: Path B остаётся on-policy к Path A, а Path A не учится. **Прогноз Stage 77 без изменения основы: survival 185-195, gate ≥200 не пройден.**

Проблема **не в forward simulation**. Проблема в основе, над которой forward simulation будет работать.

---

## F1 — Структурная находка: два параллельных world model'я (CRITICAL)

`run_continuous_episode` (`src/snks/agent/continuous_agent.py:215-317`) на каждом шаге переключается между двумя независимыми механизмами:

```python
# continuous_agent.py:248-267
sdm_ready = len(sdm) >= min_sdm_size
if sdm_ready and n_similar >= bootstrap_k and recalled:
    action_scores = score_actions(recalled, inv, tracker)
    action_str = select_action(action_scores, temperature, rng)   # Path B
else:
    action_str = _bootstrap_action(...)                            # Path A
```

### Path A — символический планировщик
- `perception.select_goal` → `concept_store.plan` (backward chaining)
- Знания из YAML (`configs/crafter_textbook.yaml`, 10 правил) → `ConceptStore.causal_links`
- Goal выбирается по формуле `urgency = 1/steps_until_zero` над фиксированным набором переменных
- State representation: `vf.near_concept` (string) + `inventory` (dict)
- Learning update: `concept_store.verify` → `link.confidence += ±0.15`

### Path B — эпизодическая память
- `StateEncoder.encode` → 4096-bit SDR
- `EpisodicSDM.recall` → popcount top-k
- `score_actions` — deficit × delta взвешивание
- State representation: 4096-bit sparse vector, без отображения на concept_id
- Learning update: `sdm.write(Episode)` — append

### Что системы делят, а что — нет

| | Path A (planner) | Path B (SDM) |
|---|---|---|
| Body vars | `HOMEOSTATIC_VARS = {health,food,drink,energy}` захардкожен (`perception.py:38,696,788`) | `tracker.observed_variables()` — динамически (`episodic_sdm.py:147`) |
| `conditional_rates` | читает через `get_rate()` + «Strategy 2» | не читает вообще |
| `observed_max` | не читает | читает через `deficit = obs_max - current` |
| State representation | `(near_concept, inventory)` | 4096-bit SDR |
| Learning signal | `link.confidence` | `Episode` append |
| Reasoning | backward chaining N шагов | 1-step scoring |

**Не существует функции**, которая читает что-то из одной системы и обновляет другую. Системы делят один объект (`HomeostaticTracker`) и даже его используют асимметрично.

Это не интеграция. Это **bypass** — один из двух агентов рулит за шаг, второй спит.

### Последствие для метрик Stage 75/76

Memory on-policy к bootstrap: первые ~10-12 эпизодов buffer пуст → работает Path A → все действия генерируются планировщиком. Buffer заполняется → Path B активируется → но `recall` возвращает только действия, которые Path A уже выбирал в похожих состояниях. `score_actions` + softmax = **шумная имитация Path A**.

Все 4 прогона (Stage 75 baseline + 3 варианта Stage 76) попали в 166-177 **по построению**. Это не дисперсия — это структурная эквивалентность.

### Отсутствие в `IDEOLOGY.md`

Таблица «Уровни мозга» из IDEOLOGY.md описывает **одну** цепочку: V1 (CNN) → ConceptStore (перцепция+каузалка) → CrafterSpatialMap → Drives → планирование. SDM в этой таблице **отсутствует вообще**. Она появилась в Stage 76 как «memory to substitute forward simulation» и легла сбоку, не встроившись в архитектуру. Это не расхождение «декларация vs реализация» — это **рост вбок**, не отражённый в идеологии.

---

## F2 — Активная интерференция Path A/B, пропущенная в Stage 76 report (HIGH)

Stage 76 report формулирует диагноз как «memory ≡ scripted baseline». Но при пересмотре gate-таблицы обнаруживается **регрессия wood collection в smoke-тесте** (БЕЗ врагов):

| Вариант | Survival | Wood ≥3 (smoke) |
|---|---:|---:|
| Stage 75 (чистый Path A) | 178 | **13/20 (65%)** |
| Stage 76 v1 FIFO | 177 | 8/20 (40%) ↓ |
| Stage 76 v2 attention | 166 | 6/20 (30%) ↓ |
| Stage 76 v2.1 +body filter | 173 | **3/20 (15%)** ↓↓ |

Smoke — это чистый тест способности собирать дерево. Path B не просто «не добавил value» — он **активно мешал** Path A делать то, что Path A умел.

### Механизм интерференции

`score_actions` (`episodic_sdm.py:116-163`) взвешивает **все** body_delta одновременно со своим deficit. Когда wood однажды попал в `tracker.observed_max` (агент собрал 9 wood), wood начинает фигурировать в scoring как «ещё одна дефицитная переменная». SDM-путь учится оценивать действия как «универсально хорошие по сумме дефицитов» — и тянет одеяло с Path A, который в этот момент сфокусирован на **одной** цели (`goal=wood`, backward chain).

### Парадигмы несовместимы

- **Path A — goal-directed:** один goal за раз, backward chaining, execute plan. Objective: «собрать wood».
- **Path B — utility-maximizing:** взвешенная сумма по всем дефицитам, softmax. Objective: «действие, максимизирующее ∑ deficit × delta».

Переключать их через `if/else` каждые N шагов = резко менять optimization criterion. Агент дёргается между «иду к дереву по плану» и «иду туда, где в среднем было неплохо».

Stage 76 v2.1 +body filter («attention только по body_variables») — это была попытка починить симптом (wood мешает): отфильтровать wood из scoring. Result: survival +7 steps, но wood collection упал до 15%. Потому что **починили не ту вещь**. Проблема не в том, что wood в scoring — проблема в том, что две парадигмы пересекаются.

---

## F3 — `HOMEOSTATIC_VARS` hardcoded при декларации "no hardcoded drives" (HIGH)

### Evidence

`perception.py:38` — module-level константа:
```python
HOMEOSTATIC_VARS = {"health", "food", "drink", "energy"}
```

5 вхождений, все в живом пути:
- `:67` — default для `self.rates`
- `:111` — итерация в `HomeostaticTracker.update`
- `:696` — **итерация в `select_goal`** (это Path A, запускается каждый шаг через bootstrap)
- `:788` — `get_drive_strengths` (UI helper)

### Декларация

IDEOLOGY.md, Stage 73: *«Drives не хардкодятся. Они НАБЛЮДАЮТСЯ. Приоритет = скорость падения homeostatic переменной. Не формула программиста, а наблюдение агента за своим телом.»*

Stage 76 design spec: *«Replace hardcoded drive list with `tracker.observed_variables()`.»*

Stage 76 добавил механизм (`observed_variables()`, `observed_max`) и использует его в `episodic_sdm.score_actions` — но **не заменил** `HOMEOSTATIC_VARS` в `select_goal`. Старый и новый механизмы существуют параллельно.

### Почему Gate 5 lint это пропустил

`tests/test_stage76_no_hardcode.py` сканирует только `src/snks/memory/`. `perception.py` вне области проверки. Поэтому формально Stage 76 Gate 5 PASS, а фактически ideology нарушена в том самом файле, где задекларирована.

### Связь с F1

Это прямое следствие F1: есть два мира, и один применяет новые правила, другой — старые.

---

## F4 — Hardcoded strategies под видом emergent behavior (HIGH)

### Evidence

`perception.py:704-773` — `select_goal` содержит три явно написанные программистом ветки:

```python
# perception.py:704 — Preparation drive
for (concept_id, var), rate in tracker.conditional_rates.items():
    if var == "health" and rate < -0.5 and concept_id != "_background":
        ...
        urgencies["preparation"] = min(0.1, abs(rate) * 0.05)
```
4 магических числа: `-0.5` порог угрозы, `0.05` масштаб, `0.1` cap, и неявно — «health» как приоритетная переменная.

```python
# perception.py:751 — Strategy 1: direct restore
plan = concept_store.plan(f"restore_{critical}", inventory)
if plan:
    return f"restore_{critical}", plan

# perception.py:757 — Strategy 2: find cause → remove it
if tracker:
    worst_cause = None
    ...
    if worst_cause:
        ...
```

### Декларация

IDEOLOGY.md, Stage 73: *«НИКОГДА не хардкодить стратегию, приоритеты или рефлексы. Всё должно вытекать из цели + мировая модель + опыт. Если хочется написать `if zombie: fight` — СТОП.»*

### Как это нарушается

«Strategy 1 сначала, Strategy 2 если первая не сработала» — это написанный программистом приоритет. Агент не может выучить, что в этой ситуации Strategy 2 важнее. Нет механизма, который это выбрал бы эмпирически.

«Preparation drive activated when rate < -0.5» — это программист решил что `-0.5` значит «серьёзно». Агент не наблюдал эту границу.

Это буквально «if zombie: fight», только завёрнуто в чтение `conditional_rates`.

### Дополнительно: hardcoded defaults и магические константы

| Место | Что |
|---|---|
| `perception.py:697` | `inventory.get(var, 9)` — «магическая 9» для отсутствующих body vars |
| `perception.py:729` | `urgencies["curiosity"] = 0.5` когда `spatial_map is None`, при том что `compute_curiosity` возвращает макс ~0.03 — inconsistency в 15× |
| `perception.py:383` | `_STAT_GAIN_TO_NEAR = {"food": "cow", "drink": "water"}` — прямая Crafter-таблица |
| `state_encoder.py:278-285` | `presence_exact = {"table", "furnace"}`, `presence_suffixes = ("_sword", "_pickaxe")` при комментарии «not hardcoded to Crafter specifics» |

---

## F5 — Confidence feedback loop мёртв (CRITICAL)

### Evidence

`link.confidence` — **записывается** в 4 местах:
- `concept_store.py:295-297` — `ConceptStore.verify` (confirm/refute)
- `concept_store.py:339-344` — `verify_after_action` (Prediction object path)

`link.confidence` — **читается** в 1 месте в живом пути:
- `perception.py:630` — `compute_curiosity` для среднего знаний (`mean_conf` → `knowledge_gap` → curiosity drive)

**Не читается:**
- `Concept.find_causal` (`concept_store.py:52-86`) — ищет правило по action+requires, без учёта confidence
- `ConceptStore.plan` (`concept_store.py:213-279`) — backward chaining, не учитывает confidence
- `select_goal` (`perception.py:678-776`) — не учитывает
- `predict` / `predict_before_action` — возвращают link с его confidence, но ничего с этим не делают

### Что это значит

Агент каждый шаг обновляет confidence правил. Правило «do water restores health» верифицируется в каждом эпизоде — работает оно или нет, confidence ползёт. Но `select_goal` → `plan` → `find_causal` **не спрашивают** этот confidence при выборе правила. Правило с confidence=0.0 будет выбрано точно так же, как правило с confidence=1.0.

Единственное, что confidence влияет — это мера невежества (curiosity). Это не принятие решения, это телеметрия.

### Декларация

IDEOLOGY.md, Stage 72: *«Каждое действие → verify prediction → update confidence. Непрерывное обучение. Модель мира обновляется непрерывно.»*

### Следствие

Главный задекларированный механизм обучения существует как writer-only цикл. Агент не умеет разочароваться в ложном правиле — оно просто остаётся в `causal_links` и продолжает выбираться. Это прямая причина того, почему textbook может содержать ложь (F7) и система продолжает работать: ложное правило не проваливает планирование, оно просто возвращает план, который не сработает, и никто этого не замечает.

---

## F6 — Body rules prior тает через EMA (MEDIUM)

### Evidence

`exp136_continuous_learning.py:321` → `tracker.init_from_body_rules(tb.body_rules)` — загружает 4 innate rate из YAML:
```yaml
body:
  - { concept: _background, variable: food, rate: -0.04 }
  - { concept: _background, variable: drink, rate: -0.04 }
  - { concept: _background, variable: energy, rate: -0.03 }
  - { concept: zombie, variable: health, rate: -2.0 }
```

`init_from_body_rules` пишет их в `self.rates[var]` (фоновые) и `self.conditional_rates[(concept,var)]` (zombie).

`HomeostaticTracker.update` (`perception.py:111-124`) применяет EMA с `RATE_EMA_ALPHA = 0.05` **без различения** prior и observation:
```python
old = self.rates.get(var, 0.0)
self.rates[var] = old * (1 - RATE_EMA_ALPHA) + delta * RATE_EMA_ALPHA
```

### Расчёт затухания

Через N шагов prior сохраняется как `(1 - 0.05)^N = 0.95^N`:
- N=60: 0.046 (≈5% от prior)
- N=100: 0.006 (≈0.6%)
- N=200: <0.0001

Первый эпизод живёт ~178 шагов. **К концу первого эпизода innate prior практически стёрт.**

### Следствие

Агент каждый episode «с нуля» узнаёт что food падает медленно и что zombie кусает. Preparation drive (F4), который активируется по `conditional_rates` от zombie, тухнет как только observation начинает доминировать. Через ~60 шагов tracker показывает «zombie почти не влияет на health», даже если агент был укушен 10 раз — потому что большую часть времени health не менялся (agent спал, шёл, делал craft) и средний delta около нуля.

### Декларация

IDEOLOGY.md: *«Как родитель объясняет ребёнку ... pre-wired. Не учится видеть — рождается с V1.»*

Но в коде prior — это просто «начальное значение», которое observation стирает. Это не innate в биологическом смысле. Это default.

---

## F7 — Textbook содержит преднамеренную ложь (MEDIUM)

### Evidence

`configs/crafter_textbook.yaml:47-50`:
```yaml
# Indirect: in Crafter, food/drink > 0 regenerates HP passively.
# Textbook teaches this as direct causality so restore_health plans exist.
- "do cow restores health"
- "do water restores health"
```

Комментарий **прямо объясняет**: это неправда. В Crafter `do cow` → `+food`, а HP восстанавливается **пассивно** когда food>0. Прямой причинности `do cow → +health` нет.

### Почему так сделано

`ConceptStore.plan("restore_health")` использует backward chaining. Chaining ищет `causal_link` с `result="restore_health"`. Если такого правила нет — plan пустой, `select_goal` Strategy 1 возвращает `("explore", [])`.

Чтобы planner мог сгенерировать «иди к корове» для восстановления здоровья, пришлось **записать в учебник прямую причинность**. Это workaround для того, что `ConceptStore` не умеет в **непрямые** причинные цепочки вида «сохранение food>0 ⇒ passive HP regen».

### Декларация

IDEOLOGY.md, Stage 71: *«Textbook = как родитель объясняет ребёнку. Правила даны, но визуальное распознавание — только из опыта.»*

Родитель не врёт ребёнку, чтобы ребёнок принял нужное решение. Это нарушение духа идеологии.

### Масштаб проблемы

Stateful/indirect причинность — это НЕ только health. Любая «подготовка» (крафт заранее, накопление ресурсов для будущего goal'а) требует того же механизма. Текущий planner не умеет ни одно из этого без workaround'ов в YAML.

---

## F8 — Planner conflates nouns/verbs (MEDIUM)

### Evidence

`configs/crafter_textbook.yaml:41`: `"do zombie with wood_sword kills zombie"`
→ парсер (`crafter_textbook.py:42-48`) создаёт:
```python
CausalLink(action="do", result="kill_zombie", requires={"wood_sword": 1})
```

`select_goal` Strategy 2 (`perception.py:757-773`):
```python
for (concept_id, var), rate in tracker.conditional_rates.items():
    if var == critical and rate < worst_rate:
        worst_cause = concept_id  # "zombie"
...
for link in cause_concept.causal_links:
    if link.action == "do":
        cause_plan = concept_store.plan(link.result, inventory)  # plan("kill_zombie", ...)
```

`ConceptStore.plan("kill_zombie", inventory)` → `_plan_recursive`:
```python
# concept_store.py:252
if inventory is not None and inventory.get(goal_id, 0) >= 1:
    return  # goal satisfied
```

### Проблема

`kill_zombie` — это **поведение** (behavior), а не предмет. `inventory.get("kill_zombie", 0)` всегда 0. Проверка «goal satisfied» **никогда не срабатывает** для поведенческих целей.

Planner работает для Crafter только потому, что `kill_zombie` — терминальный одиночный глагол (нет рекурсии через него). Если бы textbook содержал «do kill_zombie gives peace», planner бы зациклился.

Более глубоко: `ConceptStore` — это **storage for nouns** (`concepts: dict[str, Concept]`, где каждый `Concept` — объект в мире). Но через `CausalLink.result` туда пролезли псевдо-существительные типа `kill_zombie`, `flee`, `restore_food`. Эти «concepts» никогда не будут grounded визуально, их нельзя положить в inventory, они не имеют атрибутов. Это — **verbs masquerading as nouns**.

### Следствие

Для текущего Crafter'а работает. Для любого расширения (несколько типов врагов, персистентные эффекты, двухступенчатые цели) сломается концептуально, не на бытовом уровне.

---

## F9 — `next_state_sdr` — dead storage, ~40 MB впустую (LOW сейчас, INFRASTRUCTURAL для Stage 77)

### Evidence

`Episode` dataclass (`episodic_sdm.py:27-44`) содержит `next_state_sdr: np.ndarray`.

**Записывается** каждый шаг в `continuous_agent.py:301,313`:
```python
next_state_sdr = encoder.encode(...)
sdm.write(Episode(
    state_sdr=state_sdr,
    action=action_str,
    next_state_sdr=next_state_sdr,
    body_delta=body_delta,
    ...
))
```

**Не читается** нигде:
- `EpisodicSDM.recall` — итерирует `ep.state_sdr`
- `score_actions` — читает `ep.action`, `ep.body_delta`
- `count_similar` — `ep.state_sdr`

### Расчёт

4096 bits × 10K capacity × 2 (state + next_state) = ~10 MB для bool. С boxing в Python и dataclass overhead — ~40 MB RAM.

### Значение

Поле заложено под Stage 77 forward sim (подтверждает, что дизайн Stage 76 уже планировал это расширение). Сейчас — неиспользуемый балласт, но инфраструктурно готов.

---

## F10 — Spatial map без confidence tracking (MEDIUM)

### Evidence

`CrafterSpatialMap.update` (`crafter_spatial_map.py:34-43`):
```python
def update(self, player_pos: tuple[int, int], near_str: str) -> None:
    y, x = int(player_pos[0]), int(player_pos[1])
    self._map[(y, x)] = near_str   # overwrite
    self._visited.add((y, x))
```

Простая перезапись. Нет confidence, нет «второй свидетель», нет истории.

`continuous_agent.py:222-227` пишет ВЕСЬ viewport каждый шаг:
```python
vf = perceive_tile_field(pixels, segmenter)
spatial_map.update((px_player, py_player), vf.near_concept)
for cid, _conf, gy, gx in vf.detections:
    wx = px_player + (gx - center_c)
    wy = py_player + (gy - (center_r - 1))
    spatial_map.update((wx, wy), cid)
```

### Расчёт шума

Сегментер 82% accuracy (Stage 75 report). Viewport 7×9 = 63 тайла. За шаг: ~63 × 0.18 = **~11 неправильных записей в spatial_map**. За 178 шагов эпизода: ~2000 неправильных записей.

Поскольку `find_nearest(target, pos)` возвращает минимальное расстояние до **любой** позиции, помеченной как `target`, один ошибочный тик на дальней дистанции может перебить правильную близкую позицию.

### Следствие

`find_nearest("table")` может возвращать позиции, где table никогда не было — один неверный классификационный тик и позиция «помечена table навсегда» пока агент не пройдёт через неё повторно и не перезапишет.

Stage 75 report: *«placed table detection unreliable: Plan cannot navigate back to them after place_table succeeds»* — это и есть симптом. Проблема не в placement detection, а в том, что spatial_map не отличает «видел table один раз с confidence 0.3» от «видел table 20 раз подряд с confidence 0.95».

---

## F11 — Три разных `perceive` функции в двух файлах (LOW, диагностично)

### Evidence

1. `perception.perceive_field` (`perception.py:189-251`) — cosine matching, Stage 72-74. **Dead** в Stage 76.
2. `perception.perceive_tile_field` (`perception.py:254-315`) — tile classification через `encoder.classify_tiles`, Stage 75 ранняя версия. **Dead** в живом пути.
3. `continuous_agent.perceive_tile_field` (`continuous_agent.py:52-83`) — отдельная копия, использует `segmenter.classify_tiles` напрямую. **Живая**.

Сигнатуры разные:
- (1) принимает `encoder: Any, concept_store: ConceptStore`
- (2) принимает `encoder: Any` (использует `encoder.classify_tiles`)
- (3) принимает `segmenter: Any` (тот же метод, но изолированный от `CNNEncoder`)

### Значение

Кто-то открывший `perception.py` не поймёт с первого раза, какая функция реально крутится. `grep perceive_field` вернёт две разных функции в двух файлах. Это диагностично, не катастрофично.

---

## F12 — 10 из 17 публичных символов в `perception.py` мёртвы (MEDIUM)

Прогрепал все публичные функции по всему репозиторию:

| Символ | Живой Stage 76 | Писано для |
|---|:-:|---|
| `perceive_field` | ❌ | Stage 72-74 (cosine matching) |
| `perceive` (legacy) | ❌ | Stage 72 |
| `perceive_tile_field` | ❌ (живой путь использует свою копию) | Stage 75 ранняя версия |
| `ground_empty_on_start` | ❌ | Stage 72 |
| `ground_zombie_on_damage` | ❌ | Stage 72 |
| `on_action_outcome` | ❌ | Stage 72 |
| `should_retrain` | ❌ | Stage 74 (metric learning) |
| `retrain_features` | ❌ | Stage 74 (metric learning) |
| `babble_probability` | ❌ | Stage 72 |
| `explore_action` | ❌ | Stage 72 |
| `compute_curiosity` | ⚠️ только через `select_goal` | Stage 73 |
| `compute_drive` | ✅ | Stage 74 |
| `select_goal` | ✅ (только Path A) | Stage 74 |
| `verify_outcome` | ✅ | Stage 71 |
| `outcome_to_verify` | ✅ | Stage 71 |
| `HomeostaticTracker` | ✅ | Stage 74, Stage 76 extensions |
| `VisualField` | ✅ | Stage 73 |

**~500 строк из 797 (≈63%) мертвы относительно живого пути.**

Stage 72/73/74 код живёт бок о бок со Stage 76 в одном файле. Каждый новый reader должен угадывать, что архитектура, что археология.

---

## F13 — 60+ модулей в `src/snks/agent/`, большинство — археология (MEDIUM)

Минимум 4 разных world model: `causal_world_model.py`, `cls_world_model.py`, `vsa_world_model.py`, `unified_world_model.py`. Ни один не в живом пути — живой путь использует `concept_store.py`.

Агенты MiniGrid era (Stages 47-64):
- `sdm_doorkey_agent.py`, `sdm_lockedroom_agent.py`, `sdm_obstructed_agent.py`
- `keycorridor_agent.py`, `putnext_agent.py`, `multi_room_nav.py`
- `demo_guided_agent.py`, `boss_level_agent.py`
- `partial_obs_agent.py`, `pure_daf_agent.py`, `embodied_agent.py`, `instructed_agent.py`, `integration_agent.py`
- `attractor_navigator.py`, `nav_policy.py`, `mission_model.py`, `pathfinding.py`, `subgoal_planning.py`, `tiered_planner.py`, `chain_generator.py`, `abstraction_engine.py`
- `daf_causal_model.py`, `stochastic_simulator.py`

Все в `src/snks/agent/` — вперемешку с живым кодом.

Файловая структура — это буквально **засушенная карта всех гипотез**, что мы пробовали. Когда пользователь говорит «хаотично пробуем гипотезы», эта карта в `ls src/snks/agent/` так и выглядит. Это не только косметика: любой grep по архитектурному понятию (например `world_model`, `near_detector`, `planner`) возвращает мусор из 10+ файлов, и приходится каждый раз выяснять, какой из них реально живой.

---

## F14 — Мис-диагноз стены в Stage 76 report (HIGH, методологическая)

### Что говорит Stage 76 report

> *Memory alone can't close the gap, because the memory is on-policy ... reactive decisions based on past similar states cannot learn to AVOID a threat that hasn't yet touched the current state.*

Это сформулировано как **дефицит forward simulation** → отсюда пивот к Stage 77.

### Что на самом деле происходит

«Memory on-policy» — это верно, но описывает **симптом**, а не причину. Причина глубже:

1. **Нет источника новизны в системе.** Path B on-policy к Path A (F1). Path A не обновляет правила по опыту (F5 — confidence не читается). Оба пути детерминистичны относительно текущего состояния.
2. **Единственный источник случайности — softmax temperature.** Это не exploration, это noise.
3. **Path A не «генерирует плохие действия, которые память запоминает, и потом память избегает».** Path A генерирует одинаковые действия раз за разом, потому что его правила не меняются. Память просто их копирует.

### Почему это важно для диагноза

Если причина — «дефицит forward simulation», то решение — добавить rollouts. Stage 77 так и формулируется.

Если причина — «нет новизны из-за замкнутого цикла Path A/B, не учащего на опыте» — то forward simulation **не** решит проблему, потому что rollouts над детерминистичной политикой просто симулируют ту же детерминистичность дальше в будущее.

Этот мис-диагноз критичен, потому что он определяет формулировку всего следующего этапа. Stage 76 report не видел интерференцию wood collection (F2) и не отрефлексировал бездействие confidence loop (F5) — поэтому предложил решение, адресующее не ту проблему.

---

## F15 — Прогноз Stage 77 в текущей формулировке (HIGH)

### План Stage 77

Из Stage 76 report:
> *Stage 77 will reuse the Stage 76 memory substrate but replace the 1-step recall with multi-step rollouts:*
> *- For a candidate action A, query SDM for (state, A, next_state) tuples*
> *- Follow the resulting next_state as a new query, get (next_state, best_A, next_next)*
> *- Repeat for N steps, accumulate expected body_delta along the trajectory*
> *- Pick the action whose N-step rollout best closes current deficits*

### Что улучшится

- Горизонт оценки действия: N шагов вперёд вместо 1.
- Раннее детектирование «через 40 шагов health достигнет 0».
- Возможно: избежит смертей, в которых «ещё 5-10 шагов назад был разворот».

### Что НЕ починится

1. **Новизна всё ещё ноль.** Transitions в SDM записаны Path A. Rollout имитирует то, что Path A **сам бы сделал** в гипотетическом будущем. Path A уже даёт 178. +forward sim = +better Q-estimate на той же политике.
2. **Интерференция Path A/B (F2) усилится.** `score_actions` — всё ещё utility-maximizing. Path A — всё ещё goal-directed. Теперь их конфликт — на длинном горизонте.
3. **Confidence loop (F5) всё ещё мёртв.** Rollout не учит, какие правила ложны. Textbook-ложь (F7) продолжает порождать невыполнимые планы.
4. **Body prior (F6) всё ещё тает.** Zombie «перестаёт кусать» к концу первого эпизода. Forward sim не восстановит prior.

### Численный прогноз

- **Survival:** 185-195 avg (модест улучшение от глубины rollout'а)
- **Wood smoke:** 8-12/20 (интерференция остаётся, F2)
- **Gate ≥200:** скорее **не пройдено**, на границе
- **После 2-3 реверсов** (rollout length tuning, attention на rollout, priority sampling, возможно rollout ensemble): формулировка Stage 78 про следующий scope expansion

### Большая картина

Stage 74 (homeostatic) → Stage 75 (per-tile perception) → Stage 76 (memory) → Stage 77 (forward sim) — **четыре витка одной петли**. Каждый стейдж строил новый слой над одной и той же несогласованной основой, не трогая основу. Stage 73 preparation drive, Stage 75 plan verification fixes, Stage 76 attention weights, Stage 77 rollouts — все это способы «умнее использовать» ConceptStore и SDM, не трогая того, что они **не делятся** друг с другом.

Продолжение этого паттерна предсказуемо: Stage 78, 79, 80 будут новыми слоями, каждый добавляющий +5-10 survival, каждый упирающийся в тот же ±10 разброс.

---

## Course of action — Forward simulation через ConceptStore

Единственный путь, который когерентен с `IDEOLOGY.md`. Он не добавляет этап поверх текущей архитектуры — он **возвращается к этапу, который должен был быть реализован вместо Stage 76**.

### Почему один курс, а не выбор из нескольких

IDEOLOGY.md (Stage 73) прямо говорит:
> *«Стратегия = forward simulation через мировую модель ... Это не RL. Это model-based planning из каузальных правил. World model уже есть — нужно научить agent'а её использовать для принятия решений.»*

Мировая модель по идеологии = `ConceptStore`. Forward simulation = рекурсивное применение причинных правил. SDM в таблице «Уровни мозга» (Stage 72) отсутствует — она появилась в Stage 76 как обход, не как реализация идеологии.

Все альтернативы, которые сохраняют или подчиняют SDM, нарушают идеологию в той или иной степени. См. Appendix C (отклонённые альтернативы).

### Принцип: три категории знания

Чтобы избежать путаницы между «textbook = cheat» и «self-learning», зафиксируем три **разных** категории:

| Категория | Что это | Где живёт | Пример |
|---|---|---|---|
| **1. Факты о мире** | декларативные причинные правила; «учитель мог бы рассказать ребёнку» | `configs/crafter_textbook.yaml` | `do tree gives wood`, `zombie moves toward player`, `food > 0 → passive HP regen` |
| **2. Механизмы** | алгоритмы, оперирующие знаниями (но не сами знания) | `src/snks/` код | forward simulation, confidence update, backward chaining, EMA, CNN forward pass |
| **3. Выученное из опыта** | то, что агент может получить только через интеракцию | `ConceptStore` runtime state | visual grounding, confidence значения, новые правила из surprise, observed body rates, spatial map |

**Стратегии, приоритеты и рефлексы — НИ в одной из трёх.** Они выводятся на лету forward simulation'ом из (1) + (3), механизмом из (2).

### Практический тест textbook

*«Мог бы учитель сказать это ребёнку вслух, не раскрывая стратегию и не решая задачу за него?»*

| Утверждение | В textbook? | Почему |
|---|:-:|---|
| «Дерево даёт wood если ударить» | ✅ | факт |
| «Zombie движется в твою сторону» | ✅ | факт о поведении сущности |
| «Food > 0 → health медленно восстанавливается» | ✅ | факт о теле |
| «Food падает −0.04/шаг» | ✅ | innate rate |
| «Sword убивает zombie» | ✅ | факт о причинности |
| «Если видишь zombie без меча — беги» | ❌ | **стратегия** — эмерджит из rollout |
| «Крафти меч заранее» | ❌ | **приоритет** — эмерджит |
| «Urgency = 1/steps_until_zero» | ❌ | **формула принятия решения** — это механизм, и он живёт в коде, не в YAML |
| «Health падает быстрее еды → важнее» | ❌ | **приоритет**, агент выводит из observed rates |

**Ключ:** textbook = manual, guide с прохождением — нигде.

### Что удаляется

- **`src/snks/memory/`** целиком — `episodic_sdm.py`, `state_encoder.py`, `sdr_encoder.py`. Это был обход, он больше не нужен
- **`HOMEOSTATIC_VARS`** const и все 5 вхождений в `perception.py` (F3)
- **Strategy 1/2/Preparation** ветки в `select_goal` (F4) — заменяются на единую функцию «выбрать наиболее срочный deficit → катить forward sim для кандидатов → выбрать лучшую траекторию»
- **Textbook-ложь** про прямое `do cow restores health` (F7) — заменяется на stateful правило passive regen
- **`_STAT_GAIN_TO_NEAR`** hardcode в `perception.py:383` — эту информацию должен давать textbook, а не Python-словарь
- **Dead code** в `perception.py` (F12): `perceive_field`, `perceive`, `ground_empty_on_start`, `ground_zombie_on_damage`, `on_action_outcome`, `should_retrain`, `retrain_features`, `babble_probability`, `explore_action`
- **Вторая копия** `perceive_tile_field` в `continuous_agent.py` — одна каноничная функция в `perception.py` (F11)
- **`HomeostaticTracker.HOMEOSTATIC_VARS` iteration** — остаётся только `observed_variables()`

### Что добавляется в `configs/crafter_textbook.yaml`

Новые факты, которых текстбук не содержит, но которые учитель мог бы рассказать:

```yaml
# Движение сущностей
- "zombie moves toward player"
- "cow moves randomly"
- "skeleton shoots arrow at distance 5"       # если skeleton релевантен

# Stateful body dynamics (заменяют F7-ложь)
- "food > 0 restores health slowly"
- "food = 0 damages health"
- "drink = 0 damages health"
- "energy = 0 damages health"

# Combat cost (уточнение уже существующего zombie rule)
- "do zombie without wood_sword damages health"
```

Все остальные правила (gather, craft, combat, base body rates) — уже есть, менять не надо.

### Что добавляется в код

1. **`ConceptStore.simulate_forward(state, action, horizon)`** — рекурсивное применение правил. Возвращает траекторию с накопленным `body_delta` и произведением confidence по всем применённым правилам.

2. **`ConceptStore.score_plan(state, action_candidates, horizon)`** — для каждого кандидатного действия катит `simulate_forward`, оценивает суммарное улучшение body deficits с дисконтом по confidence. Возвращает ранжирование.

3. **Поддержка stateful rules** в парсере textbook и в `simulate_forward`: правила с условиями (`food > 0`), временными эффектами (`slowly`), entity movement.

4. **Surprise → new rule candidate** — реализация deferred question из `IDEOLOGY.md`. `record_surprise` копит паттерны; если `(near, action, actual_outcome)` повторяется N раз без matching rule — создаётся кандидатное правило с confidence=0.1. Это **единственный честный источник новизны**, помимо confidence erosion на существующих правилах.

5. **Non-decaying innate body prior** в `HomeostaticTracker`: отдельные поля `innate_rates` (из textbook, не меняются) и `observed_rates` (EMA). `get_rate()` комбинирует через weighted average, где вес наблюдения растёт с количеством samples. Чинит F6.

6. **Confidence gating** в `find_causal` и `plan`: правило с низкой confidence не отбрасывается, но в rollout'е даёт **неопределённый** исход (предсказание с вероятностью <1). Это органично порождает curiosity — rollout в область с низким confidence даёт unstable score → агент **активно** пробует неуверенные ветки. Чинит F5.

7. **Spatial map с count/confidence** (F10): `update(pos, concept)` инкрементит счётчик; `find_nearest` взвешивает по количеству подтверждений.

8. **Разделение nouns и verbs на уровне Concept** (F8): `Concept.kind: "entity" | "behavior"`. `_plan_recursive` для behavior не проверяет inventory, а проверяет семантический флаг завершения.

### Что остаётся из опыта (как задумано идеологией, но подключено)

| Что | Механизм | Стадия по идеологии |
|---|---|---|
| Визуальный grounding концептов | CNN → `ConceptStore.query_visual` / tile_segmenter → `Concept.visual` (EMA) | Мгновенное + Постепенное |
| Confidence на правилах | `verify_outcome` → `link.confidence ±0.15`; **теперь читается планировщиком** | Постепенное |
| Новые правила | Surprise accumulator → rule candidate → confidence build-up | Мгновенное (candidate) + Постепенное (stabilization) |
| Observed body rates | `HomeostaticTracker.update` EMA на `observed_rates` (отдельно от innate) | Постепенное |
| Spatial map | `CrafterSpatialMap.update` с count | Постепенное |
| CNN retrain (если нужен) | Фоновая задача | Фоновое |

### Как это решает стену 178

Новизна в системе появляется из трёх источников:

1. **Forward rollout через правила** видит далеко вперёд: rule «zombie moves toward player» + rule «zombie adjacent damages health» даёт «через 4 шага умру, если продолжу wood».
2. **Confidence-weighted rollouts** порождают curiosity: низкая уверенность в rule → rollout даёт wide distribution исходов → агент активно тестирует.
3. **Surprise → new rule** открывает gap'ы в текущей модели: что-то неожиданное → новое правило → rollout становится точнее.

Агент может заранее сравнить траектории:
- «продолжить wood» → rollout: zombie adjacent в шаге 4, −2 health/step, мёртв в шаге 8
- «отступить к stone» → rollout: zombie не догоняет, health stable

Это не on-policy имитация текущего поведения. Это честный model-based search по символьной world model. Стена 178 ломается потому, что forward rollout видит **причины смерти до того, как они наступают** — что Stage 76 report правильно диагностировал как недостающую способность, но предложил реализовать через SDM rollouts (где новизны нет) вместо ConceptStore rollouts (где она есть).

### Цена и риски

**Цена:**
- Объём работ: сравним с Course 2 из отклонённых (большой рефакторинг `perception.py` + `concept_store.py` + удаление `src/snks/memory/`).
- Surprise → rule mechanism — **отдельный нетривиальный подэтап**. Можно начать без него (только предзагруженные правила в textbook + confidence loop) и добавить позже.
- Stateful rules требуют нового парсера в `crafter_textbook.py`.

**Риски:**
- Может оказаться, что `simulate_forward` медленный на CPU (N=20 × 17 actions = 340 rollouts per step). Нужен бенчмарк.
- Rule movement («zombie moves toward player») — приближение. Реальная Crafter механика может отличаться. Surprise detector закроет gap, но не мгновенно.
- Stage 77 в текущей формулировке откладывается (фактически заменяется на этот курс).

**Что НЕ трогается:**
- Tile segmenter (Stage 75) — V1 уже в идеологии, оставляем как есть
- Crafter environment wrappers — не трогаем
- Homeostatic tracker core logic — только дополняем innate vs observed разделением
- MiniGrid regression tests — не затрагивает

### Почему это переформулировка Stage 77, а не новый этап

Stage 77 в текущей формулировке: *«replace 1-step recall with N-step rollouts over SDM»*.

Stage 77 по идеологии: *«implement forward simulation through ConceptStore causal rules, as Stage 73 top-down design always demanded. Remove EpisodicSDM substrate as a deviation.»*

Это не дополнительный этап. Это переоформленный Stage 77, который **возвращается к тому, что должно было быть сделано вместо Stage 76**.

---

## Phase B (по запросу)

Этот документ — только Phase A (диагностика + предварительный курс действий). Phase B — детальный проектный план для выбранного курса (ConceptStore forward simulation):

1. Дизайн `ConceptStore.simulate_forward` — сигнатура, dispatch правил, обработка stateful/movement rules, confidence-weighted outcome distribution
2. Grammar и парсер для новых типов правил в textbook (stateful, movement, temporal)
3. Surprise accumulator / rule discovery mechanism — threshold'ы, lifecycle кандидатного правила
4. План удаления `src/snks/memory/`, `HOMEOSTATIC_VARS`, dead code в perception.py — с учётом dependency порядка
5. Бенчмарк compute cost `simulate_forward` и решение о структуре rollout (exhaustive vs beam search)
6. Gate критерии: survival ≥200, wood ≥50%, no ideology violations в automated lint

Phase B делается после того как пользователь:

1. Прочитает этот документ
2. Подтвердит, оспорит или уточнит находки
3. Одобрит курс действий либо попросит дополнительную экспертизу по конкретному аспекту

---

## Appendix A — Grep evidence (воспроизводимость)

Основные greps, на которых базируется диагноз:

**F3 (`HOMEOSTATIC_VARS` hardcoded):**
```
$ rg -n "HOMEOSTATIC_VARS" src/
src/snks/agent/perception.py:38:HOMEOSTATIC_VARS = {"health", "food", "drink", "energy"}
src/snks/agent/perception.py:67:        v: 0.0 for v in HOMEOSTATIC_VARS
src/snks/agent/perception.py:111:        for var in HOMEOSTATIC_VARS:
src/snks/agent/perception.py:696:    for var in HOMEOSTATIC_VARS:
src/snks/agent/perception.py:788:    for var in HOMEOSTATIC_VARS:
```

**F5 (`link.confidence` dead):**
```
$ rg -n "link\.confidence|\.confidence >=|\.confidence <|confidence_threshold" src/snks
```
Живой путь: writes в `concept_store.py:295,297,339-344`. Reads — только `perception.py:630` (compute_curiosity). `find_causal`, `plan`, `select_goal` — ноль вхождений.

**F9 (`next_state_sdr` dead):**
```
$ rg -n "next_state_sdr" src/snks
src/snks/memory/episodic_sdm.py:34-41  (field definition + docstring)
src/snks/agent/continuous_agent.py:301,313  (writes only)
```
Никаких reads.

**F12 (dead functions в `perception.py`):**
```
$ rg -l "retrain_features|should_retrain|on_action_outcome|ground_zombie_on_damage|perceive_field|explore_action|babble_probability|get_drive_strengths" --type py
```
Вхождения только в: `perception.py` (определения), `tests/test_stage72.py`, `tests/test_stage73.py`, `demos/crafter_demo/agent_loop.py` (Stage 72 era), `experiments/exp130-135`, `src/snks/agent/sdm_*_agent.py` (MiniGrid). В живом пути (`continuous_agent.py`, `exp136`) — ноль вхождений.

---

## Appendix B — Scope этого ревью

**В scope:**
- `src/snks/agent/continuous_agent.py`, `concept_store.py`, `perception.py`, `crafter_textbook.py`, `crafter_spatial_map.py`
- `src/snks/memory/{state_encoder,episodic_sdm,sdr_encoder}.py`
- `src/snks/encoder/tile_segmenter.py` (упоминается, не аудитировался)
- `experiments/exp136_continuous_learning.py`
- `configs/crafter_textbook.yaml`
- `docs/IDEOLOGY.md` как эталон
- `docs/reports/stage-75-report.md`, `stage-76-report.md`

**Вне scope (упомянуто одной строкой если релевантно):**
- `src/snks/daf/`, `src/snks/sks/`, `src/snks/gws/`, `src/snks/dcam/`, `src/snks/metacog/`, `src/snks/language/` — осцилляторные компоненты, в живом пути не используются, идеология их явно deferred
- MiniGrid-агенты в `src/snks/agent/` (Stages 47-64)
- `src/snks/encoder/cnn_encoder.py`, `predictive_trainer.py`, `tile_head_trainer.py` — обучение сегментера, отдельная тема
- Все Stage 76 unit тесты — они тестируют то, что есть, но не проверяют когерентность

---

## Appendix C — Отклонённые альтернативы

Первая итерация этого документа предлагала три альтернативных курса. При повторной проверке против `IDEOLOGY.md` все три оказались несогласованными с идеологией в той или иной степени. Сохранено для документированности процесса и чтобы не предлагать их снова.

### C1 — Fix Path A, запустить Stage 77 поверх

**Идея:** удалить hardcoded strategies, замкнуть confidence loop, оставить SDM, запустить Stage 77 (forward sim через rollouts) поверх починенного Path A.

**Почему отклонено:** полумера. После C1 остаются **два world model'я** — F1 не закрыт. Path A становится когерентнее сам по себе, но интерференция с Path B (F2) сохраняется. Stage 77 поверх этого всё равно работает с двумя парадигмами принятия решений (goal-directed vs utility-maximizing).

### C2 — ConceptStore как dynamics prior для SDM

**Идея:** SDM остаётся главным decision-maker. При попадании rollout'а в unexplored state вызывается `ConceptStore.predict` как backup симулятор. Path A становится prior'ом для Path B.

**Почему отклонено:** **инвертирует иерархию идеологии.** IDEOLOGY Stage 72-73 говорит: ConceptStore — это мысль (forward sim через правила). SDM в идеологии отсутствует. C2 подчиняет символьное векторному, то есть переворачивает правильный порядок: делает ConceptStore подсказкой для SDM вместо того чтобы сделать SDM ненужным.

### C3a — Убить SDM, доработать Path A

**Идея:** удалить `src/snks/memory/`, превратить `ConceptStore.plan` в model-based planner с forward simulation.

**Почему не отклонено, а переформулировано:** это **правильное направление**, но в первой версии документа я описал его цену как «теряется половина IDEOLOGY» — это была ошибка. SDM в идеологии нет, её удаление не теряет ничего из идеологии, а наоборот приводит код к идеологии. После уточнения таксономии (3 категории знания) этот курс стал основным — см. «Course of action — Forward simulation через ConceptStore» выше.

### C3b — Убить ConceptStore, SDM как единственный world model

**Идея:** удалить textbook и causal_links. SDM становится единственной моделью мира. Exploration через честный off-policy RL.

**Почему отклонено:** **прямое нарушение IDEOLOGY Тезиса 1** («Модульная bio-inspired архитектура ... каузальное планирование через subgoals, НЕ reward shaping»). Убирает top-down символьный планировщик, который идеология считает центральным. Отказ от textbook противоречит Stage 73 («textbook — справочник, как родитель объясняет ребёнку»).
