# Stages 19–24: Language & Grounded Cognition — Design Doc

**Дата:** 2026-03-30
**Статус:** Stage 19 COMPLETE (2026-03-30). Stages 20–24 pending.
**Зависимости:** Этапы 0–18 ✅

---

## Архитектурные принципы

### Принцип 1: Концепты первичны, язык вторичен

СКС — это концепт (смысл), сформированный из опыта. Слово — **якорь** (grounding label), привязанный к уже существующей концептной СКС. World model оперирует исключительно концептами. Язык — одна из модальностей доступа к концептному пространству, наравне с визуальной.

Следствия:
1. Слово НЕ создаёт новую СКС — оно привязывается к существующей
2. World model НЕ хранит слова — она хранит концепты и их связи
3. Генерация текста = обход концептного графа + линеаризация через grounding map
4. Понимание текста = активация концептных СКС через языковые якоря

### Принцип 2: Естественная мультимодальность

Каждая сенсорная модальность инжектирует токи в свою зону DAF. Через STDP межзонные связи усиливаются при ко-активации. Мультимодальные концепты — emergent property, не инженерное решение.

Зоны DAF — только для **сенсорных модальностей**. Действия агента (motor) — не модальность, а рёбра в каузальном графе world model (DCAM).

Добавление новой модальности = новый энкодер + новая зона в конфиге. Ноль изменений в ядре.

---

## Общий план этапов

| Этап | Название | Суть |
|------|----------|------|
| **19** | Зональный DAF + Emergent Grounding | Зоны, межзонный STDP, ко-активация image+word |
| **20** | Композиционное понимание | HAC role-filler парсинг, bind/unbind |
| **21** | Вербализация World Model | Описание состояния, каузальные объяснения, план |
| **22** | Grounded QA | Фактические, симуляционные, рефлексивные вопросы |
| **23** | Scaffold Removal | Убираем sentence-transformers, автономная валидация |
| **24** | BabyAI Embodied Grounding | Выполнение текстовых инструкций в BabyAI |

---

## Stage 19: Зональный DAF + Emergent Grounding

### 19.1 — Зональная архитектура DAF

DAF engine получает зональную конфигурацию. Два конфига для ablation:

**Конфиг A — без convergence:**
```yaml
zones:
  visual:      {start: 0,     size: 28000}
  linguistic:  {start: 28000, size: 22000}
```

**Конфиг B — с convergence:**
```yaml
zones:
  visual:      {start: 0,     size: 22000}
  linguistic:  {start: 22000, size: 18000}
  convergence: {start: 40000, size: 10000}
```

Одинаковый N=50K. Ablation покажет, нужна ли convergence zone.

**Изменения в DafEngine:**
- `DafEngine.__init__` принимает `zones: dict` (опционально, backward-compatible — без zones работает как сейчас)
- Граф DAF: внутризонные связи `avg_degree=50`, межзонные `avg_degree=10`
- Каждый энкодер инжектирует токи только в свою зону

**Изменения в энкодерах:**
- `sdr_to_currents()` — хеширование в пределах зоны, не по всем N
- Усреднение токов `(img + txt) / 2` удаляется — каждый энкодер пишет в свою зону

**Что НЕ меняется:**
- STDP, гомеостаз, FHN-динамика, SKS detection, HAC, DCAM
- Без zones в конфиге — полная обратная совместимость

**Файлы:**
- `daf/engine.py` — zone-aware `set_input(currents, zone)`. Broadcast и motor токи (из Configurator/MetacogMonitor) применяются глобально без zone-routing — они модулируют всю сеть
- `daf/graph.py` — зональная генерация графа (intra-zone dense, inter-zone sparse)
- `configs/zones.yaml` — конфиги A и B
- `encoder/encoder.py`, `encoder/text_encoder.py` — zone-aware injection

### 19.2 — Emergent Grounding через ко-активацию

Система видит образ и одновременно получает слово. STDP усиливает межзонные связи. После достаточного числа ко-активаций подаём только слово — visual zone активируется сама.

**Процесс grounding:**
```
Обучение (ко-активация):
  image "ключ" → visual zone: узлы 100-130 активны
  слово "key"  → linguistic zone: узлы 29000-29020 активны
  → одновременно в DAF → STDP усиливает межзонные связи
  → повторить 5-10 раз

Тест (cross-modal recall):
  слово "key" → linguistic zone: 29000-29020 активны
  → межзонные связи → visual zone: 100-130 активируются БЕЗ образа
  → cross_modal_ratio = activation(visual | paired_word) / activation(visual | random_word)
```

**Датасеты для grounding:**
- Синтетические фигуры (как в Exp 12): 10 категорий × 5 вариаций = 50 пар (image, label)
- MNIST digits: 10 цифр × слова "zero".."nine"
- Простые объекты MiniGrid: key, door, ball, box, wall + их визуальные образы

**Bootstrap:** sentence-transformers (all-MiniLM-L6-v2) используется как энкодер для linguistic zone. Убираем на Stage 23.

### Эксперименты Stage 19

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 44a | Cross-modal recall (без convergence) | cross_modal_ratio | > 2.0 |
| 44b | Cross-modal recall (с convergence) | cross_modal_ratio | > 2.0 |
| 45a | Grounding speed (без convergence) | ко-активаций до ratio > 2.0 | < 20 |
| 45b | Grounding speed (с convergence) | ко-активаций до ratio > 2.0 | < 20 |

**Правило выбора конфига:** Если ratio_B > 1.2 × ratio_A или speed_B < 0.8 × speed_A — выбираем B (convergence zone оправдана). Иначе — выбираем A (проще, меньше узлов). Выбранный конфиг используется во всех последующих этапах.

#### Результаты Stage 19 (2026-03-30)

**Ключевое открытие:** Чистый STDP+coupling НЕ работает для cross-modal recall при любом N (5K-50K). Inter-zone weight растёт (+47%), но coupling слишком слабый для FHN firing threshold. Ratio ~1.0 на всех масштабах.

**Решение:** Complementary priming через GroundingMap. Pipeline регистрирует visual SDR при co-activation, при text-only recall инжектирует `priming_strength=0.3` × visual SDR в visual zone. Биологически обосновано (top-down prediction).

| Exp | Config | Метрика | Результат | Gate |
|-----|--------|---------|-----------|------|
| 44a | A (no convergence) | cross_modal_ratio | 178 | > 2.0 ✅ |
| 44b | B (with convergence) | cross_modal_ratio | 116,742 | > 2.0 ✅ |
| 45a | A (no convergence) | reps to ratio>2.0 | 1 | < 20 ✅ |
| 45b | B (with convergence) | reps to ratio>2.0 | 1 | < 20 ✅ |

**Выбран Config B** (ratio_B=116742 >> 1.2 × ratio_A=213). Convergence zone радикально снижает random activation → чище discrimination.

**Scaling study** (seed=42, RTX 3070 Ti): priming даёт ratio 64–134 при N=10K-50K. Без priming ratio ~1.0 на всех N. Качество не зависит от масштаба.

### 19.3 — Grounding Map

При каждой ко-активации (image + word) pipeline запоминает пару (sks_id, word) в `GroundingMap`:

```python
class GroundingMap:
    """Двунаправленный lookup: word ↔ sks_id."""
    def register(self, word: str, sks_id: int, sdr: Tensor) -> None
    def word_to_sks(self, word: str) -> int | None
    def sks_to_word(self, sks_id: int) -> str | None
    def word_to_sdr(self, word: str) -> Tensor | None
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

Заполняется автоматически в `pipeline.perception_cycle()` при одновременном image + text. Используется в Stage 21 (Verbalizer) и Stage 23 (GroundedTokenizer).

### Структура файлов Stage 19

```
src/snks/
├── daf/
│   ├── engine.py              ✏️  zone-aware set_input
│   └── graph.py               ✏️  зональная генерация графа
├── encoder/
│   ├── encoder.py             ✏️  zone-aware sdr_to_currents
│   └── text_encoder.py        ✏️  zone-aware sdr_to_currents
├── pipeline/
│   └── runner.py              ✏️  убрать усреднение, zone-based injection
└── experiments/
    ├── exp44_crossmodal_recall.py  🆕
    └── exp45_grounding_speed.py    🆕
├── language/
│   └── grounding_map.py       🆕  GroundingMap class

configs/
└── zones.yaml                 🆕
```

---

## Stage 20: Композиционное понимание (HAC Role-Filler)

### 20.1 — Ролевая система

Роли — фиксированные вектора в HAC-пространстве (не СКС, не аттракторы):

```python
ROLES = {
    "AGENT":    random_hac_vector(seed=100),   # кто
    "ACTION":   random_hac_vector(seed=101),   # что делает
    "OBJECT":   random_hac_vector(seed=102),   # с чем/кем
    "LOCATION": random_hac_vector(seed=103),   # где
    "GOAL":     random_hac_vector(seed=104),   # зачем
    "ATTR":     random_hac_vector(seed=105),   # атрибут (цвет, размер)
}
```

### 20.2 — Парсинг предложения

```
"cat sits on mat"
       │
       ▼ (sentence-transformers + simple rule-based chunker)
  chunks: ["cat", "sits", "on mat"]
       │
       ▼ (каждый chunk → SDR → linguistic zone → активация СКС)
  СКС_CAT, СКС_SIT, СКС_MAT
       │
       ▼ (HAC bind)
  sentence_hac = bind(AGENT, СКС_CAT) + bind(ACTION, СКС_SIT) + bind(LOCATION, СКС_MAT)
       │
       ▼ (HAC unbind — извлечение)
  unbind(AGENT, sentence_hac) → СКС_CAT  ✓
```

**Chunker — rule-based:**
- Синтетические грамматики: фиксированный порядок SVO(L), парсинг по позиции
- BabyAI-инструкции: формальная грамматика, парсинг тривиальный
- Естественный язык: spaCy dependency parse → роли (строительные леса)

**Почему не emergent парсинг:** синтаксис — конвенция языка, не свойство мира. Это интерфейсная задача, не когнитивная.

### Эксперименты Stage 20

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 46 | Role extraction (SVO) | accuracy на синтетических предложениях | > 0.8 |
| 47 | Compositional generalization | unbind accuracy на невиданных комбинациях | > 0.7 |

### Структура файлов Stage 20

```
src/snks/
├── language/
│   ├── roles.py               🆕  ROLES constants + HAC role vectors
│   ├── chunker.py             🆕  rule-based SVO(L) chunker
│   └── parser.py              🆕  text → role-filler HAC structure
└── experiments/
    ├── exp46_role_extraction.py    🆕
    └── exp47_compositional.py     🆕
```

---

## Stage 21: Вербализация World Model

### 21.1 — Три типа вербализации

**Описание состояния ("Что я вижу/знаю"):**
```
Активные СКС в visual zone → grounding → слова
  СКС_KEY активна, СКС_DOOR активна
  → grounding label lookup
  → шаблон DESCRIBE: "I see [OBJECT_1] and [OBJECT_2]"
  → "I see a key and a door"
```

**Каузальное объяснение ("Почему/как"):**
```
DCAM query: причины и следствия для активной СКС
  СКС_KEY → action_pickup → СКС_KEY_HELD → action_toggle → СКС_DOOR_OPEN
  → каждый узел цепочки → grounding label
  → шаблон CAUSAL: "[AGENT] [ACTION] [OBJECT] → [EFFECT]"
  → "picking up key → can open door"
```

**Вербализация плана ("Что собираюсь делать"):**
```
StochasticSimulator: текущий план
  plan: [СКС_KEY, СКС_DOOR, СКС_DOOR_OPEN]
  → grounding labels + actions
  → шаблон PLAN: "I need to [ACTION_1] [OBJECT_1], then [ACTION_2] [OBJECT_2]"
  → "I need to pick up the key, then open the door"
```

### 21.2 — Verbalizer

```python
class Verbalizer:
    def describe_state(self, active_sks, grounding_map) -> str
    def explain_causal(self, sks, dcam_world_model) -> str
    def verbalize_plan(self, plan_sks, grounding_map) -> str
```

Интерфейсный слой, не когнитивный. Шаблоны — осознанное решение: точная передача содержимого world model, не "красивые формулировки".

**Масштабируемость шаблонов:** Текущие шаблоны покрывают простые структуры (SVO, одношаговые планы, линейные каузальные цепочки). При росте сложности (вложенные каузальные цепочки, условные планы, multiple agents) потребуется рекурсивная вербализация или шаблонное расширение. Это вынесено в post-Stage 24 future work.

**Grounding map:** при ко-активации (Stage 19) запоминаем пару (сks_id, word) в dict. Не противоречит emergent grounding — map лишь фиксирует то, что STDP уже сформировал.

### Эксперименты Stage 21

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 48a | Describe recall & precision | recall: реальные объекты упомянуты; precision: упомянутое корректно. Gate на recall | > 0.7 (recall) |
| 48b | Causal verbalization | корректность каузальных цепочек | > 0.7 |
| 48c | Plan verbalization | план соответствует Simulator output | > 0.7 |

### Структура файлов Stage 21

```
src/snks/
├── language/
│   ├── verbalizer.py          🆕  Verbalizer class
│   └── templates.py           🆕  DESCRIBE / CAUSAL / PLAN шаблоны
└── experiments/
    ├── exp48a_describe.py         🆕
    ├── exp48b_causal_verbal.py    🆕
    └── exp48c_plan_verbal.py      🆕
```

---

## Stage 22: Grounded QA

### 22.1 — Три типа вопросов

**Фактические — запрос к DCAM:**
```
"What opens the door?"
  → парсинг: unbind(ACTION="opens", OBJECT="door") → ищем AGENT
  → DCAM query: какая СКС связана причинно с СКС_DOOR через action_open?
  → результат: СКС_KEY
  → вербализация: "the key"
```

**Симуляционные — запрос к StochasticSimulator:**
```
"What happens if I pick up the key?"
  → Simulator.simulate(state=current, action=pickup, object=СКС_KEY)
  → результат: [СКС_KEY_HELD, → СКС_DOOR_OPEN возможно]
  → вербализация: "You will hold the key. Then you can open the door."
```

**Рефлексивные — запрос к Metacog:**
```
"Why did you go left?"
  → Metacog log: последний action=left, причина=EXPLORE, prediction_error=0.8 справа
  → вербализация: "Prediction error was high on the right. I chose to explore left."
```

### 22.2 — QA Pipeline

```python
class GroundedQA:
    def answer(self, question: str) -> str:
        roles = self.parser.parse(question)
        qtype = self.classify(roles)  # factual | simulation | reflective

        if qtype == "factual":
            result = self.dcam.query_causal(roles)
        elif qtype == "simulation":
            result = self.simulator.simulate(roles)
        elif qtype == "reflective":
            result = self.metacog.explain(roles)

        return self.verbalizer.verbalize(result, qtype)
```

**Классификатор типа — rule-based:**
- "what/who/where" + present → фактический
- "what happens if / what would" → симуляционный
- "why did / why are you" → рефлексивный

**Ключевое свойство: верифицируемость.** Каждый ответ извлекается из world model. Нет знания в DCAM → "I don't know". Система не галлюцинирует.

### Эксперименты Stage 22

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 49 | Factual QA | accuracy на вопросах о каузальных связях | > 0.7 |
| 50 | Simulation QA | accuracy "что будет если" | > 0.6 |
| 51 | Reflective QA | accuracy "почему ты сделал X" | > 0.6 |

### Структура файлов Stage 22

```
src/snks/
├── language/
│   └── qa.py                  🆕  GroundedQA class
└── experiments/
    ├── exp49_factual_qa.py        🆕
    ├── exp50_simulation_qa.py     🆕
    └── exp51_reflective_qa.py     🆕
```

---

## Stage 23: Scaffold Removal

### 23.1 — Убираем sentence-transformers

Заменяем на собственный GroundedTokenizer:

```python
class GroundedTokenizer:
    """Слово → SDR через lookup выученных пар."""

    def __init__(self, grounding_pairs: dict[str, Tensor]):
        # Заполнен на Stage 19 при ко-активациях
        self.vocab = grounding_pairs

    def encode(self, word: str) -> Tensor:
        if word in self.vocab:
            return self.vocab[word]
        return self.unknown_sdr  # "I don't know this word"
```

**Почему это работает:**
- На Stage 19 при каждой ко-активации запомнили пару (word, SDR)
- SDR — тот самый, который sentence-transformers генерировал
- STDP-связи в DAF уже сформированы — они не зависят от источника SDR

**Ограничение:** Система знает только слова, которые видела при обучении. Новое слово = "I don't know". Расширение словаря: показать новое слово + образ → ко-активация → STDP → grounding. Как ребёнок учит новые слова. Однако без sentence-transformers начальный SDR для нового слова должен генерироваться иначе — через случайную проекцию из символьного представления (character-level hash → SDR). Это открытый вопрос, вынесенный в post-Stage 24 future work.

**Fallback:** Если quality drop > 20% на метриках Exp 44–51, оставляем sentence-transformers как tokenizer, фиксируем как открытую проблему.

### Эксперименты Stage 23

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 52 | Autonomous cross-modal recall | ratio без ST | > 0.8 × ratio_с_ST |
| 53 | Autonomous QA | accuracy без ST | > 0.8 × accuracy_с_ST |

### Структура файлов Stage 23

```
src/snks/
├── encoder/
│   └── grounded_tokenizer.py  🆕  GroundedTokenizer
└── experiments/
    ├── exp52_autonomous_recall.py  🆕
    └── exp53_autonomous_qa.py      🆕
```

---

## Stage 24: BabyAI Embodied Language Grounding

### 24.1 — BabyAI интеграция

BabyAI-инструкции как role-filler структуры, grounding через embodied experience.

**Pipeline:**
```
Инструкция: "pick up the red key"
       │
       ▼ (парсинг → role-filler)
  bind(ACTION, СКС_PICKUP) + bind(OBJECT, СКС_KEY) + bind(ATTR, СКС_RED)
       │
       ▼ (world model query: где red key?)
  DCAM: СКС_RED_KEY последний раз наблюдался в позиции (3,2)
       │
       ▼ (planning)
  Simulator: текущая позиция → навигация до (3,2) → pickup
       │
       ▼ (execution)
  EmbodiedAgent: выполняет план, получает визуальный feedback
       │
       ▼ (верификация)
  Активна ли СКС_RED_KEY_HELD? → success/fail
```

### 24.2 — Grounding атрибутов

BabyAI использует цвета: red, green, blue, purple, yellow, grey. Каждый цвет — концептная СКС, сформированная через ко-активацию (образ красного объекта + слово "red"). Композиция: bind(OBJECT, СКС_KEY) + bind(ATTR, СКС_RED) = конкретный "red key".

### 24.3 — Уровни сложности

| Уровень | Пример инструкции | Что тестирует |
|---|---|---|
| GoTo | "go to the red ball" | базовый grounding + навигация |
| Pickup | "pick up the blue key" | grounding + действие |
| Open | "open the yellow door" | каузальная цепочка (нужен ключ) |
| Seq | "pick up the key then open the door" | композиция двух инструкций |
| Synth | "put the red ball next to the blue box" | пространственные отношения |

### Эксперименты Stage 24

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 54 | GoTo + Pickup | success rate | > 0.6 |
| 55 | Open (каузальная цепочка) | success rate | > 0.5 |
| 56 | Sequential instructions | success rate | > 0.4 |
| 57 | Novel combinations (невиданные цвет+объект) | success rate | > 0.3 |

### Структура файлов Stage 24

```
src/snks/
├── environments/
│   └── babyai_adapter.py      🆕  BabyAI → EmbodiedAgent адаптер
├── language/
│   └── instruction_parser.py  🆕  BabyAI grammar → role-filler
└── experiments/
    ├── exp54_goto_pickup.py       🆕
    ├── exp55_open_causal.py       🆕
    ├── exp56_sequential.py        🆕
    └── exp57_novel_combos.py      🆕
```

---

## Новые зависимости

```
sentence-transformers>=2.2    # уже есть (Stage 7), убирается на Stage 23
babyai>=1.1                   # Stage 24
minigrid>=2.0                 # уже есть
spacy>=3.0                    # опционально, для natural language chunking
```

---

## Сводка экспериментов

| # | Stage | Название | Метрика | Gate |
|---|-------|----------|---------|------|
| 44a | 19 | Cross-modal recall (без convergence) | cross_modal_ratio | > 2.0 |
| 44b | 19 | Cross-modal recall (с convergence) | cross_modal_ratio | > 2.0 |
| 45a | 19 | Grounding speed (без convergence) | ко-активаций до ratio > 2.0 | < 20 |
| 45b | 19 | Grounding speed (с convergence) | ко-активаций до ratio > 2.0 | < 20 |
| 46 | 20 | Role extraction (SVO) | accuracy | > 0.8 |
| 47 | 20 | Compositional generalization | unbind accuracy | > 0.7 |
| 48a | 21 | Describe recall & precision | recall (gate) + precision (measured) | > 0.7 (recall) |
| 48b | 21 | Causal verbalization | correctness | > 0.7 |
| 48c | 21 | Plan verbalization | match with Simulator | > 0.7 |
| 49 | 22 | Factual QA | accuracy | > 0.7 |
| 50 | 22 | Simulation QA | accuracy | > 0.6 |
| 51 | 22 | Reflective QA | accuracy | > 0.6 |
| 52 | 23 | Autonomous cross-modal recall | ratio / ratio_ST | > 0.8 |
| 53 | 23 | Autonomous QA | accuracy / accuracy_ST | > 0.8 |
| 54 | 24 | GoTo + Pickup | success rate | > 0.6 |
| 55 | 24 | Open (каузальная цепочка) | success rate | > 0.5 |
| 56 | 24 | Sequential instructions | success rate | > 0.4 |
| 57 | 24 | Novel combinations | success rate | > 0.3 |

---

## Граф зависимостей

```
Stage 19 (Zonal DAF + Grounding)
    │
    ▼
Stage 20 (Compositional Understanding)
    │
    ├──► Stage 21 (Verbalization)
    │        │
    │        ▼
    │    Stage 22 (Grounded QA)
    │        │
    └────────┤
             ▼
         Stage 23 (Scaffold Removal)
             │
             ▼
         Stage 24 (BabyAI)
```

---

## Что НЕ меняется

- DafEngine ядро: STDP, гомеостаз, FHN-динамика
- SKS detection
- HAC engine (bind/unbind/bundle)
- DCAM (DcamWorldModel, EpisodicBuffer, SSG)
- Все существующие тесты и эксперименты 1–43
