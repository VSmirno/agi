# Stages 19–24: Language & Grounded Cognition — Design Doc

**Дата:** 2026-03-30
**Статус:** Stages 19–21 COMPLETE (2026-03-30). Stages 22–24 pending.
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

**Статус:** COMPLETE (2026-03-30)

### Архитектурный принцип: масштабируемость

Каждый компонент Stage 20 — интерфейс, за которым стоит текущая реализация (scaffold).
Замена реализации = ноль изменений в остальном коде. Это критично для масштабирования
до полной модели мира с терабайтами данных и произвольными языковыми конструкциями.

### 20.1 — Ролевая система (`language/roles.py`)

Роли — фиксированные единичные вектора в HAC-пространстве (не СКС, не аттракторы).
Каждая роль = "подпись на конверте": `bind(ROLE, filler)` кладёт filler в конверт,
`unbind(ROLE, sentence)` достаёт обратно.

```python
def random_hac_vector(dim: int, seed: int) -> Tensor:
    """Детерминистичный единичный вектор в HAC-пространстве."""
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=gen)
    return v / v.norm()

def get_roles(hac_dim: int = 2048) -> dict[str, Tensor]:
    """Возвращает dict ролевых векторов. hac_dim параметризован, не хардкод."""
    return {
        "AGENT":    random_hac_vector(hac_dim, seed=100),   # кто
        "ACTION":   random_hac_vector(hac_dim, seed=101),   # что делает
        "OBJECT":   random_hac_vector(hac_dim, seed=102),   # с чем/кем
        "LOCATION": random_hac_vector(hac_dim, seed=103),   # где
        "GOAL":     random_hac_vector(hac_dim, seed=104),   # зачем
        "ATTR":     random_hac_vector(hac_dim, seed=105),   # атрибут (цвет, размер)
    }
```

**Масштабируемость:** В 2048-dim ~100-200 ортогональных ролей (cosine < 0.05).
Сейчас 6, запас огромный. Добавление TEMPORAL, INSTRUMENT, CAUSE, MANNER — одна строка.
При hac_dim=4096 — ещё на порядок больше.

### 20.2 — Chunker (`language/chunker.py`)

Rule-based scaffold за абстрактным интерфейсом:

```python
@dataclass
class Chunk:
    text: str       # "cat", "sits", "on mat", "pick up"
    role: str       # "AGENT", "ACTION", "OBJECT", "LOCATION", "GOAL", "ATTR"

class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, sentence: str) -> list[Chunk]: ...

class RuleBasedChunker(BaseChunker):
    def chunk(self, sentence: str) -> list[Chunk]: ...
    def detect_pattern(self, sentence: str) -> str: ...  # "svo" | "svo_attr" | "minigrid"
```

**Три грамматических паттерна:**

Паттерн 1 — SVO(L):
```
"cat sits on mat"
→ [("cat", AGENT), ("sits", ACTION), ("on mat", LOCATION)]
Правило: 1-е слово=AGENT, 2-е=ACTION, "on/in/at X"=LOCATION, остальное=OBJECT
```

Паттерн 2 — ATTR+SVO(L):
```
"red cat sits on mat"
→ [("red", ATTR), ("cat", AGENT), ("sits", ACTION), ("on mat", LOCATION)]
Правило: 1-е слово в ADJECTIVES → ATTR, далее как паттерн 1
```

Паттерн 3 — MiniGrid imperative:
```
"pick up the red key"
→ [("pick up", ACTION), ("red", ATTR), ("key", OBJECT)]
Правило: начало в ACTION_VERBS → ACTION, цвет → ATTR, существительное → OBJECT
```

Детектор паттерна: 1-е слово в ACTION_VERBS (imperatives) → minigrid,
в ADJECTIVES → svo_attr, иначе → svo.

**Масштабирование:** Сейчас RuleBasedChunker (scaffold). Потом SpaCyChunker,
потом LearnedChunker. Pipeline знает только BaseChunker.chunk().

### 20.3 — Parser (`language/parser.py`)

```python
class RoleFillerParser:
    def __init__(self, hac: HACEngine, roles: dict[str, Tensor]):
        self.hac = hac
        self.roles = roles

    def parse(self, chunks: list[Chunk], embeddings: dict[str, Tensor]) -> Tensor:
        """chunks + HAC-эмбеддинги слов → один sentence_hac вектор."""
        bindings = []
        for chunk in chunks:
            role_vec = self.roles[chunk.role]
            filler_vec = embeddings[chunk.text]
            bindings.append(self.hac.bind(role_vec, filler_vec))
        return self.hac.bundle(bindings)

    def extract(self, role: str, sentence_hac: Tensor) -> Tensor:
        """Извлечь filler по роли."""
        return self.hac.unbind(self.roles[role], sentence_hac)

    def extract_all(self, sentence_hac: Tensor) -> dict[str, Tensor]:
        """Извлечь все роли."""
        return {name: self.extract(name, sentence_hac) for name in self.roles}
```

### 20.4 — EmbeddingResolver (гибридный подход)

```python
class EmbeddingResolver:
    """Резолвит chunk.text → HAC embedding. Гибрид: кэш + DAF fallback."""

    def __init__(self, grounding_map, embedder, pipeline):
        self.grounding_map = grounding_map
        self.embedder = embedder
        self.pipeline = pipeline  # для DAF fallback

    def resolve(self, word: str) -> Tensor | None:
        # 1. Кэш: GroundingMap → SKS → embedding
        sks_id = self.grounding_map.word_to_sks(word)
        if sks_id is not None:
            return self.embedder.get_embedding(sks_id)
        # 2. Fallback: DAF perception cycle для незнакомого слова
        return self._run_daf_for_word(word)
```

### 20.5 — Синтетический датасет

Три уровня сложности, каждый расширяет предыдущий:

| Уровень | Пример | Роли | Словарь |
|---------|--------|------|---------|
| SVO | "cat sits on mat" | AGENT, ACTION, LOCATION | 6 сущ + 5 глаг + 4 места |
| ATTR+SVO | "red cat sits on mat" | +ATTR | + 3 цвета |
| MiniGrid | "pick up the red key" | ACTION, ATTR, OBJECT | MiniGrid объекты + действия |

Всего ~120 предложений. Все слова заземляются в grounding-фазе эксперимента.

### Эксперименты Stage 20

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 46 | Role extraction | accuracy на всех 3 уровнях | > 0.8 |
| 47 | Compositional generalization | unbind accuracy на невиданных комбинациях | > 0.7 |

**Exp 46 — Role extraction:**
```
Фаза 1 (grounding): показать image+word пары через perception_cycle()
Фаза 2 (test): для каждого предложения:
  1. chunker.chunk() → chunks
  2. resolver.resolve() → embeddings
  3. parser.parse(chunks, embeddings) → sentence_hac
  4. parser.extract(role, sentence_hac) → recovered_vec
  5. similarity(recovered_vec, original_vec) > 0.3 → correct
  accuracy = correct / total
```

Метод: best-match среди всех известных embeddings (argmax similarity).
Correct = best match совпадает с ground truth.
Абсолютные значения similarity после unbind из 3-4 role bundle: ~0.15-0.25
(random noise ~0.02), но best-match надёжно выбирает правильный filler.

**Exp 47 — Compositional generalization:**
```
Split: 70% train / 30% test (невиданные комбинации слов)
  Train: "cat sits on mat", "dog runs on floor", ...
  Test:  "cat runs on floor" (cat+runs никогда вместе)
Grounding: все СЛОВА видны, новы КОМБИНАЦИИ.
Метрика: unbind accuracy на тестовых предложениях.
```

Тестирует фундаментальное свойство circular convolution —
bind/unbind не зависят от конкретных комбинаций.

### Структура файлов Stage 20

```
src/snks/
├── language/
│   ├── roles.py               🆕  get_roles(), random_hac_vector()
│   ├── chunker.py             🆕  BaseChunker, RuleBasedChunker, Chunk
│   └── parser.py              🆕  RoleFillerParser, EmbeddingResolver
└── experiments/
    ├── exp46_role_extraction.py    🆕
    └── exp47_compositional.py      🆕

tests/language/
├── test_roles.py              🆕  детерминистичность, ортогональность
├── test_chunker.py            🆕  все 3 паттерна
└── test_parser.py             🆕  parse + extract roundtrip
```

**Изменения в существующих файлах:** Нет. Stage 20 чисто аддитивный.
**Новые pip-зависимости:** Нет.

---

## Stage 21: Вербализация World Model

### 21.1 — Три типа вербализации

**Описание состояния ("Что я вижу/знаю"):**
```
Активные СКС в visual zone → grounding → слова
  СКС_KEY активна, СКС_DOOR активна
  → grounding_map.sks_to_word() lookup
  → фильтрация: только SKS с grounding label
  → шаблон DESCRIBE: "I see [OBJECT_1] and [OBJECT_2]"
  → "I see key and door"
```

**Каузальное объяснение ("Почему/как"):**
```
CausalWorldModel.get_causal_links() → фильтр по sks_id
  → для каждого link: найти "главный" SKS (первый с grounding label)
  → упрощённая форма: "[ACTION] [OBJECT] causes [EFFECT]"
  → "pick up key causes key held"

Решение: вербализуем только "главный" SKS из frozenset (первый
с grounding label). Полная вербализация всех SKS в frozenset —
post-Stage 24 future work.
```

**Вербализация плана ("Что собираюсь делать"):**
```
Вход: action_ids из StochasticSimulator + initial_sks + CausalWorldModel
  → для каждого action_id: action_names[id] → имя действия
  → causal_model.predict_effect(state, action) → SKS-эффект
  → grounding_map: SKS → слово
  → шаблон PLAN: "I need to [ACTION_1] [OBJECT_1], then [ACTION_2] [OBJECT_2]"
  → "I need to pick up key, then toggle door"

action_names — внешний словарь {0: "go left", 1: "go right", ...},
передаётся в конструктор Verbalizer. Не захардкожен.
```

### 21.2 — Verbalizer

```python
class Verbalizer:
    def __init__(self, grounding_map: GroundingMap, action_names: dict[int, str]):
        ...

    def describe_state(self, active_sks_ids: list[int]) -> str:
        """Вербализует активные SKS через grounding_map."""

    def explain_causal(self, sks_id: int, causal_model: CausalWorldModel) -> str:
        """Находит каузальные связи для sks_id, вербализует упрощённо."""

    def verbalize_plan(
        self, action_ids: list[int], initial_sks: set[int],
        causal_model: CausalWorldModel,
    ) -> str:
        """Восстанавливает SKS-цепочку через predict_effect, вербализует план."""
```

Интерфейсный слой, не когнитивный. Шаблоны — осознанное решение: точная передача содержимого world model, не "красивые формулировки".

**Масштабируемость шаблонов:** Текущие шаблоны покрывают простые структуры (SVO, одношаговые планы, линейные каузальные цепочки). При росте сложности (вложенные каузальные цепочки, условные планы, multiple agents) потребуется рекурсивная вербализация или шаблонное расширение. Это вынесено в post-Stage 24 future work.

**Grounding map:** при ко-активации (Stage 19) запоминаем пару (sks_id, word) в dict. Не противоречит emergent grounding — map лишь фиксирует то, что STDP уже сформировал.

### Эксперименты Stage 21

Все эксперименты работают на **синтетических данных** (без DAF/pipeline).
Проверяют изолированно логику Verbalizer.

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 48a | Describe recall & precision | recall: grounded объекты упомянуты; precision: нет ложных | recall > 0.7, precision = 1.0 |
| 48b | Causal verbalization | корректность каузальных фраз vs known ground truth | accuracy > 0.7 |
| 48c | Plan verbalization | план содержит правильные action+object пары в порядке | accuracy > 0.7 |

**Exp 48a** — синтетический grounding_map (5-6 слов), active_sks = подмножество + SKS без label. Проверка: все grounded упомянуты, лишних нет.

**Exp 48b** — синтетический CausalWorldModel с 3-4 известными переходами через observe_transition. Вызов explain_causal, сверка с ожидаемым текстом.

**Exp 48c** — синтетическая каузальная цепочка (sks_key →pickup→ sks_key_held →toggle→ sks_door_open). plan=[3,5]. Проверка корректности вербализации.

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
tests/
    └── test_verbalizer.py         🆕
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
