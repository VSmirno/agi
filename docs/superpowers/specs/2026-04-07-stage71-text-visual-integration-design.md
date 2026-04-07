# Stage 71: Text-Visual Integration

**Дата:** 2026-04-07
**Статус:** DESIGN
**Предыдущий:** Stage 70 (ScenarioCurriculum) COMPLETE

---

## 1. Цель

Соединить текстовый и визуальный пайплайны СНКС в единую архитектуру на домене Crafter. Текст = канал обучения каузальных правил (родитель учит ребёнка). Vision = перцептивное заземление. Агент получает знания из "учебника", заземляет их визуально, верифицирует через опыт.

### Принципы

- **Концепты первичны, язык вторичен** — слово привязывается к уже существующей перцептивной СКС
- **Обучение через демонстрации** — каузальные правила приходят из текста, не открываются exploration
- **Единое хранилище** — одна сущность = один концепт со всеми модальностями (омнимодальность)
- **Верификация через опыт** — правила из учебника верифицируются prediction error, не принимаются на веру

### Что это даёт глобально

Текстовая модальность — не подпорка для Crafter, а архитектурный компонент world model:
- В Crafter: учебник = 15 правил
- В реальном мире: учебник = Wikipedia, инструкции, учебники
- Архитектура одна: текст → каузальные правила → visual grounding

---

## 2. ConceptStore — единое хранилище концептов

Заменяет GroundingMap + CausalWorldModel. Один концепт = одна запись со всеми модальностями.

### Структуры данных

```python
@dataclass
class CausalLink:
    action: str                    # "do", "make", "place"
    result: str                    # concept_id результата ("wood")
    requires: dict[str, int]       # {wood_pickaxe: 1}
    condition: str | None          # "nearby"
    confidence: float              # 0.5 (из текста) → 1.0 (верифицировано)

@dataclass
class Concept:
    id: str                        # "tree"
    visual: torch.Tensor | None    # z_real (2048) от CNNEncoder
    text_sdr: torch.Tensor | None  # SDR от GroundedTokenizer
    attributes: dict[str, Any]     # {category: "resource", dangerous: False}
    causal_links: list[CausalLink] # do → gives(wood)
    confidence: float              # общая confidence концепта

    def find_causal(self, action: str,
                    check_requires: dict | None = None) -> CausalLink | None:
        """Найти каузальную связь по action, опционально проверив requires"""
```

### Интерфейс ConceptStore

```python
class ConceptStore:
    concepts: dict[str, Concept]
    
    # Регистрация
    def register(self, id: str, attributes: dict) -> Concept
    def add_causal(self, concept_id: str, link: CausalLink) -> None
    
    # Grounding (заполнение модальностей)
    def ground_visual(self, id: str, z_real: torch.Tensor) -> None
    def ground_text(self, id: str, text_sdr: torch.Tensor) -> None
    
    # Query (поиск по любой модальности)
    def query_visual(self, z_real: torch.Tensor) -> Concept | None  # cosine sim
    def query_text(self, word: str) -> Concept | None               # exact match
    
    # Каузальное рассуждение
    def predict(self, concept_id: str, action: str,
                inventory: dict) -> CausalLink | None
    def plan(self, goal_id: str) -> list[PlannedStep]  # backward chaining
    
    # Верификация
    def verify(self, concept_id: str, action: str,
               actual_outcome: str | None) -> None     # PE → confidence update
    def record_surprise(self, outcome: str, action: str) -> None  # log unexpected
    
    # Persistence
    def save(self, path: str) -> None
    def load(self, path: str) -> None
```

### Жизненный цикл концепта

```
Этап 1: Загрузка учебника
  Concept("tree", visual=None, text_sdr=None,
          attributes={category: "resource"},
          causal_links=[do → wood], confidence=0.5)

Этап 2: Text grounding
  GroundedTokenizer("tree") → text_sdr
  concept.text_sdr = text_sdr

Этап 3: Visual grounding (co-activation session)
  CrafterControlledEnv.reset_near("tree") → pixels
  CNNEncoder(pixels) → z_real
  concept.visual = z_real

Этап 4: Верификация через опыт
  agent does "do" near tree → OutcomeLabeler → wood +1
  concept.causal_links[0].confidence: 0.5 → 0.65

Этап 5: Многократное подтверждение
  confidence: 0.65 → 0.77 → 0.86 → 0.95
```

---

## 3. CrafterTextbook — атомарные правила

Статический YAML с атомарными каузальными правилами. Никаких цепочек, никаких prerequisites — мировая модель выводит зависимости сама через backward chaining.

```yaml
# configs/crafter_textbook.yaml
domain: crafter

vocabulary:
  # Видимые объекты (визуально заземляемые)
  - { id: tree, category: resource }
  - { id: stone, category: resource }
  - { id: coal, category: resource }
  - { id: iron, category: resource }
  - { id: table, category: crafted }
  - { id: empty, category: terrain }
  - { id: zombie, category: enemy, dangerous: true }
  # Инвентарные предметы (только текстовый grounding)
  - { id: wood, category: item }
  - { id: stone_item, category: item }
  - { id: coal_item, category: item }
  - { id: iron_item, category: item }
  - { id: wood_pickaxe, category: tool }
  - { id: stone_pickaxe, category: tool }
  - { id: wood_sword, category: weapon }

rules:
  - "do tree gives wood"
  - "do stone gives stone_item requires wood_pickaxe"
  - "do coal gives coal_item requires wood_pickaxe"
  - "do iron gives iron_item requires stone_pickaxe"
  - "place table on empty requires wood"
  - "make wood_pickaxe near table requires wood"
  - "make stone_pickaxe near table requires wood and stone_item"
  - "make wood_sword near table requires wood"
  - "do zombie with wood_sword kills zombie"
  - "zombie nearby without wood_sword means flee"
```

### Загрузка

```python
class CrafterTextbook:
    def __init__(self, path: str):
        self.data = yaml.safe_load(open(path))

    @property
    def vocabulary(self) -> list[dict]
    @property
    def rules(self) -> list[str]

    def load_into(self, store: ConceptStore) -> None:
        """Parse vocabulary → register concepts,
           parse rules → add causal links.
           Парсинг правил формата 'do tree gives wood requires X'
           встроен в CrafterTextbook (regex, не RuleBasedChunker —
           формат правил фиксированный, SVO chunker для него избыточен)."""
```

### Переносимость

Для нового домена — только новый YAML. Код не меняется.

---

## 4. GroundingSession — visual grounding через co-activation

"Родитель показывает ребёнку объекты и называет их."

### Механизм

```python
class GroundingSession:
    def __init__(self, env: CrafterControlledEnv,
                 encoder: CNNEncoder,
                 tokenizer: GroundedTokenizer,
                 store: ConceptStore):
        ...

    def ground_all(self, rng) -> GroundingReport:
        """Заземлить все визуально наблюдаемые концепты"""

    def ground_one(self, concept_id: str, rng) -> None:
        """Показать объект K=5 раз (разные seeds),
           усреднить z_real, привязать text_sdr"""
```

### Для каждого объекта

```
ground_one("tree"):
  repeat K=5 times (разные seeds):
    env.reset_near("tree") → pixels (3, 64, 64)
    encoder(pixels) → z_real (2048)
    accumulate z_real

  concept.visual = L2_normalize(mean(z_reals))
  concept.text_sdr = tokenizer.encode("tree")
```

K=5: один кадр = один ракурс. Среднее по нескольким — устойчивый прототип.

### Что заземляется визуально

| Концепт | Визуально? | Причина |
|---------|-----------|---------|
| tree, stone, coal, iron, table, empty | Да | Видимые объекты рядом с агентом |
| zombie | Да | Видимый враг |
| wood, wood_pickaxe, stone_pickaxe, wood_sword | Нет | Инвентарь = проприоцепция, не перцепция |

### Выход

```python
@dataclass
class GroundingReport:
    grounded: list[str]           # визуально заземлённые
    skipped: list[str]            # инвентарные (только текст)
    visual_sim_matrix: dict       # попарные cosine sim для диагностики
```

---

## 5. ChainGenerator — планирование из каузальных правил

Backward chaining через ConceptStore. Заменяет хардкоженные TREE_CHAIN, COAL_CHAIN.

### Интерфейс

```python
class ChainGenerator:
    def __init__(self, store: ConceptStore):
        ...

    def plan(self, goal: str) -> list[PlannedStep]:
        """Backward chaining: goal → prerequisites → base resources"""

    def generate_chain(self, goal: str) -> list[ScenarioStep]:
        """plan() → ScenarioStep для ScenarioRunner"""
```

### PlannedStep

```python
@dataclass
class PlannedStep:
    action: str            # "do", "make", "place"
    target: str            # concept_id ("tree", "iron")
    near: str | None       # "table" для craft, None для gather
    expected_gain: str     # "wood", "iron_item"
    requires: dict         # inventory snapshot needed
```

### Backward chaining пример

```
Goal: "iron_item"

1. do iron gives iron_item requires stone_pickaxe
   → нужен stone_pickaxe + найти iron

2. make stone_pickaxe near table requires wood + stone_item
   → нужны wood, stone_item, table

3. do stone gives stone_item requires wood_pickaxe
   → нужен wood_pickaxe + найти stone

4. make wood_pickaxe near table requires wood
   → нужны wood, table

5. place table on empty requires wood
   → нужен wood

6. do tree gives wood
   → базовый ресурс, нет prerequisites

Forward (план исполнения):
  [do tree ×N, place table, make wood_pickaxe,
   do stone, make stone_pickaxe, do iron]
```

### Конвертация в ScenarioStep

`generate_chain()` превращает PlannedStep → ScenarioStep (формат ScenarioRunner):
- `navigate_to` = target concept id
- `action` = Crafter action name
- `near_label` = target (для NearDetector training)
- `prerequisite_inv` = requires
- `repeat` = количество ресурса (вычислено из зависимостей)

---

## 6. Zombie handling — реактивное поведение

Не отдельная система — расширение основного цикла ScenarioRunner.

### Механизм

```python
def reactive_check(self, near: str, inventory: dict) -> str | None:
    """Проверить reactive правило для текущей ситуации.
    Returns: action override или None (продолжать план)"""

    concept = self.store.query_text(near)
    if concept is None or not concept.attributes.get("dangerous"):
        return None

    combat = concept.find_causal(action="do", check_requires=inventory)
    if combat:
        return "do"    # атакуем (есть меч)

    return "flee"      # убегаем (нет меча)
```

### Flee

Простая эвристика: 3-5 шагов в случайном направлении (кроме направления к zombie). После отхода — продолжить план с текущего шага.

### Интеграция в ScenarioRunner

```
while not done:
    near = NearDetector(pixels)

    # Reactive layer (приоритет над планом)
    override = reactive_check(near, inventory)
    if override == "do":   → attack
    if override == "flee": → отойти
    else:                  → execute planned step

    # Prediction error loop
    prediction = predict(near, action)
    outcome = OutcomeLabeler(inv_before, inv_after)
    verify(prediction, outcome)
```

### NearDetector — zombie training data

Zombie уже есть в NEAR_OBJECTS/NEAR_CLASSES. Проблема не в классе, а в отсутствии training data (enemies отключены через `_balance_chunk` monkeypatch). Решение:
- Отключаем monkeypatch (enemies ON) для сбора zombie frames
- GroundingSession заземляет zombie визуально через spawn
- ScenarioRunner с включёнными врагами собирает zombie frames для NearDetector

---

## 7. Prediction Error Loop — верификация правил

### Top-down (перед действием)

```python
def predict_before_action(self, near: str, action: str,
                           inventory: dict) -> Prediction | None:
    concept = self.store.query_text(near)
    if concept is None:
        return None
    link = concept.find_causal(action=action, check_requires=inventory)
    if link is None:
        return None
    return Prediction(expected=link.result, confidence=link.confidence)
```

### Bottom-up (после действия)

```python
def verify_after_action(self, prediction: Prediction | None,
                         action: str, inv_before: dict,
                         inv_after: dict) -> None:
    actual = OutcomeLabeler.label(action, inv_before, inv_after)

    if prediction is None:
        if actual is not None:
            self.store.record_surprise(actual, action)
        return

    if actual == prediction.expected:
        link.confidence = min(1.0, link.confidence + CONFIRM_DELTA)
    else:
        link.confidence = max(0.0, link.confidence - REFUTE_DELTA)
```

### Параметры

- `CONFIRM_DELTA = 0.15`
- `REFUTE_DELTA = 0.15`
- Начальная confidence правил из учебника: `0.5`

---

## 8. Допущения и ограничения

| # | Допущение | Почему ок сейчас | Когда убрать |
|---|-----------|-----------------|--------------|
| 1 | PrototypeMemory не интегрирован в ConceptStore | ConceptStore = 1 z_real на концепт, PrototypeMemory = тысячи экземпляров. Разные задачи | Будущий stage: episodic memory |
| 2 | Confidence delta фиксированный (±0.15) | Достаточно для 15 правил | Когда правил >50 |
| 3 | Surprise только логируется, не порождает правила | Нужен отдельный дизайн: noise vs реальное правило | Отдельный stage: autonomous discovery |
| 4 | Flee = простая эвристика (3-5 шагов) | Zombie медленный | Быстрые враги |
| 5 | Один тип врага (zombie) | Самый частый. Расширить YAML | Следующий survival stage |
| 6 | Нет стратегии "сначала крафт меча" | Planning-level, не reactive | Stage 72+: risk-aware planning |
| 7 | Нет decay правил | В Crafter правила статичны | Динамический домен |
| 8 | Planner не оптимизирует порядок | Корректность > эффективность | Episode length bottleneck |
| 9 | Nav encoder Phase 0 на exp122 (символьные траектории) | Фокус = text-visual, не nav cleanup | Stage 72: pixel-only nav |
| 10 | `use_semantic_nav=True` остаётся для редких объектов | Controlled env обходит навигацию | Stage 72 |

### Открытый вопрос

**Surprise mechanism**: как агент должен обрабатывать неожиданные события? Логирование недостаточно для world model. Автоматическое создание правил из одного наблюдения опасно (noise). Требует отдельной проработки: порог подтверждений, фильтрация noise, интеграция в ConceptStore.

---

## 9. Gate-критерии

| # | Gate | Метрика | Порог |
|---|------|---------|-------|
| 1 | Grounding | Визуально заземлённые концепты | ≥7 (tree, stone, coal, iron, table, empty, zombie) |
| 2 | Causal load | Правила загружены и предсказывают | ≥10 правил, predict accuracy ≥90% |
| 3 | Backward chaining | Корректные цепочки для goal items | plan("iron_item") = валидная цепочка ≥5 шагов |
| 4 | Cross-modal QA | Текстовый вопрос → ответ через ConceptStore | accuracy ≥80% на 10+ вопросов |
| 5 | Zombie survival | Выживаемость с reactive rules | episode length: reactive > baseline × 1.5 |
| 6 | Verification loop | Confidence растёт после опыта | confidence ≥0.8 после 5 подтверждений для ≥3 правил |
| 7 | Regression | Smoke/QA не хуже Stage 70 | smoke ≥60%, QA ≥85% |

### Тестирование

- Gates 1-4, 6: локально (pytest, без GPU)
- Gates 5, 7: minipc (полный Crafter, GPU)

---

## 10. Архитектурная диаграмма

```
┌──────────────────────────────────────────────────────────────┐
│                    Stage 71: Text-Visual Integration          │
│                                                              │
│  ┌─────────────┐     parse      ┌──────────────────────┐    │
│  │  Textbook    │──────────────→│  ConceptStore         │    │
│  │  (YAML)      │               │                      │    │
│  │  ~15 rules   │               │  Concept:             │    │
│  └─────────────┘               │   .id                  │    │
│                                 │   .visual (z_real)     │    │
│  ┌─────────────┐   ground      │   .text_sdr            │    │
│  │  Grounding   │──────────────→│   .attributes          │    │
│  │  Session     │               │   .causal_links        │    │
│  │  (co-activ.) │               │   .confidence          │    │
│  └─────────────┘               │                        │    │
│        ↑                        │  Methods:              │    │
│  ┌─────┴───────┐               │   .predict(near, act)  │    │
│  │ Controlled   │               │   .plan(goal)          │    │
│  │ Env + CNN    │               │   .verify(pred, actual)│    │
│  └─────────────┘               │   .query_visual(z)     │    │
│                                 │   .query_text(word)    │    │
│                                 └───────┬──────┬────────┘    │
│                                         │      │             │
│                            ┌────────────┘      └──────┐      │
│                            ▼                          ▼      │
│                   ┌────────────────┐      ┌───────────────┐  │
│                   │ ChainGenerator │      │ ReactiveCheck │  │
│                   │                │      │               │  │
│                   │ plan(goal) →   │      │ near=zombie?  │  │
│                   │ backward chain │      │ sword→attack  │  │
│                   │ → ScenarioStep │      │ no sword→flee │  │
│                   └───────┬────────┘      └───────┬───────┘  │
│                           │                       │          │
│                           ▼                       ▼          │
│                   ┌─────────────────────────────────────┐    │
│                   │         ScenarioRunner               │    │
│                   │                                      │    │
│                   │  while not done:                     │    │
│                   │    near = NearDetector(pixels)        │    │
│                   │    if reactive_check(near) → react   │    │
│                   │    else → execute planned step        │    │
│                   │    prediction = predict(near, action) │    │
│                   │    outcome = OutcomeLabeler(inv delta) │   │
│                   │    verify(prediction, outcome)        │    │
│                   └─────────────────────────────────────┘    │
│                                                              │
│  Существующие (без изменений):                               │
│    CNNEncoder, OutcomeLabeler, CrafterPixelEnv,              │
│    CrafterControlledEnv, CrafterSpatialMap                   │
│                                                              │
│  Расширяемые:                                                │
│    NearDetector (+zombie training data), ScenarioRunner      │
│    (+reactive layer)                                         │
│                                                              │
│  Новые:                                                      │
│    ConceptStore, CrafterTextbook, ChainGenerator,            │
│    GroundingSession, ReactiveCheck                            │
│                                                              │
│  Поглощаются ConceptStore:                                   │
│    GroundingMap, CausalWorldModel                            │
└──────────────────────────────────────────────────────────────┘
```

### Файловая структура

```
src/snks/
  agent/
    concept_store.py        # ConceptStore, Concept, CausalLink
    crafter_textbook.py     # CrafterTextbook (YAML loader)
    chain_generator.py      # ChainGenerator (backward chaining)
    grounding_session.py    # GroundingSession (co-activation)
    reactive_check.py       # ReactiveCheck (zombie flee/attack)

configs/
  crafter_textbook.yaml     # Атомарные правила Crafter

tests/
  test_stage71.py           # Gates 1-7
```

### Порядок реализации

1. ConceptStore + CrafterTextbook (загрузка правил)
2. GroundingSession (visual + text grounding)
3. ChainGenerator (backward chaining → ScenarioChains)
4. Интеграция в ScenarioRunner (prediction error loop)
5. NearDetector +zombie, ReactiveCheck
6. Gates: pytest локально → exp на minipc
