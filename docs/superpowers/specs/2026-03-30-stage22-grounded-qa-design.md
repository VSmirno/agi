# Stage 22: Grounded QA — Design Doc

**Дата:** 2026-03-30
**Статус:** Pending implementation
**Зависимости:** Stage 19 (GroundingMap), Stage 20 (RoleFillerParser, Chunker), Stage 21 (Verbalizer)
**Родительский дизайн:** docs/superpowers/specs/2026-03-30-stages19-24-language-grounding-design.md

---

## Цель

Система отвечает на вопросы трёх типов, извлекая знания из world model. Каждый ответ верифицируем: если знания нет — "I don't know", без галлюцинаций.

## Scope

Синтетический (как Stages 20-21). World model заполняется вручную, вопросы из фиксированного набора. QA-логика тестируется изолированно от DAF/DCAM.

---

## Архитектура

### Подход: тонкие backend-адаптеры

`GroundedQA` — orchestrator. Получает 3 backend-объекта через конструктор. Каждый backend реализует Protocol с методом `query()`. Для синтетических тестов — `DictBackend` (lookup). Позже (Stage 24) — реальные адаптеры к DCAM/Simulator/Metacog.

```
question: str
    │
    ▼ chunker.chunk(question)
chunks: list[Chunk]
    │
    ▼ grounding_map → resolve words to sks_ids
resolved_roles: dict[str, int]
    │
    ▼ classifier.classify(question)
qtype: QuestionType
    │
    ▼ backends[qtype].query(resolved_roles)
result: QAResult | None
    │
    ▼ if None → "I don't know"
    ▼ else → verbalizer methods
answer: str
```

---

## Компоненты

### QuestionType

```python
class QuestionType(Enum):
    FACTUAL = "factual"
    SIMULATION = "simulation"
    REFLECTIVE = "reflective"
```

### QAResult

```python
@dataclass
class QAResult:
    answer_sks: list[int]       # SKS IDs, найденные backend-ом
    confidence: float           # 0.0–1.0
    source: QuestionType        # откуда ответ
    metadata: dict              # backend-specific (action_name, reason, etc.)
```

`metadata` — расширяемый словарь для backend-specific данных:
- factual: `{"action": "opens"}` — какое действие связывает причину и следствие
- simulation: `{"action": "pick up", "steps": [...]}` — что произойдёт
- reflective: `{"reason": "explore", "pe": 0.8}` — почему агент так поступил

### QABackend (Protocol)

```python
class QABackend(Protocol):
    def query(self, roles: dict[str, int]) -> QAResult | None:
        """Query the knowledge source.

        Args:
            roles: resolved role → sks_id mapping.
                Keys: "AGENT", "ACTION", "OBJECT", "LOCATION", etc.
                Values: integer SKS IDs from GroundingMap.

        Returns:
            QAResult if knowledge found, None otherwise ("I don't know").
        """
        ...
```

### QuestionClassifier

Rule-based. Порядок проверок критичен: simulation-паттерны ("what happens if") перед factual ("what").

```python
class QuestionClassifier:
    # Compiled regex patterns, ordered by specificity (most specific first).
    SIMULATION_PATTERNS = [
        re.compile(r"what (happens|would happen|will happen) if\b", re.I),
        re.compile(r"what would\b", re.I),
    ]
    REFLECTIVE_PATTERNS = [
        re.compile(r"why (did|are|do|were) (you|i)\b", re.I),
    ]
    FACTUAL_PATTERNS = [
        re.compile(r"(what|who|where|which)\b", re.I),
    ]

    def classify(self, question: str) -> QuestionType:
        q = question.strip()
        for pat in self.SIMULATION_PATTERNS:
            if pat.match(q):
                return QuestionType.SIMULATION
        for pat in self.REFLECTIVE_PATTERNS:
            if pat.match(q):
                return QuestionType.REFLECTIVE
        for pat in self.FACTUAL_PATTERNS:
            if pat.match(q):
                return QuestionType.FACTUAL
        raise ValueError(f"Cannot classify question: {question!r}")
```

Неклассифицируемый вопрос → `ValueError` (не "I don't know" — это ошибка программиста, а не отсутствие знания).

### GroundedQA

```python
class GroundedQA:
    def __init__(
        self,
        classifier: QuestionClassifier,
        grounding_map: GroundingMap,
        chunker: BaseChunker,
        verbalizer: Verbalizer,
        factual: QABackend,
        simulation: QABackend,
        reflective: QABackend,
    ) -> None: ...

    def answer(self, question: str) -> str:
        """Full QA pipeline: classify → resolve → query → verbalize."""

        # 1. Classify question type.
        qtype = self.classifier.classify(question)

        # 2. Parse question into chunks, resolve to SKS IDs.
        chunks = self.chunker.chunk(question)
        roles: dict[str, int] = {}
        for chunk in chunks:
            sks_id = self.grounding_map.word_to_sks(chunk.text)
            if sks_id is not None:
                roles[chunk.role] = sks_id

        # 3. Route to appropriate backend.
        backend = {
            QuestionType.FACTUAL: self.factual,
            QuestionType.SIMULATION: self.simulation,
            QuestionType.REFLECTIVE: self.reflective,
        }[qtype]
        result = backend.query(roles)

        # 4. Verbalize.
        if result is None:
            return "I don't know"
        return self._verbalize(result, qtype)

    def _verbalize(self, result: QAResult, qtype: QuestionType) -> str:
        """Convert QAResult to text using Verbalizer and templates."""
        words = []
        for sks_id in result.answer_sks:
            w = self.grounding_map.sks_to_word(sks_id)
            if w is not None:
                words.append(w)

        if qtype == QuestionType.FACTUAL:
            # "the key" / "key and ball"
            return factual_answer_template(words)
        elif qtype == QuestionType.SIMULATION:
            action = result.metadata.get("action", "do something")
            return simulation_answer_template(action, words)
        elif qtype == QuestionType.REFLECTIVE:
            reason = result.metadata.get("reason", "unknown")
            return reflective_answer_template(reason)
        return "I don't know"
```

### Новые шаблоны (в templates.py)

```python
def factual_answer_template(objects: list[str]) -> str:
    """Format factual QA answer: 'the key' / 'key and ball'."""
    if not objects:
        return "I don't know"
    if len(objects) == 1:
        return f"the {objects[0]}"
    return ", ".join(objects[:-1]) + " and " + objects[-1]

def simulation_answer_template(action: str, effects: list[str]) -> str:
    """Format simulation answer: 'If you pick up, you will have key'."""
    if not effects:
        return "nothing happens"
    effect_str = " and ".join(effects)
    return f"you will have {effect_str}"

def reflective_answer_template(reason: str) -> str:
    """Format reflective answer from metacog reason."""
    return reason
```

Шаблоны намеренно минимальны — exact transmission, не NLG.

---

## Структура файлов

```
src/snks/language/
├── qa.py                      🆕  QuestionType, QAResult, QABackend, QuestionClassifier, GroundedQA
├── templates.py               ✏️  +3 шаблона (factual_answer, simulation_answer, reflective_answer)
├── grounding_map.py           (без изменений)
├── verbalizer.py              (без изменений)
├── parser.py                  (без изменений)
└── chunker.py                 (без изменений)

tests/
└── test_grounded_qa.py        🆕  ~18 unit-тестов

src/snks/experiments/
├── exp49_factual_qa.py        🆕
├── exp50_simulation_qa.py     🆕
└── exp51_reflective_qa.py     🆕
```

---

## Unit-тесты: test_grounded_qa.py

| Группа | Тест | Ожидание |
|--------|------|----------|
| **QuestionClassifier** | "What opens the door?" | FACTUAL |
| | "Who is near the wall?" | FACTUAL |
| | "Where is the key?" | FACTUAL |
| | "What happens if I pick up the key?" | SIMULATION |
| | "What would happen if I open the door?" | SIMULATION |
| | "Why did you go left?" | REFLECTIVE |
| | "Why are you exploring?" | REFLECTIVE |
| | "Hello world" → ValueError | ValueError |
| **QABackend** | DictFactualBackend: known query → QAResult | answer_sks non-empty |
| | DictFactualBackend: unknown query → None | None |
| **GroundedQA.answer** | Factual question, known answer | "the key" |
| | Factual question, unknown → "I don't know" | "I don't know" |
| | Simulation question, known effects | "you will have ..." |
| | Simulation question, no effects | "nothing happens" |
| | Reflective question, known reason | reason string |
| | Reflective question, unknown → "I don't know" | "I don't know" |
| **Edge cases** | Question with unknown word in roles | роль пропущена, backend может вернуть None |
| | All templates produce non-empty strings | no empty returns for valid input |

Тесты CPU-only, без GPU/DAF. Синтетические DictBackend-ы определяются в conftest или внутри тестового файла.

---

## Эксперименты

### Exp 49: Factual QA

**Метрика:** accuracy > 0.7
**Данные:** 20+ фактических вопросов, синтетический world model.

World model (dict):
```python
facts = {
    ("opens", "door"): ["key"],
    ("near", "wall"): ["agent"],
    ("holds", "key"): ["agent"],
    ("color", "ball"): ["red"],
    # ... 10+ фактов
}
```

Вопросы покрывают: что/кто/где/какой. Включают "I don't know" кейсы (вопрос о факте, которого нет в модели).

Валидация: сравнение `result.answer_sks` с expected SKS IDs (точное совпадение множеств). Дополнительно: вербализованный ответ содержит grounded-слова для answer_sks.

### Exp 50: Simulation QA

**Метрика:** accuracy > 0.6
**Данные:** 15+ симуляционных вопросов, синтетический simulator backend.

Simulator backend (dict):
```python
effects = {
    ("pick up", "key"): {"answer_sks": ["key_held"], "action": "pick up"},
    ("open", "door"): {"answer_sks": ["door_open"], "action": "open"},
    # state-dependent: "open door" without key → []
}
```

Включает: базовые эффекты, зависимые от состояния ("open without key" → nothing), цепочки ("pick up key then open door").

### Exp 51: Reflective QA

**Метрика:** accuracy > 0.6
**Данные:** 15+ рефлексивных вопросов, синтетический metacog log.

Metacog log (list of dicts):
```python
log = [
    {"action": "left", "reason": "Prediction error was high. I explored left.", "pe": 0.8},
    {"action": "pick up", "reason": "It was my goal to get the key.", "target": "key"},
    # ...
]
```

Валидация: ответ содержит ключевую причину (reason substring match).

---

## Зависимости

Существующие компоненты (без изменений):
- `GroundingMap` (Stage 19): `word_to_sks()`, `sks_to_word()`
- `RuleBasedChunker` (Stage 20): `chunk()` → `list[Chunk]`
- `Verbalizer` (Stage 21): не вызывается напрямую, но переиспользуем templates.py

Новые зависимости: нет. Всё на stdlib + существующем коде.

---

## Не входит в scope

- Реальная интеграция с DCAM/StochasticSimulator/MetacogMonitor (Stage 24)
- NLG / "красивые" ответы (шаблоны — exact transmission)
- Парсинг сложных вопросов (nested clauses, negation, comparatives)
- Dialogue context / multi-turn QA
