# Stage 24a: InstructionParser + Attribute Grounding — Design Doc

**Дата:** 2026-03-31
**Статус:** Pending implementation
**Зависимости:** Stage 20 (RuleBasedChunker, RoleFillerParser), Stage 19 (GroundingMap)
**Родительский дизайн:** docs/superpowers/specs/2026-03-30-stages19-24-language-grounding-design.md

---

## Цель

Расширить RuleBasedChunker для парсинга всех 5 уровней BabyAI инструкций. Проверить grounding атрибутов (цвета) через GroundingMap. Синтетический scope.

## Scope

- Расширение RuleBasedChunker: +sequential, +spatial паттерны, +SEQ_BREAK маркер
- Grounding атрибутов проверяется на синтетических данных (GroundingMap заполнен вручную)
- Нет execution, нет MiniGrid, нет planning

---

## Архитектура

### Новые паттерны в RuleBasedChunker

Существующие (без изменений):
- SVO: "cat sits on mat" → AGENT, ACTION, OBJECT
- ATTR+SVO: "red cat sits on mat" → ATTR, AGENT, ACTION, OBJECT
- MiniGrid: "pick up the red key" → ACTION, ATTR, OBJECT

Новые:

**Sequential** — split по "then" / "and then":
```
"pick up the key then open the door"
→ [Chunk("pick up", ACTION), Chunk("key", OBJECT),
   Chunk("", SEQ_BREAK),
   Chunk("open", ACTION), Chunk("door", OBJECT)]
```

Детекция: наличие " then " в тексте (case-insensitive).
Алгоритм: split по " then " / " and then ", каждую часть парсим как отдельную инструкцию (minigrid/SVO), вставляем SEQ_BREAK между ними.

**Spatial** — "put X next to Y" / "go to X near Y":
```
"put the red ball next to the blue box"
→ [Chunk("put", ACTION), Chunk("red", ATTR), Chunk("ball", OBJECT),
   Chunk("blue", ATTR), Chunk("box", LOCATION)]
```

Детекция: наличие "next to" / "beside" после ACTION+OBJECT.
Алгоритм: split по spatial preposition, первую часть парсим как minigrid, вторую — как LOCATION (с ATTR если есть).

### SEQ_BREAK

```python
@dataclass
class Chunk:
    text: str
    role: str  # "AGENT", "ACTION", "OBJECT", "LOCATION", "GOAL", "ATTR", "SEQ_BREAK"
```

SEQ_BREAK — маркер границы между последовательными инструкциями. `text=""`, `role="SEQ_BREAK"`. Потребители (InstructionPlanner в Stage 24b) split по SEQ_BREAK для получения отдельных инструкций.

### Attribute Grounding

Цвета BabyAI: red, green, blue, purple, yellow, grey. Уже в `ADJECTIVES` set в chunker.py — парсятся как `ATTR` role.

Для синтетического теста: GroundingMap заполняется вручную с цветами + объектами. Проверяем что "red key" → ATTR="red" (sks=N) + OBJECT="key" (sks=M), оба резолвятся.

### Порядок детекции паттернов

```python
def detect_pattern(self, sentence: str) -> str:
    lower = sentence.lower().strip()
    # 1. Sequential (most specific — contains "then")
    if " then " in lower:
        return "sequential"
    # 2. MiniGrid imperative
    for phrase in ACTION_PHRASES:
        if lower.startswith(phrase):
            return "minigrid"
    # 3. Spatial (contains "next to" / "beside")
    if "next to" in lower or "beside" in lower:
        return "spatial"
    # 4. ATTR+SVO
    words = lower.split()
    if words and words[0] in ADJECTIVES:
        return "svo_attr"
    # 5. SVO (default)
    return "svo"
```

Sequential проверяется первым — "pick up X then open Y" начинается с action phrase, но содержит "then".

---

## Файлы

```
src/snks/language/
└── chunker.py               ✏️  +sequential, +spatial, +SEQ_BREAK, detect_pattern updated

tests/
└── test_instruction_parser.py   🆕  ~15 unit-тестов

src/snks/experiments/
└── exp54a_instruction_parsing.py  🆕
```

---

## Unit-тесты: test_instruction_parser.py

| Группа | Тест | Ожидание |
|--------|------|----------|
| **Sequential** | "pick up the key then open the door" | 2 инструкции с SEQ_BREAK |
| | "go to the ball and then pick up the key" | "and then" тоже работает |
| | "open the door then go to the goal then pick up the key" | 3 инструкции |
| **Spatial** | "put the ball next to the box" | ACTION+OBJECT+LOCATION |
| | "put the red ball next to the blue box" | ACTION+ATTR+OBJECT+ATTR+LOCATION |
| **Attributes** | "pick up the red key" | ACTION+ATTR+OBJECT |
| | "go to the blue ball" | ACTION+ATTR+OBJECT |
| | "open the yellow door" | ACTION+ATTR+OBJECT |
| **Combined** | "pick up the red key then open the yellow door" | sequential + attrs |
| | "go to the green ball then put it next to the box" | sequential + spatial |
| **Grounding** | ATTR "red" + OBJECT "key" → GroundingMap resolves both | sks_ids non-None |
| | ATTR "purple" (unknown) → GroundingMap returns None | None for unknown |
| | All 6 BabyAI colors registered → all resolve | 6/6 non-None |
| **Edge cases** | "then" at start → single instruction | no SEQ_BREAK |
| | Empty second part "pick up the key then" | single instruction, no crash |

---

## Эксперимент 54a: Instruction Parsing Accuracy

**Метрика:** accuracy > 0.9
**Данные:** 30+ инструкций, все 5 уровней.

```python
TEST_INSTRUCTIONS = [
    # GoTo (6)
    ("go to the red ball", [("go to", "ACTION"), ("red", "ATTR"), ("ball", "OBJECT")]),
    ("go to the door", [("go to", "ACTION"), ("door", "OBJECT")]),
    ...
    # Pickup (6)
    ("pick up the blue key", [("pick up", "ACTION"), ("blue", "ATTR"), ("key", "OBJECT")]),
    ...
    # Open (4)
    ("open the yellow door", [("open", "ACTION"), ("yellow", "ATTR"), ("door", "OBJECT")]),
    ...
    # Sequential (8)
    ("pick up the key then open the door", [...SEQ_BREAK...]),
    ...
    # Spatial (6)
    ("put the red ball next to the blue box", [...LOCATION...]),
    ...
]
```

Валидация: для каждой инструкции проверяем что roles (text, role) совпадают с expected. Partial match допускается (если из 5 chunks 4 верны = 0.8 для этой инструкции).

Дополнительно: grounding resolve — для каждого chunk проверяем что `GroundingMap.word_to_sks(chunk.text)` возвращает non-None для known words.

---

## Зависимости

Существующие (модифицируется):
- `chunker.py` (Stage 20): +2 паттерна, +SEQ_BREAK, обновлённый detect_pattern

Существующие (без изменений):
- `GroundingMap` (Stage 19): word_to_sks()
- `Chunk` dataclass (Stage 20)

Новые зависимости: нет.

---

## Не входит в scope

- Execution в MiniGrid (Stage 24c)
- InstructionPlanner / planning (Stage 24b)
- Реальные QA backends (Stage 24b)
- SpaCy / learned chunker (future work)
