# Stage 31: Abstract Pattern Reasoning — Design Specification

**Дата:** 2026-03-31
**Статус:** IN PROGRESS
**Ветка:** stage31-abstract-pattern
**Эксперименты:** exp77, exp78, exp79

---

## Цель

Доказать, что СНКС может обнаруживать абстрактные паттерны в последовательностях
концептов (СКС) и предсказывать недостающие элементы — аналог Raven's Progressive
Matrices, но на уровне нативных концептов, а не пикселей.

## Что доказывает

1. Агент обнаруживает **трансформационные правила** между СКС в последовательности
2. Правила выражаются как **HAC-операции** (bind/unbind/permute) — алгебра на концептах
3. Правила **применяются** к новым элементам для предсказания (pattern completion)
4. Работает на **нескольких типах паттернов**: прогрессия, аналогия, чередование

## Философская корректность

- Паттерны = отношения между концептами (СКС), НЕ символьные правила
- Обнаружение = алгебраические операции в HAC-пространстве, НЕ перебор шаблонов
- Предсказание = применение обнаруженной трансформации, НЕ lookup в таблице
- Полностью согласовано с философией SPEC.md: мышление = оперирование концептами

---

## Подходы (brainstorming)

### Подход A: HAC Transform Discovery
Для каждой пары соседних элементов (e_i, e_{i+1}) вычисляется трансформация
через unbind: T = unbind(e_i, e_{i+1}). Если T консистентна по всей
последовательности (cosine similarity > threshold), правило найдено.
Предсказание = bind(e_last, T).

**Trade-off:** Простой и элегантный. Работает для линейных трансформаций.
Может не справиться с нелинейными паттернами (XOR-like).

### Подход B: Multi-rule decomposition
Каждый элемент кодируется как bundle нескольких аттрибутов
(shape, color, position). Правила ищутся для каждого аттрибута отдельно.

**Trade-off:** Более мощный, но требует явной декомпозиции на аттрибуты,
что может быть ad hoc.

### Подход C: Neural sequence prediction
HAC-последовательность → простая RNN/MLP → предсказание.

**Trade-off:** Не соответствует философии СНКС (нет backprop).

### Выбор: Подход A + элементы B

**Обоснование:**
- HAC Transform Discovery — чистая алгебра на концептах, нет backprop
- Согласовано с существующим HAC engine (bind/unbind/permute)
- Для сложных матричных паттернов добавляем multi-attribute unbind (подход B)
  через row/column decomposition — стандартная HAC-операция
- Подход C отвергнут — нарушает философию СНКС

---

## Архитектура

### Новые модули

#### 1. `PatternElement` (dataclass)
```python
@dataclass
class PatternElement:
    sks_ids: frozenset[int]     # концепты в этой ячейке
    embedding: Tensor           # HAC-вектор (2048D)
    position: tuple[int, ...]   # (row, col) или (index,)
```

#### 2. `PatternMatrix` (dataclass)
```python
@dataclass
class PatternMatrix:
    elements: list[PatternElement]  # row-major order
    shape: tuple[int, int]          # (rows, cols), e.g. (3, 3)
    missing: int                    # index of missing element
```

#### 3. `TransformRule` (dataclass)
```python
@dataclass
class TransformRule:
    transform_vector: Tensor    # HAC-вектор трансформации
    axis: str                   # "row" | "column" | "sequence"
    consistency: float          # mean cosine similarity
```

#### 4. `AbstractPatternReasoner` (main class)
```python
class AbstractPatternReasoner:
    def __init__(self, hac: HACEngine, threshold: float = 0.6):
        ...

    def discover_rules(self, matrix: PatternMatrix) -> list[TransformRule]:
        """Найти трансформационные правила по строкам и столбцам."""

    def predict_missing(self, matrix: PatternMatrix) -> tuple[Tensor, float]:
        """Предсказать embedding недостающего элемента."""

    def select_answer(self, prediction: Tensor, options: list[Tensor]) -> int:
        """Выбрать ближайший вариант ответа (Raven's-style)."""
```

### Алгоритм

#### Discover Rules:
1. Для каждой строки: T_row_i = unbind(e[i,0], e[i,1]), проверить T_row_i ≈ unbind(e[i,1], e[i,2])
2. Для каждого столбца: T_col_j = unbind(e[0,j], e[1,j]), проверить T_col_j ≈ unbind(e[1,j], e[2,j])
3. Консистентность = средний cosine similarity между T в одном направлении
4. Правило принимается если consistency > threshold

#### Predict Missing (для позиции [2,2] в 3x3):
1. По строке: prediction_row = bind(e[2,1], T_row)
2. По столбцу: prediction_col = bind(e[1,2], T_col)
3. Финальный prediction = bundle([prediction_row, prediction_col])

#### Select Answer:
1. cosine_similarity(prediction, option_i) для каждого варианта
2. argmax → ответ

---

## Типы паттернов для тестирования

### 1. Constant Row Transform
Каждая строка: A → bind(A, T) → bind(bind(A, T), T)
Трансформация T одинакова для всех строк.

### 2. Progressive Sequence
Линейная последовательность: permute(e, k), permute(e, 2k), ...

### 3. Row-Column Double Rule
Строки имеют T_row, столбцы имеют T_col, оба правила работают одновременно.

### 4. Analogy Pattern (A:B :: C:?)
unbind(A, B) ≈ unbind(C, D) → D = bind(C, unbind(A, B))

---

## Gate-критерии

| Exp | Метрика | Gate | Описание |
|-----|---------|------|----------|
| 77 | rule_consistency | ≥ 0.7 | Средняя consistency обнаруженных правил |
| 77 | rule_found_rate | ≥ 0.8 | Доля матриц где правило найдено |
| 78 | completion_accuracy | ≥ 0.8 | Точность предсказания на 3x3 матрицах |
| 78 | analogy_accuracy | ≥ 0.8 | Точность A:B :: C:? |
| 79 | multi_rule_accuracy | ≥ 0.7 | Точность при двух правилах (row+col) |

---

## Файлы

| Файл | Описание |
|------|----------|
| `src/snks/language/pattern_element.py` | PatternElement, PatternMatrix dataclasses |
| `src/snks/language/abstract_pattern_reasoner.py` | AbstractPatternReasoner |
| `tests/test_abstract_pattern.py` | Unit tests |
| `src/snks/experiments/exp77_pattern_rules.py` | Rule discovery |
| `src/snks/experiments/exp78_pattern_completion.py` | Completion accuracy |
| `src/snks/experiments/exp79_multi_rule.py` | Double rule patterns |
| `demos/stage-31-abstract-pattern.html` | Web demo |

---

## Зависимости

- `HACEngine` (src/snks/dcam/hac.py) — bind, unbind, bundle, permute
- `SKSEmbedder` (src/snks/sks/embedder.py) — для генерации test embeddings
- Нет зависимости от MiniGrid/env — чисто концептные операции
