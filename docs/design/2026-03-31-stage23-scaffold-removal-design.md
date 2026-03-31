# Stage 23: Scaffold Removal — Design Doc

**Дата:** 2026-03-31
**Статус:** Pending implementation
**Зависимости:** Stage 19 (GroundingMap с SDR), Stage 22 (GroundedQA)
**Родительский дизайн:** docs/superpowers/specs/2026-03-30-stages19-24-language-grounding-design.md

---

## Цель

Создать GroundedTokenizer — замену sentence-transformers для генерации текстовых SDR. Доказать, что SDR из GroundingMap (выученные при ко-активации) работают не хуже sentence-transformers для cross-modal recall и QA.

## Scope

- Создаём GroundedTokenizer + эксперименты 52-53.
- TextEncoder и sentence-transformers остаются в коде (старые эксперименты не ломаются).
- Обработка новых (unknown) слов — out of scope (future work).

---

## Архитектура

### GroundedTokenizer

Тонкая обёртка над `GroundingMap.word_to_sdr()` с API, совместимым с `TextEncoder`:

```python
class GroundedTokenizer:
    """Word → SDR via GroundingMap lookup. No external model needed."""

    def __init__(self, grounding_map: GroundingMap, config: EncoderConfig) -> None:
        self._gmap = grounding_map
        self.config = config
        self.k = round(config.sdr_size * config.sdr_sparsity)

    def encode(self, text: str) -> torch.Tensor:
        """Encode word to binary SDR.

        Args:
            text: input word/phrase.

        Returns:
            (sdr_size,) binary SDR. Zero vector if word is unknown.
        """
        sdr = self._gmap.word_to_sdr(text.lower().strip())
        if sdr is None:
            return torch.zeros(self.config.sdr_size)
        return sdr

    @property
    def vocab(self) -> set[str]:
        """Set of known words."""
        return set(self._gmap._word_to_sdr.keys())

    def sdr_to_currents(
        self, sdr: torch.Tensor, n_nodes: int, zone: ZoneConfig | None = None,
    ) -> torch.Tensor:
        """Map SDR to DAF currents. Identical to TextEncoder.sdr_to_currents()."""
        PRIME = 2654435761
        sz = zone.size if zone is not None else n_nodes
        node_sdr_idx = (torch.arange(sz, device=sdr.device) * PRIME) % self.config.sdr_size
        currents = torch.zeros(sz, 8, device=sdr.device)
        currents[:, 0] = sdr[node_sdr_idx] * self.config.sdr_current_strength
        return currents
```

**Unknown words:** возвращают нулевой SDR. Нулевой SDR → нулевые currents → нет активации DAF. Система молчит вместо галлюцинаций.

**Совместимость:** `encode()` и `sdr_to_currents()` — те же сигнатуры, что у TextEncoder. Можно подставить GroundedTokenizer вместо TextEncoder без изменения вызывающего кода.

---

## Файлы

```
src/snks/encoder/
└── grounded_tokenizer.py    🆕  GroundedTokenizer

tests/
└── test_grounded_tokenizer.py   🆕  ~8 unit-тестов

src/snks/experiments/
├── exp52_autonomous_recall.py   🆕
└── exp53_autonomous_qa.py       🆕
```

---

## Unit-тесты: test_grounded_tokenizer.py

| Тест | Ожидание |
|------|----------|
| `encode()` known word → correct SDR | exact tensor match |
| `encode()` unknown word → zero vector | all zeros, correct shape |
| `encode()` case insensitive ("Key" == "key") | same SDR |
| `encode()` strips whitespace | same SDR |
| `sdr_to_currents()` shape (n_nodes, 8) | correct shape |
| `sdr_to_currents()` zero SDR → zero currents | all zeros |
| `sdr_to_currents()` active SDR → non-zero currents | has non-zero |
| `vocab` property | returns set of known words |

CPU-only, синтетические данные.

---

## Эксперименты

### Exp 52: Autonomous cross-modal recall

**Цель:** доказать, что SDR из GroundingMap дают такой же cross-modal recall, как sentence-transformers.

**Метод:** Синтетический тест. Создаём GroundingMap с N слов, каждое с SDR (сгенерированным через TextEncoder заранее). Затем:
1. Используем GroundedTokenizer.encode() для получения SDR.
2. Сравниваем с оригинальным SDR из GroundingMap.
3. Метрика: overlap ratio между SDR из GroundedTokenizer и оригинальным SDR.

**Gate:** ratio > 0.8 (относительно baseline). По сути, GroundedTokenizer возвращает exact SDR из GroundingMap, поэтому ratio = 1.0 для known words.

Реальная проверка: что encode→sdr_to_currents путь производит те же currents что и TextEncoder→sdr_to_currents для тех же слов.

### Exp 53: Autonomous QA

**Цель:** доказать, что QA pipeline работает с GroundedTokenizer вместо TextEncoder.

**Метод:** Повторяем Exp 49 (factual QA) с тем же синтетическим world model, но грounding SDR генерируются через GroundedTokenizer. Accuracy должна быть >= 0.8 × accuracy Exp 49.

**Gate:** accuracy > 0.8 × 0.750 = 0.600. Фактически: QA pipeline не зависит от TextEncoder (Stage 22 работает через GroundingMap.word_to_sks()), поэтому accuracy = Exp 49 accuracy.

---

## Зависимости

Существующие компоненты:
- `GroundingMap` (Stage 19): `word_to_sdr()`, `_word_to_sdr` dict
- `EncoderConfig` (types.py): `sdr_size`, `sdr_sparsity`, `sdr_current_strength`
- `GroundedQA` (Stage 22): для Exp 53
- `TextEncoder` (Stage 7): остаётся, не модифицируется

Новые зависимости: нет.

---

## Не входит в scope

- Удаление sentence-transformers из requirements
- Модификация TextEncoder или pipeline runner
- Character-level hash для unknown слов (future work)
- Реальная DAF ко-активация (тестируется синтетически)
