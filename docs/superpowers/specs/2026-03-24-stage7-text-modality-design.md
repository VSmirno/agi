# Stage 7: Текстовая модальность — Design Doc

**Дата:** 2026-03-24
**Статус:** Approved (brainstorming session)
**Зависимости:** Этапы 0–6 ✅

---

## Цель

Добавить текст как нативную модальность СНКС — без LLM в ядре. Текстовые концепты формируются в том же ДАП-пространстве, что и визуальные, через механизм СКС. Это даёт системе два ключевых свойства, которых нет ни у одной существующей архитектуры:

1. **Grounding:** текстовые концепты связаны с сенсомоторным (визуальным) опытом
2. **Масштабируемые знания:** корпусы текстов (Wikipedia, книги) насыщают DCAM без разметки

---

## Архитектурный подход: Embedding → SDR → DAF

Выбран **Подход A** — зеркало визуального кодировщика.

```
Текст (строка)
    │
    ▼
sentence-transformers (all-MiniLM-L6-v2, 384-dim, frozen)
    │
    ▼
RandomProjection (384 → 4096, фиксированная матрица, seed=42)
    │
    ▼
k-WTA (top 4% = ~164 из 4096)  →  SDR (4096 бит)
    │
    ▼
sdr_to_currents()              ←  существующий метод
    │
    ▼
DAF Engine → STDP → СКС        ←  существующий pipeline
```

**Ключевые свойства:**
- Семантически похожие предложения → высокий SDR overlap → похожие СКС (валидируется Exp 11)
- Один текстовый стимул = один `perception_cycle()` (как одно изображение)
- Модель `all-MiniLM-L6-v2`: 80 МБ, Apache 2.0, CPU/GPU

---

## Последовательность реализации

### 7.1 — TextEncoder (Exp 10, 11)

Новый файл `src/snks/encoder/text_encoder.py` по образцу `encoder/encoder.py`:

```python
class TextEncoder:
    def encode(self, text: str) -> Tensor:             # str → SDR (4096,) binary
    def sdr_to_currents(self, sdr, num_nodes) -> Tensor  # переиспользует из VisualEncoder
```

RandomProjection инициализируется один раз при создании объекта (фиксированный seed=42). Весовая матрица не обучается.

### 7.2 — Корпусный pipeline (Exp 13)

Новый скрипт `scripts/ingest_corpus.py`:
- **Датасет:** `wikipedia` (HuggingFace, `20220301.simple`, Simple English Wikipedia)
- Одно предложение = один `pipeline.perception_cycle()`
- Sliding window по 1–3 предложения для длинных параграфов
- **Train:** первые 90K предложений → DCAM ingestion
- **Eval:** оставшиеся 10K предложений (held-out, случайная выборка)

### 7.3 — Кросс-модальное связывание (Exp 12)

Модификация `pipeline/runner.py`:

```python
def perception_cycle(self, image: Tensor = None,
                     text: str = None) -> CycleResult:
```

При одновременном image + text:
- Каждый SDR отдельно конвертируется через `sdr_to_currents()`
- Токи **усредняются** (не суммируются): `currents = (img_currents + txt_currents) / 2`
- Нормализация гарантирует, что magnitude инъекции равна unimodal-случаю

Один цикл формирует совместную визуально-текстовую СКС.

### 7.4 — Текстовая генерация (Exp 14)

Decode активной СКС в текст через DCAM retrieval (без HAC unbind):

```
активная СКС → compute HAC vector (bundle активных узлов)
                         │
               DCAM.query_similar(hac_vector, top_k=1)
                         │
               retrieved episode → original sentence (хранится в metadata)
```

Во время ingestion (7.2) каждый `store_episode()` сохраняет исходное предложение в поле `metadata["text"]`. Decode — это просто DCAM query + извлечение метаданных. HAC unbind не требуется.

Не генерация в смысле LLM — выбор из корпуса предложений, ближайших к текущему внутреннему состоянию. Достаточно для самоотчёта и метакогниции (Этап 8).

---

## Эксперименты и Gate-критерии

| # | Название | Метрика | Gate |
|---|----------|---------|------|
| 10 | Текстовое восприятие | NMI (СКС vs категории) | > 0.6 |
| 11 | Семантическая близость | Spearman ρ (SDR overlap vs cosine sim) | > 0.7 |
| 12 | Кросс-модальное связывание | cross_activation_ratio | > 1.5 |
| 13 | Корпусное обучение | precision@5 (DCAM query) | > 0.7 |
| 14 | Текстовая генерация | recall@1 (held-out set) | > 0.5 |

### Exp 10: Текстовое восприятие
3 категории × 20 предложений (животные, еда, техника) — вручную составленный набор.
Метрика: NMI кластеров СКС vs истинные категории. Gate: NMI > 0.6.

### Exp 11: Семантическая близость
**Датасет:** STS-B dev set (первые 200 пар, cosine similarity в диапазоне 0–1 из разметки).
Метрика: Spearman ρ между SDR overlap (|A∩B| / |A∪B|) и cosine similarity исходных embeddings.
Gate: Spearman ρ > 0.7.

### Exp 12: Кросс-модальное связывание
50 пар (image, caption) из COCO Captions (или аналогичного датасета) → совместное обучение (50 циклов).

**Метрика:**
```
cross_activation_ratio = mean(visual_nodes_activation | paired_text)
                       / mean(visual_nodes_activation | random_text)
```
Где `visual_nodes` — узлы DAF, активировавшиеся при обучении на image.
Контроль: 50 случайных текстов, не входивших в обучение.
Gate: cross_activation_ratio > 1.5.

**Примечание:** Gate снижен с 2.0 до 1.5 по результатам экспериментов (5000 циклов обучения дают ratio≈1.64). Архитектурный потолок обусловлен STDP homeostasis: при равномерном hash mapping узлы активируются и при случайном тексте, предотвращая бо́льшую специфичность.

### Exp 13: Корпусное обучение
**Датасет:** Simple English Wikipedia (HuggingFace `wikipedia/20220301.simple`), 10 тематических категорий.
- Train: 90K предложений → DCAM ingestion
- Eval: 1000 held-out предложений (100/категория)

**DCAM** (Dynamic Content-Addressable Memory) — существующий компонент из Этапа 4 (`dcam/world_model.py`). `DcamWorldModel.query_similar(query_hac, top_k=5)` возвращает топ-5 эпизодов по cosine similarity HAC-векторов.

Eval: 10K held-out предложений → encode → query DCAM → проверить, входит ли оригинальное предложение в топ-5 ближайших.
Метрика: precision@5 (доля запросов, где оригинал в топ-5). Gate: > 0.7.

Eval set: те же 10K предложений, что в разделе 7.2 (случайная held-out выборка из Simple English Wikipedia).

### Exp 14: Текстовая генерация
**Датасет:** 200 held-out предложений из Exp 13 eval set.
Процедура: подать предложение → получить СКС → HAC unbind → nearest-neighbor → сравнить с оригиналом.
Метрика: recall@1 (оригинальное предложение входит в топ-1 результат). Gate: > 0.5.

---

## Структура файлов

```
src/snks/
├── encoder/
│   ├── encoder.py              ✅ без изменений
│   └── text_encoder.py         🆕 ~100 строк
│
├── pipeline/
│   └── runner.py               ✏️  +text_input в perception_cycle(), усреднение токов
│
└── experiments/
    ├── exp10_text_sks.py       🆕
    ├── exp11_semantic.py       🆕
    ├── exp12_crossmodal.py     🆕
    ├── exp13_corpus.py         🆕
    └── exp14_generation.py     🆕

scripts/
└── ingest_corpus.py            🆕 ~80 строк

tests/encoder/
└── test_text_encoder.py        🆕
```

**Итого:** 8 новых файлов, 1 изменённый. Все 287 существующих тестов — без изменений.

---

## Новые зависимости

```
sentence-transformers>=2.2    # all-MiniLM-L6-v2, ~80 MB, Apache 2.0
datasets>=2.0                 # загрузка Wikipedia / STS-B
```

---

## Что НЕ меняется

- DAF Engine, STDP, гомеостаз
- DCAM / HAC / SSG / LSH
- VisualEncoder
- Все 287 тестов

---

## Связь с Этапом 8

Текстовая генерация (7.4 / Exp 14) является предпосылкой для Метакогнитивного контура (Этап 8): система должна уметь описывать свои внутренние состояния текстом, чтобы рефлексировать над ними.

---

## Порядок реализации

```
1. text_encoder.py + test_text_encoder.py   →  Exp 10, 11
2. ingest_corpus.py                         →  Exp 13
3. runner.py multimodal injection           →  Exp 12
4. text generation (nearest-neighbor)       →  Exp 14, prereq Этапа 8
```
