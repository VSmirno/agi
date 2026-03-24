# Stage 7: Implementation Plan — Текстовая модальность

**Spec:** `docs/superpowers/specs/2026-03-24-stage7-text-modality-design.md`
**Branch:** `stage7-text-modality`
**Цель:** 5 экспериментов PASS (Exp 10–14), все 287 существующих тестов не сломаны.

---

## Фаза 1: TextEncoder (Exp 10, 11)

### Задача 1.1 — Зависимости
```bash
source venv/bin/activate
pip install sentence-transformers>=2.2 datasets>=2.0
```
Добавить в `requirements.txt`.

**Проверка:** `python -c "from sentence_transformers import SentenceTransformer; print('ok')"`

---

### Задача 1.2 — `src/snks/encoder/text_encoder.py`

Создать класс `TextEncoder` по образцу `VisualEncoder`:

```python
class TextEncoder:
    def __init__(self, config: EncoderConfig, device=None):
        # 1. Загрузить sentence-transformers: SentenceTransformer("all-MiniLM-L6-v2")
        # 2. Создать RandomProjection матрицу: torch.randn(384, config.sdr_size, generator=...)
        #    seed=42, нормировать столбцы (L2)
        # 3. self.k = round(config.sdr_size * config.sdr_sparsity)
        pass

    def encode(self, text: str) -> torch.Tensor:
        # 1. model.encode(text) → numpy (384,) → torch float32
        # 2. projection: (384,) @ (384, 4096) → (4096,)
        # 3. kwta(result, self.k) → binary SDR (4096,)
        pass

    def sdr_to_currents(self, sdr: torch.Tensor, n_nodes: int) -> torch.Tensor:
        # Переиспользовать логику из VisualEncoder.sdr_to_currents()
        # (можно импортировать функцию или скопировать метод)
        pass
```

**Ключевые детали:**
- `SentenceTransformer` — frozen, `.eval()`, `torch.no_grad()`
- RandomProjection: `torch.Generator().manual_seed(42)`, `torch.randn(..., generator=g)`; нормировать по столбцам: `proj / proj.norm(dim=0)`
- `kwta` импортировать из `snks.encoder.sdr`
- `EncoderConfig` уже содержит `sdr_size` и `sdr_sparsity`

**Проверка:** `pytest tests/encoder/test_text_encoder.py -v`

---

### Задача 1.3 — `tests/encoder/test_text_encoder.py`

Написать тесты ДО реализации:

```python
def test_encode_returns_binary_sdr():
    # encode("hello world") → shape (4096,), dtype bool/float, sum ≈ 164

def test_similar_texts_higher_overlap():
    # overlap("cat sits", "dog runs") > overlap("cat sits", "CPU overheats")

def test_sdr_to_currents_shape():
    # sdr_to_currents(sdr, n_nodes=1000) → shape (1000, 8)

def test_deterministic():
    # encode("test") дважды → одинаковый результат
```

**Проверка:** все тесты PASS.

---

### Задача 1.4 — `src/snks/experiments/exp10_text_sks.py`

```python
# 3 категории × 20 предложений (животные, еда, техника)
# Для каждого предложения: TextEncoder.encode() → SDR
# Кластеризация SDR (k-means k=3) → NMI vs истинные категории
# Gate: NMI > 0.6
# Вывод: print("Exp 10 NMI:", nmi, "— PASS/FAIL")
```

Запускать локально (CPU, без DAF, только TextEncoder + NMI).

---

### Задача 1.5 — `src/snks/experiments/exp11_semantic.py`

```python
# Датасет: STS-B dev set (первые 200 пар)
# Загрузить через datasets: load_dataset("stsb_multi_mt", "en", split="dev")
# Для каждой пары: SDR overlap = |A∩B| / |A∪B|
# Spearman ρ между SDR overlap и cosine similarity из разметки (normalized 0-1)
# Gate: Spearman ρ > 0.7
```

---

**Контрольная точка Фазы 1:**
```
pytest tests/encoder/test_text_encoder.py          # все PASS
python -m snks.experiments.exp10_text_sks          # NMI > 0.6
python -m snks.experiments.exp11_semantic          # ρ > 0.7
pytest tests/ -x -q                                # 287+ тестов, без регрессий
```

---

## Фаза 2: Корпусный pipeline (Exp 13)

### Задача 2.1 — `scripts/ingest_corpus.py`

```python
# Аргументы CLI: --n-train 90000 --n-eval 10000 --output-dir /opt/agi/corpus/
# 1. Загрузить: load_dataset("wikipedia", "20220301.simple", split="train")
# 2. Разбить на предложения (nltk.sent_tokenize или простой split по ". ")
# 3. Train: первые n_train предложений → pipeline.perception_cycle(text=s)
#    (Pipeline с TextEncoder, маленький DAF config для теста: 10K узлов)
# 4. Eval set: следующие n_eval предложений → сохранить в JSON (sentence + embedding)
# 5. Прогресс: tqdm + логирование каждые 1000 предложений
```

**Детали хранения eval set:**
```json
{"sentence": "The cat sat on the mat.", "embedding": [0.1, 0.2, ...]}
```

---

### Задача 2.2 — `src/snks/experiments/exp13_corpus.py`

```python
# 1. Загрузить eval set из JSON (10K предложений)
# 2. Для каждого предложения: encode → query DCAM top-5
# 3. Проверить: оригинальное предложение в топ-5 результатов
# Gate: precision@5 > 0.7
```

**Запуск:** на mini-PC (ROCm, после `ingest_corpus.py`).

---

**Контрольная точка Фазы 2:**
```
python scripts/ingest_corpus.py --n-train 1000 --n-eval 100  # smoke test локально
# Полный запуск — на mini-PC:
bash scripts/remote.sh sync && ssh gem@10.253.0.179 -p 2244 \
  "cd /opt/agi && python scripts/ingest_corpus.py --n-train 90000 --n-eval 10000"
python -m snks.experiments.exp13_corpus            # precision@5 > 0.7
```

---

## Фаза 3: Кросс-модальное связывание (Exp 12)

### Задача 3.1 — Модификация `src/snks/pipeline/runner.py`

Изменить сигнатуру `perception_cycle()`:

```python
def perception_cycle(
    self,
    image: torch.Tensor | None = None,
    text: str | None = None,
) -> CycleResult:
```

Логика инъекции токов:
```python
currents = None

if image is not None:
    sdr = self.encoder.encode(image)
    currents = self.encoder.sdr_to_currents(sdr, n_nodes).to(device)

if text is not None:
    text_sdr = self.text_encoder.encode(text)
    text_currents = self.text_encoder.sdr_to_currents(text_sdr, n_nodes).to(device)
    if currents is None:
        currents = text_currents
    else:
        currents = (currents + text_currents) / 2.0  # усреднение, не сумма

if currents is None:
    raise ValueError("perception_cycle: укажите image или text")
```

`Pipeline.__init__()` — добавить `self.text_encoder = TextEncoder(config.encoder, device)`.

**Обратная совместимость:** существующие вызовы `perception_cycle(image)` работают без изменений (positional argument).

---

### Задача 3.2 — `src/snks/experiments/exp12_crossmodal.py`

```python
# Датасет: 50 пар (image_path, caption) — можно использовать MNIST + ручные подписи
#   или простые пары: shapes (circle, square, triangle) + текстовые описания
# Протокол:
#   1. Обучение: 50 циклов с perception_cycle(image=img, text=caption)
#   2. Запомнить: какие DAF-узлы активировались при visual input
#      (visual_nodes = set узлов из CycleResult при image-only цикле)
#   3. Тест: perception_cycle(text=caption) только
#      → измерить активацию visual_nodes
#   4. Контроль: perception_cycle(text=random_text) × 50
#      → измерить активацию visual_nodes
# cross_activation_ratio = mean(test) / mean(control)
# Gate: > 2.0
```

---

**Контрольная точка Фазы 3:**
```
pytest tests/ -x -q                             # все тесты PASS, включая regression
python -m snks.experiments.exp12_crossmodal     # ratio > 2.0
```

---

## Фаза 4: Текстовая генерация (Exp 14)

### Задача 4.1 — `src/snks/experiments/exp14_generation.py`

```python
# 1. Загрузить eval set (200 предложений из 10K held-out Exp 13)
# 2. Для каждого предложения:
#    a. perception_cycle(text=sentence) → получить активную СКС
#    b. Вычислить HAC-вектор: bundle активных узлов
#       (используем DcamWorldModel._hac.bundle(active_node_vectors))
#    c. DCAM.query_similar(hac_vector, top_k=1)
#    d. retrieved_episode.metadata["text"] == original_sentence?
# 3. recall@1 = доля правильных восстановлений
# Gate: > 0.5
```

**Детали:** при `store_episode()` в `ingest_corpus.py` передавать `metadata={"text": sentence}`.

---

**Контрольная точка Фазы 4:**
```
python -m snks.experiments.exp14_generation     # recall@1 > 0.5
```

---

## Финал: Проверка и коммит

```bash
# Все тесты
pytest tests/ -v -q

# Все эксперименты
python -m snks.experiments.exp10_text_sks      # NMI > 0.6
python -m snks.experiments.exp11_semantic      # ρ > 0.7
python -m snks.experiments.exp12_crossmodal    # ratio > 2.0
python -m snks.experiments.exp13_corpus        # precision@5 > 0.7  (mini-PC)
python -m snks.experiments.exp14_generation    # recall@1 > 0.5     (mini-PC)

# Обновить SPEC.md: Stage 7 → ✅
# git commit
```

---

## Сводка

| Фаза | Задачи | Запуск | Файлов |
|------|--------|--------|--------|
| 1: TextEncoder | 1.1–1.5 | Локально | 3 новых |
| 2: Корпус | 2.1–2.2 | mini-PC | 2 новых |
| 3: Кросс-модальность | 3.1–3.2 | Локально | 1 изменён, 1 новый |
| 4: Генерация | 4.1 | mini-PC | 1 новый |

**Порядок важен:** Фаза 1 → 2 → 3 → 4. Фазы 2 и 3 независимы (можно параллельно после Фазы 1).
