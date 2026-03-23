# СНКС MVP — Спецификация

**Версия:** 0.2.0
**Дата:** 2026-03-23
**Статус:** Все этапы (0–5) завершены — MVP готов

> Детальные спецификации этапов: [`specs/`](specs/)

---

## 1. Цель проекта

Минимальный proof-of-concept **СНКС** (Система Непрерывного Когнитивного Синтеза) — AGI-архитектура, принципиально отличающаяся от LLM/Transformer/RL.

### Что доказывает MVP

1. **ДАП** формируют устойчивые представления (СКС) из визуальных данных — без учителя
2. **Локальные правила** (STDP + гомеостаз) достаточны для самоорганизации — без backpropagation
3. **Непрерывное обучение** без catastrophic forgetting
4. **DCAM-формат** пригоден для хранения и восстановления модели мира

### Философия

- Вычисления = эволюция динамической системы к аттракторам
- Знания = устойчивые паттерны осцилляций (СКС)
- Обучение = локальная модификация связей (STDP)
- Память = двухкодовое хранение (HAC + SSG)

---

## 2. Аппаратные требования

| Машина | GPU | VRAM | Роль |
|--------|-----|------|------|
| Mini-PC | AMD (ROCm) | 92 GB | Полные эксперименты |
| Рабочая станция | NVIDIA RTX 3090 | 24 GB | Разработка |
| Любая | — | — | CPU-режим, тесты |

Единый код для всех сценариев. Различие только в конфигах.

---

## 3. Стек

| Слой | Технология |
|------|-----------|
| Язык | Python 3.11+ |
| Вычисления | PyTorch 2.x (CUDA / ROCm / CPU) |
| Sparse | torch.scatter_add, torch.sparse |
| FFT | torch.fft (HAC bind/unbind) |
| Кластеризация | scikit-learn (DBSCAN, KMeans) |
| Фильтры | scikit-image (Gabor kernels) |
| Сериализация | safetensors + JSON |
| Конфиги | YAML |
| Визуализация | FastAPI + WebSocket + D3.js |
| Тесты | pytest |

**Не используем:** Rust, custom CUDA kernels, torch.compile.

---

## 4. Архитектура

```
┌──────────────────────────────────────────────────────┐
│                      Pipeline                         │
│                                                       │
│   Image 64×64 ──→ [Visual Encoder] ──→ SDR 4096 бит  │
│                                           │           │
│                                           ▼           │
│                    ┌────────────────────────┐         │
│                    │      DAF Engine         │         │
│                    │  FHN oscillators (10K+) │         │
│                    │  STDP + homeostasis     │         │
│                    │  Prediction error mod.  │         │
│                    └──────────┬─────────────┘         │
│                               │                       │
│                               ▼                       │
│                    ┌────────────────────┐              │
│                    │  SKS Detection     │              │
│                    │  Rate-based + KMeans│              │
│                    └────────┬───────────┘              │
│                             │                         │
│                             ▼                         │
│                    ┌────────────────────┐              │
│                    │  DCAM Storage      │              │
│                    │  HAC + SSG + LSH   │              │
│                    └───────────────────┘              │
└──────────────────────────────────────────────────────┘
```

### Компоненты (кратко)

| # | Компонент | Вход | Выход | Файл |
|---|-----------|------|-------|------|
| 1 | Visual Encoder | image 64×64 | SDR 4096 бит (4% sparse) | `encoder/encoder.py` |
| 2 | DAF Engine | SDR currents | states (N,8) + fired_history | `daf/engine.py` |
| 3 | STDP | fired_history | weight updates | `daf/stdp.py` |
| 4 | SKS Detection | states/fired | clusters {id: nodes} | `sks/detection.py` |
| 5 | Prediction | SKS activations | predicted next, PE | `daf/prediction.py` |
| 6 | DCAM | SKS + context | world model | `dcam/world_model.py` |
| 7 | Pipeline | image stream | CycleResult | `pipeline/runner.py` |

---

## 5. Контракты (интерфейсы)

### VisualEncoder
```python
class VisualEncoder:
    def encode(self, image: Tensor) -> Tensor:        # (64,64) → (4096,) binary
    def sdr_to_currents(self, sdr: Tensor, num_nodes: int) -> Tensor:  # → (N,8)
```

### DafEngine
```python
class DafEngine:
    states: Tensor          # (N, 8)
    graph: SparseDafGraph   # COO (2,E) + attr (E,4)
    def set_input(self, currents: Tensor) -> None:
    def step(self, n_steps: int) -> StepResult:
    def get_fired_history(self) -> Tensor | None:  # (T,N) bool
```

### SKS Detection
```python
def phase_coherence_matrix(states, top_k) -> tuple[Tensor, Tensor]:
def cofiring_coherence_matrix(fired_history, top_k) -> tuple[Tensor, Tensor]:
def detect_sks(coherence, eps, min_samples, min_size) -> list[set[int]]:
```

### Pipeline
```python
class Pipeline:
    def perception_cycle(self, image: Tensor) -> CycleResult:
    def train_on_dataset(self, images: Tensor, labels: Tensor, epochs: int) -> TrainResult:
```

### DCAM (Этап 4 — контракт)
```python
class HACEngine:
    def bind(self, a: Tensor, b: Tensor) -> Tensor:       # circular convolution
    def unbind(self, a: Tensor, bound: Tensor) -> Tensor:  # circular correlation
    def bundle(self, vectors: list[Tensor]) -> Tensor:     # sum + normalize
    def similarity(self, a: Tensor, b: Tensor) -> float:   # cosine

class DcamWorldModel:
    def store_episode(self, active_nodes, context, importance) -> int:
    def query_similar(self, query_hac, top_k) -> list[tuple[int, float]]:
    def consolidate(self) -> ConsolidationReport:
    def save(self, path: str) -> None:
    def load(self, path: str) -> None:
```

---

## 6. Конфигурации

| Параметр | small (dev) | default (50K) | full (AMD) |
|----------|-------------|---------------|------------|
| num_nodes | 10,000 | 50,000 | 500,000 |
| avg_degree | 30 | 50 | 50 |
| oscillator_model | fhn | fhn | fhn |
| hac_dim | 2048 | 2048 | 4096 |
| steps_per_cycle | 200 | 200 | 200 |
| VRAM (est.) | ~2 GB | ~8 GB | ~60 GB |

---

## 7. Эксперименты и результаты

### Сводка

| # | Эксперимент | Метрика | Порог | Результат | Статус |
|---|-------------|---------|-------|-----------|--------|
| 1 | Формирование СКС | NMI | > 0.7 | **0.823** | ✅ PASS |
| 2 | Непрерывное обучение | Retention | > 85% | **103.3%** | ✅ PASS |
| 3 | Предсказание последовательностей | Mean accuracy | > 70% | **72.9%** (L3:64%, L5:75%, L7:79%) | ✅ PASS |
| 4 | Устойчивость к шуму | Graceful degradation | Нет обрыва | **0% drop** (NMI=0.77 на всех σ) | ✅ PASS |
| 5 | Персистентность (DCAM) | Δ accuracy | ≤ 1% | **Δ 0.0%** | ✅ PASS |
| 6 | MNIST unsupervised | NMI | > 0.6 | **0.609** | ✅ PASS |

### Критерии MVP

| Приоритет | Критерий | Статус |
|-----------|----------|--------|
| **Must** | СКС формируются (NMI > 0.7) | ✅ |
| **Must** | Нет catastrophic forgetting (> 80%) | ✅ |
| **Must** | Без backpropagation | ✅ |
| Should | Предсказание (> 70%) | ✅ |
| Should | Персистентность (Δ ≤ 1%) | ✅ |
| Should | GPU perf (≥ 10K steps/sec) | ✅ |
| Nice | MNIST unsupervised (NMI > 0.6) | ✅ |
| Nice | Real-time dashboard (≥ 5 FPS) | ✅ |

---

## 8. Этапы реализации

| Этап | Название | Статус | Детали |
|------|----------|--------|--------|
| 0 | Инфраструктура | ✅ | [specs/stage0.md](specs/stage0.md) |
| 1 | ДАП-Движок | ✅ | [specs/stage1.md](specs/stage1.md) |
| 2 | Визуальный кодировщик | ✅ | [specs/stage2.md](specs/stage2.md) |
| 3 | СКС + Эксперименты 1–4 | ✅ | [specs/stage3.md](specs/stage3.md) |
| 4 | DCAM Хранилище | ✅ | [specs/stage4.md](specs/stage4.md) |
| 5 | Визуализация + Интеграция | ✅ | [specs/stage5.md](specs/stage5.md) |

### Граф зависимостей

```
Этап 0 ✅ ──→ Этап 1 ✅ (DAF) ──→ Этап 3 ✅ (SKS + Experiments)
                   │                       │
                   │  Этап 2 ✅ (Encoder) ─┘
                   │                  │
                   └─→ Этап 4 ✅ (DCAM)─┴──→ Этап 5 ✅ (Integration)
```

---

## 9. Структура проекта

```
src/snks/
├── device.py                 # GPU auto-detection
├── daf/                      # Этап 1 ✅ + Этап 3 ✅
│   ├── types.py              # Все конфиги (DafConfig, SKSConfig, PipelineConfig, ...)
│   ├── graph.py              # SparseDafGraph (COO)
│   ├── oscillator.py         # FHN / Kuramoto
│   ├── coupling.py           # scatter_add
│   ├── integrator.py         # Euler-Maruyama
│   ├── stdp.py               # STDP + homeostasis
│   ├── homeostasis.py        # threshold adaptation
│   ├── structural.py         # edge add/remove
│   ├── prediction.py         # PredictionEngine
│   └── engine.py             # DafEngine facade
├── encoder/                  # Этап 2 ✅
│   ├── gabor.py              # GaborBank (128 filters)
│   ├── sdr.py                # kwta, overlap
│   └── encoder.py            # VisualEncoder
├── sks/                      # Этап 3 ✅
│   ├── detection.py          # coherence + DBSCAN
│   ├── tracking.py           # SKSTracker
│   └── metrics.py            # NMI, stability, separability
├── dcam/                     # Этап 4 ⏳
├── data/                     # stimuli, shapes, sequences
├── pipeline/                 # Pipeline runner
└── experiments/              # exp1–exp5

tests/                        # 248 тестов, все проходят
configs/                      # small.yaml, default.yaml, full.yaml
```

---

## 10. Ключевые архитектурные решения

| Решение | Причина |
|---------|---------|
| FHN вместо Kuramoto для экспериментов | Kuramoto на случайном графе не формирует кластеры через STDP |
| dt=0.01, fhn_I_base=0.0 | Стабильность + нейроны молчат без стимула |
| State reset между стимулами | Без него state carryover разрушает паттерны |
| Rate-based detection | O(N) vs O(K²), threshold = mean + 3σ |
| k-means на firing rate vectors для NMI | Вместо tracker-based assignment |
| Hash-based SDR→node mapping | Multiplicative hash для равномерного покрытия |
| GratingGenerator для gate (не ShapeGenerator) | Gabor — ориентационные детекторы V1 |

---

## 11. Риски и митигации

| Риск | Вероятность | Импакт | Митигация | Статус |
|------|-------------|--------|-----------|--------|
| ДАП не сходится | Высокая | Критичный | Kuramoto → FHN | ✅ Решено |
| STDP не формирует СКС | Средняя | Критичный | Rate-based Hebbian + FHN | ✅ Решено |
| scatter_add bottleneck | Средняя | Серьёзный | CSR spmv + CUDA Graphs | ✅ Решено |
| Phase coherence O(N²) | Средняя | Серьёзный | Rate-based O(N) заменяет | ✅ Решено |
| ROCm несовместимость | Низкая | Серьёзный | Только стандартные torch ops | — |
