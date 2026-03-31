# СНКС MVP — Спецификация

**Версия:** 0.9.0
**Дата:** 2026-03-31
**Статус:** Этапы 0–17 MVP COMPLETE. Stages 19–23 (Language Grounding, Compositional, Verbalization, QA, Scaffold Removal) COMPLETE.

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
- Мышление = оперирование концептами (СКС), НЕ словами
- Язык = интерфейс ввода-вывода к концептному пространству, НЕ основа мышления

### Принцип: Концепты первичны, язык вторичен

СКС — это концепт (смысл), сформированный из опыта, а не из языка. Слово — это **якорь** (grounding label), привязанный к уже существующей концептной СКС. World model оперирует исключительно концептами. Язык — одна из модальностей доступа к концептному пространству, наравне с визуальной и моторной.

```
Концептный уровень (World Model):
    СКС_A [концепт] ──причина──► СКС_B [концепт]
       ▲         ▲                    ▲         ▲
       │         │                    │         │
    🖼️ образ  📝 слово            🖼️ образ  📝 слово
    🎮 действие                    🎮 действие
```

Следствия:
1. Слово НЕ создаёт новую СКС — оно привязывается к существующей
2. World model НЕ хранит слова — она хранит концепты и их связи
3. Генерация текста = обход концептного графа + линеаризация через grounding map
4. Понимание текста = активация концептных СКС через языковые якоря

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
| 7 | Каузальное обучение | Precision/Recall | > 0.8 / 0.7 | **1.000 / 0.750** | ✅ PASS |
| 8 | Ментальная симуляция | Sim accuracy / Planning | > 0.7 / 0.5 | **0.750 / 0.850** | ✅ PASS |
| 9 | Curiosity exploration | Coverage ratio | > 1.2× random | **1.258** | ✅ PASS |
| 10–14 | Текстовая модальность (Stage 7) | SKS NMI / cross-modal / corpus | см. stage7 | — | ✅ PASS |
| 15 | ГРП-победитель | mean_stability | > 0.7 | **0.837** | ✅ PASS |
| 16 | Calibration viability | mean_confidence(focused) | >= 0.5 | **0.984** | ✅ PASS |
| 17 | NoisePolicy адаптация | NMI(noise)≥NMI(null)−0.05 AND std↓ | оба условия | **PASS** | ✅ PASS |
| 21 | Meta-embedding stability | intra_sim > inter_sim + 0.05 | > 0.05 | **intra=0.846, inter=0.008, margin=0.839** | ✅ PASS |
| 22 | Hierarchical prediction | L2_acc > L1_acc + 0.05 | > +0.05 | **L1=0.0, L2=1.0, margin=1.0** | ✅ PASS |
| 23 | Multi-future planning | stoch_rate > det_rate × 1.2 | > 1.2× | **stoch=1.0, det=0.09, ratio=11.1** | ✅ PASS |
| 24 | Stochastic robustness | success[N=16] > success[N=1] | monotone ↑ | **N=1→73%, N=16→100%, monotone=4/4** | ✅ PASS |
| 25 | Cost-driven exploration | icm_cov / random_cov ≥ 1.0 | ≥ 1.0 | **icm=0.49, random=0.49, ratio=1.0** | ✅ PASS |
| 26 | Goal-directed navigation | goal_success_rate > 0.7 | > 0.7 | **goal=1.0, baseline=0.2** | ✅ PASS |
| 27 | Adaptive configuration | adaptive_score > fixed × 1.1 | > 1.1× | **adaptive=0.589, fixed=0.500, ratio=1.178** | ✅ PASS |
| 28 | Context switching | cycles_to_switch ≤ 20 | ≤ 20 | **cycles=5, CONSOLIDATE→EXPLORE ✓** | ✅ PASS |
| 29 | EmbodiedAgent integration | coverage ≥ 0.30 | ≥ 0.30 | **coverage=0.362** | ✅ PASS |
| 30 | EmbodiedAgent ablation | все варианты coverage ≥ 0.25 | ≥ 0.25 | **min=0.351** | ✅ PASS |
| 31 | Scaling (miniPC) | steps_per_sec ≥ 9 | ≥ 9 | **steps_per_sec=9.41 (CPU, N=5K)** | ✅ PASS |
| 44 | Cross-modal recall | cross_modal_ratio | > 2.0 | **A=178, B=116742** | ✅ PASS |
| 45 | Grounding speed | reps to ratio>2.0 | < 20 | **1 rep** | ✅ PASS |
| 46 | Role extraction | accuracy | > 0.8 | **1.000 (64/64)** | ✅ PASS |
| 47 | Compositional generalization | unbind accuracy (novel) | > 0.7 | **1.000 (30/30)** | ✅ PASS |
| 49 | Factual QA | accuracy | > 0.7 | **0.750 (15/20)** | ✅ PASS |
| 50 | Simulation QA | accuracy | > 0.6 | **1.000 (15/15)** | ✅ PASS |
| 51 | Reflective QA | accuracy | > 0.6 | **0.867 (13/15)** | ✅ PASS |
| 52 | Autonomous cross-modal recall | sdr/currents match | > 0.8 | **1.000 (15/15)** | ✅ PASS |
| 53 | Autonomous QA | accuracy без ST | > 0.6 | **1.000 (15/15)** | ✅ PASS |
| 54a | Instruction parsing | accuracy (5 types) | > 0.9 | **1.000 (30/30)** | ✅ PASS |

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
| 6 | Каузальный агент | ✅ | [specs/stage6.md](specs/stage6.md) |
| **7** | **Текстовая модальность** | ✅ | [specs/2026-03-24-stage7-text-modality-design.md](superpowers/specs/2026-03-24-stage7-text-modality-design.md) |
| **8** | **ГРП + Метакогниция** | ✅ | [specs/2026-03-24-stage8-gws-metacog-design.md](superpowers/specs/2026-03-24-stage8-gws-metacog-design.md) |
| **9** | **SKS-Space Prediction** | ✅ | [specs/2026-03-25-stage9-sks-space-prediction-design.md](superpowers/specs/2026-03-25-stage9-sks-space-prediction-design.md) |
| **10** | **Hierarchical Prediction** | ✅ | [docs/superpowers/specs/2026-03-25-stages10-13-architecture.md](docs/superpowers/specs/2026-03-25-stages10-13-architecture.md) |
| **11** | **Multi-Future Simulation** | ✅ | [docs/superpowers/specs/2026-03-25-stages10-13-architecture.md](docs/superpowers/specs/2026-03-25-stages10-13-architecture.md) |
| **12** | **Intrinsic Cost Module** | ✅ | [docs/superpowers/specs/2026-03-25-stages10-13-architecture.md](docs/superpowers/specs/2026-03-25-stages10-13-architecture.md) |
| **13** | **Configurator / Meta-Control** | ✅ | [docs/superpowers/specs/2026-03-25-stages10-13-architecture.md](docs/superpowers/specs/2026-03-25-stages10-13-architecture.md) |
| **14** | **EmbodiedAgent** | ✅ | [specs/2026-03-25-stage14-embodied-agent.md](specs/2026-03-25-stage14-embodied-agent.md) |
| **15** | **Закрытие долгов** | ✅ | [specs/stage15-debt-closure.md](specs/stage15-debt-closure.md) |
| **16** | **DCAM Tiered Consolidation** | ✅ | [docs/superpowers/specs/2026-03-26-stage16-dcam-consolidation-design.md](docs/superpowers/specs/2026-03-26-stage16-dcam-consolidation-design.md) |
| **17** | **Full-Scale Validation** | 🔄 | [specs/stage17.md](specs/stage17.md) |

### Приоритеты (зафиксировано 2026-03-26)

- Этапы 0–9 завершены: 21 эксперимент PASS.
- Stage 9 (2026-03-25): SKSEmbedder + HACPredictionEngine + BroadcastPolicy + per-winner PE.
  Exp 16b ratio=1.993 ✅, Exp 18 ratio=2.846 ✅, Exp 19 NMI=1.000 ✅, Exp 20 HAC=66.7% ✅.
- Stages 10–13 (2026-03-25): код реализован, 38 unit tests PASS (RTX + AMD ROCm). Эксперименты 21–28 PASS (2026-03-25).
  Exp 21 margin=0.839 ✅, Exp 22 L2=1.0 ✅, Exp 23 ratio=11.1 ✅, Exp 24 N=16→100% ✅,
  Exp 25 ratio=1.0 ✅, Exp 26 goal=1.0 ✅, Exp 27 ratio=1.178 ✅, Exp 28 cycles=5 ✅.
- Stage 14 (2026-03-25): EmbodiedAgent реализован — тонкий оркестровый слой: CausalAgent +
  StochasticSimulator + Configurator FSM → полный агентный цикл в MiniGrid KeyDoor.
  Exp 29 coverage=0.362 ✅, Exp 30 min_cov=0.351 ✅, Exp 31 steps/s=9.41 (CPU, N=5K) ✅.
- Stage 15 (2026-03-26): COMPLETE. 5 критических долгов закрыты. Исправления:
  _split_context (stable perceptual hash IDs), EpisodicHACPredictor (K-pair deque),
  causal_context_bins=64, plan_min_confidence configurable, DCAM minimal integration.
  Exp 32 GOAL_SEEKING=4980 steps ✅, Exp 33 episodic_pe=0.488<bundle_pe=0.500 ✅,
  Exp 34 NMI retention=100% (N=5K) ✅. Спецификация: specs/stage15-debt-closure.md.
- Stage 16 (2026-03-26): COMPLETE. DCAM Tiered Memory Consolidation.
  AgentTransitionBuffer, ConsolidationScheduler, TieredPlanner, ReplayEngine.
  Exp 35 cold_overrides=4000 ✅, Exp 36 sim=1.0 edges=4 ✅, Exp 37 stdp_updates=1 ✅.
- Stage 17 (2026-03-29): COMPLETE. Full-Scale Validation.
  Exp 38 N=50K 16.47 steps/sec ✅, Exp 39 replay+coverage (N=5000, uniform) ✅, Exp 40 goal_seeking=49028 ✅.
  Key findings: replay requires N≥5000 + uniform sampling (importance toxic for dangerous envs).
- Stage 18 (2026-03-30): IN PROGRESS. Multi-Env Validation + Transfer Learning.
  Exp 41/42/43 реализованы, ночной запуск на minipc.
- Stage 19 (2026-03-30): COMPLETE. Zonal DAF + Language Grounding + Complementary Priming.
  Zonal architecture (visual/linguistic/convergence zones), GroundingMap, cross-modal priming.
  Key finding: pure STDP coupling insufficient for cross-modal recall at any N (5K-50K).
  Complementary priming (top-down visual SDR injection at 0.3× strength) solves it.
  Exp 44 Config A ratio=178, Config B ratio=116742 ✅. Exp 45 grounding in 1 rep ✅.
  Chosen: Config B (with convergence zone). Spec: specs/stages19-24-language-grounding-design.md.
- Stage 20 (2026-03-30): COMPLETE. Композиционное понимание (HAC Role-Filler).
  Ролевая система (6 ролей в HAC-пространстве), rule-based SVO(L) chunker (3 паттерна),
  RoleFillerParser (bind/unbind), EmbeddingResolver (гибрид: GroundingMap cache + DAF fallback).
  Exp 46 accuracy=1.000 (64/64) ✅, Exp 47 test_acc=1.000 (30/30 novel combos) ✅.
- Stage 22 (2026-03-30): COMPLETE. Grounded QA (factual/simulation/reflective).
  QuestionClassifier (rule-based), GroundedQA orchestrator, pluggable QABackend protocol.
  Key finding: chunker designed for declarative sentences, questions need prefix-stripping + word-level resolve.
  Exp 49 accuracy=0.750 ✅, Exp 50 accuracy=1.000 ✅, Exp 51 accuracy=0.867 ✅.
- Stage 23 (2026-03-31): COMPLETE. Scaffold Removal (GroundedTokenizer).
  GroundedTokenizer replaces sentence-transformers for SDR generation via GroundingMap lookup.
  Exp 52 sdr_match=1.000 ✅, Exp 53 accuracy=1.000 ✅.
- Stage 24a (2026-03-31): COMPLETE. InstructionParser + Attribute Grounding.
  RuleBasedChunker extended: +sequential (SEQ_BREAK), +spatial patterns. 5 BabyAI levels.
  Exp 54a parsing accuracy=1.000 ✅, grounding resolve=1.000 ✅.

### Граф зависимостей

```
Этап 0 ✅ ──→ Этап 1 ✅ (DAF) ──→ Этап 3 ✅ (SKS + Experiments)
                   │                       │
                   │  Этап 2 ✅ (Encoder) ─┘
                   │                  │
                   └─→ Этап 4 ✅ (DCAM)─┴──→ Этап 5 ✅ (Integration)
                                                       │
                                                       └──→ Этап 6 ✅ (Causal Agent)
                                                                   │
                                                                   └──→ Этап 7 ✅ (Text Modality)
                                                                                   │
                                                                                   └──→ Этап 8 ✅ (GWS + Metacognition)
                                                                                                   │
                                                                                                   └──→ Этап 9 ✅ (SKS-Space Prediction)
                                                                                                                   │
                                                                                                     ┌─────────────┘
                                                                                                     ↓
                                                                                               Этап 10 ✅ (Hierarchical Prediction)
                                                                                                     │
                                                                                                     ↓
                                                                                               Этап 11 ✅ (Multi-Future Simulation)
                                                                                                     │
                                                                                                     ↓
                                                                                               Этап 12 ✅ (Intrinsic Cost Module)
                                                                                                     │
                                                                                                     ↓
                                                                                               Этап 13 ✅ (Configurator)
                                                                                                     │
                                                                                                     ↓
                                                                                               Этап 14 ✅ (EmbodiedAgent)
```

### Роадмап 9–13: JEPA-inspired extensions (зафиксировано 2026-03-25)

> Основан на анализе работ Яна ЛеКуна (JEPA). Подробности: [`research/lecun_jepa_research.md`](research/lecun_jepa_research.md)

**Принцип переноса:** архитектурные идеи JEPA переносятся, механизмы обучения — нет (backprop запрещён).

| Этап | Идея из JEPA | Реализация в СНКС |
|------|-------------|-------------------|
| 9 | Предсказание в continuous latent space | SKS embedding (векторы), continuous predictor |
| 10 | H-JEPA: иерархия временных масштабов | Мета-СКС, L1/L2 предикторы |
| 11 | Латентная переменная z, множество будущих | N стохастических траекторий, planning через cost |
| 12 | Unified Cost Module (LeCun) | homeostatic + epistemic + goal cost, energy landscape |
| 13 | Configurator (мета-модуль) | Расширение MetacogMonitor до dynamic reconfiguration |

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
├── dcam/                     # Этап 4 ✅
├── data/                     # stimuli, shapes, sequences
├── pipeline/                 # Pipeline runner
├── env/                      # Этап 6 ✅ — MiniGrid среда
├── agent/                    # Этап 6 ✅ — каузальный агент
├── gws/                      # Этап 8 ✅ — GlobalWorkspace, GWSState
├── metacog/                  # Этап 8 ✅ — MetacogMonitor, policies
└── experiments/              # exp1–exp17

tests/                        # 287 тестов, все проходят
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
