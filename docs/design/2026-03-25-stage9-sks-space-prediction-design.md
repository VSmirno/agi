# Stage 9: SKS-Space Prediction Enhancement — Design Spec

**Дата:** 2026-03-25
**Версия:** 1.0
**Статус:** Approved
**Контекст:** JEPA-inspired extension. Основан на [`research/lecun_jepa_research.md`](../../../research/lecun_jepa_research.md)

---

## Цель

Заменить дискретный cluster-ID predictor непрерывным предсказанием в HAC-пространстве СКС.
Закрыть debt: Exp 16 (per-winner PE) и Exp 18 (BroadcastPolicy).

**Принцип переноса из JEPA:** архитектурные идеи — да, backpropagation — нет.

---

## Ключевые решения

| Решение | Выбор | Обоснование |
|---------|-------|-------------|
| SKS embedding | HAC bundle узлов-членов | Гарантия разнообразия, анти-коллапс встроен |
| Item memory | Фиксированные случайные векторы | HAC ортогональность в 2048D, не требует обучения |
| Предиктор | HAC ассоциативная память | STDP-совместим, нет backprop, bind/unbind |
| Per-winner PE | Косинусное расстояние в HAC-пространстве | Непрерывный PE, решает Exp 16 debt |
| BroadcastPolicy | Ток в winner_nodes при высоком confidence | Global ignition, решает Exp 18 debt |
| Архитектура | Параллельный трек (старый + новый предиктор) | Честное сравнение для Exp 20 |

---

## Новые модули

### `src/snks/sks/embedder.py` — SKSEmbedder

```python
class SKSEmbedder:
    def __init__(self, n_nodes: int, hac_dim: int, device: str) -> None:
        # Инициализирует item_memory: (n_nodes, hac_dim) — фиксированные случайные
        # единичные векторы. Никогда не обновляются.
        self._item_memory: Tensor  # (n_nodes, hac_dim)

    def embed(self, sks_clusters: dict[int, set[int]]) -> dict[int, Tensor]:
        """Для каждого СКС: bundle HAC-векторов узлов → единичный вектор (hac_dim,).

        bundle = normalize(sum(item_memory[nodes]))
        """
```

**Инвариант:** item_memory создаётся один раз, не изменяется. Разнообразие embedding гарантировано математикой HAC.

### `src/snks/daf/hac_prediction.py` — HACPredictionEngine

```python
class HACPredictionEngine:
    def __init__(self, hac: HACEngine, config: HACPredictionConfig) -> None:
        self._memory: Tensor | None          # bundle пар bind(e_t, e_{t+1})
        self._prev_embeddings: dict[int, Tensor] | None

    def observe(self, embeddings: dict[int, Tensor]) -> None:
        """Обновить ассоциативную память.

        Для каждого sks_id в embeddings:
            new_pair = hac.bind(prev_embed, curr_embed)
            memory = normalize(memory * decay + new_pair)
        Обновить _prev_embeddings.
        """

    def predict_next(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        """Предсказать следующий aggregate embedding.

        aggregate = hac.bundle(list(embeddings.values()))
        predicted = normalize(hac.unbind(aggregate, memory))
        Возвращает (hac_dim,) или None если память пуста.
        """

    def compute_winner_pe(self, predicted: Tensor, actual_winner_embed: Tensor) -> float:
        """PE = (1 - cosine(predicted, actual)) / 2  ∈ [0, 1]."""
```

---

## Изменённые модули

### `src/snks/metacog/policies.py` — BroadcastPolicy

```python
class BroadcastPolicy:
    """При confidence >= threshold инжектирует ток в winner_nodes следующего цикла.

    Реализует global ignition: broadcast победителя GWS в сеть.
    """
    def __init__(self, strength: float = 1.0, threshold: float = 0.6) -> None: ...

    def apply(self, state: MetacogState, config: DafConfig) -> None:
        # Запоминает winner_nodes если confidence >= threshold
        ...

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        # Возвращает (n_nodes,) ток — strength на winner_nodes, 0 везде
        # Сбрасывает _pending после вызова
        ...
```

**Расширение базового интерфейса:** `NullPolicy`, `NoisePolicy`, `STDPPolicy` получают дефолтный метод `get_broadcast_currents() → None`.

### `src/snks/metacog/monitor.py` — MetacogState

```python
@dataclass
class MetacogState:
    confidence: float         # без изменений
    dominance: float          # без изменений
    stability: float          # без изменений
    pred_error: float         # без изменений (глобальный)
    winner_pe: float = 0.0    # НОВОЕ: HAC PE для победителя GWS ∈ [0, 1]
    winner_nodes: set[int] = field(default_factory=set)  # НОВОЕ: для BroadcastPolicy
```

**Логика confidence в `MetacogMonitor.update()`:**

`_CycleResultProxy` расширяется полем `winner_pe: float`. Монитор выбирает источник PE явно:

```python
# Если winner_pe > 0 — HACPredictionEngine активен и вернул результат
# Если winner_pe == 0 — fallback на глобальный pred_error_norm (как раньше)
pe_for_confidence = winner_pe if winner_pe > 0.0 else pred_error_norm
confidence = alpha * dominance + beta * stability + gamma * (1 - pe_for_confidence)
```

Это явная логика переключения, не зависящая от конфига — только от наличия результата.

### `src/snks/daf/types.py` — новые конфиги

```python
@dataclass
class SKSEmbedConfig:
    hac_dim: int = 2048     # совпадает с DcamConfig.hac_dim

@dataclass
class HACPredictionConfig:
    memory_decay: float = 0.95   # затухание памяти (вытеснение старых пар)
    enabled: bool = True

# PipelineConfig расширяется:
@dataclass
class PipelineConfig:
    # ... существующие поля ...
    sks_embed: SKSEmbedConfig = field(default_factory=SKSEmbedConfig)
    hac_prediction: HACPredictionConfig = field(default_factory=HACPredictionConfig)
```

### `src/snks/pipeline/runner.py` — CycleResult + поток

```python
@dataclass
class CycleResult:
    # Существующие поля — без изменений (backward compatible)
    sks_clusters: dict[int, set[int]]
    n_sks: int
    mean_prediction_error: float
    n_spikes: int
    cycle_time_ms: float
    gws: GWSState | None = None
    metacog: MetacogState | None = None
    # Новые поля (опциональны)
    winner_pe: float = 0.0
    winner_embedding: Tensor | None = None
    hac_predicted: Tensor | None = None
```

**Новый порядок шагов `perception_cycle()`:**

```
0. Reset states
1. Encode → currents
1b. Dual injection (motor)
1c. [NEW] Inject broadcast currents (self._broadcast_currents из предыдущего цикла)
2. Step DAF
3. Detect SKS
4. Track
4b. [NEW] SKSEmbedder.embed(tracked) → embeddings: dict[int, Tensor]
5.  PredictionEngine.predict/observe/compute_pe  (старый, без изменений) → mean_pe
5b. [NEW] HACPredictionEngine.predict_next(embeddings) → hac_predicted
          HACPredictionEngine.observe(embeddings)
          # winner_pe вычисляется ПОСЛЕ шага 7 (см. шаг 7b)
6.  STDP modulation
7.  GWS.select_winner(tracked) → gws_state (winner_id теперь известен)
7b. [NEW] если hac_predicted is not None и gws_state is not None:
              winner_pe = compute_winner_pe(hac_predicted, embeddings[winner_id])
          иначе: winner_pe = 0.0
8.  MetacogMonitor.update(gws_state, proxy(mean_pe, winner_pe)) → metacog_state
    MetacogMonitor.apply_policy(metacog_state, config)
8b. [NEW] policy.get_broadcast_currents(n_nodes) → self._broadcast_currents
          (применится в шаге 1c следующего цикла — однотактовый сдвиг, намеренно)
9.  Return CycleResult (расширенный)
```

> **Заметка о broadcast-сдвиге:** BroadcastPolicy инжектирует ток на **следующем** цикле (шаг 1c). Это семантически корректно: победитель текущего цикла "поджигает" сеть в следующем. Однотактовый сдвиг не нарушает global ignition — нейробиологический аналог (reentrant signaling) также не мгновенный.

**`Pipeline.__init__` добавляет:**

```python
self.embedder = SKSEmbedder(
    n_nodes=config.daf.num_nodes,
    hac_dim=config.dcam.hac_dim,
    device=config.device,
)
self.hac_prediction = HACPredictionEngine(
    hac=self.dcam.hac,  # переиспользуем существующий HACEngine из DCAM
    config=config.hac_prediction,
)
self._broadcast_currents: Tensor | None = None
```

---

## Эксперименты

### Exp 16 (debt): Confidence Ratio Gate

**Метрика:** `confidence(focused) / confidence(noise) > 1.5`

**Как тестировать:**
1. Прогнать pipeline на focused стимулах (чёткие паттерны) → собрать `winner_pe_focused`
2. Прогнать на noise стимулах → собрать `winner_pe_noise`
3. `confidence(focused)` вычисляется с `winner_pe_focused` (низкий PE → высокий confidence)
4. `confidence(noise)` вычисляется с `winner_pe_noise` (высокий PE → низкий confidence)
5. Проверить ratio > 1.5

**Почему теперь достижимо:** `winner_pe` — непрерывное расстояние в семантическом пространстве. Focused стимул → предсказуемый winner → малый `winner_pe`. Noise → непредсказуемый winner → большой `winner_pe`.

---

### Exp 18 (debt): BroadcastPolicy

**Метрика:** `cross_activation_ratio = mean_firing_rate(non_winner_after) / mean_firing_rate(non_winner_before) > 1.2`

**Как тестировать:**
1. Включить `BroadcastPolicy(strength=1.0, threshold=0.6)`
2. После цикла с высоким confidence: зафиксировать `fired_history` до и после broadcast
3. Вычислить `cross_activation_ratio` для non-winner узлов
4. Проверить ratio > 1.2

---

### Exp 19: SKS Embedding Quality

**Метрика:** NMI между nearest-neighbors в HAC-пространстве и ground truth классами > 0.7

**Как тестировать:**
1. Прогнать pipeline на labeled датасете (MNIST или shapes)
2. Для каждого стимула получить `winner_embedding`
3. Для каждого embedding найти K ближайших соседей по cosine similarity
4. Вычислить NMI: предсказанные кластеры (NN-граф) vs true labels
5. Проверить NMI > 0.7

---

### Exp 20: HAC vs Discrete Predictor

**Метрика:** accuracy HAC predictor ≥ accuracy discrete predictor (baseline: 72.9% из Exp 3)

**Как тестировать:**

Обе метрики унифицированы как **top-1 nearest-neighbor accuracy**:
- *HAC predictor:* `argmax_k cosine(hac_predicted, embed_k)` → predicted cluster label. Верно если совпадает с actual winner label.
- *Discrete predictor:* predicted cluster IDs → выбрать ID с наибольшим confidence → сравнить с actual winner ID.

Одна и та же схема: предсказываем один winner, проверяем совпадение. Baseline 72.9% (Exp 3) применяется к обеим.

---

## Граф зависимостей экспериментов

```
                    ┌── требует SKSEmbedder ──┐
                    │                         │
              Exp 19                       Exp 20
         (embedding quality)          (HAC predictor)
                    │                         │
                    └─── оба требуют ──────────┘
                              │
                         winner_pe
                         (из шага 7b)
                              │
                    ┌─────────┴──────────┐
                    │                    │
                 Exp 16               Exp 18
             (ratio gate)          (broadcast)
            требует winner_pe    требует winner_nodes
                                  в MetacogState
```

**Рекомендуемый порядок:** Exp 19 → Exp 20 → Exp 16 → Exp 18

Exp 19 и Exp 20 независимы друг от друга, но оба требуют `SKSEmbedder`. Exp 16 и Exp 18 независимы, но оба требуют изменений в `MetacogState`.

---

## Риски

| Риск | Вероятность | Митигация |
|------|-------------|-----------|
| HAC capacity limit (~45 пар) при большом числе СКС | Низкая (обычно 3–10 СКС) | memory_decay вытесняет старые пары |
| Degeneracy мета-уровня (все embeddings похожи) | Низкая (HAC ортогональность) | Exp 19 явно проверяет NMI |
| BroadcastPolicy дестабилизирует сеть | Средняя | threshold=0.6 — только при высоком confidence |
| winner_pe = 0 при пустой памяти (первые циклы) | Высокая | Fallback на глобальный pred_error_norm |

---

## Что НЕ делаем в Stage 9

- Backpropagation в любом виде
- ViT/Transformer encoder
- VICReg / EMA для target encoder
- Удаление старого `PredictionEngine` (остаётся до Exp 20)
- Иерархия предсказания (Stage 10)
