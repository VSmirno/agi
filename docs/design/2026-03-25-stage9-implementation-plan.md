# Stage 9: Implementation Plan

**Спека:** `2026-03-25-stage9-sks-space-prediction-design.md`
**Подход:** specification-driven — тесты перед реализацией
**Ветка:** `stage9-sks-space-prediction`

---

## Task 1: Новые конфиги в types.py

**Файл:** `src/snks/daf/types.py`

**Шаги:**
1. Добавить `SKSEmbedConfig(hac_dim: int = 2048)` после `SKSConfig`
2. Добавить `HACPredictionConfig(memory_decay: float = 0.95, enabled: bool = True)` после `PredictionConfig`
3. Расширить `PipelineConfig` полями `sks_embed` и `hac_prediction`

**Верификация:** `python -c "from snks.daf.types import SKSEmbedConfig, HACPredictionConfig, PipelineConfig; c = PipelineConfig(); print(c.sks_embed, c.hac_prediction)"`

---

## Task 2: SKSEmbedder — тесты

**Файл:** `tests/test_sks_embedder.py`

**Тесты:**
- `test_embed_returns_unit_vectors` — результат каждого embedding имеет норму ≈ 1.0
- `test_different_clusters_different_embeddings` — два разных кластера → cosine < 0.99
- `test_same_cluster_same_embedding` — тот же кластер → cosine ≈ 1.0 (детерминированность)
- `test_empty_clusters_returns_empty` — `{}` → `{}`
- `test_item_memory_not_updated` — после 10 вызовов `embed()` item_memory неизменна

**Верификация:** `venv/Scripts/pytest tests/test_sks_embedder.py -x` → все тесты FAIL (файла нет)

---

## Task 3: SKSEmbedder — реализация

**Файл:** `src/snks/sks/embedder.py`

```python
class SKSEmbedder:
    def __init__(self, n_nodes: int, hac_dim: int, device: str) -> None:
        # torch.manual_seed не трогаем — случайная, но стабильная при одном запуске
        vecs = torch.randn(n_nodes, hac_dim)
        norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._item_memory = (vecs / norms).to(device)  # (n_nodes, hac_dim), requires_grad=False

    def embed(self, sks_clusters: dict[int, set[int]]) -> dict[int, Tensor]:
        result = {}
        for sks_id, nodes in sks_clusters.items():
            idx = torch.tensor(sorted(nodes), dtype=torch.long, device=self._item_memory.device)
            vecs = self._item_memory[idx]          # (k, hac_dim)
            bundle = vecs.sum(dim=0)               # (hac_dim,)
            norm = bundle.norm().clamp(min=1e-8)
            result[sks_id] = bundle / norm
        return result
```

**Верификация:** `venv/Scripts/pytest tests/test_sks_embedder.py -x` → все PASS

---

## Task 4: HACPredictionEngine — тесты

**Файл:** `tests/test_hac_prediction.py`

**Тесты:**
- `test_predict_next_returns_none_before_memory` — до observe() → None
- `test_observe_builds_memory` — после одного observe() memory не None
- `test_predict_next_after_observe` — после observe(A), observe(B) → predict_next(A) возвращает вектор
- `test_repeated_ab_improves_similarity` — A→B 10 раз → cosine(predict(A), embed_B) > cosine после 1 раза
- `test_compute_winner_pe_identical` — cosine(v, v) = 1 → pe = 0.0
- `test_compute_winner_pe_orthogonal` — pe ≈ 0.5 для ортогональных векторов
- `test_memory_decay_reduces_old_associations` — после decay старые ассоциации слабее

**Верификация:** `venv/Scripts/pytest tests/test_hac_prediction.py -x` → все FAIL

---

## Task 5: HACPredictionEngine — реализация

**Файл:** `src/snks/daf/hac_prediction.py`

```python
class HACPredictionEngine:
    def __init__(self, hac: HACEngine, config: HACPredictionConfig) -> None:
        self.hac = hac
        self.config = config
        self._memory: Tensor | None = None
        self._prev_embeddings: dict[int, Tensor] | None = None

    def observe(self, embeddings: dict[int, Tensor]) -> None:
        if self._prev_embeddings is not None and embeddings:
            new_pairs = []
            for sks_id, curr in embeddings.items():
                if sks_id in self._prev_embeddings:
                    pair = self.hac.bind(self._prev_embeddings[sks_id], curr)
                    new_pairs.append(pair)
            if new_pairs:
                new_bundle = self.hac.bundle(new_pairs)
                if self._memory is None:
                    self._memory = new_bundle
                else:
                    # decay + add
                    combined = [self._memory * self.config.memory_decay, new_bundle]
                    self._memory = self.hac.bundle(combined)
        self._prev_embeddings = dict(embeddings)

    def predict_next(self, embeddings: dict[int, Tensor]) -> Tensor | None:
        if self._memory is None or not embeddings:
            return None
        aggregate = self.hac.bundle(list(embeddings.values()))
        predicted = self.hac.unbind(aggregate, self._memory)
        norm = predicted.norm().clamp(min=1e-8)
        return predicted / norm

    def compute_winner_pe(self, predicted: Tensor, actual_winner_embed: Tensor) -> float:
        cos = self.hac.similarity(predicted, actual_winner_embed)
        return float((1.0 - cos) / 2.0)  # ∈ [0, 1]
```

**Верификация:** `venv/Scripts/pytest tests/test_hac_prediction.py -x` → все PASS

---

## Task 6: BroadcastPolicy — тесты

**Файл:** `tests/test_broadcast_policy.py`

**Тесты:**
- `test_null_policy_get_broadcast_returns_none` — NullPolicy.get_broadcast_currents() → None
- `test_broadcast_below_threshold_no_currents` — confidence=0.3 < threshold=0.6 → None
- `test_broadcast_above_threshold_returns_currents` — confidence=0.8 → Tensor с ненулевыми winner_nodes
- `test_broadcast_strength_scales_current` — strength=2.0 → ток вдвое больше
- `test_broadcast_clears_after_get` — второй вызов get → None (one-shot)
- `test_winner_nodes_get_current` — только winner_nodes ненулевые

**Верификация:** `venv/Scripts/pytest tests/test_broadcast_policy.py -x` → FAIL

---

## Task 7: BroadcastPolicy — реализация

**Файл:** `src/snks/metacog/policies.py`

1. Добавить `get_broadcast_currents(self, n_nodes: int) -> Tensor | None` в `NullPolicy` (return None)
2. Добавить дефолтный метод в `NoisePolicy` и `STDPPolicy` (return None, наследуют от NullPolicy или явно)
3. Добавить `BroadcastPolicy`:

```python
class BroadcastPolicy:
    def __init__(self, strength: float = 1.0, threshold: float = 0.6) -> None:
        self.strength = strength
        self.threshold = threshold
        self._pending_nodes: set[int] | None = None

    def apply(self, state: MetacogState, config: DafConfig) -> None:
        if state.confidence >= self.threshold and state.winner_nodes:
            self._pending_nodes = set(state.winner_nodes)

    def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
        if self._pending_nodes is None:
            return None
        currents = torch.zeros(n_nodes)
        for node in self._pending_nodes:
            if node < n_nodes:
                currents[node] = self.strength
        self._pending_nodes = None
        return currents
```

4. Добавить `"broadcast"` в `MetacogMonitor.__init__` switch.

**Верификация:** `venv/Scripts/pytest tests/test_broadcast_policy.py -x` → PASS

---

## Task 8: MetacogState + MetacogMonitor — тесты

**Файл:** `tests/test_metacog_stage9.py`

**Тесты:**
- `test_metacog_state_has_winner_pe` — MetacogState имеет поле winner_pe
- `test_metacog_state_has_winner_nodes` — MetacogState имеет поле winner_nodes
- `test_monitor_uses_winner_pe_when_nonzero` — winner_pe=0.5 влияет на confidence
- `test_monitor_fallback_when_winner_pe_zero` — winner_pe=0.0 → использует pred_error_norm
- `test_monitor_passes_winner_nodes_to_state` — gws_state.winner_nodes попадает в MetacogState

**Верификация:** `venv/Scripts/pytest tests/test_metacog_stage9.py -x` → FAIL

---

## Task 9: MetacogState + MetacogMonitor — реализация

**Файл:** `src/snks/metacog/monitor.py`

1. Расширить `MetacogState`:
```python
winner_pe: float = 0.0
winner_nodes: set[int] = field(default_factory=set)
```

2. Расширить `_CycleResultProxy` полем `winner_pe: float = 0.0`

3. В `MetacogMonitor.update()`:
```python
winner_pe = getattr(cycle_result, 'winner_pe', 0.0)
pe_for_confidence = winner_pe if winner_pe > 0.0 else pred_error_norm
confidence = alpha * dominance + beta * stability + gamma * (1.0 - pe_for_confidence)
winner_nodes = set(gws_state.winner_nodes) if gws_state else set()
return MetacogState(..., winner_pe=winner_pe, winner_nodes=winner_nodes)
```

4. Добавить `get_broadcast_currents(self, n_nodes: int) -> Tensor | None` в `MetacogMonitor`:
```python
def get_broadcast_currents(self, n_nodes: int) -> Tensor | None:
    return self._policy.get_broadcast_currents(n_nodes)
```

**Верификация:** `venv/Scripts/pytest tests/test_metacog_stage9.py tests/test_metacog.py -x` → все PASS

---

## Task 10: Pipeline интеграция — тесты

**Файл:** `tests/test_pipeline_stage9.py`

**Тесты:**
- `test_cycle_result_has_winner_pe` — CycleResult.winner_pe float
- `test_cycle_result_has_winner_embedding` — CycleResult.winner_embedding
- `test_cycle_result_has_hac_predicted` — CycleResult.hac_predicted
- `test_winner_pe_zero_before_memory_built` — первый цикл → winner_pe = 0.0 (нет памяти)
- `test_winner_pe_nonzero_after_sequence` — после 3 циклов с одним стимулом → winner_pe > 0
- `test_broadcast_currents_injected_next_cycle` — BroadcastPolicy → второй цикл получает ток

**Верификация:** `venv/Scripts/pytest tests/test_pipeline_stage9.py -x` → FAIL

---

## Task 11: Pipeline — реализация

**Файл:** `src/snks/pipeline/runner.py`

1. Расширить `CycleResult`:
```python
winner_pe: float = 0.0
winner_embedding: Tensor | None = None
hac_predicted: Tensor | None = None
```

2. В `Pipeline.__init__` добавить:
```python
from snks.sks.embedder import SKSEmbedder
from snks.daf.hac_prediction import HACPredictionEngine
from snks.dcam.hac import HACEngine

self._hac = HACEngine(dim=config.dcam.hac_dim, device=resolve_device(config.device))
self.embedder = SKSEmbedder(config.daf.num_nodes, config.dcam.hac_dim, config.device)
self.hac_prediction = HACPredictionEngine(self._hac, config.hac_prediction)
self._broadcast_currents: Tensor | None = None
```

3. В `perception_cycle()`:

**Шаг 1c** (после dual injection):
```python
if self._broadcast_currents is not None:
    currents = currents + self._broadcast_currents.to(self.engine.device)
    self._broadcast_currents = None
```

**Шаг 4b** (после Track):
```python
embeddings = self.embedder.embed(tracked)
```

**Шаг 5b** (после старого predict):
```python
hac_predicted = self.hac_prediction.predict_next(embeddings)
self.hac_prediction.observe(embeddings)
# winner_pe вычисляется ПОСЛЕ шага 7
```

**Шаг 7b** (после GWS.select_winner):
```python
winner_pe = 0.0
winner_embedding = None
if hac_predicted is not None and gws_state is not None:
    winner_id = gws_state.winner_id
    if winner_id in embeddings:
        winner_embedding = embeddings[winner_id]
        winner_pe = self.hac_prediction.compute_winner_pe(hac_predicted, winner_embedding)
```

**Шаг 8** — передать winner_pe в proxy:
```python
metacog_state = self.metacog.update(gws_state, _CycleResultProxy(mean_pe, winner_pe))
```

**Шаг 8b**:
```python
broadcast = self.metacog.get_broadcast_currents(self.engine.config.num_nodes)
if broadcast is not None:
    self._broadcast_currents = broadcast
```

**Шаг 9** — расширить CycleResult:
```python
return CycleResult(
    ...,  # существующие поля
    winner_pe=winner_pe,
    winner_embedding=winner_embedding,
    hac_predicted=hac_predicted,
)
```

**Верификация:** `venv/Scripts/pytest tests/test_pipeline_stage9.py tests/test_pipeline.py -x` → все PASS

---

## Task 12: Exp 19 — SKS Embedding Quality

**Файл:** `experiments/exp19_sks_embedding_quality.py`

**Метрика:** NMI(nearest-neighbors в HAC-space, ground truth labels) > 0.7

**Алгоритм:**
1. Загрузить shapes датасет (или MNIST small)
2. Прогнать pipeline на N стимулов с labels
3. Собрать `winner_embedding` за все циклы
4. Для каждого embedding найти K=5 ближайших соседей по cosine similarity
5. Построить граф соседей → предсказанные кластеры (connected components)
6. Вычислить NMI(predicted, true_labels)
7. Assert NMI > 0.7

**Верификация:** `venv/Scripts/python experiments/exp19_sks_embedding_quality.py` → PASS

---

## Task 13: Exp 20 — HAC vs Discrete Predictor

**Файл:** `experiments/exp20_hac_vs_discrete_predictor.py`

**Метрика:** HAC top-1 accuracy ≥ discrete top-1 accuracy (baseline ≈ 72.9%)

**Алгоритм:**
1. Прогнать pipeline на последовательных стимулах (повторяющиеся паттерны)
2. HAC accuracy: `argmax_k cosine(hac_predicted, embed_k)` → predicted label, сравнить с actual winner
3. Discrete accuracy: `argmax_k confidence(edge_k)` → predicted ID, сравнить с actual winner
4. Assert HAC_acc ≥ discrete_acc

**Верификация:** `venv/Scripts/python experiments/exp20_hac_vs_discrete_predictor.py` → PASS

---

## Task 14: Exp 16 (debt) — Confidence Ratio Gate

**Файл:** `experiments/exp16_confidence_ratio_gate.py`

**Метрика:** `mean_confidence(focused) / mean_confidence(noise) > 1.5`

**Алгоритм:**
1. Создать два набора стимулов: focused (чёткий паттерн) и noise (случайный)
2. Прогнать pipeline с `HACPredictionConfig(enabled=True)`
3. Собрать `metacog.confidence` для обоих наборов
4. Вычислить ratio
5. Assert ratio > 1.5

**Верификация:** `venv/Scripts/python experiments/exp16_confidence_ratio_gate.py` → PASS

---

## Task 15: Exp 18 (debt) — BroadcastPolicy

**Файл:** `experiments/exp18_broadcast_policy.py`

**Метрика:** `cross_activation_ratio > 1.2`

**Алгоритм:**
1. Включить `BroadcastPolicy(strength=1.0, threshold=0.6)` в MetacogConfig
2. Прогнать pipeline: собрать `n_spikes` до broadcast цикла и после
3. `cross_activation_ratio = n_spikes_after / n_spikes_before` для non-winner узлов
4. Assert > 1.2

**Верификация:** `venv/Scripts/python experiments/exp18_broadcast_policy.py` → PASS

---

## Task 16: Полный прогон тестов

```bash
venv/Scripts/pytest tests/ -x --tb=short
```

Все тесты PASS, нет регрессий для Exp 1–17.

---

## Зависимости задач

```
Task 1 (конфиги)
    ├── Task 2+3 (SKSEmbedder)
    ├── Task 4+5 (HACPredictionEngine)
    ├── Task 6+7 (BroadcastPolicy)
    └── Task 8+9 (MetacogState)
              │
        Task 10+11 (Pipeline)
              │
    ┌─────────┼──────────┐
    │         │          │
Task 12    Task 13    Task 14+15
(Exp 19)  (Exp 20)  (Exp 16+18)
              │
          Task 16
       (полный прогон)
```
