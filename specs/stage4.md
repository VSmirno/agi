# Этап 4: DCAM Хранилище ⏳

**Статус:** Не начат
**Срок:** Неделя 6–8
**Зависимости:** Этап 1 ✅

## Цель

Реализовать DCAM (Dual-Code Associative Memory) — двухкодовое хранилище модели мира:
- **HAC** (Holographic Associative Codes) — содержание (что)
- **SSG** (Structured Sparse Graph) — структура (как связано)

## Модули

| Модуль | Назначение |
|--------|-----------|
| `dcam/hac.py` | HACEngine: bind/unbind (FFT), bundle, permute, encode_scalar |
| `dcam/lsh.py` | LSH Index: SimHash (32 таблицы × 16 бит), O(1) поиск |
| `dcam/ssg.py` | SSG: 4 слоя (structural, causal, temporal, modulatory) |
| `dcam/episodic.py` | Episodic buffer: 10K эпизодов, importance-weighted eviction |
| `dcam/consolidation.py` | Консолидация (аналог сна): STC, co-activation, causal extraction |
| `dcam/persistence.py` | Save/load: safetensors + JSON |
| `dcam/world_model.py` | DcamWorldModel facade |

## Контракты

### HACEngine
```python
class HACEngine:
    def __init__(self, dim: int = 2048, device: str = "cpu"):
    def random_vector(self) -> Tensor:                    # (D,)
    def bind(self, a: Tensor, b: Tensor) -> Tensor:       # circular convolution via FFT
    def unbind(self, a: Tensor, bound: Tensor) -> Tensor:  # circular correlation
    def bundle(self, vectors: list[Tensor]) -> Tensor:     # sum + normalize
    def permute(self, v: Tensor, k: int) -> Tensor:        # cyclic shift
    def encode_scalar(self, value: float) -> Tensor:       # fractional power encoding
    def similarity(self, a: Tensor, b: Tensor) -> float:   # cosine similarity
    def batch_bind(self, A: Tensor, B: Tensor) -> Tensor:          # (batch, D)
    def batch_similarity(self, query: Tensor, keys: Tensor) -> Tensor: # (M,)
```

### LSH Index
```python
class LSHIndex:
    def __init__(self, dim: int, n_tables: int = 32, n_bits: int = 16):
    def insert(self, key: Tensor, value: int) -> None:
    def query(self, key: Tensor, top_k: int = 10) -> list[tuple[int, float]]:
    def remove(self, value: int) -> None:
```

### SSG
```python
class StructuredSparseGraph:
    def add_edge(self, src: int, dst: int, layer: str, weight: float) -> None:
    def get_neighbors(self, node: int, layer: str) -> list[tuple[int, float]]:
    def update_edge(self, src: int, dst: int, layer: str, delta: float) -> None:
    def prune(self, threshold: float) -> int:
```

### DcamWorldModel
```python
class DcamWorldModel:
    def __init__(self, config: DcamConfig, device: str = "cpu"):
    def store_episode(self, active_nodes: dict, context: Tensor, importance: float) -> int:
    def query_similar(self, query_hac: Tensor, top_k: int = 10) -> list[tuple[int, float]]:
    def consolidate(self) -> ConsolidationReport:
    def save(self, path: str) -> None:
    def load(self, path: str) -> None:
```

## Gate

1. **HAC fidelity:** unbind(a, bind(a, b)) cosine > 0.9
2. **Persistence:** save → load → test accuracy, Δ ≤ 1%
3. **Эксперимент 5:** обучить → save DCAM → reload → retest

## Параметры

```python
@dataclass
class DcamConfig:
    hac_dim: int = 2048
    lsh_tables: int = 32
    lsh_bits: int = 16
    episodic_capacity: int = 10_000
    consolidation_stc_threshold: float = 0.5
    consolidation_coact_min: int = 3
    ssg_layers: tuple = ("structural", "causal", "temporal", "modulatory")
```

## Порядок реализации (TDD)

1. `hac.py` + тесты (bind/unbind roundtrip, bundle similarity, batch ops)
2. `lsh.py` + тесты (insert/query recall, remove)
3. `ssg.py` + тесты (CRUD, prune, multi-layer)
4. `episodic.py` + тесты (store, capacity eviction, importance ordering)
5. `consolidation.py` + тесты (STC, co-activation, causal)
6. `persistence.py` + тесты (save/load roundtrip)
7. `world_model.py` + тесты (facade integration)
8. `exp5_persistence.py` — Эксперимент 5

## Справочный материал

Подробная теория DCAM: [`World_Model_Encoding.md`](../World_Model_Encoding.md)
