# Stage 16: DCAM Tiered Memory Consolidation

**Версия:** 1.4
**Дата:** 2026-03-26
**Статус:** DRAFT (после spec review v2)

---

## Цель

Завершить петлю долгосрочной памяти СНКС: EpisodicBuffer → SSG → агентный цикл. Закрыть последний MVP-тезис: "DCAM-формат пригоден для хранения и восстановления модели мира."

**Gate:** exp35 PASS + exp36 PASS + exp37 PASS
- Функциональный: агент в сессии 2 работает не хуже сессии 1 (SSG загружен с диска)
- Структурный: SSG содержит осмысленные рёбра, replay не ломает DAF

---

## Архитектура: Tiered Memory

```
┌──────────────────────────────────────────────────────┐
│                    EmbodiedAgent                      │
│                                                       │
│  ┌───────────────┐   ┌──────────────────────────────┐ │
│  │  HOT memory   │   │       COLD memory            │ │
│  │ CausalWorld   │   │  ConsolidationScheduler      │ │
│  │ Model         │   │  ├─ _node_registry: dict     │ │
│  │ (transition   │   │  ├─ _edge_actions: dict      │ │
│  │  table)       │   │  ├─ SKSIDEmbedder            │ │
│  └──────┬────────┘   │  └─ DcamWorldModel.graph     │ │
│         │            │     (SSG causal layer)        │ │
│         ▼            └──────────────┬───────────────┘ │
│  ┌───────────────┐                  │ cold_override   │
│  │ TieredPlanner │◄─────────────────┘ (weight > θ)   │
│  └───────────────┘                                    │
│                                                       │
│  AgentEpisodicBuffer ─[every N ep]─► ConsolidationScheduler._run()
│                                           │           │
│                                      SSG edges +      │
│                                      scheduler.save() │
│                                           │           │
│                                      ReplayEngine     │
└──────────────────────────────────────────────────────┘
```

**Принцип:** hot memory (CausalWorldModel) — быстрый lookup текущего эпизода. Cold memory (ConsolidationScheduler поверх SSG) — накопленное структурное знание между сессиями. TieredPlanner использует cold override когда SSG уверен.

---

## Ключевые архитектурные решения

### AgentTransitionBuffer — новый буфер переходов

`EpisodicBuffer` в `dcam/episodic.py` хранит `Episode(active_nodes, context_hac, importance)` — нет `pre_sks`, `post_sks`, `action`. Stage 15 подключил его через `store()`, но не добавил интерфейс переходов.

Для consolidation нужен отдельный `AgentTransitionBuffer` — простой deque из `AgentTransition(pre_sks, post_sks, action, importance)`. Живёт в `agent/transition_buffer.py`. Подключается в `CausalAgent.observe_result()` рядом с существующим `episodic_buffer.store()`.

```python
@dataclass
class AgentTransition:
    pre_sks: set[int]
    action: int
    post_sks: set[int]
    importance: float

class AgentTransitionBuffer:
    """Simple fixed-capacity buffer for (pre_sks, action, post_sks, importance) transitions."""

    def __init__(self, capacity: int = 200):
        self._buf: deque[AgentTransition] = deque(maxlen=capacity)

    def add(self, pre_sks: set[int], action: int,
            post_sks: set[int], importance: float) -> None:
        self._buf.append(AgentTransition(pre_sks, action, post_sks, importance))

    def get_top_k(self, k: int, by: str = "importance") -> list[AgentTransition]:
        return sorted(self._buf, key=lambda t: t.importance, reverse=True)[:k]

    def __len__(self) -> int:
        return len(self._buf)
```

Добавить в `CausalAgent.observe_result()` (рядом с `episodic_buffer.store()`):
```python
if self.transition_buffer is not None:
    self.transition_buffer.add(self._pre_sks, self._last_action, post_sks,
                                importance=prediction_error)
```

`transition_buffer: AgentTransitionBuffer | None` добавить в `CausalAgent.__init__()` (опционально, по конфигу).

---

### SKSIDEmbedder — детерминированный cross-session embedder

`SKSEmbedder` (Stage 9) инициализируется `torch.randn` → разный embedding space при каждом запуске → cross-session несовместимость.

**Решение:** новый `SKSIDEmbedder` — детерминированный. Вектор для SKS ID `s` генерируется с `torch.Generator().manual_seed(s)`. Одинаковый ID всегда даёт одинаковый вектор в любой сессии. Не требует сохранения.

```python
class SKSIDEmbedder:
    """Deterministic, session-stable HAC embedding for integer SKS IDs.

    Each ID → fixed unit vector via seeded torch.Generator.
    Same ID always produces the same vector, regardless of session.
    No persistence needed.
    """
    def __init__(self, hac_dim: int, device: torch.device):
        self._dim = hac_dim
        self._device = device
        self._cache: dict[int, Tensor] = {}

    def embed_id(self, sks_id: int) -> Tensor:
        if sks_id not in self._cache:
            g = torch.Generator()
            g.manual_seed(int(sks_id) % (2 ** 32))
            vec = torch.randn(self._dim, generator=g, device=self._device)
            self._cache[sks_id] = vec / vec.norm().clamp(min=1e-8)
        return self._cache[sks_id]

    def encode_sks_set(self, sks_ids: set[int], hac: HACEngine) -> Tensor | None:
        if not sks_ids:
            return None
        vecs = [self.embed_id(s) for s in sks_ids]
        return hac.bundle(vecs) if len(vecs) > 1 else vecs[0]
```

### SSG используется без изменений

`StructuredSparseGraph` остаётся неизменным. ConsolidationScheduler записывает в **"causal" layer** через существующие `add_edge()` / `update_edge()`. Векторные данные (node_registry) хранятся в самом `ConsolidationScheduler`, не в SSG.

### Persistence — только ConsolidationScheduler

`DcamWorldModel.save()` уже сохраняет SSG (через `persistence.py`). ConsolidationScheduler дополнительно сохраняет свой `_node_registry` и `_edge_actions` через `torch.save()`. `SKSIDEmbedder` персистентности не требует (детерминированный).

### Node ID Registry

ConsolidationScheduler владеет реестром нод:
- `_node_registry: dict[int, Tensor]` — node_id → HAC вектор (центроид)
- `_next_node_id: int = 0`
- `_node_threshold: float = 0.7` — порог для переиспользования ноды

`_get_or_create_node(vec)` — линейный скан по `_node_registry`, возвращает существующий node_id если `cosine_sim > _node_threshold`, иначе создаёт новый.

---

## Компоненты

### 1. SKSIDEmbedder (`dcam/consolidation_sched.py`)

Описан выше. Живёт в том же файле что `ConsolidationScheduler`.

### 2. ConsolidationScheduler (`dcam/consolidation_sched.py`)

```python
@dataclass
class ConsolidationSummary:
    n_episodes_processed: int
    n_edges_added: int
    total_causal_edges: int
    total_nodes: int

class ConsolidationScheduler:
    def __init__(
        self,
        agent_buffer,           # AgentEpisodicBuffer (из agent/)
        dcam: DcamWorldModel,   # для доступа к SSG + HACEngine
        every_n: int = 10,
        top_k: int = 50,
        node_threshold: float = 0.7,
        save_path: str | None = None,  # None → не сохранять
    ):
        self.embedder = SKSIDEmbedder(dcam.hac.dim, dcam.device)
        self._node_registry: dict[int, Tensor] = {}
        self._next_node_id: int = 0
        self._edge_actions: dict[tuple[int, int], int] = {}
        ...

    def maybe_consolidate(self, episode: int) -> ConsolidationSummary | None:
        if episode > 0 and episode % self.every_n == 0:
            return self._run()
        return None

    def query(self, context_sks: set[int],
              threshold: float = 0.3) -> tuple[int | None, float]:
        """Encode context_sks → HAC vec, find nearest node, return (action, weight).

        Returns (None, 0.0) if registry empty or best similarity < threshold.
        """
        vec = self.embedder.encode_sks_set(context_sks, self.dcam.hac)
        if vec is None or not self._node_registry:
            return None, 0.0
        best_nid, best_sim = self._nearest_node(vec)
        if best_sim < threshold:
            return None, 0.0
        # Find highest-weight outgoing edge in causal layer
        neighbors = self.dcam.graph.get_neighbors(best_nid, layer="causal")
        if not neighbors:
            return None, 0.0
        dst_id, weight = max(neighbors, key=lambda x: x[1])
        action = self._edge_actions.get((best_nid, dst_id))
        return action, weight

    def save_state(self, path: str) -> None:
        """Save node registry and edge actions. Called after _run() if save_path set."""
        torch.save({
            "node_registry": self._node_registry,
            "edge_actions": self._edge_actions,
            "next_node_id": self._next_node_id,
        }, path + "_sched.pt")
        self.dcam.save(path)  # saves SSG (causal layer) + buffer

    def load_state(self, path: str) -> None:
        """Restore node registry and edge actions."""
        state = torch.load(path + "_sched.pt", map_location=self.dcam.device)
        self._node_registry = state["node_registry"]
        self._edge_actions = state["edge_actions"]
        self._next_node_id = state["next_node_id"]
        self.dcam.load(path)  # restores SSG

    def _run(self) -> ConsolidationSummary:
        episodes = self.agent_buffer.get_top_k(k=self.top_k, by="importance")
        n_added = 0
        for ep in episodes:
            ctx_vec  = self.embedder.encode_sks_set(ep.pre_sks,  self.dcam.hac)
            next_vec = self.embedder.encode_sks_set(ep.post_sks, self.dcam.hac)
            if ctx_vec is None or next_vec is None:
                continue
            src_id = self._get_or_create_node(ctx_vec)
            dst_id = self._get_or_create_node(next_vec)
            self.dcam.graph.update_edge(src_id, dst_id, layer="causal",
                                        delta=ep.importance)
            self._edge_actions[(src_id, dst_id)] = ep.action  # latest wins
            n_added += 1
        if self.save_path:
            self.save_state(self.save_path)
        total = sum(len(dsts) for dsts in
                    self.dcam.graph._layers["causal"].values())
        return ConsolidationSummary(
            n_episodes_processed=len(episodes),
            n_edges_added=n_added,
            total_causal_edges=total,
            total_nodes=len(self._node_registry),
        )

    def _get_or_create_node(self, vec: Tensor) -> int:
        if self._node_registry:
            nid, sim = self._nearest_node(vec)
            if sim > self._node_threshold:
                return nid
        nid = self._next_node_id
        self._next_node_id += 1
        self._node_registry[nid] = vec.detach()
        return nid

    def _nearest_node(self, vec: Tensor) -> tuple[int, float]:
        best_nid, best_sim = -1, -1.0
        for nid, nvec in self._node_registry.items():
            sim = self.dcam.hac.similarity(vec, nvec)
            if sim > best_sim:
                best_sim, best_nid = sim, nid
        return best_nid, best_sim
```

### 3. TieredPlanner (`agent/tiered_planner.py`)

```python
class TieredPlanner:
    def __init__(
        self,
        causal_model: CausalWorldModel,
        scheduler: ConsolidationScheduler,
        cold_threshold: float = 0.3,
        n_actions: int = 7,
    ):
        ...

    def plan(self, context_sks: set[int]) -> tuple[int, str]:
        """Returns (action, source) where source in {'hot', 'cold', 'random'}."""
        hot_action, hot_conf = self.causal_model.best_action(context_sks)
        cold_action, cold_weight = self.scheduler.query(context_sks,
                                                         threshold=self.cold_threshold)
        if cold_action is not None and cold_weight > hot_conf:
            return cold_action, 'cold'
        if hot_action is not None:
            return hot_action, 'hot'
        return random.randint(0, self.n_actions - 1), 'random'
```

### 4. ReplayEngine (`dcam/replay.py`)

```python
@dataclass
class ReplayReport:
    n_replayed: int
    stdp_updates: int

class ReplayEngine:
    def __init__(self, daf_engine: DafEngine, stdp: STDPModule,
                 top_k: int = 10, n_steps: int = 50):
        ...

    def replay(self, agent_buffer) -> ReplayReport:
        """Replay top_k episodes by importance through DAF + STDP."""
        episodes = agent_buffer.get_top_k(k=self.top_k, by="importance")
        stdp_updates = 0
        for ep in episodes:
            node_ids = [s for s in ep.pre_sks if s < self.daf_engine.num_nodes]
            if not node_ids:
                continue
            self.daf_engine.inject_external_currents(node_ids, value=1.0)
            result = self.daf_engine.step(n_steps=self.n_steps)
            if result.fired_history is not None:
                self.stdp.update(result.fired_history)
                stdp_updates += 1
        return ReplayReport(n_replayed=len(episodes), stdp_updates=stdp_updates)
```

### 5. DafEngine extension (`daf/engine.py`)

```python
def inject_external_currents(self, node_ids: list[int], value: float = 1.0) -> None:
    """Inject external current to specified nodes before next step.

    Writes to _external_currents[:, 0] (I_ext channel), NOT to states (voltage).
    Effect is transient — _external_currents is reset after each step() call.
    Used by ReplayEngine for approximate SKS re-activation.
    """
    valid = [nid for nid in node_ids if 0 <= nid < self.num_nodes]
    if valid:
        self._external_currents[valid, 0] += value
```

---

## Конфигурация

Добавить в `daf/types.py`:

```python
@dataclass
class ConsolidationConfig:
    enabled: bool = False
    every_n: int = 10
    top_k: int = 50
    cold_threshold: float = 0.3
    node_threshold: float = 0.7   # для _get_or_create_node
    save_path: str | None = None  # None = no save

@dataclass
class ReplayConfig:
    enabled: bool = False
    top_k: int = 10
    n_steps: int = 50
```

Добавить в `PipelineConfig` / конфиг агента. Оба `enabled=False` по умолчанию.

---

## Интеграция в EmbodiedAgent

Предполагается что агент уже имеет `agent_buffer` (AgentEpisodicBuffer из Stage 15).

В `EmbodiedAgent.__init__()` (опционально):
```python
if cfg.consolidation.enabled:
    self.consolidation_scheduler = ConsolidationScheduler(
        agent_buffer=self.agent_buffer,
        dcam=self.dcam,
        every_n=cfg.consolidation.every_n,
        top_k=cfg.consolidation.top_k,
        node_threshold=cfg.consolidation.node_threshold,
        save_path=cfg.consolidation.save_path,
    )
    self.tiered_planner = TieredPlanner(
        causal_model=self.causal_agent.causal_model,
        scheduler=self.consolidation_scheduler,
        cold_threshold=cfg.consolidation.cold_threshold,
    )
```

В `EmbodiedAgent.step(obs)`, после извлечения SKS из result:
```python
if self.tiered_planner is not None:
    action, plan_source = self.tiered_planner.plan(current_sks)
    self._last_plan_source = plan_source
```

В конце каждого эпизода:
```python
if self.consolidation_scheduler is not None:
    report = self.consolidation_scheduler.maybe_consolidate(self._episode_idx)
    if report is not None and cfg.replay.enabled:
        self.replay_engine.replay(self.agent_buffer)
```

**Приоритет действий:** cold override активен только когда `cold_weight > hot_conf`. Конфигуратор FSM (GOAL_SEEKING, EXPLORE, CONSOLIDATE) сохраняет управление в остальных режимах.

---

## Эксперименты

### Exp 35: Cross-session Persistence

**Файл:** `src/snks/experiments/exp35_persistence.py`

```
Окружение: MiniGrid-FourRooms-v0
N=500, avg_degree=10

Сессия 1: n_episodes=40
  consolidation.enabled=True, every_n=10, top_k=50
  save_path="/tmp/snks_s16"
  Записать: SR_s1, mean_steps_s1, summary.total_nodes, summary.total_causal_edges

Сессия 2: n_episodes=40, новый EmbodiedAgent
  Порядок инициализации:
    1. agent2 = EmbodiedAgent(config)     # dcam инициализирован с пустым SSG
    2. scheduler2 = ConsolidationScheduler(agent2.transition_buffer, agent2.dcam, ...)
    3. scheduler2.load_state("/tmp/snks_s16")
       # восстанавливает: _node_registry, _edge_actions, SSG causal layer
       # SKSIDEmbedder НЕ требует загрузки (детерминированный)
    4. agent2.tiered_planner = TieredPlanner(agent2.causal_model, scheduler2, ...)
  TieredPlanner активен с эпизода 0
  Записать: SR_s2, mean_steps_s2, cold_override_count

Gate:
  cold_override_count > 0                    [PRIMARY] SSG реально использовался
  SR_s2 >= SR_s1 * 0.9                       нет регрессии
  mean_steps_s2 <= mean_steps_s1 * 1.1       overhead приемлем
```

### Exp 36: SSG Structural Quality

**Файл:** `src/snks/experiments/exp36_ssg_quality.py`

```
Загрузить checkpoint из exp35 сессии 1.
Верифицировать ConsolidationScheduler state:

  1. len(scheduler._node_registry) > 0
     (ноды созданы)

  2. Топ-10 рёбер causal layer по весу: все weight > cold_threshold (0.3)
     edges = sorted(ssg.get_all_edges("causal"), key=lambda x: -x[2])[:10]

  3. Для каждой из топ-10 нод: re-encode(same sks_ids) с новым SKSIDEmbedder
     → similarity(stored_vec, re_encoded_vec) > 0.99
     (детерминированность: один и тот же ID → один и тот же вектор)

Gate: все три проверки PASS
```

### Exp 37: Replay Quality

**Файл:** `src/snks/experiments/exp37_replay.py`

```
Два варианта, каждый 20 эпизодов в MiniGrid-FourRooms:
  no_replay:   consolidation.enabled=True, replay.enabled=False
  with_replay: consolidation.enabled=True, replay.enabled=True

Метрика: mean winner_pe на шагах 10–20 каждого эпизода
  (EpisodicHACPredictor.compute_winner_pe — уже реализован в Stage 15)

Gate:
  replay_report.stdp_updates > 0                          [PRIMARY]
  mean_pe(with_replay) <= mean_pe(no_replay) * 1.05       replay не ухудшает PE
```

---

## Новые файлы

| Файл | Назначение |
|------|-----------|
| `src/snks/agent/transition_buffer.py` | AgentTransition + AgentTransitionBuffer |
| `src/snks/dcam/consolidation_sched.py` | SKSIDEmbedder + ConsolidationScheduler + ConsolidationSummary |
| `src/snks/agent/tiered_planner.py` | TieredPlanner (hot/cold арбитраж) |
| `src/snks/dcam/replay.py` | ReplayEngine + ReplayReport |
| `src/snks/experiments/exp35_persistence.py` | Cross-session persistence |
| `src/snks/experiments/exp36_ssg_quality.py` | SSG structural quality |
| `src/snks/experiments/exp37_replay.py` | Replay quality |
| `tests/test_sks_id_embedder.py` | Детерминированность: same id → same vec |
| `tests/test_transition_buffer.py` | add/get_top_k; capacity eviction |
| `tests/test_consolidation_sched.py` | Edges появляются после _run(); save/load roundtrip |
| `tests/test_tiered_planner.py` | cold override при weight > threshold; fallback на hot |
| `tests/test_replay.py` | stdp_updates > 0; пустой буфер → 0 updates |

## Изменяемые файлы

| Файл | Изменение |
|------|-----------|
| `src/snks/daf/types.py` | ConsolidationConfig, ReplayConfig |
| `src/snks/daf/engine.py` | inject_external_currents(node_ids, value) + `@property num_nodes → config.num_nodes` |
| `src/snks/agent/causal_model.py` | `best_action(context_sks, n_actions=7) → (int\|None, float)` — итерация по actions, возврат argmax confidence |
| `src/snks/agent/agent.py` | `transition_buffer: AgentTransitionBuffer \| None` в `__init__`; вызов `transition_buffer.add()` в `observe_result()` |
| `src/snks/agent/embodied_agent.py` | TieredPlanner в `step()`; ConsolidationScheduler + ReplayEngine в конце эпизода |

**Что НЕ меняется:** `dcam/ssg.py`, `dcam/persistence.py`, `dcam/world_model.py`, `dcam/consolidation.py`, `dcam/episodic.py`.

---

## Gate для перехода к Stage 17

- `exp35`: PASS — cold_override_count > 0, SR_s2 >= SR_s1 * 0.9
- `exp36`: PASS — ноды есть, рёбра с весами > threshold, детерминированность подтверждена
- `exp37`: PASS — replay запускался, PE не деградировал

---

## Что НЕ делаем в Stage 16

- Language grounding
- GWS competitive selection
- Multi-agent
- FHN → Reservoir Computing замена
- Обучаемый SKSEmbedder
- SSG edge pruning / growth
- Изменения в DcamConsolidation (legacy, backward compat)
- Изменения в dcam/persistence.py
