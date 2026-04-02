# Stage 45: VSA World Model — Design Spec

**Date:** 2026-04-02
**Phase:** 1 — Живой DAF
**Type:** New mechanism — world model foundation

## Мотивация

Stage 44 (Foundation Audit) показал:
- DAF-ядро корректно как perception engine (26/26 tests PASS)
- Но нет пути от reward к action selection — STDP меняет coupling weights, а не policy
- Naked DAF на DoorKey-5x5: 0% success (3% = random walk)
- Проблема не в компонентах, а в отсутствии world model

Нужен **отдельный модуль world model**, task-agnostic, который:
- Запоминает transitions из опыта
- Предсказывает последствия действий
- Позволяет планировать через SDM lookup
- Отвечает на произвольные queries (позже)
- Работает без backprop, через локальные операции

## Позиция в фазе

**Фаза 1 маркеры:** Pure DAF ≥ 50% DoorKey-5x5.
Stage 45 адресует core bottleneck (нет world model / credit assignment). Это не reactive patch, а архитектурное решение на основе исследования (VSA + SDM).

## Что НЕ входит в scope

- Изменения в DAF/oscillator pipeline
- Semantic memory / consolidation / "sleep" (Phase 3 roadmap)
- Multi-step planning (depth > 1)
- QA interface / language queries
- Интеграция VSA с FHN через Resonator Networks (будущий stage)

---

## Архитектура

```
obs (7×7×3 symbolic) → VSAEncoder → state_vsa (512-bit)
                                         ↓
                                    SDMMemory.write(state, action, next_state, reward)
                                         ↓
                                    SDMPlanner.select(state) → action
                                         ↓
                                    env.step(action)
```

Параллельно (не блокирует, для сравнения):
```
obs → SymbolicEncoder → SDR → DAF Pipeline → SKS (как сейчас)
```

### Компонент 1: VSACodebook

Фиксированный random codebook для Binary Spatter Code (BSC).

**Параметры:**
- `dim`: int = 512 (параметризовано, можно увеличить)
- Операции:
  - `bind(a, b) → XOR(a, b)` — ассоциация (обратимая: bind(bind(a,b), b) = a)
  - `bundle(*vecs) → majority_vote(vecs)` — объединение
  - `similarity(a, b) → normalized Hamming distance`

**Codebook entries:**
- Roles: `agent_pos`, `key_pos`, `key_color`, `door_pos`, `door_state`, `has_key`, `goal_pos`
- Fillers: `pos_R_C` для каждой (row, col) в 7×7, `locked`, `open`, `yes`, `no`, `color_0`..`color_5`
- Actions: `action_0`..`action_6`
- Special: `reward_positive`, `reward_negative`, `reward_zero`

Все entries — random binary 512-bit vectors, генерируются один раз при создании.

### Компонент 2: VSAEncoder

Кодирует MiniGrid symbolic observation в structured VSA vector.

**Input:** obs (7×7×3) int tensor — object_type, color, state per cell.

**Process:**
1. Scan obs, extract facts: agent position, key position/color, door position/state, goal position, inventory
2. Encode each fact as `bind(role, filler)`
3. Bundle all facts: `state = bundle(fact1, fact2, ...)`

**Output:** 512-bit binary vector.

**Пример:**
```
obs shows: agent at (3,2), blue key at (1,4), locked door at (2,3), goal at (4,4), no key in hand

state = bundle(
    bind(role_agent_pos, filler_pos_3_2),
    bind(role_key_pos,   filler_pos_1_4),
    bind(role_key_color, filler_color_2),
    bind(role_door_pos,  filler_pos_2_3),
    bind(role_door_state, filler_locked),
    bind(role_has_key,   filler_no),
    bind(role_goal_pos,  filler_pos_4_4),
)
```

### Компонент 3: SDMMemory

Sparse Distributed Memory для хранения transitions.

**Параметры:**
- `n_locations`: int = 10000 (hard locations)
- `dim`: int = 512 (address and content width)
- `activation_radius`: int — calibrated at init:
  1. Compute pairwise Hamming distances between 1000 random hard location addresses
  2. Set radius = median(distances) × 0.45
  3. Verify: random query activates 1-5% of locations (100-500 out of 10K)
  4. If outside range — adjust factor until 1-5% hit. Log final radius and activation %.

**Storage layout:**
- Address space: 512-bit binary
- Each hard location has:
  - `address`: 512-bit (random, fixed at init)
  - `content_next`: 512-bit int counters (accumulate writes)
  - `content_reward`: 512-bit int counters (accumulate reward signals)

**Write:**
```python
def write(self, state_vsa, action_vsa, next_state_vsa, reward: float):
    address = bind(state_vsa, action_vsa)
    activated = locations_within_radius(address)  # ~1-5% of locations
    for loc in activated:
        loc.content_next += (2 * next_state_vsa - 1)  # ±1 update
        loc.content_reward += reward_to_vsa(reward)     # ±1 update
```

**Read:**
```python
def read_next(self, state_vsa, action_vsa) -> tuple[Tensor, float]:
    address = bind(state_vsa, action_vsa)
    activated = locations_within_radius(address)
    summed_next = sum(loc.content_next for loc in activated)
    summed_reward = sum(loc.content_reward for loc in activated)
    predicted_next = (summed_next > 0).float()  # threshold to binary
    confidence = activation_count / expected_count
    return predicted_next, confidence

def read_reward(self, state_vsa, action_vsa) -> float:
    """Signed reward score: positive = good, negative = bad, zero = unknown."""
    address = bind(state_vsa, action_vsa)
    activated = locations_within_radius(address)
    summed = sum(loc.content_reward for loc in activated)
    thresholded = (summed > 0).float()
    sim_pos = similarity(thresholded, reward_positive_vsa)
    sim_neg = similarity(thresholded, reward_negative_vsa)
    return sim_pos - sim_neg  # signed: learn to avoid bad actions too
```

**GPU implementation:**
- Hard location addresses: (10000, 512) binary tensor
- Content: (10000, 512) int16 tensor × 2 (next + reward)
- Activation: `hamming_distance(query, all_addresses)` — single batched XOR + popcount
- Total memory: ~15MB

### Компонент 4: SDMPlanner

Action selection через 1-step lookahead в SDM.

```python
def select(self, current_state_vsa) -> int:
    scores = []
    confidences = []
    for action_idx in range(n_actions):
        action_vsa = codebook.action(action_idx)
        reward_score = sdm.read_reward(current_state_vsa, action_vsa)  # signed: pos - neg
        _, confidence = sdm.read_next(current_state_vsa, action_vsa)
        scores.append(reward_score * confidence)
        confidences.append(confidence)
    
    if max(confidences) < min_confidence:
        return random_action()  # exploration: SDM doesn't know
    
    return argmax(scores) with epsilon noise
```

**Exploration strategy:**
- Если SDM не знает (все confidence < threshold) → random action
- Если SDM знает → exploit best predicted reward + epsilon=0.1 noise
- Epsilon fixed at 0.1 (не decay — exploration important for filling SDM)

### Компонент 5: WorldModelAgent

Объединяет всё.

```python
class WorldModelAgent:
    def __init__(self, config: WorldModelConfig):
        self.codebook = VSACodebook(dim=config.dim)
        self.encoder = VSAEncoder(self.codebook)
        self.sdm = SDMMemory(
            n_locations=config.n_locations,
            dim=config.dim,
        )
        self.planner = SDMPlanner(
            sdm=self.sdm,
            codebook=self.codebook,
            n_actions=config.n_actions,
            min_confidence=config.min_confidence,
            epsilon=config.epsilon,
        )
        self._prev_state = None
        self._prev_action = None
    
    def step(self, obs) -> int:
        state = self.encoder.encode(obs)
        action = self.planner.select(state)
        self._prev_state = state
        self._prev_action = action
        return action
    
    def observe(self, obs, reward):
        if self._prev_state is None:
            return  # first step — nothing to record
        next_state = self.encoder.encode(obs)
        self.sdm.write(
            self._prev_state,
            self.codebook.action(self._prev_action),
            next_state,
            reward,
        )
    
    def run_episode(self, env, max_steps=200):
        obs = env.reset()
        total_reward = 0
        for step in range(max_steps):
            action = self.step(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            self.observe(obs, reward)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward > 0, step + 1, total_reward
```

---

## Конфигурация

```python
@dataclass
class WorldModelConfig:
    dim: int = 512              # VSA vector dimensionality
    n_locations: int = 10000    # SDM hard locations
    n_actions: int = 7          # MiniGrid action count
    min_confidence: float = 0.1 # below this → random exploration
    epsilon: float = 0.1        # exploitation noise
    max_episode_steps: int = 200
```

---

## Эксперименты

### Exp 105a: VSA encoding accuracy
- Encode 100 разных MiniGrid observations
- Unbind каждый role → проверить что filler correct
- Gate: unbinding accuracy ≥ 90%

### Exp 105b: SDM prediction accuracy
- Записать 1000 transitions из random episodes
- Read: predict next_state для seen (state, action) pairs
- Gate: prediction similarity ≥ 0.6 для seen transitions

### Exp 105c: WorldModelAgent на DoorKey-5x5
- 200 эпизодов, symbolic obs (7×7×3) напрямую
- Gate (primary): success_rate ≥ 0.15 за 200 эпизодов
- Gate (stretch): success_rate ≥ 0.30
- Gate (learning): last 50 eps success > first 50 eps success (improvement trend)
- Gate (intermediate): при 100 эпизодах success > 5% (early signal)
- Reference: tabular Q-learning ~80% за 200 ep, naked DAF = 3% (random)

### Exp 105d: Сравнение с PureDafAgent (Naked DAF)
- Тот же env, 200 эпизодов
- WorldModelAgent vs PureDafAgent (exp104 baseline = ~3%)
- Gate: WorldModelAgent > 3× PureDafAgent success rate

---

## Порядок выполнения

| Шаг | Что | Где | Время |
|-----|-----|-----|-------|
| 1 | VSACodebook + tests | CPU | ~15 мин |
| 2 | VSAEncoder + tests | CPU | ~15 мин |
| 3 | SDMMemory + tests | CPU | ~20 мин |
| 4 | SDMPlanner + tests | CPU | ~15 мин |
| 5 | WorldModelAgent + tests | CPU | ~15 мин |
| 6 | Exp 105a (VSA accuracy) | CPU | ~5 мин |
| 7 | Exp 105b (SDM prediction) | CPU | ~10 мин |
| 8 | Exp 105c (DoorKey-5x5) | GPU minipc | ~30-60 мин |
| 9 | Exp 105d (comparison) | GPU minipc | ~30 мин |

Шаги 1-7 на CPU (local), 8-9 на GPU (minipc).

---

## Future integration hook

Для будущей интеграции VSA с DAF perception (Resonator Networks, SKS→VSA):

```python
class VSAEncoder:
    def encode_from_sdr(self, sdr: Tensor) -> Tensor:
        """Encode from DAF SDR/SKS output instead of symbolic obs.
        Stub — будет реализован при интеграции с DAF."""
        raise NotImplementedError("Stage 45 uses symbolic obs; DAF integration is future work")
```

Это placeholder — не реализуется в Stage 45, но определяет interface point.

## Что дальше (не в scope этого stage)

- **Multi-step planning:** depth > 1, tree search через SDM
- **Semantic consolidation:** periodic extraction of patterns из SDM → long-term memory
- **Resonator Networks:** VSA factorization через FHN oscillators (bridge DAF ↔ VSA)
- **Language interface:** encode text facts как VSA vectors, store в SDM
- **DAF integration:** SKS → VSA encoding (вместо symbolic obs → VSA)
