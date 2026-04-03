# Stage 58: SDM Retrofit — Learned Agent for DoorKey with Partial Obs

**Дата:** 2026-04-03
**Milestone:** M4 — Масштаб (пятый этап)
**Gate:** ≥30% DoorKey-5x5 partial obs (200 seeds) + SDM ≥1000 transitions
**Тип:** RETROFIT — первый learned stage после 11-этапного symbolic drift

---

## Позиция в фазе

**M4 маркеры:**
- [x] Partial observability (Stage 54 — symbolic baseline)
- [x] Exploration strategy (Stage 55 — symbolic baseline)
- [x] Complex environment 5+ object types (Stage 56 — symbolic baseline)
- [x] Long subgoal chains 5+ (Stage 57 — symbolic baseline)
- [ ] **SDM scaling ≥1000 + learned agent** ← этот этап (RETROFIT)
- [ ] Transfer learning (Stage 59)
- [ ] Integration test (Stage 60)

**Архитектурный контекст:** Stages 47-57 = pure symbolic BFS. Stage 58 = pivot point, возврат к СНКС learned pipeline. Символические агенты остаются как upper-bound baselines.

---

## Pipeline integration

```
obs 7×7 partial → SpatialMap.update() → AbstractStateEncoder (VSA) → SDM write/read
                                                                          ↓
                                        FrontierExplorer (exploration)  SDM reward lookahead
                                               ↓                          ↓
                                        exploration actions           planning actions
                                               ↓                          ↓
                                           SDM.write()              action selection
```

**СНКС компоненты задействованы:**
- **VSA:** AbstractStateEncoder — кодирует абстрактные features (has_key, door_state, agent_pos) в 512-dim binary vector
- **SDM:** SDMMemory — хранит (state, action) → (next_state, reward) transitions из exploration
- **Planner:** SDMPlanner — 1-step reward lookahead через SDM + BackwardChainPlanner (trace matching)
- **SpatialMap:** из Stage 54 — накопление partial obs в полную карту (perception helper)

**НЕ задействованы (с обоснованием):**
- DAF: R1 негативный вердикт, excitable regime бесполезен
- Language: DoorKey не требует инструкций

---

## Архитектура

### Ключевая идея: Abstract State VSA

Stage 45 VSAEncoder кодирует raw 7×7 obs → слишком шумный для SDM (49 клеток × 3 канала = 147 facts). SDM не может обобщать с таким количеством деталей.

**Решение: AbstractStateEncoder** — кодирует компактные high-level features:

```python
class AbstractStateEncoder:
    """Encode DoorKey state as compact VSA vector.
    
    Features (role-filler pairs):
    - agent_pos: (row, col) — discretized position
    - has_key: bool
    - door_state: locked / closed / open  
    - key_visible: bool (known on spatial map)
    - door_visible: bool
    - goal_visible: bool
    - exploration_progress: fraction of map explored
    """
```

Это даёт ~7-10 VSA facts вместо ~147 → SDM может реально обобщать между similar states.

### Два режима работы

**1. Exploration phase (первые N эпизодов):**
- FrontierExplorer из Stage 54 выбирает actions (эффективное покрытие карты)
- Каждый transition записывается в SDM: `sdm.write(state_vsa, action_vsa, next_state_vsa, reward)`
- Дополнительно: автоматическое toggle дверей, pickup ключей (symbolic helpers)
- Цель: наполнить SDM transitions, обнаружить reward signal

**2. Planning phase (после N эпизодов):**
- SDMPlanner.select(state) — reward-weighted action selection через SDM read
- BackwardChainPlanner — matching текущего state к successful traces
- Fallback: FrontierExplorer если SDM confidence < threshold
- Symbolic helpers остаются (toggle doors, pickup keys)

### Symbolic helpers — осознанный компромисс

Агент использует symbolic helpers для **низкоуровневых действий**:
- Toggle closed door when facing it
- Pickup key when facing it and not carrying
- These are reflexes, not planning

**Planning** (куда идти, в каком порядке) — через SDM. Это честный learned компонент: агент УЧИТСЯ что "state с has_key=True + near door → toggle → reward path opens".

### Wrapper: SDMDoorKeyAgent

```python
class SDMDoorKeyAgent:
    spatial_map: SpatialMap          # perception accumulator
    encoder: AbstractStateEncoder    # state → VSA
    sdm: SDMMemory                   # transition memory
    planner: SDMPlanner              # reward lookahead
    backward: BackwardChainPlanner   # trace matching
    explorer: FrontierExplorer       # exploration strategy
    
    def select_action(self, obs_7x7, agent_col, agent_row, agent_dir):
        # 1. Update spatial map
        self.spatial_map.update(obs_7x7, ...)
        
        # 2. Symbolic reflexes (toggle door, pickup key)
        reflex = self._check_reflexes(obs_7x7)
        if reflex is not None:
            return reflex
        
        # 3. Encode abstract state
        state_vsa = self.encoder.encode(
            agent_row, agent_col, has_key, door_state, 
            key_known, door_known, goal_known, exploration_pct
        )
        
        # 4. Choose action via SDM or exploration
        if self._exploring:
            action = self.explorer.select_action(...)
        else:
            action = self._sdm_select(state_vsa)
        
        # 5. Record transition in SDM
        if self._prev_state is not None:
            self.sdm.write(self._prev_state, action_vsa, state_vsa, reward)
        
        return action
```

---

## Implementation plan

1. **AbstractStateEncoder** — compact VSA encoding of DoorKey state features
2. **SDMDoorKeyAgent** — integrates SpatialMap + VSA + SDM + FrontierExplorer
3. **SDMDoorKeyEnv** — wrapper for real MiniGrid DoorKey-5x5 with partial obs
4. **Tests** — unit + integration
5. **Learning phase** — exploration episodes on minipc, measure SDM metrics
6. **Experiments** — 200 seeds DoorKey-5x5, compare with symbolic baseline

---

## Gate criteria

| Metric | Gate | Measurement |
|--------|------|-------------|
| Success rate (DoorKey-5x5 partial obs, 200 seeds) | ≥30% | After N exploration episodes |
| SDM unique transitions | ≥1000 | sdm.n_writes after exploration |
| Random baseline | report | Success rate with 0 exploration episodes |
| Symbolic baseline comparison | report | vs Stage 54 (100%) |
| Exploration episodes needed | report | Min episodes for ≥30% |

---

## Risks

1. **SDM interference:** Many similar states (agent at different positions) → SDM may confuse transitions
2. **Abstract state too lossy:** Losing position info → SDM can't distinguish "near key" from "near door"
3. **Exploration budget:** 50 episodes may not be enough for 200 random seeds (each seed = different layout)
4. **Reward sparsity:** DoorKey gives reward only at goal — SDM needs many successful episodes to learn
