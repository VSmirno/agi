# Stage 38_fix: Curiosity-Driven Action Selection

**Дата:** 2026-04-01
**Тип:** Fix-подэтап (TD-001)
**Приоритет:** Выше нового Stage 41

## Контекст

Exp97 GPU показал 0% success, 0 modulations. Root cause analysis выявил 3 бага:

1. **goal_embedding never set** — `run_episode()` не вызывает `set_goal_from_obs()`, навигатор отключен, агент = random
2. **Action space mismatch** — `CausalAgent.N_ACTIONS=5` (hardcoded), но MiniGrid DoorKey требует toggle=5 (7 действий). `MotorEncoder` создаётся с 5 действиями → IndexError при action≥5
3. **epsilon=0.3 too low** — недостаточно exploration на GPU с 50K нод

## Подходы

### A: Автоматический set_goal_from_obs
- В `run_episode()` вызывать `set_goal_from_obs(goal_obs)` из env
- **Проблема:** в DoorKey цель за дверью, агент не видит её до открытия двери. Не универсально.

### B: Goal-free navigator (lower similarity threshold)
- Снизить `min_similarity` до -1.0, navigator всегда активен
- **Проблема:** без goal_embedding навигатор возвращает random. Не решает корневую проблему.

### C: Curiosity/PE-driven action selection (ВЫБРАН)
- **Primary:** PE-biased exploration (из Stage 39 PredictionErrorExplorer)
- **Secondary:** count-based IntrinsicMotivation (fallback)
- **Optional:** goal-directed navigator (когда goal_embedding доступен)
- **Обоснование:** агент не нуждается в явной цели — он исследует через любопытство, STDP подкрепляет успешные пути через reward modulation

## Выбранный подход: C

### Изменения

#### 1. Fix action space (Bug #2)
- `PureDafAgent.__init__`: создать собственный `MotorEncoder(n_actions=config.n_actions)`, заменив `CausalAgent`'s default motor
- `PureDafConfig.n_actions` по умолчанию = 7 (MiniGrid full action space)

#### 2. Integrate PE exploration (Bug #1)
- Добавить `PredictionErrorExplorer` в `PureDafAgent`
- Добавить `EpsilonScheduler` в `PureDafAgent`
- Изменить `step()`:
  ```
  if random < epsilon:
      action = pe_explorer.select_with_bonus()  # PE-biased exploration
  elif goal_embedding AND current_embedding:
      action = navigator.select_action(...)     # goal-directed
  else:
      action = motivation.select_action(...)    # count-based curiosity
  ```
- `observe_result()`: записывать PE per action в pe_explorer
- `run_training()`: вызывать epsilon.step() после каждого эпизода

#### 3. Epsilon decay (Bug #3)
- `PureDafConfig`: добавить `epsilon_initial=0.7`, `epsilon_decay=0.95`, `epsilon_floor=0.1`
- `EpsilonScheduler` управляет epsilon вместо фиксированного значения

#### 4. IntrinsicMotivation update
- `observe_result()`: вызывать `motivation.update()` с PE данными
- Это позволяет count-based curiosity учитывать visited states

#### 5. Effective n_actions safety
- `_effective_n_actions = min(config.n_actions, motor.n_actions)`
- Все action selection clamp к [0, effective_n_actions - 1]

### Не меняется
- `CausalAgent` — оставить N_ACTIONS=5 (backward compat для старых Stage)
- `AttractorNavigator` — без изменений
- `DafCausalModel` — без изменений
- `IntrinsicMotivation` — без изменений (добавляется вызов update)

### Gate Criteria (TD-001)
- `success_rate >= 0.10` на DoorKey-5x5
- `modulations > 0` (STDP работает)
- Доп.: `exploration_coverage > random` (PE exploration лучше random)

## Позиция в фазе

**Фаза 1** — Живой DAF. Маркеры:
- [x] Pure DAF agent без scaffolding (Stage 38)
- [ ] **Curiosity-driven exploration** ← этот fix
- [x] Curriculum learning (Stage 39)
- [x] Learnable encoding (Stage 40)
- [ ] Temporal credit assignment (Stage 41)

Fix продвигает маркер "curiosity-driven exploration" — агент исследует через PE/novelty без внешней цели.

## Файлы
- `src/snks/agent/pure_daf_agent.py` — основные изменения
- `src/snks/experiments/exp97_pure_daf.py` — обновить config
- `tests/test_pure_daf_agent.py` — новые тесты
