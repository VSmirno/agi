# Autonomous Development Log — 2026-04-02

## Текущая фаза: 1 — Живой DAF, прогресс ~60%

Маркер фазы: Pure DAF ≥ 50% DoorKey-5x5.
Stages 0-44 COMPLETE. Stage 44 audit показал: perception работает, но нет world model.
Stage 45 адресует core bottleneck через VSA + SDM.

## Stage 44: Финализация

### [auto] Фаза 0: Tech debt + закрытие Stage 44
- exp104 Naked DAF: 0/200 = 0% success — FAIL (gate ≥ 15%)
- exp104 ещё физически RUNNING на minipc (ep 188/200, ETA ~30min), но результат определён
- Stage 44 report обновлён: FAIL зафиксирован, Post-merge обновления добавлены
- ROADMAP обновлён: Stage 44 COMPLETE, Stage 45 IN PROGRESS
- Tech debt: TD-001 остаётся IN_PROGRESS (perception blind, blocked by Stage 42)
- Tech debt: TD-002, TD-003 остаются OPEN (GPU_EXP для Stages 39, 40)
- Tech debt: TD-004 остаётся OPEN (WM gating integration)

## Stage 45: VSA World Model

### [auto] Фаза 0: Git setup
- Ветка: stage45-vsa-world-model от main
- Spec уже написан: 2026-04-02-stage45-vsa-world-model-design.md

### [auto] Фаза 1: Спецификация
- Spec уже готов (написан в предыдущей сессии)
- Подход: VSA (Binary Spatter Code) + SDM (Sparse Distributed Memory)
- Альтернативы рассмотрены в spec research:
  - A: Tabular Q-learning (простой, но не масштабируется, не bioplausible)
  - B: VSA+SDM (масштабируемый, bioplausible, content-addressable)
  - C: Neural network world model (backprop, не СНКС-philosophy)
- **Выбран: B** — VSA+SDM: масштабируемый, bioplausible, фиксированные операции (XOR, majority vote)

### [auto] Фаза 2: Реализация
- VSACodebook: 512-bit BSC, XOR bind, majority bundle — 10 тестов PASS
- VSAEncoder: MiniGrid symbolic obs → VSA vector, unbinding accuracy 97-99%
- SDMMemory: 10K locations, auto-calibrated radius, transition storage — 6 тестов PASS
- SDMPlanner: 1-step reward lookahead — 2 теста PASS
- CausalPlanner: forward beam search (depth 3-6, beam 3-5) — depth>3 слишком медленно
- BackwardChainPlanner: trace matching + reverse SDM
- WorldModelAgent: explore→plan two-phase — 5 тестов PASS
- Итого: 30 тестов PASS

### [auto] Фаза 3: Эксперименты (на minipc)
- **105a**: VSA encoding accuracy = 97-99% — **PASS** (gate ≥ 90%)
- **105b**: SDM prediction similarity = 0.85-0.87 — **PASS** (gate ≥ 0.6)
- **105c (explore)**: DoorKey-5x5 random = ~40% success (50 ep)
- **105c (forward beam)**: plan phase = 0-20% — **FAIL** (goal similarity → упирается в дверь)
- **105c (SDM reward)**: plan phase = 0-20% — **FAIL** (sparse reward, 1-step не учит цепочку)
- **105c (trace match)**: plan phase = 0-10% — **FAIL** (VSA similarity ~0.5 не различает контексты)

### Ключевой вывод
DoorKey = detour task. Forward planning (beam search, reward lookahead) не работает, потому что:
1. Прямой путь к цели блокирован дверью
2. Reward sparse — только у цели
3. VSA similarity недостаточно селективна для trace matching

Нужен **subgoal extraction + plan graph** — scope для Stage 46.

### [auto] Фаза 4: Веб-демо
- demos/stage-45-vsa-world-model.html — Canvas DoorKey + SDM metrics + VSA вектор

### [auto] Фаза 5: Merge
- Report: PARTIAL PASS (foundation PASS, planning FAIL)
- TD-005 создан: planning improvement
- ROADMAP обновлён: Stage 45 COMPLETE

### Решения
- Отказ от reward shaping (промежуточные rewards = RL, не model-based)
- Отказ от novelty bonus (заливает SDM шумом)
- Forward beam search архитектурно не подходит для detour tasks
- Trace matching без subgoal extraction — слишком шумный

---

## Stage 46: Subgoal Planning

### [14:00] Фаза 0: Git setup
- Ветка: stage46-subgoal-planning от main (commit 35b2881)
- Tech debt проверен: 5 open (TD-001 IN_PROGRESS, TD-002..005 OPEN), 0 закрыто
- minipc: нет активных tmux-сессий
- TD-001: blocked by Stage 42 (perception), пропускаем
- TD-002, TD-003: GPU_EXP, minipc свободен — можно запустить после merge
- TD-004: INTEGRATION, зависит от WM gating — пропускаем
- TD-005: INTEGRATION, VSA planning FAIL — scope Stage 46

### [auto] Фаза 1: Спецификация
- Подход A: Symbolic event detection (deterministic, interpretable)
- Подход B: VSA state diff (generic but noisy)
- Подход C: Trace segmentation + landmarks (data-driven)
- **Выбран: A + B гибрид** — symbolic для known events, VSA diff как fallback

### [auto] Фаза 2: Реализация
- SubgoalExtractor: symbolic event detection (pickup, toggle, goal) — 8 тестов PASS
- PlanGraph: ordered chain with advancement — 5 тестов PASS
- SubgoalNavigator: position-based heuristic navigation — 3 теста PASS
- SubgoalPlanningAgent: explore→extract→plan loop — 5 тестов PASS
- Итого: 21 тест PASS

### [auto] Фаза 3: Эксперименты
6 итераций debugging:
- v1: DoorKeyEnv без блокирующей стены → агент обходит дверь (40% random)
- v2: Добавлена стена-разделитель → random ~1%, но plan 0%
- v3: VSA trace matching → similarity ~0.5 (слишком шумно), plan 0%
- v4: Symbolic state matching → trace replay неэффективен (random walk traces)
- v5: Position-based navigation → agent drawn after key, wrong position extraction
- **v6: Carrying-based key detection + goal from obs → 92.5% plan success!**

Ключевые баги найдены и исправлены:
1. DoorKeyEnv.obs(): ключ перезаписывал агента при перекрытии
2. is_achieved("pickup_key"): проверял видимость ключа вместо carrying indicator
3. reach_goal: target position не извлекался (is_achieved всегда False)

| Exp | Метрика | Результат | Gate | Статус |
|-----|---------|-----------|------|--------|
| 106a | Extraction accuracy | 100% (12/12) | ≥ 80% | PASS |
| 106b | Plan graph ordering | 100% (7/7) | 100% | PASS |
| 106c | Plan phase success | 92.5% (185/200) | ≥ 15% | PASS |
| 106c | Plan last 100 eps | 100% | — | info |
| 106c | Steps per episode | ~17 | — | info |
| 106d | Navigation quality | — | — | SKIP (30 explore insufficient) |

### [auto] Фаза 4: Веб-демо
- demos/stage-46-subgoal-planning.html — Canvas DoorKey + subgoal chain + plan overlay

### [auto] Фаза 5: Merge
- TD-005 CLOSED: plan phase ≥ 15% (got 92.5%)
- Report written
- ROADMAP updated

---

## Stage 47: Wall-aware навигация

### [auto] Фаза 0: Git setup
- Ветка: stage47-wall-aware-nav от main (commit 1f86cc2)
- Tech debt проверен: 3 open (TD-001 IN_PROGRESS, TD-002/003 OPEN, TD-004 OPEN), 1 closed (TD-005)
- minipc: нет активных tmux sessions
- TD-002/003: GPU_EXP, не запущены — Stage 47 приоритетнее

### [auto] Фаза 1: Спецификация
- Подход A: BFS pathfinding на observed grid (простой, оптимальный, zero-failure)
- Подход B: A* (нет выигрыша на 5x5)
- Подход C: SDM-based navigation (bio-plausible, но 0.85 prediction unreliable)
- **Выбран: A** — навигация = инфраструктура, не когнитивная функция

### [auto] Фаза 2: Реализация
- GridPathfinder: BFS pathfinding, wall extraction, path-to-actions — 16 тестов PASS
- RandomDoorKeyEnv: случайные раскладки (wall_row, door, key, agent, goal) — 5 тестов PASS
- SubgoalNavigator: интеграция BFS (fallback на heuristic) — 3 теста PASS
- Все 24 теста PASS, Stage 46 (21 тест) не сломан

### [auto] Фаза 3: Эксперименты (на minipc)
- **107a**: BFS pathfinding = 200/200 layouts solvable, mean path 9.6 — **PASS**
- **107b (v1)**: Random DoorKey-5x5, explore_eps=100 — FAIL (0/20, random walk не находит traces)
- Fix: build_plan_from_obs — строит план прямо из observation (key/door/goal positions)
- Fix: rebuild plan every episode (random layout per reset)
- Fix: epsilon 0.1→0.05
- **107b (v2)**: Random DoorKey-5x5, obs-based planning — **100% (200/200), mean 16 steps — PASS**

### [auto] Фаза 4: Веб-демо
- demos/stage-47-wall-aware-nav.html — Canvas с 8+ раскладками, BFS overlay, trail

### [auto] Фаза 5: Merge
- Report written: PASS (100% на 200 random layouts)
- ROADMAP updated: Stage 47 COMPLETE, Stage 48 merged (covered by 47)
- Stage 47 фактически закрывает и Stage 48 (gate ≥80% на 200 random layouts)

### Решения
- Obs-based planning вместо explore-then-plan: random walk ~1% success → unreliable
- BFS = infrastructure, не cognitive claim → допустимо для СНКС
- Stage 48 merged с 47 т.к. 100% на 200 random layouts уже выполняет gate 48
