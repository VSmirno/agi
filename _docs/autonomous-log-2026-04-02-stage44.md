# Autonomous Development Log — 2026-04-02 (Stage 44)

## Текущая фаза: 1 — Живой DAF, прогресс ~70%

Stages 0-43 COMPLETE. Stage 44: Foundation Audit — проверка корректности DAF-ядра.
Мотивация: scaffolding маскировал слабости, обнаружен скрытый дефект (perception blind),
нет гарантий отсутствия аналогичных багов в других слоях.

## Stage 44: Foundation Audit

### Фаза 0: Git setup
- Ветка: stage44-foundation-audit от main (commit 07e1361)
- Tech debt: 4 open (TD-001 IN_PROGRESS, TD-002/003/004 OPEN), 0 closed
- minipc: no active tmux sessions

### Фаза 1: Спецификация
- Spec: `docs/superpowers/specs/2026-04-02-stage44-foundation-audit-design.md`
- Spec review: APPROVED after 1 iteration (5 issues fixed)
- Commit: `07e1361`

### Фаза 2: Реализация
- 26 audit tests в `tests/test_stage44_audit.py`:
  - Phase 1.1 FHN Dynamics: 7 tests PASS
  - Phase 1.2 STDP: 6 tests PASS
  - Phase 1.2b Coupling: 4 tests PASS
  - Phase 1.3 SKS Detection: 4 tests PASS
  - Phase 1.4 Pipeline E2E: 3 tests PASS
  - Phase 1.5 Action Selection: 2 tests PASS
- **26/26 PASS** на CPU (local) и GPU (minipc)

### Фаза 3: Эксперименты

#### Exp 103: Golden Path (Phase 0)
- Среда: SimpleGridEnv 3×3, goal=(2,2), agent=(0,0)
- Агент: 50K FHN нод, SymbolicEncoder, NO extras
- **Результат: 17/30 = 56.7% success — GATE PASS (>50%)**
- Weight delta: 0.098 — GATE PASS (>0)
- **НО:** action selection почти random (epsilon-greedy без real signal)
- Random baseline на 3×3 с 20 шагами ≈ 30-40%
- **Вывод:** gate формально пройден, но improvement over random сомнителен

#### Exp 104: Naked DAF (Phase 2) — RUNNING
- Среда: MiniGrid-DoorKey-5x5-v0
- Агент: 50K FHN нод, SymbolicEncoder, NO WM/eligibility/curriculum/navigator
- 200 эпизодов, EpsilonScheduler 0.7→0.1
- Запущен на minipc (tmux session: exp104)
- **Ep 1:** success=False, steps=200, PE=0.126, SKS=64, time=153.3s
- **ETA:** ~8.5 часов (508 мин) для 200 эпизодов
- Gate: success >= 15% = PASS, 5-15% = PARTIAL, <5% with learning signal = LEARNING_SIGNAL

### Ключевые находки аудита

1. **FHN не осциллирует самостоятельно.** При default params (I_base=0.5, tau=12.5)
   одиночный FHN нейрон НЕ осциллирует — он excitable, не oscillatory.
   Система полагается на noise + coupling для ongoing spiking activity.

2. **Coupling propagation медленная.** С K=0.1 нужно ~5000 steps (0.5 model sec)
   для передачи активности на не-стимулированные ноды. Pipeline использует
   steps_per_cycle=100 (0.01 model sec) — за один цикл coupling почти не работает!

3. **Structural pruning меняет graph topology.** После обучения количество edges
   отличается от начального — нужно сравнивать mean weight, а не per-edge delta.

### Решения
- Спецификация написана до реализации, автоматический spec review
- Тесты FHN документируют что excitability ≠ oscillation
- Golden Path использует SimpleGridEnv (не MiniGrid) для изоляции от env dependencies
- Naked DAF exp104 запущен с MiniGridAdapter на minipc GPU
