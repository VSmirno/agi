# СНКС — Roadmap v3 (Real Learning на top of Rules)

**Последнее обновление:** 2026-04-11
**Статус:** Stage 77a COMPLETE (partial PASS, wall ~180) → Stage 78a COMPLETE (FAIL, DAF residual dropped) → **Stage 78b COMPLETE (PASS, synthetic gate met)** → Stage 79 NEXT (surprise accumulator + rule nursery, no-LLM)
**Стратегия v3:** [docs/superpowers/specs/2026-04-11-strategy-real-learning-design.md](docs/superpowers/specs/2026-04-11-strategy-real-learning-design.md)
**История v2:** [docs/superpowers/specs/2026-04-02-roadmap-v2-design.md](docs/superpowers/specs/2026-04-02-roadmap-v2-design.md)

## Vision

Crafter — минимальная, но почти полноценная копия мировой модели. Ограниченный домен для наших ресурсов. Прицел: Crafter → другие 2D envs → Minecraft → grant → hardware → AGI.

**Architectural invariant:** rules (textbook) как база + self-organizing parametric learning поверх. Не reward shaping, не supervised backprop на labels. Только self-supervised predictive signals + symbolic priors.

---

## Принципы

- **Модульная bio-inspired архитектура:** DAF (сенсорная кора) + VSA (ассоциативная кора) + SDM (гиппокамп) + Subgoal Planner (PFC) + Language (Брока/Вернике)
- Каждый milestone — чёткий gate-критерий (достигнут или нет)
- Каузальное планирование через subgoals, НЕ reward shaping
- Язык = интерфейс, НЕ основа мышления
- Stage 46 = реальный фундамент, Блоки 1-5 = exploration phase

---

## Milestones

### M1: Генерализация ⚠️ SYMBOLIC BASELINE
**Gate:** ≥80% random DoorKey-5x5, ≥60% MultiRoom-N3
**⚠️ Stages 47-49 = pure symbolic BFS, no learned components. Gates passed via hardcoded pathfinding.**

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 47 | Wall-aware навигация | COMPLETE ⚠️ SYMBOLIC | 100% DoorKey-5x5 — BFS pathfinding, no learning |
| 48 | Random layouts | COMPLETE ⚠️ SYMBOLIC | merged with 47 |
| 49 | Multi-room | COMPLETE ⚠️ SYMBOLIC | 100% MultiRoom-N3 — BFS + door toggle, no learning |

### M2: Языковой контроль ⚠️ SYMBOLIC BASELINE
**Gate:** ≥70% success с языковой инструкцией
**⚠️ Stages 50-51 = regex parsing + fixed random VSA vectors, no learned encoding.**

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 50 | Reconnect language pipeline | COMPLETE ⚠️ SYMBOLIC | regex parsing, fixed binary vectors (not trained) |
| 51 | Language-guided planning | COMPLETE ⚠️ SYMBOLIC | BFS + regex, no SDM/VSA learning |

### M3: Концепция доказана ⚠️ SYMBOLIC BASELINE
**Gate:** M1 + M2 + интеграция ≥50% random MultiRoom-N3 с инструкцией + R1 вердикт

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 52 | Integration test | COMPLETE ⚠️ SYMBOLIC | env detection + delegation to symbolic agents |
| 53 | Architecture report | COMPLETE (2026-04-02) | R1 negative, M4 plan |

### M4: Масштаб
**Gate:** новый env 5+ типов объектов, partial observability, subgoal chains 5+
**R1 решение:** негативный вердикт → DAF = perception only

| Stage | Название | Статус | Gate |
|-------|----------|--------|------|
| 54 | Partial Observability | COMPLETE ⚠️ SYMBOLIC | 100% DoorKey — SpatialMap + BFS, no learning |
| 55 | Exploration Strategy | COMPLETE ⚠️ SYMBOLIC | 100% MultiRoom — FrontierExplorer + BFS |
| 56 | Complex Environment | COMPLETE ⚠️ SYMBOLIC | 99.5% PutNext — state machine + BFS |
| 57 | Long Subgoal Chains | COMPLETE ⚠️ SYMBOLIC | 40% KeyCorridor — ChainPlanner + BFS |
| 58 | SDM Retrofit | COMPLETE ⚠️ NEGATIVE (2026-04-03) | SDM не добавляет value — heuristic=100% на DoorKey, honest ablation |
| 59 | VSA Causal Induction | COMPLETE (2026-04-03) | **100% generalization unseen colors**, bind(X,X)=identity proof |
| 60 | World Model via Demos | COMPLETE (2026-04-03) | **100% QA-A/B/C**, causal world model из 5 синтетических демо |
| 61 | Demo-Guided Agent | COMPLETE (2026-04-03) | **100% DoorKey + 100% LockedRoom**, causal planning + BFS, ablation delta=100% |
| 62 | CLS World Model | COMPLETE (2026-04-04) | **100% QA L1-L4**, neocortex+hippocampus+consolidation, 2-tier CLS |
| 63 | Abstraction + Crafter | COMPLETE (2026-04-04) | **100% Crafter QA**, 25 auto-categories, 97% MiniGrid regression |

### Roadmap v4: Scaffolding Removal (2026-04-04)
**Crafter = основной домен.** MiniGrid — только regression.

| Stage | Убираем | Gate |
|-------|---------|------|
| 64 | Синтетику → демо + exploration | COMPLETE (2026-04-04) — 93% Crafter QA, 97% MG, 9 discovered, 5 taught, 0 synthetic |
| 65 | 100% уверенность → uncertainty | COMPLETE (2026-04-04) — Brier=0.12, ρ=0.17, ideal calibration curve, 87% Crafter, 94% MG |
| 66 | Symbolic features → пиксели | COMPLETE (2026-04-06) — 100% Crafter QA (7/7) from pixels, prototype memory k-NN, JEPA+SupCon encoder |
| 67 | Symbolic near → CNN near | COMPLETE (2026-04-06) — 99% smoke, 100% QA gate, 100% regression. NearDetector (near_weight=1.0), CrafterPixelEnv без _to_symbolic() |
| 68-71 | Pixel navigation, scenario curriculum, text-visual integration | COMPLETE (2026-04-07) |
| 72 | IDEOLOGY pivot: CNN=V1, continuous learning | COMPLETE (2026-04-07) |
| 73 | Autonomous craft: backward chaining + grounding | COMPLETE (2026-04-07) |
| 74 | Homeostatic agent | COMPLETE (2026-04-08) — axis bug fix, survival 173 |
| 75 | Per-tile perception | COMPLETE (2026-04-09) — 82% per-tile acc, survival 178, coord mapping fix |
| 76 | Continuous memory learning (SDM) | COMPLETE (2026-04-10) — reverted, memory ≡ bootstrap, architecture review |
| 77a | ConceptStore forward sim + MPC | **COMPLETE partial PASS (2026-04-10)** — 140/140 tests, wall 180 |

### Roadmap v5: Real Learning on Top of Rules (2026-04-11)
**Новая цель**: self-organizing parametric learning поверх rules. Стратегия v5 — см. [strategy spec](docs/superpowers/specs/2026-04-11-strategy-real-learning-design.md).

**2026-04-11 revision after Stage 78a FAIL:** DAF-as-residual novelty dropped (substrate does not carry conditional dynamics in any of 7 tested regimes — see `docs/reports/stage-78a-report.md`). Branch A neural residual is now the primary path using the published Dreamer-CDP pattern as scaffolding. Two of three novelties remain (no-LLM surprise rule induction, three-category ideology).

**Stage 78a — Fair DAF spike test — COMPLETE FAIL (2026-04-11)**

| Stage | Название | Gate | Status |
|-------|----------|------|--------|
| 78a | Fair DAF spike on synthetic (7 regimes × linear baseline) | At least one DAF regime ≤ 1.2× baseline conj health mse with disc ≥ 0.10 | **FAIL 2026-04-11** — all 7 regimes 10–12× worse than baseline |

Key findings: R3 (oscillatory default tau, Stage 44 R1.1 untested case) saturates — disc=0.064. R4 (fast tau) rescues discrimination to 0.309 but MLP still can't extract conditional signal. SKS cluster discovery: 0 clusters across R5/R7 (no attractor structure forms at 5K nodes × 2000 steps). STDP warmup adds ~2% (R4 vs R6) — no lift.

**Branch A (primary now)** — Neural MLP residual over ConceptStore rules, Dreamer-CDP pattern

| Stage | Название | Gate | Est |
|-------|----------|------|-----|
| 78b | MLP Residual Trainer (Dreamer-CDP scaffolding) | Forked from `predictive_trainer.py`: neg cosine similarity loss, stop-gradient target, small MLP residual on top of `simulate_forward` prediction. Synthetic test: учит conjunctive rule за <5 эпох, residual bottleneck kept small | **COMPLETE PASS 2026-04-11** — conj_health_mse=0.0064 (gate ≤0.008), gen_health_mse=0.0002 (gate ≤0.012). 9/9 unit tests. Report: `docs/reports/stage-78b-report.md` |
| 79 | Surprise Accumulator + Rule Nursery (no-LLM) | Per-context surprise bucketing with two-level context key (coarse (visible, action), refined by body quartiles). Template-based candidate rule emission from mean observed delta. Verification gate → promotion to ConceptStore.learned_rules. Sketch: `docs/superpowers/specs/2026-04-11-stage79-surprise-accumulator-sketch.md` | 2 нед |
| 80 | Residual + Rule Nursery Crafter integration | Both mechanisms live in MPC tick loop. Eval on Crafter: survival ≥200 (close Stage 77a wall), wood ≥30% | 1-2 нед |
| 81 | Alternating training (Neuro-Symbolic Synergy pattern) | Residual fine-tune только на rule-uncovered трассах, rules bump confidence on trivially-matched traces. Eval: stable gate с growing rule count per episode | 1 нед |

**Branch B (deferred — only if Branch A stalls)** — Symbolic OneLife-style induction

| Stage | Название | Gate | Est |
|-------|----------|------|-----|
| 78B | Symbolic law grammar | (precondition, effect) pairs with weights θ_i | 1 нед |
| 79B | Candidate generation w/o LLM | Template-based + surprise clustering | 2 нед |
| 80B | Law weight learning via L-BFGS on MLE | OneLife-style observable prediction | 1 нед |

**Branch C (research bet, not on Gate 1 critical path)** — DAF / Active Predictive Coding

Stage 78a empirically closes the "DAF as learnable dynamics predictor"
question at 5K-node scale — substrate does not carry conditional
signal. Larger networks (50K+), phase coherence readouts, and different
projection layers remain as follow-ups for the grant-track research bet
but are not required for the Crafter Gate 1 path.

### M5: Transfer to Second Environment (after Branch A Gate closed)
**Gate:** сохранение архитектуры + работающая агент на 2-ом discrete env (MiniHack, NetHack, or custom).

| Stage | Название | Gate |
|-------|----------|------|
| 82 | Environment abstraction layer | Unified env interface, rules loader per env |
| 83 | Transfer test | Residual JEPA works on new env с минимальными изменениями |

### M6: Scaling — Minecraft (requires more compute / grant)
**Gate:** end-to-end working agent in Minecraft. Grant proposal material.

| Stage | Название | Gate |
|-------|----------|------|
| 84 | VPT dataset offline pretraining | V-JEPA 2-AC style на gameplay videos |
| 85 | H-JEPA hierarchical (crafting / exploration / combat timescales) | Multi-level predictors |
| 86 | Minecraft basic survival | Day 1 gameplay: wood, food, shelter |

### M7: AGI vision (long-term, hardware-dependent)
- H-JEPA 3+ levels
- Active Predictive Coding if Branch C proved viable
- Open-ended discovery
- Real-world sensorimotor

### Architecture Pivot (Stage 59, 2026-04-03)
**Stages 47-58: символический BFS + SDM = тупик.** SDM паразитирует на BFS — либо BFS решает всё (SDM не нужен), либо BFS не справляется (SDM не получает данных).

**Новый подход: обучение через демонстрации, не exploration.**
- Stage 59: VSA bind(X,X)=identity ✅ — few-shot generalization доказана (3 demos → 100% unseen colors)
- Stage 60: Causal world model ✅ — 5 правил из синтетических демо, 3 уровня QA PASS
- Stage 61: Agent использует world model — exploration только для layout, не для rule discovery
- Аналогия: младенца УЧАТ, а не он сам открывает правила за миллион попыток

### M5: Автономия (vision, детализация после M4)
Self-directed goals, compositional subgoals, meta-cognition

---

## Research-трек R1: Осцилляторная динамика
**Вердикт: НЕГАТИВНЫЙ** (Stage 53, 2026-04-02)

FHN в текущей конфигурации — excitable regime, не oscillatory. Coupling timescale mismatch 50×. SKS формируются через SDR injection, не coupling. DAF остаётся как perception layer.

| ID | Вопрос | Статус |
|----|--------|--------|
| R1.1 | Oscillatory FHN (I_base > 1.0) | NEGATIVE — I_base=0.5 = excitable (Stage 44 audit) |
| R1.2 | Timescale fix (steps_per_cycle=5000) | NOT APPLIED — 50× slowdown, impractical |
| R1.3 | SKS quality при oscillatory params | NOT MEASURED — SKS = input-driven |
| R1.4 | Downstream impact на VSA+SDM | N/A — VSA+SDM work without DAF oscillations |
| R1.5 | GPU tech debt (TD-002, TD-003) | SUPERSEDED — modular architecture replaced Pure DAF |

---

## Exploration phase (Stages 0-46)

### Блок 1: Foundation (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 0-6 | Core DAF + Agent | COMPLETE |
| 7-9 | Language + Prediction | COMPLETE |
| 10-13 | Hierarchical Planning | COMPLETE |
| 14-15 | Embodied Agent + Debt | COMPLETE |
| 16-17 | Memory + Validation | COMPLETE |

### Блок 2: Language Pipeline (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 19-24 | Grounding, Compositional, Verbalizer, QA, Scaffold, BabyAI | COMPLETE |

### Блок 3-5: Reasoning, Self-Directed, Social (COMPLETE)
| Stage | Название | Статус |
|-------|----------|--------|
| 25-35 | Goal Composition → Integration Demo | COMPLETE (2026-03-31) |

### Блок 6: Scaling & Real Learning (COMPLETE)
| Stage | Название | Статус | Результат |
|-------|----------|--------|-----------|
| 36 | Spatial Abstraction | COMPLETE | 100% DoorKey-16x16 |
| 37 | Multi-Room Navigation | COMPLETE | 96% MultiRoom-N4 |
| 38 | Pure DAF Agent | COMPLETE | Чистый DAF pipeline, 12% DoorKey-5x5 |
| 39 | Curriculum Learning | COMPLETE | EpsilonDecay + PE exploration |
| 40 | Learnable Encoding | COMPLETE | HebbianEncoder +14% SDR |
| 41 | Temporal Credit Assignment | COMPLETE | Eligibility traces λ=0.92 |
| 42 | Spatial Representation | COMPLETE | Symbolic+CNN encoders |
| 43 | Working Memory | COMPLETE | Selective reset, sustained activation |
| 44 | Foundation Audit | COMPLETE | 26/26 PASS, naked DAF 0% — world model needed |
| 45 | VSA World Model | COMPLETE | 97% encoding, planning FAIL (detour) |
| 46 | Subgoal Planning | COMPLETE | **92.5% DoorKey-5x5** |

---

## Как обновлять этот файл

При завершении каждого stage:
1. Обновить статус в таблице milestone (COMPLETE + дата)
2. Добавить краткий результат
3. Commit с сообщением `docs: update ROADMAP — Stage N COMPLETE`
