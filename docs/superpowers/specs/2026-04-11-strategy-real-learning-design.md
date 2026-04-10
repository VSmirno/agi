# Стратегия: «настоящее обучение» поверх rule-based базы

**Дата:** 2026-04-11
**Контекст:** Stage 77a даёт partial PASS на Crafter (wall ~180), идеологически мы дрейфовали в expert system, пользователь требует **self-organizing parametric learning** поверх rules. Vision: Crafter → Minecraft → real-world AGI.
**Статус:** Phase A (research + branches). Phase B (план стейджей) в нижней секции.

---

## 0. TL;DR

1. **Dreamer-CDP (март 2026)** — уже опубликован и проверен на Crafter. **16.2% vs DreamerV3 14.5%, с prioritized replay 19.4%.** Reconstruction-free, JEPA-style, negative cosine similarity loss, stop-gradient target. **Ближайший известный ответ на наш точный вопрос.**
2. **Neuro-Symbolic Synergy (феврал 2026)** — rules + neural с alternating training. **+4-40pp улучшение при 35-60% меньше данных.** Symbolic scores модулируют neural distribution. Rules-from-residual синтезируются автоматически.
3. **LeJEPA (ноябрь 2025, LeCun+Balestriero)** — SIGReg вместо VICReg/EMA. **~50 строк кода**, no heuristics, no stop-gradient. Теоретически обоснован.
4. **V-JEPA 2 / V-JEPA 2-AC (июнь 2025, Meta)** — action-conditioned, но 300M params, pretraining на 1M часов видео, continuous 7D actions. **Не для нашего scale напрямую**, но дизайн-референс.
5. **OneLife (октябрь 2025)** — pure symbolic probabilistic laws от unguided exploration. **Тестирован на Crafter-OO.** Альтернатива без нейросети.
6. **Active Predictive Coding (Rao 2024)** — bio-plausible hierarchical predictive coding. Hypernetworks для state-action иерархий. **Research bet для будущего.**

**У нас в кодбейзе уже есть:**
- `src/snks/encoder/predictive_trainer.py` — базовый JEPA (pred + VICReg + SupCon). Не используется с Stage 75. Fork в Dreamer-CDP-style ≈ 200 строк.
- `src/snks/daf/` — полная нейродинамика (FHN + STDP + homeostasis). 119 тестов pass. Не интегрирована с Crafter.

---

## 1. Карта литературы — что где

### 1.1 Reconstruction-free world models на Crafter

| Paper | Год | Crafter score | Reconstruction? | Action | Идеология |
|---|---|---|---|---|---|
| **Dreamer-CDP** | 2026-03 | 16.2% (19.4% w/ PR) | **Нет** (JEPA-style) | Discrete + cont | **Матчит наши требования** |
| DyMoDreamer | 2025 | 832 (нормализованное) | Да | Discrete | ❌ reconstruction |
| MuDreamer | 2024 | соизмеримо | **Нет** | Both | ⚠️ предиктит value function |
| DreamerPro | 2022 | 4.7% | **Нет** | Both | ⚠️ устарел |
| DreamerV3 | 2023 | 14.5% | Да | Both | ❌ reconstruction |

**Вывод:** Dreamer-CDP — единственный на момент 2026-04-11 reconstruction-free world model, который:
- Побеждает DreamerV3 на Crafter
- Использует именно ту форму loss, которую мы хотим (neg cosine similarity + stop-gradient)
- Опубликован с полной архитектурой в доступном HTML preprint
- Был разработан в Zenke Lab (который вообще занимается bio-plausible learning)

### 1.2 Rules + learned correction

| Paper | Pattern | Применимость |
|---|---|---|
| **Neuro-Symbolic Synergy** (2026-02) | Neural + symbolic alternating, rules модулируют neural prob via `p̃ = p × exp(γE)` | **Прямо применимо** для нашего case |
| **ReDRAW** (2025-04) | Latent residual MLP на top of frozen base model, KL loss | Прямо применимо, но base — neural |
| **OneLife** (2025-10) | Probabilistic mixture programmatic laws + weight inference | Альтернатива без нейросети |
| SWMPO | Neurosymbolic FSM | Менее зрелый |

**Neuro-Symbolic Synergy критически важен:** он показал что паттерн «fine-tune neural **только** на трассах которые rules не покрывают, добавлять новые rules для остатков ошибок» даёт **+40pp** на Webshop при 60% меньше данных. Это буквально наш план Stage 77b + 78.

### 1.3 JEPA line (LeCun)

| Paper | Год | Что добавил |
|---|---|---|
| I-JEPA | 2023 | Joint embedding prediction без аугментаций |
| V-JEPA | 2024 | Видео → feature prediction в latent space |
| **V-JEPA 2** | 2025-06 | **Масштаб 300M + action-conditioned (V-JEPA 2-AC)** |
| **LeJEPA** | 2025-11 | **~50 строк кода, SIGReg, no heuristics** |

**V-JEPA 2-AC** — прямой ответ на вопрос «как JEPA с действиями». Но:
- 300M параметров, 1M часов видео pretraining — **не наш scale**
- Continuous 7D actions (end-effector) — не Crafter's 17 discrete
- Batch training на offline данных — не online
- **НО** action integration pattern (temporal interleaving `(a_k, s_k, z_k)`) — полезен как дизайн-референс

**LeJEPA** — наоборот, наш scale:
- Single hyperparameter
- Linear time/memory
- Бесплатно 50 строк
- **No EMA target, no stop-gradient, no schedulers** — это **именно то** что пользователь хочет от "без эвристик"
- Но: не action-conditioned из коробки, не RL-ready

### 1.4 Bio-plausible predictive coding

| Paper | Год | Что |
|---|---|---|
| **Active Predictive Coding** (Rao) | 2024 | Hypernetworks, state-action hierarchies, self-supervised + RL |
| Temporal PC for RL | 2025 | Addresses partial observability |
| Rao & Ballard | 1999 | Foundation: hierarchical prediction with local updates |

**Active Predictive Coding** — единственная современная работа которая явно комбинирует predictive coding + RL + hierarchical planning. Тестировалось на MNIST/FashionMNIST/Omniglot — не на Crafter. Research bet для будущего, не для текущего этапа.

### 1.5 Symbolic rule induction

| Paper | Год | Что |
|---|---|---|
| **OneLife** | 2025-10 | Probabilistic laws от unguided exploration, Crafter-OO |
| IDEA | 2024-08 | Induction/Deduction/Abduction с LLM |
| Logical Neural Networks | 2024 | Differentiable rule learning |

**OneLife** — закрывает правую границу spectrum'а: чисто символьный world model без нейросети. Использует LLM для синтеза кандидатов законов, weights θ_i учатся через L-BFGS на observed transitions. **Тестирован на Crafter**, +7.9% Rank@1 над PoE-World. Но: требует LLM доступа для синтеза законов.

---

## 2. Space решений — дерево ветвлений

```
                        «Real learning» на Crafter
                                   │
               ┌───────────────────┼───────────────────┐
               │                   │                   │
            Branch A            Branch B            Branch C
         Neural residual   Symbolic induction   Bio-substrate
        (Dreamer-CDP +        (OneLife +         (DAF / APC +
         Neuro-Symbolic       surprise           predictive
         Synergy)             accumulator)       coding)
               │                   │                   │
         Low risk            Mid risk            High risk
         Fast                Mid speed           Slow
         Published           Published partial   Research
```

### Branch A — Neural residual (Dreamer-CDP + Neuro-Symbolic Synergy)

**Идея:** существующие rules как base (mean predictor), small neural residual на top, обучается self-supervised predictive loss. Rules модулируют neural output (Neuro-Symbolic Synergy pattern). Rules-from-residual extraction — автоматическая по паттерну surprise accumulator.

**Что берём из литературы:**
- Loss function от Dreamer-CDP: `L_CDP = -cos(SG(u_t), û_t)` + `L_dyn, L_rep` (KL на latent states)
- Residual formulation от ReDRAW: `ẑ_{t+1} = rules_prediction + residual_MLP(z_t, a_t)`
- Alternating training от Neuro-Symbolic Synergy: neural fine-tuned только на «rules-uncovered» трассах
- Architecture от predictive_trainer.py: уже есть JEPAPredictor, VICReg, stop-gradient pattern

**Плюсы:**
- ✅ **Максимально опубликованный путь** — все компоненты из peer-reviewed работ
- ✅ **Идеологически чистый:** no labels, no external reward, pure intrinsic predictive signal
- ✅ **Переиспользует наш код** (`predictive_trainer.py` fork)
- ✅ **Online-capable:** gradient step per env step, replay buffer
- ✅ **Rules остаются:** они центральны, neural только дополняет
- ✅ **Incremental rule growth:** surprise → candidate → verification → promotion

**Минусы:**
- ⚠️ Используется backprop (хотя на **intrinsic** signal, не supervised labels — это LeCun-approved)
- ⚠️ Dreamer-CDP использует full Dreamer machinery (RSSM, reward head, continuation head); нужно адаптировать под нашу архитектуру без reward signal

**Compute:** CPU-friendly (Dreamer-CDP запускается на single V100). Small predictor ~100K params.

**Риски:**
- Residual может выучить всё вместо "только residual" → решается **малым размером** predictor (bottleneck constraint, как в ReDRAW)
- Collapse without proper regularization → решается LeJEPA SIGReg или VICReg

**Gate:** survival ≥200 на Crafter eval, wood ≥50%, без нарушений идеологии.

### Branch B — Symbolic induction (OneLife + surprise accumulator)

**Идея:** **никакой нейросети вообще**. World model = probabilistic mixture programmatic laws. Candidates генерируются через **surprise accumulator** (не LLM — наша замена их LLM-based synthesis). Weights θ_i учатся через MLE на observed transitions (L-BFGS).

**Что берём:**
- Law representation от OneLife: `law_i = (precondition_i, effect_i)`, `p(o|s,a;θ) ∝ ∏ φ_i^θ_i`
- Dynamic computation graph: only active laws get gradient credit
- Observable extractor: `ℰ: S → O` mapping
- Surprise accumulator (наш): per-context bucket → high residual → emit candidate law

**Плюсы:**
- ✅ **Zero neural network** — полностью символьно, максимально интерпретируемо
- ✅ Rules are THE world model, не добавление к нему
- ✅ Естественный continual learning: новые observations → гребенчатые веса
- ✅ **Нет collapse / catastrophic forgetting** — discrete rules

**Минусы:**
- ❌ **Нет LLM**, а OneLife использует LLM для генерации кандидатов. Наш "surprise accumulator" должен заменить LLM — **непроверенная замена**.
- ❌ Expressiveness ограничена грамматикой законов (precondition + effect). Conjunctive rules сложнее выразить.
- ⚠️ Не покрывает continuous/latent regularities которые не ложатся в atomic laws

**Compute:** Минимальный. L-BFGS быстрый, нет больших матриц.

**Риски:**
- Candidate generation без LLM — **open question**. Возможные подходы: template-based (rules по схеме "if X visible and Y action then Z delta"), surprise clustering, differential testing.
- Expressiveness: сложные правила могут потребовать многопараметрических preconditions

**Gate:** то же что Branch A + OneLife-style evaluation (Rank@1 на test states).

### Branch C — Bio-plausible (DAF revival + Active Predictive Coding)

**Идея:** Вернуть DAF substrate (уже есть) как parametric representation learner, использовать Active Predictive Coding paradigm для обучения через **локальные** правила (STDP, Hebbian). Rules остаются как high-level scaffolding, DAF работает как learnable conditional dynamics predictor.

**Что берём:**
- DAF engine (свой, в `src/snks/daf/`) — 119 тестов проходят
- Active Predictive Coding architecture (Rao 2024) — hierarchical state-action models
- Hypernetworks для композиционных representations
- Local learning rules: STDP + eligibility traces (уже есть)

**Плюсы:**
- ✅ **Самый идеологически чистый** — настоящее bio-plausible learning, никакого backprop
- ✅ Масштабируется от Crafter к неограниченному (DAF engine поддерживает 50K+ nodes на GPU)
- ✅ **Уже построен** substrate (не нужно писать с нуля)
- ✅ Hypernetworks дают hierarchical abstraction для планирования на разных timescales

**Минусы:**
- ❌ **Нет опубликованного precedent** на Crafter-scale environments. APC тестировался только на MNIST/FashionMNIST.
- ❌ DAF + Crafter bridge никогда не строился. Нужен projection layer CNN features → DAF nodes.
- ❌ Stage 44 audit показал что FHN в текущей конфигурации excitable, не oscillatory. Нужно тестировать oscillatory режим (I_base > 1) — никогда не делалось.
- ❌ **Мой спайк-тест показал что наивная интеграция не работает** — нужно несколько итераций.

**Compute:** GPU mandatory для DAF engine, но есть minipc.

**Риски:**
- Research risk высок — нет гарантии что DAF может учить то что нам нужно
- Длительная отладка substrate prior to agent integration
- Возможно обнаружить принципиальное ограничение после месяца работы

**Gate:** Phase 1 — DAF substrate learns synthetic (state, action) → delta via STDP на contrived task (bio-plausible alternative to my spike test). Phase 2 — integration with Crafter.

---

## 3. Сравнительная таблица

| Критерий | Branch A (Neural residual) | Branch B (Symbolic OneLife) | Branch C (DAF bio) |
|---|---|---|---|
| **Опубликованный precedent на Crafter** | ✅ (Dreamer-CDP, март 2026) | ⚠️ (OneLife на Crafter-OO) | ❌ |
| **Self-supervised (no labels)** | ✅ | ✅ | ✅ |
| **Бо́льшая часть кода уже есть** | ✅ (predictive_trainer.py) | ⚠️ (частично в ConceptStore) | ⚠️ (substrate есть, bridge нет) |
| **Bio-plausible** | ⚠️ (intrinsic signal, но backprop) | ✅ (no neural) | ✅ (STDP local) |
| **Time to Gate 1** (≥200 survival) | ~1-2 недели | ~2-3 недели | ~1-2 месяца |
| **Compute** | CPU ok, GPU helpful | CPU trivial | GPU mandatory |
| **Interpretability** | ⚠️ (residual opaque) | ✅ (полностью) | ⚠️ (spiking patterns) |
| **Scaling Crafter → Minecraft** | ✅ (V-JEPA 2 pattern extends) | ⚠️ (grammar expressiveness limit) | ✅ (если substrate заработает) |
| **Risk of failure** | Low | Mid | High |
| **Учит ли conjunctive rules?** | ✅ (MLP learns it) | ⚠️ (зависит от grammar) | ✅ (STDP correlational) |

---

## 4. Рекомендация — **hybrid trajectory**

Не одна ветка, а **последовательность** веток где каждая использует следующую как safety net:

**Phase 1 (1-2 недели): Branch A — доказать что идея работает**

Построить Dreamer-CDP-style residual JEPA над ConceptStore rules. Цель: **закрыть Gate 1 на Crafter** как fastest proof. Используется максимально опубликованный код и наработки.

Если работает → у нас есть reliable baseline, базовая функциональность, и можно экспериментировать дальше.
Если не работает → понимаем конкретно почему и откатываемся к Branch B или C.

**Phase 2 (2-4 недели после Phase 1): добавить Branch B — rule induction**

Над работающей Branch A добавить автоматическое извлечение rules из neural-detected surprise clusters. Паттерн от Neuro-Symbolic Synergy. Цель: **neural постепенно уступает место rules**, интерпретируемость растёт.

После этого — система учится сама обновлять свой textbook, что и есть "real learning" в широком смысле.

**Phase 3 (grant-proposal, параллельно Phase 1-2): Branch C — research bet**

Параллельно тренируем DAF substrate на synthetic tasks (bio-plausible spike test, **другой** чем тот что я делал). Цель: empirically определить способен ли substrate учить то что нам нужно, прежде чем тратить большие ресурсы.

Если DAF substrate **работает** на synthetic → это наш путь к Minecraft/AGI с уникальной архитектурой которую можно показать в грантовой заявке.
Если **не работает** → мы явно знаем что backpropagation path нам нужен для follow-through, и не тратим время на нереалистичные research bets.

---

## 5. Progressive expansion: Crafter → Minecraft → AGI

### Crafter (текущий этап)
- Цель: работающий rule-based + learning agent с Gate 1 закрытым
- Метрика: survival ≥200, wood ≥50%, craft chains executed
- Timeline: 1-2 месяца

### Phase 2 envs: другие 2D мири
- Новые discrete-action envs (MiniHack, NetHack, custom)
- Цель: доказать что архитектура переносится
- Timeline: 1 месяц after Crafter Gate closed

### Minecraft
- Compute requirement растёт значительно
- V-JEPA 2-AC pretraining pattern: используем VPT dataset (2000+ часов human gameplay)
- H-JEPA hierarchical для multi-scale planning (crafting chains, exploration, combat)
- Этот этап — **grant-proposal material**: показываем что архитектура работает end-to-end на Crafter, предлагаем masштабировать на Minecraft
- Timeline: 3-6 месяцев

### Real world / AGI
- H-JEPA с 3+ уровнями иерархии
- Bio-plausible substrate (если Branch C окажется viable)
- Hypothetical long-term

---

## 6. Что мы **НЕ** делаем

1. ❌ **DreamerV3 / DyMoDreamer full neural world model.** Reconstruction-based, заменит rules, не наш идеологический путь.
2. ❌ **V-JEPA 2 из коробки.** Wrong scale, wrong domain, wrong actions.
3. ❌ **DeepProbLog / differentiable rules.** Differentiating through rules unstable, batch-oriented.
4. ❌ **LLM в живой loop'е.** Нет API доступа на minipc, не scalable, introduces unpredictable latency.
5. ❌ **AND/OR conjunction grammar в textbook.** Расширение textbook expressivity для покрытия eval gaps — нарушение `feedback_self_induced_rules.md` (textbook = parent knowledge).
6. ❌ **Supervised backprop на labeled loss.** Было рассмотрено, отвергнуто пользователем.

---

## 7. Links to papers (verified 2026-04-11)

**Dreamer-CDP:**
- [arxiv:2603.07083](https://arxiv.org/html/2603.07083v1) — full HTML
- [Zenke Lab announcement](https://zenkelab.org/2026/03/new-paper-dreamer-cdp-reconstruction-free-world-models-for-reinforcement-learning/)

**LeJEPA:**
- [arxiv:2511.08544](https://arxiv.org/abs/2511.08544)
- [github.com/rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa)

**Neuro-Symbolic Synergy:**
- [arxiv:2602.10480](https://arxiv.org/html/2602.10480)

**ReDRAW:**
- [arxiv:2504.02252](https://arxiv.org/abs/2504.02252)

**V-JEPA 2:**
- [arxiv:2506.09985](https://arxiv.org/abs/2506.09985)
- [Meta blog](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)

**OneLife:**
- [arxiv:2510.12088](https://arxiv.org/html/2510.12088)

**Active Predictive Coding:**
- [arxiv:2210.13461](https://arxiv.org/abs/2210.13461)
- [MIT Neural Computation 2024](https://direct.mit.edu/neco/article/36/1/1/118264/Active-Predictive-Coding-A-Unifying-Neural-Model)

**TWISTER (контекст):**
- [OpenReview ICLR 2025](https://openreview.net/forum?id=YK9G4Htdew)
- [github.com/burchim/TWISTER](https://github.com/burchim/TWISTER)

---

## 8. Status этого документа

Phase A (research + strategy) — COMPLETE.
Phase B (step-by-step implementation plan) — см. `ROADMAP.md` update (Stage 78+).
Phase C (execution) — начнётся после утверждения этой стратегии пользователем.
