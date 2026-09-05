# Roadmap SNKS AGI

**Версия:** 2  
**Дата:** 2026-04-17  
**Статус:** Living document. Пересматривать после закрытия каждой крупной фазы.

> Этот roadmap выводится из `docs/IDEOLOGY.md` и проверяется против
> `docs/CONCEPT_SUCCESS_CRITERIA.md`.
> Любой claim про “архитектурный прогресс”, сделанный по ходу roadmap-а,
> дополнительно должен проходить `docs/ANTI_TUNING_CHECKLIST.md`, чтобы
> Crafter не стал скрытой конечной целью вместо proving ground.
>
> Вопрос теперь не "какой следующий stage?", а
> **"какое доказательство работоспособности концепции ещё отсутствует?"**

---

## Текущая позиция

Проект вышел из фазы локальных исправлений и дошёл до более честной архитектурной картины:

- `facts / mechanisms / experience / stimuli` разделены заметно лучше, чем на Stage 74-81
- post-mortem, death hypotheses и textbook promotion механически работают
- ideologically-clean baseline сильнее прежних гибридных вариантов
- главный bottleneck теперь понятен: **мир моделируется недостаточно динамически, а promotion пока переносит корреляции лучше, чем причинно полезное знание**

### Последний подтверждённый статус

**2026-09-05 — Learning-core development pilot: PARTIAL (реализация), без
подтверждённого выигрыша.** Добавлен отдельный обучаемый цикл опыта/динамики/
планирования. Первый HyperPC pilot: success 0/4, sensor prediction хуже
persistence, real/shuffled actions практически равны. Это не закрывает ни одну
proof obligation ниже; следующий шаг — локализация отсутствия полезного
action-conditioned прогноза. [Результаты и ограничения](reports/learning-core-pilot.md).

Follow-up локализовал ошибку абсолютного sensor prediction и проверил residual
profile на одном сохранённом replay. В controlled fixture delta-real получает
wood раньше/больше, action ablation роняет результат до нуля, а наблюдавшееся
эффективное действие получает rank1. Но shuffled control тоже проходит бинарный
gate и prediction хуже persistence. Это первый development-сигнал правильного
механизма, но proof obligations 1–5 остаются открыты; следующий барьер —
разделение real/shuffled на более информативном соседнем случае и transfer.

Штатный residual pilot затем подтвердил real<shuffled prediction error на всех
H1/H3/H5/H10, real<persistence до H5 и source success4/4. A skill сохранился
после B updates, но door/push B оказались floor/ceiling без transfer contrast.
Таким образом obligations 1 и 5 получили development evidence только в одном
controlled source case; obligations 3–4 и confirmatory повторяемость не закрыты.

Push-1 follow-up локализовал следующий wall. Terminal-aware depth-local MPC
устранил две ошибки планирования; balanced salient replay дал в diagnostic
WEIGHTS_REPLAY 4/4 против FRESH2/4 и WEIGHTS0/4. Штатный профиль воспроизвёл
только B2/4 с A4/4, поэтому transfer PASS нет. Raw JEPA latent distance ранжирует
визуальное сходство, а не достижимость; следующий bounded hypothesis — learned
goal-conditioned temporal distance на будущих состояниях реальных эпизодов.

Этот bounded hypothesis дал ограниченный положительный результат. Directed
temporal probe на frozen real outcomes обошёл latent MSE и shuffled control;
larger fresh world model с этим score получила closed-loop Push-1 4/6 против
0/6 raw-MSE и 0/6 shuffled-score. Residual correction, RGB salience и
action-contrastive objective причинные gates не прошли. Результат пока зависит
от цвета и одной фиксированной topology, поэтому следующий обязательный шаг —
новые комбинации layout/start/goal с разными первыми действиями. До этого нет
ни neighbor-domain transfer, ни AGI/concept PASS.

Layout-disjoint exp144 дал ordered/raw/shuffled totals 56/12/30 на пяти runs,
но строгий F3 прошёл только 2/5. Это подтверждает полезность temporal order
относительно raw goal distance, но не стабильность learned score относительно
matched shuffled control. Следующий bounded comparison — hindsight
goal-conditioned control на том же replay против текущего MPC; не принимать
temporal probe в production до устойчивого causal преимущества.

Hindsight на всех future pairs провалил closed-loop (0/24), несмотря на верный
первый поворот: policy затем зацикливалась на `forward`. Отбор последних
terminal-success pairs дал direct controller 64/72 на трёх training runs против
0/72 shuffled-action и покрыл все четыре unseen layouts в каждом run. Но строгий
superiority gate против raw/temporal MPC прошёл только 1/3. Это подтверждает
полезность successful experience selection, но не замену MPC и не
self-supervised learning: отбор использует `termination == success`. До
интеграции нужен random-encoder control вклада world-model representation.

Random-encoder control закрыл этот узкий вопрос: learned/random/shuffled-action
получили 60/0/0 successes из 72, representation gate прошёл 3/3. Goal-blind
получил 48/72, а общий superiority gate против MPC — 2/3. Поэтому learned
representation полезно, но GoalSpec-зависимость и источник representation ещё
не изолированы. Следующий gate отключает termination loss и terminal-priority
sampling при обучении backbone, сохраняя тот же replay и terminal-only policy
dataset.

Predictive-only ablation прошёл: encoder без termination loss и salient replay
дал terminal policy 72/72 против full64, random0, shuffled-action0, ordered44 и
raw8 на трёх runs. Predictive representation/controller gates прошли 3/3.
Значит, learned predictive representation полезно независимо от success labels
backbone; success supervision пока остаётся только в отборе 118 policy examples.
Следующий proof obligation — перенести этот же механизм через physics/ruleset
split, а не интегрировать результат одного Push-1 семейства как AGI-компонент.

**Stage 88 — CLOSED (2026-04-16, 1/2 gates)**  
**Stage 89 — PARTIAL (2026-04-19)**

- `gen1=189.4`, `gen5=179.7`, `ratio=0.949`
- secondary PASS (`n_promoted=2`)
- primary FAIL

Вывод:
- knowledge flow **механически** работает,
- но **концептуально** ещё не доказан, потому что следующее поколение не стало лучше предыдущего.
- projectile perception / tracking / imminent threat modeling теперь в целом работают честно,
  но общий survival wall не снят: оставшийся bottleneck уже выше, в общей survival policy
  против `zombie/skeleton`, а не в arrow-telemetry как таковой.

Это означает: проект пока не проходит `docs/CONCEPT_SUCCESS_CRITERIA.md#1`.

---

## Новый принцип roadmap-а

Roadmap строится вокруг **5 proof obligations**:

1. Система должна правильно моделировать опасную динамику мира.
2. Система должна извлекать из опыта каузально полезное знание, а не корреляции.
3. Это знание должно давать межпоколенческий выигрыш.
4. Та же архитектура должна выдержать хотя бы один соседний домен.
5. Только после этого можно говорить, что концепция работает.

Crafter остаётся **главным proving ground**, но roadmap сознательно
не заканчивается Crafter-успехом. Финал roadmap-а — transfer + concept validation.

---

## Phase I — Dynamic World Model

**Цель:** закрыть текущую structural wall, где агент плохо моделирует
короткую опасную динамику: стрелы, бой, приближение угроз, локальную геометрию траекторий.

**Почему это первая фаза:**
- Stage 88 показал, что ceiling теперь определяется не только виталами,
  а боевой динамикой.
- Без точной динамической модели любая causal learning-фаза будет учить шум.

### Предлагаемые stages

**Stage 89 — Arrow Trajectory Modeling** — `PARTIAL`
- добавить стрелу как динамическую сущность с направлением и коротким horizon forecast
- цель: чтобы dodge возникал из планирования, а не из рефлекса
- фактический итог:
  - `exp137` perception + textbook fixes восстановили честный projectile path
  - old `defensive_action_rate` оказался visibility-biased telemetry, а не чистым planner failure
  - arrow-specific local capability подтверждена, но global survival improvement не доказан

**Stage 90 — Threat Interaction Model** — `CLOSED VIA STAGE 90R`
- изолированный threat-model stage сам по себе не остался финальной формой работы
- его диагностические и modeling debts были поглощены более широким `Stage 90 / 90R` line

**Stage 90R — Consolidated Local Survival / Emergency Control Line** — `PASS`
- cause-finding -> viewport-first reset -> actor/ranker contract repairs -> first-class emergency controller
- canonical closeout baseline:
  - implementation commit `71d1e29`
  - closeout docs:
    - `docs/reports/stage-90r-emergency-controller-report.md`
    - `docs/reports/stage-90r-closeout.md`
- final bounded rescue proof:
  - `avg_survival = 190.0` vs frozen `9083357 = 179.25`
  - `early_hostile_deaths_without_rescue = 0`

**Stage 91 — Rescue Robustness Validation**
- проверить, что новый emergency-controller baseline держится вне узкого bounded slice
- расширить compare по seed / episode coverage
- закрепить artifact wiring так, чтобы baseline comparison был самодостаточен в eval payload
- держать GPU online path как соседний infra risk, но не смешивать его снова с архитектурным stage

### Exit gates

- Stage 90 / 90R line has one canonical closed baseline
- next stage compares against that baseline, not against intermediate 90R sub-slices
- broader rescue robustness is proven without reopening 90R as an umbrella for new debt
- improvement claims remain tied to one named stage with one active success criterion

---

## Phase II — Causal Learning

**Цель:** научить систему извлекать и удерживать **каузально полезное** знание.

**Почему это вторая фаза:**
- Stage 88 показал, что promotion сейчас может правильно сохранять структуру,
  но не различает cause vs consequence достаточно надёжно.
- Без этой фазы knowledge flow будет переносить корреляции.

### Предлагаемые stages

**Stage 92 — Causal Hypothesis Filter**
- ввести явную проверку `operational usefulness before promotion`
- hypothesis должна не просто коррелировать с гибелью, а менять prediction/planning

**Stage 93 — Verification Before Promotion**
- promotion only after repeated out-of-sample confirmation
- disconfirmation must lower confidence or block promotion

**Stage 94 — Causal Rule Demonstration**
- показать хотя бы 1-2 clean кейса:
  `observation -> hypothesis -> verification -> retained rule -> better later behavior`

### Exit gates

- at least one new promoted rule is shown to be causally useful
- false-correlation patterns of the `zombie + low drink/food` type are explicitly rejected
- promoted rule changes planner choice in the intended direction
- phase passes `docs/CONCEPT_SUCCESS_CRITERIA.md#3` locally inside Crafter

---

## Phase III — Inter-Generation Knowledge Flow

**Цель:** доказать, что следующее поколение реально стартует умнее.

**Почему это отдельная фаза:**
- Stage 88 already proved persistence mechanics
- but persistence mechanics != knowledge flow success

### Предлагаемые stages

**Stage 95 — Stable Promotion Pipeline**
- harden persistence, merge, loading and inheritance policy
- separate clearly:
  what remains runtime experience vs what becomes promoted fact

**Stage 96 — Multi-Run Generation Proof**
- run repeated generation experiments with identical protocol
- require inspectable inherited knowledge and repeatable generational gain

### Exit gates

- `genN+1 > genN` repeats across multiple independent runs
- inherited knowledge responsible for the gain is identified and inspectable
- later generations improve because they start with a better world model
- phase passes `docs/CONCEPT_SUCCESS_CRITERIA.md#1`

---

## Phase IV — Neighbor-Domain Transfer

**Цель:** доказать, что архитектура не заперта в Crafter.

**Принцип:**
- не parallel multi-domain from day one
- а **Crafter-first with forced transfer checkpoint**

### Требования к соседнему домену

Домен должен быть достаточно близким, чтобы проверять архитектуру, а не запускать отдельный research project:

- partial observability
- resources / affordances
- dynamic threats or moving hazards
- need for short-horizon planning
- возможность textbook-style facts + runtime experience

### Предлагаемые stages

**Stage 97 — Neighbor Domain Port**
- перенести ту же архитектурную схему в соседний домен
- разрешены новые `facts`, environment semantics, labels/textbook entries
- запрещено переписывать planner под case-specific policy logic

**Stage 98 — Transfer Validation**
- показать, что новый домен проходит на той же логике:
  facts + mechanisms + experience + stimuli + promotion

### Exit gates

- second domain works without bespoke control architecture
- new environment support is mostly local to facts / parser / env adapter
- no new Crafter-like reactive special-case layer appears
- phase passes `docs/CONCEPT_SUCCESS_CRITERIA.md#4`

---

## Phase V — Concept Validation

**Цель:** дать честный ответ на вопрос: работает ли концепция?

Эта фаза не про "ещё один механизм". Она про финальную проверку claim’ов.

### Предлагаемый stage

**Stage 99 — Concept Validation Report**
- собрать итоговый architecture report
- проверить проект против `docs/CONCEPT_SUCCESS_CRITERIA.md`
- разделить:
  - what is proven
  - what is promising but unproven
  - what is explicitly disproven or still blocked

### Exit gates

Все 5 пунктов из `docs/CONCEPT_SUCCESS_CRITERIA.md` должны быть закрыты:

1. cross-generation benefit demonstrated
2. benefit comes from the correct architectural layer
3. causally useful retained knowledge demonstrated
4. neighboring-domain transfer demonstrated
5. better planning follows from better world understanding

Только после этого допустимо утверждение:
**"концепция работает"**

---

## Dependency Graph

```text
Phase I  Dynamic World Model
   │
   ▼
Phase II Causal Learning
   │
   ▼
Phase III Inter-Generation Knowledge Flow
   │
   ▼
Phase IV Neighbor-Domain Transfer
   │
   ▼
Phase V Concept Validation
```

Почему порядок именно такой:

- без dynamic world model causal learning будет захватывать шум
- без causal learning inter-generation transfer будет переносить корреляции
- без inter-generation gain нельзя утверждать knowledge flow success
- без neighbor-domain transfer нельзя говорить о масштабируемости концепции

---

## What Is Explicitly Not The Center Of The Roadmap

- Crafter-specific optimization presented as architecture progress
- threshold tuning ради одного gate
- новые learning modules без явного ideological debt
- расширение списка entity types как самоцель
- claims of concept success before transfer and generation gain

---

## How To Read Progress

Крупная фаза считается закрытой только если:

1. phase exit gates выполнены
2. `docs/ASSUMPTIONS.md` обновлён
3. stage/phase reports объясняют, **почему** improvement architectural
4. `docs/ANTI_TUNING_CHECKLIST.md` не даёт оснований считать результат просто environment tuning
4. при необходимости пройден соответствующий пункт из `docs/CONCEPT_SUCCESS_CRITERIA.md`

Stage numbers сохраняются как execution-level units, но roadmap теперь управляется фазами, а не наоборот.
