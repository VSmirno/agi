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

**Stage 90 — Threat Interaction Model**
- расширить short-horizon model для skeleton/zombie contact windows, line-of-fire, threat arrival
- убрать blind spots между "вижу врага" и "умираю через 2-5 тиков"

**Stage 91 — Dynamic Planning Validation**
- доказать, что planner выбирает лучшие действия именно из-за новой динамической модели
- не тюнинг score, а сравнение prediction quality и decision quality

### Exit gates

- prediction error on dangerous short-horizon dynamics statistically below current baseline
- deaths from arrows reduced by at least 50% versus pre-Phase-I baseline
- at least one new defensive behavior emerges from planning alone
- improvement explained as world-model improvement, not threshold/scoring tweaks

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
