# Идеология проекта SNKS AGI

**Версия:** 2 (полная переписка)
**Дата:** 2026-04-11
**Статус:** Living document. Обновлять при каждом крупном insights / провале.

> Этот документ — манифест и lessons learned. Не how-to-implement guide. Если
> у тебя возникает вопрос **«как написать конкретный код для X»** — этот
> документ не ответит. Если возникает вопрос **«правильно ли я думаю о
> проблеме X»** — этот документ должен помочь.

---

## 0. Зачем это переписано

Первая версия `IDEOLOGY.md` (Stage 72-74, апрель) была написана как реакция
на конкретное расхождение между декларируемой архитектурой («самоорганизация,
непрерывное обучение») и реальной реализацией («supervised backprop, batch
training, ground truth scaffolding»). Это расхождение мы починили
— но оказалось что у нас есть **другие**, более глубокие, которые v1
не зафиксировала.

За Stages 78-81 мы:

- Нашли и зафиксили 8 багов в planner / sim / training pipeline
- Несколько раз воспроизвели один и тот же failure mode под разными
  именами (discrimination paradox)
- Месяцами измеряли learning approaches **против заведомо неправильного
  baseline** (sleeping agent), не подозревая что baseline сломан
- Обнаружили что собственные «universal» механизмы у нас постоянно
  оказывались Crafter-specific хаками
- Подняли warmup_a baseline на +50 шагов, но **не сломали** Crafter
  wall в eval
- Узнали что наша главная проблема не в **learning algorithm**, а в
  **где живёт знание и как оно течёт**

Эта v2 фиксирует то что **мы поняли**, и явно называет ловушки в которые
**регулярно попадались**. Это рабочий документ, не финальная философия.

---

## 1. Три категории знания

Главный концептуальный фрейм проекта — три категории знания, которые
живут в **разных местах** и обновляются **разными способами**.
Смешивание этих категорий — наша самая частая ошибка.

### Категория 1 — Facts (факты)

**Где живёт сейчас:** `configs/crafter_textbook.yaml`. Также в любом
другом structured store (SQLite, knowledge graph) который мы можем
завести позже.

**Что это:** declarative propositions about the world. Каузальные
правила, причинно-следственные цепочки, attributes концептов, env
семантика, magnitudes (примерные) различных эффектов. То что
«учитель» **знает** про мир и **передаёт** агенту до того как он
начал жить.

**Как обновляется:** редко. Через явное действие — человек добавляет
запись в textbook, либо runtime nursery promote'ит learned rule в
textbook (см. principle 5 ниже).

**Примеры:**

- «do tree → wood +1»
- «sleep → energy +5 (clamped at max=9)»
- «skeleton at range 5 → health -0.5»
- «tree is impassable» (фактически в Crafter; пока в коде, должно
  быть в textbook)
- «do uses facing tile, не proximity» (фактически env semantic; пока
  захардкожено в planner, должно быть в textbook)

**Правило:** если факт known to teacher и stable across episodes,
он принадлежит этой категории. Не нужно писать ML pipeline чтобы
агент его выучил.

### Категория 2 — Mechanisms (механизмы)

**Где живёт:** Python код в `src/snks/agent/*`. Главным образом
`ConceptStore.simulate_forward`, `_apply_tick`, planner-loop в
`mpc_agent`, perception (`perceive_tile_field`), spatial map.

**Что это:** алгоритмы которые **читают** facts (категория 1) и
**применяют** их. Универсальные процедуры — генерация candidate
plans, scoring trajectories, applying tick phases, обработка
surprise. Также адаптеры между representations (sim ↔ env, pixels
↔ concepts).

**Как обновляется:** очень редко. Через осознанный refactor —
изменение фундаментальной архитектуры. Не каждый эпизод. Не каждый
stage. Только когда обнаружили что **алгоритм неправильно
интерпретирует facts**, не когда хочется добавить новый поведенческий
паттерн.

**Примеры:**

- `simulate_forward` = generic rollout machinery, не знает про tree
  vs water
- `_apply_tick` Phase 6 = диспетчер «do/place/make/sleep» по primitive
- `score_trajectory` = lex-tuple ranking
- `update_spatial_map_from_viewport` = perception → world map
- `nursery.tick` = surprise → candidate → verify → promote

**Правило:** если механизм работает **только для одного env / одного
типа сущностей**, это **не mechanism**, это hardcoded fact в плохом
месте. Перенести в категорию 1.

### Категория 3 — Experience (опыт)

**Где живёт:** в runtime структурах — `HomeostaticTracker.observed_rates`,
`ConceptStore.learned_rules`, `surprise_accumulator`, `spatial_map._map`.

**Что это:** знание которое **только** агент может получить. Текущая
карта мира этого конкретного эпизода. Уточнённые rates после N
наблюдений. Правила которые textbook не предусмотрел и которые
discoveрятся через surprise. Locally adapted hyperparameters
поведения.

**Как обновляется:** непрерывно. Каждое env step → tracker обновляется,
spatial_map обновляется, surprise accumulator получает новый record,
nursery эмитит/верифицирует/промоутит rules.

**Примеры:**

- Tracker observed_rate("food") = 0.0185 (refined from textbook prior 0.02)
- spatial_map: tree at (28, 32)
- learned_rule: «sleep + (food=0 OR drink=0) → health -0.067»
- surprise_record: predicted +5 drink, actual 0 → gap -5

**Правило:** experience должен быть **тем что учитель не мог знать
заранее**. Если можно записать в textbook — это fact, не experience.
Заставлять агента discover'ить факты которые учитель знает — это
**антипаттерн** (см. anti-pattern 1).

### Принцип распределения

Каждый раз когда хочется добавить новое поведение / новое знание в
систему — **сначала** спросить:

1. **Это про мир в целом или про конкретный эпизод?**
   - Про мир → category 1 (textbook)
   - Про конкретный эпизод → category 3 (experience)

2. **Это знание которое учитель имеет или которое учитель не может
   иметь?**
   - Имеет → category 1
   - Не может → category 3

3. **Это процедура (как читать/применять знания) или само знание?**
   - Процедура → category 2 (mechanism)
   - Знание → category 1 или 3

Только после этого решения — писать код.

---

## 2. Пять принципов

### Принцип 1 — Три категории, не одна

Уже описано в разделе 1. Главное напоминание: каждый раз когда
смешиваешь категории — ты копаешь себе яму на следующие 3 stages.

Конкретный пример из нашей истории: когда мы ставили цель «agent
discoverит conjunctive rule sleep+starvation», мы пытались научить
**experience** тому что было **fact**. Stage 78a / 78c / 79 потратили
много времени и кода ради этого. Если бы мы сразу спросили «может
ли учитель это знать?» — ответ «да» — и записали в textbook одной
строкой YAML, всех этих стадий не случилось бы.

### Принцип 2 — Top-down

Цель сначала. Стратегия эмерджит из взаимодействия (цель × world
model × experience), не пишется руками.

Если хочется написать `if zombie_visible: drives["wood_sword"] = 5.0`
— стоп. Спросить:

- Знает ли world model что zombie damages вitals?
- Знает ли world model что wood_sword defeats zombies?
- Если оба «да» — почему planner не выводит сам что приоритет
  растёт?
- Если planner не выводит — где лажа? В world model? В планировщике?
  В цели?

Никогда не подменять missing reasoning hardcoded if-else'ом. Каждый
такой if — это **скрытое признание** что архитектура ниже по стеку
что-то не делает.

Этот принцип сохранён из v1. Stage 80 диагностика подтвердила
важность: мы обнаружили что 70% времени агент спит, и попытка
подкрутить residual / nursery **не помогала**, потому что мы
тюнили mechanism вместо того чтобы спросить «почему planner вообще
выбирает sleep». Когда задали правильный вопрос — нашли Bug 3 в
score function за 30 минут.

### Принцип 3 — Continuous learning

Нет фаз «собрать данные → обучить модель → задеплоить». World model
обновляется **в процессе жизни**.

Concrete:

- Каждый env.step → tracker.update + spatial_map.update + surprise
- Каждая верификация rule → confidence update
- Каждое promotion candidate из nursery → store.add_learned_rule
- Periodic / on event → learned_rules → textbook (см. Принцип 5)

**Никаких** offline training runs где модель замораживается. **Никакого**
build-deploy split. Если что-то учится batch-style — это либо
perception layer (CNN encoder, который в идеологии «врождённый»),
либо это нарушение принципа.

Этот принцип сохранён из v1. Реализация в текущем коде в основном
верна (tracker, nursery работают continuous). Один pinch-point —
CNN encoder, который мы обучали batch-style и не тренируем в
online. Это допустимое исключение **если** мы относимся к нему как к
«врождённому V1» (см. принцип 4) и не позволяем ему расти. Когда
надумаем заменить CNN — заменяющий подход должен быть continuous.

### Принцип 4 — Vision ≠ Knowledge

Перцепция (CNN, V1, near_head) — это **hardware**. Она группирует
визуально похожие пиксели в classes. Не знает что они значат.

Семантика — это **experience**. «class_7 = tree» агент учит через
действие («сделал do рядом с class_7, получил wood»).

CNN классифицирует, agent именует.

Сохранено из v1 вербатим, потому что это всё ещё корректно. Stage 75
segmenter всё ещё работает по этой схеме.

**Caveat который мы обнаружили в Stage 80:** segmenter может
**ошибаться** — Stage 75 обнаруженно классифицировал player sprite
(или его underlayer) как «tree» в spatial_map. Это perception
artifact, и mechanism layer (find_nearest, Bug 5 fix) пришлось
defend-in-depth: skip player's own tile. Принцип не нарушен —
perception ошибается, mechanism защищается. Но это напоминание что
**врождённое V1** не значит «безошибочное».

### Принцип 5 — Knowledge flow

Опыт **становится** фактом для следующего поколения. Учитель
**не вечный** статический файл — это бывшие experience'ы прошлых
эпизодов, отфильтрованные и promoted.

Конкретно:

```
              ┌──────────────┐
              │   TEACHER    │ ← initial seed
              │    YAML      │   (sparse, rough)
              └───────┬──────┘
                      │ load
                      ↓
              ┌──────────────┐
              │  WORLD MODEL │
              │ ConceptStore │
              └─┬──────────┬─┘
                │          ↑
        load    │          │ promote
                ↓          │ (after N stable obs)
      ┌──────────────┐   ┌────────────────┐
      │  SIMULATE    │   │ LEARNED_RULES  │
      │   _FORWARD   │   │    (Stage 79)  │
      └─┬────────────┘   └────────────────┘
        │                       ↑
  rollout                       │
        ↓                       │
   ┌────────┐  surprise  ┌──────────────┐
   │  agent ├──────────→ │  NURSERY     │
   │ in env │            │  accumulator │
   └────────┘            └──────────────┘
```

В этой картине:

- Зелёная стрелка снизу-справа («surprise → nursery → learned_rules
  → world_model») **уже работает** — Stage 79 implemented это.
- Стрелка сверху («learned_rules → teacher YAML») **не работает**.
  Learned rules умирают с runtime store. Это **главный пробел**
  Strategy v5.

Когда промоушн заработает:

- First-generation agent с минимальным textbook discoverит
  conjunctive sleep rule, promoteит, **записывает в textbook**.
- Second-generation agent стартует со enriched textbook — conjunctive
  rule уже факт. Не нужно discover'ить заново.
- К generation 100 textbook стабилизируется. New experience редкий.

Это **не просто механизм persistence**. Это другая ментальная
модель: **textbook это living document**, не статический контракт
от программиста. Учитель = аккумулированный опыт прошлых поколений.

Этот принцип **не реализован**. Это roadmap-level вещь. Stage 82+
candidate.

---

## 3. Антипаттерны (вещи которые мы реально делали неправильно)

Эти антипаттерны не гипотетические. Каждый из них стоил нам кода,
времени и багов. Я перечисляю их с конкретными ссылками на наши
собственные stages так чтобы будущая версия меня (или сменщик) знала
**как это выглядит изнутри** до того как впадёт в ту же яму.

### Antipattern 1 — Учить тому что учитель уже знает

**Как выглядит:** «Нам нужно научить агента что sleep с
food=0 даёт health -0.067». Пишется ML pipeline (residual MLP, DAF
substrate, symbolic nursery, etc) который должен это discover'ить
из noisy observations.

**Что не так:** этот rule **известен учителю**. Это physics, не
emergent property. Учитель может прямо записать его в textbook
одной строкой YAML.

**Где мы это делали:**

- **Stage 78a** — DAF substrate × MLP head пытался выучить
  conjunctive sleep+starvation rule на synthetic data. 7 регимов,
  все 10× хуже linear baseline. Мы назвали это «discrimination
  paradox» и долго думали что это что-то про substrate.
- **Stage 78c** — MLP residual + online SGD пытался выучить тот же
  паттерн на real Crafter rollouts. Тот же failure mode под другим
  именем.
- **Stage 79** — symbolic nursery с explicit preconditions пытался
  выучить тот же паттерн через surprise accumulation. **Synthetic
  test PASSED** (доказано что induction работает), но Crafter
  neutral. Та же история.

**Урок:** **discrimination paradox это симптом**. Корневая причина —
мы три раза подряд пытались learn что было fact. Каждый раз говорили
себе «теперь с другим mechanism» и каждый раз получали тот же
failure mode потому что **сама задача неправильно поставлена**.

**Detection rule:** перед тем как писать learning pipeline для
конкретного rule — спросить «может ли я (или человек) **записать**
этот rule прямо сейчас?». Если да — это fact. Записать. Не learn.

### Antipattern 2 — Зашивать env-семантику в mechanisms

**Как выглядит:** «Crafter `do` action interactит с facing tile, не
с adjacent. Поправлю planner так чтобы он проверял facing». Пишется
patch в `expand_to_primitive` или `_apply_tick`.

**Что не так:** «Crafter `do` interactит с facing» это **env fact**.
Должен жить в category 1 (textbook), не в category 2 (mechanism).
Иначе:

- Mechanism становится Crafter-specific
- Перенос на другой env требует переписать mechanism
- Невозможно сравнить «как этот env устроен» — нужно читать Python код
- Учитель не может объяснить env агенту, потому что mechanism
  «знает» env semantics неявно

**Где мы это делали:**

- **Bug 2** (Stage 79) — захардкодил facing check в `expand_to_primitive`
- **Bug 4** (Stage 80) — захардкодил sim/env mismatch handling
  («env blocks impassable, sim doesn't, поэтому в sim
  oscillates → return do при dist=0»)
- **Bug 5** (Stage 80) — захардкодил «player can't be on resource
  tile» в find_nearest
- **Bug 6 v1** (Stage 80) — захардкодил список Crafter resource
  items прямо в planner. Это **юзер заметил** и заставил
  refactor'ить через `gatherable_items()`.
- **Bug 8** (Stage 81) — захардкодил blocking semantics через
  heuristic `impassable_concepts()` (читает textbook но heuristic
  «do-rule with positive inv_delta = blocking» Crafter-specific)

5 из 8 багов сессии Stage 78-81 — этот антипаттерн.

**Урок:** новая способность → сначала спросить «можно ли это
выразить как factualный attribute концепта?». Если да — пишем
**один** generic mechanism который reads attribute, и **N**
declarative entries в textbook. Если нет — тогда обоснованно пишем
in code. Но **обосновать**, не дефолтно.

**Detection rule:** если patch в mechanism содержит string literal
с именем concept'а из конкретного env (e.g. `"tree"`, `"wood"`, `"do"`),
**стоп**. Это fact в плохом месте. Refactor.

### Antipattern 3 — Не смотреть что агент делает

**Как выглядит:** «Eval avg_len = 178. Это плохо. Попробуем подкрутить
score function / residual / nursery / encoder». Стадии тратятся на
mechanism tweaks без проверки что **агент фактически делает в
эпизодах**.

**Что не так:** мы оптимизируем metric не зная **поведения**. Все
решения становятся гадание-в-темноте.

**Где мы это делали:** **Stages 78c, 79, 80 (early)**. Мы тюнили
learning approaches три stage'а. Никто не спрашивал «а **что
агент собственно делает** в этих 178 шагах?». Когда в Stage 80
**наконец** запустили простую diagnostic (5 episodes с trace +
action_counts), оказалось что **70% действий sleep**. Агент вообще
никогда не пытался gather. Вся предыдущая работа была против
sleeping baseline.

**Это самый дорогой антипаттерн в плане потерянного времени.** 4-5
stages плюс несколько reports описывали failure mode который имел
**одну корневую причину** и эта причина обнаруживалась за 3 минуты
diagnostic'ом. Мы её не запустили потому что были заняты «правильной»
работой.

**Урок:** перед каждым новым experimental stage — **сначала** запустить
behaviour diagnostic. action_counts, action_entropy, что в final_inv,
что в visible, что в spatial_map. **Затем** уже формулировать
гипотезу что менять.

**Detection rule:** если ты не можешь ответить на вопрос «что агент
**делает** в провальных эпизодах» (не «какие у него metrics»), то
ты не имеешь права писать новый learning approach. Сначала diagnose,
потом fix.

### Antipattern 4 — Sleeping baselines / silently broken comparisons

**Как выглядит:** «Stage 78c residual_off eval = 169.2. Stage 78c
residual_on eval = 152.1. Δ -17.1. Approach broken.» Мы записываем
этот вердикт в отчёт и закрываем стадию.

**Что не так:** если **baseline contaminated** — все выводы про
delta'ы неправильные. В Stage 78c v1 baseline (residual_off) был
сломан Bug 2 (proximity-vs-facing) **который мы нашли только в
Stage 79 retro**. Реальный baseline после Bug 2 fix = 180.4, не
169.2. Stage 78c вердикт «MLP residual approach broken» был
**частично контаминирован**: residual_on тоже broken, но gap между
180.4 и 154.5 это другая история чем gap между 169.2 и 152.1.

**Где мы это делали:** **Stage 78c v1**. И каждый stage от 78c до 80
оперировал баг-зараженным baseline.

**Урок:** перед тем как делать выводы из A vs B сравнения — спросить
«а baseline (B) **сам** работает корректно?». **Не предполагать**.
Запустить baseline в **той же** конфигурации **дважды**, посмотреть
variance. Если variance большой — копать в baseline до того как
интерпретировать delta.

**Detection rule:** «Baseline = X» это claim который требует
доказательства, не axiom. Каждый раз когда baseline number используется
для decision, добавь в отчёт «baseline measured in run R, repeatable
from commit C». Если воспроизвести нельзя — baseline не использовать.

### Antipattern 5 — Локальные fixes без вопроса «правильный ли уровень»

**Как выглядит:** Bug 1 fix → не помог → Bug 2 fix → не помог → Bug 3
fix → не помог → ... → 8 bug fixes → eval всё ещё 167. Мы каждый раз
радовались что нашли «корневую причину», a в действительности находили
**локальную** причину **локального** failure.

**Что не так:** некоторые проблемы — это симптомы более глубокой
архитектурной дыры. Локальный fix работает локально (баг исчез) но
metric не двигается (потому что архитектура держит wall где-то ещё).
Пытаясь чинить bugs мы избегали ставить вопрос «**где** настоящий
bottleneck».

**Где мы это делали:** **вся session Stage 78-81**. 8 багов. Wall
сдвинулся с 180 до 167. Каждый bug fix имел «правдоподобное
объяснение почему именно он сломал стену». Все объяснения были
postnice. Стену они не сломали.

**Урок:** после ~3 bug fixes без movement — **остановиться** и спросить
«может проблема не на этом уровне?». Архитектурный диагноз требует
**stepping back**, не «один маленький fix больше».

В Stage 80 мы получили честный архитектурный диагноз: planner с
лексикографическим scoring всегда picks shortest greedy plan, никогда
не commit'ит на multi-step crafting chain. Это **архитектурное**
ограничение. Никакой bug fix этого не починит — нужно либо
hierarchical planning, либо subgoal commitment, либо score redesign
**deeper than has_gain tweak**.

**Detection rule:** считать bugs. Если N ≥ 3 и main metric не сдвинулась
на ≥ 10% от baseline → **остановиться**. Не fix N+1. Спросить «правильный
ли вообще layer мы атакуем?». Может быть decision-making, не perception.
Может быть scoring, не learning. Может быть env semantic, не algorithm.

---

## 4. Открытые вопросы (на которые мы не знаем ответа)

Этот раздел — честный список того, что мы **не знаем**. Здесь нет
рекомендаций, потому что у нас нет проверенных решений. Будущим
stages придётся ответить, или явно отложить.

### Q1 — Где live world model должна жить?

**Проблема:** YAML textbook не масштабируется. 30 правил Crafter — fine.
30K правил Minecraft — медленный linear scan, hand-curated bottleneck,
нет fuzzy matching, нет parametric updates, нет cross-rule consistency.

**Варианты:** SQLite / knowledge graph (Neo4j-like) / event-sourced
log + compaction / hybrid neural-symbolic / pure neural. Ни одну мы не
пробовали.

**Что мы знаем:** YAML годится как (a) initial seed, (b) export для
человеческого review, (c) cross-agent transfer format. **Не годится**
как primary runtime store при scale.

**Что мы НЕ знаем:** какой формат правильный. Какой допустимо
сложный. Когда переходить.

**Когда придётся решать:** при первом env с >100 концептами или
>500 правилами. Crafter не давит. Следующий env будет.

### Q2 — Параметрическое vs символическое

**Проблема:** symbolic rules (`do tree → wood +1`) не generalizуются.
«do oak → wood +1» это отдельное правило. «similar to tree» нет.

**Варианты:**

- Hierarchical concept ontology (oak < tree < resource) — generalization
  через inheritance
- Concept embeddings (vector per concept, similarity = cosine) —
  generalization через distance
- Rule templates с slot filling (`do <gatherable_X>` → `<gathered_item_of_X>` +1)
  — generalization через schema
- Parametric rule attributes (rate = function of context) —
  generalization через learned parameters
- Все вышеперечисленное в hybrid

**Что мы знаем:** наш текущий ConceptStore имеет flat list of rules.
Никакой generalization machinery. Любое новое concept требует новых
rules с нуля.

**Что мы НЕ знаем:** какой compromise между «всё символическое
и interpretable» и «параметрическое и scalable» правильный для AGI
goal. Стоит ли сразу инвестировать в hierarchies или ждать pain point.

### Q3 — Сколько автономии у planner до override учителя

**Проблема:** Если planner discoverит что текстbook ошибается («textbook
говорит skeleton damage 0.5/tick на range 5, но я наблюдаю 0.3»), что
делать?

**Варианты:**

- **Submissive:** агент всегда верит textbook. Surprise игнорируется
  или только логируется.
- **Bayesian:** агент имеет prior из textbook, posterior обновляется
  observations. Confidence textbook'а влияет на скорость override'а.
- **Aggressive:** агент верит experience > textbook когда они
  conflictят. Может быть полностью wrong если environment весь
  обманчив.
- **Symmetric:** один textbook, обновляется и человеком и агентом,
  conflicts разрешаются через confidence + recency + provenance.

**Что мы знаем:** Stage 79 nursery promoted rules **заменяют**
textbook predictions через Phase 7 в `_apply_tick`. Это эффективно
agressive — learned rules фactически override (или дополняют) textbook.
Мы не имеем conflict resolution.

**Что мы НЕ знаем:** какой режим правильный для balance между «trust
the teacher» и «adapt to current world». В реальном мире teacher часто
неполон / устарел / wrong про конкретный environment. Полный submissive
не работает. Полный agressive рискует agent'у выучить bad model из
sample bias.

### Q4 — Crafter как goal или как data source

**Проблема:** мы 5+ stages пытались сломать Crafter survival wall ~180.
Не сломали. Wall is bounded by enemy combat which requires multi-step
crafting (sword) which requires multi-step planning depth.

**Варианты framing'а:**

- **Crafter as goal:** наша задача — survive дольше, gather больше,
  defeat enemies. Wall это enemy. Метрики направлены на main eval.
- **Crafter as data source:** наша задача — выработать процесс
  который превращает env exploration в structured world model.
  Wall не имеет значения. Метрики направлены на качество выученного
  knowledge graph.

**Что мы знаем:** мы implicitly шли первым framing'ом и не
двигались. Brainstorm в Stage 81 показал что второй framing более
интересный для AGI roadmap (transfer, generations, knowledge
accumulation).

**Что мы НЕ знаем:** какое framing «правильное» для долгосрочной
цели проекта. Это вопрос про **что мы хотим продемонстрировать**,
а не про конкретный stage.

### Q5 — Когда останавливать tactical iteration и пересматривать стратегию

**Проблема:** мы 30+ commits в одной сессии, 8 bug fixes, не сломали
wall. Каждое изменение «должно было помочь». Иногда нужно
**остановиться** и думать вместо коммитить.

**Что мы знаем:** мы в эту яму попадаем регулярно. Есть feedback
memory `feedback_top_down_goal.md`, `feedback_no_hardcoded_reflexes.md`,
есть IDEOLOGY.md (этот документ), но они не предохраняют от
впадения.

**Что мы НЕ знаем:** какой автоматический trigger использовать.
«После N bugs»? «После M минут»? «После непрерывного недвижения
metrics»? Все эти thresholds произвольны.

---

## 5. Чего этот документ НЕ говорит

Чтобы избежать неправильного использования — этот документ
**не предписывает**:

- **Конкретные алгоритмы.** Stage 79 nursery, Stage 78c residual
  — это experiments, не доктрина. Они могут быть выкинуты или
  заменены.
- **Конкретные stage numbers / timeline.** Stages 78-81 это history,
  не roadmap. Будущие stages не обязаны следовать той же нумерации
  или последовательности.
- **Использовать ли neural networks.** Neural perception (CNN, V1)
  допустимо как hardware. Neural reasoning / world model — open
  question, не предписано.
- **Какой env таргетировать после Crafter.** Зависит от Q4 (Crafter
  as goal vs as data source) и от Q1 (storage scale).
- **Сколько кода писать на stage.** Stage может быть «1 строчка YAML
  + 1 commit» если это правильное движение. Или «2 недели refactor»
  если архитектура того требует. Размер не показатель ценности.
- **DAF/SKS / oscillator подход.** Parked в Stage 78a методологические
  gaps. Может вернуться, может не вернуться. Не догма.

Если ты не можешь решить проблему опираясь на этот документ — не
переинтерпретируй его в **более конкретный** инструмент. Скорее всего
проблема за рамками идеологии и нужно решить её на другом уровне
(implementation, debugging, experimentation).

---

## Приложение A — Brain analogy table (preserved from v1)

Сохранено из v1. Эта таблица отражает **decomposition** ответственности
между компонентами агента. Её нужно перечитывать каждый раз когда
возникает вопрос «куда положить логику X».

| Уровень | Аналогия | Реализация | Обучение |
|---------|----------|------------|----------|
| Зрительная кора (V1/V2) | Сетчатка → features | CNN encoder (frozen) | Предобучен, дообучается редко по триггеру |
| Распознавание объектов | "Это дерево" | TileSegmenter / ConceptStore.query_visual | Мгновенное (one-shot grounding из опыта) |
| Уточнение перцепции | "Точнее — берёза" | EMA обновление prototype | Каждая встреча с объектом |
| Каузальная модель мира | "do tree → wood" | ConceptStore causal rules + confidence | Мгновенное (verify ±0.15 на каждом действии) |
| Пространственная память | "Дерево было слева" | CrafterSpatialMap | Каждый шаг |
| Рефлексы | "Zombie = бежать!" | Не реализовано (анти-паттерн в Stage 73 показал почему) | — |
| Потребности | "Хочу есть" | HomeostaticTracker + score_trajectory body component | Непрерывно (физиология) |
| Планирование | "Нужна кирка → нужен стол → нужно дерево" | generate_candidate_plans + plan_toward_rule (backward chaining) | Использует каузальную модель |
| Удивление | "Ожидал камень, нашёл пусто" | Stage 79 SurpriseAccumulator | Мгновенное (prediction error) |
| Промоушн опыта в учебник | "Это правило теперь известно" | Не реализовано (Stage 82+ candidate) | — |

Два уровня помечены как «не реализовано». Это известные пробелы. См.
Q4 в открытых вопросах.

---

## Приложение B — Три скорости обучения (preserved from v1)

1. **Мгновенное** — один опыт меняет world model:
   - Verify outcome → confidence ±0.15 на каузальном правиле
   - Zombie укусил → можно записать «zombie = dangerous» с одного
     наблюдения (не реализовано, но в идеологии правильно)
   - Visual prototype записан с одного успешного «do»

2. **Постепенное** — уточнение с каждым опытом:
   - Tracker observed_rate (Bayesian update от prior_strength=20)
   - EMA на visual prototypes (0.9 × old + 0.1 × new) — если CNN
     заменён
   - Confidence стабилизируется после десятков подтверждений
   - Surprise accumulator buckets (Stage 79) — N observations прежде
     чем emit candidate

3. **Фоновое** — редко, по триггеру:
   - CNN encoder retrain (если perception quality деградирует)
   - Promotion learned_rule → textbook YAML (Stage 82+ candidate)
   - Compaction / pruning learned_rules после N эпизодов
   - Schema migration (если world model storage меняется)

Три скорости отражают **разную природу** знания. Instant — это
single-shot updates на discrete observations. Gradual — это refinement
of continuous parameters. Background — это structural changes которые
не должны мешать жизни агента.

Все три должны существовать **одновременно**. Отсутствие одной из
них = архитектурный дефект.

---

## Изменения относительно v1

Удалено:
- Конкретные ссылки на NearDetector, semantic GT navigation, controlled
  env grounding (закрытые проблемы Stage 72)
- Spec про DAF/SKS как future plan (parked в Stage 78a methodology)
- «Социальное обучение через диалог с учителем» (слишком далеко)

Сохранено (преимущественно verbatim):
- Top-down design principle
- V1 ≠ knowledge insight
- Three speeds of learning
- Brain analogy table
- «Textbook = учитель, не чит» (но переформулировано в категорию 1)

Добавлено:
- Three categories framing (главное новое)
- Knowledge flow principle (Stage 81 brainstorm)
- Anti-patterns с конкретными ссылками на наши собственные failures
- Open questions (честный список того что мы не знаем)
- Раздел «что этот документ НЕ говорит»

Дисциплинарное обещание для следующих stages:
- Каждый новый stage начинается с проверки **в какой категории
  лежит наша новая задача**
- Каждый bug review проверяет не нарушает ли fix Antipattern 2
  (env-specific in mechanism)
- Каждая stage end review проверяет нет ли **sleeping baseline**
  (Antipattern 4) и не **накопилось ли локальных fixes без**
  **architectural movement** (Antipattern 5)
