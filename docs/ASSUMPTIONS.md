# Допущения и ограничения по этапам

Файл фиксирует упрощения, принятые на каждом stage. Обновляется при завершении каждого stage.

---

## 2026-09-05 — Transferable learning core: быстрый эксперимент

Начата реализация [плана](superpowers/plans/2026-09-05-transferable-learning-core.md)
по согласованной спеке. По уточнению пользователя приоритет — небольшой сквозной
эксперимент, не полная инфраструктура подтверждающей кампании. Старый агент не
меняется и не подключается как скрытый fallback.

Ограничения: CNN/GRU с нуля, короткий дискретный search, явные наблюдаемые сенсоры,
replay реальных эпизодов, updates между эпизодами, ручные GoalSpec. Это разрешённое
спекой исследовательское отступление от старой идеологии непрерывных обновлений
на каждом шаге. SIGReg не является доказательством полезного представления,
ансамблевое расхождение — не калиброванная уверенность. Малый development pilot
не закрывает G0–G6, перенос или цель AGI. Полная preregistration automation,
масштабная статистическая кампания и исчерпывающая проверка форматов отложены.

Разработка локально, тесты и эксперименты на HyperPC. Доступны две RTX 3090;
существующие чужие процессы не останавливаются.

Первый [pilot](reports/learning-core-pilot.md), код `a096de0`: 100 source updates,
end-to-end success 0/4. Fixed-E sensor MSE на horizon 1: real 3.56337,
shuffled 3.56328, persistence 0.02419. Полезное action-conditioned знание не
показано; падение loss не считается прогрессом к AGI. Crafter — controlled
local-source fixture; B — фиксированная геометрия, не held-out maps. Полный
roadmap и доказательство переноса остаются открытыми.

Найденные обзором дефекты исправлены в `8e51eae`: transfer-only burn-in=0 сохраняет supervision
одношаговых успехов ценой отсутствия предыстории до replay window; одинаково для
всех веток. Present corrupt inventory теперь ошибка, не missing sensor; FRESH
не наследует source training cost. Первый transfer-001 с прежним burn-in=1
сохранён, но не используется как доказательство корректной B adaptation.
Исправленный transfer-002: fresh/weights/weights+replay одинаковы — door 0/2,
push 2/2 до и после. Push уже решается без обучения; A skill исходно отсутствует.
Текущий короткий фиксированный probe не измеряет полезный transfer/retention.

Follow-up диагностика: source GRU candidate abs≈0.997, action gradients малы;
обнуление сенсорного вклада усиливает различия действий, обнуление z — почти нет.
Отдельная нормализация sensor projection уменьшила насыщение, но не дала полезного
обучения (снова 0/4, real/shuffled MSE≈3.536). Не делать её default по одной
активационной метрике. Повторный Crafter collection не побитно детерминирован:
при одинаковых actions/sensors один эпизод имел отличия RGB. Последующие парные
сравнения обучаются на одном сохранённом replay. `predict_sensor_delta` — новый
opt-in профиль, не замена baseline: sensor heads прогнозируют изменение, loss
сравнивает абсолютный прогноз с реальным target; на следующих rollout шагах
прибавляется собственное предсказанное состояние, не будущие истинные сенсоры.

Парный exp140 на одном replay дал первый ограниченный положительный механизм:
delta-real 4/4 против untrained0/4 и zero-action0/4; action5 rank1 на трёх
wood-events против rank5 shuffled. Real получает wood2 на step2, shuffled wood1
на step5. Но shuffled тоже 4/4, prediction всё ещё хуже persistence, только
3held-out wood events/1seed/fixed fixture. Это не перенос и не stage/concept PASS.

Штатный delta pilot воспроизвёл source 4/4; real-actions sensor MSE ниже shuffled
на H1/H3/H5/H10 и ниже persistence до H5, но не на H10. Delta checkpoint после
B updates сохраняет A2/2 в WEIGHTS и WEIGHTS_REPLAY. Это retention весов, не
доказанный transfer: door остаётся0/2, push уже2/2 у FRESH. Следующий B probe
должен избежать floor/ceiling; сначала используем существующий push_1 ruleset.

Push-1 показал, что initial success может быть случайным: FRESH выбрал готовую
3-action последовательность до обучения, а sparse random adaptation её разрушила.
После natural random64/1000 updates action matching обучается, но raw latent goal
distance не является мерой прогресса. Planner теперь использует absorbing
predicted terminals и depth-local state cost; это generic исправления, не знание
правил Grid. `salient_fraction=0.5` смешивает uniform windows с окнами вокруг
terminated/sensor-change без reward. В production WEIGHTS_REPLAY сохранил A4/4
и получил B2/4, но initial B был4/4 и один fixed layout не доказывает transfer.
Terminal-only replay отвергнут из-за забывания начала. Следующая гипотеза —
goal-conditioned temporal reachability; до неё campaign не масштабируется.

Exp143 уточнил термин: из episode order без counterfactual oracle учится
policy-dependent directed temporal proximity, не shortest-path reachability.
На frozen реальных исходах ordered probe дал MRR1.0, но малый world model не
переносил сигнал на imagined root. Mixed anchors, RGB salience, residual
correction и action-contrastive objective не прошли заранее выбранные causal
controls; снижение MSE не засчитано как улучшение решения. Fresh 256/128 model
с ordered temporal cost дала closed-loop 4/6 против raw0/6 и shuffled0/6, но
только на fixed Push-1: blue/purple прошли, yellow провалился. Seeds здесь не
новые задачи; topology/start/goal неизменны. Следующая проверка обязана делить
данные по целым layout/start/goal комбинациям и требовать другие первые actions.
Goal template сейчас зависит от push rule через pose и остаётся потенциальной
утечкой до перекрёстной physics×goal-pose интервенции.

Evaluator fixture теперь поддерживает `PushLayout`; layout является metadata
среды и не входит в Observation/model. Exp144 split четыре train/four test
layouts заранее и получил на пяти training runs ordered/raw/shuffled
56/12/30 successes из 120. Ordered обошёл raw во всех runs, но matched shuffled
control — неустойчиво; строгий gate 2/5. CUDA run с тем же seed не оказался
побитно воспроизводимым, поэтому seed означает declared initialization/sampling,
не уникальный deterministic checkpoint. Within-run arms остаются paired.
Debiased difference score провалился. Temporal score остаётся research probe,
не production default. Следующий контроль может заменить imagined MPC на
self-supervised hindsight goal-conditioned policy, но его успех будет claim о
goal control, а не доказательство world-model understanding или transfer A→B.

All-future hindsight policy на том же replay получила 0/24: правильный первый
поворот сменялся циклом `forward`. Terminal-only successful-episode selection
дала 64/72 на трёх runs, shuffled-action 0/72, goal-blind 44/72, ordered/raw MPC
56/72 и 28/72. Все runs покрыли четыре unseen layouts, но predeclared superiority
gate прошёл 1/3. Результат нельзя называть self-supervised: policy отбирает опыт
по `termination == success`, а backbone также видел termination supervision.
Goal-blind не является чистым causal control, потому что goal tile виден в
current RGB. До production integration проверяется frozen random encoder на тех
же 118 terminal pairs.

Frozen random encoder получил 0/72 против 60/72 у learned encoder на трёх runs;
action-shuffled также 0/72. Значит, случайных CNN features недостаточно, а
backbone training даёт полезное представление. Но backbone одновременно обучался
на predictive losses, `termination == success` и terminal-priority replay,
поэтому вклад JEPA/dynamics отдельно ещё не доказан. Следующий matched ablation
обнуляет termination loss и salience fraction только для backbone training;
terminal-only policy всё ещё остаётся success-supervised.

Matched predictive-only backbone (`termination_weight=0`, `salient_fraction=0`)
дал 72/72 против full64/72 и random0/72; predictive controller gate прошёл 3/3.
Таким образом, termination supervision и terminal-priority sampling не нужны для
полезного encoder в текущем Push-1 layout split. Это всё ещё не label-free agent:
policy dataset состоит из 118 пар, выбранных по успешному terminal outcome.
Перенос через новую physics/ruleset не проверен.

Exp145 mixed source-only run проверил long-distance split, но остановился раньше
physics gate. Идентичный fixed corpus содержал **2048 episodes / 130676
transitions**, fit terminal episodes east/west/south/north **2/4/2/5 = 13** и **104 terminal
examples**; all-future local examples — **741288**, batches balanced 50:50.
Runtime составил около **30 минут**. Real loss снизился **1.64197→1.04862**,
shuffled — **1.67165→1.57132**.
Source-geometry и controls дали **0/24** (shuffled **0/24**, frozen-random
**0/24**); physics transfer gate — `null`, не запускался. Во всех 24 real traces
первый turn был выбран правильно, затем controller повторял forward до box
(`turn=6`, `forward=186` на layout). Mixed local+terminal hindsight не решил
source prerequisite; это указывает на reactive-composition failure /
label-objective mismatch, а не на отсутствие turn recognition или physics
transfer failure. Controller остаётся reactive и сбрасывает representation после
каждого observation.

Exp146 проверил existing action-conditioned dynamics + beam planner на том же
source-only protocol. Dynamics loss был **1.35259→0.58296**, held-out ordered
probe balanced accuracy **0.7140** против shuffled **0.4914**, но ordered H3,
ordered H1, shuffled H3 и raw H3 получили **0/24** каждый. Ordered/raw H3 во
всех cases правильно поворачивались, затем семь раз выбирали заблокированный
`forward`; полный `55`-candidate search и neutral termination исключают
budget/terminal объяснение.

Один заранее локализованный late fork после canonical prefix проверил все
**125** трёхшаговых продолжений. Единственный success
`[interact,forward,interact]` был rank **1/125** по actual ordered/raw endpoint,
но rank **54/125** и **42/125** по predicted endpoint. Лучшие predicted планы
неуспешны, а exhaustive predicted ranking уже неверен; pruning вторичен. Это
evidence learned rollout error только для одного deterministic fork, не
доказательство общей неспособности encoder или world-model class.

Сохранённый checkpoint позволяет следующим дешёвым diagnostic разделить
one-step action prediction и autoregressive compounding без повторного обучения.
Exp147 показал one-step failure: два canonical `interact` хуже persistence в
**63.7x/130.2x**, а blocked `forward` даёт MSE **0.43084** при нулевой
persistence error. Свободный `forward` лучше persistence. Autoregressive MSE
растёт **0.14325→0.15883→0.23801**, но это не чистый compounding failure,
поскольку one-step prerequisite уже нарушен.

Проверен один deterministic unseen-layout fork. Пока неизвестно, является ли
contact error следствием sparse source coverage/objective или проявляется только
при layout generalization. Следующий checkpoint-only source-vs-unseen split
ответил: contact и blocked-noop failures присутствуют на **4/4 source** и
**4/4 unseen** layouts. Median interact prediction/persistence ratio равен
**49.14x/72.96x**, blocked-forward MSE **0.3321/0.3501**, тогда как свободный
forward лучше persistence (`0.2256x/0.2673x`). Это shared one-step failure, а
не преимущественно layout-generalization failure.

Пока не измерено, сколько state-changing interact и blocked no-op transitions
есть в исходном fixed replay. До этого нельзя различить data coverage и
objective/parameterization. Result не является physics-transfer failure;
Push-2 не запускался.

Exp149 воспроизвёл fixed corpus: full terminal counts east/west/south/north
**2/7/2/7**, fit-cutoff **2/4/2/5**. RGB-changing interact составляет
**1925/26125 = 7.37%** interact и **1.47%** всех transitions, но встречается в
**1281/2048** episodes. No-change coverage велико: interact **24200**, forward
**9358**, noop **26082**. Поэтому contact-change действительно недовзвешен
uniform objective, но отсутствие данных не объясняет failure, особенно для
identity/no-op. Следующий matched residual/persistence arm должен сохранить тот
же replay; event-balanced sampling — только отдельный causal control.

Exp150 добавляет экспериментальную residual latent parameterization:
`member_z = current_z + learned_delta`, с нулевой инициализацией только latent
heads. Encoder/GRU, predictive objective, fixed replay, seed и chunked update
protocol сохранены как в exp146. Production modules не меняются. Сохранённый
absolute checkpoint служит frozen baseline для того же exp148 one-step split.
Gates применимы только при совпадении полных default budgets и training config
с baseline; tiny smoke проверяет артефакты, но не scientific outcome.
Residual checkpoint имеет format version 2 и явный `residual_zero_init`, чтобы
absolute-only loader exp147 не интерпретировал delta heads как absolute state.
Это одна параметризация, один seed и одно семейство задач, без Push-2 и без
доказательства AGI/JEPA/physics transfer.

Полный `exp150-residual-dynamics-001` завершился с `exact_protocol=true` и
exit 0 на **2048 episodes / 130676 transitions**. Dynamics loss снизился
**1.31142→0.49581**; ordered probe validation balanced accuracy составила
**0.7138** против **0.4671** shuffled. Однако residual не снял prerequisites:
на source contact/blocked failures остались **4/4 и 4/4** (median interact
ratio **34.06x**, blocked MSE **0.3650**, free-forward ratio **0.1032**), а на
unseen — также **4/4 и 4/4** (**54.63x**, **0.4169**, **0.0944**). Late-fork
predicted rank улучшился с baseline **54/125** до **18/125** (raw **8/125**
против **42/125**), но predicted winners `[1,3,3]` остались неуспешны.
Ordered H3/H1, shuffled H3 и raw H3 дали по **0/24**. One-step,
source-compositional и composition gates — `false`, physics — `null`.
Следующий causal control — event-balanced sampling без одновременного изменения
planner, parameterization или objective.

Exp151 выполнил этот matched control на HyperPC. Артефакт
`output_to_user/core/exp151-event-balanced-dynamics-001` завершился с
`exact_protocol=true`, exit 0 и точным corpus **2048 episodes / 130676
transitions**. Pool содержал **1894 event** и **124686 ordinary** transitions;
sampler использовал **8000/8000** event/ordinary anchors, а с multi-step
targets итоговая supervision составила **8545 event / 39455 ordinary**
(**17.80% event**). State RGB-change при action 3 служит Push-local proxy
события, а не семантической меткой `interact` или универсальным task fact.

Dynamics loss снизился **1.31198895→0.42985901**; ordered probe validation
balanced accuracy равна **0.71299148** против **0.48725784** shuffled. Frozen
exp150 residual source baseline воспроизвёл contact/blocked failures **4/4 и
4/4** с medians interact **34.05918**, blocked **0.365026**, free **0.103175**.
Event-balanced source также остался на **4/4 и 4/4**, хотя medians interact и
blocked улучшились до **26.21614** и **0.151566**; free ухудшился до
**0.287251**. Unseen arm дал **4/4 и 4/4**, medians **40.01761**, **0.241659**
и **0.248932**.

На canonical late fork успешная последовательность получила predicted ordered
rank **95/125**, raw **104/125** и endpoint MSE **1.19277**; predicted winners
неуспешны. Все четыре eval arms дали по **0/24**. One-step, source-compositional
и composition gates — `false`, physics — `null`; Push-2 не запускался.
Event balancing частично уменьшает contact/blocked metrics, но не закрывает
failures и ухудшает rollout ranking. Поэтому ни sparse frequency сама по себе,
ни residual parameterization не являются root fix. Следующий bounded diagnostic
до архитектурного изменения — frozen-encoder representation separability по
event и free-vs-blocked labels.

Observability: **577** progress records, maximum gap **2.793 s**; полны
`run.log`, manifest, results, checkpoint, rows и traces; elapsed **134.097 s**.

## 2026-05-21 — Stage9X local survival affordances and remaining hostile wall
**Что сделано:** Ветка `feature/stage9x-capability-goal-handoff` доведена до
commit `b89e462`. Закрыты несколько локальных Stage9X gaps: goal-conditioned
frontier exploration для unknown goal targets, dynamic hostile targetability,
outcome-conditioned interaction continuation, и opportunistic local survival
affordances для textbook-declared positive body effects.

**Идеологическая граница:** правило не формулируется как `if water then drink`
или `hit exactly N times`. Факты остаются в `configs/crafter_textbook.yaml`
(`do <target> -> body +...`, `do <entity> -> remove_entity`), механизм
обобщённо применяет их к локально доступным действиям и уступает immediate
emergency threats.

**Проверка:** Focused pytest на minipc: 8/8. Seed17 planner-only 350-step
forensic run on `b89e462`: `episode_steps=253`, `death_cause=zombie`,
`opportunistic:water:do_survival_buffer` сработал на steps 20-21. Предыдущая
видимая dehydration-смерть в этом inspected window снята, но episode всё ещё
заканчивается hostile-pressure failure with depleted vitals
(`final_body={health:0, food:0, drink:0, energy:1}`).

**Ограничение:** это не Stage/phase PASS claim. Следующий bottleneck —
multi-threat combat/survival arbitration under depleted vitals, а не
resource discovery/execution.

## 2026-05-02 — Stage 90R Emergency Safety Controller
**Что сделано:** введён first-class emergency safety controller, который
активируется по explicit danger/vitals/outcome features, а не главным образом по
`actor != planner`; emergency-relevant Crafter facts частично вынесены в
`configs/crafter_textbook.yaml`; rescue telemetry расширена до activation
reasons / override source / utility components / immediate outcome delta.

**Результаты:**
- implementation commit: `71d1e29`
- focused tests: `23 passed`
- bounded HyperPC compare (`mixed_control_rescue`, symbolic, smoke-lite, seed 7,
  4 episodes, CPU-only):
  - candidate `avg_survival = 190.0`
  - frozen `9083357 avg_survival = 179.25`
  - `early_hostile_deaths_without_rescue = 0`
  - `hostile_deaths_without_rescue = 0`
- Stage decision: **PASS**

**Допущения/ограничения:**
- **Proof пока CPU-only.** Online GPU-path hang остаётся отдельной соседней
  проблемой и не закрыт этим stage.
- **Fallback field в eval payload не самодостаточен.** В clean hyper checkout
  baseline summary не подтянулся в JSON, поэтому verdict делался по frozen
  repository artifact, а не по встроенному `fallback_criterion.status`.
- **Textbook migration bounded.** Вынесены только факты, прямо нужные для
  emergency path; repo-wide cleanup Crafter literals не является частью этого
  stage.
- **Stimuli-layer вопрос остаётся открытым.** Этот PASS не доказывает, что
  полный stimuli refactor не нужен; он показывает, что правильный controller
  layer уже даёт измеримый rescue-side gain.

---

## 2026-05-02 — Stage 91 CUDA Eval Path Fix
**Что установлено:** HyperPC bare-command drift шёл не от shell activation, а от
editable-install в conda env: `__editable__.snks-0.1.0.pth` тащил `snks` из
`/opt/cuda/agi/src`, поэтому голый запуск не был self-contained относительно
verify checkout. После принудительного перехода на canonical checkout
обнаружился реальный GPU bug: mixed-control eval строил local advisory tensors
на CPU при том, что local evaluator был уже на CUDA.

**Что сделано:** в `experiments/stage90r_eval_local_policy.py` для eval path
явно прокинут `local_advisory_device=device`; helper `_eval_episode_rng(...)`
сохраняет deterministic per-episode arbitration для eval-only mixed-control
сценария. Добавлен focused test `tests/test_stage90r_eval_local_policy.py`.

**Результаты:**
- focused tests:
  - `tests/test_stage90r_eval_local_policy.py`
  - `tests/test_vector_mpc_agent.py`
  - result: `6 passed`
- HyperPC minimal GPU repro после фикса: **PASS**
- HyperPC full `seed=7` GPU compare после фикса: **PASS**
  - `avg_survival = 157.75`
  - `rescue_rate = 0.437`
  - `planner_dependence = 0.452`
  - `learner_control_fraction = 0.222`

**Canonical CUDA rule:**
- не полагаться на bare interpreter import path из env
- запускать verify checkout self-contained с явным
  `PYTHONPATH=<verify>/src:<verify>:<verify>/experiments`
- использовать тот же env interpreter:
  `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`

**Допущения/ограничения:**
- **CUDA path исправлен, но Stage 91 nondeterminism не закрыт.** GPU eval теперь
  проходит, однако multi-seed robustness regression остаётся отдельной задачей.
- **Editable-install env всё ещё drift-prone.** Пока `snks` editable install
  смотрит в `/opt/cuda/agi/src`, bare recorded command не является canonical
  способом воспроизведения verify checkout.
- **Исторический Stage 90R PASS остаётся CPU-based proof point.** Этот CUDA fix
  исправляет инфраструктурный eval path и device placement, но не переписывает
  исходное доказательство Stage 90R задним числом.

**Follow-up instrumentation:** для дальнейшего Stage 91 root-cause rerun в
`mixed_control_rescue` eval добавлены diagnostic fields `rescue_trace_tail`,
`local_trace_tail` и опциональный `death_trace_bundle`. Это не меняет policy
path; цель - перестать диагностировать слабые seed только по первым 8 шагам
trace и получить terminal-step evidence у смерти.

**Follow-up measurement fix:** summary теперь отдельно считает hostile deaths
`without_rescue`, `after_prior_rescue`, `with_terminal_rescue` и
`without_terminal_rescue`. Старое поле `hostile_deaths_without_rescue` само по
себе оказалось слишком слабым: оно скрывало failure mode “rescue активировался
много раз, но hostile death всё равно произошла”.

---

## 2026-04-20 — Stage 90 Reset: Viewport-First Local Survival
**Что установлено:** Stage 90 cause-finding и последующие diagnostics нашли реальные mechanism-баги
в short-horizon симуляции (`zombie/skeleton proximity damage`, сохранение `predicted_health=0`,
ложные remote `do target` gains), но даже после их фиксов главный survival wall не снят.
Видео и trace review показали более глубокую проблему: агент часто действует слабо связно
с текущей локальной сценой, пропускает ближайшие полезные affordance и плохо реагирует на
немедленную угрозу даже при видимом viewport evidence.

**Новые ограничения на следующий stage:**
- **`viewport-first`** — primary truth для выбора действия должен быть текущий viewport.
- **`near_concept` не является policy primitive** — допустим только для debug/compatibility.
- **Знания вне viewport вторичны** — `spatial_map` и дальняя память не должны вести policy path.
- **Локальная геометрия не должна схлопываться в один label** — нужен spatial local scene/tensor,
  а не ручной агрегат.
- **Local behavior должен учиться** — threat response, local opportunity и affordance не задаются
  новыми ручными эвристиками.

**Допущения/ограничения:**
- Возможно, текущий `planner + spatial_map` стек уже слишком перегружен для честного локального
  survival behavior; следующий stage должен проверить это напрямую.
- Улучшение выживания без улучшения локальной coherence считается подозрительным и должно
  трактоваться как потенциально тактическое.
- Следующий stage должен сначала доказать полезность viewport-first local behavior, и только
  потом возвращаться к усилению памяти, planner depth или Stage 91 validation.

**Промежуточные execution-выводы (smoke):**
- **Viewport-first dataset path работает end-to-end.** На `minipc` собран smoke-срез:
  `4` эпизода, `683` local samples, `avg_episode_steps=170.75`, death breakdown = `zombie: 4`.
- **Offline local evaluator trainable, но bias уже виден.** Smoke training на `683` samples дал
  `best_valid_loss=1.2884`, `valid_survival_acc=0.90`; значит short-horizon heads чему-то учатся,
  но это ещё не значит, что learned policy уже адекватна online.
- **Smoke local-only canary схлопнулся в константный action.** Первый online eval с smoke-checkpoint
  выбрал `move_right` на всех `690/690` шагах (`avg_survival=172.5`, deaths: `zombie=1`,
  `arrow=1`, `skeleton=1`, `alive=1`). Это важный warning:
  текущий малый dataset слишком policy-biased и пока учит directional prior, а не честное
  локальное поведение.

**Новые ограничения из smoke-run:**
- Перед claim про `viewport-first gain` нужен более широкий dataset, чем `4` smoke episodes.
- Offline метрики без online diversity-проверки недостаточны: модель может хорошо fit'ить survival head
  и всё равно выдавать дегenerate single-action policy.
- Следующий шаг должен проверять и устранять action-collapse, а не трактовать smoke-checkpoint как успех.

---

## Stage 88 — Knowledge Flow: Textbook Promotion (2026-04-16)
**Что сделано:** TextbookPromoter (YAML persistence), HypothesisTracker merge fix (accumulated n_obs across generations), PostMortemLearner.from_promoted() с консервативным bump=0.3. Дополнительно: exp136 добавил класс `arrow` в TileSegmenter; `_detect_sources` исправлен (entity-specific ranges, cow исключена); killing blow DamageEvent фикс; arrow регистрирован в entity_tracker.
**Результаты (88f, 5 gen × 20 ep, minipc):** gen1=189.4, gen5=179.7, ratio=0.949. Gates: **1/2 — secondary PASS (n_promoted=2 ✓), primary FAIL (ratio=0.949 < 1.20, gen5=179 < 210 ✗)**.
**Ключевые открытия:**
- **death=unknown устранены** — с entity-specific ranges (zombie≤6, skeleton≤10, arrow≤2) и killing blow фиксом: 0 unknown из 30 диагностических эпизодов.
- **arrow attribution работает** — exp136 сегментер + arrow в entity_tracker: arrow = 27% смертей (diag), атрибутируется напрямую.
- **Knowledge flow structural wall** — gen1 всегда лучший (189.4 > gen2-5). Гипотезы zombie+drink/food формируются корректно, но корреляция ложная: drink/food были низкими как следствие боя с zombie, не причина. Поднятие порогов виталов не влияет на zombie-боевую выживаемость.
- **Arrow dodge insight** — стрела летит 1 тайл/шаг по прямой, dodge механически возможен. Требует моделирования trajectory в VectorWorldModel (Stage 89).
**Допущения/ограничения:**
- **Primary gate неверно калиброван** — gate предполагал что vital thresholds — полезное знание против zombie. Оказалось нет. Survival ceiling ~190 определяется zombie-боями.
- **from_promoted() bump тюнинг** — bump=2.0 (88e) катастрофичен (ratio=0.85), bump=0.3 (88f) нейтрален (ratio=0.95). Знание нейтральное, не позитивное.
- **arrow_acc=25.8%** — backbone exp135 не имеет фич для стрел. Точность ограничена при frozen backbone.

---

## 2026-04-18 — exp137 Perception Agreement Retrain
**Что сделано:** новый retrain `exp137_segmenter_agreement.py` для `TileSegmenter` с другой objective:
не survival proxy, а agreement с semantic backend. Изменения:
- full fine-tune всего segmenter, а не только `head`
- input crop до реального world viewport `49x63` без HUD / black band
- hard-negative mining на кадрах, где `exp136` расходился с `semantic`
- отдельный eval `diag_perception_agreement.py`
- fix в `perception.py`: `near_concept` теперь выбирает non-empty concept в central `2x2`
  patch при tie, а не первый `empty` по scan order

**Результаты:**
- baseline `exp136` agreement (seed 42..45, 64 samples):
  - `near_match_rate = 0.859`
  - `mean_jaccard = 0.462`
  - `pixel_only_by_concept`: `cow=191`, `arrow=111`, `tree=46`, `skeleton=27`
  - `pixel_only row 5 = 290`
- holdout `exp136` agreement (seed 200..207, 185 samples):
  - `near_match_rate = 0.800`
  - `mean_jaccard = 0.487`
  - `pixel_only_by_concept`: `cow=546`, `arrow=320`, `tree=169`, `skeleton=69`
  - `pixel_only row 5 = 841`
- `exp137` train:
  - `8000` cropped frames (`4000 general + 2000 skeleton + 2000 hard negatives`)
  - `epoch119 val_tile_acc = 0.992`
- holdout `exp137` after near-fix (seed 200..207, 179 samples):
  - `near_match_rate = 1.000`
  - `mean_jaccard = 0.999`
  - only residual disagreement:
    - `pixel_only_by_concept = {"arrow": 3}`
    - `symbolic_only_by_concept = {"arrow": 1}`

**Ключевые открытия:**
- Основной bottleneck действительно был в CNN perception path, а не в Crafter render.
- Нижняя часть `64x64` frame (HUD + black band) загрязняла признаки; world-crop дал большой эффект.
- После retrain almost-all remaining mismatches оказались не ошибкой segmenter, а багом
  в `near_concept` tie-break.
- Визуальный audit на real GUI render подтвердил, что `exp136` дорисовывал `arrow/cow/skeleton`
  на траве и раздувал footprint’ы.

**Допущения/ограничения:**
- `exp137` пока валидирован только на agreement с semantic backend, а не на agent-level outcome.
- `mean_jaccard = 0.999` на holdout очень сильный результат; его ещё нужно проверять в живом
  agent loop, чтобы исключить hidden distribution gap между diagnostic sampling и policy rollout.
- Остаточный disagreement сосредоточен в `arrow`; dynamic-threat eval после perception fix
  обязателен перед любыми новыми claims про Stage 89 success.
- В первом `stage89 + exp137` smoke telemetry ошибочно выглядела как `arrow_threat_steps=0`.
  Root cause оказался не в `exp137`, а в missing fact: `arrow` отсутствовал в textbook vocabulary,
  поэтому `VectorWorldModel` не создавал concept, а `DynamicEntityTracker` не регистрировал projectile
  как dynamic entity. После добавления `arrow` в `configs/crafter_textbook.yaml` tracker сразу начал
  трекать projectile и восстанавливать velocity на live run.
- Следующий diagnostic bias оказался уже в самой telemetry Stage 89: `arrow_threat_steps`
  считались как "любая видимая стрела", хотя для большинства таких шагов `predicted_baseline_loss=0`
  и defensive action не требуется. Targeted seed44 diagnostic после фикса `arrow:proximity` дал:
  `arrow_visible_steps=66`, но из них только `imminent_steps=13`, и planner выбрал защитное движение
  на всех `13/13` imminent cases. Значит низкий `defensive_action_rate` на visibility-denominator
  переоценивал planner failure; threat telemetry должна быть привязана к imminent damage within horizon,
  а не к простой projectile visibility.
- Следующий structural bug оказался не в механике Crafter, а в нашем perception→map layer.
  Trace по `seed=44` и чтение исходника Crafter показали:
  - `tree` в Crafter всегда даёт `wood`
  - `sapling` приходит только из `grass`
  - когда агент получал `sapling` на supposedly `tree:do`, реальный `env_material_before`
    на facing tile был `grass`
  Root cause состоял из двух lower-layer ошибок:
  1. **viewport→world off-by-one по Y** — detections в `spatial_map` и `DynamicEntityTracker`
     писались со сдвигом на `+1` по второй координате;
  2. **stale off-center labels** — perception не эмитил `empty` вне центрального patch,
     поэтому старые `tree`-метки не затирались, когда тайл уже стал `grass`.
  После фиксов:
  - `seed=44` short trace: `n_frustrated_tree_do = 0`, `n_successful_tree_do = 3`
  - на успешных шагах `facing_label_before = tree`, `env_material_before = tree`,
    `inventory_delta = {"wood": 1}`
  Значит странный `tree/do` loop был не planner-магией и не "неоднородной семантикой дерева",
  а рассинхроном карты мира с реальным Crafter tile truth.
- Replay-audit на соседних seed после тех же фиксов показал, что это был не узкий single-seed кейс:
  - `seed=43` short trace: `n_frustrated_tree_do = 0`, `n_successful_tree_do = 5`
  - `seed=48` short trace: `n_frustrated_tree_do = 0`, `n_successful_tree_do = 5`
  - на успешных шагах в обоих seed'ах `facing_label_before = tree`, `env_material_before = tree`,
    `inventory_delta = {"wood": 1}`
  Значит adjacent resource interaction на fresh stack больше не выглядит главным bottleneck.
  Следующий structural wall теперь выше: broad survival policy и hostile-contact management
  против `zombie/skeleton`, а не perception/resource execution.

---

## Stage 87 — Curiosity About Death (2026-04-15)
**Что сделано:** DeathHypothesis (корреляция причины смерти с уровнем витала) + HypothesisTracker (накапливает per-episode данные, порождает верифицируемые гипотезы). CuriosityStimulus обновлён: `U = weight × avg_surprise × death_relevance`, где death_relevance ∈ [1.0, 2.0] — близость витала к порогу гипотезы. PostMortemLearner.build_stimuli() добавляет CuriosityStimulus при наличии активной гипотезы.
**Результаты (20 эп, minipc):** avg_survival=186.85. n_verifiable=4, curiosity_active_episodes=17/20. Gates: **3/3 PASS**.
**Допущения/ограничения:**
- **Гипотезы корреляционные, не каузальные** — `zombie + drink < 3` означает "при low drink чаще умираю от zombie", не "drink вызывает zombie". Механизм не объясняется, только коррелируется.
- **Пороги фиксированы** — `{food: 3.0, drink: 3.0, health: 4.0, energy: 2.0}`. Адаптивные пороги (через PostMortemLearner) — Stage 88 scope.
- **Гипотезы не персистируются** — сбрасываются при новом запуске. Persistence — Stage 88 (Knowledge Flow).
- **cow как причина смерти** — агент иногда получает урон рядом с коровой (collision?). `_detect_sources` улавливает любой entity в `dist <= 2`. Корова как источник смерти не идеологически осмыслена.
- **death_relevance только по виталам** — entity proximity в VectorTrajectory недоступна (нет в VectorState). Trajectory relevance вычисляется только через body dict.

---

## Stage 86 — Post-Mortem Learning (2026-04-15)
**Что сделано:** DamageEvent log (накопление per-step при health_delta<0), PostMortemAnalyzer (temporal-decay attribution, многофакторный), PostMortemLearner (обновляет HomeostasisStimulus thresholds + health_weight между эпизодами). HomeostasisStimulus переведён на deficit-based scoring с per-vital thresholds.
**Результаты (20+20 эп, minipc):** avg_survival(with_pm)=179.7. zombie_deaths early=6→late=3, starvation with_pm=0 < without_pm=1. Gates: **3/3 PASS**.
**Допущения/ограничения:**
- **Gate 2 узкий** — starvation deaths: 0 vs 1. Агент редко умирает от голода (GoalSelector хорошо справляется с food). Разница статистически мала.
- **death_cause=alive баг** — мгновенная смерть (лава / урон в последний шаг) даёт пустой damage_log → dominant_cause="alive". Финальный damage не фиксируется т.к. break до следующего body read.
- **food_threshold почти не рос** — food редко = 0 при смерти. Агент умирает преимущественно от зомби, health_weight вырос 1.0→2.43.
- **Параметры не персистируются** — сбрасываются при новом запуске. Cross-run persistence — Stage 88 scope.

---

## Stage 85 — Goal Selector Design (2026-04-15)
**Что сделано:** GoalSelector — выбор цели из textbook rules. `total_gain` заменён на `Goal.progress(trajectory)`. Proactive crafting chain: нет дерева + нет меча → `gather_wood`. VectorTrajectory.confidences + vital_delta/inventory_delta/item_gained. CuriosityStimulus определён (Stage 87 debt).
**Результаты (20 эп, minipc):** avg_survival=197.0, wood_ge3_pct=10%, no_total_gain=✓. Gates: **3/3 PASS**.
**Допущения/ограничения:**
- **wood_ge3_pct=10% на грани** — только 2 эпизода из 20 с wood≥3 (ep3=5, ep10=4). Зависит от плотности деревьев в map seed. Стена = stale spatial_map + segmenter ghost trees.
- **Proactive crafting threshold** — `chain_cost` = сумма всех требований по material (для wood=5). После сбора 5 дерева цель переключается на explore. Если витали упадут раньше, wood=1-2.
- **Goal.explore() + sleep** — исправлен: self-action trajectories дают explore_progress=0, иначе агент спал при полных виталах.
- **Spatial map ghosts** — segmenter иногда метит тайл игрока как "tree" (near=tree при H9/F9). `find_nearest` пропускает player_pos (Bug 5), но stale entries в других позициях всё ещё вводят в заблуждение.

---

## Stage 84 — Real Stimuli Infrastructure (2026-04-15)
**Что сделано:** Vital fix (body читается из `info["inventory"]`, не из top-level `info`) + StimuliLayer (Category 4): SurvivalAversion + HomeostasisStimulus вынесены из `score_trajectory`.
**Результаты (20 эп, minipc):** avg_survival=178.9, wood=0%, sleep%=0%. Gates: 2/3 (survival ✓, sleep_not_stuck ✓, wood ✗).
**Допущения/ограничения:**
- **Wood=0** — плановая стена, не регрессия. Агент не собирает дерево из-за отсутствия curiosity сигнала. `total_gain` знает про wood (Crafter-специфично), но плановщик не генерирует цепочки gather → craft. Stage 85 scope.
- **sleep%=0%** — sleep не выбирается потому что реальные витали почти всегда полные в начале эпизода (пассивный decay не учтён в симуляции). Это ожидаемо и правильно.
- **Passive body decay не в симуляции.** `simulate_forward` не применяет ambient decay (еда/питьё падают ~1/step). Агент не предсказывает "через 5 шагов food=2". Stage 85 может адресовать через CuriosityStimulus.
- **HomeostaticTracker** получает `inv` без body-переменных (фикс Stage 84). Один-эпизод transient в `observed_rates` при первом запуске после апгрейда — безвреден.
- **score_trajectory 4-tuple → 3-tuple.** Все callers обновлены. `diag_stage83` файлы архивированы.

---

## Stage 44 — Foundation Audit
**Что сделано:** Аудит DAF-ядра, выявление ограничений FHN-осцилляторов.
**Допущения/ограничения:**
- FHN работает в возбудимом режиме (I_base=0.5), а не осциллирующем — coupling для планирования непригоден.
- SKS формируются через SDR-инъекции, а не через динамику coupling.
- DAF остаётся как perception layer без возможности планирования.

---

## Stage 47–49 — M1: Generalization (DoorKey, MultiRoom)
**Что сделано:** 100% DoorKey + 100% MultiRoom-N3.
**Допущения/ограничения:**
- **Полная наблюдаемость** — BFS работает на полной карте. Реального обобщения нет, задача тривиальна при full obs.
- SDM не используется для навигации — BFS решает всё.

---

## Stage 54 — Partial Observability
**Что сделано:** 100% DoorKey с 7×7 view через SpatialMap.
**Допущения/ограничения:**
- SpatialMap строится из символьных observations (тип/цвет клетки), не из пикселей.
- Агент помнит карту идеально — нет шума, нет forgetting.

---

## Stage 59 — VSA Causal Induction
**Что сделано:** 100% generalization на unseen colors через bind(X,X)=identity.
**Допущения/ограничения:**
- Тестируется только цветовая генерализация, не объектная (key→ball не проверялось).
- Демонстрации синтетические, не из реальной среды.

---

## Stage 60–61 — World Model + Demo-Guided Agent
**Что сделано:** 100% QA L1-L3, 100% DoorKey + LockedRoom.
**Допущения/ограничения:**
- Правила получены из синтетических демонстраций, не из реального взаимодействия со средой.
- Per-rule SDMs ограничены ~50 элементами — не масштабируются на большие домены.

---

## Stage 62 — CLS World Model
**Что сделано:** 100% QA L1-L4, neocortex + hippocampus.
**Допущения/ограничения:**
- Неокортекс = обычный Python dict (exact match). Нет generalization за пределами VSA identity property.
- Navigation policy на SDM сделала навигацию хуже (16% vs 44%) — BFS оставлен без изменений.
- Write-on-surprise: 77% записей пропускается, но hippocampus недообучен на редких ситуациях.

---

## Stage 63 — Abstraction + Crafter
**Что сделано:** 100% Crafter QA, 25 auto-категорий.
**Допущения/ограничения:**
- Абстракции извлекаются из символьных правил, не из пикселей.
- 25 категорий покрывают Crafter, но не тестировались на других доменах.
- Craft-действия (make_*) не покрыты prototype memory — 0 прототипов из-за редкости ситуаций.

---

## Stage 64 — No Synthetic
**Что сделано:** 93% Crafter QA без синтетических демо.
**Допущения/ограничения:**
- Exploration curiosity-driven, но не affordance-based — агент случайно натыкается на правила.
- 7% miss rate остаётся — редкие объекты (diamond, iron) не обнаруживаются за отведённое время.

---

## Stage 65 — Calibrated Uncertainty
**Что сделано:** Brier=0.12, calibration curve близка к идеальной.
**Допущения/ограничения:**
- Калибровка проверена только на known объектах. Поведение на out-of-distribution не тестировалось.
- Confidence threshold подобран эмпирически, не обоснован теоретически.

---

## Stage 66 — Pixels (Prototype Memory)
**Что сделано:** 100% Crafter QA из пикселей, prototype memory k-NN.
**Допущения/ограничения:**
- **Conv2d сегфолтит на AMD ROCm** через MIOpen backend. Исправлено: `torch.backends.cudnn.enabled=False` включает fallback kernel (медленнее, но работает). Обучение теперь на GPU (1.8x speedup vs CPU).
- VQ Patch Codebook отброшен: decode→symbols теряет информацию на каждом шаге.
- Prototype search в Phase 3 использует ground truth near (символьный), не CNN.
- make_* правила по-прежнему не покрыты (0 прототипов для craft actions).

---

## Stage 67 — Symbolic Near → CNN Near
**Что сделано:** NearDetector (CNN→argmax), CrafterPixelEnv без _to_symbolic(), smoke 99%, QA 100%.
**Допущения/ограничения:**
- **Инвентарь** берётся из `info["inventory"]` — проприоцепция (агент помнит что взял). Менять не нужно.
- **Навигация** — случайный walk + `_detect_near_from_info(info["semantic"])`. Stage 68 убирает info["semantic"].
- near_labels для обучения CNN = ground truth из той же символики (circular dependency: убираем символику, обучаясь на ней).
- Prototype collection в Phase 3 использует ground truth near для поиска ситуаций, не NearDetector.
- make_* правила по-прежнему не покрыты.
- Smoke test 99% оптимистичен: большинство кадров — пустое поле (easy "empty" class).

---

## Stage 68 — Pixel Navigation (когнитивная карта)
**Что сделано:** CrafterSpatialMap + find_target_with_map. Nav smoke 72%, QA 100%, regression 100%.
**Допущения/ограничения:**
- `info["player_pos"]` остаётся — проприоцепция.
- `info["inventory"]` остаётся — проприоцепция.
- CrafterSpatialMap: nav map=72% vs random=69% — небольшое преимущество (знакомые позиции).
- coal/iron/diamond: 1/0/0 из 50 seed — редкие объекты не покрыты навигацией.
- make_* прототипов 0 — table не создаётся при random walk (нет wood в инвентаре).
- near_labels для обучения CNN по-прежнему из символики (circular dependency: Stage 69).

---

## Stage 70 — ScenarioCurriculum (2026-04-06)
**Что сделано:** FSM-цепочки сценариев с OutcomeLabeler. 6 классов (empty/tree/stone/coal/iron/table). Smoke=68.8%, QA=100%, regression=100%.
**Компоненты:**
- ScenarioRunner: FSM executor с directional probing (do в 4 направлениях), window labeling W=5.
- CrafterControlledEnv: прямое редактирование мира — reset_near() и reset_with_items().
- _collect_empty_walk_frames(): random walk + semantic GT для "empty" (соответствие test distribution).
- STONE/COAL/IRON_CHAIN: controlled env для редких объектов (100% success vs ~3% natural).

**Допущения/ограничения:**
- **Nav encoder Phase 0** по-прежнему через exp122 (Stage 68 pipeline, символьные траектории). Circular dependency не устранена на уровне nav encoder. Stage 71 устраняет.
- **Stone в smoke=4.5%** из-за domain gap: controlled stone помещается в grassland, а в random walk stone появляется в горах. Smoke проходит за счёт empty=97.5%.
- **use_semantic_nav=True** используется в TREE/STONE/COAL/IRON_CHAIN для навигации — semantic scaffolding. Stage 71 убирает.
- **_balance_chunk monkeypatching**: отключение врагов через lambda. Работает, но хрупко.
- **table smoke=0/0**: таблицы не встречаются в random walk (только player-placed), поэтому GT=0 для table в smoke test.
- **do near coal 6/50, iron 17/50** в QA: coal/iron embedded в камне, навигатор не находит их за 300 шагов. QA проходит за счёт controlled prototype collection в phase4.

---

## Stage 71 — Text-Visual Integration (2026-04-07)
**Что сделано:** Соединение текстового и визуального пайплайнов. ConceptStore как единое омнимодальное хранилище. Каузальные правила из текстового "учебника" (YAML), visual grounding через co-activation, backward chaining planner, reactive zombie handling.
**Компоненты:**
- ConceptStore: unified Concept (visual + text_sdr + attributes + causal_links + confidence).
- CrafterTextbook: YAML с 10 атомарными правилами, regex-парсер, load_into(store).
- ChainGenerator: backward chaining через ConceptStore.plan() → ScenarioStep.
- GroundingSession: co-activation — K=5 visual samples + text SDR per concept.
- ReactiveCheck: zombie nearby → sword? attack : flee.
- ScenarioRunner.run_chain_with_concepts(): reactive layer + prediction error loop.

**Допущения/ограничения:**
- **PrototypeMemory не интегрирован** в ConceptStore — ConceptStore хранит 1 z_real на концепт, PrototypeMemory хранит тысячи экземпляров. Разные задачи.
- **Confidence delta фиксированный** (±0.15), не байесовский update.
- **Surprise только логируется** — неожиданные события не порождают новые правила автоматически. Требует отдельной проработки.
- **Flee = простая эвристика** (3-5 случайных шагов от врага).
- **Один тип врага** (zombie). Skeleton, arrow — будущий stage.
- **Нет стратегии "сначала крафт меча"** — reactive слой не планирует заранее.
- **Нет decay правил** — confidence не падает со временем.
- **Planner не оптимизирует порядок** — может собирать wood дважды.
- **Nav encoder Phase 0 на exp122** — фокус этого stage = text-visual, не nav cleanup.
- **use_semantic_nav=True остаётся** для редких объектов.
- **find_causal disambiguation** — при одинаковых requires выбирается наиболее специфичный match (по количеству requires items). Wood_sword/wood_pickaxe неразличимы по inventory.
- **Zombie + Survival Gate 5 PASS** (exp128d) — zombie_deaths 41→1, episode length 169→446 (2.65x). Survival rules (food/drink/energy) = основной вклад в выживаемость. Zombie flee = дополнительный.

---

## Stage 72 — Perception Pivot (2026-04-07)
**Что сделано:** Замена supervised NearDetector на ConceptStore.query_visual_scored() (cosine sim). Убран GT semantic nav. Автономный цикл perceive→decide→act→learn. Experiential grounding (one-shot + EMA). Drive-based goal selection. Prediction-verification loop.
**Компоненты:**
- perception.py: perceive(), on_action_outcome(), select_goal(), get_drive_strengths().
- ConceptStore.query_visual_scored(): возвращает (concept, similarity) для детекции "unknown".
- agent_loop.py: автономный цикл вместо ScenarioRunner chains.
- engine.py: use_semantic_nav=False, spatial_map в engine state.

**Допущения/ограничения:**
- **CNN encoder frozen** (exp128) — не дообучается в runtime. Фичи могут быть недостаточны для stone vs coal.
- **One-shot grounding noisy** — первый z_real может быть нетипичным. EMA сглаживает.
- **Cosine threshold=0.5** — подобран эмпирически, не адаптивный.
- **Spatial map cold start** — пустая карта в начале эпизода, random walk для заполнения.
- **Drive competition = max()** — нет GWS winner-take-all, простой argmax по drive strengths.
- **Sleep = 3 шага** — фиксированное количество, не адаптивное.
- **Probing = 4 directions** — для "do" пробуем все стороны. Неэффективно, но надёжно.
- **Replan interval=20** — фиксированный, не событийный.
- **DAF/SKS не интегрированы** — ConceptStore.query_visual() заменяет оба. Oscillator perception deferred.
- **info["player_pos"] остаётся** — проприоцепция.
- **info["inventory"] остаётся** — проприоцепция.
- **NearDetector code сохранён** — используется для backward compat (zombie tracking в wrapper).
- **exp130 результаты:** tree nav 60% PASS, stone 0%, coal not grounded, survival 74, verification 0.
- **exp131 результаты (Stage 74, homeostatic):** tree nav 50.5% PASS, 7 concepts grounded (incl stone), survival 138, verification 3 PASS.
  - HomeostaticTracker: body rates from observation, preparation drive (proactive sword craft).
  - Relative matching (margin ≥0.1): fixes 256-dim inter-class confusion.
  - Sword emergence: plan reaches step 2/4 (make sword near table) but table recognition fails.
  - Root cause: 256-dim inter-class similarity too high (stone vs water: 0.82).

---

## Stage 74 — Homeostatic Agent (2026-04-08)
**Что сделано:** Убраны ВСЕ hardcoded drives и ReactiveCheck. Поведение из body rates + world model + curiosity. CNN 256→512 channels.
**Компоненты:**
- HomeostaticTracker: rate of change body variables + conditional rates (STDP-like).
- compute_drive: urgency = 1/steps_until_zero (pure body physics).
- compute_curiosity: model incompleteness (biological drive).
- Preparation drive: trace known threat → plan to remove cause → proactive craft.
- Strategy 2: health drops → cause=zombie → kill_zombie → sword chain.
- Relative matching: margin ≥0.1 between best/second-best (fixes inter-class confusion).
- CNN 512 channels: retrained, +9% survival.
- ReactiveCheck REMOVED: flee wastes steps, drives handle zombie correctly.

**Результаты (exp131, 500 episodes):**
- Tree nav: 54.4% PASS
- Grounding: 6 concepts PASS (tree, table, empty, zombie, water, cow)
- Verification: 4 rules PASS (tree.do→wood 1.00, empty.place→table 1.00, water.do→restore_drink 1.00, cow.do→restore_food 0.80)
- Survival: 173 steps FAIL (gate ≥200). Стабильно, не растёт с обучением.
- Stone: 0% FAIL. Sword: 0/500.

**Допущения/ограничения:**
- **Survival 173 — потолок текущей архитектуры.** 500 эпизодов не улучшают — learning saturated.
- **Sword 0/500** — agent выводит правильный план (kill_zombie→sword→table→wood) но не успевает собрать 3 wood до смерти (~80 шагов на 3 дерева, zombie убивает за ~100-170).
- **512-dim features:** intra-class 0.99, inter-class 0.55-0.82. Relative matching помогает но не решает.
- **Bottleneck = perception speed.** Agent тратит ~80 шагов на поиск 3 деревьев при текущем качестве cosine matching. Нужно либо быстрее perception, либо рicher features.
- **ReactiveCheck убран** — flee хуже чем терпеть удары. Drives правильно выводят "крафти меч".
- **Архитектура чистая:** zero hardcoded strategy. Sword emergence подтверждён (1 craft в exp131 ранних итерациях).
- **SupCon on center features не помог** (exp132 supcon: survival 169 vs 173 без). Classification features ≠ metric features — разный training objective не решается добавлением contrastive loss.
- **500 эпизодов — learning saturated.** Survival стабильно 169-173, не растёт. Bottleneck архитектурный.
- **Лучший результат:** exp131 без ReactiveCheck, 512ch CNN 4×4 grid: survival 173, tree 54%, 6 concepts, 4 rules verified.
- **8×8 grid (256ch, 3 layers) не помог:** survival 164, tree 48%. Больше позиций (64 vs 16) но quality не улучшилась.
- **Sandbox curriculum не помог:** survival 165. Prototypes не стали точнее — CNN features fundamentally не подходят для cosine matching.
- **Diagnostic: 100% stale map entries.** perceive_field возвращает "tree" на траве. 4 тайла в одной ячейке 4×4 → смешанные features. 8×8 (~1 тайл) не решило — проблема в training objective, не разрешении.
- **512ch×8×8 grid: survival 169.** Лучше чем 256ch×8×8 (164) но хуже чем 512ch×4×4 (173). Чистые ячейки (1 tile) не помогают — CNN features всё равно не metric space.
- **Root cause подтверждён окончательно:** classification CNN features (cross-entropy) не образуют metric space для cosine matching. Проблема в training objective, не в resolution/channels/grid. Протестировано: 256ch×4×4, 512ch×4×4, 256ch×8×8, 512ch×8×8, SupCon, sandbox, 500 episodes. Ничего не помогает. Нужен metric learning или near_head для detection.

  - 3 концепта grounded из опыта: tree, water, cow.
  - Motor babbling (15% prob) → action outcome → one-shot grounding → perception bootstrap.
  - Survival +50% (49→74) через grounding cow/water для еды/питья.
  - Stone FAIL = не перцепция, а planning execution (craft chain не реализован).
  - Verification FAIL = predict/verify не подключён в babble path.

## Stage 75 — Per-Tile Visual Field (2026-04-09)
**Что сделано:** Заменил cosine matching на classification CNN с no-stride FCN architecture.
tile_head через Conv1×1 на output feature map. Per-tile labels через semantic map как teacher.
Полный viewport 7×9 tiles (49×63 px из 64×64), исключая inventory bar и черную границу.

**Компоненты:**
- TileSegmenter: 3× Conv3×3 (stride=1) + BN + ReLU → AdaptiveAvgPool(7,9) → Conv1×1(64,12). 57K params.
- viewport_tile_label: корректный coordinate mapping с учётом render transpose + sprite offset +1.
- ConceptStore.plan(goal, inventory): skip prerequisites уже в inventory.
- Textbook restore_health rules: do cow/water restores health (matches Crafter implicit regen).
- Homeostatic bugs fixed: tb.body_rules property access, plan verification before advance,
  cumulative requires for do, probe_dirs rotation, make/place unconditional advance.

**Результаты (exp135):**
- Tile accuracy: **82%** PASS (was 39% in Stage 74)
- Wood collection: **4.7/ep avg, 65% reach ≥3 in 17 steps** PASS
- Survival with enemies: **178 avg** FAIL (gate ≥200). Variance 94-264.
- Per-class acc: water 83%, tree 67%, stone 65-85%, coal 97%, cow 81%, zombie 100% (small n), skeleton 0-63%.

**Допущения/ограничения:**
- **Survival ≥200 — architectural limit.** 11+ кодовых фиксов не дали результата выше 189.
  Pattern: каждый фикс решает симптом, но рождает новый — signal of architectural gap.
- **Root cause:** plan execution linear/blind. Agent commits to kill_zombie (4 steps), нет
  forward simulation "выживу ли я во время этого плана?". Zombies attack during execution, dies.
- **Skeleton detection 0-63%:** training distribution has 168 skeleton tiles of 670K total (0.025%).
  Class-weighted CE помогает частично но не решает. Accepted limit.
- **Placed table detection unreliable:** 70 table samples in training. Agent кладёт table,
  не находит потом для make_wood_sword. Spatial_map manual update — procedural patch, reverted.
- **Coordinate mapping discovered via visual debug.** Crafter canvas.transpose((1,0,2)) + sprite
  offset +1 row. Labels were wrong in Stage 74, per-tile accuracy был ограничен noise в GT,
  не feature quality. See `feedback_visual_debug.md`.
- **Procedural patches rejected per ideology:** hardcoded flee reflex, flee_timer panic,
  stuck detection random, range-based threat check, manual spatial_map updates. Все были tried
  and reverted. See `feedback_no_hardcoded_reflexes.md`.
- **Cumulative requires check correct:** do(tree) stays until sum of requires across all
  subsequent plan steps is met. Prevents place_table at 2 wood when make_sword also needs 1.
- **explore_action babble conflicts with plan:** during plan execution, babble may consume
  resources (make_wood_pickaxe съедает wood для sword). Solution was to use random walk в
  fallback, но это тоже procedural patch и reverted.
- **Best variance observed:** survival ranges 94-264 within single 20-episode run. Stochastic
  due to zombie spawn positions. Increasing sample size would stabilize but not move mean above 200.
- **Next:** Stage 76 — continuous model learning / forward simulation. Model-based планирование
  через ConceptStore + tracker + learned value function. See
  `docs/superpowers/specs/2026-04-09-stage76-continuous-model-learning-design.md`.


---

## Stage 76 — Continuous Memory-Based Learning
**Что сделано:** Полный memory pipeline: SDR state encoder (4096 bits), EpisodicSDM (FIFO 10K),
deficit × delta action scoring, softmax selection, opt-in AttentionWeights. 90 тестов, Gate 5
автоматический lint на ideology-нарушения (нет hardcoded drives/derived features).

**Компоненты:**
- `bucket_encode`, `FixedSDRRegistry`, `SpatialRangeAllocator` — SDR primitives
- `StateEncoder` — raw inventory/visible/spatial → 4096-bit SDR, bit layout body/inv/vis/known
- `EpisodicSDM` — FIFO buffer, popcount recall, bootstrap gate via `count_similar`, `min_sdm_size` threshold
- `continuous_agent.run_continuous_episode` — decision loop, branch: SDM path OR bootstrap (ConceptStore plan)
- `tile_segmenter.py` — Stage 75 checkpoint extracted to reusable module, GPU-aware loader

**Результаты (exp136, 3 × 20 eval episodes на minipc):**
- Stage 76 v1 FIFO: **survival 177** (eval runs 184/184/163), wood ≥3: 8/20 (40%)
- Stage 76 v1+priority (A+B): **166** — buffer не заполнился, priority не сработал, rng drift, reverted
- Stage 76 v2 attention: **166** — wood deficit дoминировал mask, reverted
- Stage 76 v2.1 attention+body_vars filter: **173** — warmup-enemy улучшился до 190, но wood=0, reverted

**Допущения/ограничения:**
- **Gate 1 (survival ≥200) FAIL — architectural wall.** 4 запуска (Stage 75 baseline + 3 Stage 76
  варианта) дали результаты 166-177, все в ±10 от 178. Reactive memory-based policy ≡ scripted bootstrap.
- **Root cause:** reactive policy не может избежать угрозу, которую ещё не получила. Рекол возвращает
  single-step (state, action, outcome) tuples, а не многошаговые траектории. Scoring суммирует вклады
  отдельных одношаговых решений — это fancy 1-step Q-learning.
- **Smoke test (no enemies, T=0.3): 17-19/20 эпизодов доживают до max_steps=200 с H=9.** В safe режиме
  SDM policy полностью компетентна. Проблема именно в enemy avoidance.
- **Cause of death pattern:** `cause=health` в 80%+ случаев с food/drink=3-7 (не голод, а прямой урон
  от зомби/скелетов). Один эпизод warmup-safe дожил до 500 шагов (max) — доказательство что bootstrap
  path может выживать в safe mode.
- **Density не попала в целевой 5%:** landed ~10-13% (400-550 active bits) потому что window=40
  нужен для ≥80% adjacent-value similarity. Spec's nominal sparsity target был математически недостижим.
- **SDM capacity 10K at 180 avg eps wraps in ~55 eps.** Единственный 500-step 'alive' эпизод был
  вытеснен до eval phase. Priority eviction не помогла (buffer не филлится полностью в 50K).
- **Attention's `observed_max[inventory_item]` grows unboundedly.** Wood's deficit=9 стал сравним с
  health's. body_variables() filter помогает survival но ломает wood collection. Tradeoff не даёт gate.
- **Bootstrap→SDM transition через min_sdm_size=2000** работает корректно — первые ~10-12 eps pure
  bootstrap. Но на momentum bootstrap и SDM path дают одинаковые результаты.
- **Все ideology gates PASS:** no hardcoded drive list, no derived features, no argmax over drives,
  no `if inv.get("X") < N` patterns, attention mask композируется через тracker.body_variables()
  (legit через textbook), update учится для всех observed_variables.
- **Next:** Stage 77 — forward simulation через SDM transitions. Query SDM as a simulator, roll
  forward N шагов, scoring по накопленным body_delta по траектории. Re-uses Stage 76 substrate.


---

## Stage 77a — ConceptStore Forward Simulation + MPC
**Что сделано:** Полный MPC loop через `simulate_forward` по structured `RuleEffect` правилам.
Stage 76 memory substrate полностью удалён (Commit 7). Три категории знаний —
facts (textbook YAML) / mechanisms (simulate_forward dispatch) / experience (tracker + spatial_map)
— разделены и протестированы. 140 stage77 тестов зелёные, полный пайплайн без хардкода.

**Компоненты:**
- `forward_sim_types.py` — RuleEffect (10 kinds), StatefulCondition, SimState, SimEvent,
  Trajectory, Failure, Plan, PlannedStep (unified, no legacy fields)
- `concept_store.simulate_forward(plan, state, tracker, horizon)` — 6-фазный tick dispatch
  (body_rate → stateful → action_triggered → clamp → spatial → movement → step)
- `concept_store.plan_toward_rule(rule, state, store)` — backward chain с resolved prerequisites
- `concept_store.find_remedies(failure)` — query world model for counter-rules
- `HomeostaticTracker` — innate/observed split + Bayesian `w·innate + (1-w)·observed`,
  `vital_mins` только для catastrophic death, `init_from_textbook` идемпотентна
- `mpc_agent.run_mpc_episode` — ре-планирование каждый тик, 5-7 кандидатов, лексикографический
  score `(survived, neg_time_to_death, resources_gained, exploration)`, execute first primitive only
- `crafter_spatial_map._blocked` — observation-based blocked-tile learning
  (`prev_move && prev_pos==pos → mark_blocked`), без хардкоденных wall-avoidance reflexes
- `configs/crafter_textbook.yaml` — structured-YAML only (regex fallback removed), rough
  directional priors (`body decay -0.02`, `zombie spatial -0.5`, `skeleton range=5 -0.5`)

**Результаты (exp137_run8, minipc):**
| Phase | avg_len | Notes |
|---|---|---|
| Warmup A (no enemies) | 222 | Tracker накапливает background rates |
| Warmup B (enemies on) | 203 | Spatial/stateful damage conditioning |
| Eval run 0 (20 eps) | 193 | wood=0.4, cause=health×20, max=393 |
| Eval run 1 (20 eps) | 171 | wood=0.1, cause=health×20, max=396 |
| Eval run 2 (20 eps) | 175 | wood=0.1, cause=health×20, max=250 |
| **Overall eval** | **180** | per-run ≥200: False, overall: False |

- **Gate 1 (survival ≥200) FAIL at 180** — same wall as Stage 76 (178). Variance 171-193 между
  runs, max-per-episode 393/396/250 → архитектура имеет запас, но без runtime rule induction
  не может найти правильный план стабильно.
- **Gate 3 (wood ≥3) FAIL at 0/20** — MPC scoring `(survival > wood)` лексикографически
  душит wood gathering. Это следствие vital_mins и rough priors, не архитектурный баг.

**Допущения/ограничения:**
- **Rough directional priors only.** Textbook хранит качественные значения (`body -0.02`,
  `damage -0.5`), не точные данные из Crafter source. Точные ставки должны приходить через
  `tracker.observed_rates` — Bayesian combination уверенно сдвигает rate к наблюдению после
  ~200 observations. Идеологическое решение от user: "я за идеологию, пусть метрики хуже".
- **No surprise-driven rule induction.** Когда `simulate_forward` предсказывает health=9 а
  наблюдается health=3, обсервация попадает в `tracker.observed_rates`, но **нового правила
  не появляется**. Агент не может узнать, что zombie-at-distance-2 даёт урон — только что
  middle rate health'а падает. Это Stage 77b scope.
- **No conditional rates.** `observed_rates` глобальная per-variable; не кондиционирована
  на visible_concepts/inventory. `food rate while zombie visible` не может дивергировать от
  `food rate in open field`.
- **No when-clause conjunction grammar.** `passive_stateful` поддерживает только один
  предикат (`food > 0`), не AND/OR цепочки. Stage 77b.
- **No enemy spawn modelling.** MPC видит только текущие позиции DynamicEntity — не может
  учесть "скелет появится через 5 тиков с вероятностью 0.3". Нет passive spawn-rate rule.
- **Lava/water hazards не моделируются.** Пропущено после Run 6, когда выяснилось что real
  cause early deaths — skeleton arrows range=5, а не environment hazards.
- **`ConceptStore.save/load` stubbed to NotImplementedError.** RuleEffect нужен dedicated
  JSON serializer; реинтродуцировать когда понадобится. Сейчас textbook всегда грузится из
  YAML, experience живёт в tracker/spatial_map, persistence не критична.
- **Confidence threshold 0.1** — rules ниже не fire в simulate_forward. Магическое число,
  TODO(77b): probabilistic weighted firing через multi-rollout.
- **Все stage71/72/73 тесты удалены** — покрывали Stage 72-74 dead perception code
  (ground_/retrain_/select_goal). Stage 75 tests остались (визуальный encoder) но
  TestStepToward преexisting fail (axis convention change до моих работ).
- **Preexisting failures в full suite:** test_encoder/test_replay/test_stage15/47/66 —
  6 failures, все не трогают ConceptStore/perception/mpc. Unrelated к Stage 77a.
- **Next:** Stage 77b — runtime rule induction from surprise. Когда предсказание `sim`
  расходится с наблюдением > threshold, emit candidate rule c confidence=0.3, verify при
  следующей похожей ситуации. Плюс when-clause grammar и conditional rate learning.
