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

Exp145 mixed source-only run не прошёл prerequisite этого gate. Fixed corpus был
идентичен: **2048 episodes / 130676 transitions**; fit terminal episodes
east/west/south/north — **2/4/2/5 = 13**, **104 terminal examples**, all-future local examples
— **741288**, batches balanced 50:50. Real loss был **1.64197→1.04862**,
shuffled **1.67165→1.57132**, runtime — около **30 минут**. Source-geometry,
shuffled и frozen-random arms
дали **0/24**; physics transfer gate — `null` (не запускался). Real traces
выбирали правильный первый turn во всех 24 случаях, затем повторяли forward до
box (`turn=6`, `forward=186` на layout). Значит, mixed local+terminal hindsight
не решил source prerequisite и локализует reactive-composition failure /
label-objective mismatch; это не отсутствие turn recognition и не physics
transfer failure. Controller остаётся reactive и сбрасывает representation после
каждого observation.

Этот bounded direction проверен в exp146. Predictive dynamics loss снизился
**1.35259→0.58296**, а ordered temporal probe на held-out real pairs получил
balanced accuracy **0.7140** против **0.4914** shuffled. Но ordered H3, ordered
H1, shuffled H3 и raw H3 получили **0/24** каждый; Push-2 не запускался.
Ordered/raw H3 после правильного первого turn семь раз выбирали заблокированный
`forward`. Search выполнил полные `55` model calls, termination была
нейтрализована, поэтому лимит planner и terminal handling не объясняют failure.

Late-prefix exhaustive fork на `east_row4_left/seed=20000` отделил endpoint
score от rollout. Единственная успешная трёхшаговая последовательность
`interact,forward,interact` была rank **1/125** у actual ordered и actual raw,
но rank **54/125** и **42/125** на predicted endpoints. Лучшие predicted планы
были неуспешны; exhaustive ranking тоже неверен, поэтому расширение beam не
лечит основной дефект. На этом fork подтверждён learned rollout error, но не
общая неспособность representation.

Следующий bounded diagnostic использует сохранённый checkpoint без retraining:
teacher-forced one-step против autoregressive rollout и persistence baseline на
трёх canonical переходах. Exp147 закрыл его: contact `interact` one-step MSE
хуже persistence в **63.7x** и **130.2x**, а заблокированный `forward` имеет
prediction MSE **0.43084** при нулевой persistence error и rank **5/5**.
Свободный `forward` лучше persistence (`0.13401 vs 0.42547`). Autoregressive
ошибка растёт `0.14325→0.15883→0.23801`, но основной failure уже one-step.

Следующий bounded split по сохранённому checkpoint — те же one-step contact/no-op
метрики на source против unseen layouts. Exp148 получил contact и blocked-noop
failure на **4/4 source** и **4/4 unseen** layouts. Median interact ratio к
persistence — **49.14x/72.96x**, blocked-forward MSE — **0.3321/0.3501**;
свободный forward, напротив, лучше persistence (`0.2256x/0.2673x`). Значит,
layout generalization не является основной причиной: one-step dynamics уже
неверна на source geometries.

Следующий bounded diagnostic — измерить coverage исходного fixed replay по
state-changing `interact` и blocked no-op transitions. Exp149 воспроизвёл
**130676** transitions: `interact` меняет RGB в **1925/26125 = 7.37%** случаев
и встречается хотя бы раз в **1281/2048** episodes; одновременно есть **24200**
no-change interact, **9358** no-change forward и **26082** noop. Опыт contact
разрежен по transitions, но не отсутствует, а identity transitions обильны.

Exp150 выполнил matched residual/persistence comparison на том же replay
(**2048 episodes / 130676 transitions**, `exact_protocol=true`). Residual
dynamics снизила loss **1.31142→0.49581** и сохранила ordered probe signal
(balanced accuracy **0.7138** против **0.4671** shuffled). Она улучшила median
free-forward ratio на source с **0.2256** до **0.1032** и late-fork predicted
rank с **54/125** до **18/125**, но contact и blocked-noop failures остались
**4/4**; residual source medians — interact **34.06x**, blocked MSE **0.3650**.
Все четыре MPC arms получили **0/24**, one-step/source-compositional/composition
gates не прошли, physics gate остался `null`; Push-2 не запускался.

Exp151 завершил event-balanced causal control на том же fixed replay
(**2048 episodes / 130676 transitions**, `exact_protocol=true`). Sampling pool
содержал **1894** event и **124686** ordinary transitions; оба anchor budgets
заполнены **8000/8000**, поэтому среди всех supervised targets было
**8545/48000 = 17.80%** event. Dynamics loss снизился
**1.31199→0.42986**, probe сохранил ordered signal (**0.7130** balanced
accuracy против **0.4873** shuffled), а source contact/blocked failures остались
**4/4 и 4/4**. Относительно frozen exp150 baseline event-balanced arm снизил
source median interact ratio **34.06x→26.22x** и blocked MSE
**0.3650→0.1516**, но ухудшил free-forward ratio **0.1032→0.2873**;
на unseen failures также остались **4/4 и 4/4**. Canonical late-fork ordered
rank ухудшился до **95/125** (raw **104/125**), все четыре MPC arms дали
**0/24**, а one-step/source-compositional/composition gates не прошли.

Значит, sparse frequency — частичный фактор contact/blocked error, но ни она,
ни residual parameterization не объясняют и не снимают корневой rollout
failure.

Exp152 закрыл следующий bounded diagnostic на exact corpus **130676**
transitions и checkpoint `afdf53e`. Episode-disjoint linear probes по
frozen current `z + action` разделили both preregistered signals:
`interact` change/no-change дал ordered balanced accuracy **0.88372185**
против **0.56801987** shuffled, а `forward` blocked/moving —
**0.94262883** против **0.31591830**. Для обоих tasks balanced
accuracy, per-class recall и ordered-minus-shuffled margin прошли пороги
**0.8 / 0.7 / 0.2**; outcome — `representation_signal_evidence`.

Это снимает encoder availability как текущий root bottleneck, но не
доказывает AGI, JEPA или transfer. Следующий минимальный experiment-only
gate — transition-state conditioning/gating на current `z + action` при uniform
replay, без event labels и без изменения objective или planner.

Exp153 проверил этот matched arm на коде `49877e4` и uniform replay
(`exact_protocol=true`). Multiplicative identity gate снизил dynamics loss
**1.31142→0.46271** и подавил contact hallucination: source/unseen contact
failures стали **0/4 и 0/4**, median interact ratios — **0.9794/0.9851**.
Однако blocked-noop failures остались **4/4 и 4/4** с MSE **0.2620/0.2400**,
а free-forward ratio ухудшился относительно exp150 baseline с **0.1032** до
**0.2021/0.1775**. Contact criterion прошёл только потому, что near-persistence
слегка лучше persistence на changed transitions; это не evidence надёжного
effect modeling.

Gate в основном выучил action prior, а не устойчивое within-action state
discrimination: actions 0/1 получили примерно **0.998–0.999**, actions 3/4 —
малые значения, а forward contexts — примерно **0.690/0.758/0.935** по трём
canonical steps. Late fork дал ordered/raw rank **24/22** и endpoint MSE
**0.1508**, predicted winners остались неуспешны; все eval arms снова получили
**0/24**. One-step/source-compositional/composition gates — `false`, physics —
`null`; Push-2 не запускался. Значит, multiplicative identity bias переносимо
подавляет interact hallucination, но не решает blocked-forward или composition.
Следующий минимальный arm сохраняет архитектуру и uniform replay и меняет только
objective: self-supervised RGB change/no-change auxiliary для gate с
class balancing внутри action, без task-success labels, planner changes и Push-2.

Exp154 проверил этот objective-only arm на коде `9f896a3`, сохранив
architecture exp153, uniform replay и planner (`exact_protocol=true`).
Predictive loss снизился **1.31142→0.47395**, auxiliary loss —
**0.62084→0.50037**; ordered probe сохранил signal (**0.7129** balanced
accuracy против **0.5034** shuffled). Но auxiliary вернул contact failure:
source/unseen получили **4/4 и 4/4**, а blocked-noop также остался
**4/4 и 4/4**. Source medians составили interact **20.90x**, blocked MSE
**0.0932**, free-forward ratio **0.7502**; unseen — **21.80x**, **0.2396** и
**1.7898**. Ни один preregistered gate не прошёл.

Canonical gate margins остались малы или получили неверный знак: source
forward **0.0065/0.0143**, interact **−0.0560**; unseen forward
**0.0107/0.0077**, interact **−0.0328**, при пороге **0.15**. Late fork
ухудшился до ordered/raw ranks **91/88** и endpoint MSE **0.5536**;
predicted winner неуспешен. Все четыре MPC arms снова дали **0/24**,
physics gate — `null`, Push-2 не запускался. Значит, raw visual-change
auxiliary не выучил условную blocked/contact transition и повредил interaction
dynamics. Следующий минимальный вопрос — learned state-transition/event
representation либо factorized object-centric delta target; дальнейшая
настройка этого gate objective данными не обоснована.

Exp155 выполнил дешёвую checkpoint-only проверку более узкого вопроса: может ли
вообще оптимальная scalar amplitude спасти frozen residual delta exp150.
На source/unseen persistence имеет contact failures **4/4**, blocked failures
**0/4**, interact ratio **1**, blocked MSE **0** и free-forward ratio **1**.
Ungated exp150 воспроизвёл source **4/4 и 4/4** с medians **34.0592**,
**0.3650**, **0.1032**, а unseen — **4/4 и 4/4** с **54.6281**, **0.4169**,
**0.0944**. Shared и per-member scalar oracles сняли blocked failures до
**0/4**, не ухудшив free-forward ratio exp150, но contact остался persistence-
level: source **4/4**, unseen **3/4**, interact ratio **1**. Gate — `false`.

Следовательно, даже hindsight-optimal scalar amplitudes не исправляют
направления frozen exp150 residual deltas; oracle лишь возвращает
persistence-level contact result. Это не отвергает другие или jointly learned
delta directions и не разрешает action-specific scalar training. Следующий
минимальный шаг —
checkpoint-only raw-delta oracle audit exp153/154 до нового обучения.

Exp156 выполнил этот audit по точным checkpoint heads exp153 `49877e4` и
exp154 `9f896a3`. У exp153 native gate source/unseen contact failures были
**0/4 и 0/4**, blocked **4/4 и 4/4**; raw per-member oracle получил на source
**0/4 и 0/4** с medians free **0.1722**, interact **0.8997**, blocked MSE
**0**, а на unseen — **0/4 и 0/4**, **0.1532**, **0.9312**, **0**. Gate
прошёл. У exp154 native failures были **4/4 и 4/4** на обоих splits; raw
oracle исправил source до **0/4 и 0/4** с medians **0.6666**, **0.9288**,
**0**, но unseen остался contact **1/4**, blocked **0/4** с **0.8281**,
**0.7991**, **0**. Source gate прошёл, unseen proof — нет.

Значит, raw delta directions обеих моделей выразительны на source, а wall
находится в learnability/objective либо expressivity текущего gate. Exp153
directions дополнительно проходят unseen и являются лучшей основой. Следующий
минимальный training arm: frozen exp153 encoder/recurrent/raw deltas и
action-specific gates, обучаемые только latent predictive objective — без RGB/
task labels и без изменения planner.

Exp157 выполнил этот arm: **1,403,398** параметров backbone остались frozen и
неизменными, обучались только **3,855** параметров action-specific gates по
uniform latent predictive MSE. Loss снизился **0.23227→0.10500**, frozen
ordered probe сохранил balanced accuracy **0.7136** против **0.4719** shuffled.
Но candidate ухудшил source с baseline contact/blocked **0/4 и 4/4** до
**4/4 и 4/4** (interact ratio **1.4361**, blocked MSE **0.1627**, free ratio
**0.2495**); unseen также стал **4/4 и 4/4** с **2.3803**, **0.1503** и
**0.2128**. Gate не выучил нужную границу: action 2 дал **0.5564/0.6379/
0.8679** на blocked/changed/blocked contexts, action 3 — почти одинаковые
**0.0927/0.1540** changed против **0.0947** no-change.

Все MPC arms остались **0/24**; canonical ordered/raw ranks **24/26**,
endpoint MSE **0.1918**, winner неуспешен. Gates — `false`, physics — `null`.
Значит, action-specific boundary при uniform latent MSE недостаточна, хотя
exp156 доказал expressivity frozen raw directions. Следующий minimal arm
сохраняет тот же frozen backbone/gates, но балансирует latent prediction error
внутри `(action, RGB-change/no-change)`; RGB используется только как вес
наблюдённого transition, без BCE, новой architecture, task labels или planner
changes.

Exp158 проверил этот единственный weighting change при тех же **1,403,398**
frozen и **3,855** trainable gate parameters. Exact class weights по actions
0–4: `[[0,1],[0,1],[1.395704,.779110],[.539773,6.785714],[1,0]]`; RGB
использовался только для веса observed transition, target оставался latent.
Sampled loss изменился **0.22120→0.26392**, но эта шумная пара не считается
improvement. Candidate снова дал source/unseen contact/blocked failures
**4/4 и 4/4**: source medians free/interact/blocked **0.2600/1.9063/0.1614**,
unseen **0.2147/2.8078/0.1493**.

Граница gate осталась неверной: action 2 source — **0.5326** blocked,
**0.6164** changed, **0.8678** blocked; action 3 — **0.1028/0.1928** changed
против **0.1051** no-change. Все MPC arms получили **0/24**, late-fork
ordered/raw ranks **24/26**, endpoint MSE **0.1932**, winner `[1,4,4]`
неуспешен; gates — `false`, physics — `null`. Fixed action/change class weights
не решают amplitude learning под latent MSE; coefficient sweep не оправдан.
Следующий дешёвый шаг — checkpoint audit independent-member analytic amplitude
target до любого нового долгого training run.

Exp159 выполнил этот audit. Native exp153 на source/unseen имел contact/blocked
failures **0/4 и 4/4**; independent-member oracle получил **0/4 и 0/4** на
обоих splits. Source medians free/interact/blocked стали
**0.1722/0.8997/0**, unseen — **0.1532/0.9312/0**; joint control дал те же
агрегаты. Independent gate прошёл. При этом **360** analytic amplitudes не
сводятся к бинарной маске: min/median/max **0/0.046656/1**, counts
**144 zero / 72 one / 144 interior**.

Outcome `target licensed` означает только, что self-supervised independent
amplitude target совместим с требуемой one-step геометрией frozen raw deltas.
Следующий exp160 сохраняет frozen exp153/action-specific architecture и fixed
action/change weights, но прямо регрессирует эти analytic targets; raw deltas и
planner не меняются. Это не AGI/transfer evidence.

Exp160 напрямую обучил licensed analytic targets, оставив **1,403,398**
backbone parameters frozen/неизменными и **3,855** trainable gate parameters.
Amplitude loss снизился **0.15659→0.05136** на **144000** sampled targets
(**47724 zero / 16458 one / 79818 interior**, mean **0.506558**), но one-step
поведение не исправилось. Candidate source/unseen получил contact/blocked
failures **4/4 и 4/4**: source medians free/interact/blocked
**0.2218/1.6447/0.2021**, unseen **0.1842/2.2435/0.1894**.

Gate снова не разделил contexts: action 2 source **0.6050** blocked,
**0.6909** moving, **0.8353** blocked; action 3 **0.1149/0.1406** changed
против **0.1237** no-change. Все MPC arms — **0/24**; late-fork ordered/raw
ranks **23/24**, endpoint MSE **0.1810**, winner `[1,4,4]` неуспешен. Gates —
`false`, physics — `null`. Следующий шаг — дешёвый episode-disjoint,
teacher-forced probe analytic amplitude: `z`-linear против `z+hidden`-linear;
`z`-MLP только при необходимости минимального tie-break. До этого новый
architecture training не обоснован; AGI claim отсутствует.

Exp161 выполнил episode-disjoint teacher-forced comparison на **1536/512**
episodes с overlap **0**. Held-out amplitude MSE снизился от `z`-linear
**0.03260** (weighted **0.02944**) до `z+hidden`-linear **0.00884** (weighted
**0.00883**), но behavioral gate не прошёл ни у одного arm. `z` получил
source/unseen contact/blocked failures **4/4 и 4/4**, free/interact ratios
**0.2181/3.1853** и **0.1838/5.7260**. `z+hidden` также дал **4/4 и 4/4**:
source free/interact/blocked **0.2652/1.5196/0.0885**, unseen
**0.1929/1.0005/0.1171**.

Outcome — `both_linear_inputs_fail`: recurrent hidden state существенно
улучшает amplitude regression, но linear probes всё ещё проваливают критическое
one-step behavior. Следующий минимальный probe проверяет нелинейность либо
object-centric transition target; более долгое переобучение текущего gate не
обосновано. Это не AGI/transfer evidence.

Exp162 проверил минимальную нелинейность: per-action `z+hidden` MLP с одним
128-wide ReLU layer и **400** updates на том же episode-disjoint split.
Held-out MSE стала **0.00436**, weighted **0.00466**, против exp161 linear
**0.00884/0.00883** (примерно **2.03x** лучше по plain MSE). One-step contact
также улучшился: source contact failures **1/4** с interact ratio **0.9110**,
unseen **0/4** с **0.9599**, против linear **4/4** на обоих splits. Но
blocked-noop остался **4/4** source/unseen, несмотря на MSE **0.0578/0.0120**;
free ratios — **0.2626/0.3040**. Gate — `false`.

Нелинейные interactions materially помогают movement/contact prediction, но не
дают требуемую no-op physics. Следующий минимальный diagnostic проверяет
object-centric state/target; более долгое обучение этого MLP не обосновано.
Composition, transfer и AGI не доказаны.

Exp163 проверил, можно ли исправить zero atom простой per-action calibration
frozen exp162 MLP. Пороги равны
`[0.672216,0.569839,0.251421,0.037978,0.027148]`; source leakage отсутствует.
Held-out latent MSE изменилась лишь **0.046140→0.046053** (около **0.19%**).
При этом native source/unseen contact/blocked failures **1/4,4/4** и
**0/4,4/4** после calibration стали **3/4,2/4** и **4/4,1/4**; interact ratio
стал **1.0** на обоих splits, free ratios остались **0.2626/0.3040**.

Для action 3 threshold подавляет **99.82%** zero targets, но сохраняет только
**6.99%** positive: contact effect уничтожен. Gate — `false`. Это score-overlap/
ranking-state failure, а не недостаточная длина calibration. Следующий
минимальный шаг — evaluator-only relational object-state diagnostic; новое
обучение до него не обосновано. AGI/composition claim отсутствует.

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
