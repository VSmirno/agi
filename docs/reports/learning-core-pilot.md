# Transferable learning core: первый development pilot

Дата: 2026-09-05. Ветка: `feature/stage9x-sparse-option-failures`.
Исходный код запуска: `a096de025b74323b668df4f920d91611cb00b6a4`.
Это измерение прототипа по [плану](../superpowers/plans/2026-09-05-transferable-learning-core.md),
не завершение всей подтверждающей кампании.

## Результат пилота

Полезность обучения **не показана**. Технически цикл реальный опыт → replay →
обучение → планирование → checkpoint работает, но это не доказательство
улучшения решений или достижения AGI.

- Сбор: 8 эпизодов, 248 действий + 8 resets, 100 gradient updates.
- End-to-end validation: **0/4** успешных эпизодов, 15 872 planner-candidate calls
  (не весь compute: обновление belief по реальному наблюдению сюда не входит).
- Fixed-E initial / real-actions / shuffled-actions: **0/4** у каждого.
- Случайная обучающая политика получила ресурс в 8/8 эпизодах. Это другой
  набор seeds, поэтому не парная оценка превосходства random над planner.
- Общий held-out corpus: 124 перехода, 14 переходов с изменениями сенсоров;
  сенсор wood менялся в 3 переходах. Corpus не добавлялся в replay.
- Хеши замороженного encoder совпадают до/после во всех контрольных условиях.
- Время пилота по manifest: 19.25 s на RTX 3090; agent/infrastructure failures: 0.

Ошибка сенсоров (MSE; меньше лучше), одинаковый held-out corpus и encoder:

| Horizon | Initial dynamics | Real actions | Shuffled actions | Persistence |
|---|---:|---:|---:|---:|
| 1 | 19.34936 | 3.56337 | 3.56328 | 0.02419 |
| 3 | 19.24274 | 3.52448 | 3.52434 | 0.07069 |
| 5 | 19.14253 | 3.48300 | 3.48287 | 0.11667 |
| 10 | 18.80448 | 3.34675 | 3.34662 | 0.23864 |

`Initial` здесь — dynamics после общего source pretraining, до дополнительных
fixed-E updates, а не случайно инициализированная модель. Уменьшение ошибки
относительно этого состояния есть, но простое сохранение последнего сенсорного
значения существенно лучше. Перемешивание действий не ухудшило прогноз.
Следовательно, этот запуск не подтверждает полезное action-conditioned знание.
Падение train loss и latent MSE отдельно не меняет этот вывод.

## Первый transfer probe и итоговый обзор

`transfer-001` на исходном коде завершился с exit 0 за 11.46 s: 4 общих
adaptation episodes, 2 eval episodes, лимит 16 steps, 20 updates на ветку.
Door: 64 B steps с resets, все три ветки 0 → 0 success.
Push: 33 B steps; fresh и weights 1 → 0, weights+replay 1 → 1.
Все ветки стартовали с push success 2/2 **до** адаптации: это не обученная
способность. A success везде был 0 до/после, поэтому retention неинформативен.

Единственный интеграционный обзор нашёл три дефекта:

1. `burn_in=1` исключает supervision у легитимных одношаговых успехов push_2;
   all-short batch может остановить обучение. Поэтому B-результаты этого запуска
   не интерпретируются как проверка корректной адаптации.
2. Present corrupt inventory ошибочно превращается в missing sensor.
3. FRESH ошибочно приписана source training cost, хотя веса/опыт не наследуются.

Первый запуск сохранён без изменений. Исправления: commit `8e51eae`.
Transfer использует явно записанный `effective_config.burn_in=0` во всех ветках;
source checkpoint и настройки source pilot не менялись. Это сокращает доступный
контекст внутри обучающего replay window: recurrent-механизм остаётся, но
transfer probe не проверяет обучение с историей до начала этого окна.
Повреждённые present sensors теперь вызывают ошибку, отсутствующие остаются
masked. FRESH source cost равен нулю. Эти дефекты не объясняют отрицательный
source pilot; полезность source learning не показана.

### Повтор после исправления: transfer-002

Exit 0, статус `completed`, время 10.90 s, `effective_config.burn_in=0`;
те же seeds и бюджеты.

| B family | Fresh, до → после | Weights, до → после | Weights + replay, до → после |
|---|---:|---:|---:|
| door_key/key_consumed | 0/2 → 0/2 | 0/2 → 0/2 | 0/2 → 0/2 |
| push_box/push_2 | 2/2 → 2/2 | 2/2 → 2/2 | 2/2 → 2/2 |

Все ветки получили 20 updates; mixed replay фактически использовал 6 A и 14 B
updates против 20 B у остальных. B corpus общий для сравниваемых веток: 64 steps
на door и 33 на push, включая resets. FRESH source training cost теперь 0;
наследующие ветки учитывают 256 source steps и 100 source updates.

Вывод: после исправления исчезло ухудшение push в fresh/weights; отдельного
выигрыша от переноса **нет**. Push имеет потолок уже на checkpoint 0, door —
нулевой результат. A success всё ещё 0 до/после: retention не проверен полезным
исходным навыком. Это диагноз неинформативности текущего мини-probe, не опровержение
всего подхода world models/JEPA и не подтверждение AGI.

Команда повтора (то же окружение ниже):

```bash
timeout 240s /opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python \
  experiments/exp138_learning_core.py --stage transfer \
  --config configs/core_pilot.yaml \
  --checkpoint output_to_user/core/pilot-001/source_checkpoint.pt \
  --out output_to_user/core/transfer-002 \
  --episodes 4 --eval-episodes 2 --steps 16 --updates 20 --max-seconds 180
```

## Воспроизведение и происхождение

Разработка локально; runtime только на `cuda@192.168.98.56` (HYPERPC).
Рабочий снимок: `/opt/cuda/agi-core-20260905-hO9zUs`.
Python: `/opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python`.
PyTorch: `2.5.1+cu121`. Использовалась GPU 0; чужие процессы не останавливались.

Из рабочего снимка, с указанным Python:

```bash
export PYTHONPATH=/opt/cuda/agi-core-20260905-hO9zUs/src:/opt/cuda/agi-core-20260905-hO9zUs
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
timeout 240s /opt/cuda/miniforge3/envs/agi-stage90r-py311/bin/python \
  experiments/exp138_learning_core.py --stage pilot \
  --config configs/core_pilot.yaml --out output_to_user/core/pilot-001 \
  --family resource_acquisition --ruleset default \
  --episodes 8 --eval-episodes 4 --steps 32 --updates 100 --max-seconds 180
```

Для повторения нужен **новый** `--out`. Существующие артефакты не перезаписывать.
Локальная копия: `output_to_user/core/pilot-001/` (results, manifest, traces,
checkpoint и replay). В удалённом git archive нет `.git`, поэтому manifest честно
пишет `git_commit: unknown`. Все 21 source/config SHA256 из manifest сверены
с чистыми tracked-файлами указанного локального commit: совпадают.
Checkpoint SHA256: `a70aab8d238a395238e54def9fe8692aad0caaec63ca34bd7821875f293d4c62`.

Предыдущие проверки на HyperPC: 30 новых focused tests и 32 существующих
env/Crafter regression tests прошли; GPU smoke завершился. Полный репозиторный
suite не запускался. Повтор объединённых suites перед исправлениями:
**62 passed in 4.03 s**. Точные focused suites:

```bash
python -m pytest tests/test_core_env.py tests/test_core_model.py \
  tests/test_core_runtime.py tests/test_core_controls.py \
  tests/test_core_experience.py -q --tb=short
python -m pytest tests/test_env.py tests/test_crafter_pixel_env_67.py -q --tb=short
```

После исправлений эти же suites плюс `tests/test_core_transfer_probe.py`:
**71 passed in 4.04 s** на HyperPC (39 core, 32 существующих regression).
Минимальные дополнительные проверки: present corrupt inventory; фактический
gradient update из одношагового эпизода; стоимость FRESH/WEIGHTS/WEIGHTS_REPLAY.
До переноса исправления пять corruption cases воспроизвели ошибку на старом коде.

## Ограничения и принятые сокращения

- CNN/GRU с нуля, маленькая модель, короткий search, ручные сенсорные/image goals.
- Crafter использует контролируемый native fixture с источниками рядом со spawn,
  не естественное распределение мира. B использует фиксированную геометрию:
  разные seeds не означают held-out maps.
- Один training seed, нет confidence intervals или подтверждающих G0–G6 claims.
- Полная preregistration, SOURCE_CONTROL и причинные interventions отложены.
- Один общий обзор вместо циклов обзора каждой задачи; исправляются только
  существенные ошибки. Работа в текущей ветке с разделённым владением файлами,
  интеграция и удалённые снимки сериализуются координатором.

## Follow-up diagnosis: влияние сенсорного входа

Исходный checkpoint `pilot-001`, восемь первых реальных наблюдений из replay,
все 17 действий при одном и том же наблюдении. Это диагностическое вмешательство
во вход модели, **не** новая среда, training data или оценка способности:

| Вход | Средний диапазон sensor prediction по действиям | Среднее abs(hidden) | Доля abs(hidden)>0.99 |
|---|---:|---:|---:|
| Без вмешательства | 0.06677 | 0.94400 | 0.15786 |
| Только z=0 | 0.07492 | 0.93669 | 0.16728 |
| Только sensors=0 (mask сохранена) | 0.62898 | 0.31411 | 0 |

На первом наблюдении latent RMS=0.05937, sensors `[9,9,9,9,0]`, прогноз после
action 5 `[3.782,3.741,3.430,3.772,0.575]`. Раздувания latent здесь нет.
Обнуление сенсоров меняет физический смысл входа и не является исправлением;
оно локализует подавление различий действий сенсорным вкладом в GRU.
Причина плохого обучения **в целом** этим ещё не установлена.

Независимый аудит существующих traces: 248 source transitions, action 5 встречается
17 раз, wood меняется 10 раз (9 при action 5, 1 при action 8). Первый переход
эпизода, исключённый burn-in, содержит 1 из этих 10 событий. Значит, обучающий
сигнал редкий, но не полностью отсутствует. В source validation action 5 не
выбран ни разу: 116 действий 13 и 8 действий 15. Разброс root-cost на первом
шаге падает с ~0.0375 до ~0.00312 после fixed-E updates; это не точные ties.

Следующая узкая гипотеза в рамках утверждённого profile selection: нормировать
**только проекцию сенсоров** перед сложением с action embedding. Baseline остаётся
без нормализации, новый профиль получает отдельное имя. Encoder, SIGReg, цели,
loss, seeds, сбор данных и число updates остаются прежними. Это проверка одного
механизма, не подбор evaluator под положительный результат. Критерий полезности:
real-actions должен отделиться от shuffled и persistence, затем улучшить решения.

`exp139_core_diagnosis.py` подтвердил для source checkpoint: вклад condition в
GRU preactivation RMS=3.658 против z=0.137; abs(candidate)=0.9971, gradient L2
action embedding=0.00702 против sensor heads=13.578. Для fresh action gradient
0.1977. Результаты: `output_to_user/core/diagnosis-001/results.json`.

### Condition normalization: отрицательный результат

Отдельный `core_condition_norm.yaml` включает неаффинную LayerNorm только после
sensor projection; baseline default=False, encoder/SIGReg не менялись.
`pilot-condition-norm-001` завершился: source и все controls **0/4**; H1 sensor
MSE real=3.53605, shuffled=3.53657, persistence=0.02419. H10 real=3.30391,
shuffled=3.30386. Полезного разделения действий и улучшения решений нет.

Диагностика нового checkpoint показывает снижение abs(hidden) с 0.9418 до
0.8211, доля abs(hidden)>0.99 стала 0, action gradient вырос до 0.01986.
Но абсолютные сенсоры всё ещё предсказываются примерно как 3.1–3.4 вместо 9.
Улучшение промежуточной активационной метрики не равнозначно полезному обучению.

Оговорка повторного сбора: actions, sensors и остальные replay arrays совпали,
но у одного эпизода различались 2295 before-RGB и 2358 after-RGB элементов.
Поэтому малую разницу MSE между профилями нельзя приписывать только нормализации.
Внутри каждого fixed-E сравнения corpus общий. Следующий probe использует один
**сохранённый** replay для всех вариантов, без повторного сбора training pixels.

Следующая гипотеза: параметризовать sensor prediction как текущие сенсоры плюс
обучаемое изменение. Только отдельный `predict_sensor_delta` профиль; baseline
и normalization default=False сохраняются. Сравнение absolute-real / delta-real /
delta-shuffled начинает веса с нуля одинаково и получает одинаковые replay batches.
Это не изменение целей, sensor labels или environment-specific policy.

### Парный residual probe: первый ограниченный положительный сигнал

`exp140_core_sensor_delta.py` обучил все варианты с одинаковых начальных весов
на буквально одинаковых 100 replay batches из `pilot-001`; training corpus hash
`4ec7a9df...e1d90b54`. Общий held-out corpus собран один раз и не попал в replay.

| Вариант | Updates | Success | H1 sensor MSE | H10 sensor MSE | First wood | Max wood |
|---|---:|---:|---:|---:|---:|---:|
| absolute-real | 100 | 0/4 | 19.38487 | 18.83367 | — | 0 |
| delta-untrained | 0 | 0/4 | 0.11989 | 11.33727 | — | 0 |
| delta-real-zero-action | 100 inherited | 0/4 | 0.02579 | 0.46302 | — | 0 |
| delta-shuffled | 100 | 4/4 | 0.02788 | 0.68709 | step 5 | 1 |
| delta-real | 100 | 4/4 | 0.02457 | 0.33583 | **step 2** | **2** |

Persistence MSE: H1=0.02419, H3=0.07069, H5=0.11667, H10=0.23864 —
то есть delta-real пока хуже persistence на каждом горизонте. На 15 изменившихся
sensor entries холодный H1 MSE: real=0.96989, shuffled=0.97764; разница мала.

Тем не менее action-зависимость ограниченно подтверждена. На трёх held-out
переходах с ростом wood delta-real ранжирует action 5 первым; shuffled — пятым,
untrained — седьмым, absolute — десятым. Обнуление обученного action embedding
роняет success 4/4 → 0/4. Real выполняет action 5 дважды на эпизод и получает
wood=2; shuffled выполняет его пять раз, позже, и получает wood=1.

Честный вывод: residual-параметризация устранила грубую ошибку абсолютного
baseline и позволила обучению создать причинно релевантное различие действий в
одном controlled fixture. Но shuffled также проходит слишком слабый бинарный
gate, общий prediction всё ещё хуже persistence, события всего три, один seed и
карта/fixture фиксированы. Это **development evidence**, не stage PASS, transfer,
generalization, JEPA validation или AGI.

Артефакты: `output_to_user/core/paired-sensor-delta-002/`.

### Штатный residual pilot и первый retention

После commit `b0d0928` штатный `exp138 --stage pilot` с отдельным
`core_sensor_delta.yaml` завершился за 19.40 s: source success **4/4**.
Fixed-E real/shuffled/initial также 4/4, поэтому бинарный success остаётся слабым.

| Horizon | Initial | Real actions | Shuffled actions | Persistence |
|---|---:|---:|---:|---:|
| 1 | 0.02422 | **0.02298** | 0.02415 | 0.02419 |
| 3 | 0.07204 | **0.06614** | 0.07325 | 0.07069 |
| 5 | 0.12045 | **0.11155** | 0.13018 | 0.11667 |
| 10 | 0.27065 | **0.24467** | 0.31390 | 0.23864 |

Real-actions лучше shuffled на всех горизонтах и лучше persistence на H1/H3/H5;
на H10 persistence ещё лучше. После дополнительных fixed-E updates real получает
первый wood на step2, shuffled — step3. Source checkpoint:
`output_to_user/core/pilot-sensor-delta-001/`. Все 21 source/config hashes из
manifest совпадают с tracked files `b0d0928`.

Transfer этого checkpoint (`transfer-sensor-delta-001`, 20 updates/arm) впервые
имеет ненулевой A baseline. WEIGHTS и WEIGHTS_REPLAY сохраняют A success 2/2
после обеих B-задач; FRESH имеет A=0. Это механическая retention learned weights,
но не преимущество replay: WEIGHTS сохраняет A так же. На B transfer benefit не
измерился: door у всех 0/2 до/после, push у всех 2/2 до/после. Следовательно,
текущий B catalog даёт floor/ceiling и непригоден для вывода о переносе.

### Push-1: локализация цели, planning и replay

`exp141_push1_transfer.py` убрал ceiling, но показал новый ложный baseline:
случайно инициализированный FRESH дважды выбрал точную последовательность
`[interact, forward, interact]`, после 20 updates все ветки стали 0/2. В 64
переходах shared B replay были 12 `interact`, три первых толчка и ни одного
перехода в goal. Это regression после adaptation, а не отрицательный transfer.

`exp142_grid_action_confusion.py` отделил три причины на реальных forked outcomes.
Encoder geometry не схлопывалась, но action matching в необходимых состояниях
не выдерживал обучения на sparse random corpus. Oracle-capacity control с теми
же 4×16 steps и 20 updates, но с полными evaluator-known траекториями, дал rank1
на всех трёх решениях у FRESH/WEIGHTS/WEIGHTS_REPLAY. Это доказывает локальную
ёмкость модели, но oracle data не является результатом агента.

Natural random64 corpus содержал 972 steps, 38 движений box и 6 успехов. После
1000 uniform updates action matching стал существенно лучше mismatched, но raw
latent goal geometry считала реальный `turn_right` ближе к goal, чем первый push.
Два generic planner дефекта были подтверждены отдельными RED тестами:

- rollout продолжался после predicted termination по необученной post-terminal
  динамике;
- промежуточные latent distances суммировались как path cost, хотя они не
  калиброваны как мера прогресса и штрафуют полезный detour.

Planner теперь держит predicted terminal как absorbing state и ранжирует
depth-local reached state. Natural uniform end-to-end улучшился с 0/4 до 2/4
только у FRESH; transfer arms остались 0/4. Terminal-only sampling переобучился
на конец и забыл начало. Balanced 50/50 uniform+terminal-window diagnostic при
1000 updates дал FRESH 2/4, WEIGHTS 0/4 и WEIGHTS_REPLAY 4/4.

Production `salient_fraction=0.5` выбирает половину окон вокруг доступного без
reward сигнала: `terminated` или наблюдаемого sensor change. Штатный transfer
на том же natural corpus дал WEIGHTS_REPLAY B=2/4 и A retention=4/4; FRESH и
WEIGHTS B=0/4 и потеряли случайный A baseline. Это первый совместный сигнал от
source replay и event windows, но не положительное обучение: B checkpoint0 у
всех был случайно 4/4, один seed и два цвета из четырёх не перенеслись.

Честный следующий барьер — goal-conditioned reachability/temporal distance.
JEPA latent prediction стала action-sensitive, но евклидова близость общего
embedding к goal image не обязана кодировать достижимость или прогресс.

`exp143_temporal_proximity.py` проверил этот барьер temporal-head без reward,
координат и правил Push. При этом frozen backbone уже обучался с termination
supervision (`Grid termination == success`) и terminal-priority sampling, поэтому
опыт не является label-free. Directed probe обучался по порядку кадров целых
real эпизодов и честно оценивает только
`P(target встретится <= H | dataset policy)`, а не оптимальную достижимость. На
малой frozen-модели он идеально ранжировал
реальные fork outcomes (MRR 1.000 против 0.553 у latent MSE и 0.400 у
shuffled-endpoint control), но на imagined root правильный первый push оставался
последним. Mixed real/imagined anchors это не исправили.

Три ограниченные альтернативы отвергнуты. RGB-change salience ухудшила F1/F2.
Zero-init residual correction улучшила held-out H1 MSE с 0.120 до 0.072 и H3 с
0.171 до 0.157, но не прошла ranking gate: root MRR 0.221→0.333 совпала с
shuffled-actions control. Action-contrastive loss также не прошёл причинный
контроль: shuffled-label arm ранжировал root лучше real-label arm. Увеличение
`z/h` с 64/32 до 256/128 не устранило one-step temporal root failure, поэтому
простая нехватка размерности не считается причиной.

При этом larger fresh model дала первый положительный closed-loop сигнал именно
от learned goal score: ordered temporal cost решил Push-1 в **4/6**, raw latent
MSE — **0/6**, shuffled endpoint — **0/6**; beam width 4 и 5 дали одинаковые
4/6. Успешные реальные traces были `[interact, interact, forward, interact]`
или `[noop, interact, forward, interact]`. Оба yellow seeds провалились
повторением `forward`, blue/purple прошли. Это development evidence цепочки
experience order → imagined scoring → changed action → real success, но не
transfer: topology/start фиксированы, seed в основном меняет один из трёх
цветов, а goal pose может выдавать ruleset. Следующий gate — episode-disjoint
train/test по новым layout/start/goal комбинациям с разными первыми действиями.

`exp144_layout_generalization.py` добавил evaluator-owned `PushLayout` и разделил
четыре train/four test layouts до построения пар. Во всех train layouts random
corpus содержал terminal experience. На пяти training runs по одинаковым 120
layout-disjoint evaluation cases ordered temporal cost дала **56/120**, raw MSE
**12/120**, shuffled endpoint **30/120**. Ordered была лучше raw во всех пяти
runs, но строгий per-run F3 gate прошёл лишь **2/5**; один run дал ordered=8 и
shuffled=8, другой ordered=12 и shuffled=16. Разбивка ordered по unseen layouts:
10/30, 10/30, 18/30, 18/30. Score subtraction
`ordered_logit - shuffled_logit` на худшем run дала только 2/24 и отвергнута.

Следовательно, temporal order содержит полезный goal signal и переносится между
частью пространственных конфигураций, но MPC неустойчив к некалиброванной
геометрии score. Это `PARTIAL`: raw image distance как default опровергнута,
но temporal probe ещё не принят в production.

Direct hindsight controller на всех future pairs снизил training loss
1.61→1.12 и правильно выбрал первый поворот во всех 24 unseen-layout cases,
после чего зациклился на `forward` и получил **0/24**. Terminal-only arm отобрал
118 последних `<=8`-step pairs из 17 успешных fit episodes. На трёх training
runs он дал **64/72** против action-shuffled **0/72**, goal-blind **44/72**,
ordered MPC **56/72** и raw MPC **28/72**; каждый run решил все четыре unseen
layouts. Однако заранее заданный superiority gate против всех controls прошёл
только **1/3**, поэтому замена MPC не принята.

Это положительный сигнал, что редкий успешный опыт может обучить прямое
visual/state-conditioned действие с layout generalization. Но arm явно
success-supervised через `Grid termination == success`, current RGB уже показывает
goal tile, а fit corpus содержит всего 17 успешных эпизодов.

Frozen random-encoder control на тех же 118 pairs дал **0/72**, тогда как
world-model encoder arm дал **60/72**; action-shuffled также получил **0/72**,
goal-blind **48/72**, ordered MPC **42/72**, raw MPC **24/72**. Representation
gate прошёл **3/3**, общий superiority gate — **2/3**. Это сильный development
signal, что обучение backbone создаёт полезное для прямого control представление,
но пока не отделяет predictive dynamics от success-equivalent termination loss и
terminal-priority sampling. Следующий causal control — обучить тот же backbone
на том же replay без termination objective и terminal salience, затем повторить
terminal-only policy learning.

Этот matched ablation дал decisive development result: predictive-only encoder
(`termination_weight=0`, `salient_fraction=0`) решил **72/72**, full encoder
**64/72**, frozen random encoder **0/72**, shuffled-action **0/72**, goal-blind
**44/72**, ordered MPC **44/72**, raw MPC **8/72**. Predictive representation и
controller gates прошли **3/3**; full encoder не превосходил predictive arm ни в
одном run. Следовательно, для этой задачи полезное представление возникает из
predictive training без success-equivalent supervision backbone. Сама policy
остаётся success-supervised: 118 примеров выбраны из terminal episodes.

`exp145_physics_transfer.py` поднял следующий gate: четыре long-distance Push-1
source layouts и четыре unseen layouts, проверяемые сначала под Push-1, затем
zero-shot под Push-2. Canonical target goal использовал Push-1 pose для обеих
физик; initial RGB совпадал, native goal RGB различался. Fixed corpus, включая
идентичный mixed source-only run, — **2048 episodes / 130676 transitions**;
fit terminal episodes: east **2**, west **4**, south **2**, north **5**
(всего **13**), то есть **104 terminal
examples**. Mixed replay добавил **741288 all-future local examples** и
использовал balanced 50:50 batches. Loss real снизился **1.64197→1.04862**,
shuffled — **1.67165→1.57132**. Runtime составил около **30 минут**.

Несмотря на это, source-geometry qualification получила **0/24**; shuffled
получил **0/24**, frozen-random — **0/24**, а physics transfer gate остался
`null` (не запускался). Во всех 24 real traces controller выбирал правильный
первый turn, затем повторял `forward` до box (на layout: `turn=6`,
`forward=186`). Текущий controller остаётся reactive и сбрасывает
representation после каждого observation.

Mixed local+terminal hindsight therefore did not solve the source prerequisite:
it isolates a reactive-composition failure / label-objective mismatch, not a lack
of turn recognition and not a physics-transfer failure.

Physics transfer поэтому не опровергнут: target нельзя оценивать, пока source
mechanism не переносится на новые long-distance layouts.

`exp146_temporal_mpc_physics.py` провёл следующий source-only comparison на
том же fixed corpus. Predictive dynamics loss снизился **1.35259→0.58296**;
ordered temporal probe на held-out real source pairs получил balanced accuracy
**0.7140** против **0.4914** у shuffled endpoint. Однако ordered H3, ordered
H1, shuffled H3 и raw H3 дали **0/24** каждый; Push-2 снова не запускался.
Ordered H3 и raw H3 во всех seeds выбрали правильный первый turn, затем семь
раз повторили заблокированный `forward`, не сдвинув agent или box. Search
использовал все ожидаемые `5+25+25=55` model calls; termination была
нейтрализована. Следовательно, это не action-ID mismatch, premature terminal
или исчерпание search budget.

Трассы показали imagined progress, исчезающий после re-observation. Например,
после правильного turn на west/north ordered cost для заблокированных
`[forward]`, `[forward,forward]`, `[forward,forward,forward]` менялся
**0.7854→0.1901→-0.2943**, хотя реальное состояние не менялось. Raw MSE в том
же состоянии также предпочитал фиктивный rollout. Полезный prefix
`[interact,forward]` занимал лишь 6--11 место на depth 2 и вылетал из beam=5,
но widening beam не является достаточным объяснением: exhaustive candidates,
включая правильные полные rollouts, уже получали худший predicted score.

Late-prefix exhaustive fork отделил dynamics от endpoint score. После реального
prefix `[turn_left, interact, forward, interact, forward]` на
`east_row4_left/seed=20000` все **125** трёхшаговых продолжений были оценены как
по реальным, так и по imagined endpoints. Единственная успешная последовательность
`[interact,forward,interact]` получила rank **1/125** у actual ordered score и
rank **1/125** у actual raw score, но только **54/125** и **42/125** соответственно
на learned rollout. Лучшие predicted ordered/raw последовательности были
неуспешны. Canonical ветка вылетела из beam на depth 2, но exhaustive predicted
ranking тоже был неверным. Это изолирует rollout error в одном deterministic
fork; оно не доказывает общую неспособность representation и не оправдывает
planner tuning.

Следующий минимальный вопрос — является ли ошибка уже one-step action-conditioned
prediction в реальном canonical state или возникает преимущественно при
autoregressive compounding. Его можно проверить без повторного 22-минутного
обучения по сохранённому checkpoint, сравнив teacher-forced one-step и
autoregressive ошибки с persistence baseline на трёх canonical переходах.

`exp147_rollout_localization.py` выполнил эту checkpoint-only проверку за
**0.62 s**. На canonical continuation `[interact,forward,interact]` teacher-forced
one-step MSE против persistence составили соответственно
**0.14325 vs 0.00225**, **0.13401 vs 0.42547** и
**0.15274 vs 0.00117**. То есть свободный `forward` предсказывается полезнее
persistence, но оба contact `interact` хуже неё в **63.7x** и **130.2x**.
Заблокированный `forward` в первом состоянии реально не меняет RGB и имеет
нулевую persistence error, однако learned prediction даёт MSE **0.43084** —
худший rank **5/5** среди действий этого fork.

Autoregressive canonical MSE растёт **0.14325→0.15883→0.23801** (`1.66x` от
H1 к H3), но это вторично: prerequisite «changed one-step predictions лучше
persistence» уже нарушен на обоих push-переходах. Результат классифицирован как
`one_step_failure_evidence`, не как чистый compounding failure. Следующий
диагностический split должен повторить one-step проверку на source layouts и
unseen layouts из того же checkpoint. Если source contact transitions также
плохи, проверять coverage/objective; если source хороши, а unseen плохи —
локализовать representation/dynamics generalization.

`exp148_source_target_one_step.py` выполнил этот split на четырёх source и
четырёх unseen layouts, с одинаковым seed/color и 120 real action forks.
Contact failure и blocked-noop failure воспроизвелись на **4/4 source** и
**4/4 unseen** layouts. Median `interact` prediction/persistence ratio равен
**49.14x** на source и **72.96x** на unseen; median blocked-forward prediction
MSE — **0.3321/0.3501**. При этом свободный canonical `forward` остаётся лучше
persistence с median ratio **0.2256/0.2673**.

Outcome `shared_one_step_failure_evidence` снимает layout generalization как
основное объяснение: absolute recurrent dynamics не моделирует contact/no-op
переходы даже на source geometries, хотя умеет крупный свободный movement.
Следующий вопрос — покрытие replay: сколько в исходных 130676 transitions
state-changing `interact` и blocked no-op примеров. Только после этого выбирать
между generic event-balanced training и residual/persistence parameterization.

`exp149_replay_coverage.py` повторил точный fixed collection и получил те же
**2048 episodes / 130676 transitions**. В полном corpus natural terminal counts
равны east/west/south/north **2/7/2/7**; в первых 384 episodes каждого layout,
использованных probe fit, — **2/4/2/5**. Из **26125** `interact` переходов
**1925** меняют RGB (`7.37%` action, `1.47%` всего corpus), причём хотя бы один
такой переход есть в **1281/2048** episodes. Одновременно corpus содержит
**24200** no-change `interact`, **9358** no-change `forward` и **26082** noop
transitions.

Следовательно, contact-change разрежен на уровне transitions и uniform loss
может недовзвешивать его, но опыт не отсутствует. Более важно, identity/no-change
переходы представлены десятками тысяч примеров, а learned dynamics всё равно
галлюцинирует изменение на blocked `forward`. Следующий matched arm должен
проверить residual/persistence latent parameterization на том же replay; один
event-balanced arm допустим только как контроль редкого contact-change, а не как
основное объяснение no-op failure.

Следующий matched arm реализован в `exp150_residual_dynamics.py`: fresh
residual model обучается по exp146 protocol, затем сохраняется checkpoint и
выполняются exp148 one-step diagnostic для frozen baseline/residual, residual
late-fork audit (125 rows) и четыре исходных MPC arms. По умолчанию corpus
проверяется на 130676 transitions и terminal total/fit maps exp149.

Заранее заданный residual one-step gate требует exact matched protocol,
source contact failures **0/4**, blocked-noop failures **0/4**, median
free-forward prediction/persistence ratio **<1**, а также уменьшения обоих
failure counts относительно baseline. Composition gate дополнительно требует
исходный exp146 source gate: ordered H3 **≥18/24**, **≥3/6** на каждом layout,
выигрыш **≥4** над каждым control и terminal fit coverage всех source layouts.
Physics gate остаётся `null`. Пороги не подбираются; результаты полного
обучения зафиксированы в `exp150-residual-dynamics-001`.

Run завершился с `exact_protocol=true`, exit 0 и точным corpus
**2048 episodes / 130676 transitions**. Dynamics loss снизился
**1.311422→0.495811**; ordered probe validation balanced accuracy составила
**0.713767** против **0.467147** shuffled. Baseline source воспроизвёл contact
и blocked failures **4/4 и 4/4**, median interact ratio **49.1422x**, blocked
MSE **0.332071** и free-forward ratio **0.225626**. Residual source тоже дал
**4/4 и 4/4**, но с medians **34.0592x**, **0.365026** и **0.103175**;
residual unseen — **4/4 и 4/4**, **54.6281x**, **0.416880** и **0.094417**.

На canonical late fork residual улучшил predicted ordered rank успешной
последовательности с baseline **54/125** до **18/125**; raw rank — **8/125**
против baseline **42/125**, endpoint MSE — **0.153106**. Но predicted winners
`[1,3,3]` остались неуспешны. Evaluation `ordered_h3`, `ordered_h1`,
`shuffled_h3` и `raw_h3` дала по **0/24**. Поэтому one-step gate,
source-compositional gate и composition gate — `false`, physics gate — `null`;
Push-2 не запускался. Residual parameterization дала частичное улучшение
ranking и free-motion prediction, но не прошла prerequisites и не является
доказательством AGI/JEPA.

Observability сохранена: **613** progress records, maximum gap **30.021 s**;
`run.log`, results, manifest, checkpoint, rows и traces полны.
Следующий bounded experiment — event-balanced sampling как отдельный causal
control без одновременного изменения planner, parameterization или objective.

Этот control завершён в `exp151_event_balanced_dynamics.py`. Full run
`exp151-event-balanced-dynamics-001` сохранил residual parameterization,
planner, objective, replay и protocol exp150; менялся только sampler. Run имеет
`exact_protocol=true`, exit 0 и точный corpus **2048 episodes / 130676
transitions**. Из них sampling pool классифицировал **1894 event** и
**124686 ordinary** transitions. Оба anchor budgets заполнены **8000/8000**;
с учётом multi-step targets training получил **8545 event** и **39455
ordinary** supervised targets, то есть **17.80% event**. Критерий RGB-change
при action 3 — только Push-local proxy события, не семантика действия и не
переносимый факт среды.

Dynamics loss снизился **1.31198895→0.42985901**. Ordered temporal probe на
validation сохранил сигнал: balanced accuracy **0.71299148** против
**0.48725784** shuffled. Frozen exp150 residual baseline на source снова дал
contact/blocked failures **4/4 и 4/4**, medians interact **34.05918**, blocked
MSE **0.365026**, free-forward ratio **0.103175**. Event-balanced source также
дал **4/4 и 4/4**, но medians стали **26.21614**, **0.151566** и **0.287251**.
То есть contact и blocked metrics частично улучшились, тогда как free-motion
prediction ухудшилась. Event-balanced unseen сохранил failures **4/4 и 4/4** с
medians **40.01761**, **0.241659** и **0.248932**.

Canonical late-fork audit дал successful sequence predicted ordered rank
**95/125**, raw rank **104/125** и endpoint MSE **1.19277**; predicted winners
остались неуспешны. `ordered_h3`, `ordered_h1`, `shuffled_h3` и `raw_h3` снова
получили по **0/24**. One-step, source-compositional и composition gates —
`false`, physics gate — `null`; Push-2 не запускался. Значит, повышение частоты
редких event transitions частично снижает отдельные one-step ошибки, но не
снимает failures и сильно ухудшает rollout ranking. Вместе exp150 и exp151
исключают sparse frequency alone и residual parameterization alone как
достаточное объяснение root failure.

Run оставил **577** progress records с maximum gap **2.793 s** и elapsed
**134.097 s**. `run.log`, manifest, results, checkpoint, rows и traces полны.

Exp152 закрыл этот diagnostic на frozen checkpoint head `afdf53e`.
Full HyperPC run `exp152-representation-separability-001` завершился с
`exact_protocol=true`, exit 0 и точным corpus **130676** transitions;
protocol/checkpoint gates прошли. Probe features состояли только из
frozen current `z` и action; after-RGB использовался для target,
но не подавался encoder или probe.

`interact` task имел train counts **18160 no-change / 1425 change** и
episode-disjoint held-out counts **6040 / 500**. Ordered linear probe дал
balanced accuracy **0.88372185**, recall no-change/change
**0.7834437 / 0.984** и confusion `tn=4732, fp=1308, fn=8, tp=492`;
shuffled-label control дал **0.56801987** balanced accuracy.

`forward` task имел train counts **7036 blocked/no-change / 12590
moving/change** и held-out counts **2322 / 4174**. Ordered probe дал balanced
accuracy **0.94262883**, recall blocked/moving **0.916882 / 0.968376** и
confusion `tn=2129, fp=193, fn=132, tp=4042`; shuffled-label control дал
**0.31591830**. Оба preregistered signals прошли все три порога:
balanced accuracy **≥0.8**, each-class recall **≥0.7** и
ordered-minus-shuffled margin **≥0.2**. Outcome —
`representation_signal_evidence`.

Результат снимает encoder availability как текущий root bottleneck:
representation содержит линейно доступный signal для обоих
failure classes, тогда как текущая dynamics его не использует. Но probe
проверен на одном seed, а frozen encoder до split видел весь corpus
в self-supervised training; episode-disjoint только supervised probe split.
Это signal evidence, а не доказательство AGI, JEPA или transfer.

Следующий минимальный experiment-only gate — напрямую обусловить
transition state текущими `z + action`, сохранив uniform replay и не
добавляя event labels, новый objective или planner changes. Такой
matched gate проверит, может ли dynamics использовать уже доступный
signal без смены других causal factors.

Observability: **324** progress records, maximum gap **1.313 s**, elapsed
**81.453 s**; manifest, results и `run.log` полны.

Exp153 выполнил следующий matched experiment на коде `49877e4`. Full run
`exp153-change-gated-dynamics-001` сохранил exp150 architecture, predictive
objective, uniform replay, planner, seed и protocol; единственное изменение —
per-member sigmoid gate из current `z + action embedding`, который
мультипликативно масштабирует residual delta. Run завершился с
`exact_protocol=true`, exit 0 и elapsed **1302.454 s**.

Dynamics loss снизился **1.311422→0.462709**. Ordered temporal probe сохранил
signal: balanced accuracy **0.713612** против **0.471875** shuffled. Frozen
exp150 baseline на source воспроизвёл contact/blocked failures **4/4 и 4/4**,
medians interact **34.05918**, blocked MSE **0.365026**, free-forward ratio
**0.103175**. Gated source получил contact/blocked **0/4 и 4/4**, medians
**0.979424**, **0.262021** и **0.202104**; gated unseen — **0/4 и 4/4**,
**0.985134**, **0.239985** и **0.177488**.

Нулевой contact failure — технический threshold pass: near-persistence
prediction лишь немного лучше persistence на changed transition. Он не
показывает, что модель выучила содержательный interaction effect. Более того,
gate statistics в основном соответствуют action prior, а не устойчивому
within-action state discrimination. Actions 0/1 имеют gate примерно
**0.998–0.999**; action 3 — около **0.0013/0.0019/0.009** на трёх canonical
steps, action 4 также остаётся очень малым. Для forward contexts значения
примерно **0.690** blocked, **0.758** changed и **0.935** blocked, без
надёжного разделения blocked/moving. Gate — multiplicative amplitude, не
калиброванная вероятность изменения.

Canonical late-fork audit дал successful sequence predicted ordered rank
**24/125**, raw rank **22/125** и endpoint MSE **0.150794**; predicted winners
остались неуспешны. `ordered_h3`, `ordered_h1`, `shuffled_h3` и `raw_h3`
получили по **0/24**. One-step gate — `false` из-за blocked-noop failures;
source-compositional и composition gates — `false`, physics gate — `null`,
Push-2 не запускался.

Вывод: multiplicative identity bias подавляет interact hallucination на source
и unseen layouts, но не решает blocked-forward или reactive composition.
Следующий минимальный causal arm должен сохранить эту architecture и uniform
replay, меняя только objective: добавить self-supervised RGB change/no-change
gate auxiliary с class balancing внутри action. Task-success labels, planner
changes и Push-2 в этот arm не добавляются. Ограничения остаются прежними: один
seed/task family, некалиброванный gate, отсутствие AGI/JEPA/transfer proof.

Run оставил **735** progress records с maximum gap **30.021 s**. Полны
`run.log`, manifest, results, checkpoint, one-step, gate, fork и eval artifacts.

Exp154 выполнил preregistered objective-only follow-up на protocol commit
`9f896a336f9186ea2129306afdda06313b261908`. Full HyperPC run
`exp154-auxiliary-change-gate-001` сохранил architecture exp153, uniform replay
и planner; единственное causal изменение — self-supervised RGB
change/no-change auxiliary для gate с class balancing внутри action. Run
завершился с `exact_protocol=true`, exit 0 и elapsed **1300.724 s**.

Predictive loss снизился **1.311422→0.473953**, auxiliary loss —
**0.620841→0.500368**. Ordered temporal probe сохранил signal: balanced
accuracy **0.712866** против **0.503387** shuffled. Frozen exp153 baseline на
source воспроизвёл contact/blocked failures **0/4 и 4/4**, medians interact
**0.979424**, blocked MSE **0.262021**, free-forward ratio **0.202104**.
Auxiliary source получил **4/4 и 4/4**, medians **20.896683**, **0.093231** и
**0.750244**; auxiliary unseen — **4/4 и 4/4**, **21.797050**, **0.239616** и
**1.789780**.

Gate diagnostic не прошёл. Source forward margins равны
**0.00653/0.01425**, interact margin **−0.055997**; unseen forward margins —
**0.01069/0.00772**, interact — **−0.032788**, тогда как preregistered threshold
равен **0.15**. Canonical successful continuation получила predicted
ordered/raw ranks **91/125 и 88/125**, endpoint MSE **0.553557**; predicted
winner снова неуспешен. `ordered_h1`, `ordered_h3`, `raw_h3` и `shuffled_h3`
получили по **0/24**. One-step, source-compositional и composition gates —
`false`, physics gate — `null`; Push-2 не запускался.

Вывод: raw visual-change auxiliary не извлёк требуемую within-action event
семантику, вернул contact hallucination и не решил blocked conditional
transition. Частичное снижение source blocked MSE не компенсирует degradation
contact, free motion и rollout ranking. Дальнейшее tuning этого auxiliary
objective не обосновано; следующий минимальный causal direction — learned
state-transition/event representation либо factorized object-centric delta
target. Это результат одного seed и одного Push-1 family, не AGI/JEPA/transfer
proof.

Observability сохранена: **737** progress records, maximum gap **30.0214 s**;
полны `run.log`, manifest, results, checkpoint, one-step, gate, fork и eval
artifacts, а также exact command, Git commit и exit status.

Exp155 выполнил checkpoint-only oracle audit frozen exp150 residual delta:
может ли оптимальная scalar amplitude устранить one-step failures без нового
обучения. Контракт задан RED commit `fa4db01`, implementation —
`1ec81c420907c0bcb51ed63b8f4d16f659f3c11f`. HyperPC verification дала
**15 passed, 1 skipped** за **1.83 s**, artifact check — PASS. Full run
`exp155-oracle-residual-gate-001` завершился с `exact_protocol=true`, exit 0 и
runtime **1.719312 s**.

Persistence baseline на source/unseen имеет contact/blocked failures **4/4 и
0/4**, free-forward ratio **1**, interact ratio **1** и blocked MSE **0**.
Ungated exp150 на source воспроизвёл **4/4 и 4/4**, medians free-forward
**0.103175**, interact **34.05918**, blocked MSE **0.365026**; unseen —
**4/4 и 4/4**, **0.094417**, **54.62809**, **0.416880**. Shared и per-member
scalar oracles совпали по агрегатам: source **4/4 и 0/4**, medians **0.103175**,
**1**, **0**; unseen **3/4 и 0/4**, **0.094417**, **1**, **0**. Gate — `false`.

Oracle сохраняет полезную free-motion delta, но на contact выбирает фактически
persistence-level prediction и потому не проходит критерий. Значит,
hindsight-optimal scalar amplitudes не исправляют направления frozen exp150
residual deltas. Этот результат не отвергает другие или jointly learned delta
directions и не разрешает action-specific scalar training. Следующий
минимальный diagnostic — checkpoint-only raw-delta oracle audit exp153/154 до
любого следующего training arm.

Run сохранил **120** diagnostic rows и **127** progress records с maximum gap
**0.315408 s**; `run.log`, manifest и results полны, exact command, Git commit
и exit status зафиксированы.

Exp156 проверил следующий вопрос без retraining backbone: содержат ли raw
pre-gate deltas exp153/154 правильные направления, даже если native learned
gates выбирают неверные amplitudes. Контракт задан RED commit `b810432`,
implementation — `ffe240ec9f6ca24038187718f886a6e8f34879dc`; exact checkpoint
heads — exp153 `49877e4`, exp154 `9f896a3`. HyperPC verification дала
**22 passed, 2 skipped** за **1.93 s**. Full run
`exp156-gated-delta-oracle-001` завершился с `exact_protocol=true`, exit 0 и
runtime **2.793098 s**.

Exp153 native source имеет contact/blocked failures **0/4 и 4/4**, medians
free-forward **0.202104**, interact **0.979424**, blocked MSE **0.262021**;
unseen — **0/4 и 4/4**, **0.177488**, **0.985134**, **0.239985**. Raw
per-member oracle дал source **0/4 и 0/4**, medians **0.172222**, **0.899737**,
**0**; unseen — **0/4 и 0/4**, **0.153195**, **0.931245**, **0**. Gate прошёл.

Exp154 native source имеет **4/4 и 4/4**, medians **0.750244**,
**20.896683**, **0.093231**; unseen — **4/4 и 4/4**, **1.789780**,
**21.797050**, **0.239616**. Raw oracle дал source **0/4 и 0/4**, medians
**0.666550**, **0.928792**, **0**; unseen — **1/4 и 0/4**, **0.828144**,
**0.799142**, **0**. Source gate прошёл, но exp154 unseen transfer не доказан.

Обе модели имеют выразительные raw delta directions на source, поэтому
bottleneck сужен до learnability/objective либо expressivity текущего learned
gate. Exp153 directions также проходят unseen и являются более сильной базой.
Следующий минимальный causal arm замораживает exp153 encoder, recurrent state и
raw deltas, обучая только action-specific gates по latent predictive objective;
RGB/task labels и planner changes не добавляются.

Run сохранил два файла по **120** diagnostic rows и **250** progress records с
maximum gap **0.290957 s**; `run.log`, manifest и results полны, exact command,
Git commit и exit status зафиксированы.

Exp157 проверил, можно ли выучить oracle-expressive raw directions exp153,
заморозив весь backbone и обучая лишь action-specific gates по latent
predictive MSE. Контракт задан RED commit `769d86d`, implementation `6cc2156`,
scalar-output shape исправлен commit
`1bd15b56a0fcbebe7c765dfac9d0532b8cfdeabd`. HyperPC verification дала
**14 passed, 1 skipped** за **1.07 s**, smoke завершился за **4.88 s**. Full run
`exp157-action-specific-frozen-gate-001` завершился с `exact_protocol=true`,
exit 0 и runtime **1302.761 s**.

Все **1,403,398** frozen parameters остались неизменными (`true`); trainable
были только **3,855** gate parameters. Uniform latent predictive MSE снизилась
**0.232270→0.104996**. Frozen exp153 probe воспроизвёл ordered balanced
accuracy **0.713612** против **0.471875** shuffled.

Baseline source имел contact/blocked failures **0/4 и 4/4**, medians
free-forward **0.202104**, interact **0.979424**, blocked MSE **0.262021**;
unseen — **0/4 и 4/4**, **0.177488**, **0.985134**, **0.239985**. Candidate
source ухудшился до **4/4 и 4/4**, medians **0.249516**, **1.436134**,
**0.162705**; unseen — до **4/4 и 4/4**, **0.212795**, **2.380314**,
**0.150267**.

На action 2 source gate значения равны **0.5564** blocked, **0.6379** changed
и **0.8679** blocked: второй blocked context ошибочно получает наибольшую
amplitude. Action 3 changed contexts дают **0.0927/0.1540**, no-change —
**0.0947**, без устойчивого разделения. Все MPC arms получили **0/24**;
canonical ordered/raw ranks **24/26**, endpoint MSE **0.191819**, winner
неуспешен. One-step/source-compositional/composition gates — `false`, physics
gate — `null`.

Вывод: action-specific boundary при uniform latent MSE недостаточна, хотя
exp156 показал, что frozen raw directions выразительны. Следующий matched arm
оставляет тот же frozen backbone, raw deltas и action-specific gates, но
балансирует latent predictive error внутри `(action, RGB-change/no-change)`.
RGB применяется только для weighting наблюдённого transition, latent target
остаётся next `z`; BCE, новая architecture, task labels и planner changes не
добавляются. Это не AGI/JEPA/transfer claim.

Observability: **730** progress records, maximum gap **30.021144 s**; полны
`run.log`, manifest, results, checkpoint, rows и traces, exact command, Git
commit и exit status.

Exp158 выполнил matched follow-up exp157, изменив только weighting latent
predictive error внутри `(action, RGB-change/no-change)`. Контракт задан RED
commit `c40d983`, implementation —
`15251710df58dec697900de3cd65d350c44d9b38`. Full HyperPC run
`exp158-balanced-latent-gate-001` завершился с `exact_protocol=true`, exit 0 и
runtime **1238.310 s**. Все **1,403,398** backbone parameters остались frozen
и неизменными; trainable были те же **3,855** gate parameters.

Exact class weights `[no-change, change]` для actions 0–4 равны
`[[0,1],[0,1],[1.395704,.779110],[.539773,6.785714],[1,0]]`. RGB-change
использовался только для веса observed transition; latent target оставался next
`z`, BCE/task labels не добавлялись. Первый/последний sampled loss —
**0.221201/0.263924**; из-за stochastic balanced sampling это не evidence
ухудшения или улучшения objective.

Exp153 baseline source имел contact/blocked failures **0/4 и 4/4**, unseen —
**0/4 и 4/4**. Candidate source получил **4/4 и 4/4**, medians free-forward
**0.260011**, interact **1.906250**, blocked MSE **0.161449**; unseen —
**4/4 и 4/4**, **0.214704**, **2.807827**, **0.149274**. Action 2 source gate
дал **0.5326** blocked, **0.6164** changed, **0.8678** blocked; action 3 —
**0.1028/0.1928** changed против **0.1051** no-change.

Все MPC arms получили **0/24**. Canonical late-fork ordered/raw ranks —
**24/26**, endpoint MSE **0.193162**, predicted winner `[1,4,4]` неуспешен.
One-step/source-compositional/composition gates — `false`, physics — `null`.
Fixed action/change class weights не исправляют amplitude learning под latent
MSE, поэтому coefficient sweep не запускается. Следующий дешёвый diagnostic —
checkpoint-only independent-member analytic amplitude target audit до любого
нового долгого regression run. Это не AGI/JEPA/transfer claim.

Observability: **730** progress records, maximum gap **30.020592 s**; полны
`run.log`, manifest, results, checkpoint, rows и traces, exact command, Git
commit и exit status.

Exp159 выполнил дешёвый checkpoint-only audit перед следующим regression run:
может ли independent analytic amplitude на каждом ensemble member реализовать
нужные one-step transitions frozen exp153. Контракт задан RED commit `1fb3d58`,
implementation — `2449e374fc456ea48abff0579d7c849efa28bf6f`.
HyperPC verification дала **5 passed** за **1.04 s**, artifact check — PASS.
Full run `exp159-independent-amplitude-oracle-001` завершился с
`exact_protocol=true`, exit 0 и runtime **1.7319 s**.

Native source имел contact/blocked failures **0/4 и 4/4**, medians free-forward
**0.202104**, interact **0.979424**, blocked MSE **0.262021**; unseen —
**0/4 и 4/4**, **0.177488**, **0.985134**, **0.239985**. Independent oracle
получил source **0/4 и 0/4**, medians **0.172222**, **0.899737**, **0**;
unseen — **0/4 и 0/4**, **0.153195**, **0.931245**, **0**. Joint scalar
control численно совпал по агрегатам. Independent gate прошёл; outcome —
`target licensed`.

Всего измерено **360** amplitudes: min/median/max **0/0.046656/1**; counts
**144 zero / 72 one / 144 interior**. Поэтому target является continuous и
member-specific, а не просто binary RGB-change label. Результат показывает его
совместимость с frozen raw directions на audited one-step rows, но ещё не
learnability, rollout или transfer.

Следующий exp160 сохраняет frozen exp153 backbone, action-specific gate
architecture, fixed action/change weights, raw deltas и planner. Единственное
изменение — прямая regression self-supervised independent analytic amplitude
targets. Task labels и новая architecture не добавляются; AGI/JEPA claim нет.

Observability: **127** progress records, maximum gap **0.31937 s**; `run.log`,
manifest, results и diagnostic rows полны, exact command, Git commit и exit
status зафиксированы.

Exp160 проверил learnability licensed independent analytic amplitude target при
той же frozen exp153/action-specific architecture. Контракт задан RED commit
`d1860a0`, implementation — `17abd216a5934752e109c2ac170bb850862f846c`.
Focused HyperPC verification дала **8 passed** за **1.09 s**, smoke завершился
за **86.94 s**. Full run `exp160-amplitude-supervised-gate-001` завершился с
`exact_protocol=true`, exit 0 и runtime **1258.398 s**.

Все **1,403,398** backbone parameters остались frozen/неизменными; trainable
были только **3,855** gate parameters. Amplitude loss снизился
**0.156594→0.051361**. Из **144000** sampled targets **47724** были zero,
**16458** one и **79818** interior; mean target **0.506558**.

Exp153 baseline source имел contact/blocked failures **0/4 и 4/4**, medians
free-forward **0.202104**, interact **0.979424**, blocked MSE **0.262021**;
unseen — **0/4 и 4/4**, **0.177488**, **0.985134**, **0.239985**. Candidate
source получил **4/4 и 4/4**, medians **0.221825**, **1.644737**,
**0.202125**; unseen — **4/4 и 4/4**, **0.184195**, **2.243522**,
**0.189394**.

Action 2 source gate values — **0.6050** blocked, **0.6909** moving,
**0.8353** blocked; action 3 — **0.1149/0.1406** changed против **0.1237**
no-change. Все MPC arms дали **0/24**. Canonical late-fork ordered/raw ranks —
**23/24**, endpoint MSE **0.180957**, winner `[1,4,4]` неуспешен.
One-step/source-compositional/composition gates — `false`, physics — `null`.

Снижение amplitude loss не решило rare-context behavior. Следующий bounded
diagnostic — episode-disjoint teacher-forced probe, сравнивающий `z`-linear с
`z+hidden`-linear decoding analytic amplitude; `z`-MLP добавляется только если
нужен минимальный tie-break. До этого architecture training не запускается.
Это не AGI/JEPA/transfer claim.

Observability: **730** progress records, maximum gap **30.022755 s**; полны
`run.log`, manifest, results, checkpoint, rows и traces, exact command, Git
commit и exit status.

Exp161 выполнил дешёвый episode-disjoint teacher-forced probe перед новым
architecture training. Implementation commit —
`419fe0018fc7d6584fc2c42d1ffdf74dab3e5494`; HyperPC run завершился с
`exact_protocol=true`, status `completed` и runtime **235.3017 s**. Split:
**1536** train / **512** held-out episodes, overlap **0**.

На held-out `z`-linear amplitude MSE составила **0.0325955**, weighted —
**0.0294388**. `z+hidden`-linear существенно лучше: **0.00883724** и
**0.00882724**. Но one-step evaluation не подтвердила sufficiency ни одного
input. Native source имел contact/blocked failures **0/4 и 4/4**, medians
free-forward **0.202104**, interact **0.979424**, blocked MSE **0.262021**;
unseen — **0/4 и 4/4**, **0.177488**, **0.985134**, **0.239985**.

`z`-linear source получил contact/blocked **4/4 и 4/4**, free/interact
**0.218087/3.185324**; unseen — **4/4 и 4/4**, **0.183780/5.726008**.
`z+hidden`-linear source также дал **4/4 и 4/4**, medians free/interact/blocked
**0.265161/1.519591/0.088492**; unseen — **4/4 и 4/4**,
**0.1928968/1.000498/0.117072**. Оба arm gates — `false`, outcome —
`both_linear_inputs_fail`.

Таким образом, hidden state materially улучшает aggregate amplitude regression,
но обе linear teacher-forced модели проваливают критическое one-step behavior.
Следующий минимальный diagnostic проверяет nonlinear decodability либо
object-centric transition target, а не более долгое retraining текущего gate.
Это не AGI/JEPA/transfer claim.

Persistent `run.log`, `progress.jsonl`, `results.json` и `manifest.json`
содержат прогресс, exact command, Git commit и финальный статус.

Exp162 проверил минимальную нелинейность после провала linear probes exp161.
Контракт задан RED commit `0e9e721`, implementation —
`8edec06cca48b34a8285dec7d943f5ff4332082e`. Fresh HyperPC verification дала
**7 passed** за **1.06 s**, smoke завершился за **3.222 s**. Full run завершён с
`exact_protocol=true`, exit 0 и runtime **230.7115 s**.

Тот же corpus содержит **2048 episodes**. Episode-disjoint train split:
**1536 episodes / 98037 transitions**; held-out: **512 / 32639**; overlap
**0**. Frozen exp153 parameters не изменились. Exp161 linear arm не
переобучался и использован только как reference. Новый per-action probe получает
`z+hidden`, имеет один **128 ReLU** hidden layer и обучается **400** updates.

Training loss снизился **0.224341→0.001197**. Held-out MSE равна
**0.00435795**, weighted **0.00465816**, против exp161 linear reference
**0.00883724/0.00882724** — примерно **2.03x** лучше по plain MSE. Exact
one-step source получил contact/blocked failures **1/4 и 4/4**, medians
free-forward **0.262579**, interact **0.911038**, blocked MSE **0.057772**;
unseen — **0/4 и 4/4**, **0.304037**, **0.959895**, **0.011988**. Linear
reference имел source **4/4 и 4/4**, interact **1.519591**; unseen **4/4 и
4/4**, interact **1.000498**.

Gate остаётся `false`: нелинейные interactions materially улучшают aggregate
regression, movement/contact/interact и blocked MSE, но source contact не
идеален, а blocked-noop physics не проходит ни на source, ни на unseen.
Следующий минимальный diagnostic — object-centric state/target, а не longer
training этого MLP. Composition и transfer не проверялись; AGI/JEPA claim нет.

Persistent `run.log`, `results.json`, `manifest.json` и `progress.jsonl`
содержат **2654** progress records с maximum gap **1.2763 s**, exact command,
Git commit и финальный status.

Exp163 проверил пост-hoc calibration frozen exp162 amplitude MLP: можно ли
восстановить zero atom одним per-action threshold без retraining. Контракт задан
RED commit `718c52e`, implementation — `6546387`. Fresh HyperPC verification
дала **6 passed** за **0.99 s**, smoke завершился за **4.308 s**. Full run
завершён с `exact_protocol=true`, status `completed`, exit 0 и runtime
**121.037 s**; artifact verifier — PASS. Frozen backbone сохранён, source
leakage в calibration отсутствует.

Пороги по actions:
`[0.67221558,0.56983912,0.25142145,0.03797820,0.02714755]`. Held-out latent
MSE почти не изменилась: **0.04614048→0.04605277**, около **0.19%**. Native
source имел contact/blocked failures **1/4 и 4/4**, medians free-forward
**0.262579**, interact **0.911038**; calibrated source — **3/4 и 2/4**,
**0.262579**, **1.0**. Native unseen — **0/4 и 4/4**, **0.304037**,
**0.959895**; calibrated unseen — **4/4 и 1/4**, **0.304037**, **1.0**.

Action 3 threshold подавляет **99.82%** held-out zero targets, но сохраняет
только **6.99%** positive. Calibration частично снижает blocked failures, но
уничтожает contact behavior; gate остаётся `false`. Это локализует score
overlap/ranking-state failure, а не недостаточную длину calibration/training.

Следующий минимальный diagnostic — evaluator-only relational object-state
проверка до любого нового training arm. Composition/transfer не проверены;
AGI/JEPA claim отсутствует.

Persistent `run.log`, `results.json`, `manifest.json`, checkpoint, rows и
`progress.jsonl` содержат **1092** progress records с maximum gap **1.303 s**,
exact command, Git commit и финальный status.

Exp164 выполнил evaluator-only relational object-state diagnostic поверх frozen
exp153. Контракт задан RED commit `ccdb55b`, implementation — `7a6a6ca`. Fresh
HyperPC verification дала **7 passed** за **1.06 s**, smoke завершился за
**3.570 s**. Full run завершён с `exact_protocol=true`, status `completed`,
exit 0 и runtime **321.076 s**; artifact verifier — PASS.

Sidecar выровнен со всеми **130676** corpus transitions по сохранённому digest;
canonical **120** signatures совпадают. Frozen backbone не менялся. Per-action
MLP получает `z+hidden` и четыре privileged position-based relational slots.
Probe loss снизился **0.223857→0.00112491**; held-out MSE — **0.00401893**
против exp162 **0.00435795**, около **7.8%** improvement.

Exact source exp164 имеет contact/blocked failures **1/4 и 4/4**, medians
free-forward **0.247198**, interact **0.907779**, blocked MSE **0.050738**;
exp162 reference — **1/4 и 4/4**, **0.262579**, **0.911038**, **0.057772**.
Unseen exp164 — **0/4 и 4/4**, **0.286226**, **0.957583**, **0.009072**;
exp162 — **0/4 и 4/4**, **0.304037**, **0.959895**, **0.011988**. Continuous
metrics улучшились, categorical failures остались прежними; gate — `false`.

Ограничение принципиально для интерпретации: четыре relations не включают agent
orientation/pose, тогда как PushGrid transition зависит от `agent_dir`.
Следовательно, exp164 отвергает только position-only relational slots, а не
complete object-centric Markov state. Следующий минимальный diagnostic остаётся
evaluator-only и добавляет pose/orientation. Longer training пока не обоснован;
AGI/JEPA/composition/transfer claim отсутствует.

Persistent `run.log`, `results.json`, `manifest.json`, checkpoint, rows и
`progress.jsonl` содержат **4706** progress records с maximum gap **1.713 s**,
alignment evidence, exact command, Git commit и финальный status.

Exp165 закрыл обязательный pose/orientation control для relational input.
Контракт задан RED commit `73efd9a`, implementation `e349778`, floating pose
fixture исправлен test commit `001505f`. Fresh HyperPC verification дала
**8 passed** за **1.06 s**, smoke завершился за **3.564 s**. Full run завершён
с `exact_protocol=true`, status `completed`, exit 0 и runtime **324.739 s**;
artifact verifier — PASS.

Sidecar содержит **130676** digest-aligned rows; orientation one-hot validation
и canonical **120** signature match прошли. Frozen backbone не менялся.
Training loss снизился **0.221014→0.00126149**, но held-out MSE равна
**0.00414342** против exp164 **0.00401893** — **3.10% хуже**.

Exact source exp165 имеет contact/blocked failures **1/4 и 4/4**, medians
free-forward **0.272260**, interact **0.907236**, blocked MSE **0.045784**;
exp164 reference — **1/4 и 4/4**, **0.247198**, **0.907779**, **0.050738**.
Unseen exp165 — **0/4 и 4/4**, **0.314310**, **0.957243**, **0.012235**;
exp164 — **0/4 и 4/4**, **0.286226**, **0.957583**, **0.009072**. Gate —
`false`; outcome — `pose_categorical_failures_unchanged`.

Таким образом, full position+orientation object input не снимает categorical
failure при текущем independent-amplitude MSE. Следующий минимальный diagnostic
проверяет richer transition target: отдельный zero atom плюс conditional
amplitude. Longer training и coefficient tuning не запускаются. Результат не
опровергает object-centric modeling в целом и не является AGI/JEPA/transfer
claim.

Persistent `run.log`, `results.json`, `manifest.json`, checkpoint, rows и
`progress.jsonl` содержат **4707** progress records с maximum gap **1.691 s**,
alignment evidence, exact command, Git commit и финальный status.

Exp166 проверил richer hurdle target: отдельный zero atom classifier и
conditional-positive amplitude regression. Контракт задан RED commit `c5a7b4d`,
implementation — `e1e1bc4`. Fresh HyperPC verification дала **7 passed** за
**1.06 s**, smoke завершился за **4.394 s**. Full run завершён с
`exact_protocol=true`, exit 0 и runtime **327.601 s**; artifact verifier — PASS
для **8** artifacts, frozen backbone сохранён.

Training loss снизился **0.905769→0.038797**. На held-out atom balanced
accuracy равна **0.975702**, recall zero **0.960944**, recall positive
**0.990460**. Conditional-positive MSE — **0.001694**, но exact-zero rate —
только **0.448645**.

Exact source exp166 имеет contact/blocked failures **2/4 и 0/4**, medians
free-forward **0.357238**, interact **0.929919**; unseen — **4/4 и 1/4**,
**0.614373**, **1.0**. По сравнению с exp165 blocked failures уменьшились
**4→0 source** и **4→1 unseen**, но contact выросли **1→2** и **0→4**. Gate —
`false`; outcome — `conditional_amplitude_delta_failure`.

Canonical source atom-zero rate / gate amplitude: blocked **1/0**, free
**0.4167/0.5324**, contact **0/0.0154** и **0/0.0413**. Unseen blocked —
**0.75/0.2319**, free — **0.5833/0.3627**, contact zero rate **0.25** с
amplitudes **0.00816/0.03185**. Поэтому высокий aggregate atom score не
обеспечивает корректное ранжирование critical states.

Следующий минимальный diagnostic — exact **2×2** oracle swap:
`predicted/oracle atom × predicted/oracle conditional amplitude`. Он отделит
atom error от conditional-delta error до любого нового training/tuning. Это не
AGI/JEPA/composition/transfer claim.

Persistent `run.log`, `results.json`, `manifest.json`, checkpoint, rows и
`progress.jsonl` содержат **4828** progress records с maximum gap **1.324 s**,
exact command, Git commit и финальный status.

Exp167 выполнил preregistered **2×2** oracle swap для двух hurdle components:
predicted/oracle atom × predicted/oracle conditional amplitude. Контракт задан
RED commit `58de3b2`, implementation — `3c735f5`; validator tolerance уточнён
RED `4107acc` и fix `35809ac`. Fresh HyperPC verification дала **7 passed** за
**1.05 s**.

Официальный run `exp167-hurdle-oracle-swap-002` завершился с
`exact_protocol=true`, exit 0 и runtime **1.83593 s**; verifier — PASS. `-001`
сохранён как superseded: scientific result не менялся, повтор потребовался из-за
exact-float validator. `PP` reference отличается от exp166 максимум на
**1.49e-8** при tolerance **1e-7**; `OO` reference diff — **0**.

Exact contact/blocked/free/interact metrics:

- `PP`: source **2/4, 0/4, 0.357238, 0.929919**; unseen
  **4/4, 1/4, 0.614373, 1.0**.
- `PO`: source **0/4, 0/4, 0.339605, 0.899737**; unseen
  **1/4, 0/4, 0.647844, 0.972617**.
- `OP`: source **2/4, 0/4, 0.177678, 0.929919**; unseen
  **4/4, 0/4, 0.160719, 0.992514**.
- `OO`: source **0/4, 0/4, 0.172222, 0.899737**; unseen
  **0/4, 0/4, 0.153195, 0.931245**.

Gates `PP/PO/OP` — `false`, `OO` — `true`; outcome —
`both_components_fail`. Atom match — **318/360**, с **24 FP** и **18 FN**;
ошибки есть в actions 2/3. `OP` изолирует atom failure, `PO` оставляет
conditional unseen failure, а `OO` подтверждает expressivity frozen raw delta.

Итог: learned atom и conditional amplitude оба вносят causal failure.
Продолжать scalar/hurdle gate tuning, longer training или coefficient sweeps не
обосновано; следующий шаг — mechanism/design decision. Это не
AGI/JEPA/composition/transfer claim.

Official artifact содержит **5** files и **120** rows; persistent `run.log`,
`progress.jsonl` (**130** records, maximum gap **0.306 s**), `results.json` и
`manifest.json` сохраняют exact command, Git commit и final status.

Exp168 проверил alternative transition target после exp167: вместо learned
scalar/hurdle composition напрямую предсказывать vector delta. Контракт задан
RED commit `fae0985`, implementation — `a534932`. Fresh HyperPC verification
дала **6 passed** за **1.13 s**, smoke завершился за **3.432 s**. Canonical run
завершён с `exact_protocol=true`, exit 0 и runtime **298.791 s**; artifact
verifier — PASS для **7** artifacts. Frozen backbone и split **1536/512**
сохранены.

Training loss снизился **0.617935→0.004446** за **400** recorded updates.
Held-out vector MSE равна **0.007777**, persistence MSE — **0.646083**, ratio —
**0.012038**. Exact source имеет contact/blocked failures **0/4 и 4/4**,
medians free-forward **0.124916**, interact **0.297726**, blocked MSE
**0.192491**; unseen — **0/4 и 4/4**, **0.183308**, **0.336965**,
**0.110097**.

Это materially лучше exp165/166 на movement/contact change transitions:
contact failures сняты на source и unseen, free/interact ratios резко ниже.
Однако no-change blocked остается **4/4** на обоих splits. Gate — `false`,
outcome — `vector_improvement_only`.

Decision: direct vector prediction лицензирован как полезный component, а
scalar/hurdle factorization была вредной для change transitions. Следующий
минимальный causal arm замораживает exp168 vector и обучает generic
transition-level changed/persistence atom для exact zero, не amplitude gate.
Heavier object-transition JEPA не запускается до этой проверки. Это не
AGI/JEPA/composition/transfer claim.

Persistent `run.log`, `results.json`, `manifest.json`, checkpoint, **400** loss
records, **120** rows и `progress.jsonl` с **4707** records и maximum gap
**1.319 s** сохраняют exact command, Git commit и final status.

Exp169 проверил минимальную composition, лицензированную exp168: frozen direct
vector плюс generic transition-level changed/persistence event atom для exact
zero. Контракт задан RED commit `07f0caf`, implementation — `d063aca`. Fresh
HyperPC verification дала **6 passed** за **1.07 s**. Первый smoke сохранил
failure overlap validator; второй valid smoke завершился за **7.836 s**.
Canonical run завершён с `exact_protocol=true`, exit 0 и runtime **301.131 s**;
artifact verifier — PASS для **9** artifacts.

Frozen backbone и vector checkpoint не изменились; `PP` reference diff — точно
**0**. Event training loss снизился **0.696804→0.047894**. Held-out loss равен
**0.060342**, balanced accuracy **0.966908**, recalls changed/no-change
**0.992584/0.941231**. Per-action BA: action 2 **0.960841**, action 3
**0.910417**.

Exact source получил contact/blocked failures **0/4 и 0/4**, medians
free-forward **0.132960**, interact **0.297726**: это первый complete local
transition gate в кампании. Unseen получил **1/4 и 1/4**, free ratio **1.0**,
interact **0.336965**, поэтому transfer gate не пройден. Outcome —
`source-only`; failure локализован в state/event classification transfer, не в
drift frozen vector.

Threshold tuning на unseen не запускается. Следующий mechanism-level test —
learned object-transition representation и pose-from-observation до planner
integration или AGI claims. Local source PASS не означает transfer,
composition-stage, JEPA или AGI PASS.

Persistent `run.log` (**634060 bytes**), `results.json`, `manifest.json`,
checkpoint, два файла по **120** rows и `progress.jsonl` с **4830** records и
maximum gap **1.325 s** сохраняют exact command, Git commit и final status.
Failed-overlap и valid smoke artifacts сохранены.

Артефакты: `output_to_user/core/action-confusion-*`,
`transfer-push1-random64-u1000-finalplanner-001`,
`transfer-push1-salient-u1000-001`, `exp143-temporal-proximity-002..010`,
`exp144-layout-generalization-001..006`, `exp144-hindsight-001`,
`exp144-terminal-hindsight-001..003`, `exp144-random-encoder-001..003`.
Predictive ablation: `exp144-predictive-encoder-001..003`.
Physics transfer: `exp145-physics-transfer-003`. Temporal MPC и late fork:
`exp146-temporal-mpc-source-001`, `exp146-temporal-mpc-fork-001`. Rollout
localization: `exp147-rollout-localization-002`.
Source/unseen split: `exp148-source-target-one-step-002`.
Replay coverage: `exp149-replay-coverage-003`.
Residual dynamics: `exp150-residual-dynamics-001`.
Event-balanced residual dynamics: `exp151-event-balanced-dynamics-001`.
Representation separability: `exp152-representation-separability-001`.
Change-gated residual dynamics: `exp153-change-gated-dynamics-001`.
Auxiliary change-gated dynamics: `exp154-auxiliary-change-gate-001`.
Frozen residual scalar oracle: `exp155-oracle-residual-gate-001`.
Pre-gate delta oracle audit: `exp156-gated-delta-oracle-001`.
Frozen action-specific gate training: `exp157-action-specific-frozen-gate-001`.
Balanced latent gate training: `exp158-balanced-latent-gate-001`.
Independent amplitude oracle: `exp159-independent-amplitude-oracle-001`.
Analytic amplitude supervised gate: `exp160-amplitude-supervised-gate-001`.
Amplitude input probe: exp161 HyperPC run at `419fe00`.
Nonlinear amplitude probe: exp162 HyperPC run at `8edec06`.
Frozen amplitude calibration: exp163 HyperPC run at `6546387`.
Relational slot probe: exp164 HyperPC run at `7a6a6ca`.
Relational pose probe: exp165 HyperPC run at `e349778`.
Hurdle amplitude probe: exp166 HyperPC run at `e1e1bc4`.
Hurdle oracle swap: `exp167-hurdle-oracle-swap-002` at `35809ac`.
Direct vector transition: exp168 HyperPC run at `a534932`.
Event-mode vector composition: exp169 HyperPC run at `d063aca`.

## Stage Review

**Ideological debt addressed:** отсутствие обучаемой динамики и переносимого
опыта; механическая часть добавлена, доказательный долг не закрыт.

**Layer changed:** `mechanisms`, `experience`, `stimuli` (явные GoalSpec).
Fixture и task facts остаются в среде/описании задач, не в generic policy.

**What changed:** encoder, recurrent ensemble, multi-step training, real replay,
bounded planner и frozen evaluation с контрольными условиями.

**Evidence of improvement:** residual source dynamics стала action-sensitive;
planner исправлен двумя причинными регрессиями; source replay + salient windows
дали B2/4 с A4/4; learned temporal goal score дал paired closed-loop 4/6 против
0/6 raw и 0/6 shuffled на одном Push fixture. Exp152 показал линейно
доступный event/blocked signal в frozen representation; exp153 подавил
interact hallucination на source/unseen, но blocked-forward и composition
остались нерешёнными. Exp154 показал, что raw RGB-change auxiliary не снимает
этот wall и ухудшает contact/free-motion dynamics. Стабильного transfer
improvement всё ещё нет. Exp155 дополнительно исключил scalar amplitude frozen
exp150 delta как достаточное локальное исправление: oracle возвращается к
persistence на contact вместо моделирования effect. Exp156 показал, что
pre-gate deltas exp153/154 всё же выразительны на source, а exp153 — также на
unseen: текущий wall теперь локализован в обучении/выразительности gate. Exp157
показал, что action-specific parameterization с uniform latent MSE эту
learnability проблему не снимает. Exp158 показал, что fixed balancing по
action/change classes также недостаточен и не оправдывает coefficient sweep.
Exp159 подтвердил, что independent analytic target совместим с audited raw
deltas; exp160 показал, что текущие gate features не выучивают этот target до
поведенческого порога даже при прямой supervision. Exp161 подтвердил ценность
hidden state для aggregate regression, но обе linear input модели провалили
critical one-step gate. Exp162 подтвердил nonlinear decodability contact effect,
но blocked-noop physics осталась нерешённой. Exp163 показал, что per-action
threshold tradeoff не разделяет contact и no-op contexts. Exp164 дал небольшое
continuous improvement от position-only relations, но categorical no-op wall
остался. Exp165 добавил pose/orientation и не изменил categorical failures;
следующий вопрос перенесён с input completeness на transition target. Exp166
снял большинство blocked failures hurdle target-ом, но потерял contact и не
прошёл общий gate. Exp167 локализовал causal error в обоих learned hurdle
components; only-oracle composition проходит source и unseen one-step gate.
Exp168 снял contact failure direct vector target-ом, но exact blocked no-op
остаётся открытым. Exp169 впервые прошёл local source transition gate, но
unseen transfer остался неполным при unchanged vector.

**Why this is architectural, not tactical:** механизм описывается без названия
среды, но его общность ещё не доказана экспериментально. Специальных правил
решения Crafter в planner не добавляли; fixture не считается generalization.

**Knowledge flow outcome:** веса и реальные эпизоды сохраняются; причинная
полезность этого знания и выигрыш следующего поколения пока не установлены.

**Remaining assumptions / walls:** см. выше. Learned temporal metric полезна на
одном знакомом графе, но проваливает часть цветов и ещё не проверена на новых
layout/start/goal. Не масштабировать AGI-кампанию до этой проверки.

**Decision:** `PARTIAL`; локальные механизмы подтверждены, перенос не подтверждён.
