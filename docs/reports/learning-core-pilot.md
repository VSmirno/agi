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

Артефакты: `output_to_user/core/action-confusion-*`,
`transfer-push1-random64-u1000-finalplanner-001`,
`transfer-push1-salient-u1000-001`, `exp143-temporal-proximity-002..010`,
`exp144-layout-generalization-001..006`, `exp144-hindsight-001`,
`exp144-terminal-hindsight-001..003`, `exp144-random-encoder-001..003`.
Predictive ablation: `exp144-predictive-encoder-001..003`.

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
0/6 raw и 0/6 shuffled на одном Push fixture. Стабильного transfer improvement
всё ещё нет.

**Why this is architectural, not tactical:** механизм описывается без названия
среды, но его общность ещё не доказана экспериментально. Специальных правил
решения Crafter в planner не добавляли; fixture не считается generalization.

**Knowledge flow outcome:** веса и реальные эпизоды сохраняются; причинная
полезность этого знания и выигрыш следующего поколения пока не установлены.

**Remaining assumptions / walls:** см. выше. Learned temporal metric полезна на
одном знакомом графе, но проваливает часть цветов и ещё не проверена на новых
layout/start/goal. Не масштабировать AGI-кампанию до этой проверки.

**Decision:** `PARTIAL`; локальные механизмы подтверждены, перенос не подтверждён.
