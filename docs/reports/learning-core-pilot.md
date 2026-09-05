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

## Stage Review

**Ideological debt addressed:** отсутствие обучаемой динамики и переносимого
опыта; механическая часть добавлена, доказательный долг не закрыт.

**Layer changed:** `mechanisms`, `experience`, `stimuli` (явные GoalSpec).
Fixture и task facts остаются в среде/описании задач, не в generic policy.

**What changed:** encoder, recurrent ensemble, multi-step training, real replay,
bounded planner и frozen evaluation с контрольными условиями.

**Evidence of improvement:** исполняемость подтверждена; полезного улучшения
поведения и action-conditioned прогноза в пилоте нет.

**Why this is architectural, not tactical:** механизм описывается без названия
среды, но его общность ещё не доказана экспериментально. Специальных правил
решения Crafter в planner не добавляли; fixture не считается generalization.

**Knowledge flow outcome:** веса и реальные эпизоды сохраняются; причинная
полезность этого знания и выигрыш следующего поколения пока не установлены.

**Remaining assumptions / walls:** см. выше. Ближайший вопрос — почему predictor
не использует действия достаточно для полезного прогноза, а не увеличение
масштаба AGI-кампании или настройка evaluator для получения PASS.

**Decision:** `PARTIAL` для реализации; гипотеза полезного обучения не подтверждена.
