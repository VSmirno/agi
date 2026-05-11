# SNKS — Система Непрерывного Когнитивного Синтеза

> Исследовательская архитектура агентов общего назначения, обучающихся непрерывно, с HDC/SDM-памятью и символическим top-down планировщиком — не LLM, не deep RL.

🇬🇧 [English version](README.md)

[![status](https://img.shields.io/badge/stage-91%20closed%20%E2%80%A2%20Variant%20B%20landed-blue)](docs/architecture-report-2026-05-11.md)
[![env](https://img.shields.io/badge/env-Crafter-green)](https://github.com/danijar/crafter)
[![determinism](https://img.shields.io/badge/eval-byte%20deterministic-success)](src/snks/agent/crafter_pixel_env.py)
[![tests](https://img.shields.io/badge/agent%20tests-89%2F89%20passing-brightgreen)](tests/agent)
[![license](https://img.shields.io/badge/license-TBD-lightgrey)](#лицензия)

---

## Зачем этот проект

Современные ИИ-системы — LLM, deep-RL агенты — это *функциональные
аппроксиматоры*. Им нужны огромные датасеты, они страдают от catastrophic
forgetting, и «думают» через генерацию токенов, а не через оперирование
смыслами.

SNKS — исследовательская линия о том, может ли иной субстрат — **фазово
связанная, sparse-distributed, символически-bootstrapped** когнитивная
система — дать general-агента, который:

- Учится *непрерывно*, каждый ход, без отдельной batch-фазы.
- Стартует с малого **текстбука** явных фактов и уточняет их из опыта.
- Рассматривает *восприятие*, *модель мира*, *цель* и *мотивацию* как четыре независимых слоя над общей субстратной памятью.
- Масштабирует один и тот же агент с 64×64 игровой среды на существенно более богатые миры без переписывания.

Целевое состояние — **Phase-Coupled Cognitive Substrate** (PCCS): один общий
HDC-вектор, в который перцепция, память, стимулы, цель и post-mortem
одновременно пишут и читают каждый тик, с фазовой когерентностью Kuramoto
как binding-механизмом. Текущая кодовая база Stage 91 содержит примерно
половину этого субстрата как живую, оттестированную инфраструктуру; остальное
— план на ближайший год.

---

## Четыре слоя (идеология в одном экране)

Полный документ идеологии — [`docs/IDEOLOGY.md`](docs/IDEOLOGY.md). Кратко:

```
┌────────────────────────────────────────────────────────────────┐
│  FACTS         configs/crafter_textbook.yaml                   │  ← пишет человек
│  «do near tree → wood +1», «lava range=0 → health -1»          │
├────────────────────────────────────────────────────────────────┤
│  MECHANISMS    src/snks/agent/                                 │  ← алгоритмы
│  perceive → spatial_map → planner → sim → score → act          │
├────────────────────────────────────────────────────────────────┤
│  EXPERIENCE    runtime, per-step                               │  ← агент видит
│  SDM-память, spatial_map, surprise accumulator, death log      │
├────────────────────────────────────────────────────────────────┤
│  STIMULI       src/snks/agent/stimuli.py                       │  ← «зачем»
│  SurvivalAversion, Homeostasis, Curiosity                      │
└────────────────────────────────────────────────────────────────┘

Сверху вниз: знание стабилизируется, становится «дешевле» (статичнее).
Снизу вверх: experience очищается до facts через promotion.
```

Каждый слой заменяем независимо. Новая среда требует замены **только
текстбука**; mechanisms, experience и stimuli переиспользуются. Это и
означает «масштабируется на другие задачи» — а не «отдельная кодовая база
под каждый env».

---

## Текущее состояние (Stage 91 + Variant B)

Эталонный benchmark — среда [Crafter](https://github.com/danijar/crafter),
eval под строгим CUDA-детерминизмом (byte-identical эпизоды при фиксированном
seed).

### Что агент делает сегодня

| Возможность | Источник | Состояние |
|---|---|---|
| Pixel-перцепция (CNN tile segmenter, 64×64 → 7×9 viewport) | `tile_segmenter`, `decode_head` | Pretrained, стабильно |
| Символическая перцепция (semantic ground truth) | `perception.py` | Канонический eval-путь |
| HDC/SDM world model (binary vectors, dim=16384, 50000 sparse locations) | `vector_world_model.py` | Bootstrap из textbook, online-learn в эпизоде |
| Символический MPC планнер с motion + crafting chains | `vector_mpc_agent.py` | Генерирует make/place планы, навигирует к существующим placed-тайлам |
| Goal selector, derived from textbook threats | `goal_selector.py` | Weapon-aware: переключает `fight_X` → `craft_<weapon>` если нет оружия |
| Emergency safety controller | `stage90r_emergency_controller.py` | Независимая оценка угроз, может override planner'а |
| Frozen advisory neural actor (`.pt`) | `stage90r_local_model.py` | Даёт небольшой bonus к ranking'у |
| Spatial cognitive map с placed-object памятью | `crafter_spatial_map.py` | Placed-object writes authoritative над stale "empty" |
| Post-mortem damage attribution | `post_mortem.py` | Логирует причину смерти; feedback в planning — следующий milestone |

### Эталонный benchmark — seed 17 ep 0, full-profile, strict determinism

Последний записанный эпизод после Variant B (коммит `7829711`):

```
episode_steps  : 147
death_cause    : skeleton
productive_do  : 47   (early-game wood / water / cow gathering)
place_table    : 2 успешных (steps 23, 47)
make_wood_sword: 1 (step 48 — первый успешный крафт оружия в этой линии работ)
sleep flicker  : 0 (gated on energy < 3)
```

Variant B — стек из 12 фиксов, который end-to-end связал adjacency-правило
из textbook (`make_wood_pickaxe near: table`) с реальным поведением: новое
поле `near_requirements` в world model, chain-планы `[place_X → make_Y]`
только когда экземпляра требуемого тайла нет на карте, top-band RNG fallback
чтобы редкие crafting-планы не размывались среди десятков мотoring-планов с
тем же баллом, и placed-object override в spatial_map — запись
`(28, 33) → "empty"` с conf=1.0 больше не блокирует последующую запись
`(28, 33) → "table"` с той же confidence. Каждый фикс валидировался на
single-seed видео до выкатки.

Полный список фиксов и архитектурный audit, из которого они выросли:
[`docs/architecture-report-2026-05-11.md`](docs/architecture-report-2026-05-11.md).

---

## Архитектура в одном взгляде

```
env.step → info["semantic"]
   │
   ▼
PERCEPTION  (perceive_semantic_field, _update_spatial_map, _update_spatial_map_hazards)
   │
   ▼
WORLD MODEL  (VectorWorldModel = CausalSDM; vector_sim.simulate_forward rollouts)
   │
   ▼
GOAL SELECTOR  (символический, derived from textbook threats)
   │
   ▼
PLAN GENERATION  (motion + chain + single:target:do + craft chains)
   │
   ▼
SIMULATE + SCORE  (simulate_forward(plan) → score_trajectory(stimuli, goal))
   │
   ▼
RANK + RESCUE  (EmergencySafetyController, learner-actor advisory)
   │
   ▼
ACT  (env.step(primitive))
   │
   ▼
LEARN  (model.learn(target, action, observed_delta) + spatial_map.update)
```

Символический MPC поверх HDC/SDM ассоциативной памяти. «Learned»-часть — один
advisory `.pt` actor, дающий бонус к ranking'у; при `actor_share=0` система
работает как чистый символический MPC агент над HDC world model.

### Живая поверхность (≈20 файлов из 71)

```
configs/crafter_textbook.yaml          ← FACTS — читать первым
docs/IDEOLOGY.md                       ← философия (четыре категории)
docs/architecture-report-2026-05-11.md ← отчёт о текущем состоянии

src/snks/agent/
├── crafter_pixel_env.py               (env wrapper, determinism patch)
├── crafter_textbook.py                (YAML loader)
├── crafter_spatial_map.py             (cognitive map + placed-object override)
├── perception.py                      (info["semantic"] → VisualField)
├── vector_world_model.py              (SDM — главная learned-без-градиентов память)
├── vector_bootstrap.py                (textbook → SDM seeding)
├── vector_sim.py                      (forward rollout)
├── vector_mpc_agent.py                (main per-step orchestration)
├── goal_selector.py                   (threat-priority derivation)
├── stage90r_emergency_controller.py   (safety override layer)
├── stage90r_local_model.py            (advisory neural actor)
├── stimuli.py                         (Survival, Homeostasis, Curiosity)
└── post_mortem.py                     (damage attribution)
```

Остальное в `src/snks/` — `daf/` (Dynamic Attractor Fields), `dcam/`,
`gws/` (Global Workspace), `metacog/`, `encoder/` (oscillator-based) —
честная инфраструктура из ранних стадий, ждущая интеграции в субстрат.
Около 25% репозитория dormant by design; четырёхкатегорная идеология
называет это *substrate*-слой, который активирует следующий milestone.

---

## Детерминизм

Stage 91 закрыл многомесячное расследование нондетерминизма. Eval-стек
теперь byte-identical между запусками при фиксированном seed:

- `crafter.env.Env._balance_chunk` патч сортирует объекты по `(pos, type)`
  до итерации (порядок итерации `set` в Crafter зависит от `id(obj)`,
  которое `PYTHONHASHSEED` не покрывает).
- `CausalSDM._calibrate_radius` оффлоадит `kthvalue` на CPU
  (CUDA `kthvalue` не имеет детерминистической реализации в torch 2.5.1+cu121).
- Обязательные env-переменные для eval:
  ```
  CUDA_VISIBLE_DEVICES=0
  CUBLAS_WORKSPACE_CONFIG=:4096:8
  PYTHONHASHSEED=0
  ```

---

## Roadmap

Краткосрочно (недели):

1. **Episodic Substrate Snapshots** — на каждой точке решения бандлим
   visible scene, inventory, body state, near concept, active goal и
   выбранный plan origin в HDC-вектор, пишем в персистентный episodic SDM,
   при следующем планировании делаем similarity-query — получаем
   дополнительный стимул `EpisodicMemoryStimulus`. Первое cross-episode
   обучение, не требующее новых правил в textbook.
2. **Persistent placed-object memory** — placed-tables/furnaces переживают
   уход из viewport, не приходится re-observe каждый раз.
3. **Emergency-safety craft override** — если goal=`craft_<weapon>` и
   предсказанный craft в одном ходу — не позволять `EmergencySafetyController`
   уводить агента в побег.

Среднесрочно (месяцы):

4. **TextbookPromoter activation** — закрытие петли *experience → facts*.
   Death hypotheses, повторяющиеся между эпизодами, promote'ятся в
   `promoted_hypotheses.yaml`, читаются следующим поколением агента при
   bootstrap.
5. **Замена lex-tuple scoring на substrate decode** — заменить кортеж
   `(base, goal_prog, known, -steps)` на single similarity-based readout
   из HDC-substrate'а, в который пишут все компоненты. Это устраняет
   `baseline-wins-ties → RNG fallback` failure mode end-to-end.
6. **Wire Kuramoto sync в substrate binding** — заменить XOR-bind по
   role-vector'ам фазовой когерентностью (`tests/test_kuramoto_sync.py`
   уже верифицирует примитив).

Долгосрочно (год):

7. **Phase-Coupled Cognitive Substrate (PCCS)** как orchestration-слой для
   всех четырёх категорий идеологии. Перцепция, world model, goal selector,
   stimuli, post-mortem и episodic memory становятся pluggable
   substrate-writers/readers. Решения emergent из attractor-state субстрата,
   не из фиксированного lex-tuple ranking'а. Cross-task transfer —
   «поменять textbook и substrate-role vocabulary; переиспользовать всё
   остальное».

Архитектурный audit объясняет *почему* каждый пункт в roadmap'е и куда в
коде он целит:
[`docs/architecture-report-2026-05-11.md`](docs/architecture-report-2026-05-11.md).

---

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Тестовый набор агента (89 тестов)
pytest tests/agent tests/encoder tests/learning tests/metacog tests/gws -q

# Запись single-seed видео с perception overlay
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
PYTHONPATH=src:experiments python experiments/record_stage91_seed_video.py \
  --seed 17 --episode-index 0 --full-profile \
  --local-evaluator path/to/stage90r_actor.pt \
  --out crafting_seed17_ep0.mp4
```

Для recorder'а нужен pretrained local actor checkpoint — он лежит в
`_docs/` на dev-машине и не закоммичен в git.

---

## Документация

- [`docs/IDEOLOGY.md`](docs/IDEOLOGY.md) — четырёхкатегорная идеология полностью.
- [`docs/architecture-report-2026-05-11.md`](docs/architecture-report-2026-05-11.md) — что работает сейчас, что dormant, что дальше.
- [`SPEC.md`](SPEC.md) — полная спецификация системы (старая ревизия; идеология имеет приоритет при расхождениях).
- [`ROADMAP.md`](ROADMAP.md) — исторический roadmap через Stages 0–30; текущее направление — в architecture report.
- [`docs/reports/`](docs/reports/) — отчёты по завершённым стейджам.

---

## Лицензия

Лицензия пока не финализирована. Свяжитесь с мейнтейнером по поводу условий
применения.
