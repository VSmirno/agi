# Transferable Learning Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить и проверить ядро, которое учится на реальных переходах, использует прогнозы для управления и переносит полезный опыт между Crafter и CausalGridWorld.

**Architecture:** Изолированный observation contract, компактный CNN, recurrent action-conditioned dynamics, replay исходных эпизодов и bounded beam-search MPC. End-to-end кандидат и fixed-representation контроль существуют отдельно; знание не подменяется готовой Crafter-политикой. Реализация размещается внутри существующих пакетов `snks`, legacy остаётся контрольным вариантом.

**Tech Stack:** Python 3.11+, PyTorch, NumPy, pytest, PyYAML, Gymnasium/MiniGrid, Crafter; JSON/NPZ для эпизодов и manifests, PyTorch state dictionaries для доверенных локальных checkpoints. Новая обязательная foundation model не вводится.

**Spec:** [2026-09-05-transferable-learning-core-design.md](../specs/2026-09-05-transferable-learning-core-design.md). Исполнитель читает спеку и план вместе. Пользователь подтвердил письменную спеку запросом на этот план.

## Global Constraints

**Execution amendment (2026-09-05, latest user instruction):** implement in parallel, prioritizing the smallest end-to-end hypothesis test. Batch Tasks 1–16 into environment, model, experience and integration work; keep a few behavioral tests and one consolidated review. Defer broad campaign/approval automation until the pilot demonstrates it is useful. Preserve observation isolation, equal-budget controls, frozen evaluation and honest negative results. All runtime checks remain on HyperPC. The detailed task list below is the research roadmap, not a requirement to build every supporting abstraction before the first experiment.

- «Разработка выполняется локально. Тесты, обучение и поведенческие эксперименты выполняются на HyperPC.»
- «Новые action embeddings и новые поля сенсоров инициализируются одинаково во всех сравниваемых ветках B; общий обученный backbone переносится без изменения архитектуры.»
- «Evaluation reward среды не передаётся planner или trainer; evaluator использует его только как метрику там, где это объявлено.»
- «Первый planner работает с конечным набором примитивных действий и ограниченным горизонтом.»
- «Минимум для confirmatory campaign — 5 независимых training seeds»; «Для финальных эффектов используются 95% интервалы».
- «Manifest регистрируется отдельным commit до открытия sealed test и просматривается пользователем перед длительным запуском.»
- «PASS всей версии требует G0–G6»; готовность к запуску и положительный научный результат — разные результаты.
- Python 3.11+; новые эксперименты `expNN_description.py`; код внутри ближайших существующих `src/snks/*`, тесты внутри `tests/`.
- Пользовательские untracked-файлы не удалять, не включать в коммиты и не использовать как неявные зависимости. Исходники runtime не изменялись при написании этого плана.

---

## Исполнение, ветка и единица проверки

Исходная ветка: `feature/stage9x-sparse-option-failures`; утверждённая спека: commit `f7c5448`; legacy implementation baseline: `e6799fbfd45f9c3baeae3e58817503ec6bcac587`. Не сравнивать новый агент со старым случайным рабочим каталогом HyperPC.

Перед реализацией использовать `superpowers:using-git-worktrees`, если требуется новый изолированный checkout. Сейчас worktree не создаётся. Каждая Task заканчивается самостоятельно проверяемым результатом и отдельным scoped commit. Зависимости перечислены явно; задачи не исполняются по порядку номеров, если их prerequisites не завершены.

Все `Run` ниже выполняются **на HyperPC** из checkout соответствующего commit. `.venv-core/bin/python` — интерпретатор, проверенный в Task 1, а не предположение о существующем host env. Локально выполняются правки и git-команды. Для RED и GREEN перед передачей на HyperPC создавать отдельный commit только перечисленных файлов; RED-коммиты с намеренно падающими тестами не являются release baseline. Канал передачи исходников выбирается из существующего доступа пользователя; push и изменение общей ветки не подразумеваются этим документом. Remote checkout должен получать тот же commit, каким бы способом он ни был передан.

Для каждого тестового Run сохранять `git rev-parse HEAD`, команду, exit code и log. Шаблон локального снимка (передавать только файлы текущей Task):

```bash
core_snapshot() {
    local message="$1"
    shift
    git add -- "$@" || return
    git diff --cached --check -- "$@" || return
    git commit --only -m "$message" -- "$@" || return
    git rev-parse HEAD
}
```

Если неудача вызвана отсутствием зависимости, неправильным import path или другим тестом, это не требуемый RED. Исправить протокол и повторить. В GREEN недостаточно запуска только одного показанного теста: запускать весь файл Task и указанные regression suites.

Код в плане задаёт конкретные тесты, интерфейсы и ключевые реализации; это инструкции для будущих файлов, не уже существующие API. Импорты в snippets указывать из перечисленного модуля. Не внедрять весь план одним коммитом.

## Карта файлов и ответственности

| Создать | Ответственность | Tasks |
|---|---|---|
| `src/snks/pipeline/core_config.py`, `core_preflight.py` | Строгая конфигурация, происхождение запуска, проверка устройства | 1, 17 |
| `src/snks/env/core_types.py`, `core_adapter.py` | Типизированная граница наблюдения/действия без диагностик | 2, 3 |
| `src/snks/env/core_grid.py` | B-fixtures с частичным обзором и изменяемыми правилами | 4 |
| `src/snks/pipeline/core_tasks.py`, `core_splits.py` | Каталог целей/evaluator и разделение данных | 5 |
| `src/snks/learning/core_replay.py` | Replay исходных последовательностей | 6 |
| `src/snks/encoder/core_encoder.py` | Пространственные визуальные признаки | 7 |
| `src/snks/agent/core_world_model.py` | Состояние, recurrent dynamics и outcome heads | 8 |
| `src/snks/learning/core_objective.py`, `core_trainer.py` | SIGReg, multistep loss и обновления между эпизодами | 9, 10 |
| `src/snks/learning/core_checkpoint.py` | Совместимость, атомарное сохранение и восстановление | 11 |
| `src/snks/agent/core_cost.py`, `core_planner.py` | Независимая оценка цели и bounded beam search | 12 |
| `src/snks/agent/core_agent.py`, `src/snks/pipeline/core_runner.py` | Observe/act loop, read-only evaluation и audit | 13 |
| `src/snks/pipeline/core_metrics.py`, `core_oracle.py` | Prediction metrics, доверительные интервалы, privileged diagnostics | 14 |
| `src/snks/pipeline/core_controls.py`, `core_transfer.py` | Fixed-representation сравнения и A→B→A | 15, 16 |
| `src/snks/pipeline/core_campaign.py`, `core_interventions.py`, `core_reporting.py` | Регистрация/запуск, контрфактические проверки, независимая генерация отчёта | 17–20 |
| `configs/core_smoke.yaml`, `core_pilot.yaml` | Отдельные профили, не модификация `default.yaml` | 1, 17 |
| `experiments/exp138_learning_core.py` | Тонкий CLI к pipeline, без моделирования внутри скрипта | 17 |

На текущем checkout номер 138 свободен. Если при начале реализации он занят чужой работой, выбрать следующий свободный номер и одним отдельным documentation change обновить все ссылки и команды плана до исполнения.

Изменить только по необходимости: `pyproject.toml` (optional dependency group), `docs/ASSUMPTIONS.md` (новый долг), `docs/ROADMAP.md` (ссылка на новую проверку без переписывания старых итогов). `CausalGridWorld`, `EnvAdapter`, старый trainer и legacy planner по умолчанию не меняются; новые wrappers/fixtures не должны ломать прежние импорты.

## Контрольные точки и оценки

| Milestone | Завершённые deliverables | Условие перехода |
|---|---|---|
| M1 | Tasks 1–6 | Чистая граница данных, две среды, replay и разбиения |
| M2 | Tasks 7–11 | Модель обучается на синтетических переходах, snapshot восстанавливается |
| M3 | Tasks 12–16 | Сквозной агент и исполняемые контролируемые сравнения |
| M4 | Tasks 17–18 и development/registration часть Task 19 | Весь код готов; pilot, оценка ресурсов и одобренный manifest |
| M5 | Confirmatory часть Task 19 и Task 20 | Полный отчёт G0–G6, включая отрицательные результаты |

Владелец реализации каждой Task — один исполнитель; пользователь подтверждает ресурсный manifest и смену исследовательской гипотезы. После contracts encoder/model (Tasks 7–9) можно разрабатывать независимо от adapter/tasks/replay (Tasks 3–6); Task 4 использует adapter protocol Task 3. Общая GPU-очередь HyperPC ограничивает параллельные запуски. Сам план не запускает субагентов.

Грубая оценка инженерной работы: 100 часов оптимистично, 140 наиболее вероятно, 220 пессимистично; 25% резерв поверх наиболее вероятной оценки — около 175 часов. Это оценка подготовки и проверки реализации, не срок достижения AGI и не обещание успешных gates. Время GPU и исследовательские итерации после отрицательного результата оцениваются только в pilot. Критический путь: contracts → replay/model → trainer/checkpoint → control → ablations/transfer → pilot → registered evaluation.

### Task 1: Проверяемый профиль запуска и исходная точка

**Files:** Create `src/snks/pipeline/core_config.py`, `src/snks/pipeline/core_preflight.py`, `configs/core_smoke.yaml`, `tests/test_core_preflight.py`; Modify `pyproject.toml`, `docs/ASSUMPTIONS.md`.

**Interfaces:** Consumes `Path`, `torch`, YAML. Produces `CoreConfig` (строгая dataclass), `load_core_config(path: Path) -> CoreConfig`, `check_runtime(checkout: Path, expected_commit: str, require_cuda: bool) -> dict`.

**Depends on:** утверждённая спека. **Effort:** 4–6 часов.

- [ ] Написать тест, отвергающий неверный checkout и неизвестные ключи конфигурации:

```python
def test_rejects_wrong_checkout(tmp_path):
    import pytest
    from snks.pipeline.core_preflight import check_runtime
    with pytest.raises(RuntimeError):
        check_runtime(tmp_path, "not-a-commit", require_cuda=False)
```

- [ ] RED: ` .venv-core/bin/python -m pytest tests/test_core_preflight.py -q` — отсутствует модуль/API. Перед первым Run на HyperPC создать `.venv-core` из подтверждённого CUDA Python и установить пакет `-e '.[dev,learning-core]'`; optional group `learning-core = ["crafter"]` не меняет зависимости legacy. Сохранить `pip freeze`; при CPU-only PyTorch остановиться до обучения, не устанавливать произвольную CUDA-сборку вслепую.
- [ ] Реализовать preflight с проверкой пути и commit; добавить GPU name/VRAM, Python/PyTorch/CUDA versions и dependency export в возвращаемый отчёт:

```python
import subprocess
from pathlib import Path
import snks
import torch

def check_runtime(checkout: Path, expected_commit: str, require_cuda: bool) -> dict:
    root = checkout.resolve()
    origin = Path(snks.__file__).resolve()
    if not origin.is_relative_to(root / "src"):
        raise RuntimeError(f"wrong snks import: {origin}")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != expected_commit:
        raise RuntimeError("commit mismatch")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this profile")
    return {"commit": actual, "import_root": str(origin),
            "torch": torch.__version__, "cuda": torch.version.cuda}
```

- [ ] Создать строгий `CoreConfig`: `profile`, `device`, `seed`, `z_dim`, `h_dim`, `ensemble_size`, `batch_size`, `burn_in`, `train_horizon`, `planner_horizon`, `beam_width`, `max_model_calls`, `replay_capacity`, `recent_fraction`, `learning_rate`, `sigreg_weight`, `sensor_weight`, `termination_weight`, `exploration_fraction`. Неизвестные поля вызывают `ValueError`. Smoke values: CPU, seed 0, z=64, h=32, ensemble=3, batch=4, burn_in=2, train_horizon=2, planner_horizon=3, beam=4, calls=128, replay=32, recent=0.5, lr=0.001, loss weights 0.1/1/1, exploration=0.2. Это тестовый профиль, не итоговые гиперпараметры. В ASSUMPTIONS записать ограничения первой версии и отсутствие текущих PASS claims.
- [ ] GREEN: весь `tests/test_core_preflight.py`; также проверить manifest против реально импортируемого source. Зафиксировать `feat(core): add reproducible execution profile`. **Done:** неизвестная конфигурация, wrong import и wrong commit не запускают эксперимент.

### Task 2: Типы без неявного доступа к среде

**Files:** Create `src/snks/env/core_types.py`, `tests/test_core_types.py`, `tests/core_helpers.py`, `tests/__init__.py` (пустой package marker, чтобы `tests.core_helpers` разрешался в этот checkout).

**Interfaces:** Produces `Observation`, `ActionSpec`, `GoalSpec`, `Transition`, `Episode`, `Mode`; `make_observation(value: int = 0, sensor: float = 0.0, schema: str = "toy", step: int = 0) -> Observation` в test helper. `Mode` имеет `TRAIN`, `ADAPT`, `EVALUATE`. Завершение и schema actions передаются через `Transition` и `ActionSpec`, не через произвольный словарь Observation.

**Depends on:** 1. **Effort:** 4–6 часов.

- [ ] Написать контрактный тест:

```python
def test_observation_has_no_privileged_bag():
    from tests.core_helpers import make_observation
    obs = make_observation()
    assert obs.rgb.shape == (3, 64, 64)
    assert not hasattr(obs, "info")
    assert not hasattr(obs, "env")
    assert not hasattr(obs, "ruleset")
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_types.py -q` — отсутствуют новые типы/helper.
- [ ] Реализовать dataclasses с `slots=True`; `Observation` содержит только `rgb: np.ndarray` (CHW uint8), `sensors: np.ndarray` (float32), `sensor_mask: np.ndarray` (bool), `schema: str`, `step: int`. Конструктор проверяет размеры, конечность присутствующих сенсоров, копирует массивы и ставит `writeable=False`. Нет `info`, reward или task/seed metadata. `ActionSpec(schema: str, names: tuple[str, ...])` задаёт число actions через `len(names)`; имена не используются для ветвления policy. Остальные контракты:

```python
from dataclasses import dataclass
from enum import Enum

class Mode(Enum):
    TRAIN = "train"
    ADAPT = "adapt"
    EVALUATE = "evaluate"

@dataclass(frozen=True, slots=True)
class GoalSpec:
    image: Observation | None
    ranges: dict[int, tuple[float, float]]

@dataclass(frozen=True, slots=True)
class Transition:
    before: Observation
    action: int
    after: Observation
    terminated: bool
    truncated: bool

@dataclass(frozen=True, slots=True)
class Episode:
    uid: str
    split: str
    family: str
    ruleset: str
    transitions: tuple[Transition, ...]
```

- [ ] В helper создать наблюдение `np.full((3,64,64), value, np.uint8)`, сенсор `[sensor]`, маску `[True]`, шаг 0. Добавить тесты запрещённых NaN/shape, отсутствующий сенсор против нулевого, копирование входных массивов и несовпадающие action schemas.
- [ ] GREEN: `.venv-core/bin/python -m pytest tests/test_core_types.py -q`. Commit `feat(core): define observation and episode contracts`. **Done:** trainer сможет принимать tensors без metadata, evaluator сохранит provenance отдельно.

### Task 3: Crafter boundary и честный reset

**Files:** Create `src/snks/env/core_adapter.py`, `tests/test_core_crafter_adapter.py`.

**Interfaces:** Consumes Task 2. Produces `CoreAdapter` protocol: `reset(seed: int) -> Observation`, `step(action: int) -> Transition`, `actions: ActionSpec`, `reset_transitions: int`, `close() -> None`; `CrafterCoreAdapter(sensor_names: tuple[str, ...])`. Отдельный evaluator-only protocol `DiagnosticAdapter(CoreAdapter)` добавляет `diagnostic_snapshot() -> dict` с state/RNG digest; оба production adapters реализуют его. Agent не получает ни один adapter, только Observation/Transition.

**Depends on:** 2. **Effort:** 4–8 часов.

- [ ] Написать тест allowlist, не создающий полного тяжёлого агента:

```python
def test_inventory_only_projection():
    import numpy as np
    from snks.env.core_adapter import project_crafter_observation
    info = {"inventory": {"health": 7}, "semantic": "secret",
            "player_pos": (12, 9), "seed": 123}
    obs = project_crafter_observation(np.zeros((64,64,3), np.uint8),
                                     info, ("health", "wood"), 0)
    assert obs.sensors.tolist() == [7.0, 0.0]
    assert obs.sensor_mask.tolist() == [True, False]
    assert not hasattr(obs, "player_pos")
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_crafter_adapter.py -q`.
- [ ] Реализовать `project_crafter_observation(rgb: np.ndarray, info: dict, names: tuple[str, ...], step: int) -> Observation`:

```python
inventory = info.get("inventory", {})
values = np.asarray([inventory.get(k, 0.0) for k in names], dtype=np.float32)
mask = np.asarray([k in inventory for k in names], dtype=bool)
return Observation(np.moveaxis(rgb, -1, 0), values, mask, "crafter-v1", step)
```

- [ ] Adapter владеет нативной средой, но `CoreAgent` не получает adapter. На reset использовать свежий seed и реально полученные данные. Если нужен noop для первого inventory, `reset_transitions=1` и этот переход списывается из бюджета до первого решения; не учитывать его как действие planner. Не читать semantic map для sensor_mask или action availability. Синтаксически доступны все native action IDs. Нативный done и внешний лимит развести в `terminated`/`truncated`, отдельно проверить состояние смерти на последнем разрешённом шаге. Старую `CrafterPixelEnv.reset` не переписывать ради нового профиля.
- [ ] GREEN: новый файл и `tests/test_crafter_pixel_env_67.py`. Commit `feat(core): isolate Crafter observations and step accounting`. **Done:** лишние поля info не меняют вход/маски; reset overhead не создаёт бесплатного опыта.

### Task 4: B-fixtures с occlusion и различными правилами

**Files:** Create `src/snks/env/core_grid.py`, `tests/test_core_grid.py`; Modify `src/snks/env/core_adapter.py`.

**Interfaces:** Produces `GridRules(consume_key: bool, push_distance: int)`, `CoreGridWorld(family: str, rules: GridRules, seed: int)`, `GridCoreAdapter(world: CoreGridWorld)` implementing Task 3. Допустимые family: `door_key`, `push_box`; push_distance: 1 или 2. `diagnostic_snapshot() -> dict` доступен только evaluator.

**Depends on:** 2, 3. **Effort:** 6–8 часов.

- [ ] Добавить проверку исполнимой семантики ключа и однократного эффекта:

```python
def test_consumable_key_is_removed_after_unlock():
    from snks.env.core_grid import CoreGridWorld, GridRules
    world = CoreGridWorld("door_key", GridRules(True, 1), seed=0)
    world.reset_for_test("carrying_key_facing_locked_door")
    world.step(3)
    snap = world.diagnostic_snapshot()
    assert snap["door_open"]
    assert not snap["carrying"]
    world.close()
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_grid.py -q`.
- [ ] `CoreGridWorld` переиспользует типы MiniGrid и генерацию геометрии CausalGrid, но имеет отдельный native step dispatch. `reset_for_test(name: str) -> None` создаёт evaluator-only состояния: указанное выше, `object_facing_wall`, `occluded_object`, `clear_push_lane`. Реализовать ровно одно применение действия, не делать custom interact, затем повторный native toggle:

```python
# В обработке открытия Door после проверки реально переносимого ключа:
door.is_locked = False
door.is_open = True
if self.rules.consume_key:
    self.carrying = None
# В обработке push разрешить движение только после проверки всех клеток пути.
```

- [ ] Добавить параметризованные тесты: reusable/consumed key; push 1/2 при свободном пути и блокировании; noop обновляет время/фоновые процессы, но не движение игрока; RGB не меняется при изменении объекта за непрозрачной стеной; информация о переносимом предмете доступна только как occupancy sensor. Новый factory использует partial RGB wrapper и `see_through_walls=False`, не существующий `make_level()` с полным RGB. Вариант правил остаётся внутри среды. ActionSpec во всех B-вариантах одинаков: turn_left, turn_right, forward, interact, noop; interact подбирает доступный ключ, открывает дверь или толкает объект в рамках физики среды, не стратегии агента.
- [ ] GREEN: `.venv-core/bin/python -m pytest tests/test_core_grid.py tests/test_env.py -q`. Commit `feat(core): add partially observed transfer fixtures`. **Done:** B действительно имеет различающиеся правила, а baseline CardGate не заменяет эту проверку.

### Task 5: Цели, каталог задач и sealed splits

**Files:** Create `src/snks/pipeline/core_tasks.py`, `core_splits.py`, `tests/test_core_tasks.py`, `tests/test_core_splits.py`; Create `configs/core_tasks.yaml`.

**Interfaces:** frozen dataclass `TaskCase(uid: str, family: str, ruleset: str, seed: int, split: str, goal: GoalSpec, max_steps: int)`; `build_case(config: dict) -> TaskCase`; `resolve_goal(case: TaskCase, initial: Observation) -> GoalSpec`; `score_episode(case: TaskCase, audit: list[dict]) -> bool`; `SplitRegistry(records: list[dict])`, `assert_readable(uid: str, mode: Mode) -> None`, `assert_disjoint() -> None`.

**Depends on:** 3, 4. **Effort:** 4–8 часов.

- [ ] Написать тест запрета test data в train:

```python
def test_training_cannot_read_test_episode():
    import pytest
    from snks.env.core_types import Mode
    from snks.pipeline.core_splits import SplitRegistry
    registry = SplitRegistry([{"uid": "e1", "split": "test"}])
    with pytest.raises(PermissionError):
        registry.assert_readable("e1", Mode.TRAIN)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_tasks.py tests/test_core_splits.py -q`.
- [ ] Разрешения реализовать явно, а не через отсутствие запрещённого флага:

```python
allowed = {
    Mode.TRAIN: {"train"},
    Mode.ADAPT: {"adapt"},
    Mode.EVALUATE: {"validation", "test", "zero_shot_test"},
}
if self.by_uid[uid]["split"] not in allowed[mode]:
    raise PermissionError(f"split denied: {uid}")
```

- [ ] Каталог фиксирует A-families `resource_acquisition` (наблюдаемый wood относительно начального значения) и `resource_recovery` (наблюдаемый drink относительно начального). Для этих двух A-families `case.goal.ranges` задаёт относительный прирост: `resolve_goal` прибавляет начальный sensor к обеим границам интервала, возвращает новый абсолютный GoalSpec; предел сверху — объявленный максимум сенсора. Для остальных families цель уже абсолютная. Это task compilation в evaluator, не ветвление внутри policy. Runner один раз вызывает `resolve_goal` после reset. Success требует фактического прироста 1 до termination и проверяется относительно сохранённого initial sensor. Условия старта имеют доступный локальный источник, но не передают маршрут агенту. B-families: открыть объявленную дверь и пройти в целевую клетку; переместить box в объявленное локальное целевое положение. B получает goal image, evaluator использует ground truth. Задачи частично наблюдаемы и подбираются по решаемости в development, не по успеху обучаемой модели. Goal images — локальные шаблоны желаемого состояния, без карты текущего эпизода.
- [ ] Для resource_recovery задать в task generator начальный drink ниже максимума; любую подготовительную нативную последовательность учитывать в reset_transitions. Проверять решаемость объявленной цели до выбора learnable agent; невозможный прирост при насыщенном sensor не запускать как тест обучения.
- [ ] Разделить целые эпизоды и карты; правила B-adapt и B-test могут совпадать при разных эпизодах — это few-shot оценка. Отдельные `zero_shot_test` rulesets не встречаются в adapt. Не использовать task UID, family, ruleset или seed в model input, включая хеши/filename. Проверить пересечение content hashes и запланированных episode keys до materialization.
- [ ] GREEN: оба новых файла. Commit `feat(core): define task goals and sealed data partitions`. **Done:** objective success отделён от GoalSpec/cost и метаданные не становятся обучающими признаками.

### Task 6: Replay реальных последовательностей

**Files:** Create `src/snks/learning/core_replay.py`, `tests/learning/test_core_replay.py`.

**Interfaces:** Consumes `Episode`, `Transition`, `Mode`, `SplitRegistry`. Produces `SequenceReplay(capacity: int, seed: int)`, `append(episode: Episode, mode: Mode) -> None`, `sample(batch_size: int, length: int, burn_in: int, recent_fraction: float) -> list[Episode]`, `manifest() -> dict`, `save(path: Path) -> None`, `load(path: Path) -> SequenceReplay`.

**Depends on:** 2, 5. **Effort:** 6–8 часов.

- [ ] Написать тест неизменности replay на evaluate:

```python
def test_evaluation_cannot_change_replay():
    import pytest
    from snks.env.core_types import Mode, Episode
    from snks.learning.core_replay import SequenceReplay
    replay = SequenceReplay(capacity=4, seed=1)
    before = replay.manifest()
    with pytest.raises(PermissionError):
        replay.append(Episode("e", "test", "f", "r", ()), Mode.EVALUATE)
    assert replay.manifest() == before
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/learning/test_core_replay.py -q`.
- [ ] Хранить recent ring и reservoir старых эпизодов, ограничивать capacity в эпизодах и отдельно bytes. Выборка reservoir:

```python
self.seen += 1
if len(self.old) < self.old_capacity:
    self.old.append(episode)
else:
    slot = self.rng.randrange(self.seen)
    if slot < self.old_capacity:
        self.old[slot] = episode
```

- [ ] Входы только TRAIN/ADAPT и только реальные переходы. Исключить дубликат одного uid одновременно в old/recent sampling. Сохранять CHW uint8, sensor arrays, masks, episode/step offsets в NPZ с `allow_pickle=False` при чтении и JSON manifest с SHA-256; не сохранять environment object. Для windows длиной burn_in+length не пересекать terminal/reset/schema. Короткие окна возвращать с явной valid mask на этапе tensorization; padded entries не входят в loss. Embeddings в первом варианте вообще не кэшировать, чтобы не вводить ненужную invalidation-систему.
- [ ] GREEN: новый файл и `tests/test_transition_buffer.py`. Проверить seed reproducibility, reservoir retention, отсутствие пересечения эпизодов, corrupt hash rejection и невозможность записать imagined transitions. Commit `feat(core): persist bounded real-sequence replay`. **Done:** старый опыт не вытесняется только последним эпизодом, test data нельзя добавить в replay.

### Task 7: Компактное пространственное представление

**Files:** Create `src/snks/encoder/core_encoder.py`, `tests/encoder/test_core_encoder.py`.

**Interfaces:** `CoreEncoder(z_dim: int)`; `forward(rgb: Tensor[B,3,64,64]) -> Tensor[B,z_dim]`. RGB input — float32 [0,1]. Нет sensor, reward, goal или task ID на входе encoder.

**Depends on:** 2. **Effort:** 4–6 часов.

- [ ] Написать тест формы и градиентов:

```python
def test_encoder_preserves_batch_and_backprop():
    import torch
    from snks.encoder.core_encoder import CoreEncoder
    net = CoreEncoder(z_dim=64)
    pixels = torch.rand(2, 3, 64, 64, requires_grad=True)
    z = net(pixels)
    assert z.shape == (2, 64)
    z.square().mean().backward()
    assert torch.isfinite(pixels.grad).all()
    assert pixels.grad.abs().sum() > 0
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/encoder/test_core_encoder.py -q`.
- [ ] Реализовать минимальный профиль без последней L2/LayerNorm, которая принудительно ограничивает распределение для Gaussian regularization:

```python
import torch.nn as nn

class CoreEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 2, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, z_dim),
        )

    def forward(self, rgb):
        return self.layers(rgb)
```

- [ ] Добавить тесты конечности, повтора при `eval()`, замороженных параметров и различия одинакового паттерна в разных местах кадра. Последний проверяет пространственную чувствительность конструкции, не семантическое понимание. Старый CNN с near/tile supervised heads не импортировать в новый профиль.
- [ ] GREEN: новый файл и `tests/test_perception_encoders.py`. Commit `feat(core): add compact spatial encoder`. **Done:** encoder независимо тестируется и заменяется через один интерфейс.

### Task 8: Recurrent action-conditioned dynamics

**Files:** Create `src/snks/agent/core_world_model.py`, `tests/agent/test_core_world_model.py`.

**Interfaces:** `LatentState(z: Tensor, sensors: Tensor, sensor_mask: Tensor, hidden: Tensor, schema: str)`; `Prediction(next_state: LatentState, terminated_prob: Tensor, uncertainty: Tensor, member_z: Tensor)`; `CoreWorldModel(encoder: CoreEncoder, schemas: dict[str, tuple[int,int]], h_dim: int, heads: int)`. Schema tuple = `(n_actions, n_sensors)`. Methods: `initial(obs: Observation) -> LatentState`; `initial_from_tensors(rgb: Tensor, sensors: Tensor, mask: Tensor, schema: str) -> LatentState`; `step(state: LatentState, actions: Tensor) -> Prediction`; `rollout(state: LatentState, actions: Tensor[B,H]) -> list[Prediction]`. Все prediction tensors имеют batch axis.

**Depends on:** 2, 7. **Effort:** 6–8 часов.

- [ ] Написать тест без скрытого состояния среды:

```python
def test_actions_change_prediction_without_mutating_root():
    import torch
    from tests.core_helpers import make_observation
    from snks.encoder.core_encoder import CoreEncoder
    from snks.agent.core_world_model import CoreWorldModel
    torch.manual_seed(7)
    model = CoreWorldModel(CoreEncoder(64), {"toy": (2, 1)}, 32, 3)
    root = model.initial(make_observation())
    saved = root.hidden.clone()
    left = model.step(root, torch.tensor([0]))
    right = model.step(root, torch.tensor([1]))
    assert torch.equal(root.hidden, saved)
    assert not torch.allclose(left.next_state.z, right.next_state.z)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/agent/test_core_world_model.py -q`.
- [ ] Создать shared `GRUCell(z_dim+h_dim,h_dim)`, per-schema action embeddings `(n_actions,h_dim)` и sensor projection `Linear(2*n_sensors,h_dim)`. После GRU — независимые member latent heads `Linear(h_dim,z_dim)`, schema-specific sensor heads и termination heads. Основной переход:

```python
values = state.sensors * state.sensor_mask
body = torch.cat([values, state.sensor_mask.float()], dim=-1)
condition = self.action_embeddings[state.schema](actions)
condition = condition + self.sensor_projections[state.schema](body)
hidden = self.recurrent(torch.cat([state.z, condition], dim=-1), state.hidden)
member_z = torch.stack([head(hidden) for head in self.latent_heads], dim=0)
next_z = member_z.mean(dim=0)
uncertainty = member_z.var(dim=0, unbiased=False).mean(dim=-1)
```

- [ ] `next_state` использует mean sensor predictions, ту же mask, новый hidden; termination — mean sigmoid heads. `rollout` подаёт `prediction.next_state` в следующий шаг без ground-truth подстановки. Инициализация hidden — нули на устройстве model. RGB uint8 приводится к float/255, sensors/masks копируются. Проверять отсутствующую schema и неправильный action ID явной ошибкой. В тестах — CPU/CUDA, reset hidden, разные S/A на одной модели, отсутствие in-place изменений, запрет NaN и history sensitivity. Поиск и encoder не получают число heads как число независимых world samples: это proxy, не полноценная вероятностная модель.
- [ ] GREEN: весь файл. Commit `feat(core): implement recurrent latent dynamics`. **Done:** один оператор работает для real-history и autoregressive prediction, interfaces не требуют `env`.

### Task 9: SIGReg и маскированные losses

**Files:** Create `src/snks/learning/core_objective.py`, `tests/learning/test_core_objective.py`.

**Interfaces:** `sigreg(z: Tensor[N,D], directions: Tensor[D,K]) -> Tensor` (N>=2); `masked_mse(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor`. Направления unit-normalized, передаются явно для воспроизводимости тестов; в train их создаёт checkpointed torch RNG.

**Depends on:** 7, 8. **Effort:** 4–6 часов.

- [ ] Написать тест против вырожденного распределения:

```python
def test_gaussian_scores_better_than_collapsed_embeddings():
    import torch
    import torch.nn.functional as F
    from snks.learning.core_objective import sigreg
    torch.manual_seed(11)
    normal = torch.randn(2048, 8, requires_grad=True)
    directions = F.normalize(torch.randn(8, 32), dim=0)
    good = sigreg(normal, directions)
    bad = sigreg(torch.zeros_like(normal), directions)
    assert good < bad
    good.backward()
    assert torch.isfinite(normal.grad).all()
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/learning/test_core_objective.py -q`.
- [ ] Реализовать Gaussian characteristic-function quadrature, сверив формулу с [первичным минимальным примером LeJEPA](https://github.com/galilai-group/lejepa/blob/main/MINIMAL.md). Это самостоятельная запись алгоритма, не установка полного чужого training stack:

```python
def sigreg(z, directions):
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("SIGReg requires at least two valid examples")
    points = torch.linspace(0.0, 3.0, 17, device=z.device, dtype=torch.float32)
    phases = (z.float() @ directions.float()).unsqueeze(-1) * points
    empirical = torch.complex(phases.cos().mean(0), phases.sin().mean(0))
    reference = torch.exp(-0.5 * points.square())
    integrand = (empirical - reference).abs().square() * reference
    return 2.0 * len(z) * torch.trapezoid(integrand, points, dim=-1).mean()

def masked_mse(pred, target, mask):
    expanded = torch.broadcast_to(mask, pred.shape).to(pred.dtype)
    if expanded.sum() == 0:
        return pred.sum() * 0.0
    return ((pred - target).square() * expanded).sum() / expanded.sum()
```

- [ ] Проверить constant vs normal, gradient и quad endpoints, отсутствующие сенсоры, padded timesteps, один valid example (явный reject/skip regularizer batch, не NaN), fp32 regularizer при mixed precision. Не подменять SIGReg variance-only штрафом. Regularizer не вызывается на predictions вместо реальных encoder outputs. Записать источник, выбранный recipe и отличие CNN/GRU от авторской архитектуры в комментарии модуля.
- [ ] GREEN: весь файл. Commit `feat(core): add predictive objectives and Gaussian regularization`. **Done:** loss имеет прямые численные проверки и независим от reward/семантической разметки.

### Task 10: Sequence trainer с обновлениями между эпизодами

**Files:** Create `src/snks/learning/core_trainer.py`, `tests/learning/test_core_trainer.py`; Modify `tests/core_helpers.py`.

**Interfaces:** `SequenceBatch(rgb, sensors, sensor_mask, actions, terminated, valid, schema: str, burn_in: int)`; rgb `[B,T+1,3,64,64]`, sensors/mask `[B,T+1,S]`, actions/terminated/valid `[B,T]`. `tensorize(episodes: list[Episode], burn_in: int, device: torch.device) -> SequenceBatch` rejects mixed schemas. `CoreTrainer(model: CoreWorldModel, config: CoreConfig)`, `compute_loss(batch: SequenceBatch) -> Tensor`, `update(batch: SequenceBatch, mode: Mode) -> dict`, attributes `model`, `config`, `optimizer`. Test helper `make_sequence_batch() -> SequenceBatch` builds B=2,T=3 toy sequences with alternating action and observed sensor increments, all valid, burn_in=1.

**Depends on:** 6, 8, 9. **Effort:** 6–8 часов.

- [ ] Написать read-only evaluation тест:

```python
def test_update_is_forbidden_in_evaluate():
    import pytest
    from snks.env.core_types import Mode
    from snks.encoder.core_encoder import CoreEncoder
    from snks.agent.core_world_model import CoreWorldModel
    from snks.learning.core_trainer import CoreTrainer
    from snks.pipeline.core_config import load_core_config
    from tests.core_helpers import make_sequence_batch
    from pathlib import Path
    model = CoreWorldModel(CoreEncoder(64), {"toy": (2, 1)}, 32, 3)
    trainer = CoreTrainer(model, load_core_config(Path("configs/core_smoke.yaml")))
    with pytest.raises(PermissionError):
        trainer.update(make_sequence_batch(), Mode.EVALUATE)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/learning/test_core_trainer.py -q`.
- [ ] `compute_loss` использует real-prefix burn-in, затем autoregressive state; реальные future embeddings — только targets, не вход следующего rollout. В end-to-end режиме target encoder не detach; frozen-encoder режим автоматически не имеет encoder gradients. Применить `masked_mse` к latent и sensor predictions и masked BCE к истинному `terminated`, не к `truncated`:

```python
prediction = model.step(state, batch.actions[:,t])
target_z = model.encoder(batch.rgb[:,t+1])
latent_loss = masked_mse(prediction.next_state.z, target_z,
                         batch.valid[:,t,None])
sensor_loss = masked_mse(prediction.next_state.sensors,
                         batch.sensors[:,t+1],
                         batch.sensor_mask[:,t+1] & batch.valid[:,t,None])
state = prediction.next_state
```

- [ ] В `update` первым действием проверить mode; затем `zero_grad(set_to_none=True)`, конечность loss, `backward`, конечность grads, `optimizer.step`. При численной ошибке не записывать новый checkpoint как успешный. Regularizer вычислять по valid реальным embeddings в одном schema-batch; по возможности брать разные эпизоды, не считать padding независимыми примерами. Добавить тесты отсутствия future leakage, обучения игрушечной action-conditioned зависимости, отсутствия encoder grads при frozen E, одинакового числа optimizer steps и пропуска незавершённых эпизодов. Runner вызывает update только после завершения эпизода; trainer сам не обращается к среде.
- [ ] GREEN: весь файл, objective и replay suites. Commit `feat(core): train recurrent dynamics from replay sequences`. **Done:** получается измеримый learning signal без task-specific labels; evaluate не изменяет веса.

### Task 11: Checkpoint, RNG и совместимость схем

**Files:** Create `src/snks/learning/core_checkpoint.py`, `tests/learning/test_core_checkpoint.py`; Modify `tests/core_helpers.py`.

**Interfaces:** `save_checkpoint(path: Path, model: CoreWorldModel, trainer: CoreTrainer, replay: SequenceReplay, metadata: dict) -> str` returns SHA-256; `load_checkpoint(path: Path, model: CoreWorldModel, trainer: CoreTrainer, replay: SequenceReplay, expected_schema_hash: str) -> dict`. При несовпадении схемы/manifest бросать `ValueError` до мутации model. Runtime hidden не переносится на новый эпизод.

**Depends on:** 6, 8, 10. **Effort:** 4–8 часов.

- [ ] Написать тест отказа без изменения параметров; helper `make_core_bundle()` создаёт toy model/trainer/replay из Tasks 6–10 и определяется в `tests/core_helpers.py`:

```python
def test_wrong_schema_does_not_mutate_model(tmp_path):
    import pytest, torch
    from tests.core_helpers import make_core_bundle
    from snks.learning.core_checkpoint import save_checkpoint, load_checkpoint
    model, trainer, replay = make_core_bundle()
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, trainer, replay, {"schema_hash": "a"})
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(ValueError):
        load_checkpoint(path, model, trainer, replay, "b")
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/learning/test_core_checkpoint.py -q`.
- [ ] Сначала валидировать metadata и shapes; только затем применять state dictionaries. Payload включает optimizer, torch CPU/CUDA RNG, NumPy/Python RNG в tensor/primitive представлении, replay manifest/hash, action schema, normalizations, config hash, episode counters и curriculum position. Атомарная запись в пределах одного filesystem:

```python
temporary = path.with_name(path.name + ".partial")
with temporary.open("wb") as handle:
    torch.save(payload, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, path)
```

- [ ] Новые checkpoints имеют уникальные versioned имена; `path` не перезаписывает предыдущую принятую версию. Загрузка только доверенных собственных артефактов с restricted tensor/primitive payload; произвольный pickle среды запрещён. Проверить восстановление следующего replay sample, optimizer step и RNG; corrupted hash, missing replay, mismatch action map, partially written file и сохранность исходного checkpoint. Дописать `tests/core_helpers.py` в Files/commit этой Task.
- [ ] GREEN: checkpoint, trainer и replay suites. Commit `feat(core): persist compatible learning state atomically`. **Done:** save/load сохраняет знание и воспроизводимость, не создаёт ложный новый независимый seed.

### Task 12: Независимый cost и ограниченное планирование

**Files:** Create `src/snks/agent/core_cost.py`, `core_planner.py`, `tests/agent/test_core_planner.py`.

**Interfaces:** `GoalCost(goal_z: Tensor | None, ranges: dict[int,tuple[float,float]], image_weight: float = 1.0, sensor_weight: float = 1.0, uncertainty_weight: float = 0.0, termination_weight: float = 1.0)`, `__call__(prediction: Prediction) -> Tensor[B]`; `PlanResult(actions: tuple[int, ...], cost: float, model_calls: int, trace: list[dict])`; `beam_plan(model: CoreWorldModel, root: LatentState, cost: GoalCost, n_actions: int, horizon: int, beam_width: int, max_calls: int) -> PlanResult`.

**Depends on:** 8, 11 (test bundle). **Effort:** 6–8 часов.

- [ ] Написать тест бюджета:

```python
def test_planner_never_exceeds_model_budget():
    from tests.core_helpers import make_core_bundle, make_observation
    from snks.agent.core_cost import GoalCost
    from snks.agent.core_planner import beam_plan
    model, trainer, replay = make_core_bundle()
    root = model.initial(make_observation())
    result = beam_plan(model, root, GoalCost(None, {0: (1., 2.)}),
                       n_actions=2, horizon=3, beam_width=2, max_calls=6)
    assert 0 < result.model_calls <= 6
    assert result.actions[0] in (0, 1)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/agent/test_core_planner.py -q`.
- [ ] Cost sensor interval не зависит от названия ресурса:

```python
value = prediction.next_state.sensors[:, index]
violation = torch.relu(lower - value) + torch.relu(value - upper)
score = score + self.sensor_weight * violation.square()
```

- [ ] Для image-goal добавить средний squared latent distance; цель в отсутствующем сенсоре — явная ошибка GoalSpec, не бесплатный нулевой штраф. Beam начинает с root; в каждом depth расширяет сохранённые nodes всеми syntax-valid actions, в порядке ID; считает каждый candidate transition за один model call независимо от batching; возвращает лучший полный доступный префикс. Tie-break `(cost, action_tuple)` детерминированный. Умершие branches не расширяются; terminal risk включается как declared constraint/cost, не скрытая эвристика. При исчерпании бюджета до одного кандидата — `RuntimeError`, не legacy fallback. Псевдокод ядра:

```python
from dataclasses import dataclass

@dataclass
class BeamNode:
    state: LatentState
    actions: tuple[int, ...]
    cost: float
    survival: float = 1.0

beam = [BeamNode(root, (), 0.0)]
calls = 0
device = root.z.device
for depth in range(horizon):
    expanded = []
    for node in beam:
        if node.survival == 0.0:
            expanded.append(node)
            continue
        for action in range(n_actions):
            if calls == max_calls:
                break
            prediction = model.step(node.state, torch.tensor([action], device=device))
            calls += 1
            expanded.append(BeamNode(prediction.next_state,
                                     node.actions + (action,),
                                     node.cost + node.survival * float(cost(prediction).item()),
                                     node.survival * (1.0 - float(prediction.terminated_prob.item()))))
    if not expanded:
        break
    beam = sorted(expanded, key=lambda node: (node.cost, node.actions))[:beam_width]
```

- [ ] Завершить kernel сбором `PlanResult` из лучшего BeamNode и trace каждого вызова. Первый профиль использует expected сумму costs с discount=1 и survival weight. GoalCost добавляет `termination_weight * prediction.terminated_prob` для avoided termination; значение weight и тип завершения фиксируются task profile, успешное terminal состояние не штрафуется как смерть. Без threshold по вероятности: ранние несовершенные prediction heads не должны произвольно обрезать search. Проверить tie-break, no action name branching, frozen goal, отсутствующий сенсор, NaN rejection, root immutability и toy case, где более точная динамика меняет первое действие.
- [ ] GREEN: весь файл. Commit `feat(core): plan discrete actions through learned dynamics`. **Done:** оценка желаемого состояния отделена от predictor и search имеет строгий budget.

### Task 13: Real observe/act loop и неизменяемое evaluation

**Files:** Create `src/snks/agent/core_agent.py`, `src/snks/pipeline/core_runner.py`, `tests/agent/test_core_agent.py`, `tests/test_core_runner.py`; Modify `tests/core_helpers.py`.

**Interfaces:** `CoreAgent(model: CoreWorldModel, config: CoreConfig)`, `start(obs: Observation, goal: GoalSpec) -> None`, `act(exploration_fraction: float = 0.0) -> int`, `observe(transition: Transition) -> None`, `last_trace: list[dict]`. `EpisodeResult(episode: Episode, steps: int, agent_failed: bool, infrastructure_failed: bool, audit: list[dict])`. `run_episode(adapter: CoreAdapter, agent: CoreAgent, case: TaskCase, mode: Mode, replay: SequenceReplay, trainer: CoreTrainer | None) -> EpisodeResult`.

**Depends on:** 3–6, 10–12. **Effort:** 6–8 часов.

- [ ] Helper `TinyAdapter` в `tests/core_helpers.py` реализует интерфейс Task 3: два действия, sensor+=1 для action 1, terminal после трёх шагов, без reset overhead. Helper `make_toy_case(split: str) -> TaskCase` задаёт goal sensor>=1, max_steps=3, фиксированный seed. Написать read-only integration test:

```python
def test_evaluate_keeps_weights_and_replay_unchanged():
    import torch
    from tests.core_helpers import make_core_bundle, TinyAdapter, make_toy_case
    from snks.env.core_types import Mode
    from snks.agent.core_agent import CoreAgent
    from snks.pipeline.core_runner import run_episode
    model, trainer, replay = make_core_bundle()
    before = {k: v.clone() for k,v in model.state_dict().items()}
    count = replay.manifest()
    result = run_episode(TinyAdapter(), CoreAgent(model, trainer.config),
                         make_toy_case("test"), Mode.EVALUATE, replay, None)
    assert result.steps == 3
    assert replay.manifest() == count
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/agent/test_core_agent.py tests/test_core_runner.py -q`.
- [ ] `CoreAgent.observe` сначала прогоняет исполненное действие из старого root, затем корректирует prediction реальным наблюдением:

```python
prediction = self.model.step(self.state, torch.tensor([transition.action], device=device))
observed = self.model.initial(transition.after)
self.state = LatentState(observed.z, observed.sensors, observed.sensor_mask,
                         prediction.next_state.hidden.detach(), observed.schema)
```

- [ ] Runner владеет adapter, evaluator metadata и split check; agent получает только Observation/GoalSpec/Transition. При EVALUATE включает `model.eval()` и `torch.inference_mode()`, перед/после сравнивает weight/buffer/replay hashes, запрещает optimizer updates. В TRAIN/ADAPT inference внутри эпизода также без gradient graph; после полного эпизода выполняются append и объявленное число updates. Interrupted episode не подаётся trainer. `steps` включает reset_transitions; при достижении budget последняя запись получает truncated, не fake success. Последний episode может завершить сбор раньше checkpoint budget; сравнения используют общую достигнутую сетку и не добавляют бесплатные шаги.
- [ ] Trace содержит candidate prefix, components cost, uncertainty, chosen primitive, model hash, observed deltas, failure reason. NaN/нет действия — `agent_failed=True`, success=0; высокая uncertainty остаётся обычным эпизодом. Infrastructure failure логируется отдельно и не подменяет agent failure. Запретить implicit legacy fallback прямым тестом: подставить failing planner и проверить отсутствие импорта/вызова legacy agent. Тренировочный exploration выбирает синтаксически допустимые ID seeded RNG, не имена объектов.
- [ ] GREEN: оба файла плюс planner/checkpoint suites. Commit `feat(core): connect real experience and audited agent control`. **Done:** сквозной агент можно запускать, но это ещё не G2/G3 PASS.

### Task 14: Метрики, независимые единицы и oracle diagnostics

**Files:** Create `src/snks/pipeline/core_metrics.py`, `core_oracle.py`, `tests/test_core_metrics.py`, `tests/test_core_oracle.py`.

**Interfaces:** `normalized_auc(steps: list[int], scores: list[float]) -> float`; `paired_cluster_interval(left: np.ndarray, right: np.ndarray, seed: int, n_boot: int = 10000, alpha: float = 0.05) -> tuple[float,float]` (first axis = training seed); `prediction_metrics(records: list[dict]) -> dict`; `ReplayOracle(factory: Callable[[],DiagnosticAdapter], seed: int, prefix: tuple[int,...])`, `predict(actions: tuple[int,...]) -> list[Transition]`. `OracleDynamics` — evaluator-only совместимый с `beam_plan` объект с `initial`/`step`, кодирует истинные предсказанные наблюдения замороженным encoder.

**Depends on:** 3–5, 8, 12, 13. **Effort:** 6–8 часов.

- [ ] Написать проверку неравномерной сетки и постоянного парного эффекта:

```python
def test_auc_and_paired_interval():
    import numpy as np
    import pytest
    from snks.pipeline.core_metrics import normalized_auc, paired_cluster_interval
    assert normalized_auc([0, 2, 10], [0., 1., 1.]) == pytest.approx(0.9)
    left = np.full((5, 4), 0.7)
    right = np.full((5, 4), 0.2)
    lo, hi = paired_cluster_interval(left, right, seed=0)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_metrics.py tests/test_core_oracle.py -q`.
- [ ] Реализовать численные ядра без объединения эпизодов разных training seeds в фиктивные независимые модели:

```python
def normalized_auc(steps, scores):
    x, y = np.asarray(steps), np.asarray(scores, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(x) != len(y)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(np.diff(x) <= 0) or np.any((y < 0) | (y > 1))):
        raise ValueError("invalid learning curve grid")
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0]))

def paired_cluster_interval(left, right, seed, n_boot=10000, alpha=0.05):
    if (left.ndim < 1 or left.shape != right.shape or left.shape[0] < 5
            or left.size == 0 or not np.isfinite(left).all()
            or not np.isfinite(right).all() or n_boot < 1 or not 0 < alpha < 1):
        raise ValueError("paired conditions require at least five training runs")
    differences = (left - right).reshape(len(left), -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    means = rng.choice(differences, (n_boot, len(differences)), replace=True).mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist())
```

- [ ] `prediction_metrics` считает sensor MAE/MSE на valid fields, termination Brier score, event precision/recall для заранее объявленных наблюдаемых изменений, coverage и число эпизодов/событий по горизонту 1/3/5/10. Отсутствующие положительные события дают unavailable metric, не искусственный perfect score. Geometry probes, если потребуются, обучаются только на development в evaluator, с замороженным encoder, без обратной передачи градиентов/labels в agent; их бюджет/capacity фиксирован одинаково для сравнений. Не сравнивать raw latent MSE разных encoder checkpoints как общую шкалу.
- [ ] Добавить `uncertainty_metrics(errors: np.ndarray, disagreement: np.ndarray, critical: np.ndarray, threshold: float) -> dict`: Spearman correlation, число критических errors ниже validation-selected uncertainty threshold и coverage. Threshold выбирается только на validation и фиксируется в manifest. Proxy с отсутствующей связью с ошибкой не описывать как calibrated confidence; высокую uncertainty не использовать для исключения эпизодов.
- [ ] Oracle сбрасывает fresh env с тем же seed, воспроизводит реальный prefix и candidate actions; проверяет root observation и diagnostic state/RNG digest на совпадение. Prefix replay cost логируется отдельно. `OracleDynamics` хранит evaluator-only отображение identity созданного LatentState в prefix, возвращает `Prediction` с реально полученными RGB/sensors и нулевой uncertainty; это privileged baseline, никогда обычный agent backend. Использует тот же `beam_plan`, horizon и candidate-call budget. Тест: две ветки из одного root не мутируют live env и повторяют реальный исход; несовпадение воспроизведения — explicit invalid oracle, не допустимое приближение.
- [ ] GREEN: оба файла. Commit `feat(core): add paired metrics and isolated oracle controls`. **Done:** математическая агрегация и oracle воспроизводимость проверены до expensive experiments.

### Task 15: Исполняемые G1/G2 controls с фиксированной оценкой цели

**Files:** Create `src/snks/pipeline/core_controls.py`, `tests/test_core_controls.py`.

**Interfaces:** `build_dynamics_controls(model: CoreWorldModel) -> dict[str, CoreWorldModel]` keys `initial`, `real_actions`, `shuffled_actions`; `shuffle_action_labels(batch: SequenceBatch, seed: int) -> SequenceBatch`; `run_dynamics_controls(models: dict, batches: list[SequenceBatch], config: CoreConfig) -> dict` returns checkpoints, update counts, frozen hashes; `compare_control_results(records: list[dict]) -> dict` использует Task 14.

**Depends on:** 10–14. **Effort:** 4–8 часов.

- [ ] Написать тест заморозки всех encoder параметров и buffers:

```python
def test_controls_start_with_identical_frozen_encoders():
    import torch
    from tests.core_helpers import make_core_bundle
    from snks.pipeline.core_controls import build_dynamics_controls
    model, trainer, replay = make_core_bundle()
    controls = build_dynamics_controls(model)
    assert set(controls) == {"initial", "real_actions", "shuffled_actions"}
    reference = controls["initial"].encoder.state_dict()
    for variant in controls.values():
        assert not any(p.requires_grad for p in variant.encoder.parameters())
        assert all(torch.equal(v, reference[k])
                   for k,v in variant.encoder.state_dict().items())
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_controls.py -q`.
- [ ] Построить deepcopy ветки из одного development checkpoint:

```python
controls = {name: copy.deepcopy(model)
            for name in ("initial", "real_actions", "shuffled_actions")}
for variant in controls.values():
    variant.encoder.requires_grad_(False)
    variant.encoder.eval()
```

- [ ] Goal encoding/cost normalization, search parameters и исходные последовательности общие. `initial` не дообучается; остальные имеют одинаковые batch order и update counts. Shuffle переставляет labels только внутри одной schema и valid positions, сохраняет action histogram и не изменяет obs/targets/masks. Новый label применяется и к recurrent input; служебные true labels не доступны shuffled predictor. После обновлений проверить hashes encoder/buffers/goal cost. В matched shadow использовать одну реальную историю и candidate set, hidden пересчитать отдельно каждым predictor. Проверить и prediction, и closed-loop success; не закрывать G2 только action sensitivity или изменившимся ranking.
- [ ] Добавить `PersistenceDynamics(model: CoreWorldModel)`: `initial` делегирует model, `step(state, actions) -> Prediction` возвращает неизменённые z/sensors/hidden, termination=0, uncertainty=0 и повтор z по ensemble axis. Actions игнорируются явно; на не-terminal root persistence предсказывает отсутствие завершения. Это G1 baseline, не новая ветвь основного агента.
- [ ] Добавить evaluator-only `StructuredPredictor(state_dim: int, n_actions: int)`, `forward(state: Tensor[B,D], actions: Tensor[B]) -> Tensor[B,D]`: concatenate state и one-hot action, `Linear(D+A,64)`, GELU, `Linear(64,D)`, residual к state. Источник state — diagnostic schema adapter, не Observation; состав state/probe readout объявить по каждой среде и сохранить в manifest. Обучать на тех же train episode keys и action-conditioned MSE, не test. Тест: синтетическое sensor+=action обучение уменьшает MSE на независимых последовательностях; agent не импортирует этот класс. Legacy остаётся отдельно помеченным инженерным baseline, не causal ablation. Зафиксировать два сравнения G2 `real_actions - initial`, `real_actions - shuffled_actions`.
- [ ] GREEN: весь файл и trainer/runner/metrics suites. Commit `feat(core): isolate dynamics contribution to decisions`. **Done:** changing encoder/cost запрещено тестом; улучшение G2 действительно может быть отнесено к обучению динамики.

### Task 16: Явный transfer contract и A→B→A

**Files:** Create `src/snks/pipeline/core_transfer.py`, `tests/test_core_transfer.py`; Modify `src/snks/agent/core_world_model.py`.

**Interfaces:** `TransferCondition` enum: `FRESH`, `WEIGHTS`, `WEIGHTS_REPLAY`, `SOURCE_CONTROL`; `prepare_transfer(source: CoreWorldModel, replay: SequenceReplay, condition: TransferCondition, target_schema: str, target_shape: tuple[int,int], seed: int, config: CoreConfig) -> tuple[CoreWorldModel,CoreTrainer,SequenceReplay]`; `run_transfer_grid(conditions: dict, cases: list[TaskCase], checkpoints: list[int], config: CoreConfig) -> list[dict]`. `CoreWorldModel.register_schema(name: str, shape: tuple[int,int], seed: int) -> None` добавляется в Task 8 module без изменения существующих A heads.

**Depends on:** 5, 6, 10, 11, 13, 14. **Effort:** 6–8 часов.

- [ ] Тестирует отсутствие переноса optimizer и сохранение A-specific параметров:

```python
def test_transfer_preserves_source_schema_and_resets_optimizer():
    import torch
    from tests.core_helpers import make_core_bundle
    from snks.pipeline.core_transfer import prepare_transfer, TransferCondition
    source, trainer, replay = make_core_bundle()
    model, new_trainer, new_replay = prepare_transfer(
        source, replay, TransferCondition.WEIGHTS, "grid-v1", (5, 1), 17, trainer.config)
    assert "toy" in model.schemas and "grid-v1" in model.schemas
    assert new_trainer.optimizer.state_dict()["state"] == {}
    assert torch.equal(source.action_embeddings["toy"].weight,
                       model.action_embeddings["toy"].weight)
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_transfer.py -q`.
- [ ] Сделать transfer table исполняемым: encoder, recurrent dynamics и общие heads переносятся для WEIGHTS/WEIGHTS_REPLAY; A-specific schema components сохраняются; новые B embeddings/projections/heads создаются из одной seed stream во всех условиях; optimizer всегда новый. FRESH имеет ту же полную структуру, но свежие shared weights. Эпизодический hidden всегда reset. Для schema initialization не расходовать общий случайный поток обучения:

```python
with torch.random.fork_rng(devices=[]):
    torch.manual_seed(seed)
    action_embedding = nn.Embedding(n_actions, h_dim)
    sensor_projection = nn.Linear(2 * n_sensors, h_dim)
```

- [ ] Конструировать новые компоненты на CPU, затем переносить на model device; одинаковость B init проверять tensor-by-tensor для всех условий. WEIGHTS_REPLAY получает исходный A replay; WEIGHTS/FRESH — пустой A replay. Количество updates в B одинаковое: replay меняет состав batches, не добавляет бесплатный compute. SOURCE_CONTROL использует отдельно предобученный source-control checkpoint с тем же source-transition/update budget и той же declared replay policy, а не испорченный target dataset. Конкретный источник фиксируется в Task 19 до sealed test.
- [ ] `run_transfer_grid` создаёт новую branch каждого training seed; публикует B learning curves по одной сетке actual B steps, включает reset costs, отдельно lifetime A+B cost, затем оценивает A без updates. Полные эпизоды при checkpoint boundary: terminate-by-budget как truncated, и update только после этой границы; одинаковое правило всех условий. Проверить hash A-specific parameters до B initialization, optimizer reset, no test replay, read-only return-to-A и checkpoint roundtrip. При нехватке B budget не extrapolate кривую до отсутствующего checkpoint.
- [ ] GREEN: новый файл плюс checkpoint/runner/metrics suites. Commit `feat(core): make transfer and retention conditions explicit`. **Done:** fresh/transfer не расходятся незаметно в architecture, init, бюджетах или режиме оценивания.

### Task 17: Campaign contract, CLI и интервальные gates

**Files:** Create `src/snks/pipeline/core_campaign.py`, `experiments/exp138_learning_core.py`, `tests/test_core_campaign.py`, `configs/core_pilot.yaml`; Modify `src/snks/pipeline/core_config.py`, `src/snks/pipeline/core_preflight.py`.

**Interfaces:** `CampaignManifest` содержит source commit, config/split/checkpoint hashes, cases, budgets, conditions, seeds, metric margins, rerun rules, `status: str` (`pilot`/`registered`). Одобрение — отдельный receipt, не изменяемое поле зарегистрированного manifest. `validate_campaign(manifest: dict, confirmatory: bool) -> None`; `classify_gate(lower_bound: float, upper_bound: float, practical_margin: float) -> str`; `main(argv: list[str] | None = None) -> int`; `execute_stage(stage: str, manifest: dict, out: Path, approval: dict | None = None) -> dict`.

**Depends on:** 1, 5, 11, 13–16. **Effort:** 4–8 часов.

- [ ] Написать тест запрета confirmatory без регистрации:

```python
def test_pilot_cannot_be_run_as_confirmatory():
    import pytest
    from snks.pipeline.core_campaign import validate_campaign, classify_gate
    with pytest.raises(ValueError):
        validate_campaign({"status": "pilot"}, confirmatory=True)
    assert classify_gate(-0.1, 0.3, 0.05) == "PARTIAL"
    assert classify_gate(0.1, 0.3, 0.05) == "PASS"
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_campaign.py -q`.
- [ ] Реализовать интервальный gate, отклоняющий неконечные данные:

```python
import math

def classify_gate(lower_bound, upper_bound, practical_margin):
    if (not all(math.isfinite(x) for x in
                (lower_bound, upper_bound, practical_margin))
            or lower_bound > upper_bound):
        raise ValueError("invalid interval")
    if lower_bound > practical_margin:
        return "PASS"
    if upper_bound <= practical_margin:
        return "FAIL"
    return "PARTIAL"
```

- [ ] `validate_campaign` требует все поля спеки §10, минимум 5 training seeds для confirmatory, непересекающиеся splits, checkpoint grid с началом 0, обязательные сравнения, confidence level, bootstrap seed, practically significant margins и retention tolerance. Для simultaneous comparisons первая реализация использует Bonferroni: alpha=0.05/число обязательных сравнений в paired cluster intervals Task 14. Retention effect — after-minus-before с границей `-margin`; G1 improvement — baseline-minus-model error. Недостающие events/conditions/episodes дают PARTIAL/invalid protocol, не исключаются из знаменателя. Эти правила фиксируются до test.
- [ ] CLI принимает `--stage {preflight,collect,train,shadow,evaluate,controls,transfer}`, `--manifest PATH`, `--out PATH`. Обязательный manifest читается явно, без hidden default. Скрипт импортирует `main` из pipeline; execute_stage вызывает конкретные runners Tasks 13–16. Каждый output directory новый, overwrite запрещён. Остальные исполняемые stages добавляет Task 18 до регистрации, не после test.
- [ ] Создать полный `core_pilot.yaml`: все поля smoke скопированы явно; profile=pilot, device=cuda, z_dim=256, h_dim=128, batch_size=16, train_horizon=3, beam_width=8, planner_horizon=5, max_model_calls=1024. Остальные значения из Task 1 сохраняются. Это development стартовый профиль, не обязательный научный optimum.
- [ ] GREEN: `.venv-core/bin/python -m pytest tests/test_core_campaign.py tests/test_core_preflight.py -q` и `.venv-core/bin/python experiments/exp138_learning_core.py --help`.
- [ ] Commit `feat(core): define auditable campaign contracts`. **Done:** общие stages исполняются на development, incomplete manifest не получает confirmatory статус.

### Task 18: Завершить интервенции, регистрацию и отчёт до заморозки кода

**Files:** Create `src/snks/pipeline/core_interventions.py`, `src/snks/pipeline/core_reporting.py`, `tests/test_core_interventions.py`, `tests/test_core_reporting.py`; Modify `src/snks/pipeline/core_campaign.py`, `core_tasks.py`, `src/snks/env/core_grid.py`, `tests/test_core_campaign.py`, `tests/core_helpers.py`.

**Interfaces:** In core_interventions: `intervention_cases(cases: list[TaskCase], intervention: str) -> list[TaskCase]`, `run_interventions(manifest: dict, out: Path) -> list[dict]`. In core_reporting: `aggregate_status(gates: dict[str,str]) -> str`, `render_report(results: dict, manifest: dict) -> str`. In core_campaign: `make_pilot_manifest(checkout: Path, config: Path, tasks: Path) -> dict`, `run_pilot(manifest: dict, out: Path) -> dict`, `register_campaign(manifest: dict, pilot: dict, path: Path) -> str` returns SHA-256; `run_confirmatory(manifest: dict, approval: dict, out: Path) -> dict`.

**Depends on:** 1–17. **Effort:** 8–12 часов. Вся логика кампании, включая report, создаётся здесь до pilot/registration. Task 19 не добавляет runtime code.

- [ ] Написать три независимых защитных теста в перечисленных файлах:

```python
def test_intervention_changes_rules_not_goal():
    from snks.pipeline.core_interventions import intervention_cases
    from tests.core_helpers import make_grid_case
    original = make_grid_case("adapt", consume_key=False)
    changed = intervention_cases([original], "key_consumption_switch")[0]
    assert changed.ruleset != original.ruleset
    assert changed.goal == original.goal
    assert changed.family == original.family

def test_pilot_refuses_test_cases(tmp_path):
    import pytest
    from snks.pipeline.core_campaign import run_pilot
    with pytest.raises(PermissionError):
        run_pilot({"status": "pilot", "cases": [{"split": "test"}]}, tmp_path)

def test_missing_transfer_family_prevents_full_pass():
    from snks.pipeline.core_reporting import aggregate_status
    gates = {f"G{i}": "PASS" for i in range(7)}
    gates["G3"] = "PARTIAL"
    assert aggregate_status(gates) == "PARTIAL"
```

- [ ] RED: `.venv-core/bin/python -m pytest tests/test_core_interventions.py tests/test_core_reporting.py tests/test_core_campaign.py -q`.
- [ ] Реализовать преобразование task configuration без передачи ruleset модели:

```python
from dataclasses import replace

def intervention_cases(cases, intervention):
    switches = {
        "key_consumption_switch": {"key_reusable": "key_consumed",
                                   "key_consumed": "key_reusable"},
        "push_distance_switch": {"push_1": "push_2", "push_2": "push_1"},
        "appearance_decoupling": {"color_linked": "color_independent"},
    }
    mapping = switches[intervention]
    return [replace(case, ruleset=mapping[case.ruleset]) for case in cases]
```

- [ ] Дополнить GridRules полем `appearance_mode: str = "independent"` и генератор: для color_linked несущественный цвет фона связан с consume_key; для color_independent цвет выбирается отдельным seeded RNG независимо от consume_key. Цвет настоящего ключа/двери и geometry/goal image неизменны. Каталог materializes обе величины отдельно и сохраняет их в evaluator audit, не Observation. Paired physical interventions сохраняют видимый root до диагностического действия; appearance intervention специально меняет статистическую связь между внешним видом и правилом. Одинаковое утверждение о pixel identity для этих двух разных типов контроля не применять.
- [ ] `run_interventions` создаёт paired world cases, собирает только реальные adaptation transitions, запускает matched no-update control, затем измеряет held-out alternative-action errors и planning success. Oracle counterfactual labels остаются evaluator-only. На игрушечной fixture проверить известные отличающиеся исходы действия и запрет попадания ruleset в model input.
- [ ] Первым действием `run_pilot` запрещает test cases, затем выполняет preflight и последовательность collect/train/controls/transfer/interventions только на development:

```python
if any(case["split"] not in {"train", "adapt", "validation"}
       for case in manifest["cases"]):
    raise PermissionError("pilot cannot open sealed test cases")
```

- [ ] `make_pilot_manifest` вычисляет actual HEAD, SHA-256 полного config и каталога, использует development seeds [0,1,2], 16 episodes/family/condition, max_steps=64, до 100 updates/condition. Cases генерируются каталогом Task 5 только с train/adapt/validation splits. Не вставлять фиктивный commit и не обещать confirmatory power этим трём seeds. `register_campaign` требует фактические pilot resources, заполненные численные margins/sample sizes, полный failure/rerun protocol; сохраняет новый JSON без overwrite. Выбор чисел описан в Task 19.
- [ ] Реализовать receipt validation перед materialization в `run_confirmatory`: receipt содержит manifest hash, registration commit, одобрение пользователя и утверждённый бюджет. Проверить exact HEAD==registration commit и clean tracked tree; этот commit отличается от source-code commit только зарегистрированными config/docs. Проверить hashes исполняемого source, experiment entrypoint, dependencies и tests против source-code commit; config hash — против manifest. Нельзя требовать от manifest знать hash собственного будущего commit. При несовпадении остановиться до env creation.
- [ ] Реализовать confirmatory orchestration всех training seeds/conditions, G2 fixed encoder/cost, обеих G3 families, return-to-A/save-load G4 и независимых G5 cases. Agent failure остаётся success=0; infrastructure rerun следует заранее объявленному правилу, сохраняет первую попытку и не выбирает лучшую. Тестировать orchestration маленькими synthetic runners с теми же result schemas и счётчиками; реальная confirmatory не является unit test.
- [ ] Реализовать отчётную агрегацию:

```python
def aggregate_status(gates):
    if set(gates) != {f"G{i}" for i in range(7)}:
        return "PARTIAL"
    if all(value == "PASS" for value in gates.values()):
        return "PASS"
    if any(value == "FAIL" for value in gates.values()):
        return "FAIL"
    return "PARTIAL"
```

- [ ] `render_report` строит Markdown из результатов: G0–G6, simultaneous intervals, per-family outcomes, source/registration commits, все failures, A/B/lifetime budgets и существующие artifact links. Missing key/family даёт PARTIAL. Результаты не вычисляются повторным обучением. Snapshot test проверяет наличие каждого gate и отсутствие фразы о достижении AGI.
- [ ] Добавить CLI stages `pilot-manifest,pilot,register,interventions,confirm,report` и флаги `--approval PATH`, `--results PATH`, `--config PATH`, `--tasks PATH`. Для pilot-manifest обязательны config/tasks/out; остальные modes требуют manifest/out; register/report дополнительно results; confirm дополнительно approval. `--out` всегда новый directory, JSON/report размещаются внутри с фиксированным именем. Неподходящие flags вызывают parser error, не игнорируются.
- [ ] GREEN: все четыре campaign/intervention/reporting/oracle suites, затем `.venv-core/bin/python -m pytest tests/ -x -q` на HyperPC. Проверить CLI smoke на синтетических данных, negative approval/hash cases и запуск всех stages без sealed test.
- [ ] Commit `feat(core): complete preregistration and causal evaluation workflow`. **Done:** весь исполняемый код готов до регистрации; положительные результаты ещё не заявлены.

### Task 19: Development pilot, регистрация и confirmatory A→B→A

**Files:** Create по фактическим результатам `docs/reports/learning-core-pilot.md`, `configs/core_campaign.json`; artifacts под `output_to_user/core/` на HyperPC. Runtime files не изменять в ходе зарегистрированной кампании.

**Interfaces:** Consumes исполняемые CLI/API Task 18. Produces measured pilot report, зарегистрированный manifest, отдельный approval receipt и полный result ledger.

**Depends on:** 1–18; доступ/ресурсный бюджет pilot подтверждаются перед запуском, длительная confirmatory требует отдельного одобрения зарегистрированного manifest. **Effort:** 4–8 часов сопровождения; machine time пока неизвестно.

- [ ] На HyperPC проверить exact code commit/import/CUDA через preflight. После согласования pilot budget создать development manifest:
  `.venv-core/bin/python experiments/exp138_learning_core.py --stage pilot-manifest --config configs/core_pilot.yaml --tasks configs/core_tasks.yaml --out output_to_user/core/pilot-protocol-01`.
- [ ] Запустить pilot: `.venv-core/bin/python experiments/exp138_learning_core.py --stage pilot --manifest output_to_user/core/pilot-protocol-01/manifest.json --out output_to_user/core/pilot-01`.
- [ ] Измерить peak allocated/reserved VRAM, encode/train/decision time, native transitions, model calls, failures, oracle solvability и редкие события. OOM останавливает profile; новый development profile получает новый run-id. Сначала проверять pipeline и coarse signal, не добиваться G3 подбором B-test.
- [ ] Проверить source-control: реальный locomotion-only Crafter corpus, без resource-interaction episodes и подмены targets. Объявленная collecting policy выбирает движения/noop, основной агент её не получает. Transition/update budget совпадает с A-source. Locomotion может переноситься — это публикуется, не подавляется искусственно. До регистрации сформулировать специфический эффект G3 относительно этого контроля; если сравнение не отделяет гипотезу, повторить development, не открывая test.
- [ ] На development оценить variance training seeds, event coverage и precision/power для планируемого числа eval episodes. Выбрать минимум 5 confirmatory training seeds. Обосновать practically significant margins масштабом задач: G1/G2 error reduction, G2 success, G3 normalized AUC, G4 retention tolerance, G5 event coverage. Не выбирать margin по признаку того, что pilot его уже превысил. Если ресурсы не обеспечивают требуемую точность — exploratory-only report и решение пользователя о следующем бюджете.
- [ ] Если pilot потребовал исправления кода, вернуться к затронутой Task, пройти её RED/GREEN и повторить relevant pilot на новом exact commit. Только после этого регистрировать протокол. Записать в JSON конкретные measured численные budgets/margins/seeds, а не незаполненные поля; `register_campaign` отвергает неполноту.
- [ ] Выполнить stage register с manifest-кандидатом, pilot results и новым out directory. Сохранить итоговый JSON как `configs/core_campaign.json` и фактический report через обычные контролируемые правки. Commit `docs(core): preregister measured validation campaign` меняет только эти config/docs; source-code commit остаётся зафиксированным.
- [ ] **Остановиться для просмотра пользователем manifest и бюджета.** Одобрение записывается отдельным receipt `output_to_user/core/approval.json` с exact manifest SHA-256 и registration commit. Исполнитель не выдаёт разрешение сам себе.
- [ ] После одобрения: `.venv-core/bin/python experiments/exp138_learning_core.py --stage confirm --manifest configs/core_campaign.json --approval output_to_user/core/approval.json --out output_to_user/core/confirm-01` из checkout registration commit на HyperPC.
- [ ] Сохранить все seeds, failures и незавершённые условия. Изменение runtime после открытия test закрывает текущую кампанию; новый confirmatory вывод требует новой регистрации и нового test split. Не увеличивать compute до получения PASS без нового решения пользователя.
- [ ] **Done:** полный проверяемый ledger либо явно неполный результат. M4 — согласованный протокол, M5 — полученные данные, а не гарантированный научный успех.

### Task 20: Итоговый отчёт и решение по первой версии

**Files:** Create `docs/reports/learning-core-validation.md`; Modify `docs/ASSUMPTIONS.md`, `docs/ROADMAP.md` только по фактическому результату. Report generator и его тесты уже завершены в Task 18.

**Interfaces:** Consumes immutable result ledger, manifest и `render_report` Task 18. Produces проверенный отчёт и явно сформулированный следующий исследовательский вопрос.

**Depends on:** 19. **Effort:** 4–6 часов.

- [ ] Проверить hashes manifest/checkpoints/logs, полноту зарегистрированных comparisons и ledger infrastructure incidents. Отсутствующие artifacts не реконструировать по памяти.
- [ ] На HyperPC из frozen registration checkout выполнить `.venv-core/bin/python experiments/exp138_learning_core.py --stage report --manifest configs/core_campaign.json --results output_to_user/core/confirm-01/results.json --out output_to_user/core/report-01`. Этот stage не обучает модель и не меняет outcomes.
- [ ] Сверить отчёт с `docs/STAGE_REVIEW_CRITERIA.md`, `docs/ANTI_TUNING_CHECKLIST.md` и `docs/CONCEPT_SUCCESS_CRITERIA.md`. Разделить инженерную готовность, fixed-representation G2, end-to-end behavior, перенос, удержание A и ограниченные causal cases. Для PASS показать путь от real episode к обновлению, прогнозу, решению и результату.
- [ ] Записать реальные проверки HyperPC, включая полный regression Task 18 и недоступные suites. Не выдавать focused tests за полный PASS. Для PARTIAL/FAIL назвать неподтверждённую гипотезу; не заменять её средоспецифичным patch ради gate.
- [ ] Обновить документы локально; commit `docs(core): report learning and transfer evidence`. Runtime исходники и frozen artifacts этим commit не меняются.
- [ ] **Done:** пользователь получает evidence-backed решение о следующей итерации. Ни PASS первой версии, ни завершение этого плана не означают достижения AGI.

## Общие тестовые helpers: точные начальные реализации

`tests/core_helpers.py` создаётся/расширяется по мере Tasks 2, 10, 11, 13, 18. Импорты поздних модулей держать внутри функций, чтобы ранние tasks не падали из-за ещё не созданного trainer.

```python
from pathlib import Path
import numpy as np
from snks.env.core_types import Observation, ActionSpec, GoalSpec, Transition

def make_observation(value=0, sensor=0.0, schema="toy", step=0):
    return Observation(np.full((3,64,64), value, dtype=np.uint8),
                       np.asarray([sensor], dtype=np.float32),
                       np.asarray([True]), schema, step)

def make_core_bundle():
    from snks.pipeline.core_config import load_core_config
    from snks.encoder.core_encoder import CoreEncoder
    from snks.agent.core_world_model import CoreWorldModel
    from snks.learning.core_trainer import CoreTrainer
    from snks.learning.core_replay import SequenceReplay
    config = load_core_config(Path("configs/core_smoke.yaml"))
    model = CoreWorldModel(CoreEncoder(config.z_dim), {"toy": (2, 1)},
                          config.h_dim, config.ensemble_size)
    return model, CoreTrainer(model, config), SequenceReplay(8, 0)

def make_sequence_batch():
    import torch
    from snks.learning.core_trainer import SequenceBatch
    return SequenceBatch(
        rgb=torch.rand(2,4,3,64,64),
        sensors=torch.tensor([[[0.],[1.],[1.],[2.]]]).repeat(2,1,1),
        sensor_mask=torch.ones(2,4,1, dtype=torch.bool),
        actions=torch.tensor([[1,0,1],[1,0,1]]),
        terminated=torch.zeros(2,3), valid=torch.ones(2,3,dtype=torch.bool),
        schema="toy", burn_in=1)

class TinyAdapter:
    actions = ActionSpec("toy", ("hold", "increment"))
    reset_transitions = 0

    def reset(self, seed):
        self.obs = make_observation()
        return self.obs

    def step(self, action):
        before = self.obs
        self.obs = make_observation(sensor=float(before.sensors[0]) + int(action == 1),
                                    step=before.step + 1)
        return Transition(before, action, self.obs, self.obs.step == 3, False)

    def close(self):
        return None

def make_toy_case(split):
    from snks.pipeline.core_tasks import TaskCase
    return TaskCase("toy-case", "toy", "toy-rule", 0, split,
                    GoalSpec(None, {0: (1.0, 3.0)}), 3)

def make_grid_case(split, consume_key):
    from snks.pipeline.core_tasks import TaskCase
    rule = "key_consumed" if consume_key else "key_reusable"
    return TaskCase("grid-case", "door_key", rule, 0, split,
                    GoalSpec(None, {0: (0.0, 1.0)}), 32)
```

## Матрица покрытия спеки

| Требование | Реализация и проверка |
|---|---|
| Цель, alternatives, legacy isolation | Header, Tasks 1, 3, 13, 20 |
| Observation/action contract, отсутствие privileged leakage | Tasks 2–5, 13–14 |
| Spatial encoder/history/action dynamics | Tasks 7–10 |
| SIGReg, masks, отсутствие supervised tile labels | Tasks 9–10 |
| Реальный replay и обучение между эпизодами | Tasks 6, 10, 13 |
| Checkpoint/schema/normalization/RNG и отсутствие forgetful reload | Tasks 11, 16 |
| Goals, beam search, budgets, uncertainty | Tasks 8, 12–14 |
| G0: split/mode/runtime integrity | Tasks 1–6, 13, 17 |
| G1: прогнозы, horizons, action controls | Tasks 9–10, 14–15 |
| G2: fixed representation + actual behavior | Tasks 12–15 |
| G3: A→B, обе B families, matched source control | Tasks 4–5, 14, 16, 18–19 |
| G4: A→B→A и save/load | Tasks 11, 14, 16, 18–19 |
| G5: интервенции, ложные корреляции | Tasks 4, 14, 19 |
| G6: training seeds, paired CIs, full failure ledger | Tasks 14, 17–20 |
| HyperPC pilot и preregistration/user gate | Tasks 1, 17–19 |
| ASSUMPTIONS, ROADMAP, non-claims | Tasks 1, 19–20 |

## Риски и правило остановки

- **Неполный oracle или неисполняемая B-задача:** остановить affected gate до learning claim; проверить state/RNG и целевую задачу на development.
- **Плохое покрытие редких действий/событий:** публикация coverage обязательна; новый collecting profile требует нового run/manifest, не дополнения test вручную.
- **Нет полезного переноса:** это допустимый отрицательный результат; не менять TaskCase goal или source-control задним числом.
- **Shared encoder забывает A:** report retention по семье; replay ratio можно исследовать только в новой development итерации.
- **Hypothesis collapse или planner exploiting model:** trace позволяет проверить предсказание на реальных последствиях; красивые latent rollouts не являются gate.
- **Не хватает HyperPC для confirmatory:** остановка на M4 с exploratory report и явным решением пользователя о бюджете.

## Проверка плана перед передачей

- [x] Сверить каждый раздел спеки с матрицей и конкретными Tasks.
- [x] Проверить совпадение имён типов, методов, constructor arguments, filenames и CLI flags между Tasks.
- [x] Проверить, что prospective tests/CLI не выдаются за существующие и что команды не запускаются локально.
- [x] Проверить отсутствие незаполненных обязательных steps и недоказанных PASS claims.
- [x] Выдать пользователю выбор исполнения: `superpowers:subagent-driven-development` с review по задачам либо `superpowers:executing-plans` последовательно с checkpoints. Не начинать реализацию при передаче плана.

Проверки документа 2026-09-05: все 40 Python snippets синтаксически разбираются через AST без исполнения; whitespace check не обнаружил ошибок; проверены ссылки на существующие regression suites. Это проверка плана, не запуск pytest и не свидетельство работоспособности будущего ядра.
