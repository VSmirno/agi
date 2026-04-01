# Stage 44: Foundation Audit — Design Spec

**Date:** 2026-04-02
**Phase:** 1 — Живой DAF
**Type:** Audit / Verification (не новая фича)

## Мотивация

После Stage 38 (Pure DAF) мы обнаружили, что чистый DAF-pipeline без scaffolding даёт ~5% на DoorKey-5x5 даже с идеальным perception (SymbolicEncoder). Stages 38–43 добавляли механизмы (eligibility traces, WM, curriculum) реактивно — каждый как ответ на конкретный провал, а не из архитектурного плана.

Stage 42 выявил скрытый дефект (perception blind), который маскировался месяцами. **Нет гарантий, что аналогичных дефектов нет в других слоях.**

Прежде чем добавлять новые механизмы, нужно:
1. Убедиться, что каждый существующий слой работает корректно
2. Проверить, что чистое DAF-ядро (без надстроек) способно обучаться

## Что НЕ входит в scope

- Новые механизмы, фичи, оптимизации
- Тюнинг параметров (цель — проверка корректности, не подгонка)
- Изменение архитектуры (если аудит выявит проблему — это отдельный stage)

---

## Фаза 0: Golden Path (быстрый сигнал)

**Цель:** за минимальное время ответить — DAF-ядро вообще учится?

### Среда
- Кастомная 3×3 grid, один объект (goal), никаких ключей/дверей
- Оптимальное решение: 2-3 шага
- Reward: +1 при достижении goal

### Агент
- 500 FHN нод (минимум для SKS)
- SymbolicEncoder (идеальное perception)
- Без WM, без eligibility traces, без curriculum, без AttractorNavigator
- Только: encoder → FHN → STDP → SKS → action selection (epsilon-greedy + PE)

### Ожидания
- После 20 эпизодов: success rate > 50% (random baseline ~10-15% на 3×3)
- STDP веса должны измеримо отличаться между success/fail эпизодами

### Gate
- **PASS:** success > 50% AND STDP weight delta (with reward) > 2× STDP weight delta (no reward control)
- **FAIL:** success ≤ random OR weight delta не отличается от no-reward control → СТОП, локализация проблемы

### При FAIL — диагностика
Пошагово определить, где ломается цепочка:
1. Encoder выдаёт различимые SDR для разных позиций? (overlap test)
2. FHN формирует различимые activation patterns для разных SDR? (pattern separation)
3. STDP меняет веса в ответ на reward? (weight delta test)
4. SKS кластеры стабильны и различимы между input-ами? (cluster stability)
5. Action selection учитывает SKS? (action-SKS correlation)

---

## Фаза 1: Послойный аудит

Независимо от результата Фазы 0 — проверяем каждый слой. Если Фаза 0 FAIL, аудит поможет локализовать; если PASS — убеждаемся в отсутствии скрытых дефектов.

### 1.1 FHN Oscillator Dynamics
**Файл:** `src/snks/daf/oscillator.py`

**Тесты:**
- Подать постоянный ток I=0.5 на одиночный осциллятор → проверить период и амплитуду колебаний аналитически (FHN имеет известные bifurcation properties)
- Подать I=0 → осциллятор должен быть в покое (stable fixed point)
- Подать ток на 100 осцилляторов → проверить что нет численного дрейфа за 10000 шагов (mean v, mean w стабильны)
- Проверить что dt=0.1 не вызывает нестабильности (сравнить с dt=0.01 reference)

**Что ищем:** численная нестабильность, дрейф, вырождение, неожиданное поведение при граничных значениях.

### 1.2 STDP Weight Updates
**Файл:** `src/snks/daf/stdp.py`

**Тесты:**
- Два осциллятора, pre fires before post (Δt=+5ms) → вес должен расти (LTP)
- Pre fires after post (Δt=-5ms) → вес должен падать (LTD)
- Без корреляции (случайные спайки) → вес ~стабилен
- Reward modulation: одинаковая корреляция, reward=1 vs reward=0 → разница в Δw
- Homeostasis: проверить что при target rate 5% и actual rate 20% — веса уменьшаются, но не обнуляются
- **Критический тест:** homeostasis не убивает learned signal. 100 шагов обучения → homeostasis → проверить что learned weights сохранили относительный порядок

**Что ищем:** homeostasis подавляет learning, reward modulation не работает, LTP/LTD инвертированы.

### 1.2b Coupling / Connectivity
**Файлы:** `src/snks/daf/engine.py`, coupling matrix initialization

**Тесты:**
- 10 нод, известная coupling matrix (A→B=1.0, rest=0) → спайк в A вызывает отклик в B через ~τ ms
- 50 нод, 3 группы, внутригрупповой coupling=1.0, межгрупповой=0 → группы синхронизируются внутри, не между
- Coupling matrix sparse (CSR) vs dense → идентичные результаты
- После STDP update: coupling weights реально обновляются в матрице (не теневой копии)
- Topology: проверить что connection count и distribution соответствуют DafConfig

**Что ищем:** coupling не передаёт спайки, STDP обновляет теневую копию весов, топология не соответствует конфигу.

### 1.3 SKS Detection
**Файл:** `src/snks/sks/detection.py`

**Тесты:**
- 100 нод, 3 группы по 20 с высокой внутригрупповой когерентностью → DBSCAN должен найти 3 кластера
- Те же группы, повторить 10 раз → кластеры воспроизводимы (Jaccard index > 0.8)
- Два разных input паттерна → разные SKS (overlap < 0.3)
- Один и тот же input → одинаковый SKS (overlap > 0.7)
- Шум без структуры → нет значимых кластеров (или один большой)

**Что ищем:** нестабильность кластеров, неспособность различать паттерны, ложные кластеры из шума.

### 1.4 Encoder → SKS Pipeline (end-to-end)
**Файл:** `src/snks/pipeline/runner.py`

**Тесты:**
- SymbolicEncoder: 3 разных observation → 3 различимых SDR (pairwise overlap < 0.5)
- SDR → FHN injection → perception_cycle → SKS: один и тот же input 5 раз → SKS overlap > 0.6
- Два разных input-а → SKS overlap < 0.4
- **Критический тест:** после perception_cycle reset, предыдущий input не "протекает" в следующий цикл (ghost signal test)

**Что ищем:** perception_cycle reset ломает что-то, SDR не доходит до SKS, coupling structure не передаёт signal.

### 1.5 Action Selection
**Файл:** `src/snks/agent/pure_daf_agent.py`

**Тесты:**
- При epsilon=1.0: действия равномерны (chi-square test, 1000 samples)
- При epsilon=0.0: действия определяются PE/goal-directed
- PE-driven: prediction error для одного действия искусственно выше → это действие выбирается чаще
- После STDP обучения с reward за action=2: при epsilon=0 action=2 выбирается чаще чем до обучения

- **Критический тест:** action index → environment action mapping. Action 0 = move forward, action 1 = turn left, и т.д. Проверить что нет off-by-one или permutation.

**Что ищем:** epsilon не влияет, PE не влияет на выбор, STDP learning не доходит до action selection, action mapping перепутан.

---

## Фаза 2: Naked DAF на DoorKey-5x5

**Предусловие:** Фаза 0 PASS, Фаза 1 без критических дефектов.

### Конфигурация
- 50K FHN нод (GPU, minipc)
- SymbolicEncoder (идеальное perception)
- Без WM (wm_fraction=0.0)
- Без eligibility traces
- Без curriculum
- Без AttractorNavigator
- EpsilonScheduler: 0.7 → 0.1
- 200 эпизодов (50 недостаточно для spiking network на multi-step task)

### Gate
- **PASS:** success_rate ≥ 0.15 (random baseline ~2%)
- **PARTIAL:** 0.05 ≤ success_rate < 0.15 → ядро учится, но медленно — штатная ситуация для дальнейших improvements
- **LEARNING SIGNAL:** success < 0.05, но STDP weight structure улучшается (correlation с task-relevant actions растёт) → архитектура жива, но нужно больше эпизодов или temporal credit
- **FAIL:** success_rate < 0.05 AND нет learning signal → фундаментальная проблема в архитектуре

---

## Порядок выполнения

| Шаг | Где | Зависимости | Ожидаемое время |
|-----|-----|-------------|-----------------|
| Фаза 0: Golden Path | CPU (local) | — | ~15 мин (20 эпизодов, 500 нод) |
| Фаза 1.1: FHN | CPU (local) | — | ~10 мин |
| Фаза 1.2: STDP | CPU (local) | — | ~10 мин |
| Фаза 1.2b: Coupling | CPU (local) | — | ~10 мин |
| Фаза 1.3: SKS | CPU (local) | — | ~10 мин |
| Фаза 1.4: Pipeline | CPU (local) | 1.1, 1.2, 1.2b, 1.3 | ~15 мин |
| Фаза 1.5: Action Selection | CPU (local) | 1.4 | ~10 мин |
| Фаза 2: Naked DAF DoorKey | GPU (minipc) | Фаза 0 PASS, Фаза 1 clean | ~1-2 часа (200 эпизодов, 50K нод) |

Фазы 0, 1.1–1.3 можно запускать параллельно. Если любой тест занимает >3× ожидаемого — это само по себе сигнал проблемы.

---

## Ожидаемые исходы

**Лучший случай:** Все слои чистые, golden path PASS, naked DAF ≥15%. Фундамент работает, можно планировать следующие stages из роадмапа (не реактивно).

**Средний случай:** Обнаружены 1-2 дефекта уровня "perception blind". Фиксим, перепроверяем. Это именно то, ради чего мы делаем аудит.

**Худший случай:** Golden path FAIL даже на 3×3, слой X фундаментально не работает как задумано. Это болезненно, но это ответ на вопрос "архитектура работает?" — и он ценнее чем ещё 10 reactive stages.

---

## Принципы

- **Никаких изменений в production code** до завершения аудита (только тестовый код)
- **Все тесты воспроизводимы** — фиксированные seed-ы, детерминированные условия
- **Каждый FAIL документируется** с точной локализацией и evidence
- **Не тюним параметры** — используем текущие defaults. Цель: проверка корректности, не оптимизация.
