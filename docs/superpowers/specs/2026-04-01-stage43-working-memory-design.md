# Stage 43: Working Memory — Sustained Oscillation

**Дата:** 2026-04-01
**Статус:** DESIGN
**Автор:** autonomous-dev

## Позиция в фазе

**Фаза 1: Живой DAF (~70% → 80%)**

Stage 42 доказал dual bottleneck: даже идеальная perception → 5% success. Root cause анализ показал: `perception_cycle()` **сбрасывает ВСЕ состояния осцилляторов** на каждом шаге (runner.py:203-210). Агент получает амнезию каждый цикл — не может удержать "ищу ключ" между шагами.

Маркеры Stage 43:
- [ ] Sustained oscillation: подмножество нод сохраняет активацию между cycles
- [ ] Goal retention: агент помнит подцель > 10 шагов
- [ ] Контекстное действие: action selection учитывает WM state

## Проблема

```python
# runner.py:203-210 — КАЖДЫЙ perception_cycle():
self.engine.states[:, 0] = torch.randn(N) * 0.1  # v → random noise
self.engine.states[:, 4] = 0.0                     # w_recovery → zero
```

Это означает:
1. **Нет памяти между шагами** — осцилляторы забывают всё
2. **SKS кластеры каждый раз формируются заново** — от текущего стимула, без контекста
3. **Causal model не получает consistent state** — before_action snapshot бессмыслен если state сброшен
4. **STDP weights учатся** (они сохраняются), но **активация не сохраняется**

Аналогия: это как если бы мозг каждую секунду засыпал и просыпался — синапсы есть, но рабочая память стёрта.

## Brainstorming

### Подход A: WM Buffer Zone (ВЫБРАН)

Разделить осцилляторы на две группы:
- **Perceptual zone** (~80% нод): сбрасывается каждый cycle (как сейчас) — для чистого восприятия текущего стимула
- **WM zone** (~20% нод): **НЕ сбрасывается** — sustained activation удерживает контекст

WM zone получает input через STDP-связи с perceptual zone. Когда агент видит ключ, перцептивные ноды активируют "ключ-паттерн" в WM zone через сильные связи. На следующем cycle perceptual zone сбрасывается, но WM zone **продолжает резонировать** с паттерном "ключ".

**Pros:** естественный DAF механизм (осцилляторы и так умеют sustain), минимальные изменения кода, биологически правдоподобно (prefrontal cortex = sustained activation)
**Cons:** WM zone может "залипнуть" на старом паттерне, нужен механизм gating
**Trade-off:** простота vs gating — начинаем без gating, добавим если нужно

### Подход B: Recurrent State Carry

Не сбрасывать states вообще — каждый cycle продолжает с предыдущего состояния.

**Rejected:** без сброса новый стимул не может "протолкнуть" свой паттерн — старая активация доминирует. Perception деградирует.

### Подход C: External Working Memory (отдельный буфер)

Хранить WM как отдельный тензор вне DAF engine.

**Rejected:** это не DAF mechanism, это RL-style external memory. Противоречит философии СНКС.

## Дизайн

### 1. WM Zone в DafEngine

Добавить `wm_zone` в DafConfig — диапазон нод, которые НЕ сбрасываются.

```python
@dataclass
class DafConfig:
    # ... existing ...
    wm_fraction: float = 0.2  # 20% нод = WM zone
```

В DafEngine: первые `N * (1 - wm_fraction)` нод = perceptual, последние `N * wm_fraction` = WM.

### 2. Selective Reset в Pipeline

```python
# runner.py — ВМЕСТО полного сброса:
n_perceptual = int(N * (1 - self.engine.config.wm_fraction))
# Reset ONLY perceptual zone
self.engine.states[:n_perceptual, 0] = torch.randn(n_perceptual) * 0.1
self.engine.states[:n_perceptual, 4] = 0.0
# WM zone: states PRESERVED from previous cycle
```

### 3. WM Decay (Anti-Lock)

WM zone не сбрасывается, но **затухает** с каждым cycle чтобы не "залипнуть":

```python
wm_start = n_perceptual
# Soft decay: v *= wm_decay (e.g., 0.95)
self.engine.states[wm_start:, 0] *= wm_decay
```

При `wm_decay=0.95`, activation полностью затухает за ~60 cycles если не подкрепляется.

### 4. WM → Action Selection

Добавить WM state как дополнительный context для action selection:
- Вычислить WM embedding: mean firing rate WM нод → embedding
- Конкатенировать с current perceptual embedding для navigator
- Или: WM-активные ноды участвуют в motor selection через coupling

### 5. WM Content Monitoring

Для диагностики и демо:
- Считать mean activation WM zone
- Detect dominant pattern в WM (кластеризация WM нод)
- Записывать в CycleResult: `wm_activation`, `wm_pattern_id`

## Gate-критерии

| Test | Метрика | Gate |
|------|---------|------|
| WM zone preserves state | activation at cycle t+1 > 0 after stimulus at cycle t | True |
| WM decay works | activation → 0 after 60 cycles without stimulus | True |
| Perceptual zone still resets | clean perception of new stimulus | True |
| WM helps DoorKey | success_rate(WM) > success_rate(no-WM) | True |
| No regression | existing tests pass | True |
| WM retention | stimulus remembered >= 10 cycles | True |

## Файлы

### Новые
- `src/snks/experiments/exp102_working_memory.py` — experiments
- `tests/test_working_memory.py` — unit tests

### Изменяемые
- `src/snks/daf/types.py` — wm_fraction, wm_decay
- `src/snks/pipeline/runner.py` — selective reset
- `src/snks/agent/pure_daf_agent.py` — WM context in action selection
- `demos/index.html` — карточка Stage 43
