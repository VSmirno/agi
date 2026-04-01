# Autonomous Development Log — 2026-04-01 — Stage 40

## Текущая фаза: 1 — Живой DAF, прогресс ~13%

Маркеры завершения фазы:
- Pure DAF >= 50% на DoorKey-5x5 (сейчас 12%)
- Emergent: формирование "привычек"
- Emergent: перенос Empty → DoorKey

## Stage 40: Learnable Encoding

**Цель:** Сделать encoder обучаемым через Hebbian правило (STDP-подобное, локальное).
Текущий Gabor encoder замороженный — агент не может адаптировать представление.

### Фаза 0: Git setup
- Ветка: stage40-learnable-encoding от main (commit 64674fb)
- Предыдущие stages: 0-39 COMPLETE, Pure DAF 12% DoorKey-5x5
- Ключевая проблема: frozen Gabor теряет task-relevant информацию

### Фаза 1: Спецификация
- Подход A: Oja's Hebbian Rule (self-normalizing)
- Подход B: Competitive Hebbian + lateral inhibition
- Подход C: Temporal Contrastive (backprop) — ОТКЛОНЁН (нарушает философию СНКС)
- **Выбран: A→B** — начали с Oja, перешли на Sanger's GHA + competitive selection
- Обоснование: plain Oja сводит все фильтры к PC1, Sanger декоррелирует
- Spec-review: 4 must-fix (PE timing, weight access, self-norm), исправлены

### Фаза 2: Реализация
- HebbianEncoder: 170 строк, наследует VisualEncoder
- Sanger's GHA с competitive selection (top-25% winners)
- PE modulation через Pipeline.mean_pe
- Diversity regularization каждые 50 обновлений
- 17 тестов PASS

### Фаза 3: Эксперименты
- exp99a: SDR discrimination 0.232→0.199 (improving) PASS
- exp99b: Filter diversity 0.89 (>0.5) PASS
- exp99c: Hebbian convergence delta ratio 0.28x PASS
- exp99d: DoorKey runs without error PASS
- exp99e: Learning curve positive PASS

### Фаза 4: Веб-демо
- demos/stage-40-learnable-encoding.html — фильтры до/после, SDR comparison, charts

### Фаза 5: Merge
- Merged stage40-learnable-encoding → main

### Решения
1. **Sanger вместо Oja**: plain Oja → все фильтры к PC1 (0.33 overlap). Sanger's triangular decorrelation → разные компоненты (0.20 overlap).
2. **Competitive selection**: только top-25% фильтров обновляются — латеральное торможение как в коре.
3. **PE из Pipeline.mean_pe**: доступен внутри perception_cycle после DAF step. Не из DafCausalModel (доступен позже).
4. **Weight clamp обязателен**: PE modulation ломает self-normalization Oja.
