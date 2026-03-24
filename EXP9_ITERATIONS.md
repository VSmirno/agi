# Experiment 9: Curiosity-driven Exploration — Iterations & Results

**Goal:** Coverage ratio > 1.5 (curious agent beats random by 50%)

---

## Baseline (Commit 796997b)
- **Coverage ratio:** 0.69 (FAIL)
- **Curious coverage:** 0.30
- **Random coverage:** 0.43
- **Problem:** Curious agent worse than random

---

## Iteration 1: Rotation-invariant perceptual hash + Learning progress
**Hypothesis:** DAF-шум (noisy SKS clusters) мешает learning. Нужна stable, rotation-invariant перцептивная хеш-функция + learning progress для suppression noisy-TV.

**Changes:**
- `_perceptual_hash()`: Sorted intensity-based (rotation-invariant) вместо позиционного
- `_learning_progress`: EMA delta prediction error вместо raw prediction error
- `_stable_context()`: Фильтр only perceptual hash IDs (>= 10000), игнор DAF-шум

**Result (Commit b679bc4):**
- **Coverage ratio:** 0.788 (улучшение! но FAIL)
- **Curious coverage:** 0.383
- **Random coverage:** 0.487
- **Root cause found:** Full hash + learning progress недостаточны. Causal model использует coarsened hash (16 bins) → все предсказания одинаковые → interest одинаков для всех действий.

---

## Iteration 2: Pure count-based + learning progress
**Hypothesis:** Prediction error слишком шумный. Убрать learning progress suppression, полагаться только на visit counts.

**Changes:**
- Убрать `_learning_progress` из select_action
- Простая формула: `interest = 0.6 * state_novelty + 0.4 * action_novelty`

**Result (Commit deb1bff):**
- **Coverage ratio:** 0.836 (tiny улучшение)
- **Curious coverage:** 0.390
- **Random coverage:** 0.467
- **Conclusion:** Learning progress suppression — не проблема. Основная проблема — predict_effect использует coarsened hash.

---

## Iteration 3: Pure visitor counting (NO predict_effect)
**Hypothesis:** `causal_model.predict_effect()` бесполезен (coarsened hash делает все предсказания одинаковыми). Убрать predict_effect полностью, использовать только visit counts.

**Changes:**
- Убрать `causal_model.predict_effect()` из select_action
- Оставить только: `interest = 1.0 / (1.0 + visit_count)`

**Result (Commit 175a5ef):**
- **Coverage ratio:** 1.035 ✅ (БОЛЬШЕ чем random!)
- **Curious coverage:** 0.490 > Random: 0.473
- **Avg causal links:** 498
- **BREAKTHROUGH:** Curiosity работает! Нужно optimise к 1.5.

---

## Iteration 4: Add state novelty heuristic
**Hypothesis:** Без predict_effect, можем использовать visit_count как proxy для state novelty.

**Changes:**
- `state_novelty = 1.0 - (visit_count / (visit_count + 10.0))`
- `interest = 0.8 * state_novelty + 0.2 * action_novelty`

**Result (Commit 20cd6f6):**
- **Coverage ratio:** 1.090 (улучшение!)
- **Curious coverage:** 0.487 > Random: 0.447
- **Avg causal links:** 514
- **Still FAIL:** 1.090 < 1.5, но trend правильный

---

## Iteration 5: Tune epsilon 0.2 → 0.1
**Hypothesis:** Меньше random actions → более directed exploration.

**Changes:**
- `curiosity_epsilon: 0.2` → `0.1`

**Status:** Running on minipc...

---

## Key Insights

1. **DAF noise is real:** Full hash vs coarsened hash — fundamental problem
2. **predict_effect is useless:** Coarsened hash (16 bins) делает все предсказания одинаковыми
3. **Pure count-based works:** Простой visitor counting > prediction error
4. **Rotation-invariant hash helps:** Но недостаточно без других fixes
5. **Need state exploration:** Prefer actions leading to less-visited contexts

---

## Architecture Decision

**Final approach:** Pure count-based visitor counting on stable perceptual hash:
- Перцептивная хеш = rotation-invariant + deterministic
- Action selection = только visit counts, NO prediction
- Weights: 80% state novelty, 20% action novelty
- Epsilon-greedy: 0.1 (10% random)
