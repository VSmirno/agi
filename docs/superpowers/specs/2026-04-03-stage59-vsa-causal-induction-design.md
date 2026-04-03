# Stage 59 (revised): VSA+SDM Few-Shot Causal Induction

**Дата:** 2026-04-03
**Статус:** SPEC
**Предыдущий:** Stage 58 (SDM Retrofit — negative), Stage 59 attempts on LockedRoom/ObstructedMaze (negative)

---

## Мотивация

Stages 47-58: символический BFS решает всё, SDM не добавляет value. Stage 59 attempts: heuristic оптимален (LockedRoom), exploration не работает (ObstructedMaze). Архитектурный тупик: SDM = lookup table, не learning.

**Ключевое открытие:** VSA binding имеет свойство `bind(X, X) = identity vector` для любого X. Это означает что "sameness" имеет уникальную математическую сигнатуру в VSA пространстве. SDM может хранить reward по этой сигнатуре и обобщать на unseen colors.

## Что доказываем

Few-shot causal induction: 3 демонстрации (key_color, door_color, success/fail) → SDM обобщает правило same_color → success на unseen цвета. Без нейросетей, без heuristics, чистый VSA+SDM.

## Архитектура

```
Демонстрация: (key_color, door_color, success)
    │
    ▼
VSACodebook: encode colors → binary vectors
    │
    ▼
bind(VSA(key_color), VSA(door_color)) → relationship vector
    │
    ▼  
SDM.write(relationship, reward=±1.0)
    │
Query: (new_key_color, new_door_color)
    │
    ▼
bind(VSA(new_key), VSA(new_door)) → relationship vector
    │
    ▼
SDM.read_reward(relationship) → positive=opens, negative=doesn't
```

### Почему обобщает

- `bind(red, red) = bind(blue, blue) = bind(green, green) = zero_vector` (XOR identity)
- `bind(red, blue) = random_vector_1` (unique per pair)
- SDM: zero_vector → reward=+1, random → reward=-1
- Test (green, green) → zero_vector → SDM returns +1 → correct!

## Эксперименты

### Phase A: Same-color generalization
- Train: {red, blue, yellow} — 3 same-color demos (+1.0), 6 different-color demos (-1.0)
- Test: {green, purple, grey} — all 9 pairs (3 same, 6 different)
- Gate: ≥90% accuracy on unseen colors

### Phase B: Scaling
- Train with 1, 2, 3, 4, 5 colors
- Test on remaining colors
- Gate: accuracy monotonically increases

### Phase C: Arbitrary mapping (memorization)
- Train: red→blue, blue→green, green→red (rotation, not same-color)
- Test seen pairs: accuracy ≥80%
- Test unseen pairs (yellow→?): accuracy ~50% (cannot generalize rotation)
- Gate: clear difference between generalization (Phase A) and memorization (Phase C)

## Gate Criteria

| Test | Метрика | Порог |
|------|---------|-------|
| A: generalization | accuracy unseen same/diff | ≥ 90% |
| A: ablation | trained - untrained delta | ≥ 40% |
| B: scaling | monotonic growth | yes |
| C: seen pairs | accuracy | ≥ 80% |
| C: unseen pairs | accuracy | ~50% |

## Файлы

| Файл | Описание |
|------|----------|
| `src/snks/experiments/exp114_vsa_causal.py` | Всё в одном: train, test, ablation, scaling, arbitrary |
| `tests/test_vsa_causal_induction.py` | Unit тесты: identity property, SDM storage, generalization |
