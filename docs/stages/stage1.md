# Этап 1: ДАП-Движок ✅

**Статус:** Завершён (2026-03-22)
**Срок:** Неделя 2–4
**Критический путь:** Да, блокирует всё остальное

## Модули

7 модулей в `src/snks/daf/`:

| Модуль | Назначение |
|--------|-----------|
| `graph.py` | SparseDafGraph: COO граф, scatter_add coupling |
| `oscillator.py` | FHN / Kuramoto dynamics |
| `coupling.py` | scatter_add coupling term |
| `integrator.py` | Euler-Maruyama SDE integrator |
| `stdp.py` | STDP + гомеостатический член + lr_modulation |
| `homeostasis.py` | Threshold adaptation (EMA firing rate) |
| `engine.py` | DafEngine facade: step(), set_input(), StepResult |

## Динамика

**FitzHugh-Nagumo:**
```
dv/dt = v - v³/3 - w + I
dw/dt = (v + a - b*w) / τ
```

**Kuramoto (fallback):**
```
dφ/dt = ω_i + K/N × Σ sin(φ_j - φ_i) + σ·η
```

**Интегратор:** Euler-Maruyama (стохастические ODE)
```
x(t+dt) = x(t) + dt·f(x) + √dt·σ·η
```

## STDP

```
Δw_ij = A+·exp(-|Δt|/τ+)  если pre→post (potentiation)
      = -A-·exp(-|Δt|/τ-)  если post→pre (depression)
      + λ·(w_target - w_ij)  гомеостаз
```

Параметры: A+=0.01, A-=0.012, τ=20, λ=0.001, w∈[0,1].
Rate-based Hebbian STDP добавлен для случая, когда timing STDP не работает.

## Gate

- Kuramoto r > 0.7: ✅ (1K узлов, K=3.0, dt=0.01, 5000 шагов)
- GPU perf ≥ 10K steps/sec (50K nodes): ⏳ ожидает бенчмарка

## Тесты

71 тест — все проходят.

## Отклонения от плана

- `stdp_w_target = 0.5` добавлен в DafConfig
- `enable_learning` в DafEngine для отключения STDP в тестах
- Частоты ω — Normal(0,1) с clamp(−5,5), не Cauchy
- coupling_strength ≥ 2.0 необходим для Kuramoto (K_c ≈ 1.6σ)
- Edge init weights: rand()*0.5 (было 0.1)
