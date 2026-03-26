# СНКС — Архитектурный ревью и направление развития

**Дата:** 2026-03-26
**Метод:** Multi-agent structured brainstorming (Скептик / Архитектор / Инженер / Когнитивист / Арбитр)
**Статус:** APPROVED

---

## Что реально достигнуто (честная оценка)

1. ✅ **Рабочая биологически-инспирированная динамика** — FHN + STDP + homeostasis формируют стабильные паттерны (СКС)
2. ✅ **Полный pipeline** — от пикселей до действия, все компоненты связаны (Stage 0–14)
3. ✅ **VSA/HDC система (HAC)** — математически корректная circular convolution bind/unbind/bundle
4. ✅ **Каузальный агент** — tabular (context, action) → effect, работает для small state space
5. ✅ **Инженерная работа** — device-agnostic, ROCm/CUDA/CPU, 287 тестов PASS

---

## Выявленные проблемы

### Критические (блокируют AGI-заявление)

| # | Проблема | Описание |
|---|---------|----------|
| C1 | **GOAL_SEEKING не работает** | Chicken-and-egg: goal_sks требует success, success требует goal_sks. success_rate ≈ 0% на KeyDoor. |
| C2 | **Catastrophic forgetting не тестировалось** | Ключевое свойство MVP заявлено, но ни один эксперимент его не проверяет. |
| C3 | **Generalization не тестировалась** | Нет тестов на новые окружения / unseen стимулы. |
| C4 | **HAC capacity overflow** | При >20 шагах истории bundle переполняется, предсказания деградируют в шум. |
| C5 | **DCAM не интегрирован в агентный цикл** | EpisodicBuffer и SSG реализованы, но не используются в exp29–31. |

### Архитектурные (снижают качество)

| # | Проблема | Описание |
|---|---------|----------|
| A1 | **SKSEmbedder — random static projection** | Не обучается. Называть "HAC embedding" вводит в заблуждение. |
| A2 | **Context coarsening слишком грубый** | 16 bins → огромные коллизии для тысяч SKS-конфигураций. |
| A3 | **GWS — pure size-based selection** | w_coherence=0, w_pred=0. Нет настоящей конкурентной селекции. |
| A4 | **STDP + rate-based — временной парадокс** | STDP работает на микровременной шкале, детектор — на макро. |

### Производительность

| # | Проблема | Описание |
|---|---------|----------|
| P1 | **N=5K / 9.41 steps/sec на CPU** | ROCm: torch.sparse_csr_tensor N=50K → 10+ мин инициализации. |
| P2 | **torch.compile отключён на AMD ROCm** | Re-trace per shape → навсегда. |
| P3 | **HAC на CPU** | torch.fft.rfft segfault на AMD ROCm gfx1151. |

---

## Решения по возражениям (Decision Log)

| Возражение | Решение | Статус |
|-----------|---------|--------|
| SKSEmbedder не обучается | Валидна как fixed reservoir encoder. Переименовать в "random projection". Обучаемый embedder — следующий этап. | ПРИНЯТО |
| HAC predict_next — шум при >1 паре | Нужен episodic K-pair buffer вместо единого bundle. | ПРИНЯТО |
| Context 16 bins | Заменить на LSH или frozenset-hash. | ПРИНЯТО |
| GOAL_SEEKING не работает | Bootstrap goal_sks из первого успешного эпизода (random walk). | КРИТИЧНО |
| GWS pure size | Упрощение, не нарушает MVP-цель. Полная GWT — следующий этап. | ОТЛОЖЕНО |
| STDP + rate-based | Теоретическая несогласованность, практически работает. | ИЗВЕСТНО |
| Coverage ≠ интеллект | Нужны behavioral тесты (forgetting, generalization, transfer). | КРИТИЧНО |
| HAC — short-term (decay 0.95 → ~20 steps) | Нужен replay/consolidation для долгосрочной памяти. | ПРИНЯТО |
| DCAM dead code | Требует реальной интеграции в агентный цикл. | ПРОВЕРИТЬ |
| Catastrophic forgetting не тестировано | Добавить exp33. | КРИТИЧНО |

---

## Стратегия развития

### Стратегия A: "Закрыть долги" (выбрана как ПЕРВООЧЕРЕДНАЯ)

Исправить критические проблемы текущей архитектуры перед движением вперёд.

```
Stage 15a — закрытие долгов:
├── Fix GOAL_SEEKING bootstrap (exp32)
├── Fix HAC capacity: episodic K-pair buffer (exp33)
├── Catastrophic forgetting test (exp34)
├── Context hashing: LSH/frozenset вместо 16-bin (улучшение CausalWorldModel)
└── DCAM реальная интеграция в агентный цикл
```

### Стратегия B: "Следующий уровень" (после A)

```
Stage 15b — новые возможности:
├── Language Grounding: text → goal_sks → action
├── Более сложное окружение (не 8×8 DoorKey)
└── Transfer learning test
```

### Стратегия C: "Перезапуск ядра" (долгосрочно)

Заменить FHN на Reservoir Computing (Echo State Network) для N=100K+. Сохранить надстройки (HAC, DCAM, CausalAgent, GWS). Решить проблему производительности.

---

## Финальный вердикт арбитра

**APPROVED — REVISE**

Система является **обещающим proof-of-concept нейроморфной архитектуры**, но ещё не является **AGI proof-of-concept** в заявленном смысле:
- Основные задачи (KeyDoor) не решаются
- Ключевые свойства (no catastrophic forgetting) не верифицированы
- HAC предсказания деградируют при реальном использовании

**Рекомендуется: Стратегия A (закрыть долги), затем Стратегия B.**
