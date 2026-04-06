# Идея: Survival-агент поверх Stage 66

## Суть

Расширить goal 9 (free_explore) в демо до полноценного выживания в Crafter:
пить воду, есть еду, уходить от зомби.

## Что модель умеет (проверено эмпирически)

| Задача | Источник знания | Качество |
|--------|----------------|----------|
| do near water → drink | neocortex | conf=0.95 ✓ |
| do near tree → wood | neocortex + prototype | conf=0.96 ✓ |
| Позиции zombie/skeleton | semantic map 64×64 | всегда точно ✓ |
| do near cow → food | prototype memory | conf=0.97, но neocortex путает с MiniGrid |
| do near zombie | нет прототипов | ✗ |

**Pixel-запрос near water дал "wood" (conf=0.92)** — энкодер смешивает воду и дерево.
Для надёжного выживания pixel-path недостаточен.

## Архитектура гибридного агента

```
Приоритетная очередь (эвристика):
  1. Zombie dist < 4  → flee (semantic map)
  2. drink < 4/9     → find water, do (neocortex ✓)
  3. food < 4/9      → find cow + wood_sword (prototype + plan)
  4. energy < 3/9    → sleep action
  5. health < 5/9    → avoid combat
  6. else            → explore + collect resources

Каждое действие:
  → query_from_pixels() → показываем предсказание WM в UI
  → execute
  → observe outcome
```

## Что нужно для реализации

- Только правки в `demos/stage66_demo.py` (новый `SurvivalAgent` класс)
- **Дообучение НЕ нужно** для базового выживания
- Добавить UI: полоски здоровья/еды/воды/энергии

## Что нужно для честного pixel-based выживания

Дообучить модель на переходах с выживанием:
- Прототипы `do near water` с чёткой ситуацией "рядом с водой"
- Прототипы `do near cow` (с sword в инвентаре)
- Переходы урона от zombie (агент получает урон)

Это отдельный этап — Stage 67 или расширение Stage 66.

## Ограничения без дообучения

- Корова: neocortex возвращает мусор (MiniGrid fallback)
- Атака на врага: нет прототипов
- Pixel-discriminability: вода/дерево неразличимы в z-пространстве

## Статус

Идея, не реализована. Обсуждалась после Stage 66 demo.
