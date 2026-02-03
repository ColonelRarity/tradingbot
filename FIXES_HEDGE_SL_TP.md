# Виправлення проблем з hedge та SL/TP

## Виявлені проблеми

### 1. ❌ PERCENT_PRICE error при закритті hedge

**Проблема:** Hedge не може закритися через помилку PERCENT_PRICE (-4131). Це означає, що ціна закриття занадто далеко від mark price.

**Виправлення:** Додано обробку PERCENT_PRICE error в `core/hedge_manager.py`:
- Якщо MARKET order не вдається через PERCENT_PRICE, використовується STOP_MARKET з `closePosition=True`
- Trigger price встановлюється дуже близько до mark price (0.01% відстань)

**Файли:** `core/hedge_manager.py`

### 2. ❌ Відсутні SL/TP для існуючих позицій

**Проблема:** У користувача дві SHORT позиції (SWELLUSDT, THEUSDT), обидві прибуткові, але на біржі немає SL/TP ордерів.

**Причини:**
- SL/TP могли не встановитися при відкритті (помилка)
- SL/TP були встановлені, але потім скасовані
- Позиції відкривалися без SL/TP (якась помилка)

**Потрібне виправлення:**
- Додати функцію для перевірки наявності SL/TP ордерів на біржі
- Якщо SL/TP відсутні, встановити їх на основі поточного стану позиції

### 3. ❌ Hedge відображається неправильно

**Проблема:** Користувач каже, що hedge обробляється некоректно. У нього дві SHORT позиції (SWELLUSDT, THEUSDT), обидві прибуткові, і hedge не має показуватися, якщо hedge немає.

**Поточна логіка:**
```python
if position.has_hedge:
    hedge = self.hedge_manager.get_hedge_for_parent(position.position_id)
    if hedge and hedge.is_open:
        position_label = f"{position.side}[HEDGE]"
    else:
        position.has_hedge = False
        position.hedge_id = None
```

**Примітка:** Логіка виглядає правильною - `[HEDGE]` показується тільки коли hedge існує і відкритий. Потрібно перевірити, чому `has_hedge` флаг встановлюється неправильно.
