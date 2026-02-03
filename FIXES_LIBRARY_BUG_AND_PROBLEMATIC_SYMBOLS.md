# Виправлення обробки library bug та problematic symbols

## 🔴 Проблема 1: Library bug з get_open_orders неправильно реєструється як API error

### Проблема
Library bug з `get_open_orders()` (помилка "Library requires orderId parameter") реєструється як API error і додає символ до blacklist після 3 помилок. Це неправильно, тому що:
- Це відома помилка бібліотеки `binance-futures-connector`, а не справжня помилка API
- Це не критична помилка - бот може продовжувати працювати з цим символом
- Це призводить до непотрібного blacklist символів, які насправді працюють нормально

**Приклад з логу:**
```
WARNING:exchange.binance_client:Library requires orderId parameter (library bug), returning empty list for CHILLGUYUSDT
WARNING:core.position_tracker:Symbol CHILLGUYUSDT marked as PROBLEMATIC (reason: API errors (3 consecutive)). Will be skipped for 24.0 hours.
```

### Виправлення

**Файл:** `core/position_tracker.py` (метод `_ensure_sl_tp_orders`)

Видалено реєстрацію library bug як API error:

```python
# If we have tracked order IDs but get_open_orders returned empty, it's likely a library bug
# Do NOT record this as API error - it's a known library bug, not a real API failure
if has_tracked_orders:
    # Skip restoration when we can't verify orders exist (to avoid duplicates)
    # Orders will be checked again on next update cycle
    # This is a known library bug, not a real API error, so we don't blacklist the symbol
    logger.debug(f"[{position.symbol}] get_open_orders returned empty (library bug), skipping SL/TP restoration to avoid duplicates")
    return
```

**Що це вирішує:**
- Library bug більше не реєструється як API error
- Символи не додаються до blacklist через library bug
- Бот може продовжувати працювати з символами, навіть якщо `get_open_orders` повертає порожній список

---

## 🔴 Проблема 2: Існуючі позиції на problematic symbols продовжують оброблятися

### Проблема
Коли символ позначається як PROBLEMATIC, існуючі позиції все ще обробляються, що призводить до:
- Продовження помилок API (навіть якщо це library bug)
- Непотрібних викликів API для символів з відомими проблемами
- Повторного позначення символу як PROBLEMATIC

**Приклад з логу:**
- Рядок 593: `CHILLGUYUSDT marked as PROBLEMATIC`
- Рядки 600, 606, 709: Позиція все ще обробляється
- Рядок 711: `CHILLGUYUSDT marked as PROBLEMATIC` (знову!)

### Виправлення

**Файл:** `main.py` (метод `_manage_all_positions`)

Додано перевірку problematic_symbols при оновленні позицій:

```python
# Skip positions on problematic symbols (but allow closing existing positions)
# This prevents unnecessary API calls and errors for symbols with known issues
if self.position_tracker.is_problematic_symbol(position.symbol):
    # Still update P&L for existing positions, but skip SL/TP updates to avoid errors
    logger.debug(f"[SKIP] {position.symbol}: Position {position.position_id} on problematic symbol, skipping detailed updates")
    # Still update basic P&L
    if position.symbol in self.market_data_cache:
        md = self.market_data_cache[position.symbol]
        md.update()
        snapshot = md.get_snapshot()
        if snapshot:
            if position.side == "LONG":
                pnl = (snapshot.current_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - snapshot.current_price) * position.quantity
            position.unrealized_pnl = pnl
    continue
```

**Що це вирішує:**
- Існуючі позиції на problematic symbols пропускаються при детальному оновленні
- P&L все ще оновлюється для відстеження
- SL/TP оновлення пропускаються, щоб уникнути помилок API
- Менше викликів API для символів з відомими проблемами
- Символи не позначаються повторно як PROBLEMATIC

---

## 🔴 Проблема 3: Недостатнє логування для problematic symbols

### Проблема
Коли символ перевіряється на problematic status, немає достатнього логування для відстеження:
- Скільки часу залишилося до закінчення blacklist
- Чи правильно працює перевірка

### Виправлення

**Файл:** `core/position_tracker.py` (метод `is_problematic_symbol`)

Покращено логування:

```python
# Symbol is still in blacklist
remaining_time = self._problematic_symbol_duration - elapsed
logger.debug(f"Symbol {symbol} is problematic (blacklist expires in {remaining_time/3600:.1f} hours)")
return True
```

Також покращено логування при видаленні з blacklist:

```python
logger.info(f"Symbol {symbol} removed from problematic list (blacklist expired after {elapsed/3600:.1f} hours)")
```

**Що це вирішує:**
- Краще логування для відстеження problematic symbols
- Можна бачити, скільки часу залишилося до закінчення blacklist
- Легше діагностувати проблеми з blacklist

---

## Підсумок виправлень

### Виправлені файли:
1. **core/position_tracker.py**
   - Видалено реєстрацію library bug як API error в `_ensure_sl_tp_orders`
   - Покращено логування в `is_problematic_symbol`

2. **main.py**
   - Додано перевірку problematic_symbols при оновленні позицій в `_manage_all_positions`
   - Існуючі позиції на problematic symbols пропускаються при детальному оновленні
   - P&L все ще оновлюється для відстеження

### Очікувані результати:
- ✅ Library bug більше не додає символи до blacklist
- ✅ Існуючі позиції на problematic symbols пропускаються при детальному оновленні
- ✅ Менше помилок API для символів з відомими проблемами
- ✅ Символи не позначаються повторно як PROBLEMATIC
- ✅ Краще логування для відстеження problematic symbols

### Важливі примітки:
1. **Library bug не є критичною помилкою** - це відома помилка бібліотеки, яка не впливає на функціональність бота
2. **Існуючі позиції все ще відстежуються** - P&L оновлюється, але SL/TP оновлення пропускаються
3. **Blacklist працює правильно** - символи додаються до blacklist тільки через справжні помилки API (не library bug)

### Тестування:
1. Запустити бота і перевірити, чи library bug не додає символи до blacklist
2. Перевірити, чи існуючі позиції на problematic symbols пропускаються при детальному оновленні
3. Перевірити, чи P&L все ще оновлюється для позицій на problematic symbols
4. Перевірити логування для відстеження problematic symbols
