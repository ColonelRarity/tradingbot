# Виправлення обробки помилок -2019 та -2027

## 🔴 КРИТИЧНА ПРОБЛЕМА: Постійні спроби оновити SL/TP при помилках margin/leverage

### Проблема
Бот постійно намагається оновити SL/TP ордери, навіть коли отримує помилки:
- **-2019 - Margin is insufficient** (594 рази в логах)
- **-2027 - Exceeded the maximum allowable position at current leverage** (600 разів в логах)

Це призводить до:
- Постійних помилок API
- Непотрібних викликів API
- Неможливості оновити SL/TP для прибуткових позицій
- Заблокованих позицій без можливості оновити SL/TP

**Приклади з логу:**
```
ERROR:exchange.binance_client:API ClientError: -2019 - Margin is insufficient.
ERROR:core.order_manager:SL order failed: (400, -2019, 'Margin is insufficient.', ...)
ERROR:core.position_tracker:URGENT: Failed to enforce breakeven SL for 54014c97: (400, -2019, 'Margin is insufficient.', ...)

ERROR:exchange.binance_client:API ClientError: -2027 - Exceeded the maximum allowable position at current leverage.
ERROR:core.order_manager:SL order failed: (400, -2027, 'Exceeded the maximum allowable position at current leverage.', ...)
ERROR:core.position_tracker:URGENT: Failed to enforce breakeven SL for 9d4391eb: (400, -2027, 'Exceeded the maximum allowable position at current leverage.', ...)
```

### Причини
1. **Відсутність обробки помилок -2019 та -2027** - бот не розпізнавав ці помилки як критичні
2. **Відсутність cooldown** - бот продовжував намагатися оновити SL/TP, навіть коли це неможливо
3. **Недостатня обробка в urgent breakeven check** - бот намагався оновити SL кожну секунду, навіть коли отримував помилки

### Виправлення

#### 1. Додано обробку помилок -2019 та -2027 в `core/order_manager.py`

**Для SL orders (`place_stop_loss`):**
```python
# -2019: Margin is insufficient - cannot update SL/TP
if "-2019" in error_str or "Margin is insufficient" in error_str:
    logger.warning(f"SL order failed: Margin is insufficient. Cannot update SL for {symbol}")
    return OrderResult(
        success=False,
        error_code="MARGIN_INSUFFICIENT",
        error_message="Margin is insufficient - cannot update SL"
    )
# -2027: Exceeded maximum position at current leverage - cannot update SL/TP
if "-2027" in error_str or "Exceeded the maximum allowable position" in error_str:
    logger.warning(f"SL order failed: Exceeded maximum position at current leverage for {symbol}")
    return OrderResult(
        success=False,
        error_code="MAX_POSITION_EXCEEDED",
        error_message="Exceeded maximum position at current leverage - cannot update SL"
    )
```

**Для TP orders (`place_take_profit`):**
Аналогічна обробка для TP orders.

**Що це вирішує:**
- Бот розпізнає помилки -2019 та -2027 як критичні
- Повертає спеціальні error_code для подальшої обробки
- Логує попередження замість помилок

#### 2. Додано cooldown для позицій з помилками margin/leverage в `core/position_tracker.py`

**Додано tracking cooldown:**
```python
# Cooldown for positions with margin/leverage errors: position_id -> timestamp
# Prevents repeated attempts to update SL/TP when margin is insufficient or position limit exceeded
self._sl_update_cooldown: Dict[str, float] = {}  # position_id -> cooldown_until_timestamp
self._sl_update_cooldown_duration = 300.0  # 5 minutes cooldown after margin/leverage errors
```

**Додано перевірку cooldown перед оновленням SL:**
```python
# Check cooldown for margin/leverage errors before attempting update
if position_id in self._sl_update_cooldown:
    cooldown_until = self._sl_update_cooldown[position_id]
    if time.time() < cooldown_until:
        # Still in cooldown, skip update
        remaining = cooldown_until - time.time()
        logger.debug(f"[{position.symbol}] SL update skipped due to cooldown ({remaining/60:.1f} min remaining)")
        return
```

**Додано встановлення cooldown при помилках:**
```python
error_code = result.error_code if hasattr(result, 'error_code') else ""
# Check if error is due to margin/leverage issues
if error_code in ["MARGIN_INSUFFICIENT", "MAX_POSITION_EXCEEDED"]:
    # Set cooldown to prevent repeated attempts
    self._sl_update_cooldown[position_id] = time.time() + self._sl_update_cooldown_duration
    logger.warning(f"[{position.symbol}] ❌ SL UPDATE FAILED: {error_code} | "
                  f"Cannot update SL due to margin/leverage constraints. "
                  f"Cooldown set for {self._sl_update_cooldown_duration/60:.1f} minutes.")
    # Remove from urgent check temporarily - will retry after cooldown
    if position_id in self._urgent_breakeven_check:
        del self._urgent_breakeven_check[position_id]
```

**Що це вирішує:**
- Бот не намагається оновити SL/TP, якщо позиція в cooldown
- Cooldown триває 5 хвилин після помилки
- Після cooldown бот спробує оновити SL/TP знову

#### 3. Додано обробку помилок в urgent breakeven check

**Додано перевірку cooldown в `check_and_enforce_breakeven_sl_urgent`:**
```python
# Check cooldown for margin/leverage errors
if position_id in self._sl_update_cooldown:
    cooldown_until = self._sl_update_cooldown[position_id]
    if time.time() < cooldown_until:
        # Still in cooldown, skip update
        remaining = cooldown_until - time.time()
        logger.debug(f"SL update cooldown active for {position_id} ({remaining/60:.1f} min remaining)")
        return False  # Still needs checking after cooldown
    else:
        # Cooldown expired, remove it
        del self._sl_update_cooldown[position_id]
        logger.debug(f"SL update cooldown expired for {position_id}, will retry")
```

**Додано обробку помилок при urgent check:**
```python
error_code = result.error_code if hasattr(result, 'error_code') else ""
if error_code in ["MARGIN_INSUFFICIENT", "MAX_POSITION_EXCEEDED"]:
    # Set cooldown to prevent repeated attempts
    self._sl_update_cooldown[position_id] = time.time() + self._sl_update_cooldown_duration
    logger.warning(f"URGENT: Cannot update SL for {position_id} due to {error_code}. "
                 f"Cooldown set for {self._sl_update_cooldown_duration/60:.1f} minutes.")
    # Remove from urgent check temporarily - will retry after cooldown
    if position_id in self._urgent_breakeven_check:
        del self._urgent_breakeven_check[position_id]
    return True  # Consider as "handled" for now, will retry after cooldown
```

**Що це вирішує:**
- Urgent breakeven check не намагається оновити SL, якщо позиція в cooldown
- Після cooldown бот спробує оновити SL знову
- Менше помилок API при urgent check

#### 4. Додано обробку помилок для TP updates

Аналогічна обробка для TP updates:
- Перевірка cooldown перед оновленням TP
- Встановлення cooldown при помилках -2019 та -2027
- Очищення cooldown при успішному оновленні TP

**Що це вирішує:**
- TP updates також не намагаються оновити, якщо позиція в cooldown
- Cooldown спільний для SL та TP (якщо не вдається оновити SL, TP також не оновлюється)

### Результат

1. **Менше помилок API** - бот не намагається оновити SL/TP, коли це неможливо
2. **Cooldown механізм** - бот чекає 5 хвилин перед повторною спробою
3. **Правильна обробка помилок** - бот розпізнає помилки -2019 та -2027 як критичні
4. **Менше навантаження на API** - менше викликів API для позицій з помилками
5. **Автоматичне відновлення** - після cooldown бот спробує оновити SL/TP знову

### Важливі примітки

1. **Cooldown триває 5 хвилин** - це достатньо для того, щоб маржа або позиція могли змінитися
2. **Cooldown спільний для SL та TP** - якщо не вдається оновити SL, TP також не оновлюється
3. **Cooldown очищається при успішному оновленні** - якщо SL/TP вдалося оновити, cooldown видаляється
4. **Urgent breakeven check поважає cooldown** - навіть urgent check не намагається оновити SL, якщо позиція в cooldown

### Тестування

1. Запустити бота і перевірити, чи не намагається оновити SL/TP при помилках -2019 та -2027
2. Перевірити, чи встановлюється cooldown при помилках
3. Перевірити, чи бот спробує оновити SL/TP після закінчення cooldown
4. Перевірити, чи urgent breakeven check поважає cooldown
