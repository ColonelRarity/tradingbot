# Інструкції для перевірки логів зранку

## Команди для перевірки логів

### 1. Перевірка reconciliation операцій
```powershell
Select-String -Path trading_bot.log -Pattern "\[RECONCILE\]" | Select-Object -Last 50
```

### 2. Перевірка hedge операцій
```powershell
Select-String -Path trading_bot.log -Pattern "hedge|HEDGE" | Select-Object -Last 50
```

### 3. Перевірка SL операцій
```powershell
Select-String -Path trading_bot.log -Pattern "SL PLACED|SL UPDATED|SL UPDATE FAILED" | Select-Object -Last 50
```

### 4. Перевірка помилок
```powershell
Select-String -Path trading_bot.log -Pattern "ERROR" | Select-Object -Last 20
```

### 5. Останні 100 рядків логу
```powershell
Get-Content trading_bot.log -Tail 100
```

## Що перевіряти

### ✅ Позитивні ознаки:
- Hedge позиції правильно відображаються з `[HEDGE]` labels
- Немає помилок `PositionState is not defined`
- Reconciliation операції виконуються коректно
- SL операції логуються детально

### ⚠️ Проблеми, на які звернути увагу:
- Помилки reconciliation
- Hedge позиції, які не синхронізуються з біржею
- Критичні помилки в логах
- Проблеми зі стоп-лосами

## Файли для надсилання

Якщо виникнуть проблеми, надішліть:
1. Останні 200-500 рядків `trading_bot.log`
2. Вивід з консолі (якщо можливо)
3. Конкретні помилки (якщо є)
