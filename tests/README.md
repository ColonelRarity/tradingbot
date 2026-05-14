# Тести для Trading Bot

## Запуск тестів

```bash
# Всі тести
python -m pytest tests/

# Конкретний файл
python -m pytest tests/test_cache.py

# З verbose output
python -m pytest tests/ -v

# З coverage (приклад)
python -m pytest tests/ --cov=utils --cov=core

# Сигнали, ризик, ордери (без біржі)
python -m pytest tests/test_signals_and_trading.py -v
```

## Структура тестів

- `test_bot.py` — ініціалізація `MultiPairTradingBot` з мок-клієнтом
- `test_signals_and_trading.py` — сигнал → розмір позиції → MARKET entry (моки)
- `test_cache.py` — кешування
- `test_exceptions.py` — кастомні exceptions
- `test_market_learning.py` — legacy ML (пропускається, якщо немає модуля)

## Вимоги

```bash
pip install pytest pytest-cov pytest-mock
```

## Примітки

- Тести використовують тимчасові файли для БД
- Моки для Binance API (не роблять реальні запити)
- Unit тести для критичних функцій

