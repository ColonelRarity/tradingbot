# Тести для Trading Bot

## Запуск тестів

```bash
# Всі тести
python -m pytest tests/

# Конкретний файл
python -m pytest tests/test_cache.py

# З verbose output
python -m pytest tests/ -v

# З coverage
python -m pytest tests/ --cov=utils --cov=market_learning
```

## Структура тестів

- `test_cache.py` - тести для кешування
- `test_exceptions.py` - тести для кастомних exceptions
- `test_market_learning.py` - тести для ML модуля

## Вимоги

```bash
pip install pytest pytest-cov pytest-mock
```

## Примітки

- Тести використовують тимчасові файли для БД
- Моки для Binance API (не роблять реальні запити)
- Unit тести для критичних функцій

