"""
Кастомні exceptions для trading bot.

Дозволяє краще розрізняти типи помилок та обробляти їх відповідно.
"""

from typing import Optional, Dict, Any


class TradingBotError(Exception):
    """Базовий exception для всіх помилок trading bot."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx_str})"
        return self.message


class APIError(TradingBotError):
    """Помилка при виклику Binance API."""
    
    def __init__(self, message: str, error_code: Optional[int] = None, 
                 symbol: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if error_code is not None:
            context['error_code'] = error_code
        if symbol:
            context['symbol'] = symbol
        super().__init__(message, context)
        self.error_code = error_code
        self.symbol = symbol


class RateLimitError(APIError):
    """Помилка rate limiting (429 Too Many Requests)."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None, **kwargs):
        context = kwargs.get('context', {})
        if retry_after is not None:
            context['retry_after'] = retry_after
        super().__init__(message, error_code=429, context=context, **kwargs)
        self.retry_after = retry_after


class TimestampError(APIError):
    """Помилка синхронізації часу (-1021)."""
    
    def __init__(self, message: str = "Timestamp out of sync", **kwargs):
        super().__init__(message, error_code=-1021, **kwargs)


class InsufficientBalanceError(TradingBotError):
    """Недостатньо балансу для виконання операції."""
    
    def __init__(self, message: str = "Insufficient balance", 
                 required: Optional[float] = None, available: Optional[float] = None, **kwargs):
        context = kwargs.get('context', {})
        if required is not None:
            context['required'] = required
        if available is not None:
            context['available'] = available
        super().__init__(message, context)
        self.required = required
        self.available = available


class OrderError(TradingBotError):
    """Помилка при розміщенні/скасуванні ордера."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, 
                 order_id: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        if order_id:
            context['order_id'] = order_id
        super().__init__(message, context)
        self.symbol = symbol
        self.order_id = order_id


class MarketDataError(TradingBotError):
    """Помилка отримання даних ринку."""
    
    def __init__(self, message: str, symbol: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if symbol:
            context['symbol'] = symbol
        super().__init__(message, context)
        self.symbol = symbol


class ConfigurationError(TradingBotError):
    """Помилка конфігурації бота."""
    pass


class DatabaseError(TradingBotError):
    """Помилка роботи з базою даних."""
    pass


class ValidationError(TradingBotError):
    """Помилка валідації параметрів."""
    
    def __init__(self, message: str, parameter: Optional[str] = None, value: Any = None, **kwargs):
        context = kwargs.get('context', {})
        if parameter:
            context['parameter'] = parameter
        if value is not None:
            context['value'] = value
        super().__init__(message, context)
        self.parameter = parameter
        self.value = value
