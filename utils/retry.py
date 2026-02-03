"""
Retry logic з exponential backoff для обробки тимчасових помилок.

Використання:
    from utils.retry import retry_on_error
    
    @retry_on_error(max_retries=3, backoff_base=2.0)
    def api_call():
        # ... виклик API
        return result
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple, Optional, Any
from binance.exceptions import BinanceAPIException

from utils.exceptions import (
    APIError, RateLimitError, TimestampError, TradingBotError
)

logger = logging.getLogger(__name__)


def retry_on_error(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Декоратор для retry з exponential backoff.
    
    Args:
        max_retries: Максимальна кількість спроб
        backoff_base: База для exponential backoff (секунди)
        exceptions: Типи exceptions для retry
        on_retry: Callback функція, яка викликається при retry
    
    Приклад:
        @retry_on_error(max_retries=3, backoff_base=2.0)
        def api_call():
            return client.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Якщо це остання спроба - підняти помилку
                    if attempt == max_retries - 1:
                        break
                    
                    # Розрахувати затримку (exponential backoff)
                    delay = backoff_base ** attempt
                    
                    # Спеціальна обробка для rate limit
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = float(e.retry_after)
                    
                    # Спеціальна обробка для BinanceAPIException
                    if isinstance(e, BinanceAPIException):
                        error_code = getattr(e, 'code', None)
                        if error_code == 429:  # Rate limit
                            # Спробувати отримати retry_after з headers
                            retry_after = getattr(e, 'response_headers', {}).get('Retry-After', delay)
                            delay = float(retry_after) if retry_after else delay
                        elif error_code == -1021:  # Timestamp error
                            # Для timestamp error не робимо backoff, просто retry
                            delay = 0.1
                    
                    # Викликати callback якщо вказано
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception:
                            pass
                    
                    logger.warning(
                        f"⚠️ Помилка при виклику {func.__name__} (спроба {attempt + 1}/{max_retries}): {e}. "
                        f"Повтор через {delay:.2f}s..."
                    )
                    
                    time.sleep(delay)
            
            # Якщо всі спроби не вдалися - підняти останню помилку
            raise last_exception
        
        return wrapper
    return decorator


def retry_on_api_error(
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retryable_codes: Tuple[int, ...] = (429, -1021, -1003, -1006)
):
    """
    Спеціалізований декоратор для Binance API помилок.
    
    Args:
        max_retries: Максимальна кількість спроб
        backoff_base: База для exponential backoff
        retryable_codes: Коди помилок, які можна retry
    
    Приклад:
        @retry_on_api_error(max_retries=3)
        def get_balance():
            return client.get_account()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except BinanceAPIException as e:
                    last_exception = e
                    error_code = getattr(e, 'code', None)
                    
                    # Перевірити чи можна retry цю помилку
                    if error_code not in retryable_codes:
                        # Не retry для не-retryable помилок
                        raise
                    
                    # Якщо це остання спроба - підняти помилку
                    if attempt == max_retries - 1:
                        break
                    
                    # Розрахувати затримку
                    delay = backoff_base ** attempt
                    
                    if error_code == 429:  # Rate limit
                        retry_after = getattr(e, 'response_headers', {}).get('Retry-After', delay)
                        delay = float(retry_after) if retry_after else delay
                        logger.warning(
                            f"⚠️ Rate limit exceeded (спроба {attempt + 1}/{max_retries}). "
                            f"Повтор через {delay:.2f}s..."
                        )
                    elif error_code == -1021:  # Timestamp error
                        delay = 0.1  # Швидкий retry для timestamp
                        logger.debug(f"🔄 Timestamp sync (спроба {attempt + 1}/{max_retries})")
                    else:
                        logger.warning(
                            f"⚠️ API помилка {error_code} (спроба {attempt + 1}/{max_retries}): {e}. "
                            f"Повтор через {delay:.2f}s..."
                        )
                    
                    time.sleep(delay)
                except Exception as e:
                    # Для інших помилок - не retry
                    raise
            
            # Підняти останню помилку
            raise last_exception
        
        return wrapper
    return decorator

