"""
Утиліти для валідації параметрів торгового бота.
"""

import re
from typing import Optional

try:
    from utils.exceptions import ValidationError
except ImportError:
    # Fallback якщо exceptions модуль недоступний
    class ValidationError(ValueError):
        """Помилка валідації параметрів."""
        pass


def validate_symbol(symbol: str) -> str:
    """
    Валідує формат символу торгової пари.
    
    Args:
        symbol: Символ торгової пари (наприклад, 'BTCUSDT')
        
    Returns:
        Нормалізований символ (uppercase)
        
    Raises:
        ValidationError: Якщо символ невалідний
    """
    if not symbol or not isinstance(symbol, str):
        raise ValidationError(f"Символ має бути непустим рядком, отримано: {type(symbol).__name__}")
    
    symbol = symbol.strip().upper()
    
    # Базова перевірка формату: мінімум 3 символи, зазвичай закінчується на USDT
    if len(symbol) < 3:
        raise ValidationError(f"Символ занадто короткий: {symbol}")
    
    # Перевірка на наявність заборонених символів
    if not re.match(r'^[A-Z0-9]+$', symbol):
        raise ValidationError(f"Символ містить недопустимі символи: {symbol}")
    
    return symbol


def validate_quantity(quantity: float, min_quantity: float = 0.0, max_quantity: Optional[float] = None) -> float:
    """
    Валідує кількість для торгової операції.
    
    Args:
        quantity: Кількість для торгівлі
        min_quantity: Мінімальна допустима кількість (за замовчуванням 0)
        max_quantity: Максимальна допустима кількість (опційно)
        
    Returns:
        Валідована кількість
        
    Raises:
        ValidationError: Якщо кількість невалідна
    """
    if not isinstance(quantity, (int, float)):
        raise ValidationError(f"Кількість має бути числом, отримано: {type(quantity).__name__}")
    
    quantity = float(quantity)
    
    if quantity <= 0:
        raise ValidationError(f"Кількість має бути більше 0, отримано: {quantity}")
    
    if quantity < min_quantity:
        raise ValidationError(f"Кількість {quantity} менша за мінімальну {min_quantity}")
    
    if max_quantity is not None and quantity > max_quantity:
        raise ValidationError(f"Кількість {quantity} більша за максимальну {max_quantity}")
    
    return quantity


def validate_price(price: float, min_price: float = 0.0) -> float:
    """
    Валідує ціну для торгової операції.
    
    Args:
        price: Ціна
        min_price: Мінімальна допустима ціна (за замовчуванням 0)
        
    Returns:
        Валідована ціна
        
    Raises:
        ValidationError: Якщо ціна невалідна
    """
    if not isinstance(price, (int, float)):
        raise ValidationError(f"Ціна має бути числом, отримано: {type(price).__name__}")
    
    price = float(price)
    
    if price <= 0:
        raise ValidationError(f"Ціна має бути більше 0, отримано: {price}")
    
    if price < min_price:
        raise ValidationError(f"Ціна {price} менша за мінімальну {min_price}")
    
    # Перевірка на розумні межі (занадто великі значення можуть бути помилкою)
    if price > 1e10:  # 10 мільярдів
        raise ValidationError(f"Ціна {price} занадто велика (можлива помилка)")
    
    return price


def validate_side(side: str) -> str:
    """
    Валідує напрямок угоди (BUY/SELL).
    
    Args:
        side: Напрямок угоди
        
    Returns:
        Нормалізований напрямок (uppercase)
        
    Raises:
        ValidationError: Якщо напрямок невалідний
    """
    if not side or not isinstance(side, str):
        raise ValidationError(f"Напрямок має бути непустим рядком, отримано: {type(side).__name__}")
    
    side = side.strip().upper()
    
    if side not in ('BUY', 'SELL'):
        raise ValidationError(f"Напрямок має бути 'BUY' або 'SELL', отримано: {side}")
    
    return side


def validate_percent(value: float, name: str = "Відсоток", min_value: float = 0.0, max_value: float = 100.0) -> float:
    """
    Валідує відсоткове значення.
    
    Args:
        value: Відсоткове значення
        name: Назва параметра для повідомлення про помилку
        min_value: Мінімальне значення (за замовчуванням 0)
        max_value: Максимальне значення (за замовчуванням 100)
        
    Returns:
        Валідоване значення
        
    Raises:
        ValidationError: Якщо значення невалідне
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} має бути числом, отримано: {type(value).__name__}")
    
    value = float(value)
    
    if value < min_value:
        raise ValidationError(f"{name} {value} менше за мінімальне {min_value}")
    
    if value > max_value:
        raise ValidationError(f"{name} {value} більше за максимальне {max_value}")
    
    return value

