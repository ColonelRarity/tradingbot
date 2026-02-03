"""
Налаштування структурованого логування для trading bot.

Підтримує:
- JSON логування для production
- Звичайне логування для development
- Правильні рівні логування
- Ротація логів
"""

import os
import sys
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class StructuredFormatter(logging.Formatter):
    """
    Formatter для структурованого логування (JSON).
    Корисно для збору логів у централізовані системи (ELK, Splunk, etc.)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Форматувати log record у JSON формат.
        """
        log_data: Dict[str, Any] = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Додати exception info якщо є
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Додати додаткові поля з extra
        if hasattr(record, 'symbol'):
            log_data['symbol'] = record.symbol
        if hasattr(record, 'action'):
            log_data['action'] = record.action
        if hasattr(record, 'error_code'):
            log_data['error_code'] = record.error_code
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter для додавання контексту до всіх логів.
    
    Використання:
        logger = get_logger(__name__, context={'symbol': 'BTCUSDT'})
        logger.info("Trading started")  # Автоматично додасть symbol до логу
    """
    
    def __init__(self, logger: logging.Logger, context: Optional[Dict[str, Any]] = None):
        super().__init__(logger, context or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Додати контекст до log record.
        """
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        # Додати контекст до extra для structured logging
        if self.extra:
            kwargs['extra'].update(self.extra)
        
        return msg, kwargs


def get_log_level() -> int:
    """
    Отримати рівень логування з ENV або за замовчуванням.
    
    ENV: LOG_LEVEL (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_str = os.getenv('LOG_LEVEL', 'INFO').upper()
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    return level_map.get(level_str, logging.INFO)


def get_log_format() -> str:
    """
    Отримати формат логування з ENV.
    
    ENV: LOG_FORMAT (json, text)
    """
    return os.getenv('LOG_FORMAT', 'text').lower()


def setup_logging(
    name: str = 'trading_bot',
    log_file: str = 'trading_bot.log',
    use_json: Optional[bool] = None,
    level: Optional[int] = None
) -> logging.Logger:
    """
    Налаштувати логування для trading bot.
    
    Args:
        name: Ім'я логгера
        log_file: Шлях до файлу логів
        use_json: Використовувати JSON формат (None = auto detect з LOG_FORMAT)
        level: Рівень логування (None = auto detect з LOG_LEVEL)
    
    Returns:
        Налаштований logger
    """
    logger = logging.getLogger(name)
    
    # Якщо вже налаштовано - повернути
    if logger.handlers:
        return logger
    
    # Визначити рівень та формат
    if level is None:
        level = get_log_level()
    
    if use_json is None:
        use_json = get_log_format() == 'json'
    
    logger.setLevel(level)
    
    # Formatter
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # File handler з ротацією
    try:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,  # Зберігати 5 backup файлів
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Якщо не вдалося створити file handler - використати stderr
        print(f"⚠️ Не вдалося створити file handler: {e}", file=sys.stderr)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Для Windows - використовувати UTF-8 handler
    if sys.platform == 'win32':
        class UTF8StreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    stream = self.stream
                    stream.write(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace') + self.terminator)
                    self.flush()
                except Exception:
                    self.handleError(record)
        
        console_handler = UTF8StreamHandler(sys.stdout)
        console_handler.setLevel(level)
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> ContextLoggerAdapter:
    """
    Отримати logger з контекстом.
    
    Args:
        name: Ім'я логгера (зазвичай __name__)
        context: Контекст для додавання до всіх логів
    
    Returns:
        Logger adapter з контекстом
    
    Приклад:
        logger = get_logger(__name__, context={'symbol': 'BTCUSDT'})
        logger.info("Analysis started")  # Додасть symbol до логу
    """
    base_logger = logging.getLogger(name)
    return ContextLoggerAdapter(base_logger, context)


# Глобальний logger для trading bot
_main_logger: Optional[logging.Logger] = None


def init_logging(log_file: str = 'trading_bot.log') -> logging.Logger:
    """
    Ініціалізувати глобальне логування для trading bot.
    
    Викликається один раз при старті бота.
    """
    global _main_logger
    if _main_logger is None:
        _main_logger = setup_logging('trading_bot', log_file)
    return _main_logger

