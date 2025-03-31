import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import coloredlogs
import yaml
import os
from typing import Optional, Any, Dict, Union, Callable
import queue
import atexit
from datetime import datetime
import json
import traceback
import inspect

# Global queue for async logging
_log_queue = queue.Queue(-1)  # Unlimited queue size
_queue_listener = None

# ===== Default Configuration =====
_DEFAULT_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s | %(name)s | %(levelname)s | %(message)s',  # Add this top-level format
    'file': {
        'enabled': True,
        'path': 'logs/trading_bot.log',
        'max_size': 10,  # MB
        'backups': 3,
        'json_format': True
    },
    'console': {
        'enabled': True,
        'colors': True,
        'format': '%(asctime)s | %(name)s | %(levelname)s | %(message)s'  # Keep this too
    },
    'error_reporting': {
        'capture_stack': True,
        'context_vars': True
    }
}

# ===== JSON Formatter (for structured logging) =====
class EnhancedJsonFormatter(logging.Formatter):
    """Improved JSON formatter with error handling"""
    def format(self, record: logging.LogRecord) -> str:
        try:
            log_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'process': record.process,
                'thread': record.threadName
            }

            if record.exc_info:
                log_data['exception'] = traceback.format_exc()

            if hasattr(record, 'context'):
                log_data['context'] = getattr(record, 'context')

            return json.dumps(log_data, default=str)
        except Exception as e:
            return json.dumps({
                'timestamp': datetime.utcnow().isoformat(),
                'level': 'ERROR',
                'message': f'Log formatting failed: {str(e)}',
                'original_message': record.getMessage()
            })

def _setup_handlers(config: Dict[str, Any]) -> list:
    """Configure all logging handlers"""
    handlers = []

    # File handler (always JSON)
    if config['file']['enabled']:
        os.makedirs(os.path.dirname(config['file']['path']), exist_ok=True)
        file_handler = RotatingFileHandler(
            filename=config['file']['path'],
            maxBytes=config['file']['max_size'] * 1024 * 1024,
            backupCount=config['file']['backups'],
            encoding='utf-8'
        )
        file_handler.setFormatter(EnhancedJsonFormatter())
        handlers.append(file_handler)

    # Console handler
    if config['console']['enabled']:
        if config['console']['colors']:
            coloredlogs.install(
                level=config['level'],
                fmt=config['console'].get('format', config['format']),  # Fallback to top-level format
                logger=logging.getLogger()
            )
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter(config['console'].get('format', config['format']))
            )
            handlers.append(console_handler)

    return handlers

def setup_logger(name: str, config_path: str = "config/config.yaml") -> logging.Logger:
    """Enhanced logger setup with async support"""
    global _queue_listener

    try:
        config = _load_config_with_fallback(config_path)
        logger = logging.getLogger(name)
        logger.setLevel(config['level'])

        if not logger.handlers:
            if _queue_listener is None:
                handlers = _setup_handlers(config)
                _queue_listener = QueueListener(_log_queue, *handlers)
                _queue_listener.start()
                atexit.register(_stop_queue_listener)

            queue_handler = QueueHandler(_log_queue)
            queue_handler.setLevel(config['level'])
            logger.addHandler(queue_handler)

        logger.propagate = False
        return logger

    except Exception as e:
        print(f"Critical logging failure: {str(e)}")
        return _create_fallback_logger(name)

# ===== Utility Functions =====
def _load_config_with_fallback(config_path: str) -> Dict[str, Any]:
    """Load config with graceful fallback to defaults"""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f).get('logging', {})
        return {**_DEFAULT_CONFIG, **config}  # Merge with defaults
    except Exception:
        return _DEFAULT_CONFIG

def _stop_queue_listener():
    """Cleanup async logging on shutdown"""
    global _queue_listener
    if _queue_listener:
        _queue_listener.stop()

def _create_fallback_logger(name: str) -> logging.Logger:
    """Emergency logger if setup fails"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_CONFIG.get('format', '%(message)s')))
    logger.addHandler(handler)
    return logger

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert to float with error handling"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

# ===== Contextual Logging =====
def log_with_context(
    logger: logging.Logger,
    level: int,
    msg: str,  # This was missing in your calls
    *,
    stack_info: bool = False,
    **context
) -> None:

    """
    Enhanced logging with structured context.

    Args:
        logger: Configured logger instance
        level: Logging level (e.g., logging.INFO)
        msg: Primary log message
        stack_info: Whether to include stack info
        **context: Additional context as key-value pairs
    """
    if logger.isEnabledFor(level):
        extra = {'context': context}
        if stack_info:
            extra['stack_info'] = traceback.format_stack()
        logger.log(level, msg, extra=extra)

def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    capture_args: bool = False
) -> Callable:
    """Flexible execution time logger decorator"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()

                context = {
                    'function': func.__name__,
                    'duration': duration,
                    'module': inspect.getmodule(func).__name__
                }

                if capture_args:
                    context.update({
                        'args': args,
                        'kwargs': kwargs
                    })

                if logger and logger.isEnabledFor(level):
                    logger.log(
                        level,
                        f"{func.__name__} executed in {duration:.4f}s",
                        extra={'context': context}
                    )

                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                if logger:
                    logger.error(
                        f"{func.__name__} failed after {duration:.4f}s: {str(e)}",
                        exc_info=True,
                        extra={
                            'context': {
                                'function': func.__name__,
                                'duration': duration,
                                'error': str(e),
                                'traceback': traceback.format_exc()
                            }
                        }
                    )
                raise
        return wrapper
    return decorator
