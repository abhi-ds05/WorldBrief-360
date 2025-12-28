"""
Logging configuration and utilities.
"""

import logging
import logging.config
import sys
import json
from typing import Any, Dict, Optional, Union
from pathlib import Path
from datetime import datetime

from .config import LoggingConfig, get_config


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_data.update(record.extra)
        
        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",   # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",   # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logging(
    config: Optional[LoggingConfig] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> None:
    """
    Setup logging configuration.
    
    Args:
        config: Logging configuration
        log_file: Optional log file path
        json_format: Use JSON format for logs
    """
    if config is None:
        config = get_config().logging
    
    # Create log directory if needed
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt=config.format,
            datefmt=config.date_format
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(config.level.value)
    
    if not json_format:
        console_handler.setFormatter(ColorFormatter(
            fmt=config.format,
            datefmt=config.date_format
        ))
    else:
        console_handler.setFormatter(formatter)
    
    # File handler (if log file specified)
    handlers = [console_handler]
    
    if log_file or config.file_path:
        file_path = log_file or config.file_path
        if file_path:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    filename=file_path,
                    maxBytes=config.max_file_size,
                    backupCount=config.backup_count,
                    encoding="utf-8"
                )
                file_handler.setLevel(config.level.value)
                file_handler.setFormatter(formatter)
                handlers.append(file_handler)
            except Exception as e:
                logging.warning(f"Failed to setup file logging: {e}")
    
    # Configure root logger
    logging.basicConfig(
        level=config.level.value,
        format=config.format,
        datefmt=config.date_format,
        handlers=handlers,
        force=True  # Override existing handlers
    )
    
    # Set levels for third-party libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aioredis").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    
    logging.info(f"Logging configured with level: {config.level}")


def get_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get a logger with extra context.
    
    Args:
        name: Logger name
        extra: Extra context to include in logs
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    if extra:
        # Create a filter to add extra context
        class ContextFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                for key, value in extra.items():
                    setattr(record, key, value)
                return True
        
        # Add filter if not already added
        if not any(isinstance(f, ContextFilter) for f in logger.filters):
            logger.addFilter(ContextFilter())
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with additional context."""
    
    def __init__(self, logger: logging.Logger, extra: Dict[str, Any]):
        super().__init__(logger, extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """Process log message with extra context."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"].update(self.extra)
        return msg, kwargs


# Context manager for logging sections
class LoggingContext:
    """Context manager for logging sections."""
    
    def __init__(self, logger: logging.Logger, message: str, level: int = logging.INFO):
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time: Optional[datetime] = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.log(self.level, f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        status = "completed" if exc_type is None else "failed"
        self.logger.log(
            self.level,
            f"{self.message} {status} in {duration.total_seconds():.2f}s"
        )


# Convenience functions
def get_request_logger(request_id: str) -> LoggerAdapter:
    """Get a logger adapter for request context."""
    logger = logging.getLogger("request")
    return LoggerAdapter(logger, {"request_id": request_id})


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                duration = datetime.now() - start_time
                logger.debug(
                    f"Function {func.__name__} executed in {duration.total_seconds():.2f}s"
                )
                return result
            except Exception as e:
                duration = datetime.now() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {duration.total_seconds():.2f}s: {e}"
                )
                raise
        return wrapper
    return decorator