"""
Structured Logging System

This module provides a comprehensive structured logging system with:

- JSON-formatted structured logs
- Context-aware logging with request/operation tracking
- Log levels with custom filtering
- Multiple output handlers (console, file, syslog, etc.)
- Log rotation and compression
- Correlation IDs for distributed tracing
- Performance metrics in logs
- Log aggregation and filtering
- Audit logging capabilities
"""

import os
import sys
import json
import logging
import logging.config
import logging.handlers
import traceback
import time
import uuid
from typing import Dict, Generator, List, Optional, Any, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import threading
import inspect
from contextvars import ContextVar
from pathlib import Path
import gzip
import hashlib

from sklearn.conftest import wraps
from sklearn.utils import contextmanager
from sklearn.utils import contextmanager

try:
    import ujson as json
except ImportError:
    import json

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """Convert string to LogLevel."""
        level_str = level_str.upper()
        if level_str == "WARN":
            level_str = "WARNING"
        return cls[level_str]


@dataclass
class LogContext:
    """
    Context for structured logging.
    
    This contains information that should be included in every log message
    within a particular context (e.g., a request, a background job).
    """
    # Core context fields
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Operation context
    operation: Optional[str] = None
    component: Optional[str] = None
    subcomponent: Optional[str] = None
    action: Optional[str] = None
    
    # Performance context
    duration_ms: Optional[float] = None
    start_time: Optional[datetime] = None
    
    # Business context
    tenant_id: Optional[str] = None
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        
        # Add standard fields
        for field_name, field_value in asdict(self).items():
            if field_name == "custom_fields":
                continue
            if field_value is not None:
                result[field_name] = field_value
        
        # Add custom fields
        result.update(self.custom_fields)
        
        return result
    
    def merge(self, other: 'LogContext') -> 'LogContext':
        """Merge two contexts, with other taking precedence."""
        merged = LogContext()
        
        # Copy self fields
        for field_name in self.__dataclass_fields__:
            if field_name == "custom_fields":
                continue
            value = getattr(self, field_name)
            if value is not None:
                setattr(merged, field_name, value)
        
        # Override with other fields
        for field_name in other.__dataclass_fields__:
            if field_name == "custom_fields":
                continue
            value = getattr(other, field_name)
            if value is not None:
                setattr(merged, field_name, value)
        
        # Merge custom fields
        merged.custom_fields = {**self.custom_fields, **other.custom_fields}
        
        return merged
    
    def update(self, **kwargs) -> None:
        """Update context with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_fields[key] = value


class StructuredFormatter(logging.Formatter):
    """
    Formatter that outputs logs as structured JSON.
    """
    
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = '%',
        include_traceback: bool = True,
        include_module_info: bool = True,
        flatten_context: bool = True,
        **kwargs
    ):
        super().__init__(fmt, datefmt, style)
        self.include_traceback = include_traceback
        self.include_module_info = include_module_info
        self.flatten_context = flatten_context
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record as structured JSON.
        """
        log_entry = self._create_log_entry(record)
        return json.dumps(log_entry, default=self._json_serializer, ensure_ascii=False)
    
    def _create_log_entry(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Create structured log entry from log record."""
        # Extract context from record
        context = getattr(record, 'context', LogContext()).to_dict()
        extra = getattr(record, '__dict__', {})
        
        # Base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "process_id": record.process,
            "thread_id": record.thread,
            "thread_name": record.threadName,
        }
        
        # Add module info if requested
        if self.include_module_info:
            log_entry.update({
                "module": record.module,
                "function": record.funcName,
                "file": record.filename,
                "line": record.lineno,
            })
        
        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_traceback = record.exc_info
            log_entry["exception"] = {
                "type": exc_type.__name__ if exc_type else None,
                "message": str(exc_value) if exc_value else None,
            }
            
            if self.include_traceback:
                log_entry["exception"]["traceback"] = self._format_traceback(exc_traceback)
        
        # Add context
        if context:
            if self.flatten_context:
                # Flatten context into top-level fields
                for key, value in context.items():
                    if key not in log_entry:  # Don't override existing fields
                        log_entry[key] = value
            else:
                # Keep context as nested object
                log_entry["context"] = context
        
        # Add extra fields (excluding internal fields)
        internal_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread', 'threadName', 'processName',
            'process', 'message', 'context', '__dict__'
        }
        
        for key, value in extra.items():
            if key not in internal_fields and not key.startswith('_'):
                if key not in log_entry:  # Don't override existing fields
                    log_entry[key] = value
        
        return log_entry
    
    def _format_traceback(self, exc_traceback) -> List[str]:
        """Format traceback as list of strings."""
        if not exc_traceback:
            return []
        
        tb_lines = traceback.format_tb(exc_traceback)
        return [line.strip() for line in tb_lines]
    
    def _json_serializer(self, obj):
        """JSON serializer for non-serializable objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        elif callable(obj):
            return obj.__name__
        else:
            return str(obj)


class ContextFilter(logging.Filter):
    """
    Filter that adds context to log records.
    """
    
    def __init__(self, default_context: Optional[LogContext] = None):
        super().__init__()
        self.default_context = default_context or LogContext()
        self.context_var = ContextVar('logging_context', default=self.default_context)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        current_context = self.context_var.get()
        record.context = current_context
        return True
    
    def set_context(self, context: LogContext) -> None:
        """Set current context."""
        self.context_var.set(context)
    
    def get_context(self) -> LogContext:
        """Get current context."""
        return self.context_var.get()
    
    def update_context(self, **kwargs) -> None:
        """Update current context with new values."""
        context = self.get_context()
        context.update(**kwargs)
        self.set_context(context)
    
    def clear_context(self) -> None:
        """Clear current context (reset to default)."""
        self.context_var.set(self.default_context)


class AuditLogger:
    """
    Specialized logger for audit logging.
    
    Audit logs are used for security and compliance purposes,
    recording important actions and access attempts.
    """
    
    def __init__(self, logger_name: str = "audit"):
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False  # Don't propagate to root logger
        
        # Ensure audit logger has handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set level
        self.logger.setLevel(logging.INFO)
    
    def log_access(
        self,
        user_id: str,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """
        Log access attempt.
        
        Args:
            user_id: ID of the user performing the action
            action: Action being performed (e.g., "login", "read", "update")
            resource: Type of resource being accessed
            resource_id: ID of the specific resource
            success: Whether the action was successful
            ip_address: IP address of the client
            user_agent: User agent string
            details: Additional details about the action
            context: Additional logging context
        """
        audit_data = {
            "event_type": "access",
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "resource_id": resource_id,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if details:
            audit_data["details"] = details
        
        # Create message
        status = "SUCCESS" if success else "FAILURE"
        message = f"Access {status}: {user_id} {action} {resource}"
        if resource_id:
            message += f" ({resource_id})"
        
        # Log with context
        extra = {"audit": audit_data}
        if context:
            extra.update(context.to_dict())
        
        if success:
            self.logger.info(message, extra=extra)
        else:
            self.logger.warning(message, extra=extra)
    
    def log_data_change(
        self,
        user_id: str,
        action: str,
        entity_type: str,
        entity_id: str,
        changes: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """
        Log data modification.
        
        Args:
            user_id: ID of the user making the change
            action: Action (e.g., "create", "update", "delete")
            entity_type: Type of entity being modified
            entity_id: ID of the entity
            changes: Dictionary of changes made
            previous_state: Previous state of the entity
            new_state: New state of the entity
            context: Additional logging context
        """
        audit_data = {
            "event_type": "data_change",
            "user_id": user_id,
            "action": action,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "changes": changes,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if previous_state:
            audit_data["previous_state"] = previous_state
        if new_state:
            audit_data["new_state"] = new_state
        
        # Create message
        message = f"Data change: {user_id} {action} {entity_type} ({entity_id})"
        
        # Log with context
        extra = {"audit": audit_data}
        if context:
            extra.update(context.to_dict())
        
        self.logger.info(message, extra=extra)
    
    def log_security_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """
        Log security-related event.
        
        Args:
            event_type: Type of security event
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            description: Description of the event
            user_id: ID of the user involved (if any)
            ip_address: IP address involved
            details: Additional details
            context: Additional logging context
        """
        audit_data = {
            "event_type": "security",
            "security_event_type": event_type,
            "severity": severity,
            "description": description,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if details:
            audit_data["details"] = details
        
        # Create message
        message = f"Security event ({severity}): {event_type} - {description}"
        
        # Log with context
        extra = {"audit": audit_data}
        if context:
            extra.update(context.to_dict())
        
        # Log at appropriate level
        log_level = getattr(logging, severity.upper(), logging.INFO)
        self.logger.log(log_level, message, extra=extra)


class PerformanceLogger:
    """
    Logger for performance metrics.
    """
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.logger.propagate = False
        
        # Ensure performance logger has handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = StructuredFormatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set level
        self.logger.setLevel(logging.INFO)
    
    def log_operation(
        self,
        operation: str,
        duration_ms: float,
        success: bool = True,
        component: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """
        Log operation performance.
        
        Args:
            operation: Name of the operation
            duration_ms: Duration in milliseconds
            success: Whether the operation was successful
            component: Component performing the operation
            metrics: Additional performance metrics
            context: Additional logging context
        """
        perf_data = {
            "event_type": "performance",
            "operation": operation,
            "duration_ms": duration_ms,
            "success": success,
            "component": component,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if metrics:
            perf_data["metrics"] = metrics
        
        # Create message
        status = "SUCCESS" if success else "FAILURE"
        message = f"Performance: {operation} took {duration_ms:.2f}ms ({status})"
        
        # Log with context
        extra = {"performance": perf_data}
        if context:
            extra.update(context.to_dict())
        
        self.logger.info(message, extra=extra)
    
    def log_throughput(
        self,
        operation: str,
        count: int,
        duration_ms: float,
        component: Optional[str] = None,
        context: Optional[LogContext] = None,
    ) -> None:
        """
        Log throughput metrics.
        
        Args:
            operation: Name of the operation
            count: Number of operations
            duration_ms: Total duration in milliseconds
            component: Component performing the operations
            context: Additional logging context
        """
        throughput = count / (duration_ms / 1000) if duration_ms > 0 else 0
        avg_latency = duration_ms / count if count > 0 else 0
        
        perf_data = {
            "event_type": "throughput",
            "operation": operation,
            "count": count,
            "duration_ms": duration_ms,
            "throughput_per_second": throughput,
            "avg_latency_ms": avg_latency,
            "component": component,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Create message
        message = f"Throughput: {operation} - {throughput:.2f}/s, avg latency: {avg_latency:.2f}ms"
        
        # Log with context
        extra = {"performance": perf_data}
        if context:
            extra.update(context.to_dict())
        
        self.logger.info(message, extra=extra)


class LogManager:
    """
    Central manager for logging configuration and operations.
    """
    
    def __init__(
        self,
        service_name: str = "worldbrief-360",
        environment: str = "development",
        log_level: Union[str, int] = "INFO",
        log_dir: Optional[str] = None,
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        enable_json_output: bool = True,
        enable_audit_logging: bool = True,
        enable_performance_logging: bool = True,
        **kwargs
    ):
        """
        Initialize LogManager.
        
        Args:
            service_name: Name of the service
            environment: Deployment environment
            log_level: Default log level
            log_dir: Directory for log files
            enable_file_logging: Enable file logging
            enable_console_logging: Enable console logging
            enable_json_output: Use JSON format for logs
            enable_audit_logging: Enable audit logging
            enable_performance_logging: Enable performance logging
            **kwargs: Additional configuration
        """
        self.service_name = service_name
        self.environment = environment
        self.log_level = self._parse_log_level(log_level)
        self.log_dir = self._setup_log_dir(log_dir)
        
        # Configuration flags
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.enable_json_output = enable_json_output
        self.enable_audit_logging = enable_audit_logging
        self.enable_performance_logging = enable_performance_logging
        
        # Initialize components
        self.context_filter = None
        self.audit_logger = None
        self.performance_logger = None
        
        # Statistics
        self.stats = {
            "log_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "last_log_time": None,
        }
        
        # Initialize logging
        self._setup_logging()
        
        logger.info(f"LogManager initialized for {service_name} in {environment}")
    
    def _parse_log_level(self, log_level: Union[str, int]) -> int:
        """Parse log level from string or integer."""
        if isinstance(log_level, str):
            try:
                return getattr(logging, log_level.upper())
            except AttributeError:
                return logging.INFO
        elif isinstance(log_level, int):
            return log_level
        else:
            return logging.INFO
    
    def _setup_log_dir(self, log_dir: Optional[str]) -> Optional[Path]:
        """Setup log directory."""
        if not log_dir:
            # Default to logs directory in current working directory
            log_dir = Path.cwd() / "logs"
        
        log_path = Path(log_dir)
        
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            return log_path
        except Exception as e:
            logger.error(f"Failed to create log directory {log_path}: {e}")
            return None
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Clear existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Configure root logger
        root_logger.setLevel(self.log_level)
        
        # Create context filter
        default_context = LogContext(
            service=self.service_name,
            environment=self.environment,
        )
        self.context_filter = ContextFilter(default_context=default_context)
        
        # Create formatter
        if self.enable_json_output:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Add console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(self.context_filter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if self.enable_file_logging and self.log_dir:
            # Application logs
            app_log_file = self.log_dir / f"{self.service_name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                filename=app_log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=10,
                encoding='utf-8',
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(self.context_filter)
            root_logger.addHandler(file_handler)
            
            # Error logs
            error_log_file = self.log_dir / f"{self.service_name}.error.log"
            error_handler = logging.handlers.RotatingFileHandler(
                filename=error_log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=10,
                encoding='utf-8',
            )
            error_handler.setFormatter(formatter)
            error_handler.addFilter(self.context_filter)
            error_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_handler)
        
        # Setup specialized loggers
        if self.enable_audit_logging:
            self.audit_logger = AuditLogger()
        
        if self.enable_performance_logging:
            self.performance_logger = PerformanceLogger()
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
    
    def _configure_third_party_loggers(self) -> None:
        """Configure logging for third-party libraries."""
        # Reduce noise from noisy libraries
        noisy_loggers = [
            "urllib3",
            "requests",
            "botocore",
            "boto3",
            "s3transfer",
            "azure",
            "google",
            "openai",
            "httpx",
            "asyncio",
            "uvicorn.access",
        ]
        
        for logger_name in noisy_loggers:
            lib_logger = logging.getLogger(logger_name)
            lib_logger.setLevel(logging.WARNING)
            lib_logger.propagate = True
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """
        Get a logger instance.
        
        Args:
            name: Logger name (defaults to caller's module name)
            
        Returns:
            logging.Logger: Configured logger instance
        """
        if name is None:
            # Get caller's module name
            frame = inspect.currentframe().f_back
            module = inspect.getmodule(frame)
            name = module.__name__ if module else "unknown"
        
        logger_obj = logging.getLogger(name)
        
        # Add context filter if not already present
        if not any(isinstance(f, ContextFilter) for f in logger_obj.filters):
            logger_obj.addFilter(self.context_filter)
        
        return logger_obj
    
    def set_context(self, context: LogContext) -> None:
        """
        Set logging context.
        
        Args:
            context: Logging context to set
        """
        if self.context_filter:
            self.context_filter.set_context(context)
    
    def get_context(self) -> LogContext:
        """
        Get current logging context.
        
        Returns:
            LogContext: Current logging context
        """
        if self.context_filter:
            return self.context_filter.get_context()
        return LogContext()
    
    def update_context(self, **kwargs) -> None:
        """
        Update current logging context.
        
        Args:
            **kwargs: Context fields to update
        """
        if self.context_filter:
            self.context_filter.update_context(**kwargs)
    
    def clear_context(self) -> None:
        """Clear logging context (reset to default)."""
        if self.context_filter:
            self.context_filter.clear_context()
    
    @contextmanager
    def context(self, **kwargs) -> Generator[LogContext, None, None]:
        """
        Context manager for temporary logging context.
        
        Args:
            **kwargs: Context fields to set
            
        Yields:
            LogContext: Temporary context
        """
        old_context = self.get_context()
        new_context = old_context.merge(LogContext(**kwargs))
        
        self.set_context(new_context)
        
        try:
            yield new_context
        finally:
            self.set_context(old_context)
    
    def log_with_context(
        self,
        level: Union[str, int],
        message: str,
        context: Optional[LogContext] = None,
        **kwargs
    ) -> None:
        """
        Log a message with specific context.
        
        Args:
            level: Log level
            message: Message to log
            context: Context to use for this log
            **kwargs: Additional fields to add to log
        """
        # Get appropriate logger
        caller_frame = inspect.currentframe().f_back
        caller_module = inspect.getmodule(caller_frame)
        logger_name = caller_module.__name__ if caller_module else "unknown"
        logger_obj = self.get_logger(logger_name)
        
        # Parse log level
        if isinstance(level, str):
            log_level = self._parse_log_level(level)
        else:
            log_level = level
        
        # Create log record with context
        current_context = self.get_context()
        if context:
            effective_context = current_context.merge(context)
        else:
            effective_context = current_context
        
        # Add extra kwargs to context
        if kwargs:
            effective_context = effective_context.merge(
                LogContext(custom_fields=kwargs)
            )
        
        # Temporarily set context and log
        old_context = self.get_context()
        self.set_context(effective_context)
        
        try:
            logger_obj.log(log_level, message)
            
            # Update statistics
            self.stats["log_count"] += 1
            self.stats["last_log_time"] = datetime.utcnow()
            
            if log_level >= logging.WARNING:
                self.stats["warning_count"] += 1
            if log_level >= logging.ERROR:
                self.stats["error_count"] += 1
                
        finally:
            self.set_context(old_context)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.log_with_context(logging.CRITICAL, message, **kwargs)
    
    def audit(self) -> Optional[AuditLogger]:
        """Get audit logger."""
        return self.audit_logger
    
    def performance(self) -> Optional[PerformanceLogger]:
        """Get performance logger."""
        return self.performance_logger
    
    def set_level(self, level: Union[str, int]) -> None:
        """
        Set log level for root logger.
        
        Args:
            level: Log level
        """
        log_level = self._parse_log_level(level)
        logging.getLogger().setLevel(log_level)
        logger.info(f"Log level set to {logging.getLevelName(log_level)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Add handler statistics
        root_logger = logging.getLogger()
        handler_stats = []
        
        for handler in root_logger.handlers:
            handler_info = {
                "type": handler.__class__.__name__,
                "level": logging.getLevelName(handler.level),
            }
            
            if hasattr(handler, 'baseFilename'):
                handler_info["file"] = handler.baseFilename
            
            handler_stats.append(handler_info)
        
        stats["handlers"] = handler_stats
        
        return stats
    
    def rotate_logs(self) -> None:
        """Rotate log files."""
        root_logger = logging.getLogger()
        
        for handler in root_logger.handlers:
            if hasattr(handler, 'doRollover'):
                try:
                    handler.doRollover()
                    logger.info(f"Rotated logs for {handler.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Failed to rotate logs: {e}")
    
    def cleanup(self) -> None:
        """Clean up logging resources."""
        # Flush all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.flush()
        
        logger.info("LogManager cleaned up")


# Global log manager instance
_log_manager: Optional[LogManager] = None


def setup_logging(
    service_name: str = "worldbrief-360",
    environment: str = "development",
    log_level: Union[str, int] = "INFO",
    log_dir: Optional[str] = None,
    **kwargs
) -> LogManager:
    """
    Set up logging system.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        log_level: Default log level
        log_dir: Directory for log files
        **kwargs: Additional configuration
        
    Returns:
        LogManager: Configured log manager instance
    """
    global _log_manager
    
    if _log_manager is not None:
        logger.warning("Logging already set up. Returning existing instance.")
        return _log_manager
    
    _log_manager = LogManager(
        service_name=service_name,
        environment=environment,
        log_level=log_level,
        log_dir=log_dir,
        **kwargs
    )
    
    return _log_manager


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to caller's module name)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Raises:
        RuntimeError: If logging is not set up
    """
    if _log_manager is None:
        # Setup minimal logging if not already set up
        _log_manager = setup_logging()
    
    return _log_manager.get_logger(name)


def get_log_manager() -> LogManager:
    """
    Get the global log manager instance.
    
    Returns:
        LogManager: Global log manager instance
        
    Raises:
        RuntimeError: If logging is not set up
    """
    if _log_manager is None:
        raise RuntimeError("Logging not set up. Call setup_logging() first.")
    
    return _log_manager


# Context manager for logging context
@contextmanager
def logging_context(**kwargs) -> Generator[LogContext, None, None]:
    """
    Context manager for temporary logging context.
    
    Args:
        **kwargs: Context fields to set
        
    Yields:
        LogContext: Temporary context
    """
    log_manager = get_log_manager()
    
    with log_manager.context(**kwargs) as ctx:
        yield ctx


# Decorator for logging function calls
def log_function_call(
    level: str = "INFO",
    log_args: bool = True,
    log_result: bool = False,
    log_duration: bool = True,
    include_context: bool = True,
):
    """
    Decorator to log function calls.
    
    Args:
        level: Log level
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_duration: Whether to log function duration
        include_context: Whether to include logging context
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(func.__module__)
            
            # Prepare log context
            log_extra = {}
            if include_context:
                log_manager = get_log_manager()
                context = log_manager.get_context()
                log_extra.update(context.to_dict())
            
            # Log function call
            call_message = f"Calling {func.__name__}"
            if log_args:
                call_message += f" with args={args}, kwargs={kwargs}"
            
            logger.log(getattr(logging, level.upper()), call_message, extra=log_extra)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log success
                success_message = f"{func.__name__} completed"
                if log_duration:
                    success_message += f" in {duration:.3f}s"
                if log_result:
                    success_message += f" with result={result}"
                
                logger.log(getattr(logging, level.upper()), success_message, extra=log_extra)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                error_message = f"{func.__name__} failed after {duration:.3f}s: {str(e)}"
                logger.error(error_message, extra=log_extra, exc_info=True)
                
                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger(func.__module__)
            
            # Prepare log context
            log_extra = {}
            if include_context:
                log_manager = get_log_manager()
                context = log_manager.get_context()
                log_extra.update(context.to_dict())
            
            # Log function call
            call_message = f"Calling {func.__name__}"
            if log_args:
                call_message += f" with args={args}, kwargs={kwargs}"
            
            logger.log(getattr(logging, level.upper()), call_message, extra=log_extra)
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log success
                success_message = f"{func.__name__} completed"
                if log_duration:
                    success_message += f" in {duration:.3f}s"
                if log_result:
                    success_message += f" with result={result}"
                
                logger.log(getattr(logging, level.upper()), success_message, extra=log_extra)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error
                error_message = f"{func.__name__} failed after {duration:.3f}s: {str(e)}"
                logger.error(error_message, extra=log_extra, exc_info=True)
                
                raise
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    # Setup logging
    log_manager = setup_logging(
        service_name="test-service",
        environment="development",
        log_level="DEBUG",
        enable_file_logging=True,
        enable_audit_logging=True,
        enable_performance_logging=True,
    )
    
    # Get logger
    logger = get_logger(__name__)
    
    # Basic logging
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    # Log with context
    with logging_context(
        request_id="req-123",
        user_id="user-456",
        operation="test_operation"
    ):
        logger.info("Message with context")
        
        # Update context
        log_manager.update_context(component="test_component")
        logger.info("Message with updated context")
    
    # Use decorator
    @log_function_call(level="INFO", log_args=True, log_duration=True)
    def test_function(x, y):
        time.sleep(0.1)
        return x + y
    
    result = test_function(10, 20)
    print(f"Function result: {result}")
    
    # Audit logging
    audit_logger = log_manager.audit()
    if audit_logger:
        audit_logger.log_access(
            user_id="user-123",
            action="login",
            resource="auth",
            success=True,
            ip_address="192.168.1.100",
        )
    
    # Performance logging
    perf_logger = log_manager.performance()
    if perf_logger:
        perf_logger.log_operation(
            operation="database_query",
            duration_ms=150.5,
            success=True,
            component="database",
        )
    
    # Get statistics
    stats = log_manager.get_stats()
    print(f"\nLogging statistics: {stats}")
    
    # Cleanup
    log_manager.cleanup()