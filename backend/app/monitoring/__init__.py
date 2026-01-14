"""
Monitoring and Observability Module

This module provides comprehensive monitoring, logging, metrics, and tracing
capabilities for the WorldBrief 360 application. It integrates with:

- Prometheus for metrics collection
- OpenTelemetry for distributed tracing
- Structured logging with context
- Health checks and performance monitoring
- Alerting and anomaly detection
"""

import logging
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from .metrics import setup_metrics, get_metrics_registry, record_metrics
from .tracing import setup_tracing, get_tracer, trace_function, trace_coroutine
from .logging import setup_logging, get_logger, LogContext
from .alerts import setup_alerts, AlertManager, AlertSeverity
from .health_checks import setup_health_checks, HealthCheckRegistry, HealthStatus
from .performance import PerformanceMonitor, setup_performance_monitoring

__version__ = "1.0.0"
__author__ = "WorldBrief 360 Team"
__description__ = "Monitoring and observability for WorldBrief 360"

# Module-level configuration
_config: Optional[Dict[str, Any]] = None
_initialized: bool = False

# Module components
_metrics_registry = None
_tracer_provider = None
_logger = None
_alert_manager = None
_health_registry = None
_performance_monitor = None


def init_monitoring(
    service_name: str = "worldbrief-360",
    environment: str = "development",
    metrics_enabled: bool = True,
    tracing_enabled: bool = True,
    logging_enabled: bool = True,
    health_checks_enabled: bool = True,
    performance_monitoring_enabled: bool = True,
    **kwargs
) -> None:
    """
    Initialize all monitoring components.
    
    Args:
        service_name: Name of the service for monitoring
        environment: Deployment environment (development, staging, production)
        metrics_enabled: Enable Prometheus metrics
        tracing_enabled: Enable OpenTelemetry tracing
        logging_enabled: Enable structured logging
        health_checks_enabled: Enable health checks
        performance_monitoring_enabled: Enable performance monitoring
        **kwargs: Additional configuration options
    """
    global _config, _initialized, _metrics_registry, _tracer_provider
    global _logger, _alert_manager, _health_registry, _performance_monitor
    
    if _initialized:
        logger.warning("Monitoring already initialized. Skipping re-initialization.")
        return
    
    _config = {
        "service_name": service_name,
        "environment": environment,
        "metrics_enabled": metrics_enabled,
        "tracing_enabled": tracing_enabled,
        "logging_enabled": logging_enabled,
        "health_checks_enabled": health_checks_enabled,
        "performance_monitoring_enabled": performance_monitoring_enabled,
        **kwargs
    }
    
    try:
        # Initialize logging first (needed for other components)
        if logging_enabled:
            _logger = setup_logging(
                service_name=service_name,
                environment=environment,
                **kwargs.get("logging_config", {})
            )
        
        # Initialize metrics
        if metrics_enabled:
            _metrics_registry = setup_metrics(
                service_name=service_name,
                **kwargs.get("metrics_config", {})
            )
        
        # Initialize tracing
        if tracing_enabled:
            _tracer_provider = setup_tracing(
                service_name=service_name,
                environment=environment,
                **kwargs.get("tracing_config", {})
            )
        
        # Initialize alerting
        _alert_manager = setup_alerts(
            service_name=service_name,
            environment=environment,
            **kwargs.get("alerts_config", {})
        )
        
        # Initialize health checks
        if health_checks_enabled:
            _health_registry = setup_health_checks(
                service_name=service_name,
                **kwargs.get("health_config", {})
            )
        
        # Initialize performance monitoring
        if performance_monitoring_enabled:
            _performance_monitor = setup_performance_monitoring(
                service_name=service_name,
                **kwargs.get("performance_config", {})
            )
        
        _initialized = True
        
        logger = get_logger(__name__)
        logger.info(
            "Monitoring initialized",
            extra={
                "service_name": service_name,
                "environment": environment,
                "components": {
                    "metrics": metrics_enabled,
                    "tracing": tracing_enabled,
                    "logging": logging_enabled,
                    "health_checks": health_checks_enabled,
                    "performance": performance_monitoring_enabled,
                }
            }
        )
        
    except Exception as e:
        # If initialization fails, log but don't crash
        logging.error(f"Failed to initialize monitoring: {e}", exc_info=True)
        _initialized = False


def get_config() -> Optional[Dict[str, Any]]:
    """
    Get the monitoring configuration.
    
    Returns:
        Optional[Dict[str, Any]]: Configuration dictionary or None
    """
    return _config


def is_initialized() -> bool:
    """
    Check if monitoring is initialized.
    
    Returns:
        bool: True if monitoring is initialized
    """
    return _initialized


@asynccontextmanager
async def monitoring_context(
    operation_name: str,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Context manager for monitoring a block of code.
    
    Args:
        operation_name: Name of the operation
        context: Additional context for logging/tracing
        **kwargs: Additional parameters
        
    Yields:
        Dict[str, Any]: Context dictionary with monitoring info
    """
    from contextvars import ContextVar
    
    operation_ctx = ContextVar("operation_ctx", default={})
    
    # Start tracing span if enabled
    if tracing_enabled():
        tracer = get_tracer()
        span = tracer.start_span(operation_name)
        span.set_attribute("operation.name", operation_name)
        
        if context:
            for key, value in context.items():
                span.set_attribute(f"context.{key}", str(value))
    
    # Create log context
    log_context = LogContext(
        operation=operation_name,
        trace_id=getattr(span, 'context', {}).get('trace_id', '') if tracing_enabled() else '',
        **(context or {})
    )
    
    # Record metrics start
    if metrics_enabled():
        from .metrics import Counter, Histogram
        Counter(f"{operation_name}.started").inc()
        timer = Histogram(f"{operation_name}.duration").time()
    
    try:
        # Set context for this operation
        ctx_data = {
            "operation_name": operation_name,
            "log_context": log_context,
            "span": span if tracing_enabled() else None,
            "timer": timer if metrics_enabled() else None,
            **(context or {})
        }
        
        token = operation_ctx.set(ctx_data)
        
        # Log operation start
        if logging_enabled():
            logger = get_logger(__name__)
            logger.info(
                f"Starting operation: {operation_name}",
                extra=log_context.dict()
            )
        
        yield ctx_data
        
        # Log operation success
        if logging_enabled():
            logger.info(
                f"Completed operation: {operation_name}",
                extra={**log_context.dict(), "status": "success"}
            )
        
        # Record success metrics
        if metrics_enabled():
            Counter(f"{operation_name}.success").inc()
            
    except Exception as e:
        # Log operation failure
        if logging_enabled():
            logger = get_logger(__name__)
            logger.error(
                f"Failed operation: {operation_name}",
                extra={**log_context.dict(), "status": "error", "error": str(e)},
                exc_info=True
            )
        
        # Record error metrics
        if metrics_enabled():
            Counter(f"{operation_name}.errors").inc()
        
        # Set span error
        if tracing_enabled() and 'span' in locals():
            span.record_exception(e)
            span.set_status("ERROR")
        
        raise
        
    finally:
        # End span
        if tracing_enabled() and 'span' in locals():
            span.end()
        
        # End timer
        if metrics_enabled() and 'timer' in locals():
            timer()
        
        # Reset context
        if 'token' in locals():
            operation_ctx.reset(token)


def metrics_enabled() -> bool:
    """Check if metrics are enabled."""
    return _config and _config.get("metrics_enabled", False)


def tracing_enabled() -> bool:
    """Check if tracing is enabled."""
    return _config and _config.get("tracing_enabled", False)


def logging_enabled() -> bool:
    """Check if structured logging is enabled."""
    return _config and _config.get("logging_enabled", False)


def health_checks_enabled() -> bool:
    """Check if health checks are enabled."""
    return _config and _config.get("health_checks_enabled", False)


def performance_monitoring_enabled() -> bool:
    """Check if performance monitoring is enabled."""
    return _config and _config.get("performance_monitoring_enabled", False)


def cleanup() -> None:
    """Clean up monitoring resources."""
    global _initialized, _tracer_provider
    
    if _tracer_provider:
        _tracer_provider.shutdown()
    
    if _performance_monitor:
        _performance_monitor.stop()
    
    _initialized = False
    
    logger = get_logger(__name__)
    logger.info("Monitoring cleaned up")


# Re-export important components
__all__ = [
    # Initialization
    "init_monitoring",
    "get_config",
    "is_initialized",
    "cleanup",
    "monitoring_context",
    
    # Configuration checks
    "metrics_enabled",
    "tracing_enabled",
    "logging_enabled",
    "health_checks_enabled",
    "performance_monitoring_enabled",
    
    # Metrics
    "setup_metrics",
    "get_metrics_registry",
    "record_metrics",
    
    # Tracing
    "setup_tracing",
    "get_tracer",
    "trace_function",
    "trace_coroutine",
    
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    
    # Alerts
    "setup_alerts",
    "AlertManager",
    "AlertSeverity",
    
    # Health Checks
    "setup_health_checks",
    "HealthCheckRegistry",
    "HealthStatus",
    
    # Performance
    "PerformanceMonitor",
    "setup_performance_monitoring",
    
    # Constants
    "__version__",
    "__author__",
    "__description__",
]

# Set up a default logger for the module itself
logging.getLogger(__name__).addHandler(logging.NullHandler())