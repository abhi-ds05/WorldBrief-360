"""
Application metrics collection, monitoring, and observability.
Supports Prometheus, StatsD, and custom metrics.
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict, deque
import statistics
import json
import logging

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    REGISTRY,
    CollectorRegistry,
)
from prometheus_client.exposition import start_http_server
from prometheus_client.metrics import MetricWrapperBase

from .config import get_config
from .logging_config import get_logger
from .exceptions import AppException

logger = get_logger(__name__)


class MetricType(str, Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AggregationMethod(str, Enum):
    """Aggregation methods for metrics."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


@dataclass
class MetricDefinition:
    """Definition of a metric."""
    name: str
    type: MetricType
    description: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    quantiles: Optional[List[float]] = None
    max_age_seconds: int = 60
    age_buckets: int = 5
    help_text: str = ""


@dataclass
class MetricValue:
    """A single metric value with timestamp."""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricAggregator:
    """Aggregates metric values over time windows."""
    
    def __init__(self, window_size: int = 60, max_points: int = 1000):
        """
        Initialize aggregator.
        
        Args:
            window_size: Window size in seconds
            max_points: Maximum number of points to keep
        """
        self.window_size = window_size
        self.max_points = max_points
        self.data: Dict[str, deque] = {}
        self.lock = threading.RLock()
    
    def add_value(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add a metric value."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self.lock:
            if metric_name not in self.data:
                self.data[metric_name] = deque(maxlen=self.max_points)
            
            self.data[metric_name].append((timestamp, value))
    
    def get_aggregated_value(
        self,
        metric_name: str,
        method: AggregationMethod,
        window_seconds: Optional[int] = None,
        percentile: Optional[float] = None
    ) -> Optional[float]:
        """Get aggregated value for a metric."""
        window_seconds = window_seconds or self.window_size
        
        with self.lock:
            if metric_name not in self.data:
                return None
            
            cutoff = datetime.now() - timedelta(seconds=window_seconds)
            values = [
                value for ts, value in self.data[metric_name]
                if ts >= cutoff
            ]
            
            if not values:
                return None
            
            if method == AggregationMethod.SUM:
                return sum(values)
            elif method == AggregationMethod.AVG:
                return statistics.mean(values)
            elif method == AggregationMethod.MIN:
                return min(values)
            elif method == AggregationMethod.MAX:
                return max(values)
            elif method == AggregationMethod.COUNT:
                return len(values)
            elif method == AggregationMethod.PERCENTILE and percentile is not None:
                return statistics.quantiles(values, n=100)[int(percentile) - 1]
            
            return None
    
    def clear(self, metric_name: Optional[str] = None):
        """Clear metric data."""
        with self.lock:
            if metric_name:
                if metric_name in self.data:
                    self.data[metric_name].clear()
            else:
                self.data.clear()


class MetricsCollector:
    """
    Main metrics collector supporting multiple backends.
    """
    
    def __init__(self, enable_prometheus: bool = True, enable_statsd: bool = False):
        """
        Initialize metrics collector.
        
        Args:
            enable_prometheus: Enable Prometheus metrics
            enable_statsd: Enable StatsD metrics
        """
        self.config = get_config()
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, MetricWrapperBase] = {}
        self.custom_metrics: Dict[str, MetricDefinition] = {}
        self.aggregator = MetricAggregator()
        
        # Prometheus setup
        if enable_prometheus:
            self._init_prometheus()
        
        # StatsD setup
        if enable_statsd:
            self._init_statsd()
        
        # Internal tracking
        self._start_time = datetime.now()
        self._request_counts: Dict[str, int] = defaultdict(int)
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        
        # Default metrics
        self._register_default_metrics()
    
    def _init_prometheus(self) -> None:
        """Initialize Prometheus metrics."""
        try:
            # Start Prometheus HTTP server if enabled
            if self.config.enable_metrics:
                port = getattr(self.config, 'metrics_port', 9090)
                start_http_server(port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.warning(f"Failed to start Prometheus server: {e}")
    
    def _init_statsd(self) -> None:
        """Initialize StatsD client."""
        try:
            from statsd import StatsClient
            
            statsd_host = getattr(self.config, 'statsd_host', 'localhost')
            statsd_port = getattr(self.config, 'statsd_port', 8125)
            self.statsd = StatsClient(host=statsd_host, port=statsd_port)
            logger.info(f"StatsD client initialized: {statsd_host}:{statsd_port}")
        except ImportError:
            logger.warning("StatsD client not available. Install with: pip install statsd")
        except Exception as e:
            logger.warning(f"Failed to initialize StatsD: {e}")
    
    def _register_default_metrics(self) -> None:
        """Register default application metrics."""
        
        # Application info
        self.register_gauge(
            name="app_info",
            description="Application information",
            labels=["app_name", "version", "environment"]
        )
        
        # Request metrics
        self.register_counter(
            name="http_requests_total",
            description="Total HTTP requests",
            labels=["method", "endpoint", "status_code"]
        )
        
        self.register_histogram(
            name="http_request_duration_seconds",
            description="HTTP request duration in seconds",
            labels=["method", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Error metrics
        self.register_counter(
            name="errors_total",
            description="Total errors",
            labels=["type", "source"]
        )
        
        # Cache metrics
        self.register_counter(
            name="cache_operations_total",
            description="Total cache operations",
            labels=["operation", "backend", "status"]
        )
        
        self.register_gauge(
            name="cache_size",
            description="Cache size in bytes",
            labels=["backend"]
        )
        
        # Database metrics
        self.register_counter(
            name="database_queries_total",
            description="Total database queries",
            labels=["operation", "table", "status"]
        )
        
        self.register_histogram(
            name="database_query_duration_seconds",
            description="Database query duration in seconds",
            labels=["operation", "table"],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
        )
        
        # System metrics
        self.register_gauge(
            name="memory_usage_bytes",
            description="Memory usage in bytes",
            labels=["type"]
        )
        
        self.register_gauge(
            name="cpu_usage_percent",
            description="CPU usage percentage"
        )
        
        # Custom business metrics
        self.register_counter(
            name="user_actions_total",
            description="Total user actions",
            labels=["action_type", "user_id"]
        )
        
        self.register_gauge(
            name="active_users",
            description="Number of active users"
        )
        
        # Set application info
        self.set_gauge(
            "app_info",
            1,
            labels={
                "app_name": self.config.app_name,
                "version": "1.0.0",  # Should come from version.py
                "environment": self.config.environment.value
            }
        )
    
    def register_counter(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Register a counter metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        labels = labels or []
        metric = Counter(
            name=name,
            documentation=description,
            labelnames=labels,
            registry=self.registry
        )
        self.metrics[name] = metric
    
    def register_gauge(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None
    ) -> None:
        """
        Register a gauge metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
        """
        labels = labels or []
        metric = Gauge(
            name=name,
            documentation=description,
            labelnames=labels,
            registry=self.registry
        )
        self.metrics[name] = metric
    
    def register_histogram(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        buckets: Optional[List[float]] = None
    ) -> None:
        """
        Register a histogram metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            buckets: Histogram buckets
        """
        labels = labels or []
        buckets = buckets or Histogram.DEFAULT_BUCKETS
        metric = Histogram(
            name=name,
            documentation=description,
            labelnames=labels,
            buckets=buckets,
            registry=self.registry
        )
        self.metrics[name] = metric
    
    def register_summary(
        self,
        name: str,
        description: str = "",
        labels: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None
    ) -> None:
        """
        Register a summary metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: List of label names
            quantiles: Quantiles to calculate
        """
        labels = labels or []
        quantiles = quantiles or [0.5, 0.95, 0.99]
        metric = Summary(
            name=name,
            documentation=description,
            labelnames=labels,
            registry=self.registry
        )
        self.metrics[name] = metric
    
    def increment_counter(
        self,
        name: str,
        amount: float = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Metric name
            amount: Amount to increment
            labels: Label values
        """
        if name not in self.metrics:
            logger.warning(f"Counter metric not registered: {name}")
            return
        
        metric = self.metrics[name]
        if not isinstance(metric, Counter):
            logger.error(f"Metric {name} is not a counter")
            return
        
        try:
            if labels:
                metric.labels(**labels).inc(amount)
            else:
                metric.inc(amount)
            
            # Also track in aggregator
            self.aggregator.add_value(f"{name}_counter", amount)
        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {e}")
    
    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Metric name
            value: Gauge value
            labels: Label values
        """
        if name not in self.metrics:
            logger.warning(f"Gauge metric not registered: {name}")
            return
        
        metric = self.metrics[name]
        if not isinstance(metric, Gauge):
            logger.error(f"Metric {name} is not a gauge")
            return
        
        try:
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
            
            # Also track in aggregator
            self.aggregator.add_value(f"{name}_gauge", value)
        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {e}")
    
    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observe a histogram metric value.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Label values
        """
        if name not in self.metrics:
            logger.warning(f"Histogram metric not registered: {name}")
            return
        
        metric = self.metrics[name]
        if not isinstance(metric, Histogram):
            logger.error(f"Metric {name} is not a histogram")
            return
        
        try:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
            
            # Also track in aggregator
            self.aggregator.add_value(f"{name}_histogram", value)
        except Exception as e:
            logger.error(f"Failed to observe histogram {name}: {e}")
    
    def observe_summary(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observe a summary metric value.
        
        Args:
            name: Metric name
            value: Value to observe
            labels: Label values
        """
        if name not in self.metrics:
            logger.warning(f"Summary metric not registered: {name}")
            return
        
        metric = self.metrics[name]
        if not isinstance(metric, Summary):
            logger.error(f"Metric {name} is not a summary")
            return
        
        try:
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
            
            # Also track in aggregator
            self.aggregator.add_value(f"{name}_summary", value)
        except Exception as e:
            logger.error(f"Failed to observe summary {name}: {e}")
    
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Create a timer context manager for measuring duration.
        
        Args:
            name: Metric name
            labels: Label values
            
        Returns:
            Timer context manager
        """
        return Timer(self, name, labels)
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float
    ) -> None:
        """
        Record an HTTP request.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status_code: HTTP status code
            duration: Request duration in seconds
        """
        labels = {
            "method": method,
            "endpoint": endpoint,
            "status_code": str(status_code)
        }
        
        self.increment_counter("http_requests_total", labels=labels)
        self.observe_histogram("http_request_duration_seconds", duration, labels=labels)
        
        # Track in internal stats
        key = f"{method}:{endpoint}"
        self._request_counts[key] += 1
        self._latencies[key].append(duration)
        
        # Prune old latency data
        if len(self._latencies[key]) > 1000:
            self._latencies[key] = self._latencies[key][-500:]
    
    def record_error(self, error_type: str, source: str) -> None:
        """
        Record an error.
        
        Args:
            error_type: Type of error
            source: Source of error
        """
        labels = {"type": error_type, "source": source}
        self.increment_counter("errors_total", labels=labels)
        
        # Track in internal stats
        key = f"{error_type}:{source}"
        self._error_counts[key] += 1
    
    def record_cache_operation(
        self,
        operation: str,
        backend: str,
        success: bool
    ) -> None:
        """
        Record a cache operation.
        
        Args:
            operation: Cache operation (get, set, delete, etc.)
            backend: Cache backend (memory, redis, file)
            success: Whether operation was successful
        """
        status = "success" if success else "error"
        labels = {"operation": operation, "backend": backend, "status": status}
        self.increment_counter("cache_operations_total", labels=labels)
    
    def record_database_operation(
        self,
        operation: str,
        table: str,
        success: bool,
        duration: float
    ) -> None:
        """
        Record a database operation.
        
        Args:
            operation: Database operation (select, insert, update, delete)
            table: Database table
            success: Whether operation was successful
            duration: Operation duration in seconds
        """
        status = "success" if success else "error"
        labels = {"operation": operation, "table": table, "status": status}
        
        self.increment_counter("database_queries_total", labels=labels)
        self.observe_histogram("database_query_duration_seconds", duration, labels=labels)
    
    def update_system_metrics(self) -> None:
        """Update system metrics (CPU, memory, etc.)."""
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Memory metrics
            memory_info = process.memory_info()
            self.set_gauge(
                "memory_usage_bytes",
                memory_info.rss,
                labels={"type": "rss"}
            )
            self.set_gauge(
                "memory_usage_bytes",
                memory_info.vms,
                labels={"type": "vms"}
            )
            
            # CPU metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            self.set_gauge("cpu_usage_percent", cpu_percent)
            
            # System-wide metrics
            system_memory = psutil.virtual_memory()
            self.set_gauge(
                "memory_usage_bytes",
                system_memory.used,
                labels={"type": "system_used"}
            )
            self.set_gauge(
                "memory_usage_bytes",
                system_memory.total,
                labels={"type": "system_total"}
            )
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            self.set_gauge(
                "disk_usage_bytes",
                disk_usage.used,
                labels={"type": "used", "mount": "/"}
            )
            self.set_gauge(
                "disk_usage_bytes",
                disk_usage.total,
                labels={"type": "total", "mount": "/"}
            )
            
        except ImportError:
            logger.warning("psutil not installed. Install with: pip install psutil")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def get_metrics(self, format: str = "prometheus") -> Union[str, Dict[str, Any]]:
        """
        Get all metrics in specified format.
        
        Args:
            format: Output format ("prometheus", "json")
            
        Returns:
            Metrics in requested format
        """
        if format == "prometheus":
            return generate_latest(self.registry).decode('utf-8')
        elif format == "json":
            metrics_data = {}
            
            for name, metric in self.metrics.items():
                samples = list(metric.collect()[0].samples)
                metrics_data[name] = [
                    {
                        "value": sample.value,
                        "labels": sample.labels,
                        "timestamp": sample.timestamp
                    }
                    for sample in samples
                ]
            
            # Add aggregated metrics
            metrics_data["aggregated"] = self.get_aggregated_metrics()
            
            # Add internal stats
            metrics_data["internal"] = {
                "uptime": (datetime.now() - self._start_time).total_seconds(),
                "request_counts": dict(self._request_counts),
                "error_counts": dict(self._error_counts),
                "latency_stats": {
                    key: {
                        "avg": statistics.mean(values) if values else 0,
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0,
                        "count": len(values)
                    }
                    for key, values in self._latencies.items()
                }
            }
            
            return metrics_data
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_aggregated_metrics(self, window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get aggregated metrics over a time window.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}
        
        # Define aggregations for different metrics
        aggregations = {
            "http_requests_total_counter": AggregationMethod.SUM,
            "http_request_duration_seconds_histogram": AggregationMethod.AVG,
            "errors_total_counter": AggregationMethod.SUM,
            "database_queries_total_counter": AggregationMethod.SUM,
            "cache_operations_total_counter": AggregationMethod.SUM,
        }
        
        for metric_name, method in aggregations.items():
            value = self.aggregator.get_aggregated_value(
                metric_name,
                method,
                window_seconds
            )
            if value is not None:
                aggregated[metric_name] = value
        
        return aggregated
    
    def clear_metrics(self, metric_name: Optional[str] = None) -> None:
        """
        Clear metrics data.
        
        Args:
            metric_name: Optional specific metric to clear
        """
        if metric_name:
            if metric_name in self.metrics:
                # Prometheus doesn't support clearing, but we can unregister
                self.registry.unregister(self.metrics[metric_name])
                del self.metrics[metric_name]
        else:
            # Clear all custom metrics
            for name in list(self.metrics.keys()):
                if name not in ["app_info"]:  # Keep app info
                    self.registry.unregister(self.metrics[name])
                    del self.metrics[name]
            
            # Clear aggregator
            self.aggregator.clear()
            
            # Reset internal stats
            self._request_counts.clear()
            self._error_counts.clear()
            self._latencies.clear()


class Timer:
    """Context manager for timing code execution."""
    
    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        Initialize timer.
        
        Args:
            collector: Metrics collector instance
            name: Metric name
            labels: Label values
        """
        self.collector = collector
        self.name = name
        self.labels = labels or {}
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.collector.observe_histogram(self.name, duration, self.labels)


# Decorators for metrics collection
def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to measure function execution time.
    
    Args:
        metric_name: Metric name
        labels: Label values
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with get_metrics_collector().timer(metric_name, labels):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with get_metrics_collector().timer(metric_name, labels):
                return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.
    
    Args:
        metric_name: Metric name
        labels: Label values
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            get_metrics_collector().increment_counter(metric_name, labels=labels)
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            get_metrics_collector().increment_counter(metric_name, labels=labels)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get or create metrics collector instance.
    
    Returns:
        MetricsCollector instance
    """
    global _metrics_collector
    if _metrics_collector is None:
        config = get_config()
        _metrics_collector = MetricsCollector(
            enable_prometheus=config.enable_metrics,
            enable_statsd=getattr(config, 'enable_statsd', False)
        )
    return _metrics_collector


def initialize_metrics() -> None:
    """Initialize metrics collection."""
    get_metrics_collector()


def get_metrics(format: str = "prometheus") -> Union[str, Dict[str, Any]]:
    """
    Get all metrics.
    
    Args:
        format: Output format
        
    Returns:
        Metrics in requested format
    """
    return get_metrics_collector().get_metrics(format)


def clear_metrics(metric_name: Optional[str] = None) -> None:
    """
    Clear metrics.
    
    Args:
        metric_name: Optional specific metric to clear
    """
    get_metrics_collector().clear_metrics(metric_name)


# Convenience functions
def increment_counter(
    name: str,
    amount: float = 1,
    labels: Optional[Dict[str, str]] = None
) -> None:
    """Increment a counter metric."""
    get_metrics_collector().increment_counter(name, amount, labels)


def set_gauge(
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None
) -> None:
    """Set a gauge metric value."""
    get_metrics_collector().set_gauge(name, value, labels)


def observe_histogram(
    name: str,
    value: float,
    labels: Optional[Dict[str, str]] = None
) -> None:
    """Observe a histogram metric value."""
    get_metrics_collector().observe_histogram(name, value, labels)


def record_request(
    method: str,
    endpoint: str,
    status_code: int,
    duration: float
) -> None:
    """Record an HTTP request."""
    get_metrics_collector().record_request(method, endpoint, status_code, duration)


def record_error(error_type: str, source: str) -> None:
    """Record an error."""
    get_metrics_collector().record_error(error_type, source)


def record_cache_operation(operation: str, backend: str, success: bool) -> None:
    """Record a cache operation."""
    get_metrics_collector().record_cache_operation(operation, backend, success)


def record_database_operation(
    operation: str,
    table: str,
    success: bool,
    duration: float
) -> None:
    """Record a database operation."""
    get_metrics_collector().record_database_operation(operation, table, success, duration)


# Background task for updating system metrics
async def start_metrics_updater(interval: int = 30) -> None:
    """
    Start background task to update system metrics.
    
    Args:
        interval: Update interval in seconds
    """
    collector = get_metrics_collector()
    
    while True:
        try:
            collector.update_system_metrics()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Metrics updater error: {e}")
            await asyncio.sleep(interval)


# Example usage
if __name__ == "__main__":
    # Initialize metrics
    initialize_metrics()
    
    # Record some metrics
    increment_counter("test_counter", labels={"type": "test"})
    set_gauge("test_gauge", 42.5)
    
    # Use timer
    with get_metrics_collector().timer("test_timer"):
        time.sleep(0.1)
    
    # Get metrics
    print(get_metrics("prometheus"))