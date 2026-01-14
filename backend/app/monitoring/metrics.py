"""
Metrics Collection System

This module provides comprehensive metrics collection and monitoring with:

- Prometheus-compatible metrics
- Multiple metric types (Counter, Gauge, Histogram, Summary)
- Labels and dimensions for multi-dimensional metrics
- Metric aggregation and querying
- Performance monitoring
- Business metrics tracking
- Real-time metrics collection
- Metric exporters (Prometheus, StatsD, etc.)
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import inspect
import functools
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from abc import ABC, abstractmethod

from backend.app.monitoring import logging

from backend.app.monitoring import logging

try:
    from prometheus_client import (
        Counter as PromCounter,
        Gauge as PromGauge,
        Histogram as PromHistogram,
        Summary as PromSummary,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    
    @property
    def is_monotonic(self) -> bool:
        """Check if metric type is monotonic (only increases)."""
        return self == MetricType.COUNTER


@dataclass
class MetricLabels:
    """Labels for metrics (dimensions)."""
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return self.labels.copy()
    
    def add(self, name: str, value: str) -> None:
        """Add a label."""
        self.labels[name] = value
    
    def remove(self, name: str) -> None:
        """Remove a label."""
        self.labels.pop(name, None)
    
    def merge(self, other: 'MetricLabels') -> 'MetricLabels':
        """Merge two sets of labels."""
        merged = self.labels.copy()
        merged.update(other.labels)
        return MetricLabels(labels=merged)
    
    @property
    def sorted_items(self) -> List[Tuple[str, str]]:
        """Get sorted label items for consistent ordering."""
        return sorted(self.labels.items())


class BaseMetric(ABC):
    """Base class for all metrics."""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[MetricLabels] = None,
        namespace: str = "worldbrief",
        subsystem: str = "app",
        **kwargs
    ):
        """
        Initialize base metric.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels/dimensions
            namespace: Metric namespace
            subsystem: Metric subsystem
        """
        self.name = name
        self.description = description
        self.labels = labels or MetricLabels()
        self.namespace = namespace
        self.subsystem = subsystem
        
        # Full metric name with namespace
        self.full_name = f"{namespace}_{subsystem}_{name}"
        
        # Metadata
        self.created_at = datetime.utcnow()
        self.last_updated = None
        self.help_text = kwargs.get("help", description)
        
        # Statistics
        self.observation_count = 0
    
    @abstractmethod
    def observe(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """Record an observation."""
        pass
    
    @abstractmethod
    def get_value(self, labels: Optional[MetricLabels] = None) -> float:
        """Get current value."""
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metric metadata."""
        return {
            "name": self.name,
            "full_name": self.full_name,
            "description": self.description,
            "type": self.__class__.__name__,
            "namespace": self.namespace,
            "subsystem": self.subsystem,
            "labels": self.labels.to_dict(),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "observation_count": self.observation_count,
        }


class Counter(BaseMetric):
    """
    Counter metric that only increases.
    
    Used for counting events, requests, errors, etc.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value: float = 0.0
        self._lock = threading.RLock()
        
        # Per-label values
        self._label_values: Dict[Tuple, float] = {}
    
    def inc(self, amount: float = 1.0, labels: Optional[MetricLabels] = None) -> None:
        """
        Increment the counter.
        
        Args:
            amount: Amount to increment by
            labels: Labels for this observation
        """
        if amount < 0:
            raise ValueError("Counter can only be incremented (positive values)")
        
        with self._lock:
            self.observation_count += 1
            self.last_updated = datetime.utcnow()
            
            if labels:
                label_key = tuple(labels.sorted_items)
                self._label_values[label_key] = self._label_values.get(label_key, 0) + amount
            else:
                self._value += amount
    
    def observe(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """Alias for inc."""
        self.inc(value, labels)
    
    def get_value(self, labels: Optional[MetricLabels] = None) -> float:
        """
        Get counter value.
        
        Args:
            labels: Labels to filter by
            
        Returns:
            float: Counter value
        """
        with self._lock:
            if labels:
                label_key = tuple(labels.sorted_items)
                return self._label_values.get(label_key, 0.0)
            else:
                return self._value
    
    def reset(self, labels: Optional[MetricLabels] = None) -> None:
        """
        Reset counter.
        
        Args:
            labels: Labels to reset (if None, reset all)
        """
        with self._lock:
            if labels:
                label_key = tuple(labels.sorted_items)
                self._label_values.pop(label_key, None)
            else:
                self._value = 0.0
                self._label_values.clear()
    
    def get_all_values(self) -> Dict[MetricLabels, float]:
        """Get all label-value pairs."""
        with self._lock:
            result = {}
            for label_items, value in self._label_values.items():
                labels_dict = dict(label_items)
                result[MetricLabels(labels=labels_dict)] = value
            return result


class Gauge(BaseMetric):
    """
    Gauge metric that can go up and down.
    
    Used for current values like memory usage, queue size, temperature, etc.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value: float = 0.0
        self._lock = threading.RLock()
        
        # Per-label values
        self._label_values: Dict[Tuple, float] = {}
    
    def set(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """
        Set gauge value.
        
        Args:
            value: Value to set
            labels: Labels for this observation
        """
        with self._lock:
            self.observation_count += 1
            self.last_updated = datetime.utcnow()
            
            if labels:
                label_key = tuple(labels.sorted_items)
                self._label_values[label_key] = value
            else:
                self._value = value
    
    def inc(self, amount: float = 1.0, labels: Optional[MetricLabels] = None) -> None:
        """
        Increment gauge.
        
        Args:
            amount: Amount to increment by
            labels: Labels for this observation
        """
        with self._lock:
            self.observation_count += 1
            self.last_updated = datetime.utcnow()
            
            if labels:
                label_key = tuple(labels.sorted_items)
                current = self._label_values.get(label_key, 0.0)
                self._label_values[label_key] = current + amount
            else:
                self._value += amount
    
    def dec(self, amount: float = 1.0, labels: Optional[MetricLabels] = None) -> None:
        """
        Decrement gauge.
        
        Args:
            amount: Amount to decrement by
            labels: Labels for this observation
        """
        self.inc(-amount, labels)
    
    def observe(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """Alias for set."""
        self.set(value, labels)
    
    def get_value(self, labels: Optional[MetricLabels] = None) -> float:
        """
        Get gauge value.
        
        Args:
            labels: Labels to filter by
            
        Returns:
            float: Gauge value
        """
        with self._lock:
            if labels:
                label_key = tuple(labels.sorted_items)
                return self._label_values.get(label_key, 0.0)
            else:
                return self._value
    
    def track_in_progress(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Context manager to track something in progress.
        
        Args:
            labels: Labels for this tracking
            
        Returns:
            Callable: Context manager
        """
        @contextmanager
        def track():
            self.inc(1, labels)
            try:
                yield
            finally:
                self.dec(1, labels)
        
        return track
    
    def time(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Context manager to time a block of code.
        
        Args:
            labels: Labels for this timing
            
        Returns:
            Callable: Context manager that records duration
        """
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.set(duration, labels)
        
        return timer
    
    def get_all_values(self) -> Dict[MetricLabels, float]:
        """Get all label-value pairs."""
        with self._lock:
            result = {}
            for label_items, value in self._label_values.items():
                labels_dict = dict(label_items)
                result[MetricLabels(labels=labels_dict)] = value
            return result


class Histogram(BaseMetric):
    """
    Histogram metric for observing distributions.
    
    Counts observations in configurable buckets. Useful for request durations,
    response sizes, etc.
    """
    
    def __init__(
        self,
        *args,
        buckets: Optional[List[float]] = None,
        **kwargs
    ):
        """
        Initialize histogram.
        
        Args:
            buckets: Custom bucket boundaries
            **kwargs: Additional arguments
        """
        super().__init__(*args, **kwargs)
        
        # Default buckets (in seconds for duration)
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.buckets.sort()
        
        self._lock = threading.RLock()
        
        # Structure: {label_key: {"count": X, "sum": Y, "buckets": [bucket_counts]}}
        self._data: Dict[Tuple, Dict[str, Any]] = {}
        
        # Initialize bucket counts
        self._init_data_structure()
    
    def _init_data_structure(self) -> Dict[str, Any]:
        """Initialize data structure for histogram."""
        return {
            "count": 0,
            "sum": 0.0,
            "buckets": [0] * (len(self.buckets) + 1),  # +1 for +Inf bucket
        }
    
    def observe(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """
        Observe a value.
        
        Args:
            value: Value to observe
            labels: Labels for this observation
        """
        if value < 0:
            raise ValueError("Histogram values cannot be negative")
        
        with self._lock:
            self.observation_count += 1
            self.last_updated = datetime.utcnow()
            
            label_key = tuple(labels.sorted_items) if labels else tuple()
            
            if label_key not in self._data:
                self._data[label_key] = self._init_data_structure()
            
            data = self._data[label_key]
            data["count"] += 1
            data["sum"] += value
            
            # Find bucket
            bucket_index = len(self.buckets)  # Default to +Inf bucket
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    bucket_index = i
                    break
            
            data["buckets"][bucket_index] += 1
    
    def get_value(self, labels: Optional[MetricLabels] = None) -> Dict[str, Any]:
        """
        Get histogram statistics.
        
        Args:
            labels: Labels to filter by
            
        Returns:
            Dict[str, Any]: Histogram statistics
        """
        with self._lock:
            label_key = tuple(labels.sorted_items) if labels else tuple()
            
            if label_key not in self._data:
                return self._init_data_structure()
            
            data = self._data[label_key].copy()
            
            # Calculate cumulative bucket counts
            cumulative = 0
            cumulative_buckets = []
            for count in data["buckets"]:
                cumulative += count
                cumulative_buckets.append(cumulative)
            
            data["cumulative_buckets"] = cumulative_buckets
            
            # Calculate average
            if data["count"] > 0:
                data["avg"] = data["sum"] / data["count"]
            else:
                data["avg"] = 0.0
            
            # Add bucket boundaries
            data["bucket_boundaries"] = self.buckets + ["+Inf"]
            
            return data
    
    def time(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Context manager to time a block of code.
        
        Args:
            labels: Labels for this timing
            
        Returns:
            Callable: Context manager that records duration
        """
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.observe(duration, labels)
        
        return timer
    
    async def time_async(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Async context manager to time a block of async code.
        
        Args:
            labels: Labels for this timing
            
        Returns:
            Callable: Async context manager that records duration
        """
        @asynccontextmanager
        async def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.observe(duration, labels)
        
        return timer
    
    def get_percentile(self, percentile: float, labels: Optional[MetricLabels] = None) -> Optional[float]:
        """
        Estimate percentile from histogram.
        
        Args:
            percentile: Percentile to calculate (0-100)
            labels: Labels to filter by
            
        Returns:
            Optional[float]: Estimated percentile value
        """
        if percentile < 0 or percentile > 100:
            raise ValueError("Percentile must be between 0 and 100")
        
        data = self.get_value(labels)
        if data["count"] == 0:
            return None
        
        target_count = percentile / 100 * data["count"]
        
        # Find bucket containing the target count
        cumulative = 0
        for i, count in enumerate(data["buckets"]):
            cumulative += count
            if cumulative >= target_count:
                # Simple linear interpolation within bucket
                lower_bound = self.buckets[i - 1] if i > 0 else 0
                upper_bound = self.buckets[i] if i < len(self.buckets) else float('inf')
                
                if i == 0:
                    return lower_bound
                
                # Count in this bucket before reaching target
                prev_cumulative = cumulative - count
                count_in_bucket = target_count - prev_cumulative
                fraction = count_in_bucket / count if count > 0 else 0.5
                
                return lower_bound + fraction * (upper_bound - lower_bound)
        
        return self.buckets[-1] if self.buckets else 0.0
    
    def get_all_values(self) -> Dict[MetricLabels, Dict[str, Any]]:
        """Get all label-histogram pairs."""
        with self._lock:
            result = {}
            for label_items, data in self._data.items():
                labels_dict = dict(label_items)
                result[MetricLabels(labels=labels_dict)] = data.copy()
            return result


class Summary(BaseMetric):
    """
    Summary metric for calculating quantiles over sliding time windows.
    
    Similar to histogram but calculates quantiles directly.
    """
    
    def __init__(
        self,
        *args,
        quantiles: Optional[List[float]] = None,
        max_age: timedelta = timedelta(minutes=10),
        age_buckets: int = 5,
        **kwargs
    ):
        """
        Initialize summary.
        
        Args:
            quantiles: Quantiles to calculate (default: [0.5, 0.9, 0.95, 0.99])
            max_age: Maximum age of observations
            age_buckets: Number of age buckets
            **kwargs: Additional arguments
        """
        super().__init__(*args, **kwargs)
        
        self.quantiles = quantiles or [0.5, 0.9, 0.95, 0.99]
        self.max_age = max_age
        self.age_buckets = age_buckets
        
        self._lock = threading.RLock()
        
        # Structure: {label_key: {"observations": deque, "sum": X, "count": Y}}
        self._data: Dict[Tuple, Dict[str, Any]] = {}
    
    def observe(self, value: float, labels: Optional[MetricLabels] = None) -> None:
        """
        Observe a value.
        
        Args:
            value: Value to observe
            labels: Labels for this observation
        """
        with self._lock:
            self.observation_count += 1
            self.last_updated = datetime.utcnow()
            
            label_key = tuple(labels.sorted_items) if labels else tuple()
            
            if label_key not in self._data:
                self._data[label_key] = {
                    "observations": deque(maxlen=10000),  # Limit memory usage
                    "sum": 0.0,
                    "count": 0,
                }
            
            data = self._data[label_key]
            timestamp = time.time()
            
            # Add observation
            data["observations"].append((timestamp, value))
            data["sum"] += value
            data["count"] += 1
            
            # Clean old observations
            self._clean_old_observations(label_key)
    
    def _clean_old_observations(self, label_key: Tuple) -> None:
        """Clean observations older than max_age."""
        data = self._data[label_key]
        cutoff = time.time() - self.max_age.total_seconds()
        
        # Remove old observations from deque
        while data["observations"] and data["observations"][0][0] < cutoff:
            timestamp, value = data["observations"].popleft()
            data["sum"] -= value
            data["count"] -= 1
    
    def get_value(self, labels: Optional[MetricLabels] = None) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Args:
            labels: Labels to filter by
            
        Returns:
            Dict[str, Any]: Summary statistics
        """
        with self._lock:
            label_key = tuple(labels.sorted_items) if labels else tuple()
            
            if label_key not in self._data:
                return {"count": 0, "sum": 0.0, "quantiles": {}}
            
            data = self._data[label_key]
            self._clean_old_observations(label_key)
            
            # Calculate quantiles
            observations = [obs[1] for obs in data["observations"]]
            observations.sort()
            
            quantile_values = {}
            for q in self.quantiles:
                if observations:
                    idx = int(q * (len(observations) - 1))
                    quantile_values[f"quantile_{q}"] = observations[idx]
                else:
                    quantile_values[f"quantile_{q}"] = 0.0
            
            return {
                "count": data["count"],
                "sum": data["sum"],
                "avg": data["sum"] / data["count"] if data["count"] > 0 else 0.0,
                "quantiles": quantile_values,
                "observations_count": len(data["observations"]),
            }
    
    def time(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Context manager to time a block of code.
        
        Args:
            labels: Labels for this timing
            
        Returns:
            Callable: Context manager that records duration
        """
        @contextmanager
        def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.observe(duration, labels)
        
        return timer
    
    async def time_async(self, labels: Optional[MetricLabels] = None) -> Callable:
        """
        Async context manager to time a block of async code.
        
        Args:
            labels: Labels for this timing
            
        Returns:
            Callable: Async context manager that records duration
        """
        @asynccontextmanager
        async def timer():
            start_time = time.time()
            try:
                yield
            finally:
                duration = time.time() - start_time
                self.observe(duration, labels)
        
        return timer
    
    def get_all_values(self) -> Dict[MetricLabels, Dict[str, Any]]:
        """Get all label-summary pairs."""
        with self._lock:
            result = {}
            for label_items, data in self._data.items():
                labels_dict = dict(label_items)
                result[MetricLabels(labels=labels_dict)] = self.get_value(MetricLabels(labels=labels_dict))
            return result


class MetricsRegistry:
    """
    Registry for managing metrics.
    """
    
    def __init__(
        self,
        namespace: str = "worldbrief",
        subsystem: str = "app",
        enable_prometheus: bool = True,
        **kwargs
    ):
        """
        Initialize metrics registry.
        
        Args:
            namespace: Metric namespace
            subsystem: Metric subsystem
            enable_prometheus: Enable Prometheus integration
            **kwargs: Additional configuration
        """
        self.namespace = namespace
        self.subsystem = subsystem
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        
        # Storage
        self.metrics: Dict[str, BaseMetric] = {}
        self._lock = threading.RLock()
        
        # Prometheus integration
        self.prom_registry = None
        self.prom_metrics = {}
        
        if self.enable_prometheus:
            self._setup_prometheus()
        
        # Statistics
        self.stats = {
            "metrics_registered": 0,
            "observations_recorded": 0,
            "last_observation_time": None,
        }
        
        # Default labels
        self.default_labels = MetricLabels()
        
        logger.info(f"MetricsRegistry initialized (namespace: {namespace}, subsystem: {subsystem})")
    
    def _setup_prometheus(self) -> None:
        """Setup Prometheus integration."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Disabling Prometheus integration.")
            self.enable_prometheus = False
            return
        
        # Create separate registry to avoid conflicts
        self.prom_registry = CollectorRegistry()
        
        logger.info("Prometheus integration enabled")
    
    def register(self, metric: BaseMetric) -> None:
        """
        Register a metric.
        
        Args:
            metric: Metric to register
        """
        with self._lock:
            if metric.name in self.metrics:
                logger.warning(f"Metric '{metric.name}' already registered. Overwriting.")
            
            self.metrics[metric.name] = metric
            
            # Register with Prometheus if enabled
            if self.enable_prometheus:
                self._register_prometheus_metric(metric)
            
            self.stats["metrics_registered"] += 1
            logger.info(f"Registered metric: {metric.name}")
    
    def _register_prometheus_metric(self, metric: BaseMetric) -> None:
        """Register metric with Prometheus."""
        if not self.enable_prometheus or not self.prom_registry:
            return
        
        try:
            prom_metric = None
            
            if isinstance(metric, Counter):
                prom_metric = PromCounter(
                    name=metric.full_name,
                    documentation=metric.description,
                    labelnames=list(metric.labels.labels.keys()),
                    registry=self.prom_registry,
                )
            
            elif isinstance(metric, Gauge):
                prom_metric = PromGauge(
                    name=metric.full_name,
                    documentation=metric.description,
                    labelnames=list(metric.labels.labels.keys()),
                    registry=self.prom_registry,
                )
            
            elif isinstance(metric, Histogram):
                prom_metric = PromHistogram(
                    name=metric.full_name,
                    documentation=metric.description,
                    labelnames=list(metric.labels.labels.keys()),
                    buckets=metric.buckets,
                    registry=self.prom_registry,
                )
            
            elif isinstance(metric, Summary):
                prom_metric = PromSummary(
                    name=metric.full_name,
                    documentation=metric.description,
                    labelnames=list(metric.labels.labels.keys()),
                    registry=self.prom_registry,
                )
            
            if prom_metric:
                self.prom_metrics[metric.name] = prom_metric
                logger.debug(f"Registered metric {metric.name} with Prometheus")
                
        except Exception as e:
            logger.error(f"Failed to register metric {metric.name} with Prometheus: {e}")
    
    def create_counter(
        self,
        name: str,
        description: str,
        labels: Optional[MetricLabels] = None,
        **kwargs
    ) -> Counter:
        """
        Create and register a counter.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            **kwargs: Additional arguments
            
        Returns:
            Counter: Created counter metric
        """
        counter = Counter(
            name=name,
            description=description,
            labels=labels,
            namespace=self.namespace,
            subsystem=self.subsystem,
            **kwargs
        )
        self.register(counter)
        return counter
    
    def create_gauge(
        self,
        name: str,
        description: str,
        labels: Optional[MetricLabels] = None,
        **kwargs
    ) -> Gauge:
        """
        Create and register a gauge.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            **kwargs: Additional arguments
            
        Returns:
            Gauge: Created gauge metric
        """
        gauge = Gauge(
            name=name,
            description=description,
            labels=labels,
            namespace=self.namespace,
            subsystem=self.subsystem,
            **kwargs
        )
        self.register(gauge)
        return gauge
    
    def create_histogram(
        self,
        name: str,
        description: str,
        labels: Optional[MetricLabels] = None,
        buckets: Optional[List[float]] = None,
        **kwargs
    ) -> Histogram:
        """
        Create and register a histogram.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            buckets: Custom bucket boundaries
            **kwargs: Additional arguments
            
        Returns:
            Histogram: Created histogram metric
        """
        histogram = Histogram(
            name=name,
            description=description,
            labels=labels,
            namespace=self.namespace,
            subsystem=self.subsystem,
            buckets=buckets,
            **kwargs
        )
        self.register(histogram)
        return histogram
    
    def create_summary(
        self,
        name: str,
        description: str,
        labels: Optional[MetricLabels] = None,
        quantiles: Optional[List[float]] = None,
        **kwargs
    ) -> Summary:
        """
        Create and register a summary.
        
        Args:
            name: Metric name
            description: Metric description
            labels: Metric labels
            quantiles: Quantiles to calculate
            **kwargs: Additional arguments
            
        Returns:
            Summary: Created summary metric
        """
        summary = Summary(
            name=name,
            description=description,
            labels=labels,
            namespace=self.namespace,
            subsystem=self.subsystem,
            quantiles=quantiles,
            **kwargs
        )
        self.register(summary)
        return summary
    
    def get_metric(self, name: str) -> Optional[BaseMetric]:
        """
        Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Optional[BaseMetric]: Metric if found, None otherwise
        """
        with self._lock:
            return self.metrics.get(name)
    
    def record_observation(self, metric_name: str, value: float, labels: Optional[MetricLabels] = None) -> None:
        """
        Record an observation for a metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
            labels: Labels for this observation
        """
        metric = self.get_metric(metric_name)
        if metric:
            metric.observe(value, labels)
            
            # Update statistics
            with self._lock:
                self.stats["observations_recorded"] += 1
                self.stats["last_observation_time"] = datetime.utcnow()
            
            # Update Prometheus if enabled
            if self.enable_prometheus and metric_name in self.prom_metrics:
                self._update_prometheus_metric(metric, value, labels)
    
    def _update_prometheus_metric(self, metric: BaseMetric, value: float, labels: Optional[MetricLabels] = None) -> None:
        """Update Prometheus metric."""
        try:
            prom_metric = self.prom_metrics.get(metric.name)
            if not prom_metric:
                return
            
            # Prepare labels
            label_dict = {}
            if labels:
                label_dict.update(labels.to_dict())
            if metric.labels:
                label_dict.update(metric.labels.to_dict())
            
            # Update based on metric type
            if isinstance(metric, Counter):
                prom_metric.labels(**label_dict).inc(value)
            elif isinstance(metric, Gauge):
                prom_metric.labels(**label_dict).set(value)
            elif isinstance(metric, Histogram):
                prom_metric.labels(**label_dict).observe(value)
            elif isinstance(metric, Summary):
                prom_metric.labels(**label_dict).observe(value)
                
        except Exception as e:
            logger.error(f"Failed to update Prometheus metric {metric.name}: {e}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all metrics with their current values.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of metric values
        """
        with self._lock:
            result = {}
            for name, metric in self.metrics.items():
                result[name] = {
                    "metadata": metric.get_metadata(),
                    "value": metric.get_value(),
                }
            return result
    
    def generate_prometheus_metrics(self) -> Optional[bytes]:
        """
        Generate Prometheus metrics exposition format.
        
        Returns:
            Optional[bytes]: Prometheus metrics in text format
        """
        if not self.enable_prometheus or not self.prom_registry:
            return None
        
        try:
            return generate_latest(self.prom_registry)
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get registry statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        with self._lock:
            stats = self.stats.copy()
            stats["total_metrics"] = len(self.metrics)
            stats["prometheus_enabled"] = self.enable_prometheus
            stats["prometheus_metrics"] = len(self.prom_metrics)
            
            # Calculate observation rate
            if stats["observations_recorded"] > 0 and stats["last_observation_time"]:
                time_diff = (datetime.utcnow() - stats["last_observation_time"]).total_seconds()
                if time_diff > 0:
                    stats["observations_per_second"] = stats["observations_recorded"] / time_diff
                else:
                    stats["observations_per_second"] = 0.0
            else:
                stats["observations_per_second"] = 0.0
            
            return stats
    
    def reset(self, metric_name: Optional[str] = None) -> None:
        """
        Reset metrics.
        
        Args:
            metric_name: Name of metric to reset (if None, reset all)
        """
        with self._lock:
            if metric_name:
                metric = self.metrics.get(metric_name)
                if metric and hasattr(metric, 'reset'):
                    metric.reset()
                    logger.info(f"Reset metric: {metric_name}")
            else:
                for name, metric in self.metrics.items():
                    if hasattr(metric, 'reset'):
                        metric.reset()
                logger.info("Reset all metrics")
    
    def cleanup(self) -> None:
        """Clean up metrics registry."""
        self.reset()
        logger.info("MetricsRegistry cleaned up")


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def setup_metrics(
    namespace: str = "worldbrief",
    subsystem: str = "app",
    enable_prometheus: bool = True,
    **kwargs
) -> MetricsRegistry:
    """
    Set up metrics collection.
    
    Args:
        namespace: Metric namespace
        subsystem: Metric subsystem
        enable_prometheus: Enable Prometheus integration
        **kwargs: Additional configuration
        
    Returns:
        MetricsRegistry: Configured metrics registry
    """
    global _metrics_registry
    
    if _metrics_registry is not None:
        logger.warning("Metrics already set up. Returning existing instance.")
        return _metrics_registry
    
    _metrics_registry = MetricsRegistry(
        namespace=namespace,
        subsystem=subsystem,
        enable_prometheus=enable_prometheus,
        **kwargs
    )
    
    # Register default metrics
    _register_default_metrics(_metrics_registry)
    
    return _metrics_registry


def _register_default_metrics(registry: MetricsRegistry) -> None:
    """Register default metrics."""
    # HTTP metrics
    registry.create_counter(
        name="http_requests_total",
        description="Total number of HTTP requests",
        labels=MetricLabels(labels={"method": "", "endpoint": "", "status": ""}),
    )
    
    registry.create_histogram(
        name="http_request_duration_seconds",
        description="HTTP request duration in seconds",
        labels=MetricLabels(labels={"method": "", "endpoint": ""}),
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    
    # Business metrics
    registry.create_counter(
        name="articles_processed_total",
        description="Total number of articles processed",
        labels=MetricLabels(labels={"source": "", "category": ""}),
    )
    
    registry.create_counter(
        name="briefings_generated_total",
        description="Total number of briefings generated",
        labels=MetricLabels(labels={"level": "", "topic": ""}),
    )
    
    # System metrics
    registry.create_gauge(
        name="memory_usage_bytes",
        description="Memory usage in bytes",
    )
    
    registry.create_gauge(
        name="cpu_usage_percent",
        description="CPU usage percentage",
    )
    
    registry.create_gauge(
        name="active_connections",
        description="Number of active database connections",
    )
    
    # Error metrics
    registry.create_counter(
        name="errors_total",
        description="Total number of errors",
        labels=MetricLabels(labels={"type": "", "component": ""}),
    )
    
    # Cache metrics
    registry.create_counter(
        name="cache_hits_total",
        description="Total number of cache hits",
        labels=MetricLabels(labels={"cache": ""}),
    )
    
    registry.create_counter(
        name="cache_misses_total",
        description="Total number of cache misses",
        labels=MetricLabels(labels={"cache": ""}),
    )
    
    registry.create_gauge(
        name="cache_size_bytes",
        description="Cache size in bytes",
        labels=MetricLabels(labels={"cache": ""}),
    )
    
    logger.info(f"Registered {len(registry.metrics)} default metrics")


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry.
    
    Returns:
        MetricsRegistry: Global metrics registry
        
    Raises:
        RuntimeError: If metrics are not set up
    """
    if _metrics_registry is None:
        raise RuntimeError("Metrics not set up. Call setup_metrics() first.")
    
    return _metrics_registry


def record_metrics(metric_name: str, value: float = 1.0, **label_kwargs) -> Callable:
    """
    Decorator to record metrics for function calls.
    
    Args:
        metric_name: Name of the metric to update
        value: Value to record (default: 1.0 for counters)
        **label_kwargs: Labels for the metric
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute function
            result = func(*args, **kwargs)
            
            # Record metric
            try:
                registry = get_metrics_registry()
                labels = MetricLabels(labels=label_kwargs)
                registry.record_observation(metric_name, value, labels)
            except Exception as e:
                logger.error(f"Failed to record metric {metric_name}: {e}")
            
            return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Execute async function
            result = await func(*args, **kwargs)
            
            # Record metric
            try:
                registry = get_metrics_registry()
                labels = MetricLabels(labels=label_kwargs)
                registry.record_observation(metric_name, value, labels)
            except Exception as e:
                logger.error(f"Failed to record metric {metric_name}: {e}")
            
            return result
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def time_metrics(metric_name: str, **label_kwargs) -> Callable:
    """
    Decorator to time function execution and record duration.
    
    Args:
        metric_name: Name of the histogram/summary metric
        **label_kwargs: Labels for the metric
        
    Returns:
        Callable: Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record duration
                try:
                    registry = get_metrics_registry()
                    labels = MetricLabels(labels=label_kwargs)
                    registry.record_observation(metric_name, duration, labels)
                except Exception as e:
                    logger.error(f"Failed to record timing metric {metric_name}: {e}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record duration even on error
                try:
                    registry = get_metrics_registry()
                    labels = MetricLabels(labels={**label_kwargs, "error": "true"})
                    registry.record_observation(metric_name, duration, labels)
                except Exception as e2:
                    logger.error(f"Failed to record error timing metric {metric_name}: {e2}")
                
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record duration
                try:
                    registry = get_metrics_registry()
                    labels = MetricLabels(labels=label_kwargs)
                    registry.record_observation(metric_name, duration, labels)
                except Exception as e:
                    logger.error(f"Failed to record timing metric {metric_name}: {e}")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record duration even on error
                try:
                    registry = get_metrics_registry()
                    labels = MetricLabels(labels={**label_kwargs, "error": "true"})
                    registry.record_observation(metric_name, duration, labels)
                except Exception as e2:
                    logger.error(f"Failed to record error timing metric {metric_name}: {e2}")
                
                raise
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Setup metrics
    registry = setup_metrics(
        namespace="test",
        subsystem="demo",
        enable_prometheus=False,
    )
    
    # Create custom metrics
    request_counter = registry.create_counter(
        name="demo_requests",
        description="Demo request counter",
        labels=MetricLabels(labels={"endpoint": "", "method": ""}),
    )
    
    response_time_histogram = registry.create_histogram(
        name="demo_response_time",
        description="Demo response time histogram",
        labels=MetricLabels(labels={"endpoint": ""}),
    )
    
    # Record metrics
    request_counter.inc(1, MetricLabels(labels={"endpoint": "/api/test", "method": "GET"}))
    
    with response_time_histogram.time(MetricLabels(labels={"endpoint": "/api/test"})):
        time.sleep(0.1)  # Simulate work
    
    # Use decorators
    @record_metrics("demo_function_calls", endpoint="/demo", method="POST")
    def demo_function():
        print("Demo function called")
        return "success"
    
    @time_metrics("demo_timed_function", endpoint="/timed")
    def timed_function():
        time.sleep(0.05)
        return "timed success"
    
    # Test functions
    demo_function()
    timed_function()
    
    # Get all metrics
    print("\nAll metrics:")
    all_metrics = registry.get_all_metrics()
    for name, data in all_metrics.items():
        print(f"{name}: {data['value']}")
    
    # Get statistics
    print("\nRegistry statistics:")
    stats = registry.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Cleanup
    registry.cleanup()
    
    print("\nMetrics test completed")