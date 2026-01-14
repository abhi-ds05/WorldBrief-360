"""
Performance Monitoring System

This module provides comprehensive performance monitoring with:

- Request timing and profiling
- Resource usage tracking (CPU, memory, disk, network)
- Database query performance monitoring
- Cache performance metrics
- API endpoint performance tracking
- Background job performance monitoring
- Performance bottleneck detection
- Performance trend analysis
- Automated performance reporting
"""

import logging
import logging
import time
import threading
import asyncio
import psutil
import gc
import tracemalloc
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import inspect
import functools
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
import statistics
import json
import hashlib
import socket
import resource

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class PerformanceMetric(Enum):
    """Performance metric types."""
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    DATABASE_QUERY_TIME = "database_query_time"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_LENGTH = "queue_length"
    GC_PRESSURE = "gc_pressure"
    
    @classmethod
    def from_string(cls, value: str) -> 'PerformanceMetric':
        """Convert string to PerformanceMetric."""
        try:
            return cls(value.lower())
        except ValueError:
            # Try case-insensitive match
            for metric in cls:
                if metric.value.lower() == value.lower():
                    return metric
            raise ValueError(f"Unknown performance metric: {value}")


@dataclass
class PerformanceThreshold:
    """Threshold configuration for performance alerts."""
    metric: PerformanceMetric
    component: str
    warning_threshold: float
    critical_threshold: float
    direction: str = "above"  # "above" or "below"
    window_size: int = 60  # Seconds to look back
    min_samples: int = 10  # Minimum samples needed
    
    def check(self, value: float) -> Tuple[str, Optional[str]]:
        """
        Check value against threshold.
        
        Returns:
            Tuple[str, Optional[str]]: (status, message)
            Status can be: "ok", "warning", "critical"
        """
        if self.direction == "above":
            if value >= self.critical_threshold:
                return "critical", f"{self.metric.value} ({value}) >= critical threshold ({self.critical_threshold})"
            elif value >= self.warning_threshold:
                return "warning", f"{self.metric.value} ({value}) >= warning threshold ({self.warning_threshold})"
        else:  # below
            if value <= self.critical_threshold:
                return "critical", f"{self.metric.value} ({value}) <= critical threshold ({self.critical_threshold})"
            elif value <= self.warning_threshold:
                return "warning", f"{self.metric.value} ({value}) <= warning threshold ({self.warning_threshold})"
        
        return "ok", None


@dataclass
class PerformanceSample:
    """A single performance measurement."""
    timestamp: datetime
    metric: PerformanceMetric
    component: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "metric": self.metric.value,
            "component": self.component,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "metadata": self.metadata,
        }


class PerformanceWindow:
    """
    Sliding window for performance samples.
    
    Maintains samples within a time window and provides
    statistical analysis on them.
    """
    
    def __init__(
        self,
        window_size: timedelta = timedelta(minutes=5),
        max_samples: int = 10000,
    ):
        """
        Initialize performance window.
        
        Args:
            window_size: Time window size
            max_samples: Maximum number of samples to keep
        """
        self.window_size = window_size
        self.max_samples = max_samples
        self.samples: deque[PerformanceSample] = deque(maxlen=max_samples)
        self._lock = threading.RLock()
    
    def add_sample(self, sample: PerformanceSample) -> None:
        """Add a sample to the window."""
        with self._lock:
            self.samples.append(sample)
            self._clean_old_samples()
    
    def _clean_old_samples(self) -> None:
        """Remove samples older than window size."""
        cutoff = datetime.utcnow() - self.window_size
        
        with self._lock:
            # Remove old samples from the beginning
            while self.samples and self.samples[0].timestamp < cutoff:
                self.samples.popleft()
    
    def get_samples(
        self,
        metric: Optional[PerformanceMetric] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[PerformanceSample]:
        """
        Get filtered samples.
        
        Args:
            metric: Filter by metric type
            component: Filter by component
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            List[PerformanceSample]: Filtered samples
        """
        with self._lock:
            filtered = list(self.samples)
            
            if metric:
                filtered = [s for s in filtered if s.metric == metric]
            
            if component:
                filtered = [s for s in filtered if s.component == component]
            
            if start_time:
                filtered = [s for s in filtered if s.timestamp >= start_time]
            
            if end_time:
                filtered = [s for s in filtered if s.timestamp <= end_time]
            
            return filtered
    
    def get_statistics(
        self,
        metric: PerformanceMetric,
        component: str,
        percentiles: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Get statistics for a metric and component.
        
        Args:
            metric: Metric type
            component: Component name
            percentiles: Percentiles to calculate
            
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        samples = self.get_samples(metric=metric, component=component)
        values = [s.value for s in samples]
        
        if not values:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "stddev": None,
                "percentiles": {},
            }
        
        if percentiles is None:
            percentiles = [0.5, 0.9, 0.95, 0.99]
        
        # Basic statistics
        stats = {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        
        # Calculate percentiles
        sorted_values = sorted(values)
        percentile_values = {}
        
        for p in percentiles:
            if sorted_values:
                idx = int(p * (len(sorted_values) - 1))
                percentile_values[f"p{p*100}"] = sorted_values[idx]
            else:
                percentile_values[f"p{p*100}"] = 0.0
        
        stats["percentiles"] = percentile_values
        
        # Add time range
        if samples:
            stats["start_time"] = min(s.timestamp for s in samples).isoformat()
            stats["end_time"] = max(s.timestamp for s in samples).isoformat()
        
        return stats
    
    def check_threshold(self, threshold: PerformanceThreshold) -> Tuple[str, Optional[str]]:
        """
        Check if threshold is violated.
        
        Args:
            threshold: Threshold to check
            
        Returns:
            Tuple[str, Optional[str]]: (status, message)
        """
        samples = self.get_samples(
            metric=threshold.metric,
            component=threshold.component,
            start_time=datetime.utcnow() - timedelta(seconds=threshold.window_size),
        )
        
        if len(samples) < threshold.min_samples:
            return "insufficient_data", f"Only {len(samples)} samples (need {threshold.min_samples})"
        
        values = [s.value for s in samples]
        avg_value = statistics.mean(values)
        
        return threshold.check(avg_value)
    
    def clear(self) -> None:
        """Clear all samples."""
        with self._lock:
            self.samples.clear()


class ResourceMonitor:
    """
    Monitors system resources (CPU, memory, disk, network).
    """
    
    def __init__(self, sampling_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            sampling_interval: Sampling interval in seconds
        """
        self.sampling_interval = sampling_interval
        self._running = False
        self._thread = None
        
        # Initial readings
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_used = 0
        self.memory_available = 0
        self.disk_usage = {}
        self.network_io = {}
        
        # Historical data
        self.cpu_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.memory_history = deque(maxlen=300)
        self._lock = threading.RLock()
    
    def start(self) -> None:
        """Start resource monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="ResourceMonitor",
            daemon=True,
        )
        self._thread.start()
        
        logger.info("Resource monitor started")
    
    def stop(self) -> None:
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.info("Resource monitor stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        # Initial network I/O readings
        net_io_start = psutil.net_io_counters()
        net_io_last = net_io_start
        last_time = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Memory usage
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk_usage = {}
                for partition in psutil.disk_partitions():
                    try:
                        usage = psutil.disk_usage(partition.mountpoint)
                        disk_usage[partition.mountpoint] = {
                            "total": usage.total,
                            "used": usage.used,
                            "free": usage.free,
                            "percent": usage.percent,
                        }
                    except Exception:
                        pass
                
                # Network I/O
                net_io_current = psutil.net_io_counters()
                time_diff = current_time - last_time
                
                if time_diff > 0:
                    network_io = {
                        "bytes_sent_rate": (net_io_current.bytes_sent - net_io_last.bytes_sent) / time_diff,
                        "bytes_recv_rate": (net_io_current.bytes_recv - net_io_last.bytes_recv) / time_diff,
                        "packets_sent_rate": (net_io_current.packets_sent - net_io_last.packets_sent) / time_diff,
                        "packets_recv_rate": (net_io_current.packets_recv - net_io_last.packets_recv) / time_diff,
                    }
                else:
                    network_io = {}
                
                net_io_last = net_io_current
                last_time = current_time
                
                # Update state
                with self._lock:
                    self.cpu_percent = cpu_percent
                    self.memory_percent = memory.percent
                    self.memory_used = memory.used
                    self.memory_available = memory.available
                    self.disk_usage = disk_usage
                    self.network_io = network_io
                    
                    # Update history
                    self.cpu_history.append((datetime.utcnow(), cpu_percent))
                    self.memory_history.append((datetime.utcnow(), memory.percent))
                
                # Sleep until next interval
                time_to_sleep = max(0, self.sampling_interval - (time.time() - current_time))
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                    
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.sampling_interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current resource statistics.
        
        Returns:
            Dict[str, Any]: Current resource statistics
        """
        with self._lock:
            return {
                "cpu_percent": self.cpu_percent,
                "memory_percent": self.memory_percent,
                "memory_used_bytes": self.memory_used,
                "memory_available_bytes": self.memory_available,
                "disk_usage": self.disk_usage,
                "network_io": self.network_io,
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    def get_history(
        self,
        metric: str = "cpu",
        duration: timedelta = timedelta(minutes=5),
    ) -> List[Tuple[datetime, float]]:
        """
        Get historical data.
        
        Args:
            metric: Metric name ("cpu" or "memory")
            duration: Duration to look back
            
        Returns:
            List[Tuple[datetime, float]]: Historical data points
        """
        cutoff = datetime.utcnow() - duration
        
        with self._lock:
            if metric == "cpu":
                history = self.cpu_history
            elif metric == "memory":
                history = self.memory_history
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            return [(ts, val) for ts, val in history if ts >= cutoff]
    
    def get_statistics(self, duration: timedelta = timedelta(minutes=5)) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Args:
            duration: Duration to analyze
            
        Returns:
            Dict[str, Any]: Resource statistics
        """
        cpu_data = self.get_history("cpu", duration)
        memory_data = self.get_history("memory", duration)
        
        cpu_values = [val for _, val in cpu_data] if cpu_data else []
        memory_values = [val for _, val in memory_data] if memory_data else []
        
        stats = {
            "cpu": {
                "current": self.cpu_percent,
                "samples": len(cpu_values),
                "min": min(cpu_values) if cpu_values else None,
                "max": max(cpu_values) if cpu_values else None,
                "mean": statistics.mean(cpu_values) if cpu_values else None,
                "p95": sorted(cpu_values)[int(0.95 * len(cpu_values))] if cpu_values else None,
            },
            "memory": {
                "current": self.memory_percent,
                "samples": len(memory_values),
                "min": min(memory_values) if memory_values else None,
                "max": max(memory_values) if memory_values else None,
                "mean": statistics.mean(memory_values) if memory_values else None,
                "p95": sorted(memory_values)[int(0.95 * len(memory_values))] if memory_values else None,
            },
            "duration_seconds": duration.total_seconds(),
        }
        
        return stats


class DatabasePerformanceMonitor:
    """
    Monitors database query performance.
    """
    
    def __init__(self):
        """Initialize database performance monitor."""
        self.queries: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Query fingerprinting
        self.query_fingerprints = {}
        
        # Slow query threshold (in seconds)
        self.slow_query_threshold = 1.0
        
        logger.info("Database performance monitor initialized")
    
    def record_query(
        self,
        query: str,
        duration: float,
        rows_returned: Optional[int] = None,
        connection_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Record a database query.
        
        Args:
            query: SQL query string
            duration: Query duration in seconds
            rows_returned: Number of rows returned
            connection_id: Database connection ID
            **kwargs: Additional metadata
        """
        # Create query fingerprint for grouping similar queries
        fingerprint = self._create_fingerprint(query)
        
        with self._lock:
            if fingerprint not in self.queries:
                self.queries[fingerprint] = {
                    "query_template": self._normalize_query(query),
                    "count": 0,
                    "total_duration": 0.0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                    "total_rows": 0,
                    "slow_count": 0,
                    "last_executed": None,
                    "samples": deque(maxlen=1000),  # Keep recent samples
                }
            
            query_stats = self.queries[fingerprint]
            query_stats["count"] += 1
            query_stats["total_duration"] += duration
            query_stats["min_duration"] = min(query_stats["min_duration"], duration)
            query_stats["max_duration"] = max(query_stats["max_duration"], duration)
            
            if rows_returned is not None:
                query_stats["total_rows"] += rows_returned
            
            if duration > self.slow_query_threshold:
                query_stats["slow_count"] += 1
            
            query_stats["last_executed"] = datetime.utcnow()
            
            # Add sample
            sample = {
                "timestamp": datetime.utcnow(),
                "duration": duration,
                "rows_returned": rows_returned,
                "connection_id": connection_id,
                **kwargs,
            }
            query_stats["samples"].append(sample)
    
    def _create_fingerprint(self, query: str) -> str:
        """Create fingerprint for query (group similar queries)."""
        # Normalize whitespace
        query = ' '.join(query.split())
        
        # Remove literals (numbers, strings)
        import re
        
        # Replace string literals
        query = re.sub(r"'.*?'", "?", query)
        
        # Replace number literals
        query = re.sub(r"\b\d+\b", "?", query)
        
        # Create hash
        return hashlib.md5(query.encode()).hexdigest()
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for display."""
        # Limit length
        if len(query) > 500:
            query = query[:497] + "..."
        
        return query
    
    def get_slow_queries(
        self,
        threshold: Optional[float] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get slow queries.
        
        Args:
            threshold: Slow query threshold in seconds
            limit: Maximum number of queries to return
            
        Returns:
            List[Dict[str, Any]]: Slow query statistics
        """
        if threshold is None:
            threshold = self.slow_query_threshold
        
        with self._lock:
            slow_queries = []
            
            for fingerprint, stats in self.queries.items():
                avg_duration = stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
                
                if avg_duration >= threshold:
                    query_info = {
                        "fingerprint": fingerprint,
                        "query_template": stats["query_template"],
                        "count": stats["count"],
                        "avg_duration": avg_duration,
                        "min_duration": stats["min_duration"],
                        "max_duration": stats["max_duration"],
                        "total_duration": stats["total_duration"],
                        "slow_count": stats["slow_count"],
                        "slow_percentage": (stats["slow_count"] / stats["count"]) * 100 if stats["count"] > 0 else 0,
                        "avg_rows": stats["total_rows"] / stats["count"] if stats["count"] > 0 else 0,
                        "last_executed": stats["last_executed"].isoformat() if stats["last_executed"] else None,
                    }
                    slow_queries.append(query_info)
            
            # Sort by average duration (descending)
            slow_queries.sort(key=lambda x: x["avg_duration"], reverse=True)
            
            return slow_queries[:limit]
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """
        Get overall query statistics.
        
        Returns:
            Dict[str, Any]: Query statistics
        """
        with self._lock:
            total_queries = 0
            total_duration = 0.0
            total_slow = 0
            
            for stats in self.queries.values():
                total_queries += stats["count"]
                total_duration += stats["total_duration"]
                total_slow += stats["slow_count"]
            
            avg_duration = total_duration / total_queries if total_queries > 0 else 0
            
            return {
                "unique_queries": len(self.queries),
                "total_queries": total_queries,
                "total_duration": total_duration,
                "avg_duration": avg_duration,
                "slow_queries": total_slow,
                "slow_percentage": (total_slow / total_queries) * 100 if total_queries > 0 else 0,
            }
    
    def clear(self) -> None:
        """Clear all query statistics."""
        with self._lock:
            self.queries.clear()
            logger.info("Cleared database performance statistics")


class CachePerformanceMonitor:
    """
    Monitors cache performance.
    """
    
    def __init__(self):
        """Initialize cache performance monitor."""
        self.caches: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        logger.info("Cache performance monitor initialized")
    
    def record_cache_op(
        self,
        cache_name: str,
        operation: str,
        hit: bool = True,
        key: Optional[str] = None,
        size_bytes: Optional[int] = None,
        duration: Optional[float] = None,
        **kwargs
    ) -> None:
        """
        Record a cache operation.
        
        Args:
            cache_name: Name of the cache
            operation: Operation type ("get", "set", "delete", "clear")
            hit: Whether it was a hit (for get operations)
            key: Cache key (optional)
            size_bytes: Size in bytes (for set operations)
            duration: Operation duration in seconds
            **kwargs: Additional metadata
        """
        with self._lock:
            if cache_name not in self.caches:
                self.caches[cache_name] = {
                    "gets": 0,
                    "hits": 0,
                    "sets": 0,
                    "deletes": 0,
                    "clears": 0,
                    "total_size_bytes": 0,
                    "max_size_bytes": 0,
                    "total_duration": 0.0,
                    "operation_counts": defaultdict(int),
                    "last_updated": None,
                }
            
            cache_stats = self.caches[cache_name]
            
            if operation == "get":
                cache_stats["gets"] += 1
                if hit:
                    cache_stats["hits"] += 1
            elif operation == "set":
                cache_stats["sets"] += 1
                if size_bytes:
                    cache_stats["total_size_bytes"] += size_bytes
                    cache_stats["max_size_bytes"] = max(
                        cache_stats["max_size_bytes"],
                        cache_stats["total_size_bytes"]
                    )
            elif operation == "delete":
                cache_stats["deletes"] += 1
            elif operation == "clear":
                cache_stats["clears"] += 1
                cache_stats["total_size_bytes"] = 0
            
            if duration:
                cache_stats["total_duration"] += duration
            
            cache_stats["operation_counts"][operation] += 1
            cache_stats["last_updated"] = datetime.utcnow()
    
    def get_cache_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            cache_name: Cache name (if None, get all)
            
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            if cache_name:
                if cache_name not in self.caches:
                    return {}
                
                stats = self.caches[cache_name].copy()
                
                # Calculate hit rate
                if stats["gets"] > 0:
                    stats["hit_rate"] = (stats["hits"] / stats["gets"]) * 100
                else:
                    stats["hit_rate"] = 0.0
                
                # Calculate average duration
                total_ops = sum(stats["operation_counts"].values())
                if total_ops > 0:
                    stats["avg_duration"] = stats["total_duration"] / total_ops
                else:
                    stats["avg_duration"] = 0.0
                
                return stats
            
            else:
                # Return all caches
                all_stats = {}
                for name, stats in self.caches.items():
                    all_stats[name] = self.get_cache_stats(name)
                return all_stats
    
    def get_hit_rates(self) -> Dict[str, float]:
        """
        Get hit rates for all caches.
        
        Returns:
            Dict[str, float]: Cache hit rates
        """
        hit_rates = {}
        
        with self._lock:
            for name, stats in self.caches.items():
                if stats["gets"] > 0:
                    hit_rates[name] = (stats["hits"] / stats["gets"]) * 100
                else:
                    hit_rates[name] = 0.0
        
        return hit_rates
    
    def clear_cache_stats(self, cache_name: Optional[str] = None) -> None:
        """
        Clear cache statistics.
        
        Args:
            cache_name: Cache name (if None, clear all)
        """
        with self._lock:
            if cache_name:
                if cache_name in self.caches:
                    self.caches[cache_name] = {
                        "gets": 0,
                        "hits": 0,
                        "sets": 0,
                        "deletes": 0,
                        "clears": 0,
                        "total_size_bytes": 0,
                        "max_size_bytes": 0,
                        "total_duration": 0.0,
                        "operation_counts": defaultdict(int),
                        "last_updated": None,
                    }
                    logger.info(f"Cleared statistics for cache: {cache_name}")
            else:
                self.caches.clear()
                logger.info("Cleared all cache statistics")


class GarbageCollectionMonitor:
    """
    Monitors garbage collection performance.
    """
    
    def __init__(self):
        """Initialize GC monitor."""
        self.gc_stats = {
            "collections": 0,
            "collected": 0,
            "uncollectable": 0,
            "total_time": 0.0,
            "last_collection": None,
        }
        self._lock = threading.RLock()
        
        # Enable GC debugging
        gc.set_debug(gc.DEBUG_STATS)
        
        # Set up GC callback
        gc.callbacks.append(self._gc_callback)
        
        logger.info("Garbage collection monitor initialized")
    
    def _gc_callback(self, phase: str, info: Dict[str, Any]) -> None:
        """GC callback function."""
        with self._lock:
            if phase == "start":
                # GC started
                pass
            elif phase == "stop":
                # GC completed
                self.gc_stats["collections"] += 1
                self.gc_stats["collected"] += info.get("collected", 0)
                self.gc_stats["uncollectable"] += info.get("uncollectable", 0)
                self.gc_stats["total_time"] += info.get("duration", 0.0)
                self.gc_stats["last_collection"] = datetime.utcnow()
    
    def get_gc_stats(self) -> Dict[str, Any]:
        """
        Get GC statistics.
        
        Returns:
            Dict[str, Any]: GC statistics
        """
        with self._lock:
            stats = self.gc_stats.copy()
            
            # Add current GC counts
            stats["objects"] = len(gc.get_objects())
            stats["garbage"] = len(gc.garbage)
            
            # Calculate collection rate
            if stats["collections"] > 0:
                stats["avg_collected"] = stats["collected"] / stats["collections"]
                stats["avg_time"] = stats["total_time"] / stats["collections"]
            else:
                stats["avg_collected"] = 0
                stats["avg_time"] = 0.0
            
            return stats
    
    def force_gc(self) -> Dict[str, Any]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Dict[str, Any]: GC statistics after forced collection
        """
        # Get before stats
        before_objects = len(gc.get_objects())
        before_garbage = len(gc.garbage)
        
        # Force collection
        collected = gc.collect()
        
        # Get after stats
        after_objects = len(gc.get_objects())
        after_garbage = len(gc.garbage)
        
        return {
            "collected": collected,
            "objects_before": before_objects,
            "objects_after": after_objects,
            "objects_freed": before_objects - after_objects,
            "garbage_before": before_garbage,
            "garbage_after": after_garbage,
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    def cleanup(self) -> None:
        """Clean up GC monitor."""
        # Remove callback
        gc.callbacks.remove(self._gc_callback)
        
        logger.info("Garbage collection monitor cleaned up")


class PerformanceMonitor:
    """
    Main performance monitoring orchestrator.
    """
    
    def __init__(
        self,
        service_name: str = "worldbrief-360",
        sampling_interval: float = 1.0,
        enable_resource_monitoring: bool = True,
        enable_database_monitoring: bool = True,
        enable_cache_monitoring: bool = True,
        enable_gc_monitoring: bool = True,
        **kwargs
    ):
        """
        Initialize performance monitor.
        
        Args:
            service_name: Name of the service
            sampling_interval: Resource sampling interval in seconds
            enable_resource_monitoring: Enable resource monitoring
            enable_database_monitoring: Enable database monitoring
            enable_cache_monitoring: Enable cache monitoring
            enable_gc_monitoring: Enable garbage collection monitoring
            **kwargs: Additional configuration
        """
        self.service_name = service_name
        
        # Performance windows
        self.windows = {
            "short": PerformanceWindow(window_size=timedelta(minutes=5)),
            "medium": PerformanceWindow(window_size=timedelta(minutes=30)),
            "long": PerformanceWindow(window_size=timedelta(hours=6)),
        }
        
        # Component monitors
        self.resource_monitor = None
        if enable_resource_monitoring:
            self.resource_monitor = ResourceMonitor(sampling_interval=sampling_interval)
        
        self.db_monitor = None
        if enable_database_monitoring:
            self.db_monitor = DatabasePerformanceMonitor()
        
        self.cache_monitor = None
        if enable_cache_monitoring:
            self.cache_monitor = CachePerformanceMonitor()
        
        self.gc_monitor = None
        if enable_gc_monitoring:
            self.gc_monitor = GarbageCollectionMonitor()
        
        # Performance thresholds
        self.thresholds: List[PerformanceThreshold] = []
        self._load_default_thresholds()
        
        # State
        self._running = False
        self._collector_thread = None
        
        # Statistics
        self.stats = {
            "samples_collected": 0,
            "threshold_violations": 0,
            "last_sample_time": None,
        }
        
        logger.info(f"PerformanceMonitor initialized for {service_name}")
    
    def _load_default_thresholds(self) -> None:
        """Load default performance thresholds."""
        self.thresholds = [
            # Response time thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.RESPONSE_TIME,
                component="api",
                warning_threshold=1.0,  # 1 second
                critical_threshold=3.0,  # 3 seconds
                direction="above",
            ),
            
            # Error rate thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.ERROR_RATE,
                component="api",
                warning_threshold=1.0,  # 1%
                critical_threshold=5.0,  # 5%
                direction="above",
            ),
            
            # CPU thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.CPU_USAGE,
                component="system",
                warning_threshold=80.0,  # 80%
                critical_threshold=95.0,  # 95%
                direction="above",
            ),
            
            # Memory thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.MEMORY_USAGE,
                component="system",
                warning_threshold=85.0,  # 85%
                critical_threshold=95.0,  # 95%
                direction="above",
            ),
            
            # Cache hit rate thresholds
            PerformanceThreshold(
                metric=PerformanceMetric.CACHE_HIT_RATE,
                component="redis",
                warning_threshold=90.0,  # 90%
                critical_threshold=80.0,  # 80%
                direction="below",
            ),
        ]
    
    def start(self) -> None:
        """Start performance monitoring."""
        if self._running:
            return
        
        self._running = True
        
        # Start resource monitor
        if self.resource_monitor:
            self.resource_monitor.start()
        
        # Start sample collector thread
        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            name="PerformanceCollector",
            daemon=True,
        )
        self._collector_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop(self) -> None:
        """Stop performance monitoring."""
        self._running = False
        
        # Stop resource monitor
        if self.resource_monitor:
            self.resource_monitor.stop()
        
        # Stop collector thread
        if self._collector_thread:
            self._collector_thread.join(timeout=2.0)
            self._collector_thread = None
        
        # Cleanup GC monitor
        if self.gc_monitor:
            self.gc_monitor.cleanup()
        
        logger.info("Performance monitoring stopped")
    
    def _collector_loop(self) -> None:
        """Collect performance samples periodically."""
        while self._running:
            try:
                self._collect_samples()
                time.sleep(5.0)  # Collect every 5 seconds
            except Exception as e:
                logger.error(f"Error in performance collector: {e}")
                time.sleep(5.0)
    
    def _collect_samples(self) -> None:
        """Collect performance samples from all monitors."""
        current_time = datetime.utcnow()
        
        # Collect resource samples
        if self.resource_monitor:
            resource_stats = self.resource_monitor.get_current_stats()
            
            # CPU
            self.record_sample(
                metric=PerformanceMetric.CPU_USAGE,
                component="system",
                value=resource_stats["cpu_percent"],
                unit="percent",
                tags={"type": "system"},
            )
            
            # Memory
            self.record_sample(
                metric=PerformanceMetric.MEMORY_USAGE,
                component="system",
                value=resource_stats["memory_percent"],
                unit="percent",
                tags={"type": "system"},
            )
            
            # Memory used
            self.record_sample(
                metric=PerformanceMetric.MEMORY_USAGE,
                component="system",
                value=resource_stats["memory_used_bytes"] / (1024**3),  # Convert to GB
                unit="gigabytes",
                tags={"type": "used"},
            )
        
        # Collect GC samples if available
        if self.gc_monitor:
            gc_stats = self.gc_monitor.get_gc_stats()
            
            # GC pressure (collections per minute)
            if gc_stats["last_collection"]:
                time_since_last = (current_time - gc_stats["last_collection"]).total_seconds()
                if time_since_last > 0:
                    collections_per_minute = 60 / time_since_last
                    self.record_sample(
                        metric=PerformanceMetric.GC_PRESSURE,
                        component="python",
                        value=collections_per_minute,
                        unit="collections_per_minute",
                    )
        
        self.stats["last_sample_time"] = current_time
        self.stats["samples_collected"] += 1
    
    def record_sample(
        self,
        metric: PerformanceMetric,
        component: str,
        value: float,
        unit: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a performance sample.
        
        Args:
            metric: Performance metric type
            component: Component name
            value: Metric value
            unit: Unit of measurement
            tags: Additional tags
            metadata: Additional metadata
        """
        sample = PerformanceSample(
            timestamp=datetime.utcnow(),
            metric=metric,
            component=component,
            value=value,
            unit=unit,
            tags=tags or {},
            metadata=metadata or {},
        )
        
        # Add to all windows
        for window in self.windows.values():
            window.add_sample(sample)
    
    def record_response_time(
        self,
        endpoint: str,
        method: str,
        duration: float,
        status_code: int,
        **kwargs
    ) -> None:
        """
        Record API response time.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            duration: Response duration in seconds
            status_code: HTTP status code
            **kwargs: Additional metadata
        """
        # Record response time
        self.record_sample(
            metric=PerformanceMetric.RESPONSE_TIME,
            component="api",
            value=duration * 1000,  # Convert to milliseconds
            unit="milliseconds",
            tags={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
            },
            metadata=kwargs,
        )
        
        # Record error if status >= 500
        if status_code >= 500:
            self.record_sample(
                metric=PerformanceMetric.ERROR_RATE,
                component="api",
                value=1.0,  # One error
                unit="count",
                tags={
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": str(status_code),
                },
            )
    
    def check_thresholds(self) -> List[Dict[str, Any]]:
        """
        Check all performance thresholds.
        
        Returns:
            List[Dict[str, Any]]: Threshold violations
        """
        violations = []
        
        for threshold in self.thresholds:
            for window_name, window in self.windows.items():
                status, message = window.check_threshold(threshold)
                
                if status in ["warning", "critical"]:
                    violation = {
                        "threshold": {
                            "metric": threshold.metric.value,
                            "component": threshold.component,
                            "warning_threshold": threshold.warning_threshold,
                            "critical_threshold": threshold.critical_threshold,
                            "direction": threshold.direction,
                        },
                        "status": status,
                        "message": message,
                        "window": window_name,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    violations.append(violation)
                    
                    # Update statistics
                    self.stats["threshold_violations"] += 1
        
        return violations
    
    def get_performance_report(
        self,
        duration: timedelta = timedelta(minutes=30),
        include_thresholds: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate performance report.
        
        Args:
            duration: Report duration
            include_thresholds: Include threshold violations
            
        Returns:
            Dict[str, Any]: Performance report
        """
        report = {
            "service": self.service_name,
            "generated_at": datetime.utcnow().isoformat(),
            "duration_seconds": duration.total_seconds(),
            "summary": {},
            "components": {},
        }
        
        # Get statistics for each metric and component
        window = self.windows["medium"]  # Use medium window
        
        # API performance
        api_stats = window.get_statistics(
            metric=PerformanceMetric.RESPONSE_TIME,
            component="api",
        )
        
        if api_stats["count"] > 0:
            report["components"]["api"] = {
                "response_time_ms": {
                    "p50": api_stats.get("percentiles", {}).get("p50", 0),
                    "p95": api_stats.get("percentiles", {}).get("p95", 0),
                    "p99": api_stats.get("percentiles", {}).get("p99", 0),
                    "avg": api_stats.get("mean", 0),
                    "min": api_stats.get("min", 0),
                    "max": api_stats.get("max", 0),
                    "requests": api_stats["count"],
                }
            }
        
        # System performance
        if self.resource_monitor:
            resource_stats = self.resource_monitor.get_statistics(duration)
            report["components"]["system"] = resource_stats
        
        # Database performance
        if self.db_monitor:
            db_stats = self.db_monitor.get_query_statistics()
            slow_queries = self.db_monitor.get_slow_queries(limit=5)
            
            report["components"]["database"] = {
                "query_statistics": db_stats,
                "slow_queries": slow_queries,
            }
        
        # Cache performance
        if self.cache_monitor:
            cache_stats = self.cache_monitor.get_cache_stats()
            hit_rates = self.cache_monitor.get_hit_rates()
            
            report["components"]["cache"] = {
                "stats": cache_stats,
                "hit_rates": hit_rates,
            }
        
        # Threshold violations
        if include_thresholds:
            violations = self.check_thresholds()
            report["threshold_violations"] = violations
        
        # Summary
        report["summary"] = {
            "samples_collected": self.stats["samples_collected"],
            "threshold_violations": self.stats["threshold_violations"],
            "last_sample_time": self.stats["last_sample_time"].isoformat() if self.stats["last_sample_time"] else None,
        }
        
        return report
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance monitor statistics.
        
        Returns:
            Dict[str, Any]: Statistics dictionary
        """
        stats = self.stats.copy()
        
        # Window statistics
        stats["windows"] = {}
        for name, window in self.windows.items():
            with window._lock:
                stats["windows"][name] = {
                    "samples": len(window.samples),
                    "window_size_seconds": window.window_size.total_seconds(),
                }
        
        # Component statistics
        stats["components"] = {
            "resource_monitor": self.resource_monitor is not None,
            "database_monitor": self.db_monitor is not None,
            "cache_monitor": self.cache_monitor is not None,
            "gc_monitor": self.gc_monitor is not None,
        }
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up performance monitor."""
        self.stop()
        
        # Clear windows
        for window in self.windows.values():
            window.clear()
        
        logger.info("PerformanceMonitor cleaned up")


# Performance monitoring decorators
def monitor_performance(
    metric: PerformanceMetric,
    component: str,
    unit: str = "seconds",
    tags: Optional[Dict[str, str]] = None,
):
    """
    Decorator to monitor function performance.
    
    Args:
        metric: Performance metric type
        component: Component name
        unit: Unit of measurement
        tags: Additional tags
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_sample(
                        metric=metric,
                        component=component,
                        value=duration,
                        unit=unit,
                        tags=tags or {},
                        metadata={
                            "function": func.__name__,
                            "module": func.__module__,
                        }
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_sample(
                        metric=PerformanceMetric.ERROR_RATE,
                        component=component,
                        value=1.0,
                        unit="count",
                        tags={**(tags or {}), "error": "true"},
                        metadata={
                            "function": func.__name__,
                            "module": func.__module__,
                            "error": str(e),
                        }
                    )
                
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_sample(
                        metric=metric,
                        component=component,
                        value=duration,
                        unit=unit,
                        tags=tags or {},
                        metadata={
                            "function": func.__name__,
                            "module": func.__module__,
                        }
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_sample(
                        metric=PerformanceMetric.ERROR_RATE,
                        component=component,
                        value=1.0,
                        unit="count",
                        tags={**(tags or {}), "error": "true"},
                        metadata={
                            "function": func.__name__,
                            "module": func.__module__,
                            "error": str(e),
                        }
                    )
                
                raise
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def monitor_api_performance(endpoint: str, method: str = "GET"):
    """
    Decorator to monitor API endpoint performance.
    
    Args:
        endpoint: API endpoint
        method: HTTP method
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Get FastAPI request from args if available
                request = None
                for arg in args:
                    if hasattr(arg, 'method') and hasattr(arg, 'url'):
                        request = arg
                        break
                
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Extract status code from result if possible
                status_code = 200
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                
                # Record performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_response_time(
                        endpoint=endpoint,
                        method=method,
                        duration=duration,
                        status_code=status_code,
                        request_path=endpoint,
                        request_method=method,
                    )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error performance
                monitor = get_performance_monitor()
                if monitor:
                    monitor.record_response_time(
                        endpoint=endpoint,
                        method=method,
                        duration=duration,
                        status_code=500,
                        request_path=endpoint,
                        request_method=method,
                        error=str(e),
                    )
                
                raise
        
        return wrapper
    
    return decorator


# Global performance monitor instance
_performance_monitor: Optional[PerformanceMonitor] = None


def setup_performance_monitoring(
    service_name: str = "worldbrief-360",
    sampling_interval: float = 1.0,
    **kwargs
) -> PerformanceMonitor:
    """
    Set up performance monitoring.
    
    Args:
        service_name: Name of the service
        sampling_interval: Resource sampling interval in seconds
        **kwargs: Additional configuration
        
    Returns:
        PerformanceMonitor: Configured performance monitor
    """
    global _performance_monitor
    
    if _performance_monitor is not None:
        logger.warning("Performance monitoring already set up. Returning existing instance.")
        return _performance_monitor
    
    _performance_monitor = PerformanceMonitor(
        service_name=service_name,
        sampling_interval=sampling_interval,
        **kwargs
    )
    
    # Start monitoring
    _performance_monitor.start()
    
    logger.info(f"Performance monitoring set up for {service_name}")
    
    return _performance_monitor


def get_performance_monitor() -> Optional[PerformanceMonitor]:
    """
    Get the global performance monitor instance.
    
    Returns:
        Optional[PerformanceMonitor]: Performance monitor instance
    """
    return _performance_monitor


def get_performance_report(
    duration: timedelta = timedelta(minutes=30),
    include_thresholds: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get performance report.
    
    Args:
        duration: Report duration
        include_thresholds: Include threshold violations
        
    Returns:
        Optional[Dict[str, Any]]: Performance report
    """
    monitor = get_performance_monitor()
    if not monitor:
        return None
    
    return monitor.get_performance_report(duration=duration, include_thresholds=include_thresholds)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Setup performance monitoring
    monitor = setup_performance_monitoring(
        service_name="test-service",
        sampling_interval=0.5,
        enable_resource_monitoring=True,
        enable_database_monitoring=True,
        enable_cache_monitoring=True,
        enable_gc_monitoring=True,
    )
    
    # Test performance recording
    @monitor_performance(
        metric=PerformanceMetric.RESPONSE_TIME,
        component="api",
        unit="seconds",
        tags={"endpoint": "/test", "method": "GET"}
    )
    def test_function():
        time.sleep(0.1)  # Simulate work
        return "success"
    
    # Test API monitoring
    @monitor_api_performance(endpoint="/api/test", method="GET")
    async def test_api_endpoint():
        await asyncio.sleep(0.05)  # Simulate async work
        return {"status": "ok"}
    
    # Run tests
    print("Testing performance monitoring...")
    
    # Run synchronous function
    result = test_function()
    print(f"Synchronous function result: {result}")
    
    # Run async function
    async def run_async():
        result = await test_api_endpoint()
        print(f"Async API result: {result}")
    
    asyncio.run(run_async())
    
    # Wait for some samples to be collected
    time.sleep(2)
    
    # Get performance report
    report = monitor.get_performance_report(duration=timedelta(minutes=1))
    print("\nPerformance report:")
    print(json.dumps(report, indent=2, default=str))
    
    # Get statistics
    stats = monitor.get_stats()
    print("\nPerformance monitor statistics:")
    print(json.dumps(stats, indent=2, default=str))
    
    # Check thresholds
    violations = monitor.check_thresholds()
    if violations:
        print("\nThreshold violations:")
        for violation in violations:
            print(f"  - {violation['status']}: {violation['message']}")
    
    # Cleanup
    monitor.cleanup()
    
    print("\nPerformance monitoring test completed")