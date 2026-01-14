"""
Model Monitoring System for tracking model performance, drift, and health.
Includes metrics collection, alerting, and performance analysis.
"""
import asyncio
import json
import logging
import math
import statistics
import time
import warnings
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Deque,
    Set, Iterator, ClassVar
)

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
from scipy import stats

from .base import ModelType # assuming ModelType is defined in base.py # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class MonitorType(Enum):
    """Types of monitoring metrics."""
    PERFORMANCE = "performance"
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    MEMORY_USAGE = "memory_usage"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class DriftDetectionMethod(Enum):
    """Statistical methods for drift detection."""
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    CHI_SQUARED = "chi_squared"
    PSI = "psi"  # Population Stability Index
    KL_DIVERGENCE = "kl_divergence"
    JENSEN_SHANNON = "jensen_shannon"
    ADAPTIVE_WINDOW = "adaptive_window"
    CUSUM = "cusum"  # Cumulative Sum


@dataclass
class MonitorConfig:
    """Configuration for monitoring a specific metric."""
    monitor_type: MonitorType
    model_id: str
    metric_name: str
    window_size: int = 1000  # Number of samples for sliding window
    sampling_rate: float = 1.0  # Sample every N requests
    thresholds: Dict[str, float] = field(default_factory=dict)
    drift_reference_window: int = 10000  # Reference window for drift detection
    check_interval_seconds: int = 60
    enabled: bool = True
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Collected metrics for a model."""
    model_id: str
    model_type: ModelType
    
    # Performance metrics
    inference_times: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    throughput_history: Deque[float] = field(default_factory=lambda: deque(maxlen=1440))  # 24 hours at 1-min intervals
    
    # Quality metrics
    success_count: int = 0
    error_count: int = 0
    prediction_counts: Dict[Any, int] = field(default_factory=dict)
    
    # Input/output tracking for drift detection
    input_features: Dict[str, Deque[float]] = field(default_factory=dict)
    output_distributions: Dict[str, Deque[float]] = field(default_factory=dict)
    
    # Resource metrics
    memory_usage_mb: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    gpu_utilization: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    
    # Timestamps
    last_prediction_time: Optional[datetime] = None
    first_seen_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        
        # Convert deque to list for serialization
        for key, value in result.items():
            if isinstance(value, deque):
                result[key] = list(value)
        
        return result


@dataclass
class DriftDetectionResult:
    """Result of drift detection analysis."""
    model_id: str
    feature_name: str
    drift_score: float
    drift_method: DriftDetectionMethod
    p_value: Optional[float] = None
    is_drift_detected: bool = False
    threshold: float = 0.1
    reference_size: int = 0
    current_size: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['drift_method'] = self.drift_method.value
        return result


@dataclass
class PerformanceAlert:
    """Alert for performance issues."""
    alert_id: str
    model_id: str
    alert_type: MonitorType
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['alert_type'] = self.alert_type.value
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        return result


class PerformanceThresholds(BaseModel):
    """Performance thresholds for monitoring."""
    # Latency thresholds (milliseconds)
    latency_warning_ms: float = 1000.0
    latency_error_ms: float = 5000.0
    latency_critical_ms: float = 10000.0
    
    # Throughput thresholds (requests per second)
    throughput_warning_rps: float = 10.0
    throughput_error_rps: float = 1.0
    
    # Error rate thresholds
    error_rate_warning: float = 0.05  # 5%
    error_rate_error: float = 0.10    # 10%
    error_rate_critical: float = 0.20  # 20%
    
    # Memory thresholds (MB)
    memory_warning_mb: float = 1024.0  # 1GB
    memory_error_mb: float = 2048.0    # 2GB
    
    # Drift thresholds
    drift_warning_score: float = 0.1
    drift_error_score: float = 0.2
    drift_critical_score: float = 0.3
    
    # Custom thresholds
    custom_thresholds: Dict[str, float] = Field(default_factory=dict)


class ModelMonitorConfig(BaseModel):
    """Configuration for the model monitoring system."""
    # Storage
    metrics_retention_days: int = 30
    alerts_retention_days: int = 90
    storage_backend: str = "memory"  # "memory", "sqlite", "postgresql"
    storage_path: Optional[Path] = None
    
    # Performance thresholds
    thresholds: PerformanceThresholds = Field(default_factory=PerformanceThresholds)
    
    # Drift detection
    drift_detection_enabled: bool = True
    drift_detection_method: str = "psi"
    drift_check_interval_hours: int = 1
    min_samples_for_drift: int = 100
    
    # Alerting
    alert_cooldown_seconds: int = 300  # 5 minutes
    max_alerts_per_hour: int = 10
    alert_channels: List[str] = Field(default_factory=list)  # "email", "slack", "webhook"
    
    # Monitoring frequency
    metrics_collection_interval_seconds: int = 60
    health_check_interval_seconds: int = 300  # 5 minutes
    
    # Resource limits
    max_metrics_per_model: int = 10000
    max_models_monitored: int = 100
    enable_auto_cleanup: bool = True
    
    # Feature tracking
    track_input_features: bool = True
    max_features_tracked: int = 50
    feature_sampling_rate: float = 0.1  # Sample 10% of requests for feature tracking
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        json_encoders = {Path: str}
    
    @validator('storage_path')
    def validate_storage_path(cls, v):
        """Ensure storage directory exists."""
        if v is not None:
            v = Path(v)
            v.mkdir(parents=True, exist_ok=True)
        return v


class MetricsStorage:
    """Abstract base class for metrics storage backends."""
    
    def save_metrics(self, model_id: str, metrics: ModelMetrics) -> None:
        """Save model metrics."""
        raise NotImplementedError
    
    def load_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """Load model metrics."""
        raise NotImplementedError
    
    def save_alert(self, alert: PerformanceAlert) -> None:
        """Save performance alert."""
        raise NotImplementedError
    
    def load_alerts(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceAlert]:
        """Load performance alerts."""
        raise NotImplementedError
    
    def save_drift_result(self, result: DriftDetectionResult) -> None:
        """Save drift detection result."""
        raise NotImplementedError
    
    def load_drift_results(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[DriftDetectionResult]:
        """Load drift detection results."""
        raise NotImplementedError
    
    def cleanup_old_data(self, retention_days: int) -> int:
        """Cleanup old data."""
        raise NotImplementedError


class InMemoryStorage(MetricsStorage):
    """In-memory storage backend for development/testing."""
    
    def __init__(self):
        self._metrics: Dict[str, ModelMetrics] = {}
        self._alerts: List[PerformanceAlert] = []
        self._drift_results: List[DriftDetectionResult] = []
        self._lock = Lock()
    
    def save_metrics(self, model_id: str, metrics: ModelMetrics) -> None:
        with self._lock:
            self._metrics[model_id] = metrics
    
    def load_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        with self._lock:
            return self._metrics.get(model_id)
    
    def save_alert(self, alert: PerformanceAlert) -> None:
        with self._lock:
            self._alerts.append(alert)
    
    def load_alerts(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[PerformanceAlert]:
        with self._lock:
            alerts = self._alerts.copy()
            
            if model_id:
                alerts = [a for a in alerts if a.model_id == model_id]
            
            if start_time:
                alerts = [a for a in alerts if a.timestamp >= start_time]
            
            if end_time:
                alerts = [a for a in alerts if a.timestamp <= end_time]
            
            return alerts
    
    def save_drift_result(self, result: DriftDetectionResult) -> None:
        with self._lock:
            self._drift_results.append(result)
    
    def load_drift_results(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[DriftDetectionResult]:
        with self._lock:
            results = self._drift_results.copy()
            
            if model_id:
                results = [r for r in results if r.model_id == model_id]
            
            if start_time:
                results = [r for r in results if r.timestamp >= start_time]
            
            return results
    
    def cleanup_old_data(self, retention_days: int) -> int:
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        with self._lock:
            # Clean old alerts
            initial_count = len(self._alerts)
            self._alerts = [
                a for a in self._alerts if a.timestamp >= cutoff_time
            ]
            alerts_removed = initial_count - len(self._alerts)
            
            # Clean old drift results
            initial_count = len(self._drift_results)
            self._drift_results = [
                r for r in self._drift_results if r.timestamp >= cutoff_time
            ]
            drift_removed = initial_count - len(self._drift_results)
            
            return alerts_removed + drift_removed


class ModelMonitor:
    """
    Comprehensive model monitoring system.
    
    Features:
    - Real-time performance monitoring
    - Statistical drift detection
    - Automated alerting
    - Metrics aggregation and visualization
    - Health checks and reporting
    """
    
    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], ModelMonitorConfig]] = None
    ):
        """
        Initialize the model monitor.
        
        Args:
            config: Configuration dictionary or ModelMonitorConfig instance
        """
        # Parse configuration
        if config is None:
            config = ModelMonitorConfig()
        elif isinstance(config, dict):
            config = ModelMonitorConfig(**config)
        
        self.config = config
        
        # Initialize storage
        self._storage = self._create_storage()
        
        # Active monitoring
        self._monitored_models: Set[str] = set()
        self._monitor_configs: Dict[str, List[MonitorConfig]] = {}
        self._active_alerts: Dict[str, List[PerformanceAlert]] = {}
        
        # Statistics
        self._stats = {
            'total_predictions': 0,
            'total_errors': 0,
            'total_alerts': 0,
            'total_drift_checks': 0,
            'start_time': datetime.now()
        }
        
        # Alert cooldown tracking
        self._alert_cooldowns: Dict[Tuple[str, str], datetime] = {}
        
        # Background threads
        self._monitor_thread: Optional[Thread] = None
        self._cleanup_thread: Optional[Thread] = None
        self._should_stop = False
        
        # Locks
        self._models_lock = Lock()
        self._alerts_lock = Lock()
        
        # Start background threads
        self._start_background_threads()
        
        logger.info(f"ModelMonitor initialized with config: {config.dict()}")
    
    def _create_storage(self) -> MetricsStorage:
        """Create appropriate storage backend."""
        backend = self.config.storage_backend.lower()
        
        if backend == "memory":
            return InMemoryStorage()
        elif backend == "sqlite":
            try:
                from .storage.sqlite_storage import SQLiteStorage # type: ignore
                return SQLiteStorage(self.config.storage_path)
            except ImportError:
                logger.warning("SQLiteStorage not available, falling back to memory")
                return InMemoryStorage()
        elif backend == "postgresql":
            try:
                from .storage.postgres_storage import PostgreSQLStorage # type: ignore
                return PostgreSQLStorage()
            except ImportError:
                logger.warning("PostgreSQLStorage not available, falling back to memory")
                return InMemoryStorage()
        else:
            logger.warning(f"Unknown storage backend: {backend}, using memory")
            return InMemoryStorage()
    
    def _start_background_threads(self) -> None:
        """Start background monitoring and cleanup threads."""
        # Monitor thread
        self._monitor_thread = Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ModelMonitor-Main"
        )
        self._monitor_thread.start()
        
        # Cleanup thread
        self._cleanup_thread = Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="ModelMonitor-Cleanup"
        )
        self._cleanup_thread.start()
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while not self._should_stop:
            try:
                # Check each monitored model
                with self._models_lock:
                    model_ids = list(self._monitored_models)
                
                for model_id in model_ids:
                    self._check_model_performance(model_id)
                    
                    # Check for drift (less frequently)
                    if (self.config.drift_detection_enabled and 
                        datetime.now().hour % self.config.drift_check_interval_hours == 0):
                        self._check_model_drift(model_id)
                
                # Sleep until next check
                time.sleep(self.config.metrics_collection_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _cleanup_loop(self) -> None:
        """Cleanup loop for old data."""
        while not self._should_stop:
            try:
                # Sleep for 1 hour
                time.sleep(3600)
                
                # Cleanup old data
                removed_count = self._storage.cleanup_old_data(
                    self.config.metrics_retention_days
                )
                
                if removed_count > 0:
                    logger.info(f"Cleaned up {removed_count} old data entries")
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def register_model(
        self,
        model_id: str,
        model_type: Union[str, ModelType],
        monitor_configs: Optional[List[MonitorConfig]] = None
    ) -> bool:
        """
        Register a model for monitoring.
        
        Args:
            model_id: Model identifier
            model_type: Type of model
            monitor_configs: Optional custom monitor configurations
            
        Returns:
            True if registered successfully
        """
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        with self._models_lock:
            if model_id in self._monitored_models:
                logger.warning(f"Model {model_id} is already being monitored")
                return False
            
            # Create default metrics if not exist
            metrics = self._storage.load_metrics(model_id)
            if metrics is None:
                metrics = ModelMetrics(
                    model_id=model_id,
                    model_type=model_type,
                    first_seen_time=datetime.now()
                )
                self._storage.save_metrics(model_id, metrics)
            
            # Set up monitoring configurations
            if monitor_configs is None:
                monitor_configs = self._create_default_monitors(model_id, model_type)
            
            self._monitor_configs[model_id] = monitor_configs
            self._monitored_models.add(model_id)
            
            logger.info(f"Registered model for monitoring: {model_id} ({model_type.value})")
            return True
    
    def _create_default_monitors(
        self,
        model_id: str,
        model_type: ModelType
    ) -> List[MonitorConfig]:
        """Create default monitor configurations for a model."""
        monitors = [
            MonitorConfig(
                monitor_type=MonitorType.PERFORMANCE,
                model_id=model_id,
                metric_name="latency",
                thresholds={
                    "warning": self.config.thresholds.latency_warning_ms,
                    "error": self.config.thresholds.latency_error_ms,
                    "critical": self.config.thresholds.latency_critical_ms
                },
                check_interval_seconds=60
            ),
            MonitorConfig(
                monitor_type=MonitorType.ERROR_RATE,
                model_id=model_id,
                metric_name="error_rate",
                thresholds={
                    "warning": self.config.thresholds.error_rate_warning,
                    "error": self.config.thresholds.error_rate_error,
                    "critical": self.config.thresholds.error_rate_critical
                },
                check_interval_seconds=300
            ),
            MonitorConfig(
                monitor_type=MonitorType.THROUGHPUT,
                model_id=model_id,
                metric_name="throughput",
                thresholds={
                    "warning": self.config.thresholds.throughput_warning_rps,
                    "error": self.config.thresholds.throughput_error_rps
                },
                check_interval_seconds=60
            )
        ]
        
        # Add drift monitoring for certain model types
        if model_type in [
            ModelType.TEXT_GENERATION,
            ModelType.EMBEDDING,
            ModelType.IMAGE_TO_TEXT
        ]:
            monitors.append(
                MonitorConfig(
                    monitor_type=MonitorType.DATA_DRIFT,
                    model_id=model_id,
                    metric_name="input_distribution",
                    thresholds={
                        "warning": self.config.thresholds.drift_warning_score,
                        "error": self.config.thresholds.drift_error_score,
                        "critical": self.config.thresholds.drift_critical_score
                    },
                    check_interval_seconds=3600,  # Check hourly
                    drift_reference_window=10000
                )
            )
        
        return monitors
    
    def unregister_model(self, model_id: str) -> bool:
        """
        Unregister a model from monitoring.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if unregistered successfully
        """
        with self._models_lock:
            if model_id not in self._monitored_models:
                return False
            
            self._monitored_models.remove(model_id)
            self._monitor_configs.pop(model_id, None)
            
            logger.info(f"Unregistered model from monitoring: {model_id}")
            return True
    
    def record_prediction(
        self,
        model_id: str,
        inference_time_ms: float,
        success: bool = True,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record a model prediction for monitoring.
        
        Args:
            model_id: Model identifier
            inference_time_ms: Inference time in milliseconds
            success: Whether prediction was successful
            input_data: Optional input features for drift detection
            output_data: Optional output for drift detection
            error_message: Optional error message if failed
        """
        with self._models_lock:
            # Load or create metrics
            metrics = self._storage.load_metrics(model_id)
            if metrics is None:
                # Auto-register if not monitored
                logger.info(f"Auto-registering model for monitoring: {model_id}")
                self.register_model(model_id, ModelType.TEXT_GENERATION)
                metrics = self._storage.load_metrics(model_id)
            
            if metrics is None:
                logger.warning(f"Could not record prediction for unmonitored model: {model_id}")
                return
            
            # Update statistics
            self._stats['total_predictions'] += 1
            
            if success:
                metrics.success_count += 1
                metrics.inference_times.append(inference_time_ms)
            else:
                metrics.error_count += 1
                self._stats['total_errors'] += 1
            
            # Update throughput (requests per minute)
            current_time = datetime.now()
            if metrics.last_prediction_time:
                time_diff = (current_time - metrics.last_prediction_time).total_seconds()
                if time_diff > 0:
                    throughput = 60.0 / time_diff  # Requests per minute
                    metrics.throughput_history.append(throughput)
            
            metrics.last_prediction_time = current_time
            
            # Track input features for drift detection (sampled)
            if (self.config.track_input_features and 
                input_data is not None and 
                np.random.random() < self.config.feature_sampling_rate):
                
                for feature_name, feature_value in input_data.items():
                    if feature_name not in metrics.input_features:
                        if len(metrics.input_features) >= self.config.max_features_tracked:
                            # Remove oldest feature if limit reached
                            oldest_feature = next(iter(metrics.input_features))
                            metrics.input_features.pop(oldest_feature)
                        
                        metrics.input_features[feature_name] = deque(maxlen=1000)
                    
                    # Convert to float if possible
                    try:
                        if isinstance(feature_value, (int, float)):
                            metrics.input_features[feature_name].append(float(feature_value))
                        elif isinstance(feature_value, list) and len(feature_value) > 0:
                            # Take first element for simplicity
                            val = feature_value[0]
                            if isinstance(val, (int, float)):
                                metrics.input_features[feature_name].append(float(val))
                    except (ValueError, TypeError):
                        pass
            
            # Track output distribution (for classification models)
            if output_data is not None and success:
                if isinstance(output_data, dict) and 'prediction' in output_data:
                    prediction = output_data['prediction']
                    metrics.prediction_counts[prediction] = \
                        metrics.prediction_counts.get(prediction, 0) + 1
            
            # Save updated metrics
            self._storage.save_metrics(model_id, metrics)
    
    def _check_model_performance(self, model_id: str) -> None:
        """
        Check performance metrics for a model and generate alerts if needed.
        
        Args:
            model_id: Model identifier
        """
        metrics = self._storage.load_metrics(model_id)
        if metrics is None:
            return
        
        # Get monitor configurations for this model
        monitor_configs = self._monitor_configs.get(model_id, [])
        
        for config in monitor_configs:
            if not config.enabled:
                continue
            
            try:
                if config.monitor_type == MonitorType.PERFORMANCE:
                    self._check_latency(model_id, metrics, config)
                elif config.monitor_type == MonitorType.ERROR_RATE:
                    self._check_error_rate(model_id, metrics, config)
                elif config.monitor_type == MonitorType.THROUGHPUT:
                    self._check_throughput(model_id, metrics, config)
                elif config.monitor_type == MonitorType.DATA_DRIFT:
                    # Drift is checked separately
                    pass
                elif config.monitor_type == MonitorType.CUSTOM:
                    self._check_custom_metric(model_id, metrics, config)
                    
            except Exception as e:
                logger.error(f"Error checking {config.metric_name} for {model_id}: {e}")
    
    def _check_latency(
        self,
        model_id: str,
        metrics: ModelMetrics,
        config: MonitorConfig
    ) -> None:
        """Check latency metrics."""
        if len(metrics.inference_times) < 10:
            return  # Not enough data
        
        # Calculate percentiles
        times = list(metrics.inference_times)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        thresholds = config.thresholds
        
        # Check against thresholds
        if p99 > thresholds.get("critical", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.PERFORMANCE,
                metric_name="latency_p99",
                metric_value=p99,
                threshold=thresholds.get("critical"),
                severity=AlertSeverity.CRITICAL,
                message=f"Critical latency: P99={p99:.1f}ms exceeds {thresholds.get('critical')}ms"
            )
        elif p99 > thresholds.get("error", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.PERFORMANCE,
                metric_name="latency_p99",
                metric_value=p99,
                threshold=thresholds.get("error"),
                severity=AlertSeverity.ERROR,
                message=f"High latency: P99={p99:.1f}ms exceeds {thresholds.get('error')}ms"
            )
        elif p95 > thresholds.get("warning", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.PERFORMANCE,
                metric_name="latency_p95",
                metric_value=p95,
                threshold=thresholds.get("warning"),
                severity=AlertSeverity.WARNING,
                message=f"Elevated latency: P95={p95:.1f}ms exceeds {thresholds.get('warning')}ms"
            )
    
    def _check_error_rate(
        self,
        model_id: str,
        metrics: ModelMetrics,
        config: MonitorConfig
    ) -> None:
        """Check error rate metrics."""
        total_predictions = metrics.success_count + metrics.error_count
        if total_predictions < 100:
            return  # Not enough data
        
        error_rate = metrics.error_count / total_predictions
        thresholds = config.thresholds
        
        if error_rate > thresholds.get("critical", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.ERROR_RATE,
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=thresholds.get("critical"),
                severity=AlertSeverity.CRITICAL,
                message=f"Critical error rate: {error_rate:.1%} exceeds {thresholds.get('critical'):.1%}"
            )
        elif error_rate > thresholds.get("error", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.ERROR_RATE,
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=thresholds.get("error"),
                severity=AlertSeverity.ERROR,
                message=f"High error rate: {error_rate:.1%} exceeds {thresholds.get('error'):.1%}"
            )
        elif error_rate > thresholds.get("warning", float('inf')):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.ERROR_RATE,
                metric_name="error_rate",
                metric_value=error_rate,
                threshold=thresholds.get("warning"),
                severity=AlertSeverity.WARNING,
                message=f"Elevated error rate: {error_rate:.1%} exceeds {thresholds.get('warning'):.1%}"
            )
    
    def _check_throughput(
        self,
        model_id: str,
        metrics: ModelMetrics,
        config: MonitorConfig
    ) -> None:
        """Check throughput metrics."""
        if len(metrics.throughput_history) < 5:
            return  # Not enough data
        
        # Calculate average throughput (requests per minute)
        throughput = statistics.mean(list(metrics.throughput_history)[-5:])  # Last 5 minutes
        thresholds = config.thresholds
        
        if throughput < thresholds.get("error", 0):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.THROUGHPUT,
                metric_name="throughput",
                metric_value=throughput,
                threshold=thresholds.get("error"),
                severity=AlertSeverity.ERROR,
                message=f"Low throughput: {throughput:.1f} RPM below {thresholds.get('error'):.1f} RPM"
            )
        elif throughput < thresholds.get("warning", 0):
            self._create_alert(
                model_id=model_id,
                alert_type=MonitorType.THROUGHPUT,
                metric_name="throughput",
                metric_value=throughput,
                threshold=thresholds.get("warning"),
                severity=AlertSeverity.WARNING,
                message=f"Reduced throughput: {throughput:.1f} RPM below {thresholds.get('warning'):.1f} RPM"
            )
    
    def _check_custom_metric(
        self,
        model_id: str,
        metrics: ModelMetrics,
        config: MonitorConfig
    ) -> None:
        """Check custom metric defined in configuration."""
        # This is a placeholder for custom metric checking
        # In practice, you would implement specific logic based on config.custom_config
        pass
    
    def _check_model_drift(self, model_id: str) -> None:
        """
        Check for data drift in model inputs/outputs.
        
        Args:
            model_id: Model identifier
        """
        metrics = self._storage.load_metrics(model_id)
        if metrics is None:
            return
        
        # Check if we have enough data
        total_features = sum(len(deque) for deque in metrics.input_features.values())
        if total_features < self.config.min_samples_for_drift:
            return
        
        self._stats['total_drift_checks'] += 1
        
        # Check each tracked feature for drift
        for feature_name, feature_values in metrics.input_features.items():
            if len(feature_values) < 100:
                continue  # Not enough samples
            
            # Convert to numpy array
            current_data = np.array(list(feature_values))
            
            # Get reference data (older half of samples)
            split_idx = len(current_data) // 2
            reference_data = current_data[:split_idx]
            test_data = current_data[split_idx:]
            
            if len(reference_data) < 50 or len(test_data) < 50:
                continue
            
            # Calculate drift
            drift_result = self._calculate_drift(
                reference_data,
                test_data,
                feature_name,
                model_id
            )
            
            # Save drift result
            self._storage.save_drift_result(drift_result)
            
            # Create alert if drift detected
            if drift_result.is_drift_detected:
                severity = self._determine_drift_severity(drift_result.drift_score)
                
                self._create_alert(
                    model_id=model_id,
                    alert_type=MonitorType.DATA_DRIFT,
                    metric_name=f"drift_{feature_name}",
                    metric_value=drift_result.drift_score,
                    threshold=drift_result.threshold,
                    severity=severity,
                    message=f"Data drift detected for feature '{feature_name}': "
                           f"score={drift_result.drift_score:.3f} (threshold={drift_result.threshold})"
                )
    
    def _calculate_drift(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str,
        model_id: str
    ) -> DriftDetectionResult:
        """
        Calculate drift between reference and test data.
        
        Args:
            reference_data: Reference data distribution
            test_data: Current data distribution
            feature_name: Name of the feature being tested
            model_id: Model identifier
            
        Returns:
            Drift detection result
        """
        method_str = self.config.drift_detection_method.lower()
        
        try:
            if method_str == DriftDetectionMethod.KOLMOGOROV_SMIRNOV.value:
                return self._calculate_ks_drift(
                    reference_data, test_data, feature_name, model_id
                )
            elif method_str == DriftDetectionMethod.PSI.value:
                return self._calculate_psi_drift(
                    reference_data, test_data, feature_name, model_id
                )
            elif method_str == DriftDetectionMethod.KL_DIVERGENCE.value:
                return self._calculate_kl_drift(
                    reference_data, test_data, feature_name, model_id
                )
            elif method_str == DriftDetectionMethod.JENSEN_SHANNON.value:
                return self._calculate_js_drift(
                    reference_data, test_data, feature_name, model_id
                )
            else:
                # Default to PSI
                return self._calculate_psi_drift(
                    reference_data, test_data, feature_name, model_id
                )
                
        except Exception as e:
            logger.error(f"Error calculating drift for {model_id}.{feature_name}: {e}")
            
            # Return a default result on error
            return DriftDetectionResult(
                model_id=model_id,
                feature_name=feature_name,
                drift_score=0.0,
                drift_method=DriftDetectionMethod(method_str),
                is_drift_detected=False,
                threshold=self.config.thresholds.drift_warning_score,
                reference_size=len(reference_data),
                current_size=len(test_data)
            )
    
    def _calculate_ks_drift(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str,
        model_id: str
    ) -> DriftDetectionResult:
        """Calculate drift using Kolmogorov-Smirnov test."""
        statistic, p_value = stats.ks_2samp(reference_data, test_data)
        
        threshold = self.config.thresholds.drift_warning_score
        is_drift_detected = statistic > threshold
        
        return DriftDetectionResult(
            model_id=model_id,
            feature_name=feature_name,
            drift_score=float(statistic),
            drift_method=DriftDetectionMethod.KOLMOGOROV_SMIRNOV,
            p_value=float(p_value),
            is_drift_detected=is_drift_detected,
            threshold=threshold,
            reference_size=len(reference_data),
            current_size=len(test_data)
        )
    
    def _calculate_psi_drift(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str,
        model_id: str
    ) -> DriftDetectionResult:
        """Calculate drift using Population Stability Index (PSI)."""
        # Create bins based on reference data
        min_val = min(reference_data.min(), test_data.min())
        max_val = max(reference_data.max(), test_data.max())
        
        # Use deciles for binning
        bins = np.percentile(reference_data, np.arange(0, 101, 10))
        bins = np.unique(bins)  # Remove duplicates
        
        # Ensure bins cover the range
        if len(bins) < 2:
            bins = np.array([min_val, max_val])
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference_data, bins=bins)
        test_hist, _ = np.histogram(test_data, bins=bins)
        
        # Add small epsilon to avoid division by zero
        ref_hist = ref_hist.astype(float) + 1e-10
        test_hist = test_hist.astype(float) + 1e-10
        
        # Normalize to probabilities
        ref_probs = ref_hist / ref_hist.sum()
        test_probs = test_hist / test_hist.sum()
        
        # Calculate PSI
        psi = np.sum((test_probs - ref_probs) * np.log(test_probs / ref_probs))
        
        threshold = self.config.thresholds.drift_warning_score
        is_drift_detected = psi > threshold
        
        return DriftDetectionResult(
            model_id=model_id,
            feature_name=feature_name,
            drift_score=float(psi),
            drift_method=DriftDetectionMethod.PSI,
            is_drift_detected=is_drift_detected,
            threshold=threshold,
            reference_size=len(reference_data),
            current_size=len(test_data)
        )
    
    def _calculate_kl_drift(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str,
        model_id: str
    ) -> DriftDetectionResult:
        """Calculate drift using Kullback-Leibler divergence."""
        # Create probability distributions using kernel density estimation
        ref_kde = stats.gaussian_kde(reference_data)
        test_kde = stats.gaussian_kde(test_data)
        
        # Sample points for integration
        min_val = min(reference_data.min(), test_data.min())
        max_val = max(reference_data.max(), test_data.max())
        x = np.linspace(min_val, max_val, 1000)
        
        # Calculate PDFs
        ref_pdf = ref_kde(x)
        test_pdf = test_kde(x)
        
        # Add small epsilon to avoid log(0)
        ref_pdf = ref_pdf + 1e-10
        test_pdf = test_pdf + 1e-10
        
        # Normalize
        ref_pdf = ref_pdf / ref_pdf.sum()
        test_pdf = test_pdf / test_pdf.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(test_pdf * np.log(test_pdf / ref_pdf))
        
        threshold = self.config.thresholds.drift_warning_score
        is_drift_detected = kl_div > threshold
        
        return DriftDetectionResult(
            model_id=model_id,
            feature_name=feature_name,
            drift_score=float(kl_div),
            drift_method=DriftDetectionMethod.KL_DIVERGENCE,
            is_drift_detected=is_drift_detected,
            threshold=threshold,
            reference_size=len(reference_data),
            current_size=len(test_data)
        )
    
    def _calculate_js_drift(
        self,
        reference_data: np.ndarray,
        test_data: np.ndarray,
        feature_name: str,
        model_id: str
    ) -> DriftDetectionResult:
        """Calculate drift using Jensen-Shannon divergence."""
        # Create probability distributions
        ref_kde = stats.gaussian_kde(reference_data)
        test_kde = stats.gaussian_kde(test_data)
        
        # Sample points
        min_val = min(reference_data.min(), test_data.min())
        max_val = max(reference_data.max(), test_data.max())
        x = np.linspace(min_val, max_val, 1000)
        
        # Calculate PDFs
        ref_pdf = ref_kde(x)
        test_pdf = test_kde(x)
        
        # Add epsilon and normalize
        ref_pdf = (ref_pdf + 1e-10) / (ref_pdf.sum() + 1e-10)
        test_pdf = (test_pdf + 1e-10) / (test_pdf.sum() + 1e-10)
        
        # Calculate JS divergence
        m = 0.5 * (ref_pdf + test_pdf)
        js_div = 0.5 * (stats.entropy(ref_pdf, m) + stats.entropy(test_pdf, m))
        
        threshold = self.config.thresholds.drift_warning_score
        is_drift_detected = js_div > threshold
        
        return DriftDetectionResult(
            model_id=model_id,
            feature_name=feature_name,
            drift_score=float(js_div),
            drift_method=DriftDetectionMethod.JENSEN_SHANNON,
            is_drift_detected=is_drift_detected,
            threshold=threshold,
            reference_size=len(reference_data),
            current_size=len(test_data)
        )
    
    def _determine_drift_severity(self, drift_score: float) -> AlertSeverity:
        """Determine alert severity based on drift score."""
        if drift_score > self.config.thresholds.drift_critical_score:
            return AlertSeverity.CRITICAL
        elif drift_score > self.config.thresholds.drift_error_score:
            return AlertSeverity.ERROR
        elif drift_score > self.config.thresholds.drift_warning_score:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    def _create_alert(
        self,
        model_id: str,
        alert_type: MonitorType,
        metric_name: str,
        metric_value: float,
        threshold: float,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Create a performance alert.
        
        Args:
            model_id: Model identifier
            alert_type: Type of alert
            metric_name: Name of the metric
            metric_value: Current metric value
            threshold: Threshold that was exceeded
            severity: Alert severity
            message: Alert message
            metadata: Additional metadata
        """
        # Check cooldown
        alert_key = (model_id, f"{alert_type.value}_{metric_name}")
        last_alert_time = self._alert_cooldowns.get(alert_key)
        
        if last_alert_time:
            time_since_last = (datetime.now() - last_alert_time).total_seconds()
            if time_since_last < self.config.alert_cooldown_seconds:
                return  # Still in cooldown
        
        # Create alert
        alert_id = f"{model_id}_{alert_type.value}_{datetime.now().timestamp()}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            model_id=model_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            metadata=metadata or {}
        )
        
        # Save alert
        self._storage.save_alert(alert)
        self._stats['total_alerts'] += 1
        
        # Update cooldown
        self._alert_cooldowns[alert_key] = datetime.now()
        
        # Trigger alert actions
        self._trigger_alert_actions(alert)
        
        logger.warning(f"Alert created: {message}")
    
    def _trigger_alert_actions(self, alert: PerformanceAlert) -> None:
        """Trigger actions for an alert (email, webhook, etc.)."""
        # This is a placeholder for alert action implementations
        # In practice, you would integrate with email services, Slack, webhooks, etc.
        
        for channel in self.config.alert_channels:
            try:
                if channel == "email":
                    self._send_email_alert(alert)
                elif channel == "slack":
                    self._send_slack_alert(alert)
                elif channel == "webhook":
                    self._send_webhook_alert(alert)
                elif channel == "log":
                    # Already logged above
                    pass
                    
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
    
    def _send_email_alert(self, alert: PerformanceAlert) -> None:
        """Send alert via email."""
        # Implementation would depend on your email service
        pass
    
    def _send_slack_alert(self, alert: PerformanceAlert) -> None:
        """Send alert to Slack."""
        # Implementation would depend on your Slack integration
        pass
    
    def _send_webhook_alert(self, alert: PerformanceAlert) -> None:
        """Send alert via webhook."""
        # Implementation would depend on your webhook configuration
        pass
    
    def get_model_metrics(self, model_id: str) -> Optional[ModelMetrics]:
        """
        Get metrics for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metrics or None
        """
        return self._storage.load_metrics(model_id)
    
    def get_alerts(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None
    ) -> List[PerformanceAlert]:
        """
        Get alerts with optional filtering.
        
        Args:
            model_id: Filter by model identifier
            start_time: Filter alerts after this time
            end_time: Filter alerts before this time
            severity: Filter by severity
            status: Filter by status
            
        Returns:
            List of matching alerts
        """
        alerts = self._storage.load_alerts(model_id, start_time, end_time)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts
    
    def get_drift_results(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None
    ) -> List[DriftDetectionResult]:
        """
        Get drift detection results.
        
        Args:
            model_id: Filter by model identifier
            start_time: Filter results after this time
            
        Returns:
            List of drift detection results
        """
        return self._storage.load_drift_results(model_id, start_time)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if acknowledged, False if not found
        """
        # This would require updating storage to modify alerts
        # For simplicity, this is a placeholder
        logger.info(f"Alert acknowledged: {alert_id}")
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.
        
        Args:
            alert_id: Alert identifier
            
        Returns:
            True if resolved, False if not found
        """
        # This would require updating storage to modify alerts
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        with self._models_lock:
            stats = self._stats.copy()
            stats.update({
                'monitored_models': len(self._monitored_models),
                'uptime_seconds': (datetime.now() - stats['start_time']).total_seconds(),
                'prediction_rate_per_minute': self._calculate_prediction_rate(),
            })
            return stats
    
    def _calculate_prediction_rate(self) -> float:
        """Calculate current prediction rate."""
        # Simplified calculation
        uptime = (datetime.now() - self._stats['start_time']).total_seconds()
        if uptime == 0:
            return 0.0
        
        return self._stats['total_predictions'] / (uptime / 60)
    
    def generate_report(
        self,
        model_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate a monitoring report.
        
        Args:
            model_id: Optional specific model
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Report dictionary
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=1)
        if end_time is None:
            end_time = datetime.now()
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'time_range': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {},
            'models': []
        }
        
        # Get models to report on
        if model_id:
            model_ids = [model_id]
        else:
            with self._models_lock:
                model_ids = list(self._monitored_models)
        
        for mid in model_ids:
            metrics = self._storage.load_metrics(mid)
            if not metrics:
                continue
            
            # Calculate metrics
            total_predictions = metrics.success_count + metrics.error_count
            error_rate = metrics.error_count / max(total_predictions, 1)
            
            # Get recent alerts
            alerts = self.get_alerts(
                model_id=mid,
                start_time=start_time,
                end_time=end_time
            )
            
            # Get drift results
            drift_results = self.get_drift_results(
                model_id=mid,
                start_time=start_time
            )
            
            model_report = {
                'model_id': mid,
                'model_type': metrics.model_type.value,
                'total_predictions': total_predictions,
                'success_count': metrics.success_count,
                'error_count': metrics.error_count,
                'error_rate': error_rate,
                'recent_alerts': len(alerts),
                'drift_checks': len(drift_results),
                'drift_detected': sum(1 for d in drift_results if d.is_drift_detected),
                'first_seen': metrics.first_seen_time.isoformat() if metrics.first_seen_time else None,
                'last_prediction': metrics.last_prediction_time.isoformat() if metrics.last_prediction_time else None
            }
            
            report['models'].append(model_report)
        
        # Update summary
        if report['models']:
            total_models = len(report['models'])
            total_predictions = sum(m['total_predictions'] for m in report['models'])
            total_errors = sum(m['error_count'] for m in report['models'])
            total_alerts = sum(m['recent_alerts'] for m in report['models'])
            
            report['summary'] = {
                'total_models': total_models,
                'total_predictions': total_predictions,
                'total_errors': total_errors,
                'overall_error_rate': total_errors / max(total_predictions, 1),
                'total_alerts': total_alerts,
                'models_with_alerts': sum(1 for m in report['models'] if m['recent_alerts'] > 0)
            }
        
        return report
    
    def shutdown(self) -> None:
        """Shutdown the model monitor."""
        logger.info("Shutting down ModelMonitor")
        self._should_stop = True
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        logger.info("ModelMonitor shutdown complete")


# Singleton instance for global access
_global_monitor: Optional[ModelMonitor] = None


def get_model_monitor(
    config: Optional[Union[Dict[str, Any], ModelMonitorConfig]] = None
) -> ModelMonitor:
    """
    Get or create global model monitor instance.
    
    Args:
        config: Configuration for the model monitor
        
    Returns:
        Global ModelMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = ModelMonitor(config)
    
    return _global_monitor


def shutdown_model_monitor() -> None:
    """Shutdown the global model monitor."""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.shutdown()
        _global_monitor = None