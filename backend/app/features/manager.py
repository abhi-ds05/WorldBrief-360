"""
Feature flag manager.
Singleton manager for feature flag operations including CRUD, evaluation, and monitoring.
"""

from dataclasses import dataclass, field
import json
import logging
from typing import Dict, Any, Optional, List, Union, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading
import asyncio
from collections import defaultdict
import hashlib

from .flags import FeatureFlag, FlagType, FeatureFlags, create_flag_from_dict
from .context import FeatureContext, UserContext, RequestContext
from .evaluator import FeatureEvaluator, create_evaluator, SegmentationRule, Segment
from .backends.base import FeatureBackend
from .backends.in_memory import InMemoryBackend
from .schemas import (
    FeatureFlagResponse, 
    VariantAssignment,
    FeatureFlagUpdate,
    FeatureFlagCreate,
    FeatureFlagList
)

logger = logging.getLogger(__name__)


class UpdateStrategy(str, Enum):
    """Strategies for updating feature flags."""
    POLLING = "polling"           # Periodic polling
    WEBHOOK = "webhook"           # Webhook-based updates
    STREAMING = "streaming"       # Real-time streaming
    MANUAL = "manual"             # Manual updates only


class CacheStrategy(str, Enum):
    """Caching strategies for feature flags."""
    NO_CACHE = "no_cache"         # No caching
    MEMORY = "memory"             # In-memory cache
    REDIS = "redis"               # Redis cache
    LOCAL_STORAGE = "local_storage" # Browser local storage


@dataclass
class FlagUpdate:
    """Represents a feature flag update."""
    
    flag_name: str
    old_value: Optional[FeatureFlag]
    new_value: Optional[FeatureFlag]
    timestamp: datetime
    source: str  # "manual", "api", "webhook", "polling"
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_name": self.flag_name,
            "old_value": self.old_value.to_dict() if self.old_value else None,
            "new_value": self.new_value.to_dict() if self.new_value else None,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "user_id": self.user_id,
        }


@dataclass
class EvaluationMetrics:
    """Metrics for feature flag evaluations."""
    
    flag_name: str
    total_evaluations: int = 0
    enabled_evaluations: int = 0
    disabled_evaluations: int = 0
    variant_counts: Dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    last_evaluated: Optional[datetime] = None
    
    def record_evaluation(self, enabled: bool, variant: Any = None):
        """Record an evaluation."""
        self.total_evaluations += 1
        if enabled:
            self.enabled_evaluations += 1
        else:
            self.disabled_evaluations += 1
        
        if variant is not None:
            variant_key = str(variant)
            self.variant_counts[variant_key] = self.variant_counts.get(variant_key, 0) + 1
        
        self.last_evaluated = datetime.utcnow()
    
    def record_error(self):
        """Record an evaluation error."""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "flag_name": self.flag_name,
            "total_evaluations": self.total_evaluations,
            "enabled_evaluations": self.enabled_evaluations,
            "disabled_evaluations": self.disabled_evaluations,
            "enabled_percentage": self.enabled_evaluations / max(self.total_evaluations, 1),
            "variant_counts": self.variant_counts,
            "error_count": self.error_count,
            "error_percentage": self.error_count / max(self.total_evaluations, 1),
            "last_evaluated": self.last_evaluated.isoformat() if self.last_evaluated else None,
        }


class FeatureManager:
    """
    Singleton manager for feature flag operations.
    
    Provides:
    - Feature flag CRUD operations
    - Flag evaluation with context
    - Metrics collection
    - Update strategies
    - Cache management
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FeatureManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        backend: Optional[FeatureBackend] = None,
        update_strategy: UpdateStrategy = UpdateStrategy.POLLING,
        polling_interval: int = 30,  # seconds
        cache_ttl: int = 300,  # seconds
    ):
        """Initialize feature manager (only once)."""
        if self._initialized:
            return
        
        self.backend = backend or InMemoryBackend()
        self.update_strategy = update_strategy
        self.polling_interval = polling_interval
        self.cache_ttl = cache_ttl
        
        # Evaluation
        self.evaluator = create_evaluator()
        
        # Caching
        self._flag_cache: Dict[str, FeatureFlag] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Metrics
        self.metrics: Dict[str, EvaluationMetrics] = defaultdict(lambda: EvaluationMetrics(flag_name=""))
        self.updates: List[FlagUpdate] = []
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Locks
        self._cache_lock = threading.RLock()
        self._update_lock = threading.RLock()
        
        # Background tasks
        self._polling_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Initialize with default flags
        self._load_default_flags()
        
        self._initialized = True
        logger.info("FeatureManager initialized")
    
    def _load_default_flags(self):
        """Load default feature flags."""
        default_flags = FeatureFlags.get_all_flags()
        for flag_name, flag in default_flags.items():
            if not self.backend.get(flag_name):
                self.backend.set(flag_name, flag)
                self._flag_cache[flag_name] = flag
                self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
                logger.debug(f"Loaded default flag: {flag_name}")
    
    # ==================== FLAG MANAGEMENT ====================
    
    def get_flag(self, flag_name: str, force_refresh: bool = False) -> Optional[FeatureFlag]:
        """
        Get feature flag by name.
        
        Args:
            flag_name: Name of the flag
            force_refresh: Bypass cache and fetch from backend
        
        Returns:
            FeatureFlag or None if not found
        """
        with self._cache_lock:
            # Check cache first
            if not force_refresh and flag_name in self._flag_cache:
                cache_time = self._cache_timestamps.get(flag_name)
                if cache_time and (datetime.utcnow() - cache_time).seconds < self.cache_ttl:
                    return self._flag_cache[flag_name]
            
            # Fetch from backend
            flag = self.backend.get(flag_name)
            if flag:
                self._flag_cache[flag_name] = flag
                self._cache_timestamps[flag_name] = datetime.utcnow()
            
            # Initialize metrics if needed
            if flag_name not in self.metrics:
                self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
            
            return flag
    
    def get_all_flags(self, include_disabled: bool = False) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.
        
        Args:
            include_disabled: Include disabled flags
        
        Returns:
            Dictionary of flag name -> FeatureFlag
        """
        flags = self.backend.get_all()
        
        # Update cache
        with self._cache_lock:
            for flag_name, flag in flags.items():
                self._flag_cache[flag_name] = flag
                self._cache_timestamps[flag_name] = datetime.utcnow()
                
                # Initialize metrics if needed
                if flag_name not in self.metrics:
                    self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
        
        if not include_disabled:
            flags = {name: flag for name, flag in flags.items() if flag.enabled}
        
        return flags
    
    def create_flag(
        self,
        flag_name: str,
        description: str,
        flag_type: FlagType = FlagType.BOOLEAN,
        enabled: bool = False,
        variants: Optional[Dict[str, Any]] = None,
        default_variant: Any = None,
        rollout_percentage: float = 0.0,
        target_users: Optional[List[str]] = None,
        target_segments: Optional[List[str]] = None,
        environments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> FeatureFlag:
        """
        Create a new feature flag.
        
        Args:
            flag_name: Unique name for the flag
            description: Human-readable description
            flag_type: Type of flag
            enabled: Whether flag is enabled
            variants: Available variants for multivariate flags
            default_variant: Default variant value
            rollout_percentage: Rollout percentage (0.0 to 1.0)
            target_users: Specific user IDs to target
            target_segments: User segments to target
            environments: Environments where flag is available
            metadata: Additional metadata
            user_id: User creating the flag
        
        Returns:
            Created FeatureFlag
        
        Raises:
            ValueError: If flag already exists
        """
        if self.get_flag(flag_name):
            raise ValueError(f"Feature flag '{flag_name}' already exists")
        
        flag = FeatureFlag(
            name=flag_name,
            description=description,
            flag_type=flag_type,
            enabled=enabled,
            variants=variants or {},
            default_variant=default_variant,
            rollout_percentage=rollout_percentage,
            target_users=target_users or [],
            target_segments=target_segments or [],
            environments=environments or ["development", "staging", "production"],
            metadata=metadata or {},
        )
        
        # Save to backend
        self.backend.set(flag_name, flag)
        
        # Update cache
        with self._cache_lock:
            self._flag_cache[flag_name] = flag
            self._cache_timestamps[flag_name] = datetime.utcnow()
        
        # Initialize metrics
        self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
        
        # Record update
        update = FlagUpdate(
            flag_name=flag_name,
            old_value=None,
            new_value=flag,
            timestamp=datetime.utcnow(),
            source="manual",
            user_id=user_id,
        )
        self.updates.append(update)
        
        logger.info(f"Created feature flag: {flag_name}")
        return flag
    
    def update_flag(
        self,
        flag_name: str,
        enabled: Optional[bool] = None,
        rollout_percentage: Optional[float] = None,
        target_users: Optional[List[str]] = None,
        target_segments: Optional[List[str]] = None,
        variants: Optional[Dict[str, Any]] = None,
        default_variant: Optional[Any] = None,
        environments: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Optional[FeatureFlag]:
        """
        Update an existing feature flag.
        
        Args:
            flag_name: Name of flag to update
            enabled: New enabled state
            rollout_percentage: New rollout percentage
            target_users: New target users
            target_segments: New target segments
            variants: New variants
            default_variant: New default variant
            environments: New environments
            metadata: New metadata
            user_id: User updating the flag
        
        Returns:
            Updated FeatureFlag or None if not found
        """
        old_flag = self.get_flag(flag_name)
        if not old_flag:
            logger.warning(f"Cannot update non-existent flag: {flag_name}")
            return None
        
        # Create updated flag
        new_flag = FeatureFlag(
            name=flag_name,
            description=old_flag.description,
            flag_type=old_flag.flag_type,
            enabled=enabled if enabled is not None else old_flag.enabled,
            variants=variants if variants is not None else old_flag.variants,
            default_variant=default_variant if default_variant is not None else old_flag.default_variant,
            rollout_percentage=rollout_percentage if rollout_percentage is not None else old_flag.rollout_percentage,
            target_users=target_users if target_users is not None else old_flag.target_users,
            target_segments=target_segments if target_segments is not None else old_flag.target_segments,
            environments=environments if environments is not None else old_flag.environments,
            metadata=metadata if metadata is not None else old_flag.metadata,
        )
        
        # Save to backend
        self.backend.set(flag_name, new_flag)
        
        # Update cache
        with self._cache_lock:
            self._flag_cache[flag_name] = new_flag
            self._cache_timestamps[flag_name] = datetime.utcnow()
        
        # Record update
        update = FlagUpdate(
            flag_name=flag_name,
            old_value=old_flag,
            new_value=new_flag,
            timestamp=datetime.utcnow(),
            source="manual",
            user_id=user_id,
        )
        self.updates.append(update)
        
        logger.info(f"Updated feature flag: {flag_name}")
        return new_flag
    
    def delete_flag(self, flag_name: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a feature flag.
        
        Args:
            flag_name: Name of flag to delete
            user_id: User deleting the flag
        
        Returns:
            True if deleted, False if not found
        """
        old_flag = self.get_flag(flag_name)
        if not old_flag:
            return False
        
        # Delete from backend
        self.backend.delete(flag_name)
        
        # Remove from cache
        with self._cache_lock:
            if flag_name in self._flag_cache:
                del self._flag_cache[flag_name]
            if flag_name in self._cache_timestamps:
                del self._cache_timestamps[flag_name]
        
        # Record update
        update = FlagUpdate(
            flag_name=flag_name,
            old_value=old_flag,
            new_value=None,
            timestamp=datetime.utcnow(),
            source="manual",
            user_id=user_id,
        )
        self.updates.append(update)
        
        # Remove metrics
        if flag_name in self.metrics:
            del self.metrics[flag_name]
        
        logger.info(f"Deleted feature flag: {flag_name}")
        return True
    
    def toggle_flag(self, flag_name: str, user_id: Optional[str] = None) -> Optional[FeatureFlag]:
        """
        Toggle a feature flag's enabled state.
        
        Args:
            flag_name: Name of flag to toggle
            user_id: User toggling the flag
        
        Returns:
            Updated FeatureFlag or None if not found
        """
        flag = self.get_flag(flag_name)
        if not flag:
            return None
        
        return self.update_flag(
            flag_name=flag_name,
            enabled=not flag.enabled,
            user_id=user_id,
        )
    
    # ==================== EVALUATION ====================
    
    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production",
    ) -> bool:
        """
        Check if a feature flag is enabled for a user.
        
        Args:
            flag_name: Name of feature flag
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            True if enabled, False otherwise
        """
        flag = self.get_flag(flag_name)
        if not flag:
            logger.debug(f"Flag not found: {flag_name}")
            return False
        
        try:
            result = self.evaluator.is_enabled(flag, user_id, context, environment)
            
            # Record metrics
            self._record_evaluation(flag_name, result, None)
            
            return result
        except Exception as e:
            logger.error(f"Error evaluating flag {flag_name}: {e}")
            self._record_error(flag_name)
            return False
    
    def get_variant(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production",
    ) -> Any:
        """
        Get variant value for a feature flag.
        
        Args:
            flag_name: Name of feature flag
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            Variant value or default if flag is disabled
        """
        flag = self.get_flag(flag_name)
        if not flag:
            logger.debug(f"Flag not found: {flag_name}")
            return None
        
        try:
            variant = self.evaluator.get_variant(flag, user_id, context, environment)
            
            # Determine if flag is enabled based on variant
            is_enabled = variant != flag.default_variant if flag.default_variant is not None else bool(variant)
            
            # Record metrics
            self._record_evaluation(flag_name, is_enabled, variant)
            
            return variant
        except Exception as e:
            logger.error(f"Error getting variant for flag {flag_name}: {e}")
            self._record_error(flag_name)
            return flag.default_variant
    
    def evaluate(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production",
    ) -> FeatureFlagResponse:
        """
        Evaluate a feature flag and return detailed response.
        
        Args:
            flag_name: Name of feature flag
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            FeatureFlagResponse with evaluation details
        """
        flag = self.get_flag(flag_name)
        if not flag:
            return FeatureFlagResponse(
                flag_name=flag_name,
                enabled=False,
                variant=None,
                reason="flag_not_found",
                error=True,
                error_message=f"Flag '{flag_name}' not found",
            )
        
        try:
            # Get evaluation result
            result = self.evaluator.evaluate(flag, user_id, context, environment)
            
            # Record metrics
            self._record_evaluation(flag_name, result["enabled"], result.get("variant"))
            
            # Record history
            self.evaluation_history.append({
                "flag_name": flag_name,
                "user_id": user_id,
                "enabled": result["enabled"],
                "variant": result.get("variant"),
                "timestamp": datetime.utcnow().isoformat(),
                "environment": environment,
            })
            
            # Trim history if too large
            if len(self.evaluation_history) > 10000:
                self.evaluation_history = self.evaluation_history[-5000:]
            
            return FeatureFlagResponse(
                flag_name=flag_name,
                enabled=result["enabled"],
                variant=result.get("variant"),
                reason=result.get("reason", "unknown"),
                flag_type=flag.flag_type.value,
                environment=environment,
                user_id=user_id,
                segments=result.get("segments", []),
                timestamp=result.get("timestamp"),
            )
        except Exception as e:
            logger.error(f"Error evaluating flag {flag_name}: {e}")
            self._record_error(flag_name)
            
            return FeatureFlagResponse(
                flag_name=flag_name,
                enabled=False,
                variant=flag.default_variant,
                reason="evaluation_error",
                error=True,
                error_message=str(e),
            )
    
    def evaluate_many(
        self,
        flag_names: List[str],
        user_id: Optional[str] = None,
        context: Optional[Union[Dict, FeatureContext]] = None,
        environment: str = "production",
    ) -> Dict[str, FeatureFlagResponse]:
        """
        Evaluate multiple feature flags at once.
        
        Args:
            flag_names: List of flag names to evaluate
            user_id: Optional user identifier
            context: Evaluation context
            environment: Current environment
        
        Returns:
            Dictionary of flag name -> FeatureFlagResponse
        """
        results = {}
        for flag_name in flag_names:
            results[flag_name] = self.evaluate(flag_name, user_id, context, environment)
        return results
    
    def _record_evaluation(self, flag_name: str, enabled: bool, variant: Any = None):
        """Record evaluation metrics."""
        if flag_name not in self.metrics:
            self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
        
        self.metrics[flag_name].record_evaluation(enabled, variant)
    
    def _record_error(self, flag_name: str):
        """Record evaluation error."""
        if flag_name not in self.metrics:
            self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
        
        self.metrics[flag_name].record_error()
    
    # ==================== SEGMENT MANAGEMENT ====================
    
    def add_segment(self, segment: Segment):
        """Add a user segment for targeting."""
        self.evaluator.add_segment(segment)
    
    def remove_segment(self, segment_name: str):
        """Remove a user segment."""
        self.evaluator.remove_segment(segment_name)
    
    def get_segments(self) -> Dict[str, Segment]:
        """Get all user segments."""
        return self.evaluator.segments
    
    # ==================== CACHE MANAGEMENT ====================
    
    def clear_cache(self, flag_name: Optional[str] = None):
        """
        Clear feature flag cache.
        
        Args:
            flag_name: Optional specific flag to clear, or all if None
        """
        with self._cache_lock:
            if flag_name:
                if flag_name in self._flag_cache:
                    del self._flag_cache[flag_name]
                if flag_name in self._cache_timestamps:
                    del self._cache_timestamps[flag_name]
                logger.debug(f"Cleared cache for flag: {flag_name}")
            else:
                self._flag_cache.clear()
                self._cache_timestamps.clear()
                logger.debug("Cleared all feature flag cache")
    
    def refresh_cache(self, flag_names: Optional[List[str]] = None):
        """
        Refresh cache from backend.
        
        Args:
            flag_names: Optional list of specific flags to refresh
        """
        if flag_names:
            for flag_name in flag_names:
                self.get_flag(flag_name, force_refresh=True)
        else:
            self.get_all_flags(include_disabled=True)
    
    # ==================== METRICS & MONITORING ====================
    
    def get_metrics(self, flag_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get evaluation metrics.
        
        Args:
            flag_name: Optional specific flag metrics
        
        Returns:
            Metrics dictionary
        """
        if flag_name:
            if flag_name in self.metrics:
                return self.metrics[flag_name].to_dict()
            return {}
        
        return {name: metrics.to_dict() for name, metrics in self.metrics.items()}
    
    def get_recent_updates(self, limit: int = 100) -> List[FlagUpdate]:
        """
        Get recent flag updates.
        
        Args:
            limit: Maximum number of updates to return
        
        Returns:
            List of FlagUpdate objects
        """
        return self.updates[-limit:] if self.updates else []
    
    def get_evaluation_history(
        self,
        flag_name: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get evaluation history.
        
        Args:
            flag_name: Filter by flag name
            user_id: Filter by user ID
            limit: Maximum number of records
        
        Returns:
            List of evaluation records
        """
        filtered = self.evaluation_history
        
        if flag_name:
            filtered = [r for r in filtered if r.get("flag_name") == flag_name]
        if user_id:
            filtered = [r for r in filtered if r.get("user_id") == user_id]
        
        return filtered[-limit:] if filtered else []
    
    def get_flag_status(self, flag_name: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a flag.
        
        Args:
            flag_name: Name of flag
        
        Returns:
            Status dictionary
        """
        flag = self.get_flag(flag_name)
        if not flag:
            return {"error": "Flag not found"}
        
        metrics = self.get_metrics(flag_name)
        recent_updates = [u.to_dict() for u in self.get_recent_updates(10) if u.flag_name == flag_name]
        recent_evaluations = self.get_evaluation_history(flag_name, limit=10)
        
        return {
            "flag": flag.to_dict(),
            "metrics": metrics,
            "recent_updates": recent_updates,
            "recent_evaluations": recent_evaluations,
            "cache_status": {
                "cached": flag_name in self._flag_cache,
                "cache_time": self._cache_timestamps.get(flag_name),
            },
        }
    
    # ==================== BACKGROUND TASKS ====================
    
    async def start_polling(self):
        """Start polling for flag updates."""
        if self.update_strategy != UpdateStrategy.POLLING:
            logger.warning("Polling started but update strategy is not POLLING")
            return
        
        self._running = True
        
        async def poll_loop():
            while self._running:
                try:
                    logger.debug("Polling for feature flag updates...")
                    self.refresh_cache()
                    await asyncio.sleep(self.polling_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in polling loop: {e}")
                    await asyncio.sleep(self.polling_interval)
        
        self._polling_task = asyncio.create_task(poll_loop())
        logger.info(f"Started feature flag polling (interval: {self.polling_interval}s)")
    
    async def stop_polling(self):
        """Stop polling for flag updates."""
        self._running = False
        
        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
            self._polling_task = None
        
        logger.info("Stopped feature flag polling")
    
    # ==================== UTILITY METHODS ====================
    
    def import_flags(self, flags_data: List[Dict[str, Any]], overwrite: bool = False) -> List[str]:
        """
        Import feature flags from data.
        
        Args:
            flags_data: List of flag data dictionaries
            overwrite: Overwrite existing flags
        
        Returns:
            List of imported flag names
        """
        imported = []
        
        for flag_data in flags_data:
            flag_name = flag_data.get("name")
            if not flag_name:
                logger.warning("Skipping flag without name")
                continue
            
            existing = self.get_flag(flag_name)
            if existing and not overwrite:
                logger.debug(f"Skipping existing flag: {flag_name}")
                continue
            
            try:
                flag = create_flag_from_dict(flag_data)
                self.backend.set(flag_name, flag)
                
                # Update cache
                with self._cache_lock:
                    self._flag_cache[flag_name] = flag
                    self._cache_timestamps[flag_name] = datetime.utcnow()
                
                # Initialize metrics
                if flag_name not in self.metrics:
                    self.metrics[flag_name] = EvaluationMetrics(flag_name=flag_name)
                
                imported.append(flag_name)
                logger.debug(f"Imported flag: {flag_name}")
            except Exception as e:
                logger.error(f"Error importing flag {flag_name}: {e}")
        
        logger.info(f"Imported {len(imported)} feature flags")
        return imported
    
    def export_flags(self) -> List[Dict[str, Any]]:
        """
        Export all feature flags to data.
        
        Returns:
            List of flag data dictionaries
        """
        flags = self.get_all_flags(include_disabled=True)
        return [flag.to_dict() for flag in flags.values()]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get manager statistics.
        
        Returns:
            Statistics dictionary
        """
        flags = self.get_all_flags(include_disabled=True)
        
        enabled_flags = sum(1 for flag in flags.values() if flag.enabled)
        disabled_flags = len(flags) - enabled_flags
        
        total_evaluations = sum(m.total_evaluations for m in self.metrics.values())
        total_errors = sum(m.error_count for m in self.metrics.values())
        
        return {
            "total_flags": len(flags),
            "enabled_flags": enabled_flags,
            "disabled_flags": disabled_flags,
            "cache_size": len(self._flag_cache),
            "total_evaluations": total_evaluations,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_evaluations, 1),
            "recent_updates": len(self.updates),
            "evaluation_history_size": len(self.evaluation_history),
            "segments_count": len(self.evaluator.segments),
            "backend_type": self.backend.__class__.__name__,
            "update_strategy": self.update_strategy.value,
            "polling_interval": self.polling_interval,
            "cache_ttl": self.cache_ttl,
        }


# Singleton instance getter
def get_feature_manager() -> FeatureManager:
    """
    Get the singleton FeatureManager instance.
    
    Returns:
        FeatureManager instance
    """
    return FeatureManager()


# Async context manager for polling
class FeatureManagerContext:
    """Context manager for FeatureManager with polling."""
    
    def __init__(self, manager: Optional[FeatureManager] = None):
        self.manager = manager or get_feature_manager()
    
    async def __aenter__(self) -> FeatureManager:
        """Enter context and start polling."""
        if self.manager.update_strategy == UpdateStrategy.POLLING:
            await self.manager.start_polling()
        return self.manager
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context and stop polling."""
        if self.manager.update_strategy == UpdateStrategy.POLLING:
            await self.manager.stop_polling()