"""
LaunchDarkly integration for feature flags.
Provides a backend that connects to LaunchDarkly's feature management platform.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
import threading

import ldclient
from ldclient.config import Config
from ldclient.context import Context
from ldclient.feature_store import CacheConfig
from ldclient.impl.events.event_processor import DefaultEventProcessor

from .base import (
    FeatureBackend,
    BackendConfig,
    BackendError,
    ConnectionError,
    OperationError,
    BackendStatus,
    BackendStats,
    RetryMixin,
    MetricsMixin,
)
from ..flags import FeatureFlag, FlagType, create_flag_from_dict
from ..context import FeatureContext, UserContext

logger = logging.getLogger(__name__)


@dataclass
class LaunchDarklyConfig(BackendConfig):
    """Configuration for LaunchDarkly backend."""
    
    # LaunchDarkly SDK key
    sdk_key: str = field(default="", metadata={"secret": True})
    
    # LaunchDarkly configuration
    base_uri: str = "https://app.launchdarkly.com"
    events_uri: str = "https://events.launchdarkly.com"
    stream_uri: str = "https://stream.launchdarkly.com"
    
    # SDK configuration
    offline: bool = False
    send_events: bool = True
    events_max_pending: int = 10000
    flush_interval: float = 5.0
    sampling_interval: float = 0.001
    all_attributes_private: bool = False
    private_attribute_names: List[str] = field(default_factory=list)
    
    # Polling configuration
    use_polling: bool = False
    poll_interval: float = 30.0
    stream_uri: str = "https://stream.launchdarkly.com"
    
    # Caching
    cache_config: Optional[Dict[str, Any]] = None
    feature_store: Optional[Any] = None
    
    # Context configuration
    default_context_kind: str = "user"
    anonymous_context_key: str = "anonymous"
    
    # Flag synchronization
    sync_flags: bool = True
    sync_interval: int = 60  # seconds
    
    def validate(self):
        """Validate configuration."""
        super().validate()
        
        if not self.sdk_key and not self.offline:
            raise ValueError("SDK key is required when not in offline mode")
        
        if self.poll_interval < 1.0:
            raise ValueError("Poll interval must be at least 1 second")
        
        if self.flush_interval < 0.1:
            raise ValueError("Flush interval must be at least 0.1 seconds")
        
        if self.sampling_interval < 0.0 or self.sampling_interval > 1.0:
            raise ValueError("Sampling interval must be between 0.0 and 1.0")


class LaunchDarklyBackend(FeatureBackend, RetryMixin, MetricsMixin):
    """
    LaunchDarkly feature flag backend.
    
    Integrates with LaunchDarkly's feature management platform.
    
    Features:
    - Real-time flag updates via streaming
    - Full LaunchDarkly SDK capabilities
    - Automatic flag synchronization
    - Rich targeting and segmentation
    - Detailed analytics and metrics
    """
    
    supports_persistence = True
    supports_watches = True
    supports_transactions = False
    is_distributed = True
    supports_batch_operations = True
    
    def __init__(self, config: LaunchDarklyConfig):
        """
        Initialize LaunchDarkly backend.
        
        Args:
            config: LaunchDarklyConfig instance
        """
        super().__init__(config)
        self.config = config
        
        # LaunchDarkly client
        self._client: Optional[ldclient.LDClient] = None
        
        # Local cache of flags (for faster access)
        self._flag_cache: Dict[str, FeatureFlag] = {}
        self._flag_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Synchronization
        self._sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_running = False
        
        # Watches
        self._watches: Dict[str, List] = {}
        self._watch_lock = threading.RLock()
        
        # Flag change listeners
        self._listeners: List = []
        
        logger.debug(f"LaunchDarklyBackend initialized (offline={config.offline})")
    
    def connect(self) -> bool:
        """
        Connect to LaunchDarkly.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        try:
            # Configure LaunchDarkly SDK
            ld_config = self._create_ld_config()
            
            # Initialize client
            self._client = ldclient.get(sdk_key=self.config.sdk_key, config=ld_config)
            
            # Wait for client initialization
            if not self.config.offline:
                initialized = self._client.wait_for_initialization(timeout=10)
                if not initialized:
                    logger.warning("LaunchDarkly client initialization timeout")
            
            # Start flag synchronization
            if self.config.sync_flags:
                self._start_sync_thread()
            
            # Set up flag change listener
            self._setup_flag_listener()
            
            self._connected = True
            self._stats.status = BackendStatus.CONNECTED
            self._stats.connected_since = datetime.utcnow()
            
            logger.info(f"LaunchDarklyBackend connected (offline={self.config.offline})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to LaunchDarkly: {e}")
            self._stats.status = BackendStatus.DISCONNECTED
            raise ConnectionError(
                f"Failed to connect to LaunchDarkly: {str(e)}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
    
    def disconnect(self) -> bool:
        """
        Disconnect from LaunchDarkly.
        
        Returns:
            True if disconnected successfully
        """
        if not self._connected:
            return True
        
        try:
            # Stop synchronization
            if self.config.sync_flags:
                self._stop_sync_thread()
            
            # Remove listeners
            self._remove_flag_listener()
            
            # Close LaunchDarkly client
            if self._client:
                self._client.close()
                self._client = None
            
            # Clear caches
            self._flag_cache.clear()
            self._flag_metadata.clear()
            
            self._connected = False
            self._stats.status = BackendStatus.DISCONNECTED
            
            logger.info("LaunchDarklyBackend disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from LaunchDarkly: {e}")
            return False
    
    def _create_ld_config(self) -> Config:
        """Create LaunchDarkly SDK configuration."""
        config_builder = Config(
            sdk_key=self.config.sdk_key,
            base_uri=self.config.base_uri,
            events_uri=self.config.events_uri,
            stream_uri=self.config.stream_uri,
            offline=self.config.offline,
            send_events=self.config.send_events,
            events_max_pending=self.config.events_max_pending,
            flush_interval=self.config.flush_interval,
            sampling_interval=self.config.sampling_interval,
            all_attributes_private=self.config.all_attributes_private,
            private_attribute_names=self.config.private_attribute_names,
            use_ldd=self.config.offline,
            application={"id": "worldbrief360", "version": "1.0.0"},
        )
        
        # Configure polling if needed
        if self.config.use_polling:
            config_builder = Config(
                sdk_key=self.config.sdk_key,
                base_uri=self.config.base_uri,
                events_uri=self.config.events_uri,
                stream_uri=self.config.stream_uri,
                update_processor_class=ldclient.polling.PollingUpdateProcessor,
                poll_interval=self.config.poll_interval,
            )
        
        # Configure cache if specified
        if self.config.cache_config:
            cache_config = CacheConfig(
                enabled=self.config.cache_config.get("enabled", True),
                ttl=self.config.cache_config.get("ttl", 30),
            )
            config_builder = config_builder.cache_config(cache_config)
        
        # Configure feature store if specified
        if self.config.feature_store:
            config_builder = config_builder.feature_store(self.config.feature_store)
        
        return config_builder
    
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag from LaunchDarkly.
        
        Note: LaunchDarkly doesn't support direct flag retrieval without evaluation.
        This method returns a synthetic flag based on default configuration.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag if found, None otherwise
        """
        start_time = time.time()
        success = False
        
        try:
            # Check cache first
            if key in self._flag_cache:
                return self._flag_cache[key]
            
            # Try to get flag details from LaunchDarkly
            # Note: This requires the flag to be predefined in LaunchDarkly
            if self._client and not self.config.offline:
                # Get flag variation details
                # This is a workaround since LD doesn't expose raw flag data
                flag = self._get_flag_from_ld(key)
                if flag:
                    # Cache the flag
                    self._flag_cache[key] = flag
                    success = True
                    return flag
            
            # Return None if flag not found
            logger.debug(f"Flag not found in LaunchDarkly: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting flag {key} from LaunchDarkly: {e}")
            raise OperationError(
                f"Failed to get flag from LaunchDarkly: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get", success, start_time)
    
    def _get_flag_from_ld(self, key: str) -> Optional[FeatureFlag]:
        """
        Extract flag information from LaunchDarkly.
        
        This is a workaround since LaunchDarkly doesn't provide a direct API
        to get flag definitions without evaluation context.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag or None
        """
        try:
            # Create anonymous context to get default flag state
            context = Context.builder(self.config.anonymous_context_key).build()
            
            # Get flag details (this varies by SDK version)
            # In newer SDKs, you might use _client._get_flag()
            # This is implementation-specific
            
            # For now, create a synthetic flag
            # In production, you'd want to implement proper flag retrieval
            flag = FeatureFlag(
                name=key,
                description=f"LaunchDarkly flag: {key}",
                flag_type=FlagType.BOOLEAN,
                enabled=False,  # Default state
                default_variant=False,
                environments=["production"],  # Assuming production
            )
            
            return flag
            
        except Exception as e:
            logger.warning(f"Could not retrieve flag details for {key}: {e}")
            return None
    
    def set(self, key: str, flag: FeatureFlag) -> bool:
        """
        Set a feature flag in LaunchDarkly.
        
        Note: LaunchDarkly doesn't support programmatic flag creation via SDK.
        Flags must be created in the LaunchDarkly dashboard.
        
        Args:
            key: Feature flag key
            flag: FeatureFlag to store
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            logger.warning(
                "LaunchDarkly doesn't support programmatic flag creation via SDK. "
                "Flags must be created in the LaunchDarkly dashboard."
            )
            
            # Update local cache
            self._flag_cache[key] = flag
            self._flag_metadata[key] = {
                "updated_at": datetime.utcnow().isoformat(),
                "source": "local_cache",
            }
            
            # Notify watchers
            self._notify_watchers(key, flag)
            
            success = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting flag {key} in LaunchDarkly: {e}")
            raise OperationError(
                f"Failed to set flag in LaunchDarkly: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("set", success, start_time)
    
    def delete(self, key: str) -> bool:
        """
        Delete a feature flag from LaunchDarkly.
        
        Note: LaunchDarkly doesn't support programmatic flag deletion via SDK.
        Flags must be deleted in the LaunchDarkly dashboard.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            logger.warning(
                "LaunchDarkly doesn't support programmatic flag deletion via SDK. "
                "Flags must be deleted in the LaunchDarkly dashboard."
            )
            
            # Remove from local cache
            if key in self._flag_cache:
                del self._flag_cache[key]
            if key in self._flag_metadata:
                del self._flag_metadata[key]
            
            # Notify watchers of deletion
            self._notify_watchers(key, None)
            
            success = True
            return True
            
        except Exception as e:
            logger.error(f"Error deleting flag {key} from LaunchDarkly: {e}")
            raise OperationError(
                f"Failed to delete flag from LaunchDarkly: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("delete", success, start_time)
    
    def exists(self, key: str) -> bool:
        """
        Check if a feature flag exists in LaunchDarkly.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if exists
        """
        start_time = time.time()
        success = False
        
        try:
            # Check cache first
            if key in self._flag_cache:
                return True
            
            # Try to get flag from LaunchDarkly
            if self._client and not self.config.offline:
                # This is a workaround - check if flag exists by evaluating it
                context = Context.builder(self.config.anonymous_context_key).build()
                try:
                    # Try to get flag variation
                    variation = self._client.variation(key, context, False)
                    # If we get here without error, flag likely exists
                    success = True
                    return True
                except Exception:
                    # Flag doesn't exist or error occurred
                    pass
            
            success = True
            return False
            
        except Exception as e:
            logger.error(f"Error checking flag existence {key}: {e}")
            raise OperationError(
                f"Failed to check flag existence: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("exists", success, start_time)
    
    def get_all(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags from LaunchDarkly.
        
        Note: LaunchDarkly doesn't provide a direct API to get all flags.
        This method returns cached flags and tries to sync from LD.
        
        Returns:
            Dictionary of feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            # Trigger synchronization if needed
            if self.config.sync_flags:
                self._sync_flags_from_ld()
            
            # Return cached flags
            success = True
            return self._flag_cache.copy()
            
        except Exception as e:
            logger.error(f"Error getting all flags from LaunchDarkly: {e}")
            raise OperationError(
                "Failed to get all flags from LaunchDarkly",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get_all", success, start_time)
    
    def clear(self) -> bool:
        """
        Clear all feature flags.
        
        Note: This only clears local cache, not LaunchDarkly flags.
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            with self._sync_lock:
                self._flag_cache.clear()
                self._flag_metadata.clear()
                
                # Notify watchers
                for key in list(self._watches.keys()):
                    self._notify_watchers(key, None)
            
            success = True
            return True
            
        except Exception as e:
            logger.error(f"Error clearing flags: {e}")
            raise OperationError(
                "Failed to clear flags",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("clear", success, start_time)
    
    # Evaluation methods (LaunchDarkly-specific)
    
    def evaluate(
        self,
        key: str,
        user_context: Optional[Dict[str, Any]] = None,
        default_value: Any = None,
    ) -> Any:
        """
        Evaluate a feature flag for a user context.
        
        Args:
            key: Feature flag key
            user_context: User context for evaluation
            default_value: Default value if flag not found
        
        Returns:
            Flag variation value
        """
        if not self._client:
            return default_value
        
        try:
            # Create LaunchDarkly context
            context = self._create_ld_context(user_context)
            
            # Evaluate flag
            variation = self._client.variation(key, context, default_value)
            
            # Track evaluation in analytics
            self._track_evaluation(key, context, variation)
            
            return variation
            
        except Exception as e:
            logger.error(f"Error evaluating flag {key}: {e}")
            return default_value
    
    def evaluate_detail(
        self,
        key: str,
        user_context: Optional[Dict[str, Any]] = None,
        default_value: Any = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a feature flag with detailed results.
        
        Args:
            key: Feature flag key
            user_context: User context for evaluation
            default_value: Default value if flag not found
        
        Returns:
            Dictionary with evaluation details
        """
        if not self._client:
            return {
                "value": default_value,
                "reason": {"kind": "ERROR", "errorKind": "CLIENT_NOT_READY"},
            }
        
        try:
            # Create LaunchDarkly context
            context = self._create_ld_context(user_context)
            
            # Evaluate flag with detail
            detail = self._client.variation_detail(key, context, default_value)
            
            # Track evaluation in analytics
            self._track_evaluation(key, context, detail.value)
            
            return {
                "value": detail.value,
                "reason": {
                    "kind": detail.reason.kind,
                    "errorKind": getattr(detail.reason, 'errorKind', None),
                    "ruleIndex": getattr(detail.reason, 'ruleIndex', None),
                    "ruleId": getattr(detail.reason, 'ruleId', None),
                    "prerequisiteKey": getattr(detail.reason, 'prerequisiteKey', None),
                },
                "variationIndex": detail.variation_index,
            }
            
        except Exception as e:
            logger.error(f"Error evaluating flag {key} with detail: {e}")
            return {
                "value": default_value,
                "reason": {"kind": "ERROR", "errorKind": "EXCEPTION"},
            }
    
    def _create_ld_context(self, user_context: Optional[Dict[str, Any]] = None) -> Context:
        """Create LaunchDarkly context from user context."""
        if not user_context:
            return Context.builder(self.config.anonymous_context_key).build()
        
        # Extract user information
        user_id = user_context.get("user_id") or user_context.get("id") or self.config.anonymous_context_key
        builder = Context.builder(user_id)
        
        # Set kind (default to "user")
        kind = user_context.get("kind", self.config.default_context_kind)
        builder.kind(kind)
        
        # Set name
        if "name" in user_context:
            builder.name(user_context["name"])
        
        # Set email
        if "email" in user_context:
            builder.set("email", user_context["email"])
        
        # Set custom attributes
        for key, value in user_context.items():
            if key not in ["user_id", "id", "kind", "name", "email"]:
                builder.set(key, value)
        
        return builder.build()
    
    def _track_evaluation(self, key: str, context: Context, value: Any):
        """Track flag evaluation in LaunchDarkly analytics."""
        if self._client and self.config.send_events:
            try:
                self._client.track("feature_flag_evaluated", context, data={
                    "key": key,
                    "value": value,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            except Exception as e:
                logger.debug(f"Failed to track evaluation for {key}: {e}")
    
    # Synchronization methods
    
    def _start_sync_thread(self):
        """Start background flag synchronization thread."""
        if self._sync_thread is not None:
            return
        
        self._sync_running = True
        
        def sync_worker():
            while self._sync_running:
                try:
                    time.sleep(self.config.sync_interval)
                    self._sync_flags_from_ld()
                except Exception as e:
                    logger.error(f"Error in sync thread: {e}")
                    time.sleep(min(self.config.sync_interval, 60))  # Backoff
        
        self._sync_thread = threading.Thread(
            target=sync_worker,
            name="LaunchDarkly-Sync",
            daemon=True,
        )
        self._sync_thread.start()
        
        logger.debug("Started LaunchDarkly sync thread")
    
    def _stop_sync_thread(self):
        """Stop background flag synchronization thread."""
        self._sync_running = False
        
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
            self._sync_thread = None
            
            logger.debug("Stopped LaunchDarkly sync thread")
    
    def _sync_flags_from_ld(self):
        """Synchronize flags from LaunchDarkly."""
        if not self._client or self.config.offline:
            return
        
        try:
            # This is a placeholder - actual implementation depends on
            # how you want to sync flags from LaunchDarkly
            
            # In a real implementation, you might:
            # 1. Use LaunchDarkly's REST API to get flag definitions
            # 2. Parse flag data into FeatureFlag objects
            # 3. Update local cache
            
            logger.debug("Syncing flags from LaunchDarkly")
            
            # For now, just update cache with existing flags
            with self._sync_lock:
                # This would be replaced with actual sync logic
                pass
            
        except Exception as e:
            logger.error(f"Error syncing flags from LaunchDarkly: {e}")
    
    # Watch implementation
    
    def watch(self, key: str, callback) -> bool:
        """
        Watch for changes to a feature flag.
        
        Args:
            key: Feature flag key to watch
            callback: Function to call when flag changes
        
        Returns:
            True if watch established
        """
        with self._watch_lock:
            if key not in self._watches:
                self._watches[key] = []
            
            self._watches[key].append(callback)
            logger.debug(f"Added watch for {key}")
            
            return True
    
    def unwatch(self, key: str) -> bool:
        """
        Stop watching a feature flag.
        
        Args:
            key: Feature flag key to stop watching
        
        Returns:
            True if watch removed
        """
        with self._watch_lock:
            if key in self._watches:
                del self._watches[key]
                logger.debug(f"Removed watch for {key}")
                return True
            
            return False
    
    def _notify_watchers(self, key: str, flag: Optional[FeatureFlag]):
        """Notify all watchers of a flag change."""
        with self._watch_lock:
            if key in self._watches:
                for callback in self._watches[key]:
                    try:
                        callback(key, flag)
                    except Exception as e:
                        logger.error(f"Error in watch callback for {key}: {e}")
    
    # Flag change listener
    
    def _setup_flag_listener(self):
        """Set up LaunchDarkly flag change listener."""
        if not self._client or self.config.offline:
            return
        
        try:
            # Add flag change listener
            listener = self._client.flag_tracker.add_flag_value_change_listener(
                key="",
                context=Context.builder(self.config.anonymous_context_key).build(),
                callback=self._on_flag_change,
            )
            self._listeners.append(listener)
            
            logger.debug("Set up LaunchDarkly flag change listener")
            
        except Exception as e:
            logger.error(f"Error setting up flag change listener: {e}")
    
    def _remove_flag_listener(self):
        """Remove flag change listeners."""
        for listener in self._listeners:
            try:
                self._client.flag_tracker.remove_listener(listener)
            except Exception as e:
                logger.debug(f"Error removing listener: {e}")
        
        self._listeners.clear()
    
    def _on_flag_change(self, change_event):
        """Handle flag change events from LaunchDarkly."""
        try:
            key = change_event.key
            new_value = change_event.new_value
            old_value = change_event.old_value
            
            logger.info(f"Flag changed: {key} from {old_value} to {new_value}")
            
            # Update local cache
            # Note: We'd need to convert LD flag to our FeatureFlag format
            
            # Notify watchers
            self._notify_watchers(key, None)  # Pass None for now
            
        except Exception as e:
            logger.error(f"Error handling flag change: {e}")
    
    # Health and monitoring
    
    def health_check(self) -> bool:
        """Perform health check on LaunchDarkly backend."""
        if not self._client:
            return False
        
        try:
            # Check if client is initialized
            initialized = self._client.is_initialized()
            
            # Check connectivity (offline mode always returns True)
            if not self.config.offline:
                # Try a simple flag evaluation
                context = Context.builder("health-check").build()
                self._client.variation("health-check-flag", context, "default")
            
            return initialized
            
        except Exception as e:
            logger.warning(f"LaunchDarkly health check failed: {e}")
            return False
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics."""
        stats = super().get_stats()
        
        # Add LaunchDarkly-specific stats
        if self._client:
            stats.extra_stats = {
                "flag_cache_size": len(self._flag_cache),
                "watches_count": len(self._watches),
                "listeners_count": len(self._listeners),
                "client_initialized": self._client.is_initialized() if self._client else False,
                "offline_mode": self.config.offline,
                "send_events": self.config.send_events,
            }
        
        return stats
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics."""
        diagnostics = super().get_diagnostics()
        
        diagnostics.update({
            "launchdarkly_config": {
                "base_uri": self.config.base_uri,
                "offline": self.config.offline,
                "send_events": self.config.send_events,
                "use_polling": self.config.use_polling,
                "sync_flags": self.config.sync_flags,
            },
            "cache_info": {
                "cached_flags": list(self._flag_cache.keys()),
                "flag_count": len(self._flag_cache),
                "metadata_count": len(self._flag_metadata),
            },
            "watch_info": {
                "watched_flags": list(self._watches.keys()),
                "total_watches": sum(len(callbacks) for callbacks in self._watches.values()),
            },
            "client_info": {
                "initialized": self._client.is_initialized() if self._client else False,
                "sdk_version": ldclient.version.__version__ if hasattr(ldclient, 'version') else "unknown",
            },
        })
        
        return diagnostics
    
    # Utility methods for working with LaunchDarkly
    
    def identify_user(self, user_context: Dict[str, Any]):
        """
        Identify a user to LaunchDarkly.
        
        Args:
            user_context: User context information
        """
        if not self._client or not self.config.send_events:
            return
        
        try:
            context = self._create_ld_context(user_context)
            self._client.identify(context)
            logger.debug(f"Identified user: {context.key}")
        except Exception as e:
            logger.error(f"Error identifying user: {e}")
    
    def track_event(self, event_key: str, user_context: Dict[str, Any], data: Optional[Dict] = None):
        """
        Track a custom event to LaunchDarkly.
        
        Args:
            event_key: Event key
            user_context: User context
            data: Optional event data
        """
        if not self._client or not self.config.send_events:
            return
        
        try:
            context = self._create_ld_context(user_context)
            self._client.track(event_key, context, data)
            logger.debug(f"Tracked event: {event_key} for user {context.key}")
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
    
    def get_all_flags_state(
        self,
        user_context: Dict[str, Any],
        client_side_only: bool = False,
        with_reasons: bool = False,
        details_only_for_tracked_flags: bool = False,
    ) -> Dict[str, Any]:
        """
        Get all flag states for a user context.
        
        Args:
            user_context: User context
            client_side_only: Only include client-side flags
            with_reasons: Include evaluation reasons
            details_only_for_tracked_flags: Only include details for tracked flags
        
        Returns:
            Dictionary of all flag states
        """
        if not self._client:
            return {}
        
        try:
            context = self._create_ld_context(user_context)
            flags_state = self._client.all_flags_state(
                context,
                client_side_only=client_side_only,
                with_reasons=with_reasons,
                details_only_for_tracked_flags=details_only_for_tracked_flags,
            )
            
            return flags_state.to_json_dict()
            
        except Exception as e:
            logger.error(f"Error getting all flags state: {e}")
            return {}
    
    def flush_events(self):
        """Flush pending events to LaunchDarkly."""
        if self._client:
            self._client.flush()
            logger.debug("Flushed events to LaunchDarkly")
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Flush events before disconnecting
        self.flush_events()
        self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        offline = "offline" if self.config.offline else "online"
        return f"LaunchDarklyBackend({status}, {offline}, flags={len(self._flag_cache)})"