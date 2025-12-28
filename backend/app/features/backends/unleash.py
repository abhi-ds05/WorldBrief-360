"""
Unleash integration for feature flags.
Provides a backend that connects to Unleash feature management platform.
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
import threading
from urllib.parse import urljoin

import requests
from UnleashClient import UnleashClient
from UnleashClient.strategies import Strategy # pyright: ignore[reportMissingImports]

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
class UnleashConfig(BackendConfig):
    """Configuration for Unleash backend."""
    
    # Unleash server configuration
    url: str = "http://localhost:4242"
    app_name: str = "worldbrief360"
    environment: str = "default"
    instance_id: Optional[str] = None
    project_name: Optional[str] = None
    
    # Authentication
    api_token: str = field(default="", metadata={"secret": True})
    custom_headers: Dict[str, str] = field(default_factory=dict)
    
    # Client configuration
    refresh_interval: int = 15  # seconds
    metrics_interval: int = 60  # seconds
    disable_metrics: bool = False
    disable_registration: bool = False
    
    # Cache configuration
    cache_directory: Optional[str] = None
    cache_ttl: int = 86400  # 24 hours
    
    # Network configuration
    request_timeout: int = 30
    request_retries: int = 3
    enable_compression: bool = True
    
    # Bootstrap configuration
    bootstrap_file: Optional[str] = None
    bootstrap_url: Optional[str] = None
    bootstrap_data: Optional[Dict[str, Any]] = None
    
    # Custom strategies
    custom_strategies: List[Dict[str, Any]] = field(default_factory=list)
    
    # Context fields
    default_context_fields: Dict[str, str] = field(default_factory=dict)
    
    def validate(self):
        """Validate configuration."""
        super().validate()
        
        if not self.url:
            raise ValueError("Unleash URL is required")
        
        if not self.app_name:
            raise ValueError("App name is required")
        
        if self.refresh_interval < 1:
            raise ValueError("Refresh interval must be at least 1 second")
        
        if self.metrics_interval < 1:
            raise ValueError("Metrics interval must be at least 1 second")
        
        if self.request_timeout < 1:
            raise ValueError("Request timeout must be at least 1 second")
        
        if self.request_retries < 0:
            raise ValueError("Request retries cannot be negative")


class UnleashBackend(FeatureBackend, RetryMixin, MetricsMixin):
    """
    Unleash feature flag backend.
    
    Integrates with Unleash feature management platform.
    
    Features:
    - Real-time flag updates via polling
    - Custom strategy support
    - Bootstrap configuration for offline mode
    - Detailed metrics and analytics
    - Multi-environment support
    """
    
    supports_persistence = True
    supports_watches = True
    supports_transactions = False
    is_distributed = True
    supports_batch_operations = True
    
    def __init__(self, config: UnleashConfig):
        """
        Initialize Unleash backend.
        
        Args:
            config: UnleashConfig instance
        """
        super().__init__(config)
        self.config = config
        
        # Unleash client
        self._client: Optional[UnleashClient] = None
        
        # Local cache of flags
        self._flag_cache: Dict[str, FeatureFlag] = {}
        self._flag_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Custom strategies
        self._custom_strategies: Dict[str, Strategy] = {}
        
        # Synchronization
        self._sync_lock = threading.RLock()
        self._sync_thread: Optional[threading.Thread] = None
        self._sync_running = False
        
        # Watches
        self._watches: Dict[str, List] = {}
        self._watch_lock = threading.RLock()
        
        # Event listeners
        self._listeners: Dict[str, List] = {
            "flag_changed": [],
            "error": [],
        }
        
        # API client for direct Unleash API access
        self._api_client = UnleashAPIClient(config)
        
        logger.debug(f"UnleashBackend initialized for {config.url}")
    
    def connect(self) -> bool:
        """
        Connect to Unleash.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        try:
            # Initialize Unleash client
            self._initialize_unleash_client()
            
            # Load custom strategies
            self._load_custom_strategies()
            
            # Start synchronization
            self._start_sync_thread()
            
            self._connected = True
            self._stats.status = BackendStatus.CONNECTED
            self._stats.connected_since = datetime.utcnow()
            
            logger.info(f"UnleashBackend connected to {self.config.url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Unleash: {e}")
            self._stats.status = BackendStatus.DISCONNECTED
            raise ConnectionError(
                f"Failed to connect to Unleash: {str(e)}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
    
    def disconnect(self) -> bool:
        """
        Disconnect from Unleash.
        
        Returns:
            True if disconnected successfully
        """
        if not self._connected:
            return True
        
        try:
            # Stop synchronization
            self._stop_sync_thread()
            
            # Destroy Unleash client
            if self._client:
                self._client.destroy()
                self._client = None
            
            # Clear caches
            self._flag_cache.clear()
            self._flag_metadata.clear()
            self._custom_strategies.clear()
            
            self._connected = False
            self._stats.status = BackendStatus.DISCONNECTED
            
            logger.info("UnleashBackend disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Unleash: {e}")
            return False
    
    def _initialize_unleash_client(self):
        """Initialize Unleash client."""
        try:
            # Build configuration
            unleash_config = {
                "url": self.config.url,
                "app_name": self.config.app_name,
                "environment": self.config.environment,
                "instance_id": self.config.instance_id or f"{self.config.app_name}-{time.time()}",
                "refresh_interval": self.config.refresh_interval,
                "metrics_interval": self.config.metrics_interval,
                "disable_metrics": self.config.disable_metrics,
                "disable_registration": self.config.disable_registration,
                "custom_headers": self._build_custom_headers(),
                "custom_options": {
                    "timeout": self.config.request_timeout,
                    "verify": True,
                },
                "cache_directory": self.config.cache_directory,
                "project_name": self.config.project_name,
                "verbose_log_level": 0,
            }
            
            # Add bootstrap if configured
            if self.config.bootstrap_file:
                unleash_config["bootstrap_file"] = self.config.bootstrap_file
            elif self.config.bootstrap_url:
                unleash_config["bootstrap_url"] = self.config.bootstrap_url
            elif self.config.bootstrap_data:
                unleash_config["bootstrap_data"] = self.config.bootstrap_data
            
            # Initialize client
            self._client = UnleashClient(**unleash_config)
            self._client.initialize_client()
            
            # Wait for initialization
            for _ in range(30):  # Wait up to 30 seconds
                if self._client.is_initialized:
                    break
                time.sleep(1)
            
            if not self._client.is_initialized:
                logger.warning("Unleash client initialization timeout")
            
            logger.debug(f"Unleash client initialized: {self._client.is_initialized}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Unleash client: {e}")
            raise
    
    def _build_custom_headers(self) -> Dict[str, str]:
        """Build custom headers for Unleash API."""
        headers = self.config.custom_headers.copy()
        
        # Add Authorization header if API token provided
        if self.config.api_token:
            headers["Authorization"] = self.config.api_token
        
        # Add custom headers
        headers.update({
            "User-Agent": f"WorldBrief360/{self.config.app_name}",
            "Content-Type": "application/json",
        })
        
        return headers
    
    def _load_custom_strategies(self):
        """Load custom strategies."""
        for strategy_config in self.config.custom_strategies:
            try:
                strategy_name = strategy_config["name"]
                strategy_class = strategy_config.get("class")
                parameters = strategy_config.get("parameters", {})
                
                if strategy_class:
                    # Import and instantiate custom strategy
                    module_name, class_name = strategy_class.rsplit('.', 1)
                    module = __import__(module_name, fromlist=[class_name])
                    strategy_class_obj = getattr(module, class_name)
                    
                    strategy = strategy_class_obj(parameters)
                    self._custom_strategies[strategy_name] = strategy
                    
                    logger.debug(f"Loaded custom strategy: {strategy_name}")
                    
            except Exception as e:
                logger.error(f"Failed to load custom strategy: {e}")
    
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag from Unleash.
        
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
            
            # Try to get flag from Unleash API
            flag_data = self._api_client.get_feature(key)
            if flag_data:
                flag = self._convert_unleash_flag(flag_data)
                if flag:
                    # Cache the flag
                    self._flag_cache[key] = flag
                    success = True
                    return flag
            
            # Flag not found
            logger.debug(f"Flag not found in Unleash: {key}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting flag {key} from Unleash: {e}")
            raise OperationError(
                f"Failed to get flag from Unleash: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get", success, start_time)
    
    def _convert_unleash_flag(self, unleash_data: Dict[str, Any]) -> Optional[FeatureFlag]:
        """Convert Unleash flag data to FeatureFlag."""
        try:
            # Extract flag information
            name = unleash_data.get("name", "")
            description = unleash_data.get("description", "")
            enabled = unleash_data.get("enabled", False)
            strategies = unleash_data.get("strategies", [])
            
            # Determine flag type based on strategies
            flag_type = self._determine_flag_type(strategies)
            
            # Extract variants if present
            variants = {}
            if "variants" in unleash_data and unleash_data["variants"]:
                for variant in unleash_data["variants"]:
                    variant_name = variant.get("name", "")
                    variant_weight = variant.get("weight", 0)
                    variant_payload = variant.get("payload", {})
                    
                    variants[variant_name] = {
                        "weight": variant_weight,
                        "payload": variant_payload,
                    }
            
            # Create FeatureFlag
            flag = FeatureFlag(
                name=name,
                description=description,
                flag_type=flag_type,
                enabled=enabled,
                variants=variants if variants else {},
                default_variant=False,  # Unleash default is False
                environments=[self.config.environment],
                metadata={
                    "unleash_data": unleash_data,
                    "strategies": strategies,
                    "created_at": unleash_data.get("createdAt"),
                    "project": unleash_data.get("project"),
                    "stale": unleash_data.get("stale", False),
                },
            )
            
            return flag
            
        except Exception as e:
            logger.error(f"Error converting Unleash flag: {e}")
            return None
    
    def _determine_flag_type(self, strategies: List[Dict]) -> FlagType:
        """Determine flag type from Unleash strategies."""
        if not strategies:
            return FlagType.BOOLEAN
        
        # Check for userWithId strategy (targeted users)
        for strategy in strategies:
            strategy_name = strategy.get("name", "")
            parameters = strategy.get("parameters", {})
            
            if strategy_name == "userWithId" and parameters.get("userIds"):
                return FlagType.TARGETED
            
            if strategy_name == "flexibleRollout":
                return FlagType.PERCENTAGE
            
            if "variants" in strategy:
                return FlagType.MULTIVARIATE
        
        return FlagType.BOOLEAN
    
    def set(self, key: str, flag: FeatureFlag) -> bool:
        """
        Set a feature flag in Unleash.
        
        Note: Requires Unleash admin API access.
        
        Args:
            key: Feature flag key
            flag: FeatureFlag to store
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            # Convert flag to Unleash format
            unleash_data = self._convert_to_unleash_flag(key, flag)
            
            # Create or update flag via API
            if self._api_client.feature_exists(key):
                result = self._api_client.update_feature(key, unleash_data)
            else:
                result = self._api_client.create_feature(unleash_data)
            
            if result:
                # Update local cache
                self._flag_cache[key] = flag
                self._flag_metadata[key] = {
                    "updated_at": datetime.utcnow().isoformat(),
                    "source": "api",
                }
                
                # Notify watchers
                self._notify_watchers(key, flag)
                
                success = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error setting flag {key} in Unleash: {e}")
            raise OperationError(
                f"Failed to set flag in Unleash: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("set", success, start_time)
    
    def _convert_to_unleash_flag(self, key: str, flag: FeatureFlag) -> Dict[str, Any]:
        """Convert FeatureFlag to Unleash format."""
        unleash_data = {
            "name": key,
            "description": flag.description,
            "enabled": flag.enabled,
            "strategies": [],
            "project": self.config.project_name or "default",
        }
        
        # Add strategies based on flag type
        if flag.flag_type == FlagType.TARGETED and flag.target_users:
            strategy = {
                "name": "userWithId",
                "parameters": {
                    "userIds": ",".join(flag.target_users),
                },
            }
            unleash_data["strategies"].append(strategy)
        
        elif flag.flag_type == FlagType.PERCENTAGE and flag.rollout_percentage > 0:
            strategy = {
                "name": "flexibleRollout",
                "parameters": {
                    "rollout": str(int(flag.rollout_percentage * 100)),
                    "stickiness": "default",
                    "groupId": key,
                },
            }
            unleash_data["strategies"].append(strategy)
        
        elif flag.flag_type == FlagType.MULTIVARIATE and flag.variants:
            # For multivariate flags, we need to set up variants
            unleash_data["variants"] = []
            for variant_name, variant_data in flag.variants.items():
                variant = {
                    "name": variant_name,
                    "weight": variant_data.get("weight", 100),
                    "payload": variant_data.get("payload", {}),
                }
                unleash_data["variants"].append(variant)
        
        return unleash_data
    
    def delete(self, key: str) -> bool:
        """
        Delete a feature flag from Unleash.
        
        Note: Requires Unleash admin API access.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            # Delete via API
            result = self._api_client.delete_feature(key)
            
            if result:
                # Remove from local cache
                if key in self._flag_cache:
                    del self._flag_cache[key]
                if key in self._flag_metadata:
                    del self._flag_metadata[key]
                
                # Notify watchers of deletion
                self._notify_watchers(key, None)
                
                success = True
            
            return result
            
        except Exception as e:
            logger.error(f"Error deleting flag {key} from Unleash: {e}")
            raise OperationError(
                f"Failed to delete flag from Unleash: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("delete", success, start_time)
    
    def exists(self, key: str) -> bool:
        """
        Check if a feature flag exists in Unleash.
        
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
            
            # Check via API
            exists = self._api_client.feature_exists(key)
            
            success = True
            return exists
            
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
        Get all feature flags from Unleash.
        
        Returns:
            Dictionary of feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            # Get all features from API
            features = self._api_client.get_all_features()
            
            # Convert to FeatureFlag objects
            flags = {}
            for feature_data in features:
                try:
                    flag = self._convert_unleash_flag(feature_data)
                    if flag:
                        flags[flag.name] = flag
                except Exception as e:
                    logger.warning(f"Error converting feature {feature_data.get('name')}: {e}")
            
            # Update cache
            with self._sync_lock:
                self._flag_cache.update(flags)
            
            success = True
            return flags
            
        except Exception as e:
            logger.error(f"Error getting all flags from Unleash: {e}")
            raise OperationError(
                "Failed to get all flags from Unleash",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get_all", success, start_time)
    
    def clear(self) -> bool:
        """
        Clear all feature flags.
        
        Note: This only clears local cache, not Unleash flags.
        
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
    
    # Evaluation methods (Unleash-specific)
    
    def is_enabled(
        self,
        key: str,
        context: Optional[Dict[str, Any]] = None,
        default_value: bool = False,
    ) -> bool:
        """
        Check if a feature flag is enabled for a context.
        
        Args:
            key: Feature flag key
            context: Evaluation context
            default_value: Default value if flag not found
        
        Returns:
            True if enabled
        """
        if not self._client:
            return default_value
        
        try:
            # Build Unleash context
            unleash_context = self._build_unleash_context(context)
            
            # Evaluate flag
            enabled = self._client.is_enabled(key, unleash_context, default_value)
            
            # Track evaluation
            self._track_evaluation(key, unleash_context, enabled)
            
            return enabled
            
        except Exception as e:
            logger.error(f"Error evaluating flag {key}: {e}")
            return default_value
    
    def get_variant(
        self,
        key: str,
        context: Optional[Dict[str, Any]] = None,
        default_variant: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Get variant for a feature flag.
        
        Args:
            key: Feature flag key
            context: Evaluation context
            default_variant: Default variant if flag not found
        
        Returns:
            Variant dictionary
        """
        if not self._client:
            return default_variant or {}
        
        try:
            # Build Unleash context
            unleash_context = self._build_unleash_context(context)
            
            # Get variant
            variant = self._client.get_variant(key, unleash_context, default_variant)
            
            # Track evaluation
            self._track_evaluation(key, unleash_context, variant.get("enabled", False))
            
            return variant
            
        except Exception as e:
            logger.error(f"Error getting variant for flag {key}: {e}")
            return default_variant or {}
    
    def _build_unleash_context(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build Unleash context from evaluation context."""
        if not context:
            return {}
        
        unleash_context = {}
        
        # Map standard fields
        field_mapping = {
            "user_id": "userId",
            "session_id": "sessionId",
            "ip_address": "remoteAddress",
            "environment": "environment",
            "app_name": "appName",
        }
        
        for our_field, unleash_field in field_mapping.items():
            if our_field in context:
                unleash_context[unleash_field] = context[our_field]
        
        # Add custom properties
        custom_properties = {}
        for key, value in context.items():
            if key not in field_mapping:
                custom_properties[key] = value
        
        if custom_properties:
            unleash_context["properties"] = custom_properties
        
        # Add default context fields
        unleash_context.update(self.config.default_context_fields)
        
        return unleash_context
    
    def _track_evaluation(self, key: str, context: Dict[str, Any], enabled: bool):
        """Track flag evaluation."""
        # Unleash handles tracking automatically via metrics
        pass
    
    # Synchronization methods
    
    def _start_sync_thread(self):
        """Start background flag synchronization thread."""
        if self._sync_thread is not None:
            return
        
        self._sync_running = True
        
        def sync_worker():
            while self._sync_running:
                try:
                    time.sleep(self.config.refresh_interval)
                    self._sync_flags_from_unleash()
                except Exception as e:
                    logger.error(f"Error in sync thread: {e}")
                    time.sleep(min(self.config.refresh_interval, 60))
        
        self._sync_thread = threading.Thread(
            target=sync_worker,
            name="Unleash-Sync",
            daemon=True,
        )
        self._sync_thread.start()
        
        logger.debug("Started Unleash sync thread")
    
    def _stop_sync_thread(self):
        """Stop background flag synchronization thread."""
        self._sync_running = False
        
        if self._sync_thread:
            self._sync_thread.join(timeout=5)
            self._sync_thread = None
            
            logger.debug("Stopped Unleash sync thread")
    
    def _sync_flags_from_unleash(self):
        """Synchronize flags from Unleash."""
        if not self._client:
            return
        
        try:
            # Get current features from Unleash
            features = self._api_client.get_all_features()
            
            # Update cache
            with self._sync_lock:
                for feature_data in features:
                    try:
                        flag = self._convert_unleash_flag(feature_data)
                        if flag:
                            old_flag = self._flag_cache.get(flag.name)
                            
                            # Check if flag changed
                            if old_flag != flag:
                                self._flag_cache[flag.name] = flag
                                self._notify_flag_change(flag.name, flag, old_flag)
                                
                                logger.debug(f"Flag updated: {flag.name}")
                    except Exception as e:
                        logger.warning(f"Error processing feature {feature_data.get('name')}: {e}")
            
            logger.debug(f"Synced {len(features)} flags from Unleash")
            
        except Exception as e:
            logger.error(f"Error syncing flags from Unleash: {e}")
    
    def _notify_flag_change(self, key: str, new_flag: FeatureFlag, old_flag: Optional[FeatureFlag]):
        """Notify listeners of flag change."""
        # Notify watchers
        self._notify_watchers(key, new_flag)
        
        # Notify event listeners
        for callback in self._listeners.get("flag_changed", []):
            try:
                callback(key, new_flag, old_flag)
            except Exception as e:
                logger.error(f"Error in flag change listener: {e}")
    
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
    
    # Event listener methods
    
    def add_listener(self, event_type: str, callback):
        """
        Add event listener.
        
        Args:
            event_type: Type of event (flag_changed, error)
            callback: Callback function
        """
        if event_type in self._listeners:
            self._listeners[event_type].append(callback)
            logger.debug(f"Added listener for {event_type}")
    
    def remove_listener(self, event_type: str, callback):
        """
        Remove event listener.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self._listeners and callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)
            logger.debug(f"Removed listener for {event_type}")
    
    # Health and monitoring
    
    def health_check(self) -> bool:
        """Perform health check on Unleash backend."""
        if not self._client:
            return False
        
        try:
            # Check if client is initialized
            initialized = self._client.is_initialized
            
            # Check connectivity via API
            if initialized:
                health = self._api_client.health_check()
                return health
            
            return initialized
            
        except Exception as e:
            logger.warning(f"Unleash health check failed: {e}")
            return False
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics."""
        stats = super().get_stats()
        
        # Add Unleash-specific stats
        if self._client:
            try:
                # Get metrics from Unleash client
                metrics_data = self._client.__dict__.get('metrics', {})
                
                stats.extra_stats = {
                    "flag_cache_size": len(self._flag_cache),
                    "watches_count": len(self._watches),
                    "client_initialized": self._client.is_initialized,
                    "last_sync": getattr(self._client, 'last_sync', None),
                    "metrics": metrics_data,
                }
            except Exception as e:
                logger.debug(f"Could not get Unleash metrics: {e}")
                stats.extra_stats = {
                    "flag_cache_size": len(self._flag_cache),
                    "watches_count": len(self._watches),
                    "client_initialized": self._client.is_initialized,
                }
        
        return stats
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics."""
        diagnostics = super().get_diagnostics()
        
        diagnostics.update({
            "unleash_config": {
                "url": self.config.url,
                "app_name": self.config.app_name,
                "environment": self.config.environment,
                "project_name": self.config.project_name,
                "refresh_interval": self.config.refresh_interval,
            },
            "cache_info": {
                "cached_flags": list(self._flag_cache.keys()),
                "flag_count": len(self._flag_cache),
            },
            "watch_info": {
                "watched_flags": list(self._watches.keys()),
                "total_watches": sum(len(callbacks) for callbacks in self._watches.values()),
            },
            "client_info": {
                "initialized": self._client.is_initialized if self._client else False,
                "custom_strategies": list(self._custom_strategies.keys()),
            },
        })
        
        return diagnostics
    
    # Utility methods
    
    def force_refresh(self):
        """Force refresh of flags from Unleash."""
        if self._client:
            self._client.fetch_features()
            logger.debug("Forced refresh of Unleash flags")
    
    def get_client(self) -> Optional[UnleashClient]:
        """
        Get the underlying Unleash client.
        
        Returns:
            UnleashClient instance
        """
        return self._client
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.disconnect()
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        flags = len(self._flag_cache)
        return f"UnleashBackend({status}, flags={flags}, url={self.config.url})"


class UnleashAPIClient:
    """HTTP client for Unleash API."""
    
    def __init__(self, config: UnleashConfig):
        self.config = config
        self.session = requests.Session()
        
        # Configure session
        self.session.headers.update(self._build_headers())
        self.session.timeout = config.request_timeout
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers."""
        headers = {
            "User-Agent": f"WorldBrief360-Unleash/{self.config.app_name}",
            "Content-Type": "application/json",
        }
        
        if self.config.api_token:
            headers["Authorization"] = self.config.api_token
        
        headers.update(self.config.custom_headers)
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request to Unleash API."""
        url = urljoin(self.config.url, endpoint)
        
        for attempt in range(self.config.request_retries + 1):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.RequestException as e:
                if attempt == self.config.request_retries:
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        raise OperationError("Request failed after retries", backend_type="UnleashAPIClient")
    
    def health_check(self) -> bool:
        """Check Unleash server health."""
        try:
            response = self._make_request("GET", "/health")
            data = response.json()
            return data.get("health", {}).get("healthy", False)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    def get_all_features(self) -> List[Dict[str, Any]]:
        """Get all features from Unleash."""
        try:
            endpoint = "/api/client/features"
            if self.config.project_name:
                endpoint = f"/api/admin/projects/{self.config.project_name}/features"
            
            response = self._make_request("GET", endpoint)
            data = response.json()
            
            if self.config.project_name:
                return data.get("features", [])
            else:
                return data.get("features", {}).get("features", [])
        except Exception as e:
            logger.error(f"Failed to get all features: {e}")
            return []
    
    def get_feature(self, key: str) -> Optional[Dict[str, Any]]:
        """Get a specific feature from Unleash."""
        try:
            endpoint = f"/api/admin/projects/{self.config.project_name or 'default'}/features/{key}"
            response = self._make_request("GET", endpoint)
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get feature {key}: {e}")
            return None
    
    def feature_exists(self, key: str) -> bool:
        """Check if a feature exists."""
        feature = self.get_feature(key)
        return feature is not None
    
    def create_feature(self, feature_data: Dict[str, Any]) -> bool:
        """Create a new feature in Unleash."""
        try:
            project = self.config.project_name or "default"
            endpoint = f"/api/admin/projects/{project}/features"
            
            response = self._make_request("POST", endpoint, json=feature_data)
            return response.status_code == 201
        except Exception as e:
            logger.error(f"Failed to create feature: {e}")
            return False
    
    def update_feature(self, key: str, feature_data: Dict[str, Any]) -> bool:
        """Update an existing feature in Unleash."""
        try:
            project = self.config.project_name or "default"
            endpoint = f"/api/admin/projects/{project}/features/{key}"
            
            response = self._make_request("PUT", endpoint, json=feature_data)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to update feature {key}: {e}")
            return False
    
    def delete_feature(self, key: str) -> bool:
        """Delete a feature from Unleash."""
        try:
            project = self.config.project_name or "default"
            endpoint = f"/api/admin/projects/{project}/features/{key}"
            
            response = self._make_request("DELETE", endpoint)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to delete feature {key}: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics from Unleash."""
        try:
            response = self._make_request("GET", "/api/client/metrics")
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {}