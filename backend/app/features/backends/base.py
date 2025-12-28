"""
Base classes and interfaces for feature flag backends.
Defines the abstract interface that all feature flag backends must implement.
"""

import abc
import logging
from typing import Dict, Any, Optional, List, Set, Iterator, AsyncIterator
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import json

from ..flags import FeatureFlag, FlagType, create_flag_from_dict
from ..context import FeatureContext

logger = logging.getLogger(__name__)


class BackendError(Exception):
    """Base exception for backend errors."""
    
    def __init__(self, message: str, backend_type: str = "unknown", cause: Exception = None):
        self.message = message
        self.backend_type = backend_type
        self.cause = cause
        super().__init__(f"{backend_type} backend error: {message}")


class ConnectionError(BackendError):
    """Raised when backend connection fails."""
    pass


class TimeoutError(BackendError):
    """Raised when backend operation times out."""
    pass


class ConfigurationError(BackendError):
    """Raised when backend configuration is invalid."""
    pass


class OperationError(BackendError):
    """Raised when backend operation fails."""
    pass


class BackendStatus(str, Enum):
    """Status of a backend connection."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class BackendStats:
    """Statistics for a backend."""
    
    backend_type: str
    status: BackendStatus
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_latency_ms: float = 0.0
    last_operation: Optional[datetime] = None
    connected_since: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        # Convert datetime to ISO string
        for field_name in ["last_operation", "connected_since"]:
            if value := getattr(self, field_name):
                result[field_name] = value.isoformat()
        return result
    
    def record_operation(self, success: bool, latency_ms: float):
        """Record an operation."""
        self.total_operations += 1
        if success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1
        
        # Update average latency (exponential moving average)
        alpha = 0.1  # Smoothing factor
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
        
        self.last_operation = datetime.utcnow()


@dataclass
class BackendConfig:
    """Base configuration for all backends."""
    
    # Common configuration options
    name: str = "default"
    environment: str = "production"
    key_prefix: str = "features:"
    default_ttl: Optional[int] = None  # Time-to-live in seconds
    connection_timeout: float = 5.0  # seconds
    operation_timeout: float = 2.0  # seconds
    retry_attempts: int = 3
    retry_delay: float = 0.1  # seconds
    
    # Monitoring
    enable_metrics: bool = True
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    
    # Serialization
    serializer: str = "json"  # json, pickle, msgpack
    compress: bool = False
    
    # Feature flag defaults
    default_enabled: bool = False
    default_rollout_percentage: float = 0.0
    
    def validate(self):
        """Validate configuration."""
        if not self.name:
            raise ConfigurationError("Backend name cannot be empty")
        
        if self.connection_timeout <= 0:
            raise ConfigurationError("Connection timeout must be positive")
        
        if self.operation_timeout <= 0:
            raise ConfigurationError("Operation timeout must be positive")
        
        if self.retry_attempts < 0:
            raise ConfigurationError("Retry attempts cannot be negative")
        
        if self.retry_delay < 0:
            raise ConfigurationError("Retry delay cannot be negative")
        
        if self.serializer not in ["json", "pickle", "msgpack"]:
            raise ConfigurationError(f"Unsupported serializer: {self.serializer}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class FeatureBackend(abc.ABC):
    """
    Abstract base class for feature flag backends.
    
    All feature flag backends must implement this interface.
    """
    
    # Backend capabilities
    supports_persistence: bool = False
    supports_watches: bool = False
    supports_transactions: bool = False
    is_distributed: bool = False
    supports_batch_operations: bool = False
    
    def __init__(self, config: BackendConfig):
        """
        Initialize backend with configuration.
        
        Args:
            config: Backend configuration
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.config = config
        self.config.validate()
        
        self._stats = BackendStats(
            backend_type=self.__class__.__name__,
            status=BackendStatus.DISCONNECTED,
        )
        
        self._connected = False
        self._initialized = False
        
        logger.debug(f"Initializing {self.__class__.__name__} backend")
    
    @abc.abstractmethod
    def connect(self) -> bool:
        """
        Connect to the backend.
        
        Returns:
            True if connection successful, False otherwise
        
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abc.abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the backend.
        
        Returns:
            True if disconnection successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag by key.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag if found, None otherwise
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    @abc.abstractmethod
    def set(self, key: str, flag: FeatureFlag) -> bool:
        """
        Set a feature flag.
        
        Args:
            key: Feature flag key
            flag: FeatureFlag to store
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    @abc.abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a feature flag.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    @abc.abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if a feature flag exists.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if exists, False otherwise
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    @abc.abstractmethod
    def get_all(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.
        
        Returns:
            Dictionary of all feature flags
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    @abc.abstractmethod
    def clear(self) -> bool:
        """
        Clear all feature flags.
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            OperationError: If operation fails
        """
        pass
    
    # Optional methods (provide default implementations)
    
    def get_many(self, keys: List[str]) -> Dict[str, Optional[FeatureFlag]]:
        """
        Get multiple feature flags at once.
        
        Args:
            keys: List of feature flag keys
        
        Returns:
            Dictionary mapping keys to FeatureFlags (or None if not found)
        
        Raises:
            OperationError: If operation fails
        """
        result = {}
        for key in keys:
            result[key] = self.get(key)
        return result
    
    def set_many(self, items: Dict[str, FeatureFlag]) -> Dict[str, bool]:
        """
        Set multiple feature flags at once.
        
        Args:
            items: Dictionary mapping keys to FeatureFlags
        
        Returns:
            Dictionary mapping keys to success status
        
        Raises:
            OperationError: If operation fails
        """
        result = {}
        for key, flag in items.items():
            result[key] = self.set(key, flag)
        return result
    
    def delete_many(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple feature flags at once.
        
        Args:
            keys: List of feature flag keys to delete
        
        Returns:
            Dictionary mapping keys to success status
        
        Raises:
            OperationError: If operation fails
        """
        result = {}
        for key in keys:
            result[key] = self.delete(key)
        return result
    
    def search(self, pattern: str) -> Dict[str, FeatureFlag]:
        """
        Search for feature flags by pattern.
        
        Args:
            pattern: Search pattern (backend-specific)
        
        Returns:
            Dictionary of matching feature flags
        
        Raises:
            OperationError: If operation fails
            NotImplementedError: If backend doesn't support search
        """
        raise NotImplementedError("Search not implemented for this backend")
    
    def watch(self, key: str, callback) -> bool:
        """
        Watch for changes to a feature flag.
        
        Args:
            key: Feature flag key to watch
            callback: Function to call when flag changes
        
        Returns:
            True if watch established, False otherwise
        
        Raises:
            NotImplementedError: If backend doesn't support watches
        """
        raise NotImplementedError("Watch not implemented for this backend")
    
    def unwatch(self, key: str) -> bool:
        """
        Stop watching a feature flag.
        
        Args:
            key: Feature flag key to stop watching
        
        Returns:
            True if watch removed, False otherwise
        
        Raises:
            NotImplementedError: If backend doesn't support watches
        """
        raise NotImplementedError("Unwatch not implemented for this backend")
    
    # Context manager support
    
    def __enter__(self):
        """Enter context manager."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.disconnect()
    
    # Async support (optional)
    
    async def connect_async(self) -> bool:
        """Async version of connect."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def disconnect_async(self) -> bool:
        """Async version of disconnect."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def get_async(self, key: str) -> Optional[FeatureFlag]:
        """Async version of get."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def set_async(self, key: str, flag: FeatureFlag) -> bool:
        """Async version of set."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def delete_async(self, key: str) -> bool:
        """Async version of delete."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def exists_async(self, key: str) -> bool:
        """Async version of exists."""
        raise NotImplementedError("Async not supported by this backend")
    
    async def get_all_async(self) -> Dict[str, FeatureFlag]:
        """Async version of get_all."""
        raise NotImplementedError("Async not supported by this backend")
    
    # Utility methods
    
    def serialize(self, flag: FeatureFlag) -> str:
        """Serialize feature flag for storage."""
        flag_dict = flag.to_dict()
        
        if self.config.serializer == "json":
            return json.dumps(flag_dict, default=str)
        elif self.config.serializer == "pickle":
            import pickle
            return pickle.dumps(flag_dict)
        elif self.config.serializer == "msgpack":
            import msgpack
            return msgpack.packb(flag_dict, default=str)
        else:
            raise ConfigurationError(f"Unsupported serializer: {self.config.serializer}")
    
    def deserialize(self, data: str) -> FeatureFlag:
        """Deserialize feature flag from storage."""
        if self.config.serializer == "json":
            flag_dict = json.loads(data)
        elif self.config.serializer == "pickle":
            import pickle
            flag_dict = pickle.loads(data.encode() if isinstance(data, str) else data)
        elif self.config.serializer == "msgpack":
            import msgpack
            flag_dict = msgpack.unpackb(data.encode() if isinstance(data, str) else data)
        else:
            raise ConfigurationError(f"Unsupported serializer: {self.config.serializer}")
        
        return create_flag_from_dict(flag_dict)
    
    def get_full_key(self, key: str) -> str:
        """Get full key with prefix."""
        return f"{self.config.key_prefix}{key}"
    
    def remove_key_prefix(self, full_key: str) -> str:
        """Remove key prefix from full key."""
        if full_key.startswith(self.config.key_prefix):
            return full_key[len(self.config.key_prefix):]
        return full_key
    
    # Health and monitoring
    
    def health_check(self) -> bool:
        """
        Perform health check on backend.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try a simple operation to test connectivity
            self.exists("__health_check__")
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {self.__class__.__name__}: {e}")
            return False
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics."""
        return self._stats
    
    def reset_stats(self):
        """Reset backend statistics."""
        self._stats = BackendStats(
            backend_type=self.__class__.__name__,
            status=self._stats.status,
            connected_since=self._stats.connected_since,
        )
    
    def is_connected(self) -> bool:
        """Check if backend is connected."""
        return self._connected
    
    def get_status(self) -> BackendStatus:
        """Get backend connection status."""
        return self._stats.status
    
    # Transaction support (optional)
    
    def begin_transaction(self):
        """
        Begin a transaction.
        
        Raises:
            NotImplementedError: If backend doesn't support transactions
        """
        raise NotImplementedError("Transactions not supported by this backend")
    
    def commit_transaction(self) -> bool:
        """
        Commit current transaction.
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            NotImplementedError: If backend doesn't support transactions
        """
        raise NotImplementedError("Transactions not supported by this backend")
    
    def rollback_transaction(self) -> bool:
        """
        Rollback current transaction.
        
        Returns:
            True if successful, False otherwise
        
        Raises:
            NotImplementedError: If backend doesn't support transactions
        """
        raise NotImplementedError("Transactions not supported by this backend")
    
    # Iterator support
    
    def keys(self) -> Iterator[str]:
        """
        Iterate over all feature flag keys.
        
        Returns:
            Iterator of feature flag keys
        
        Raises:
            NotImplementedError: If backend doesn't support iteration
        """
        all_flags = self.get_all()
        return iter(all_flags.keys())
    
    def values(self) -> Iterator[FeatureFlag]:
        """
        Iterate over all feature flag values.
        
        Returns:
            Iterator of FeatureFlag objects
        
        Raises:
            NotImplementedError: If backend doesn't support iteration
        """
        all_flags = self.get_all()
        return iter(all_flags.values())
    
    def items(self) -> Iterator[tuple[str, FeatureFlag]]:
        """
        Iterate over all feature flag key-value pairs.
        
        Returns:
            Iterator of (key, FeatureFlag) tuples
        
        Raises:
            NotImplementedError: If backend doesn't support iteration
        """
        all_flags = self.get_all()
        return iter(all_flags.items())
    
    def __len__(self) -> int:
        """Get number of feature flags."""
        return len(self.get_all())
    
    def __contains__(self, key: str) -> bool:
        """Check if feature flag exists."""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> FeatureFlag:
        """Get feature flag by key."""
        flag = self.get(key)
        if flag is None:
            raise KeyError(f"Feature flag not found: {key}")
        return flag
    
    def __setitem__(self, key: str, flag: FeatureFlag):
        """Set feature flag."""
        self.set(key, flag)
    
    def __delitem__(self, key: str):
        """Delete feature flag."""
        if not self.delete(key):
            raise KeyError(f"Failed to delete feature flag: {key}")
    
    # Debug and diagnostic methods
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get backend diagnostics information."""
        return {
            "backend_type": self.__class__.__name__,
            "config": self.config.to_dict(),
            "stats": self._stats.to_dict(),
            "connected": self._connected,
            "capabilities": {
                "persistence": self.supports_persistence,
                "watches": self.supports_watches,
                "transactions": self.supports_transactions,
                "distributed": self.is_distributed,
                "batch_operations": self.supports_batch_operations,
            },
            "health_check": self.health_check(),
        }


# Mixin classes for common functionality

class RetryMixin:
    """Mixin for retry logic."""
    
    def _execute_with_retry(self, operation, *args, **kwargs):
        """
        Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
        
        Returns:
            Operation result
        
        Raises:
            OperationError: If all retries fail
        """
        import time
        
        last_exception = None
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.config.retry_attempts:
                    delay = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.debug(
                        f"Operation failed, retrying in {delay:.2f}s "
                        f"(attempt {attempt + 1}/{self.config.retry_attempts + 1}): {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Operation failed after {attempt + 1} attempts: {e}")
        
        raise OperationError(
            f"Operation failed after {self.config.retry_attempts + 1} attempts",
            backend_type=self.__class__.__name__,
            cause=last_exception,
        )


class MetricsMixin:
    """Mixin for metrics collection."""
    
    def _record_operation(self, operation_name: str, success: bool, start_time: float):
        """Record operation metrics."""
        if not self.config.enable_metrics:
            return
        
        latency_ms = (datetime.utcnow().timestamp() - start_time) * 1000
        self._stats.record_operation(success, latency_ms)
        
        if success:
            logger.debug(f"Operation {operation_name} succeeded in {latency_ms:.2f}ms")
        else:
            logger.warning(f"Operation {operation_name} failed after {latency_ms:.2f}ms")


class ConnectionPoolMixin:
    """Mixin for connection pool management."""
    
    def _get_connection(self):
        """Get a connection from the pool."""
        # Implementation depends on specific backend
        pass
    
    def _release_connection(self, connection):
        """Release a connection back to the pool."""
        # Implementation depends on specific backend
        pass
    
    def _close_all_connections(self):
        """Close all connections in the pool."""
        # Implementation depends on specific backend
        pass