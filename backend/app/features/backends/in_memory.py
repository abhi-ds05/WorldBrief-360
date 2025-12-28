"""
In-memory feature flag backend.
Fast, non-persistent backend for development and testing.
"""

import threading
import time
import json
from typing import Dict, Any, Optional, List, Set, Iterator,logger
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import OrderedDict

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
from ..flags import FeatureFlag, create_flag_from_dict


@dataclass
class InMemoryConfig(BackendConfig):
    """Configuration for in-memory backend."""
    
    # Memory management
    max_size: Optional[int] = None  # Maximum number of flags to store
    eviction_policy: str = "lru"  # lru, fifo, random
    cleanup_interval: int = 300  # Cleanup interval in seconds
    
    # Performance
    enable_compression: bool = False
    enable_indexing: bool = True
    
    # Serialization format for internal storage
    storage_format: str = "dict"  # dict, json
    
    def validate(self):
        """Validate configuration."""
        super().validate()
        
        if self.max_size is not None and self.max_size <= 0:
            raise ValueError("max_size must be positive or None")
        
        if self.eviction_policy not in ["lru", "fifo", "random"]:
            raise ValueError(f"Invalid eviction policy: {self.eviction_policy}")
        
        if self.storage_format not in ["dict", "json"]:
            raise ValueError(f"Invalid storage format: {self.storage_format}")


class InMemoryBackend(FeatureBackend, RetryMixin, MetricsMixin):
    """
    In-memory feature flag backend.
    
    Features:
    - Fast in-memory storage (no persistence)
    - LRU/FIFO/Random eviction policies
    - Thread-safe operations
    - Optional size limits
    - Statistics and metrics
    """
    
    supports_persistence = False
    supports_watches = True
    supports_transactions = True
    is_distributed = False
    supports_batch_operations = True
    
    def __init__(self, config: InMemoryConfig):
        """
        Initialize in-memory backend.
        
        Args:
            config: InMemoryConfig instance
        """
        super().__init__(config)
        self.config = config
        
        # Storage structures
        self._store: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._access_order: OrderedDict = OrderedDict()
        
        # Index for faster searches
        self._index: Dict[str, Set[str]] = {}  # field_value -> set of keys
        
        # Watches
        self._watches: Dict[str, List] = {}
        
        # Transactions
        self._transaction_stack: List[Dict[str, Any]] = []
        self._transaction_changes: List[Dict[str, Any]] = []
        
        # Cleanup thread
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_running = False
        
        # Locks
        self._store_lock = threading.RLock()
        self._watch_lock = threading.RLock()
        self._transaction_lock = threading.RLock()
        
        # Initialization
        self._initialized = True
        
        logger.debug(f"InMemoryBackend initialized with max_size={config.max_size}")
    
    def connect(self) -> bool:
        """
        Connect to in-memory backend.
        
        Returns:
            Always returns True (in-memory is always "connected")
        """
        with self._store_lock:
            if not self._connected:
                self._connected = True
                self._stats.status = BackendStatus.CONNECTED
                self._stats.connected_since = datetime.utcnow()
                
                # Start cleanup thread if needed
                if self.config.max_size is not None and self.config.cleanup_interval > 0:
                    self._start_cleanup_thread()
                
                logger.info("InMemoryBackend connected")
            
            return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from in-memory backend.
        
        Returns:
            True if successful
        """
        with self._store_lock:
            if self._connected:
                # Stop cleanup thread
                self._stop_cleanup_thread()
                
                # Clear all data if not persisting
                if not self.config.enable_metrics:
                    self._store.clear()
                    self._metadata.clear()
                    self._access_order.clear()
                    self._index.clear()
                    self._watches.clear()
                
                self._connected = False
                self._stats.status = BackendStatus.DISCONNECTED
                
                logger.info("InMemoryBackend disconnected")
            
            return True
    
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag by key.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag if found, None otherwise
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            with self._store_lock:
                if full_key not in self._store:
                    logger.debug(f"Flag not found: {key}")
                    return None
                
                # Update access order for LRU
                if self.config.eviction_policy == "lru":
                    if full_key in self._access_order:
                        self._access_order.move_to_end(full_key)
                
                # Get stored data
                stored_data = self._store[full_key]
                
                # Deserialize based on storage format
                if self.config.storage_format == "json":
                    flag_data = json.loads(stored_data)
                    flag = create_flag_from_dict(flag_data)
                else:
                    # Already a dictionary
                    flag = create_flag_from_dict(stored_data)
                
                success = True
                return flag
            
        except Exception as e:
            logger.error(f"Error getting flag {key}: {e}")
            raise OperationError(f"Failed to get flag: {key}", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("get", success, start_time)
    
    def set(self, key: str, flag: FeatureFlag) -> bool:
        """
        Set a feature flag.
        
        Args:
            key: Feature flag key
            flag: FeatureFlag to store
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            with self._store_lock:
                # Check size limit
                if self.config.max_size is not None:
                    self._enforce_size_limit()
                
                # Store flag data
                if self.config.storage_format == "json":
                    flag_data = flag.to_dict()
                    stored_data = json.dumps(flag_data, default=str)
                else:
                    stored_data = flag.to_dict()
                
                self._store[full_key] = stored_data
                
                # Update metadata
                self._metadata[full_key] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat(),
                    "size": len(str(stored_data)),
                }
                
                # Update access order
                if self.config.eviction_policy == "lru":
                    self._access_order[full_key] = time.time()
                elif self.config.eviction_policy == "fifo":
                    if full_key not in self._access_order:
                        self._access_order[full_key] = time.time()
                
                # Update index if enabled
                if self.config.enable_indexing:
                    self._update_index(full_key, flag)
                
                # Notify watchers
                self._notify_watchers(full_key, flag)
                
                success = True
                return True
            
        except Exception as e:
            logger.error(f"Error setting flag {key}: {e}")
            raise OperationError(f"Failed to set flag: {key}", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("set", success, start_time)
    
    def delete(self, key: str) -> bool:
        """
        Delete a feature flag.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            with self._store_lock:
                if full_key not in self._store:
                    logger.debug(f"Flag not found for deletion: {key}")
                    return False
                
                # Remove from store
                del self._store[full_key]
                
                # Remove from metadata
                if full_key in self._metadata:
                    del self._metadata[full_key]
                
                # Remove from access order
                if full_key in self._access_order:
                    del self._access_order[full_key]
                
                # Remove from index
                if self.config.enable_indexing:
                    self._remove_from_index(full_key)
                
                # Notify watchers of deletion
                self._notify_watchers(full_key, None)
                
                success = True
                return True
            
        except Exception as e:
            logger.error(f"Error deleting flag {key}: {e}")
            raise OperationError(f"Failed to delete flag: {key}", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("delete", success, start_time)
    
    def exists(self, key: str) -> bool:
        """
        Check if a feature flag exists.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if exists
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            with self._store_lock:
                exists = full_key in self._store
                success = True
                return exists
            
        except Exception as e:
            logger.error(f"Error checking existence of flag {key}: {e}")
            raise OperationError(f"Failed to check flag existence: {key}", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("exists", success, start_time)
    
    def get_all(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags.
        
        Returns:
            Dictionary of all feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._store_lock:
                for full_key, stored_data in self._store.items():
                    key = self.remove_key_prefix(full_key)
                    
                    try:
                        if self.config.storage_format == "json":
                            flag_data = json.loads(stored_data)
                            flag = create_flag_from_dict(flag_data)
                        else:
                            flag = create_flag_from_dict(stored_data)
                        
                        result[key] = flag
                    except Exception as e:
                        logger.warning(f"Error deserializing flag {key}: {e}")
                        continue
                
                success = True
                return result
            
        except Exception as e:
            logger.error(f"Error getting all flags: {e}")
            raise OperationError("Failed to get all flags", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("get_all", success, start_time)
    
    def clear(self) -> bool:
        """
        Clear all feature flags.
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            with self._store_lock:
                self._store.clear()
                self._metadata.clear()
                self._access_order.clear()
                self._index.clear()
                
                # Notify all watchers
                with self._watch_lock:
                    for key in list(self._watches.keys()):
                        self._notify_watchers(key, None)
                
                success = True
                return True
            
        except Exception as e:
            logger.error(f"Error clearing flags: {e}")
            raise OperationError("Failed to clear flags", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("clear", success, start_time)
    
    # Override batch operations for better performance
    
    def get_many(self, keys: List[str]) -> Dict[str, Optional[FeatureFlag]]:
        """
        Get multiple feature flags at once.
        
        Args:
            keys: List of feature flag keys
        
        Returns:
            Dictionary mapping keys to FeatureFlags (or None if not found)
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._store_lock:
                for key in keys:
                    full_key = self.get_full_key(key)
                    
                    if full_key in self._store:
                        stored_data = self._store[full_key]
                        
                        # Update access order for LRU
                        if self.config.eviction_policy == "lru":
                            if full_key in self._access_order:
                                self._access_order.move_to_end(full_key)
                        
                        # Deserialize
                        if self.config.storage_format == "json":
                            flag_data = json.loads(stored_data)
                            flag = create_flag_from_dict(flag_data)
                        else:
                            flag = create_flag_from_dict(stored_data)
                        
                        result[key] = flag
                    else:
                        result[key] = None
            
            success = True
            return result
        
        except Exception as e:
            logger.error(f"Error getting multiple flags: {e}")
            raise OperationError("Failed to get multiple flags", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("get_many", success, start_time)
    
    def set_many(self, items: Dict[str, FeatureFlag]) -> Dict[str, bool]:
        """
        Set multiple feature flags at once.
        
        Args:
            items: Dictionary mapping keys to FeatureFlags
        
        Returns:
            Dictionary mapping keys to success status
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._store_lock:
                # Check size limit
                if self.config.max_size is not None:
                    self._enforce_size_limit(len(items))
                
                for key, flag in items.items():
                    try:
                        full_key = self.get_full_key(key)
                        
                        # Store flag data
                        if self.config.storage_format == "json":
                            flag_data = flag.to_dict()
                            stored_data = json.dumps(flag_data, default=str)
                        else:
                            stored_data = flag.to_dict()
                        
                        self._store[full_key] = stored_data
                        
                        # Update metadata
                        self._metadata[full_key] = {
                            "created_at": datetime.utcnow().isoformat(),
                            "updated_at": datetime.utcnow().isoformat(),
                            "size": len(str(stored_data)),
                        }
                        
                        # Update access order
                        if self.config.eviction_policy == "lru":
                            self._access_order[full_key] = time.time()
                        elif self.config.eviction_policy == "fifo":
                            if full_key not in self._access_order:
                                self._access_order[full_key] = time.time()
                        
                        # Update index if enabled
                        if self.config.enable_indexing:
                            self._update_index(full_key, flag)
                        
                        result[key] = True
                        
                        # Notify watchers
                        self._notify_watchers(full_key, flag)
                        
                    except Exception as e:
                        logger.error(f"Error setting flag {key} in batch: {e}")
                        result[key] = False
            
            success = True
            return result
        
        except Exception as e:
            logger.error(f"Error setting multiple flags: {e}")
            raise OperationError("Failed to set multiple flags", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("set_many", success, start_time)
    
    # Search implementation
    
    def search(self, pattern: str) -> Dict[str, FeatureFlag]:
        """
        Search for feature flags by pattern.
        
        Args:
            pattern: Search pattern (supports wildcards: * for any characters)
        
        Returns:
            Dictionary of matching feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._store_lock:
                # Simple pattern matching
                import fnmatch
                
                for full_key, stored_data in self._store.items():
                    key = self.remove_key_prefix(full_key)
                    
                    if fnmatch.fnmatch(key, pattern):
                        try:
                            if self.config.storage_format == "json":
                                flag_data = json.loads(stored_data)
                                flag = create_flag_from_dict(flag_data)
                            else:
                                flag = create_flag_from_dict(stored_data)
                            
                            result[key] = flag
                        except Exception as e:
                            logger.warning(f"Error deserializing flag {key} during search: {e}")
                            continue
            
            success = True
            return result
        
        except Exception as e:
            logger.error(f"Error searching flags: {e}")
            raise OperationError(f"Failed to search flags with pattern: {pattern}", backend_type=self.__class__.__name__, cause=e)
        
        finally:
            self._record_operation("search", success, start_time)
    
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
        full_key = self.get_full_key(key)
        
        with self._watch_lock:
            if full_key not in self._watches:
                self._watches[full_key] = []
            
            self._watches[full_key].append(callback)
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
        full_key = self.get_full_key(key)
        
        with self._watch_lock:
            if full_key in self._watches:
                del self._watches[full_key]
                logger.debug(f"Removed watch for {key}")
                return True
            
            return False
    
    def _notify_watchers(self, full_key: str, flag: Optional[FeatureFlag]):
        """Notify all watchers of a flag change."""
        with self._watch_lock:
            if full_key in self._watches:
                for callback in self._watches[full_key]:
                    try:
                        callback(full_key, flag)
                    except Exception as e:
                        logger.error(f"Error in watch callback for {full_key}: {e}")
    
    # Transaction implementation
    
    def begin_transaction(self):
        """
        Begin a transaction.
        
        Returns:
            Transaction ID
        """
        with self._transaction_lock:
            # Save current state
            snapshot = {
                "store": self._store.copy(),
                "metadata": self._metadata.copy(),
                "access_order": self._access_order.copy(),
                "index": {k: v.copy() for k, v in self._index.items()} if self._index else {},
            }
            
            self._transaction_stack.append(snapshot)
            self._transaction_changes.append([])
            
            transaction_id = len(self._transaction_stack)
            logger.debug(f"Started transaction {transaction_id}")
            
            return transaction_id
    
    def commit_transaction(self) -> bool:
        """
        Commit current transaction.
        
        Returns:
            True if successful
        """
        with self._transaction_lock:
            if not self._transaction_stack:
                logger.warning("No active transaction to commit")
                return False
            
            # Pop transaction stack
            self._transaction_stack.pop()
            changes = self._transaction_changes.pop()
            
            logger.debug(f"Committed transaction with {len(changes)} changes")
            return True
    
    def rollback_transaction(self) -> bool:
        """
        Rollback current transaction.
        
        Returns:
            True if successful
        """
        with self._transaction_lock:
            if not self._transaction_stack:
                logger.warning("No active transaction to rollback")
                return False
            
            # Restore snapshot
            snapshot = self._transaction_stack.pop()
            self._transaction_changes.pop()
            
            with self._store_lock:
                self._store = snapshot["store"]
                self._metadata = snapshot["metadata"]
                self._access_order = snapshot["access_order"]
                self._index = snapshot["index"]
            
            logger.debug("Rolled back transaction")
            return True
    
    # Index management
    
    def _update_index(self, full_key: str, flag: FeatureFlag):
        """Update search index for a flag."""
        if not self.config.enable_indexing:
            return
        
        # Remove old index entries
        self._remove_from_index(full_key)
        
        # Add new index entries
        flag_dict = flag.to_dict()
        
        # Index by flag type
        self._add_to_index("flag_type", flag_dict.get("flag_type"), full_key)
        
        # Index by enabled state
        self._add_to_index("enabled", str(flag_dict.get("enabled")), full_key)
        
        # Index by environment
        for env in flag_dict.get("environments", []):
            self._add_to_index("environment", env, full_key)
        
        # Index by target segments
        for segment in flag_dict.get("target_segments", []):
            self._add_to_index("segment", segment, full_key)
    
    def _add_to_index(self, field: str, value: Any, full_key: str):
        """Add entry to index."""
        if value is None:
            return
        
        index_key = f"{field}:{value}"
        if index_key not in self._index:
            self._index[index_key] = set()
        
        self._index[index_key].add(full_key)
    
    def _remove_from_index(self, full_key: str):
        """Remove flag from all indexes."""
        for index_key in list(self._index.keys()):
            if full_key in self._index[index_key]:
                self._index[index_key].remove(full_key)
                
                # Clean up empty index entries
                if not self._index[index_key]:
                    del self._index[index_key]
    
    def search_by_field(self, field: str, value: Any) -> Dict[str, FeatureFlag]:
        """
        Search flags by field value using index.
        
        Args:
            field: Field name to search
            value: Field value to match
        
        Returns:
            Dictionary of matching flags
        """
        if not self.config.enable_indexing:
            raise OperationError("Indexing is not enabled", backend_type=self.__class__.__name__)
        
        index_key = f"{field}:{value}"
        
        with self._store_lock:
            if index_key not in self._index:
                return {}
            
            result = {}
            for full_key in self._index[index_key]:
                if full_key in self._store:
                    key = self.remove_key_prefix(full_key)
                    stored_data = self._store[full_key]
                    
                    try:
                        if self.config.storage_format == "json":
                            flag_data = json.loads(stored_data)
                            flag = create_flag_from_dict(flag_data)
                        else:
                            flag = create_flag_from_dict(stored_data)
                        
                        result[key] = flag
                    except Exception as e:
                        logger.warning(f"Error deserializing flag {key}: {e}")
                        continue
            
            return result
    
    # Memory management
    
    def _enforce_size_limit(self, additional_items: int = 0):
        """Enforce maximum size limit by evicting items if needed."""
        if self.config.max_size is None:
            return
        
        current_size = len(self._store)
        if current_size + additional_items <= self.config.max_size:
            return
        
        items_to_evict = (current_size + additional_items) - self.config.max_size
        
        with self._store_lock:
            for _ in range(items_to_evict):
                if not self._store:
                    break
                
                if self.config.eviction_policy == "lru":
                    # Evict least recently used
                    full_key, _ = self._access_order.popitem(last=False)
                elif self.config.eviction_policy == "fifo":
                    # Evict first in
                    full_key, _ = self._access_order.popitem(last=False)
                elif self.config.eviction_policy == "random":
                    # Evict random item
                    import random
                    full_key = random.choice(list(self._store.keys()))
                
                # Remove from store
                if full_key in self._store:
                    del self._store[full_key]
                
                # Remove metadata
                if full_key in self._metadata:
                    del self._metadata[full_key]
                
                # Remove from index
                if self.config.enable_indexing:
                    self._remove_from_index(full_key)
                
                # Notify watchers
                self._notify_watchers(full_key, None)
                
                logger.debug(f"Evicted flag {self.remove_key_prefix(full_key)} due to size limit")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        if self._cleanup_thread is not None:
            return
        
        self._cleanup_running = True
        
        def cleanup_worker():
            while self._cleanup_running:
                try:
                    time.sleep(self.config.cleanup_interval)
                    
                    if self.config.max_size is not None:
                        self._enforce_size_limit()
                    
                    # Clean up expired metadata
                    current_time = time.time()
                    with self._store_lock:
                        keys_to_remove = []
                        for full_key, metadata in self._metadata.items():
                            # Example: remove metadata for items not accessed in 24 hours
                            # You can customize this logic
                            pass
                        
                        for full_key in keys_to_remove:
                            if full_key in self._metadata:
                                del self._metadata[full_key]
                
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {e}")
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            name="InMemoryBackend-Cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()
        
        logger.debug("Started cleanup thread")
    
    def _stop_cleanup_thread(self):
        """Stop background cleanup thread."""
        self._cleanup_running = False
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
            self._cleanup_thread = None
            
            logger.debug("Stopped cleanup thread")
    
    # Statistics and diagnostics
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics."""
        stats = super().get_stats()
        
        # Add memory-specific stats
        with self._store_lock:
            stats.extra_stats = {
                "store_size": len(self._store),
                "metadata_size": len(self._metadata),
                "index_size": len(self._index),
                "watches_count": len(self._watches),
                "active_transactions": len(self._transaction_stack),
                "memory_usage": self._estimate_memory_usage(),
            }
        
        return stats
    
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        import sys
        
        total_size = 0
        
        # Estimate store size
        for key, value in self._store.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(value)
        
        # Estimate metadata size
        for key, value in self._metadata.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(value)
        
        # Estimate index size
        for key, value in self._index.items():
            total_size += sys.getsizeof(key)
            total_size += sys.getsizeof(value)
        
        return total_size
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics."""
        diagnostics = super().get_diagnostics()
        
        with self._store_lock:
            diagnostics.update({
                "store_keys": list(self._store.keys()),
                "store_size": len(self._store),
                "metadata_keys": list(self._metadata.keys()),
                "index_fields": list(self._index.keys()),
                "watch_keys": list(self._watches.keys()),
                "transaction_depth": len(self._transaction_stack),
                "access_order_size": len(self._access_order),
                "estimated_memory_bytes": self._estimate_memory_usage(),
            })
        
        return diagnostics
    
    # Special methods
    
    def __len__(self) -> int:
        """Get number of feature flags."""
        return len(self._store)
    
    def __contains__(self, key: str) -> bool:
        """Check if feature flag exists."""
        return self.exists(key)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"InMemoryBackend(store_size={len(self._store)}, connected={self._connected})"
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.disconnect()


# Cache-specific backend (optimized for read-heavy workloads)

class InMemoryCacheBackend(InMemoryBackend):
    """
    In-memory cache backend with TTL support.
    
    Features:
    - Time-to-live (TTL) for cached flags
    - Automatic expiration
    - Write-through caching support
    - Statistics for cache hits/misses
    """
    
    def __init__(self, config: InMemoryConfig, default_ttl: int = 300):
        """
        Initialize cache backend.
        
        Args:
            config: InMemoryConfig instance
            default_ttl: Default TTL in seconds
        """
        super().__init__(config)
        self.default_ttl = default_ttl
        self._expiry_times: Dict[str, float] = {}
        
        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._expirations = 0
    
    def set(self, key: str, flag: FeatureFlag, ttl: Optional[int] = None) -> bool:
        """
        Set a feature flag with TTL.
        
        Args:
            key: Feature flag key
            flag: FeatureFlag to store
            ttl: Time-to-live in seconds (uses default if None)
        
        Returns:
            True if successful
        """
        success = super().set(key, flag)
        
        if success:
            full_key = self.get_full_key(key)
            ttl_value = ttl if ttl is not None else self.default_ttl
            
            if ttl_value is not None and ttl_value > 0:
                self._expiry_times[full_key] = time.time() + ttl_value
            elif full_key in self._expiry_times:
                del self._expiry_times[full_key]
        
        return success
    
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag, checking expiry.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag if found and not expired, None otherwise
        """
        full_key = self.get_full_key(key)
        
        # Check expiry
        if full_key in self._expiry_times:
            if time.time() > self._expiry_times[full_key]:
                # Expired, remove it
                with self._store_lock:
                    if full_key in self._store:
                        del self._store[full_key]
                    if full_key in self._expiry_times:
                        del self._expiry_times[full_key]
                    if full_key in self._metadata:
                        del self._metadata[full_key]
                    
                    self._expirations += 1
                
                self._misses += 1
                return None
        
        # Get from parent
        flag = super().get(key)
        
        if flag is not None:
            self._hits += 1
        else:
            self._misses += 1
        
        return flag
    
    def delete(self, key: str) -> bool:
        """Delete a feature flag and its expiry."""
        full_key = self.get_full_key(key)
        
        if full_key in self._expiry_times:
            del self._expiry_times[full_key]
        
        return super().delete(key)
    
    def clear_expired(self) -> int:
        """
        Clear all expired items.
        
        Returns:
            Number of items cleared
        """
        cleared = 0
        current_time = time.time()
        
        with self._store_lock:
            keys_to_remove = []
            
            for full_key, expiry in self._expiry_times.items():
                if current_time > expiry:
                    keys_to_remove.append(full_key)
            
            for full_key in keys_to_remove:
                if full_key in self._store:
                    del self._store[full_key]
                if full_key in self._metadata:
                    del self._metadata[full_key]
                del self._expiry_times[full_key]
                
                # Remove from index
                if self.config.enable_indexing:
                    self._remove_from_index(full_key)
                
                # Notify watchers
                self._notify_watchers(full_key, None)
                
                cleared += 1
                self._expirations += 1
        
        logger.debug(f"Cleared {cleared} expired items")
        return cleared
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "expirations": self._expirations,
            "hit_ratio": self._hits / max(self._hits + self._misses, 1),
            "expired_count": len([v for v in self._expiry_times.values() if time.time() > v]),
            "total_cached": len(self._store),
        }