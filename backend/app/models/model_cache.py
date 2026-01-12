"""
Model caching system to reduce load times and memory usage.
Implements LRU (Least Recently Used) caching strategy for loaded models.
"""
import asyncio
import hashlib
import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union
from typing import List, Callable

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


class CacheStrategy(Enum):
    """Caching strategies for models."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live


class CacheEvictionPolicy(Enum):
    """Eviction policies for cache items."""
    ON_MAX_SIZE = "on_max_size"
    ON_MEMORY_PRESSURE = "on_memory_pressure"
    ON_IDLE_TIMEOUT = "on_idle_timeout"


@dataclass
class CacheItem(Generic[T]):
    """Represents an item in the cache."""
    key: str
    value: T
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def access(self) -> None:
        """Update access statistics."""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()
    
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now() - self.last_accessed).total_seconds()


class ModelCache:
    """
    Cache for ML models to reduce loading times.
    Supports multiple caching strategies and eviction policies.
    """
    
    def __init__(
        self,
        max_size_mb: int = 1024,  # 1GB default
        max_items: int = 10,
        strategy: CacheStrategy = CacheStrategy.LRU,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.ON_MAX_SIZE,
        ttl_seconds: Optional[int] = 3600,  # 1 hour default
        idle_timeout_seconds: Optional[int] = 1800,  # 30 minutes
        persist_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the model cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            max_items: Maximum number of items to cache
            strategy: Caching strategy (LRU, LFU, FIFO, TTL)
            eviction_policy: When to evict items
            ttl_seconds: Time to live for cached items (for TTL strategy)
            idle_timeout_seconds: Idle timeout before eviction
            persist_dir: Directory to persist cache to disk (optional)
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_items = max_items
        self.strategy = strategy
        self.eviction_policy = eviction_policy
        self.ttl_seconds = ttl_seconds
        self.idle_timeout_seconds = idle_timeout_seconds
        
        # Cache storage
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._current_size_bytes = 0
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Disk persistence
        self.persist_dir = Path(persist_dir) if persist_dir else None
        if self.persist_dir:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_lock = threading.Lock()
        
        # Start background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
        
        logger.info(
            f"ModelCache initialized: "
            f"max_size={max_size_mb}MB, "
            f"max_items={max_items}, "
            f"strategy={strategy.value}"
        )
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self.eviction_policy in [
            CacheEvictionPolicy.ON_IDLE_TIMEOUT,
            CacheEvictionPolicy.ON_MEMORY_PRESSURE
        ]:
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                daemon=True,
                name="ModelCache-Cleanup"
            )
            self._cleanup_thread.start()
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        cleanup_interval = 60  # Check every minute
        
        while not self._stop_cleanup.is_set():
            time.sleep(cleanup_interval)
            
            try:
                with self._cleanup_lock:
                    self._perform_cleanup()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _perform_cleanup(self) -> None:
        """Perform cache cleanup based on eviction policy."""
        if self.eviction_policy == CacheEvictionPolicy.ON_IDLE_TIMEOUT:
            self._evict_idle_items()
        elif self.eviction_policy == CacheEvictionPolicy.ON_MEMORY_PRESSURE:
            self._check_memory_pressure()
    
    def _evict_idle_items(self) -> None:
        """Evict items that have been idle for too long."""
        if not self.idle_timeout_seconds:
            return
        
        with self._lock:
            items_to_evict = []
            
            for key, item in list(self._cache.items()):
                if item.idle_seconds() > self.idle_timeout_seconds:
                    items_to_evict.append(key)
            
            for key in items_to_evict:
                self._evict_item(key, reason="idle_timeout")
    
    def _check_memory_pressure(self) -> None:
        """Check system memory pressure and evict if necessary."""
        # This is a simplified implementation
        # In production, you might use psutil or similar
        memory_usage_percent = self._current_size_bytes / self.max_size_bytes
        
        if memory_usage_percent > 0.9:  # 90% full
            logger.warning(f"Cache memory pressure: {memory_usage_percent:.1%}")
            self._evict_items(count=2, reason="memory_pressure")
    
    def _generate_key(self, model_config: Dict[str, Any]) -> str:
        """
        Generate a unique cache key from model configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Unique cache key
        """
        # Sort config to ensure consistent keys
        sorted_config = sorted(
            (k, v if not isinstance(v, dict) else self._generate_key(v))
            for k, v in model_config.items()
        )
        
        config_str = str(sorted_config).encode('utf-8')
        return hashlib.sha256(config_str).hexdigest()[:32]
    
    def get(
        self,
        model_config: Dict[str, Any],
        load_func: Optional[Callable[[Dict[str, Any]], T]] = None
    ) -> Optional[T]:
        """
        Get a model from cache, or load it if not present.
        
        Args:
            model_config: Model configuration
            load_func: Function to load the model if not in cache
            
        Returns:
            The model or None if not found and no load_func provided
        """
        key = self._generate_key(model_config)
        
        with self._lock:
            if key in self._cache:
                # Cache hit
                item = self._cache[key]
                item.access()
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self._cache.move_to_end(key)
                
                self._hit_count += 1
                logger.debug(f"Cache hit for key: {key}")
                return item.value
            
            # Cache miss
            self._miss_count += 1
            logger.debug(f"Cache miss for key: {key}")
            
            if load_func is None:
                return None
            
            # Load the model
            try:
                model = load_func(model_config)
                self.put(model_config, model)
                return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
    
    async def get_async(
        self,
        model_config: Dict[str, Any],
        load_func: Optional[Callable[[Dict[str, Any]], T]] = None
    ) -> Optional[T]:
        """Async version of get."""
        # Run sync version in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.get(model_config, load_func)
        )
    
    def put(self, model_config: Dict[str, Any], model: T) -> str:
        """
        Put a model into the cache.
        
        Args:
            model_config: Model configuration
            model: The model to cache
            
        Returns:
            Cache key
        """
        key = self._generate_key(model_config)
        
        with self._lock:
            # Check if already in cache
            if key in self._cache:
                logger.debug(f"Model already in cache: {key}")
                return key
            
            # Estimate size (simplified - should be more accurate in production)
            size_bytes = self._estimate_size(model)
            
            # Check if we need to evict before adding
            self._check_and_evict_if_needed(size_bytes)
            
            # Create cache item
            item = CacheItem(
                key=key,
                value=model,
                size_bytes=size_bytes,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                metadata={
                    "config": model_config,
                    "estimated_size_mb": size_bytes / (1024 * 1024)
                }
            )
            
            # Add to cache
            self._cache[key] = item
            self._current_size_bytes += size_bytes
            
            # For LRU, new items go to the end
            if self.strategy == CacheStrategy.LRU:
                self._cache.move_to_end(key)
            
            logger.debug(
                f"Cached model: {key}, "
                f"size={size_bytes / (1024 * 1024):.2f}MB, "
                f"total_cache_size={self._current_size_bytes / (1024 * 1024):.2f}MB"
            )
            
            # Persist to disk if configured
            if self.persist_dir:
                self._persist_item(key, item)
            
            return key
    
    def _estimate_size(self, model: Any) -> int:
        """
        Estimate the memory size of a model.
        
        Args:
            model: The model to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            # Try to get actual size
            import sys
            size = sys.getsizeof(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            
            # Add some overhead for safety
            return int(size * 1.1)
        except (pickle.PickleError, AttributeError):
            # Fallback estimation
            logger.warning("Could not estimate model size accurately, using default")
            return 100 * 1024 * 1024  # 100MB default
    
    def _check_and_evict_if_needed(self, new_item_size: int) -> None:
        """Check cache limits and evict items if necessary."""
        # Check if adding this item would exceed max size
        if self._current_size_bytes + new_item_size > self.max_size_bytes:
            # Calculate how much we need to free
            need_to_free = (self._current_size_bytes + new_item_size) - self.max_size_bytes
            self._evict_by_size(need_to_free)
        
        # Check if we've reached max items
        if len(self._cache) >= self.max_items:
            items_to_evict = len(self._cache) - self.max_items + 1
            self._evict_items(items_to_evict)
    
    def _evict_by_size(self, need_to_free_bytes: int) -> None:
        """Evict items until we've freed enough space."""
        freed_bytes = 0
        
        while freed_bytes < need_to_free_bytes and self._cache:
            # Get item to evict based on strategy
            if self.strategy == CacheStrategy.LRU:
                key, item = next(iter(self._cache.items()))  # Oldest (first)
            elif self.strategy == CacheStrategy.LFU:
                # Find item with lowest access count
                key, item = min(self._cache.items(), key=lambda x: x[1].access_count)
            elif self.strategy == CacheStrategy.FIFO:
                key, item = next(iter(self._cache.items()))  # First in
            elif self.strategy == CacheStrategy.TTL:
                # Find item with oldest creation time
                key, item = min(self._cache.items(), key=lambda x: x[1].created_at)
            else:
                key, item = next(iter(self._cache.items()))
            
            freed_bytes += item.size_bytes
            self._evict_item(key, reason="size_limit")
    
    def _evict_items(self, count: int = 1, reason: str = "manual") -> None:
        """Evict a specific number of items."""
        for _ in range(min(count, len(self._cache))):
            if self.strategy == CacheStrategy.LRU:
                key = next(iter(self._cache.keys()))
            elif self.strategy == CacheStrategy.LFU:
                key = min(self._cache.items(), key=lambda x: x[1].access_count)[0]
            elif self.strategy == CacheStrategy.FIFO:
                key = next(iter(self._cache.keys()))
            elif self.strategy == CacheStrategy.TTL:
                key = min(self._cache.items(), key=lambda x: x[1].created_at)[0]
            else:
                key = next(iter(self._cache.keys()))
            
            self._evict_item(key, reason)
    
    def _evict_item(self, key: str, reason: str = "manual") -> None:
        """Evict a specific item from cache."""
        with self._lock:
            if key not in self._cache:
                return
            
            item = self._cache.pop(key)
            self._current_size_bytes -= item.size_bytes
            self._eviction_count += 1
            
            logger.debug(
                f"Evicted item {key} ({reason}): "
                f"age={item.age_seconds():.0f}s, "
                f"accesses={item.access_count}"
            )
            
            # Clean up persistent storage
            if self.persist_dir:
                self._remove_persisted_item(key)
    
    def _persist_item(self, key: str, item: CacheItem) -> None:
        """Persist a cache item to disk."""
        try:
            file_path = self.persist_dir / f"{key}.cache"
            
            # Create a serializable version
            persist_data = {
                'key': item.key,
                'value': item.value,  # Note: model must be serializable
                'size_bytes': item.size_bytes,
                'created_at': item.created_at,
                'last_accessed': item.last_accessed,
                'access_count': item.access_count,
                'metadata': item.metadata
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(persist_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Failed to persist cache item {key}: {e}")
    
    def _remove_persisted_item(self, key: str) -> None:
        """Remove persisted cache item from disk."""
        try:
            file_path = self.persist_dir / f"{key}.cache"
            if file_path.exists():
                file_path.unlink()
        except Exception as e:
            logger.error(f"Failed to remove persisted cache item {key}: {e}")
    
    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            logger.info(f"Clearing cache with {len(self._cache)} items")
            
            # Clear persistent storage
            if self.persist_dir:
                for file in self.persist_dir.glob("*.cache"):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.error(f"Failed to remove cache file {file}: {e}")
            
            self._cache.clear()
            self._current_size_bytes = 0
    
    def remove(self, model_config: Dict[str, Any]) -> bool:
        """
        Remove a specific model from cache.
        
        Args:
            model_config: Model configuration
            
        Returns:
            True if removed, False if not found
        """
        key = self._generate_key(model_config)
        
        with self._lock:
            if key in self._cache:
                self._evict_item(key, reason="manual_remove")
                return True
            return False
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_items = len(self._cache)
            total_hits = self._hit_count
            total_misses = self._miss_count
            total_requests = total_hits + total_misses
            
            hit_rate = total_hits / total_requests if total_requests > 0 else 0
            miss_rate = total_misses / total_requests if total_requests > 0 else 0
            
            # Get item statistics
            item_stats = []
            for key, item in self._cache.items():
                item_stats.append({
                    'key': key[:8] + '...',  # Truncate for readability
                    'size_mb': item.size_bytes / (1024 * 1024),
                    'age_seconds': item.age_seconds(),
                    'idle_seconds': item.idle_seconds(),
                    'access_count': item.access_count,
                    'model_type': item.metadata.get('config', {}).get('model_type', 'unknown')
                })
            
            return {
                'total_items': total_items,
                'current_size_mb': self._current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'usage_percent': (self._current_size_bytes / self.max_size_bytes * 100)
                                if self.max_size_bytes > 0 else 0,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'total_evictions': self._eviction_count,
                'hit_rate': hit_rate,
                'miss_rate': miss_rate,
                'strategy': self.strategy.value,
                'eviction_policy': self.eviction_policy.value,
                'items': item_stats
            }
    
    def get_keys(self) -> List[str]:
        """Get all cache keys."""
        with self._lock:
            return list(self._cache.keys())
    
    def contains(self, model_config: Dict[str, Any]) -> bool:
        """Check if a model is in cache."""
        key = self._generate_key(model_config)
        with self._lock:
            return key in self._cache
    
    def __contains__(self, model_config: Dict[str, Any]) -> bool:
        """Check if a model is in cache using 'in' operator."""
        return self.contains(model_config)
    
    def __len__(self) -> int:
        """Get number of items in cache."""
        with self._lock:
            return len(self._cache)
    
    def __repr__(self) -> str:
        """String representation of cache."""
        stats = self.stats()
        return (
            f"ModelCache(items={stats['total_items']}, "
            f"size={stats['current_size_mb']:.1f}MB/{stats['max_size_mb']:.0f}MB, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )
    
    def shutdown(self) -> None:
        """Shutdown the cache and cleanup resources."""
        logger.info("Shutting down ModelCache")
        self._stop_cleanup.set()
        
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        # Clear cache
        self.clear()


# Singleton instance for global cache
_global_cache: Optional[ModelCache] = None


def get_global_cache(**kwargs) -> ModelCache:
    """
    Get or create global model cache instance.
    
    Args:
        **kwargs: Cache configuration parameters
        
    Returns:
        Global ModelCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = ModelCache(**kwargs)
    
    return _global_cache


def clear_global_cache() -> None:
    """Clear the global cache."""
    global _global_cache
    
    if _global_cache:
        _global_cache.clear()


def shutdown_global_cache() -> None:
    """Shutdown the global cache."""
    global _global_cache
    
    if _global_cache:
        _global_cache.shutdown()
        _global_cache = None