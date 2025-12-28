"""
Read-Through cache strategy for WorldBrief 360.

Read-Through is a caching pattern where:
1. Application reads from cache first
2. If cache miss, data is loaded from data source
3. Data is then written to cache for future reads
4. All reads go through cache (transparent to application)
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import hashlib
import json

from app.cache.backends import get_cache_backend

logger = logging.getLogger(__name__)


@dataclass
class ReadThroughEntry:
    """Metadata for read-through cache entry."""
    key: str
    value: Any
    loaded_at: datetime
    loaded_from: str  # cache, datasource, fallback
    load_duration: float
    ttl: Optional[int]
    hits: int = 1
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key": self.key,
            "loaded_at": self.loaded_at.isoformat(),
            "loaded_from": self.loaded_from,
            "load_duration": self.load_duration,
            "ttl": self.ttl,
            "hits": self.hits,
            "last_accessed": self.last_accessed.isoformat(),
        }


class ReadThroughStrategy:
    """
    Read-Through cache strategy with automatic data loading.
    
    Features:
    - Transparent cache population on misses
    - Batch loading for multiple cache misses
    - Cache stampede prevention (dog-piling)
    - Stale-while-revalidate pattern
    - Graceful degradation on data source failures
    - Load shedding and rate limiting
    - Cache warming and preloading
    """
    
    def __init__(
        self,
        cache_backend=None,
        default_ttl: int = 300,
        stale_ttl: int = 60,
        batch_size: int = 100,
        max_concurrent_loads: int = 10,
        enable_stale_while_revalidate: bool = True,
        enable_load_shedding: bool = True,
        load_shedding_threshold: float = 0.8,  # 80% load
    ):
        """
        Initialize Read-Through strategy.
        
        Args:
            cache_backend: Cache backend instance
            default_ttl: Default TTL in seconds
            stale_ttl: TTL for stale data while revalidating
            batch_size: Batch size for loading multiple keys
            max_concurrent_loads: Maximum concurrent data source loads
            enable_stale_while_revalidate: Enable stale data serving
            enable_load_shedding: Enable load shedding under high load
            load_shedding_threshold: Load threshold for shedding (0-1)
        """
        self.cache = cache_backend or get_cache_backend()
        self.default_ttl = default_ttl
        self.stale_ttl = stale_ttl
        self.batch_size = batch_size
        self.max_concurrent_loads = max_concurrent_loads
        self.enable_stale_while_revalidate = enable_stale_while_revalidate
        self.enable_load_shedding = enable_load_shedding
        self.load_shedding_threshold = load_shedding_threshold
        
        # Load management
        self._loading_locks: Dict[str, asyncio.Lock] = {}
        self._loading_semaphore = asyncio.Semaphore(max_concurrent_loads)
        self._load_metrics: Dict[str, List[float]] = {
            "load_times": [],
            "cache_hits": [],
            "cache_misses": [],
        }
        self._current_loads = 0
        self._max_loads_seen = 0
        
        # Entry tracking
        self._entry_metadata: Dict[str, ReadThroughEntry] = {}
        self._key_patterns: Dict[str, Callable] = {}
        
        logger.info(
            f"Read-Through strategy initialized: "
            f"default_ttl={default_ttl}s, "
            f"batch_size={batch_size}, "
            f"max_concurrent={max_concurrent_loads}"
        )
    
    def register_key_pattern(
        self,
        pattern: str,
        load_func: Callable[[List[str]], Dict[str, Any]],
        ttl: Optional[int] = None,
        batch_enabled: bool = True,
    ):
        """
        Register a key pattern with its loading function.
        
        Args:
            pattern: Key pattern (e.g., "user:*", "product:*")
            load_func: Function to load data for keys matching pattern
            ttl: TTL for this pattern (overrides default)
            batch_enabled: Enable batch loading for this pattern
        """
        self._key_patterns[pattern] = {
            "load_func": load_func,
            "ttl": ttl,
            "batch_enabled": batch_enabled,
            "registered_at": datetime.utcnow(),
        }
        
        logger.info(f"Registered key pattern: {pattern}")
    
    def _get_pattern_for_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern configuration for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Pattern configuration or None
        """
        for pattern, config in self._key_patterns.items():
            if self._key_matches_pattern(key, pattern):
                return config
        
        return None
    
    def _key_matches_pattern(self, key: str, pattern: str) -> bool:
        """
        Check if key matches pattern.
        
        Args:
            key: Cache key
            pattern: Pattern with wildcards
            
        Returns:
            True if key matches pattern
        """
        if pattern == "*":
            return True
        
        if "*" in pattern:
            # Simple wildcard matching
            pattern_parts = pattern.split("*")
            key_lower = key.lower()
            
            for part in pattern_parts:
                if part and part not in key_lower:
                    return False
            
            return True
        
        return key == pattern
    
    async def _get_loading_lock(self, key: str) -> asyncio.Lock:
        """
        Get or create loading lock for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Async lock for the key
        """
        if key not in self._loading_locks:
            self._loading_locks[key] = asyncio.Lock()
        return self._loading_locks[key]
    
    async def _should_load_shed(self) -> bool:
        """
        Determine if load shedding should be activated.
        
        Returns:
            True if load shedding should be activated
        """
        if not self.enable_load_shedding:
            return False
        
        # Simple load calculation
        load_ratio = self._current_loads / self.max_concurrent_loads
        return load_ratio >= self.load_shedding_threshold
    
    async def get(
        self,
        key: str,
        load_func: Optional[Callable[[str], Any]] = None,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
        allow_stale: bool = True,
    ) -> Any:
        """
        Get value using read-through pattern.
        
        Args:
            key: Cache key
            load_func: Function to load data if not in cache
            ttl: TTL for loaded data
            force_refresh: Force refresh from data source
            allow_stale: Allow returning stale data
            
        Returns:
            Value from cache or data source
        
        Raises:
            ValueError: If key not in cache and no load_func provided
        """
        start_time = time.time()
        
        # Check for load shedding
        if await self._should_load_shed():
            logger.warning(f"Load shedding activated for key: {key}")
            # In load shedding mode, try cache only
            if allow_stale:
                cached = await self.cache.get(key)
                if cached is not None:
                    return cached
            # Fall through to regular load if cache miss
    
        # Try to get from cache first (unless forcing refresh)
        if not force_refresh:
            cached = await self.cache.get(key)
            
            if cached is not None:
                # Cache hit
                await self._record_cache_hit(key, start_time)
                return cached
        
        # Cache miss or force refresh
        lock = await self._get_loading_lock(key)
        
        async with lock:
            # Check cache again after acquiring lock (double-check)
            if not force_refresh:
                cached = await self.cache.get(key)
                if cached is not None:
                    await self._record_cache_hit(key, start_time)
                    return cached
            
            # Determine load function
            pattern_config = self._get_pattern_for_key(key)
            if load_func is None and pattern_config:
                # Use pattern-based load function
                async def pattern_load(k):
                    batch_result = await self._load_batch([k], pattern_config)
                    return batch_result.get(k)
                actual_load_func = pattern_load
            elif load_func:
                actual_load_func = load_func
            else:
                raise ValueError(f"No load function provided for key: {key}")
            
            # Load from data source with concurrency control
            async with self._loading_semaphore:
                self._current_loads += 1
                self._max_loads_seen = max(self._max_loads_seen, self._current_loads)
                
                try:
                    # Load data
                    load_start = time.time()
                    
                    if asyncio.iscoroutinefunction(actual_load_func):
                        value = await actual_load_func(key)
                    else:
                        value = actual_load_func(key)
                    
                    load_duration = time.time() - load_start
                    
                    if value is not None:
                        # Determine TTL
                        if ttl is None and pattern_config:
                            ttl = pattern_config.get("ttl")
                        if ttl is None:
                            ttl = self.default_ttl
                        
                        # Store in cache
                        await self.cache.set(key, value, ttl=ttl)
                        
                        # Record metadata
                        self._entry_metadata[key] = ReadThroughEntry(
                            key=key,
                            value=value,
                            loaded_at=datetime.utcnow(),
                            loaded_from="datasource",
                            load_duration=load_duration,
                            ttl=ttl,
                        )
                        
                        logger.debug(
                            f"Read-Through loaded: {key} "
                            f"(duration: {load_duration:.3f}s, ttl: {ttl}s)"
                        )
                    
                    await self._record_cache_miss(key, start_time, load_duration)
                    return value
                    
                except Exception as e:
                    logger.error(f"Load failed for key {key}: {e}")
                    
                    # Try to return stale data if allowed
                    if allow_stale and self.enable_stale_while_revalidate:
                        stale_data = await self.cache.get(key)
                        if stale_data is not None:
                            logger.warning(f"Returning stale data for {key} due to load failure")
                            
                            # Refresh in background
                            asyncio.create_task(self._refresh_in_background(key, actual_load_func))
                            
                            return stale_data
                    
                    raise
                    
                finally:
                    self._current_loads -= 1
    
    async def get_many(
        self,
        keys: List[str],
        load_func: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
        ttl: Optional[int] = None,
        force_refresh: bool = False,
        allow_stale: bool = True,
        batch_load: bool = True,
    ) -> Dict[str, Any]:
        """
        Get multiple values using read-through pattern.
        
        Args:
            keys: List of cache keys
            load_func: Function to load multiple items
            ttl: TTL for loaded data
            force_refresh: Force refresh from data source
            allow_stale: Allow returning stale data
            batch_load: Enable batch loading
            
        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}
        
        start_time = time.time()
        
        # Try to get from cache first
        cached = {}
        missing_keys = []
        
        if not force_refresh:
            cached = await self.cache.get_many(keys)
            missing_keys = [k for k in keys if k not in cached]
        else:
            missing_keys = keys
        
        # Return if all found in cache
        if not missing_keys:
            for key in keys:
                if key in cached:
                    await self._record_cache_hit(key, start_time)
            return cached
        
        # Determine load function for missing keys
        if load_func is None:
            # Try to group by pattern
            grouped_keys = self._group_keys_by_pattern(missing_keys)
            
            if grouped_keys and batch_load:
                # Use pattern-based batch loading
                return await self._batch_load_by_pattern(
                    keys=keys,
                    cached=cached,
                    grouped_keys=grouped_keys,
                    ttl=ttl,
                    allow_stale=allow_stale,
                    start_time=start_time,
                )
        
        # Load missing keys
        loaded = await self._load_missing_keys(
            keys=missing_keys,
            load_func=load_func,
            ttl=ttl,
            allow_stale=allow_stale,
            batch_load=batch_load,
        )
        
        # Combine results
        result = {**cached, **loaded}
        
        # Record metrics
        for key in keys:
            if key in cached:
                await self._record_cache_hit(key, start_time)
            elif key in loaded:
                await self._record_cache_miss(key, start_time)
        
        return result
    
    def _group_keys_by_pattern(self, keys: List[str]) -> Dict[str, List[str]]:
        """
        Group keys by their matching patterns.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of pattern -> list of keys
        """
        grouped = {}
        
        for key in keys:
            pattern_config = self._get_pattern_for_key(key)
            if pattern_config:
                # Find the actual pattern
                for pattern, config in self._key_patterns.items():
                    if self._key_matches_pattern(key, pattern):
                        if pattern not in grouped:
                            grouped[pattern] = []
                        grouped[pattern].append(key)
                        break
        
        return grouped
    
    async def _batch_load_by_pattern(
        self,
        keys: List[str],
        cached: Dict[str, Any],
        grouped_keys: Dict[str, List[str]],
        ttl: Optional[int],
        allow_stale: bool,
        start_time: float,
    ) -> Dict[str, Any]:
        """
        Load missing keys using pattern-based batch loading.
        
        Args:
            keys: Original keys requested
            cached: Already cached values
            grouped_keys: Keys grouped by pattern
            ttl: TTL for loaded data
            allow_stale: Allow stale data
            start_time: Start time for metrics
            
        Returns:
            Dictionary of all values
        """
        result = cached.copy()
        
        # Load each pattern group
        for pattern, pattern_keys in grouped_keys.items():
            pattern_config = self._key_patterns[pattern]
            
            if pattern_config["batch_enabled"]:
                # Load batch for this pattern
                loaded = await self._load_batch(pattern_keys, pattern_config, ttl)
                result.update(loaded)
            else:
                # Load individually
                for key in pattern_keys:
                    value = await self.get(
                        key=key,
                        ttl=ttl or pattern_config.get("ttl"),
                        force_refresh=True,
                        allow_stale=allow_stale,
                    )
                    if value is not None:
                        result[key] = value
        
        return result
    
    async def _load_batch(
        self,
        keys: List[str],
        pattern_config: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load a batch of keys for a pattern.
        
        Args:
            keys: List of keys to load
            pattern_config: Pattern configuration
            ttl: TTL for loaded data
            
        Returns:
            Dictionary of loaded values
        """
        if not keys:
            return {}
        
        load_func = pattern_config["load_func"]
        pattern_ttl = pattern_config.get("ttl")
        
        try:
            # Acquire locks for all keys to prevent stampede
            locks = [await self._get_loading_lock(key) for key in keys]
            for lock in locks:
                await lock.acquire()
            
            try:
                # Check cache again after acquiring locks
                cached = await self.cache.get_many(keys)
                still_missing = [k for k in keys if k not in cached]
                
                if not still_missing:
                    return cached
                
                # Load missing keys from data source
                load_start = time.time()
                
                if asyncio.iscoroutinefunction(load_func):
                    loaded = await load_func(still_missing)
                else:
                    loaded = load_func(still_missing)
                
                load_duration = time.time() - load_start
                
                # Store in cache
                if loaded:
                    actual_ttl = ttl or pattern_ttl or self.default_ttl
                    
                    await self.cache.set_many(loaded, ttl=actual_ttl)
                    
                    # Record metadata
                    for key, value in loaded.items():
                        self._entry_metadata[key] = ReadThroughEntry(
                            key=key,
                            value=value,
                            loaded_at=datetime.utcnow(),
                            loaded_from="datasource_batch",
                            load_duration=load_duration / len(loaded),
                            ttl=actual_ttl,
                        )
                    
                    logger.debug(
                        f"Batch loaded {len(loaded)}/{len(still_missing)} keys "
                        f"for pattern (duration: {load_duration:.3f}s)"
                    )
                
                # Combine cached and loaded
                result = {**cached, **loaded}
                return result
                
            finally:
                # Release all locks
                for lock in locks:
                    lock.release()
                    
        except Exception as e:
            logger.error(f"Batch load failed for {len(keys)} keys: {e}")
            return {}
    
    async def _load_missing_keys(
        self,
        keys: List[str],
        load_func: Optional[Callable],
        ttl: Optional[int],
        allow_stale: bool,
        batch_load: bool,
    ) -> Dict[str, Any]:
        """
        Load missing keys using the provided load function.
        
        Args:
            keys: Missing keys to load
            load_func: Load function
            ttl: TTL for loaded data
            allow_stale: Allow stale data
            batch_load: Enable batch loading
            
        Returns:
            Dictionary of loaded values
        """
        if not keys or load_func is None:
            return {}
        
        if batch_load and len(keys) > 1:
            # Try batch loading
            try:
                load_start = time.time()
                
                if asyncio.iscoroutinefunction(load_func):
                    loaded = await load_func(keys)
                else:
                    loaded = load_func(keys)
                
                load_duration = time.time() - load_start
                
                # Store in cache
                if loaded:
                    await self.cache.set_many(loaded, ttl=ttl)
                    
                    # Record metadata
                    for key, value in loaded.items():
                        self._entry_metadata[key] = ReadThroughEntry(
                            key=key,
                            value=value,
                            loaded_at=datetime.utcnow(),
                            loaded_from="datasource_custom_batch",
                            load_duration=load_duration / len(loaded),
                            ttl=ttl,
                        )
                
                return loaded
                
            except Exception as e:
                logger.error(f"Custom batch load failed: {e}")
                # Fall back to individual loading
        
        # Load individually
        loaded = {}
        
        for key in keys:
            try:
                value = await self.get(
                    key=key,
                    load_func=lambda k=key: load_func([k])[k] if load_func else None,
                    ttl=ttl,
                    force_refresh=True,
                    allow_stale=allow_stale,
                )
                if value is not None:
                    loaded[key] = value
            except Exception as e:
                logger.error(f"Individual load failed for {key}: {e}")
        
        return loaded
    
    async def _refresh_in_background(self, key: str, load_func: Callable):
        """
        Refresh cache entry in background.
        
        Args:
            key: Cache key
            load_func: Load function
        """
        try:
            # Get current value to check TTL
            current_ttl = await self.cache.ttl(key)
            
            if current_ttl > 0:
                # Set stale TTL to allow serving while refreshing
                await self.cache.expire(key, self.stale_ttl)
                
                # Refresh in background
                async def refresh():
                    try:
                        if asyncio.iscoroutinefunction(load_func):
                            new_value = await load_func(key)
                        else:
                            new_value = load_func(key)
                        
                        if new_value is not None:
                            await self.cache.set(key, new_value, ttl=self.default_ttl)
                            logger.debug(f"Background refresh completed for {key}")
                    except Exception as e:
                        logger.error(f"Background refresh failed for {key}: {e}")
                
                asyncio.create_task(refresh())
                
        except Exception as e:
            logger.error(f"Error setting up background refresh for {key}: {e}")
    
    async def _record_cache_hit(self, key: str, start_time: float):
        """Record cache hit metrics."""
        duration = time.time() - start_time
        
        # Update entry metadata
        if key in self._entry_metadata:
            entry = self._entry_metadata[key]
            entry.hits += 1
            entry.last_accessed = datetime.utcnow()
        
        # Record metrics
        self._load_metrics["cache_hits"].append(duration)
        
        # Keep only recent metrics
        for metric in self._load_metrics.values():
            if len(metric) > 1000:
                metric.pop(0)
    
    async def _record_cache_miss(self, key: str, start_time: float, load_duration: float = 0):
        """Record cache miss metrics."""
        duration = time.time() - start_time
        
        # Record metrics
        self._load_metrics["cache_misses"].append(duration)
        if load_duration > 0:
            self._load_metrics["load_times"].append(load_duration)
        
        # Keep only recent metrics
        for metric in self._load_metrics.values():
            if len(metric) > 1000:
                metric.pop(0)
    
    async def warm_cache(
        self,
        keys: List[str],
        load_func: Callable[[List[str]], Dict[str, Any]],
        ttl: Optional[int] = None,
        batch_size: int = None,
        concurrency: int = None,
    ) -> Dict[str, bool]:
        """
        Warm cache by preloading keys.
        
        Args:
            keys: List of keys to warm
            load_func: Function to load keys
            ttl: TTL for loaded data
            batch_size: Batch size for loading
            concurrency: Maximum concurrent batches
            
        Returns:
            Dictionary of key -> success status
        """
        if not keys:
            return {}
        
        batch_size = batch_size or self.batch_size
        concurrency = concurrency or self.max_concurrent_loads
        
        results = {}
        semaphore = asyncio.Semaphore(concurrency)
        
        async def warm_batch(batch_keys: List[str]) -> Dict[str, bool]:
            async with semaphore:
                try:
                    # Load batch
                    if asyncio.iscoroutinefunction(load_func):
                        loaded = await load_func(batch_keys)
                    else:
                        loaded = load_func(batch_keys)
                    
                    # Store in cache
                    if loaded:
                        await self.cache.set_many(loaded, ttl=ttl)
                    
                    # Record results
                    batch_results = {}
                    for key in batch_keys:
                        batch_results[key] = key in loaded
                    
                    return batch_results
                    
                except Exception as e:
                    logger.error(f"Cache warming batch failed: {e}")
                    return {key: False for key in batch_keys}
        
        # Process in batches
        tasks = []
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            tasks.append(warm_batch(batch))
        
        # Wait for all batches
        batch_results = await asyncio.gather(*tasks)
        
        # Combine results
        for batch_result in batch_results:
            results.update(batch_result)
        
        logger.info(f"Cache warming completed: {sum(results.values())}/{len(keys)} successful")
        return results
    
    async def prefetch(
        self,
        key_patterns: List[str],
        load_func: Callable[[List[str]], Dict[str, Any]],
        estimate_count: int = 100,
        ttl: Optional[int] = None,
    ) -> int:
        """
        Prefetch keys matching patterns.
        
        Args:
            key_patterns: List of key patterns to prefetch
            estimate_count: Estimated number of keys to prefetch
            load_func: Function to load keys
            ttl: TTL for loaded data
            
        Returns:
            Number of keys prefetched
        """
        # This is a simplified implementation
        # In production, you might want to scan existing data
        # or use a predictive algorithm
        
        logger.info(f"Prefetching keys matching patterns: {key_patterns}")
        
        # For now, just return 0 since we don't have actual keys
        # In real implementation, you would:
        # 1. Scan data source for keys matching patterns
        # 2. Load them in batches
        # 3. Store in cache
        
        return 0
    
    async def invalidate(
        self,
        key: str,
        pattern: bool = False,
        load_func: Optional[Callable] = None,
    ) -> bool:
        """
        Invalidate cache entry with optional reload.
        
        Args:
            key: Cache key or pattern
            pattern: Whether key is a pattern
            load_func: Optional function to reload data
            
        Returns:
            True if successful
        """
        try:
            if pattern:
                # Delete all keys matching pattern
                keys = await self.cache.keys(key)
                for k in keys:
                    await self.cache.delete(k)
                    if k in self._entry_metadata:
                        del self._entry_metadata[k]
                
                logger.info(f"Invalidated pattern: {key} ({len(keys)} keys)")
                return True
            else:
                # Delete single key
                success = await self.cache.delete(key)
                
                if success and key in self._entry_metadata:
                    del self._entry_metadata[key]
                
                # Reload if function provided
                if success and load_func:
                    asyncio.create_task(self.get(key, load_func=load_func))
                
                return success
                
        except Exception as e:
            logger.error(f"Invalidation failed for {key}: {e}")
            return False
    
    async def get_entry_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Entry metadata or None
        """
        if key in self._entry_metadata:
            return self._entry_metadata[key].to_dict()
        return None
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get read-through strategy statistics.
        
        Returns:
            Statistics dictionary
        """
        # Calculate metrics
        hit_count = len(self._load_metrics["cache_hits"])
        miss_count = len(self._load_metrics["cache_misses"])
        total_ops = hit_count + miss_count
        
        hit_rate = hit_count / total_ops if total_ops > 0 else 0
        
        avg_hit_time = (
            sum(self._load_metrics["cache_hits"]) / hit_count
            if hit_count > 0 else 0
        )
        avg_miss_time = (
            sum(self._load_metrics["cache_misses"]) / miss_count
            if miss_count > 0 else 0
        )
        avg_load_time = (
            sum(self._load_metrics["load_times"]) / len(self._load_metrics["load_times"])
            if self._load_metrics["load_times"] else 0
        )
        
        return {
            "performance": {
                "total_operations": total_ops,
                "hit_count": hit_count,
                "miss_count": miss_count,
                "hit_rate": hit_rate,
                "avg_hit_time_ms": avg_hit_time * 1000,
                "avg_miss_time_ms": avg_miss_time * 1000,
                "avg_load_time_ms": avg_load_time * 1000,
            },
            "load_management": {
                "current_loads": self._current_loads,
                "max_loads_seen": self._max_loads_seen,
                "max_concurrent_loads": self.max_concurrent_loads,
                "load_shedding_enabled": self.enable_load_shedding,
                "load_shedding_threshold": self.load_shedding_threshold,
            },
            "cache_state": {
                "tracked_entries": len(self._entry_metadata),
                "registered_patterns": len(self._key_patterns),
                "loading_locks": len(self._loading_locks),
            },
            "configuration": {
                "default_ttl": self.default_ttl,
                "stale_ttl": self.stale_ttl,
                "batch_size": self.batch_size,
                "enable_stale_while_revalidate": self.enable_stale_while_revalidate,
            }
        }
    
    async def clear_metadata(self):
        """Clear all metadata (for testing/maintenance)."""
        self._entry_metadata.clear()
        self._loading_locks.clear()
        self._load_metrics = {
            "load_times": [],
            "cache_hits": [],
            "cache_misses": [],
        }
        self._current_loads = 0
        self._max_loads_seen = 0
        
        logger.info("Cleared read-through metadata")
    
    async def close(self):
        """Cleanup resources."""
        await self.clear_metadata()
        logger.info("Read-through strategy closed")


# Factory function for easy creation
def create_read_through_strategy(
    cache_backend=None,
    default_ttl: int = 300,
    **kwargs
) -> ReadThroughStrategy:
    """
    Create a read-through cache strategy.
    
    Args:
        cache_backend: Cache backend instance
        default_ttl: Default TTL in seconds
        **kwargs: Additional arguments for ReadThroughStrategy
        
    Returns:
        ReadThroughStrategy instance
    """
    return ReadThroughStrategy(
        cache_backend=cache_backend,
        default_ttl=default_ttl,
        **kwargs
    )


# Example usage
async def example_usage():
    """Example of using read-through strategy."""
    
    # Create strategy
    strategy = ReadThroughStrategy(default_ttl=3600)
    
    # Register pattern
    async def load_users(user_ids: List[str]) -> Dict[str, Any]:
        """Load users from database."""
        # Simulate database call
        await asyncio.sleep(0.1)
        return {uid: {"id": uid, "name": f"User {uid}"} for uid in user_ids}
    
    strategy.register_key_pattern(
        pattern="user:*",
        load_func=load_users,
        ttl=7200,
        batch_enabled=True,
    )
    
    # Use strategy
    try:
        # Get single user (will load if not in cache)
        user1 = await strategy.get("user:123")
        print(f"User 1: {user1}")
        
        # Get multiple users
        users = await strategy.get_many(["user:123", "user:456", "user:789"])
        print(f"Users: {users}")
        
        # Get stats
        stats = await strategy.get_stats()
        print(f"Stats: {stats}")
        
        # Warm cache
        warm_results = await strategy.warm_cache(
            keys=["user:111", "user:222", "user:333"],
            load_func=load_users,
        )
        print(f"Warm results: {warm_results}")
        
    finally:
        await strategy.close()


if __name__ == "__main__":
    asyncio.run(example_usage())