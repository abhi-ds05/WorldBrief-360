# app/cache/multi_level_backend.py
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultilevelCacheBackend:
    """
    Multi-level cache implementation with read-through and write-through strategies.
    
    Cache levels are ordered from fastest (L1) to slowest (L3).
    Supports TTL (Time To Live) for cache entries.
    """
    
    def __init__(self, levels: List[Any]):
        """
        Initialize multi-level cache.
        
        Args:
            levels: List of cache backend instances in order from fastest to slowest.
                   Each backend should implement get(), set(), and delete() methods.
        """
        if not levels:
            raise ValueError("At least one cache level is required")
        
        self.levels = levels
        self.level_names = self._infer_level_names()
        logger.info(f"Initialized multi-level cache with {len(levels)} levels: {self.level_names}")
    
    def _infer_level_names(self) -> List[str]:
        """Infer names for each cache level based on their class names."""
        names = []
        for level in self.levels:
            cls_name = level.__class__.__name__
            # Try to extract meaningful name
            if 'Memory' in cls_name or 'Local' in cls_name:
                names.append('Memory')
            elif 'Redis' in cls_name:
                names.append('Redis')
            elif 'File' in cls_name or 'Disk' in cls_name:
                names.append('File')
            else:
                names.append(cls_name)
        return names
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache hierarchy (read-through).
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        # Try each cache level from fastest to slowest
        for i, (cache, level_name) in enumerate(zip(self.levels, self.level_names)):
            try:
                value = await cache.get(key)
                if value is not None:
                    # Cache hit at level i
                    logger.debug(f"Cache hit at {level_name} (L{i+1}) for key: {key}")
                    
                    # Refresh faster caches (cache promotion)
                    await self._promote_to_faster_caches(key, value, i)
                    
                    # Update metrics if available
                    self._record_hit(i)
                    return value
                    
            except Exception as e:
                logger.warning(f"Error reading from {level_name} cache (L{i+1}) for key {key}: {e}")
                continue
        
        # Cache miss at all levels
        logger.debug(f"Cache miss for key: {key}")
        self._record_miss()
        return default
    
    async def _promote_to_faster_caches(self, key: str, value: Any, hit_level: int) -> None:
        """
        Promote cache entry to faster cache levels.
        
        Args:
            key: Cache key
            value: Cache value
            hit_level: Level where cache was hit (0 = fastest)
        """
        if hit_level == 0:
            return  # Already in fastest cache
        
        # Propagate to faster caches (excluding the one we hit)
        tasks = []
        for i in range(hit_level):
            try:
                # Get TTL from the source cache if available
                ttl = await self._get_ttl_from_cache(self.levels[hit_level], key)
                task = self.levels[i].set(key, value, ttl=ttl)
                tasks.append(task)
            except Exception as e:
                logger.debug(f"Failed to promote to L{i+1}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _get_ttl_from_cache(self, cache: Any, key: str) -> Optional[int]:
        """Try to get remaining TTL from cache backend if supported."""
        try:
            # Check if cache has get_ttl method
            if hasattr(cache, 'get_ttl'):
                return await cache.get_ttl(key)
            
            # Check if cache has get_with_ttl method
            if hasattr(cache, 'get_with_ttl'):
                result = await cache.get_with_ttl(key)
                if isinstance(result, tuple) and len(result) == 2:
                    return result[1]  # Assuming (value, ttl) format
            
            # Try to get from Redis-like cache
            if hasattr(cache, 'pttl'):
                ttl_ms = await cache.pttl(key)
                if ttl_ms > 0:
                    return ttl_ms // 1000  # Convert ms to seconds
        except Exception:
            pass
        
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in all cache levels (write-through).
        
        Args:
            key: Cache key
            value: Cache value
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        # Write to all cache levels concurrently
        tasks = []
        for i, (cache, level_name) in enumerate(zip(self.levels, self.level_names)):
            try:
                task = cache.set(key, value, ttl=ttl)
                tasks.append((task, i, level_name))
            except Exception as e:
                logger.warning(f"Failed to set in {level_name} cache (L{i+1}): {e}")
                success = False
        
        # Execute all set operations
        for task, i, level_name in tasks:
            try:
                await task
                logger.debug(f"Set in {level_name} cache (L{i+1}) for key: {key}")
            except Exception as e:
                logger.warning(f"Error setting in {level_name} cache (L{i+1}): {e}")
                success = False
        
        return success
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from all cache levels.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful in at least one level
        """
        success = False
        
        # Delete from all cache levels concurrently
        tasks = []
        for cache in self.levels:
            try:
                task = cache.delete(key)
                tasks.append(task)
            except Exception as e:
                logger.warning(f"Failed to schedule delete: {e}")
        
        # Execute all delete operations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, bool) and result:
                success = True
            elif not isinstance(result, Exception):
                success = True
        
        logger.debug(f"Deleted key from all cache levels: {key}")
        return success
    
    async def clear(self) -> bool:
        """
        Clear all cache levels.
        
        Returns:
            True if successful
        """
        success = True
        
        tasks = []
        for i, (cache, level_name) in enumerate(zip(self.levels, self.level_names)):
            try:
                task = cache.clear()
                tasks.append((task, i, level_name))
            except Exception as e:
                logger.warning(f"Failed to clear {level_name} cache (L{i+1}): {e}")
                success = False
        
        for task, i, level_name in tasks:
            try:
                await task
                logger.info(f"Cleared {level_name} cache (L{i+1})")
            except Exception as e:
                logger.warning(f"Error clearing {level_name} cache (L{i+1}): {e}")
                success = False
        
        return success
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in any cache level.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists in any cache level
        """
        for i, (cache, level_name) in enumerate(zip(self.levels, self.level_names)):
            try:
                if hasattr(cache, 'exists'):
                    exists = await cache.exists(key)
                else:
                    # Fallback: try to get the value
                    value = await cache.get(key)
                    exists = value is not None
                
                if exists:
                    logger.debug(f"Key exists in {level_name} cache (L{i+1}): {key}")
                    return True
            except Exception as e:
                logger.debug(f"Error checking existence in {level_name} cache: {e}")
                continue
        
        return False
    
    async def get_with_level(self, key: str) -> tuple[Optional[Any], Optional[str]]:
        """
        Get value and the cache level where it was found.
        
        Args:
            key: Cache key
            
        Returns:
            Tuple of (value, level_name) or (None, None)
        """
        for i, (cache, level_name) in enumerate(zip(self.levels, self.level_names)):
            try:
                value = await cache.get(key)
                if value is not None:
                    # Promote to faster caches
                    await self._promote_to_faster_caches(key, value, i)
                    return value, level_name
            except Exception as e:
                logger.debug(f"Error reading from {level_name} cache: {e}")
                continue
        
        return None, None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "levels": len(self.levels),
            "level_names": self.level_names,
            "backend_types": [type(level).__name__ for level in self.levels]
        }
    
    def _record_hit(self, level: int) -> None:
        """Record cache hit (for metrics/logging)."""
        # Can be extended with actual metrics collection
        pass
    
    def _record_miss(self) -> None:
        """Record cache miss (for metrics/logging)."""
        # Can be extended with actual metrics collection
        pass
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all cache levels.
        
        Returns:
            Dictionary mapping level names to health status
        """
        health_status = {}
        
        for level_name, cache in zip(self.level_names, self.levels):
            try:
                # Try a simple operation to check health
                if hasattr(cache, 'health_check'):
                    healthy = await cache.health_check()
                else:
                    # Try to set and get a test key
                    test_key = f"__health_check_{id(self)}"
                    await cache.set(test_key, "test", ttl=1)
                    value = await cache.get(test_key)
                    healthy = value == "test"
                    await cache.delete(test_key)
                
                health_status[level_name] = healthy
            except Exception as e:
                logger.warning(f"Health check failed for {level_name}: {e}")
                health_status[level_name] = False
        
        return health_status