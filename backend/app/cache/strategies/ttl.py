"""
Time-To-Live (TTL) cache strategy.
Automatically expires cache entries after a specified time.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
from collections import OrderedDict

from app.cache.backends import get_cache_backend

logger = logging.getLogger(__name__)


class TTLStrategy:
    """
    TTL-based cache strategy with automatic expiration.
    
    Features:
    - Automatic expiration based on TTL
    - TTL refresh on access (optional)
    - Grace period for stale data
    - Batch operations with consistent TTL
    """
    
    def __init__(self, cache_backend=None, default_ttl: int = 300):
        """
        Initialize TTL strategy.
        
        Args:
            cache_backend: Cache backend instance
            default_ttl: Default TTL in seconds
        """
        self.cache = cache_backend or get_cache_backend()
        self.default_ttl = default_ttl
        self._ttl_overrides: Dict[str, int] = {}
        self._refresh_on_access: Dict[str, bool] = {}
        self._grace_periods: Dict[str, int] = {}
        
        logger.info(f"TTL strategy initialized with default TTL: {default_ttl}s")
    
    def set_ttl_override(self, key_pattern: str, ttl: int):
        """
        Set custom TTL for keys matching pattern.
        
        Args:
            key_pattern: Key pattern (supports wildcards)
            ttl: TTL in seconds
        """
        self._ttl_overrides[key_pattern] = ttl
        logger.debug(f"TTL override set: {key_pattern} -> {ttl}s")
    
    def set_refresh_on_access(self, key_pattern: str, refresh: bool = True):
        """
        Configure whether to refresh TTL on access.
        
        Args:
            key_pattern: Key pattern
            refresh: Whether to refresh TTL on access
        """
        self._refresh_on_access[key_pattern] = refresh
        logger.debug(f"Refresh on access set: {key_pattern} -> {refresh}")
    
    def set_grace_period(self, key_pattern: str, grace_seconds: int):
        """
        Set grace period for stale data.
        
        Args:
            key_pattern: Key pattern
            grace_seconds: Grace period in seconds
        """
        self._grace_periods[key_pattern] = grace_seconds
        logger.debug(f"Grace period set: {key_pattern} -> {grace_seconds}s")
    
    def _get_ttl_for_key(self, key: str) -> int:
        """
        Get TTL for a specific key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds
        """
        # Check for pattern matches
        for pattern, ttl in self._ttl_overrides.items():
            if self._pattern_matches(key, pattern):
                return ttl
        
        return self.default_ttl
    
    def _should_refresh_on_access(self, key: str) -> bool:
        """
        Check if TTL should be refreshed on access.
        
        Args:
            key: Cache key
            
        Returns:
            True if should refresh on access
        """
        for pattern, refresh in self._refresh_on_access.items():
            if self._pattern_matches(key, pattern):
                return refresh
        
        return False
    
    def _get_grace_period(self, key: str) -> int:
        """
        Get grace period for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Grace period in seconds
        """
        for pattern, grace in self._grace_periods.items():
            if self._pattern_matches(key, pattern):
                return grace
        
        return 0
    
    def _pattern_matches(self, key: str, pattern: str) -> bool:
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
    
    async def get(
        self,
        key: str,
        fetch_func: Optional[Callable] = None,
        allow_stale: bool = False,
    ) -> Any:
        """
        Get value from cache with TTL handling.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data if not in cache or stale
            allow_stale: Allow returning stale data during grace period
            
        Returns:
            Cached value or fetched data
        """
        # Check if value exists and get TTL
        exists = await self.cache.exists(key)
        ttl = await self.cache.ttl(key)
        
        if exists and ttl > 0:
            # Value exists and hasn't expired
            value = await self.cache.get(key)
            
            # Refresh TTL on access if configured
            if self._should_refresh_on_access(key):
                new_ttl = self._get_ttl_for_key(key)
                await self.cache.expire(key, new_ttl)
            
            logger.debug(f"TTL cache hit: {key} (TTL: {ttl}s)")
            return value
        
        elif exists and allow_stale:
            # Check if in grace period
            grace_period = self._get_grace_period(key)
            if ttl >= -grace_period:  # Negative TTL means expired
                value = await self.cache.get(key)
                logger.debug(f"TTL cache hit (stale): {key} (TTL: {ttl}s)")
                
                # Refresh in background if fetch_func provided
                if fetch_func:
                    asyncio.create_task(self._refresh_in_background(key, fetch_func))
                
                return value
        
        # Cache miss or expired
        if fetch_func:
            return await self.set_with_fetch(key, fetch_func)
        
        return None
    
    async def set_with_fetch(
        self,
        key: str,
        fetch_func: Callable,
        force_refresh: bool = False,
    ) -> Any:
        """
        Set cache value by fetching data.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data
            force_refresh: Force refresh even if data exists
            
        Returns:
            Fetched data
        """
        # Check if we should use existing data
        if not force_refresh:
            exists = await self.cache.exists(key)
            ttl = await self.cache.ttl(key)
            
            if exists and ttl > 0:
                value = await self.cache.get(key)
                logger.debug(f"Using existing cache: {key}")
                return value
        
        # Fetch new data
        logger.debug(f"Fetching data for: {key}")
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
        except Exception as e:
            logger.error(f"Error fetching data for {key}: {e}")
            
            # Try to return stale data as fallback
            if await self.cache.exists(key):
                stale_value = await self.cache.get(key)
                logger.warning(f"Using stale data for {key} due to fetch error")
                return stale_value
            
            raise
        
        # Store in cache with TTL
        ttl = self._get_ttl_for_key(key)
        await self.cache.set(key, value, ttl=ttl)
        
        logger.debug(f"Cached data for: {key} (TTL: {ttl}s)")
        return value
    
    async def _refresh_in_background(self, key: str, fetch_func: Callable):
        """Refresh cache value in background."""
        try:
            await self.set_with_fetch(key, fetch_func, force_refresh=True)
            logger.debug(f"Background refresh completed for: {key}")
        except Exception as e:
            logger.error(f"Background refresh failed for {key}: {e}")
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses strategy default if None)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self._get_ttl_for_key(key)
        
        success = await self.cache.set(key, value, ttl=ttl)
        
        if success:
            logger.debug(f"Set cache with TTL: {key} -> {ttl}s")
        
        return success
    
    async def set_many(
        self,
        items: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set multiple values with consistent TTL.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: TTL in seconds (uses strategy default if None)
            
        Returns:
            True if successful
        """
        if not items:
            return True
        
        # Determine TTL for each key
        items_with_ttl = {}
        for key, value in items.items():
            key_ttl = ttl or self._get_ttl_for_key(key)
            items_with_ttl[key] = (value, key_ttl)
        
        # Set using pipeline for efficiency
        try:
            async with self.cache.pipeline(transaction=True) as pipe:
                for key, (value, key_ttl) in items_with_ttl.items():
                    if key_ttl > 0:
                        pipe.setex(key, key_ttl, value)
                    else:
                        pipe.set(key, value)
                
                await pipe.execute()
            
            logger.debug(f"Set {len(items)} items with TTL strategy")
            return True
            
        except Exception as e:
            logger.error(f"Error setting multiple items: {e}")
            return False
    
    async def get_many(
        self,
        keys: List[str],
        fetch_func: Optional[Callable[[List[str]], Dict[str, Any]]] = None,
        allow_stale: bool = False,
    ) -> Dict[str, Any]:
        """
        Get multiple values with TTL handling.
        
        Args:
            keys: List of cache keys
            fetch_func: Function to fetch missing data
            allow_stale: Allow returning stale data
            
        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}
        
        # Get all values at once
        values = await self.cache.get_many(keys)
        
        # Check TTL for each key
        result = {}
        missing_keys = []
        stale_keys = []
        
        for key in keys:
            value = values.get(key)
            ttl = await self.cache.ttl(key)
            
            if value is not None and ttl > 0:
                # Valid cache entry
                result[key] = value
                
                # Refresh TTL if configured
                if self._should_refresh_on_access(key):
                    new_ttl = self._get_ttl_for_key(key)
                    await self.cache.expire(key, new_ttl)
                    
            elif value is not None and allow_stale:
                # Check grace period
                grace_period = self._get_grace_period(key)
                if ttl >= -grace_period:
                    result[key] = value
                    stale_keys.append(key)
                    
            else:
                # Missing or expired
                missing_keys.append(key)
        
        # Fetch missing data if function provided
        if missing_keys and fetch_func:
            try:
                if asyncio.iscoroutinefunction(fetch_func):
                    fetched = await fetch_func(missing_keys)
                else:
                    fetched = fetch_func(missing_keys)
                
                # Cache fetched data
                for key in missing_keys:
                    if key in fetched:
                        value = fetched[key]
                        result[key] = value
                        
                        ttl = self._get_ttl_for_key(key)
                        await self.cache.set(key, value, ttl=ttl)
                        
                logger.debug(f"Fetched {len(fetched)} missing items")
                
            except Exception as e:
                logger.error(f"Error fetching missing data: {e}")
        
        # Background refresh for stale data
        if stale_keys and fetch_func:
            for key in stale_keys:
                asyncio.create_task(self._refresh_in_background(
                    key,
                    lambda k=key: fetch_func([k])[k] if fetch_func else None
                ))
        
        logger.debug(f"TTL get_many: {len(result)} found, {len(missing_keys)} missing")
        return result
    
    async def cleanup_expired(self, pattern: str = "*") -> int:
        """
        Clean up expired cache entries.
        
        Args:
            pattern: Key pattern to clean
            
        Returns:
            Number of entries cleaned
        """
        # Redis automatically expires keys, but we can clean up manually if needed
        # This is mostly useful for other backends or for reporting
        
        keys = await self.cache.keys(pattern)
        expired_count = 0
        
        for key in keys:
            ttl = await self.cache.ttl(key)
            if ttl == -2:  # Key doesn't exist (shouldn't happen with keys())
                await self.cache.delete(key)
                expired_count += 1
            elif ttl < 0:  # Expired
                await self.cache.delete(key)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired cache entries")
        
        return expired_count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get TTL strategy statistics.
        
        Returns:
            Statistics dictionary
        """
        total_keys = len(await self.cache.keys("*"))
        
        # Sample keys to get TTL distribution
        sample_keys = await self.cache.scan_keys(count=100)
        ttl_samples = []
        
        for key in sample_keys:
            ttl = await self.cache.ttl(key)
            if ttl > 0:
                ttl_samples.append(ttl)
        
        return {
            "total_keys": total_keys,
            "sample_size": len(ttl_samples),
            "ttl_stats": {
                "min": min(ttl_samples) if ttl_samples else 0,
                "max": max(ttl_samples) if ttl_samples else 0,
                "avg": sum(ttl_samples) / len(ttl_samples) if ttl_samples else 0,
            } if ttl_samples else {},
            "config": {
                "default_ttl": self.default_ttl,
                "ttl_overrides": len(self._ttl_overrides),
                "refresh_patterns": len(self._refresh_on_access),
                "grace_periods": len(self._grace_periods),
            }
        }