"""
Least Recently Used (LRU) cache strategy.
Evicts least recently used items when cache is full.
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import OrderedDict
from dataclasses import dataclass, field

from app.cache.backends import get_cache_backend

logger = logging.getLogger(__name__)


@dataclass
class LRUEntry:
    """Represents an entry in LRU cache."""
    key: str
    value: Any
    size: int
    access_time: float = field(default_factory=time.time)
    access_count: int = 1
    
    @property
    def score(self) -> float:
        """Calculate LRU score (lower = more likely to evict)."""
        # Simple LRU: only based on access time
        return self.access_time
        
        # Alternative: combine recency and frequency
        # return self.access_time / (self.access_count + 1)


class LRUStrategy:
    """
    LRU cache strategy with size limits.
    
    Features:
    - Automatic eviction when size limit reached
    - Configurable size limits (count or memory)
    - Access tracking for recency/frequency
    - Batched operations with LRU awareness
    """
    
    def __init__(
        self,
        cache_backend=None,
        max_size: int = 1000,
        max_memory_mb: Optional[int] = None,
        eviction_policy: str = "lru",  # lru, lfu, arc
    ):
        """
        Initialize LRU strategy.
        
        Args:
            cache_backend: Cache backend instance
            max_size: Maximum number of items
            max_memory_mb: Maximum memory in MB (None for unlimited)
            eviction_policy: Eviction policy (lru, lfu, arc)
        """
        self.cache = cache_backend or get_cache_backend()
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        self.eviction_policy = eviction_policy
        
        # Track LRU metadata
        self._access_tracker: Dict[str, LRUEntry] = {}
        self._total_size_bytes = 0
        
        # Adaptive Replacement Cache (ARC) components
        if eviction_policy == "arc":
            self._t1 = OrderedDict()  # Recent entries
            self._t2 = OrderedDict()  # Frequent entries
            self._b1 = OrderedDict()  # Recent ghost entries
            self._b2 = OrderedDict()  # Frequent ghost entries
            self._p = 0  # Target size for T1
        
        logger.info(
            f"LRU strategy initialized: "
            f"max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, "
            f"policy={eviction_policy}"
        )
    
    async def _track_access(self, key: str, value: Any = None, size: int = None):
        """
        Track access to a key.
        
        Args:
            key: Cache key
            value: Cache value (optional)
            size: Value size in bytes (optional)
        """
        now = time.time()
        
        if key in self._access_tracker:
            # Update existing entry
            entry = self._access_tracker[key]
            entry.access_time = now
            entry.access_count += 1
            
            if value is not None:
                # Update size if value changed
                old_size = entry.size
                entry.size = size or len(str(value).encode())
                self._total_size_bytes += entry.size - old_size
        else:
            # Create new entry
            if size is None and value is not None:
                size = len(str(value).encode())
            elif size is None:
                size = 0
            
            self._access_tracker[key] = LRUEntry(
                key=key,
                value=value,
                size=size,
                access_time=now,
                access_count=1,
            )
            self._total_size_bytes += size
    
    async def _needs_eviction(self) -> bool:
        """
        Check if eviction is needed.
        
        Returns:
            True if eviction is needed
        """
        # Check size limit
        if len(self._access_tracker) >= self.max_size:
            return True
        
        # Check memory limit
        if (self.max_memory_bytes and 
            self._total_size_bytes >= self.max_memory_bytes):
            return True
        
        return False
    
    async def _evict(self, count: int = 1) -> List[str]:
        """
        Evict items based on policy.
        
        Args:
            count: Number of items to evict
            
        Returns:
            List of evicted keys
        """
        if self.eviction_policy == "arc":
            return await self._evict_arc(count)
        elif self.eviction_policy == "lfu":
            return await self._evict_lfu(count)
        else:  # lru (default)
            return await self._evict_lru(count)
    
    async def _evict_lru(self, count: int) -> List[str]:
        """Evict using LRU policy."""
        # Sort by access time (oldest first)
        entries = sorted(
            self._access_tracker.values(),
            key=lambda e: e.score
        )
        
        to_evict = entries[:count]
        evicted_keys = []
        
        for entry in to_evict:
            await self.cache.delete(entry.key)
            self._total_size_bytes -= entry.size
            del self._access_tracker[entry.key]
            evicted_keys.append(entry.key)
        
        if evicted_keys:
            logger.debug(f"LRU evicted {len(evicted_keys)} items: {evicted_keys[:5]}...")
        
        return evicted_keys
    
    async def _evict_lfu(self, count: int) -> List[str]:
        """Evict using LFU (Least Frequently Used) policy."""
        # Sort by access count (least frequent first)
        entries = sorted(
            self._access_tracker.values(),
            key=lambda e: e.access_count
        )
        
        to_evict = entries[:count]
        evicted_keys = []
        
        for entry in to_evict:
            await self.cache.delete(entry.key)
            self._total_size_bytes -= entry.size
            del self._access_tracker[entry.key]
            evicted_keys.append(entry.key)
        
        if evicted_keys:
            logger.debug(f"LFU evicted {len(evicted_keys)} items: {evicted_keys[:5]}...")
        
        return evicted_keys
    
    async def _evict_arc(self, count: int) -> List[str]:
        """Evict using ARC (Adaptive Replacement Cache) policy."""
        evicted_keys = []
        
        for _ in range(count):
            if len(self._t1) >= max(1, self._p):
                # Evict from T1 (recent) or B1 (recent ghost)
                if self._t1:
                    key, _ = self._t1.popitem(last=False)
                else:
                    key, _ = self._b1.popitem(last=False)
            else:
                # Evict from T2 (frequent) or B2 (frequent ghost)
                if self._t2:
                    key, _ = self._t2.popitem(last=False)
                else:
                    key, _ = self._b2.popitem(last=False)
            
            # Remove from access tracker and cache
            if key in self._access_tracker:
                entry = self._access_tracker[key]
                self._total_size_bytes -= entry.size
                del self._access_tracker[key]
            
            await self.cache.delete(key)
            evicted_keys.append(key)
        
        if evicted_keys:
            logger.debug(f"ARC evicted {len(evicted_keys)} items: {evicted_keys[:5]}...")
        
        return evicted_keys
    
    async def _update_arc(self, key: str, hit: bool):
        """Update ARC data structures."""
        if hit:
            if key in self._t1:
                # Move from T1 to T2 (promote to frequent)
                self._t1.pop(key)
                self._t2[key] = time.time()
            elif key in self._t2:
                # Already in T2, update access time
                self._t2[key] = time.time()
        else:
            if key in self._b1:
                # Increase target size for T1
                self._p = min(self._p + max(1, len(self._b2) // len(self._b1)), self.max_size)
                self._replace(key)
                self._b1.pop(key)
                self._t2[key] = time.time()
            elif key in self._b2:
                # Decrease target size for T1
                self._p = max(self._p - max(1, len(self._b1) // len(self._b2)), 0)
                self._replace(key)
                self._b2.pop(key)
                self._t2[key] = time.time()
            else:
                # New entry, add to T1
                if len(self._t1) + len(self._b1) == self.max_size:
                    if len(self._t1) < self.max_size:
                        self._b1.popitem(last=False)
                        self._replace(key)
                    else:
                        self._t1.popitem(last=False)
                elif len(self._t1) + len(self._b1) < self.max_size:
                    total = len(self._t1) + len(self._b1) + len(self._t2) + len(self._b2)
                    if total >= self.max_size:
                        if total == 2 * self.max_size:
                            self._b2.popitem(last=False)
                        self._replace(key)
                
                self._t1[key] = time.time()
    
    def _replace(self, key: str):
        """Replace an item in ARC."""
        if self._t1 and (key in self._b2 or len(self._t1) > self._p):
            old_key, _ = self._t1.popitem(last=False)
            self._b1[old_key] = time.time()
        else:
            old_key, _ = self._t2.popitem(last=False)
            self._b2[old_key] = time.time()
    
    async def get(
        self,
        key: str,
        fetch_func: Optional[Callable] = None,
    ) -> Any:
        """
        Get value from cache with LRU tracking.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data if not in cache
            
        Returns:
            Cached value or fetched data
        """
        # Try to get from cache
        value = await self.cache.get(key)
        
        if value is not None:
            # Cache hit - update access tracking
            await self._track_access(key, value)
            
            # Update ARC if using that policy
            if self.eviction_policy == "arc":
                await self._update_arc(key, hit=True)
            
            logger.debug(f"LRU cache hit: {key}")
            return value
        
        # Cache miss
        if fetch_func:
            return await self.set_with_fetch(key, fetch_func)
        
        return None
    
    async def set_with_fetch(
        self,
        key: str,
        fetch_func: Callable,
    ) -> Any:
        """
        Set cache value by fetching data.
        
        Args:
            key: Cache key
            fetch_func: Function to fetch data
            
        Returns:
            Fetched data
        """
        # Fetch data
        logger.debug(f"Fetching data for LRU cache: {key}")
        try:
            if asyncio.iscoroutinefunction(fetch_func):
                value = await fetch_func()
            else:
                value = fetch_func()
        except Exception as e:
            logger.error(f"Error fetching data for {key}: {e}")
            raise
        
        # Store in cache with LRU tracking
        await self.set(key, value)
        
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
    ) -> bool:
        """
        Set value in cache with LRU tracking.
        
        Args:
            key: Cache key
            value: Value to cache
            
        Returns:
            True if successful
        """
        # Check if we need to evict before adding
        if await self._needs_eviction():
            await self._evict()
        
        # Store in cache
        success = await self.cache.set(key, value)
        
        if success:
            # Track access
            await self._track_access(key, value)
            
            # Update ARC for miss
            if self.eviction_policy == "arc":
                await self._update_arc(key, hit=False)
            
            logger.debug(f"LRU cache set: {key}")
        
        return success
    
    async def set_many(
        self,
        items: Dict[str, Any],
    ) -> bool:
        """
        Set multiple values with LRU awareness.
        
        Args:
            items: Dictionary of key-value pairs
            
        Returns:
            True if successful
        """
        if not items:
            return True
        
        # Check if we need to evict
        new_count = len(items)
        current_count = len(self._access_tracker)
        
        if current_count + new_count > self.max_size:
            evict_count = current_count + new_count - self.max_size
            await self._evict(evict_count)
        
        # Estimate memory usage
        if self.max_memory_bytes:
            new_size = sum(len(str(v).encode()) for v in items.values())
            if self._total_size_bytes + new_size > self.max_memory_bytes:
                # Need to evict based on memory
                # Simple approach: evict until we have enough space
                while (self._total_size_bytes + new_size > self.max_memory_bytes and
                       self._access_tracker):
                    await self._evict(1)
        
        # Store items
        try:
            success = await self.cache.set_many(items)
            
            if success:
                # Track access for all items
                for key, value in items.items():
                    await self._track_access(key, value)
                    
                    # Update ARC for misses
                    if self.eviction_policy == "arc":
                        await self._update_arc(key, hit=False)
            
            logger.debug(f"LRU set_many: {len(items)} items")
            return success
            
        except Exception as e:
            logger.error(f"Error setting multiple items: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        success = await self.cache.delete(key)
        
        if success and key in self._access_tracker:
            # Remove from tracking
            entry = self._access_tracker[key]
            self._total_size_bytes -= entry.size
            del self._access_tracker[key]
            
            # Remove from ARC if using that policy
            if self.eviction_policy == "arc":
                for cache_dict in [self._t1, self._t2, self._b1, self._b2]:
                    if key in cache_dict:
                        del cache_dict[key]
        
        return success
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        success = await self.cache.clear()
        
        if success:
            # Clear tracking
            self._access_tracker.clear()
            self._total_size_bytes = 0
            
            # Clear ARC structures
            if self.eviction_policy == "arc":
                self._t1.clear()
                self._t2.clear()
                self._b1.clear()
                self._b2.clear()
                self._p = 0
        
        return success
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get LRU strategy statistics.
        
        Returns:
            Statistics dictionary
        """
        total_keys = len(self._access_tracker)
        
        # Calculate access statistics
        if total_keys > 0:
            access_times = [e.access_time for e in self._access_tracker.values()]
            access_counts = [e.access_count for e in self._access_tracker.values()]
            sizes = [e.size for e in self._access_tracker.values()]
            
            stats = {
                "total_keys": total_keys,
                "total_size_mb": self._total_size_bytes / (1024 * 1024),
                "avg_size_bytes": sum(sizes) / len(sizes) if sizes else 0,
                "access_stats": {
                    "min_time": min(access_times) if access_times else 0,
                    "max_time": max(access_times) if access_times else 0,
                    "avg_count": sum(access_counts) / len(access_counts) if access_counts else 0,
                },
                "eviction_stats": {
                    "policy": self.eviction_policy,
                    "max_size": self.max_size,
                    "current_usage_percent": (total_keys / self.max_size * 100) if self.max_size else 0,
                    "memory_usage_percent": (
                        (self._total_size_bytes / self.max_memory_bytes * 100) 
                        if self.max_memory_bytes else 0
                    ),
                }
            }
        else:
            stats = {
                "total_keys": 0,
                "total_size_mb": 0,
                "access_stats": {},
                "eviction_stats": {
                    "policy": self.eviction_policy,
                    "max_size": self.max_size,
                    "current_usage_percent": 0,
                    "memory_usage_percent": 0,
                }
            }
        
        # Add ARC-specific stats if applicable
        if self.eviction_policy == "arc":
            stats["arc_stats"] = {
                "t1_size": len(self._t1),
                "t2_size": len(self._t2),
                "b1_size": len(self._b1),
                "b2_size": len(self._b2),
                "p_value": self._p,
            }
        
        return stats
    
    async def get_hot_keys(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get most frequently accessed keys.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of (key, access_count) tuples
        """
        entries = sorted(
            self._access_tracker.values(),
            key=lambda e: e.access_count,
            reverse=True
        )
        
        return [(e.key, e.access_count) for e in entries[:limit]]
    
    async def get_cold_keys(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get least recently accessed keys.
        
        Args:
            limit: Maximum number of keys to return
            
        Returns:
            List of (key, seconds_since_access) tuples
        """
        now = time.time()
        entries = sorted(
            self._access_tracker.values(),
            key=lambda e: e.access_time
        )
        
        return [(e.key, now - e.access_time) for e in entries[:limit]]