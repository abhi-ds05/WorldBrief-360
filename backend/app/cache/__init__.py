"""
Cache module for WorldBrief 360.

This module provides a unified caching interface with support for multiple backends
(Redis, in-memory, file-based) and strategies (LRU, TTL, write-through).

Features:
- Multiple cache backends (Redis, memory, file)
- Different caching strategies
- Cache invalidation patterns
- Metrics and monitoring
- Type-safe operations
"""

import json
import pickle
import hashlib
import time
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Dict, List, Callable, TypeVar, Generic
from enum import Enum
from datetime import datetime, timedelta
from functools import wraps
import logging

from app.core.config import settings

"""
Cache package for WorldBrief 360.
Provides caching utilities, backends, and invalidation strategies.
"""

from app.cache.backends import get_cache_backend
from app.cache.invalidators import CacheInvalidator
from app.cache.strategies import (
    CacheStrategy,
    TTLStrategy,
    LRUStrategy,
    WriteThroughStrategy
)

__all__ = [
    'get_cache_backend',
    'CacheInvalidator',
    'CacheStrategy',
    'TTLStrategy',
    'LRUStrategy',
    'WriteThroughStrategy',
]

# Type variables for generic caching
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

# Logger
logger = logging.getLogger(__name__)


class CacheBackendType(Enum):
    """Supported cache backend types."""
    REDIS = "redis"
    MEMORY = "memory"
    FILE = "file"
    MULTI_LEVEL = "multi_level"


class CacheStrategy(Enum):
    """Supported caching strategies."""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"


class CacheKey:
    """Helper class for generating consistent cache keys."""
    
    @staticmethod
    def generate(*args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Generated cache key
        """
        # Convert args and kwargs to string representation
        parts = []
        
        # Add args
        for arg in args:
            parts.append(str(arg))
        
        # Add kwargs (sorted for consistency)
        for key in sorted(kwargs.keys()):
            parts.append(f"{key}:{kwargs[key]}")
        
        # Join and hash
        key_string = ":".join(parts)
        
        # For long keys, use hash
        if len(key_string) > 100:
            return hashlib.md5(key_string.encode()).hexdigest()
        
        return key_string
    
    @staticmethod
    def for_model(model_name: str, model_id: Union[int, str]) -> str:
        """
        Generate cache key for a model instance.
        
        Args:
            model_name: Name of the model
            model_id: ID of the model instance
            
        Returns:
            str: Cache key
        """
        return f"model:{model_name}:{model_id}"
    
    @staticmethod
    def for_list(model_name: str, filters: Optional[Dict] = None) -> str:
        """
        Generate cache key for a list of model instances.
        
        Args:
            model_name: Name of the model
            filters: Filter parameters
            
        Returns:
            str: Cache key
        """
        if filters:
            filter_str = json.dumps(filters, sort_keys=True)
            return f"list:{model_name}:{hashlib.md5(filter_str.encode()).hexdigest()}"
        return f"list:{model_name}:all"
    
    @staticmethod
    def for_user(user_id: Union[int, str], resource: str, *args) -> str:
        """
        Generate cache key for user-specific data.
        
        Args:
            user_id: User ID
            resource: Resource name
            *args: Additional arguments
            
        Returns:
            str: Cache key
        """
        parts = [f"user:{user_id}:{resource}"]
        parts.extend(str(arg) for arg in args)
        return ":".join(parts)
    
    @staticmethod
    def for_topic(topic_id: Union[int, str], data_type: str) -> str:
        """
        Generate cache key for topic data.
        
        Args:
            topic_id: Topic ID
            data_type: Type of data (articles, briefings, etc.)
            
        Returns:
            str: Cache key
        """
        return f"topic:{topic_id}:{data_type}"
    
    @staticmethod
    def for_incident(incident_id: Union[int, str], data_type: str = "details") -> str:
        """
        Generate cache key for incident data.
        
        Args:
            incident_id: Incident ID
            data_type: Type of data
            
        Returns:
            str: Cache key
        """
        return f"incident:{incident_id}:{data_type}"


class CacheBackend(ABC, Generic[K, V]):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: K) -> Optional[V]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    async def set(self, key: K, value: V, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: K) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: K) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    async def keys(self, pattern: str = "*") -> List[K]:
        """Get keys matching pattern."""
        pass
    
    @abstractmethod
    async def increment(self, key: K, amount: int = 1) -> int:
        """Increment integer value."""
        pass
    
    @abstractmethod
    async def decrement(self, key: K, amount: int = 1) -> int:
        """Decrement integer value."""
        pass
    
    @abstractmethod
    async def ttl(self, key: K) -> int:
        """Get TTL for key in seconds."""
        pass
    
    @abstractmethod
    async def expire(self, key: K, ttl: int) -> bool:
        """Set TTL for key."""
        pass


class CacheMetrics:
    """Cache performance metrics collector."""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_operations = 0
        self.start_time = time.time()
    
    def record_hit(self):
        """Record a cache hit."""
        self.hits += 1
        self.total_operations += 1
    
    def record_miss(self):
        """Record a cache miss."""
        self.misses += 1
        self.total_operations += 1
    
    def record_set(self):
        """Record a cache set operation."""
        self.sets += 1
        self.total_operations += 1
    
    def record_delete(self):
        """Record a cache delete operation."""
        self.deletes += 1
        self.total_operations += 1
    
    def record_error(self):
        """Record a cache error."""
        self.errors += 1
        self.total_operations += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        uptime = time.time() - self.start_time
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_operations": self.total_operations,
            "hit_rate": self.get_hit_rate(),
            "uptime_seconds": uptime,
            "operations_per_second": self.total_operations / uptime if uptime > 0 else 0,
        }


# Import backend implementations
try:
    from .redis_client import RedisBackend
except ImportError:
    logger.warning("Redis backend not available")
    RedisBackend = None

from .strategies.lru import LRUCacheStrategy
from .strategies.ttl import TTLCacheStrategy
from .strategies.write_through import WriteThroughStrategy
from .invalidators.model_invalidator import ModelInvalidator

# Cache manager instance
_cache_manager = None


class CacheManager:
    """
    Main cache manager that provides unified interface to all caching operations.
    """
    
    def __init__(self):
        self.backends: Dict[str, CacheBackend] = {}
        self.strategies: Dict[str, CacheStrategy] = {}
        self.metrics = CacheMetrics()
        self.default_ttl = settings.CACHE_DEFAULT_TTL
        self._setup_backends()
        self._setup_strategies()
    
    def _setup_backends(self):
        """Setup cache backends based on configuration."""
        # Setup Redis backend if configured
        if settings.REDIS_URL and RedisBackend:
            try:
                redis_backend = RedisBackend(
                    url=settings.REDIS_URL,
                    prefix=settings.CACHE_KEY_PREFIX,
                    default_ttl=self.default_ttl
                )
                self.backends["redis"] = redis_backend
                logger.info("Redis cache backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Redis backend: {e}")
        
        # Setup in-memory backend as fallback
        MemoryBackend = None
        try:
            # Try absolute import first (works when running as an installed package),
            # then fall back to a relative import when running as a package/module.
            import importlib
            mod = None
            try:
                mod = importlib.import_module("app.cache.memory_backend")
            except (ImportError, ModuleNotFoundError):
                try:
                    # Use package-relative import; __package__ should be 'app.cache' when used as a package.
                    mod = importlib.import_module(".memory_backend", package=__package__)
                except Exception as e:
                    logger.debug(f"Relative import for MemoryBackend failed: {e}")
                    mod = None

            if mod:
                MemoryBackend = getattr(mod, "MemoryBackend", None)
            else:
                logger.warning("Memory backend not available")
        except Exception as e:
            logger.warning(f"Memory backend import error: {e}")
            MemoryBackend = None

        if MemoryBackend:
            try:
                memory_backend = MemoryBackend(max_size=settings.CACHE_MEMORY_MAX_SIZE)
                self.backends["memory"] = memory_backend
                logger.info("Memory cache backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Memory backend: {e}")
        else:
            logger.warning("Skipping memory backend initialization because MemoryBackend is unavailable")
        
        # Setup file backend for persistent caching
        if settings.CACHE_FILE_ENABLED:
            FileBackend = None
            try:
                import importlib
                mod = None
                try:
                    mod = importlib.import_module("app.cache.file_backend")
                except (ImportError, ModuleNotFoundError):
                    try:
                        mod = importlib.import_module(".file_backend", package=__package__)
                    except Exception as e:
                        logger.debug(f"Relative import for FileBackend failed: {e}")
                        mod = None

                if mod:
                    FileBackend = getattr(mod, "FileBackend", None)
                else:
                    logger.warning("File backend not available")
            except Exception as e:
                logger.warning(f"File backend import error: {e}")
                FileBackend = None

            if FileBackend:
                try:
                    file_backend = FileBackend(
                        base_dir=settings.CACHE_FILE_DIR,
                        default_ttl=self.default_ttl
                    )
                    self.backends["file"] = file_backend
                    logger.info("File cache backend initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize File backend: {e}")
            else:
                logger.warning("Skipping file backend initialization because FileBackend is unavailable")
    
    def _setup_strategies(self):
        """Setup caching strategies."""
        # Get primary backend
        primary_backend = self.backends.get("redis") or self.backends.get("memory")
        
        if primary_backend:
            # LRU Strategy
            self.strategies["lru"] = LRUCacheStrategy(
                backend=primary_backend,
                max_size=settings.CACHE_LRU_MAX_SIZE
            )
            
            # TTL Strategy
            self.strategies["ttl"] = TTLCacheStrategy(
                backend=primary_backend,
                default_ttl=self.default_ttl
            )
            
            # Write-through Strategy
            self.strategies["write_through"] = WriteThroughStrategy(
                backend=primary_backend,
                write_func=None  # Will be set per use case
            )
            
            logger.info(f"Initialized {len(self.strategies)} cache strategies")
    
    def get_backend(self, name: str = "default") -> Optional[CacheBackend]:
        """
        Get cache backend by name.
        
        Args:
            name: Backend name (redis, memory, file)
            
        Returns:
            Optional[CacheBackend]: Cache backend or None
        """
        if name == "default":
            # Return Redis if available, otherwise memory
            return self.backends.get("redis") or self.backends.get("memory")
        return self.backends.get(name)
    
    def get_strategy(self, name: str) -> Optional[Any]:
        """
        Get cache strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Optional[Any]: Cache strategy or None
        """
        return self.strategies.get(name)
    
    async def get(self, key: str, strategy: str = "ttl", **kwargs) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            strategy: Caching strategy to use
            **kwargs: Additional arguments for strategy
            
        Returns:
            Optional[Any]: Cached value or None
        """
        try:
            cache_strategy = self.strategies.get(strategy)
            if not cache_strategy:
                # Fall back to direct backend access
                backend = self.get_backend()
                if backend:
                    value = await backend.get(key)
                    if value is not None:
                        self.metrics.record_hit()
                    else:
                        self.metrics.record_miss()
                    return value
                return None
            
            value = await cache_strategy.get(key, **kwargs)
            if value is not None:
                self.metrics.record_hit()
            else:
                self.metrics.record_miss()
            return value
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  strategy: str = "ttl", **kwargs) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            strategy: Caching strategy to use
            **kwargs: Additional arguments for strategy
            
        Returns:
            bool: True if successful
        """
        try:
            cache_strategy = self.strategies.get(strategy)
            if not cache_strategy:
                # Fall back to direct backend access
                backend = self.get_backend()
                if backend:
                    success = await backend.set(key, value, ttl or self.default_ttl)
                    if success:
                        self.metrics.record_set()
                    return success
                return False
            
            success = await cache_strategy.set(key, value, ttl or self.default_ttl, **kwargs)
            if success:
                self.metrics.record_set()
            return success
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str, strategy: str = "ttl") -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            strategy: Caching strategy to use
            
        Returns:
            bool: True if successful
        """
        try:
            cache_strategy = self.strategies.get(strategy)
            if not cache_strategy:
                # Fall back to direct backend access
                backend = self.get_backend()
                if backend:
                    success = await backend.delete(key)
                    if success:
                        self.metrics.record_delete()
                    return success
                return False
            
            success = await cache_strategy.delete(key)
            if success:
                self.metrics.record_delete()
            return success
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def get_or_set(self, key: str, factory: Callable[[], Any],
                         ttl: Optional[int] = None, strategy: str = "ttl") -> Any:
        """
        Get value from cache or set it using factory function.
        
        Args:
            key: Cache key
            factory: Function to generate value if not in cache
            ttl: Time to live in seconds
            strategy: Caching strategy to use
            
        Returns:
            Any: Cached or newly generated value
        """
        # Try to get from cache
        cached = await self.get(key, strategy=strategy)
        if cached is not None:
            return cached
        
        # Generate value using factory
        try:
            value = factory()
            if value is not None:
                await self.set(key, value, ttl, strategy=strategy)
            return value
        except Exception as e:
            logger.error(f"Error in cache factory for key {key}: {e}")
            raise
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Pattern to match keys
            
        Returns:
            int: Number of keys invalidated
        """
        try:
            backend = self.get_backend()
            if not backend:
                return 0
            
            keys = await backend.keys(pattern)
            deleted = 0
            for key in keys:
                if await backend.delete(key):
                    deleted += 1
                    self.metrics.record_delete()
            
            logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
            return deleted
            
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        try:
            success = True
            for name, backend in self.backends.items():
                try:
                    if not await backend.clear():
                        success = False
                        logger.warning(f"Failed to clear backend: {name}")
                except Exception as e:
                    logger.error(f"Error clearing backend {name}: {e}")
                    success = False
            
            if success:
                logger.info("All cache backends cleared successfully")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing all cache: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache metrics.
        
        Returns:
            Dict[str, Any]: Cache metrics
        """
        return self.metrics.get_metrics()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all cache backends.
        
        Returns:
            Dict[str, Any]: Health status for each backend
        """
        health_status = {}
        
        for name, backend in self.backends.items():
            try:
                # Try a simple operation to check health
                test_key = f"health_check:{name}:{int(time.time())}"
                await backend.set(test_key, "test", 10)
                await backend.get(test_key)
                await backend.delete(test_key)
                
                health_status[name] = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            except Exception as e:
                health_status[name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                }
        
        return health_status


def get_cache_manager() -> CacheManager:
    """
    Get or create cache manager instance.
    
    Returns:
        CacheManager: Cache manager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


# Decorator for caching function results
def cached(ttl: Optional[int] = None, key_prefix: str = "func", 
           strategy: str = "ttl", ignore_args: List[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        strategy: Caching strategy to use
        ignore_args: List of argument indices to ignore in key generation
    
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_args = list(args)
            cache_kwargs = kwargs.copy()
            
            # Remove ignored arguments
            if ignore_args:
                for idx in sorted(ignore_args, reverse=True):
                    if idx < len(cache_args):
                        del cache_args[idx]
            
            # Remove 'self' for instance methods
            if cache_args and hasattr(cache_args[0], '__class__'):
                # It's an instance method, include class name but not instance
                cache_args[0] = cache_args[0].__class__.__name__
            
            key = CacheKey.generate(key_prefix, func.__name__, *cache_args, **cache_kwargs)
            
            # Try to get from cache
            cache_manager = get_cache_manager()
            cached_result = await cache_manager.get(key, strategy=strategy)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}, executing")
            result = await func(*args, **kwargs)
            
            if result is not None:
                await cache_manager.set(key, result, ttl, strategy=strategy)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic for synchronous functions
            import asyncio
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Convenience functions for common caching operations
async def cache_model(model_name: str, model_id: Union[int, str], data: Any,
                      ttl: Optional[int] = None) -> bool:
    """
    Cache a model instance.
    
    Args:
        model_name: Name of the model
        model_id: ID of the model instance
        data: Model data to cache
        ttl: Time to live in seconds
        
    Returns:
        bool: True if successful
    """
    key = CacheKey.for_model(model_name, model_id)
    cache_manager = get_cache_manager()
    return await cache_manager.set(key, data, ttl)


async def get_cached_model(model_name: str, model_id: Union[int, str]) -> Optional[Any]:
    """
    Get cached model instance.
    
    Args:
        model_name: Name of the model
        model_id: ID of the model instance
        
    Returns:
        Optional[Any]: Cached model data or None
    """
    key = CacheKey.for_model(model_name, model_id)
    cache_manager = get_cache_manager()
    return await cache_manager.get(key)


async def invalidate_model(model_name: str, model_id: Union[int, str]) -> bool:
    """
    Invalidate cached model instance.
    
    Args:
        model_name: Name of the model
        model_id: ID of the model instance
        
    Returns:
        bool: True if successful
    """
    key = CacheKey.for_model(model_name, model_id)
    cache_manager = get_cache_manager()
    return await cache_manager.delete(key)


async def cache_list(model_name: str, filters: Optional[Dict], data: List[Any],
                     ttl: Optional[int] = None) -> bool:
    """
    Cache a list of model instances.
    
    Args:
        model_name: Name of the model
        filters: Filter parameters
        data: List data to cache
        ttl: Time to live in seconds
        
    Returns:
        bool: True if successful
    """
    key = CacheKey.for_list(model_name, filters)
    cache_manager = get_cache_manager()
    return await cache_manager.set(key, data, ttl)


async def get_cached_list(model_name: str, filters: Optional[Dict] = None) -> Optional[List[Any]]:
    """
    Get cached list of model instances.
    
    Args:
        model_name: Name of the model
        filters: Filter parameters
        
    Returns:
        Optional[List[Any]]: Cached list data or None
    """
    key = CacheKey.for_list(model_name, filters)
    cache_manager = get_cache_manager()
    return await cache_manager.get(key)


# Export public API
__all__ = [
    # Classes
    "CacheManager",
    "CacheBackend",
    "CacheKey",
    "CacheMetrics",
    "CacheBackendType",
    "CacheStrategy",
    
    # Functions
    "get_cache_manager",
    "cached",
    "cache_model",
    "get_cached_model",
    "invalidate_model",
    "cache_list",
    "get_cached_list",
    
    # Backends
    "RedisBackend",
    
    # Strategies
    "LRUCacheStrategy",
    "TTLCacheStrategy",
    "WriteThroughStrategy",
    
    # Invalidators
    "ModelInvalidator",
]