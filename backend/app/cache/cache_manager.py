"""
Cache Manager - Main interface for all caching operations in WorldBrief 360.
This module provides a unified API for caching with support for multiple backends
and strategies, with built-in metrics, monitoring, and automatic invalidation.
"""
import asyncio
import time
import json
import hashlib
from typing import Any, Optional, Dict, List, Union, Callable, TypeVar, Generic
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from enum import Enum
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.metrics import metrics_client
from app.db.models import CacheMetadata

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheOperation(str, Enum):
    """Types of cache operations."""
    GET = "get"
    SET = "set"
    DELETE = "delete"
    INVALIDATE = "invalidate"
    CLEAR = "clear"
    HIT = "hit"
    MISS = "miss"


class CacheBackend(str, Enum):
    """Supported cache backends."""
    REDIS = "redis"
    MEMORY = "memory"
    FILE = "file"
    DATABASE = "database"
    MULTI_LEVEL = "multi_level"


class CacheStrategy(str, Enum):
    """Caching strategies."""
    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    WRITE_THROUGH = "write_through"
    WRITE_BACK = "write_back"
    WRITE_AROUND = "write_around"
    READ_THROUGH = "read_through"


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class CacheConfig:
    """Cache configuration."""
    default_ttl: int = 300  # 5 minutes
    max_size_mb: int = 100  # 100 MB max cache size
    backend: CacheBackend = CacheBackend.REDIS
    strategy: CacheStrategy = CacheStrategy.TTL
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack
    namespace: str = "wb360"


class CacheKeyBuilder:
    """Builds consistent cache keys with namespacing."""
    
    def __init__(self, namespace: str = "wb360"):
        self.namespace = namespace
    
    def build(self, *parts: Any, separator: str = ":") -> str:
        """
        Build a cache key from parts.
        
        Args:
            *parts: Parts to include in the key
            separator: Separator between parts
            
        Returns:
            str: Built cache key
        """
        # Convert all parts to strings and clean
        str_parts = [self.namespace]
        for part in parts:
            if part is None:
                continue
            if isinstance(part, (list, dict)):
                # Hash complex objects
                part_hash = hashlib.md5(
                    json.dumps(part, sort_keys=True).encode()
                ).hexdigest()
                str_parts.append(part_hash[:8])
            else:
                str_parts.append(str(part).lower().replace(" ", "_"))
        
        return separator.join(str_parts)
    
    def for_model(self, model_name: str, model_id: Any, field: str = None) -> str:
        """Build key for model instance."""
        parts = ["model", model_name, model_id]
        if field:
            parts.append(field)
        return self.build(*parts)
    
    def for_list(self, model_name: str, filters: Dict = None, page: int = None, 
                 per_page: int = None) -> str:
        """Build key for list of models."""
        parts = ["list", model_name]
        if filters:
            parts.append("filtered")
            # Add filter hash
            filter_hash = hashlib.md5(
                json.dumps(filters, sort_keys=True).encode()
            ).hexdigest()
            parts.append(filter_hash[:12])
        if page is not None:
            parts.append(f"page_{page}")
            if per_page:
                parts.append(f"per_{per_page}")
        return self.build(*parts)
    
    def for_user(self, user_id: Any, resource: str, *args) -> str:
        """Build key for user-specific data."""
        parts = ["user", user_id, resource, *args]
        return self.build(*parts)
    
    def for_topic(self, topic_id: Any, data_type: str = "summary") -> str:
        """Build key for topic data."""
        return self.build("topic", topic_id, data_type)
    
    def for_incident(self, incident_id: Any, data_type: str = "details") -> str:
        """Build key for incident data."""
        return self.build("incident", incident_id, data_type)
    
    def for_briefing(self, briefing_id: Any, format: str = "html") -> str:
        """Build key for briefing data."""
        return self.build("briefing", briefing_id, format)
    
    def for_chat(self, session_id: str, message_count: int = None) -> str:
        """Build key for chat session."""
        parts = ["chat", session_id]
        if message_count:
            parts.append(f"messages_{message_count}")
        return self.build(*parts)


class CacheManager:
    """
    Main cache manager with support for multiple backends and strategies.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize cache manager."""
        if self._initialized:
            return
        
        self.config = CacheConfig(
            default_ttl=settings.CACHE_DEFAULT_TTL,
            max_size_mb=settings.CACHE_MAX_SIZE_MB,
            backend=CacheBackend(settings.CACHE_BACKEND),
            strategy=CacheStrategy(settings.CACHE_STRATEGY),
            compression=settings.CACHE_COMPRESSION,
            serialization=settings.CACHE_SERIALIZATION,
            namespace=settings.CACHE_NAMESPACE,
        )
        
        self.key_builder = CacheKeyBuilder(self.config.namespace)
        self.stats = CacheStats()
        self._backends = {}
        self._strategies = {}
        self._initialized = True
        
        self._init_backends()
        self._init_strategies()
        
        logger.info(f"CacheManager initialized with backend: {self.config.backend}")
    
    def _init_backends(self):
        """Initialize cache backends."""
        # Redis backend
        if self.config.backend in [CacheBackend.REDIS, CacheBackend.MULTI_LEVEL]:
            try:
                from .redis_client import RedisCacheBackend
                redis_backend = RedisCacheBackend(
                    url=settings.REDIS_URL,
                    default_ttl=self.config.default_ttl,
                    compression=self.config.compression,
                )
                self._backends[CacheBackend.REDIS] = redis_backend
                logger.info("Redis cache backend initialized")
            except ImportError as e:
                logger.warning(f"Redis backend not available: {e}")
        
        # Memory backend
        try:
            import importlib
            module = None
            for module_name in ("app.cache.memory_backend", "app.cache.backends.memory_backend", f"{__package__}.memory_backend"):
                try:
                    module = importlib.import_module(module_name)
                    break
                except ImportError:
                    module = None
            
            if module is None:
                raise ImportError("memory_backend module not found in known locations")
            
            MemoryCacheBackend = getattr(module, "MemoryCacheBackend")
            memory_backend = MemoryCacheBackend(
                max_size_mb=self.config.max_size_mb,
                default_ttl=self.config.default_ttl,
            )
            self._backends[CacheBackend.MEMORY] = memory_backend
            logger.info("Memory cache backend initialized")
        except Exception as e:
            logger.warning(f"Memory backend not available: {e}")
        
        # File backend (for large objects)
        if self.config.backend == CacheBackend.FILE:
            try:
                import importlib

                # Try several possible import paths to accommodate different package layouts
                module = None
                for module_name in ("app.cache.file_backend", "app.cache.backends.file_backend", f"{__package__}.file_backend"):
                    try:
                        module = importlib.import_module(module_name)
                        break
                    except ImportError:
                        module = None

                if module is None:
                    raise ImportError("file_backend module not found in known locations")

                FileCacheBackend = getattr(module, "FileCacheBackend")
                file_backend = FileCacheBackend(
                    cache_dir=settings.CACHE_FILE_DIR,
                    default_ttl=self.config.default_ttl,
                )
                self._backends[CacheBackend.FILE] = file_backend
                logger.info("File cache backend initialized")
            except Exception as e:
                logger.warning(f"File backend not available: {e}")
        
        # Multi-level cache (L1: memory, L2: redis, L3: file)
        if self.config.backend == CacheBackend.MULTI_LEVEL:
            try:
                from app.cache.multi_level_backend import MultiLevelCacheBackend
                # Order matters: faster -> slower
                levels = []
                if CacheBackend.MEMORY in self._backends:
                    levels.append(self._backends[CacheBackend.MEMORY])
                if CacheBackend.REDIS in self._backends:
                    levels.append(self._backends[CacheBackend.REDIS])
                if CacheBackend.FILE in self._backends:
                    levels.append(self._backends[CacheBackend.FILE])
                
                if len(levels) > 1:
                    multi_backend = MultiLevelCacheBackend(levels=levels)
                    self._backends[CacheBackend.MULTI_LEVEL] = multi_backend
                    logger.info(f"Multi-level cache initialized with {len(levels)} levels")
            except ImportError as e:
                logger.warning(f"Multi-level backend not available: {e}")
    
    def _init_strategies(self):
        """Initialize caching strategies."""
        # Get primary backend
        primary_backend = self._backends.get(
            self.config.backend, 
            self._backends.get(CacheBackend.MEMORY)
        )
        
        if not primary_backend:
            logger.error("No cache backend available")
            return
        
        # TTL Strategy
        try:
            from app.cache.strategies.ttl import TTLStrategy
            self._strategies[CacheStrategy.TTL] = TTLStrategy(
                backend=primary_backend,
                default_ttl=self.config.default_ttl,
            )
        except Exception as e:
            logger.warning(f"TTL strategy not available: {e}")
        
        # LRU Strategy
        try:
            from app.cache.strategies.lru import LRUStrategy
            self._strategies[CacheStrategy.LRU] = LRUStrategy(
                backend=primary_backend,
                max_size=self.config.max_size_mb * 1024 * 1024,  # Convert to bytes
            )
        except Exception as e:
            logger.warning(f"LRU strategy not available: {e}")
        
        # Write-through Strategy
        try:
            from app.cache.strategies.write_through import WriteThroughStrategy
            self._strategies[CacheStrategy.WRITE_THROUGH] = WriteThroughStrategy(
                backend=primary_backend,
            )
        except Exception as e:
            logger.warning(f"Write-through strategy not available: {e}")
        
        # Read-through Strategy
        try:
            from app.cache.strategies.read_through import ReadThroughStrategy
            self._strategies[CacheStrategy.READ_THROUGH] = ReadThroughStrategy(
                backend=primary_backend,
            )
        except Exception as e:
            logger.warning(f"Read-through strategy not available: {e}")
        
        logger.info(f"Initialized {len(self._strategies)} cache strategies")
    
    def get_backend(self, backend_name: CacheBackend = None) -> Any:
        """
        Get cache backend.
        
        Args:
            backend_name: Name of backend to get
            
        Returns:
            Cache backend instance
        """
        if backend_name is None:
            backend_name = self.config.backend
        
        backend = self._backends.get(backend_name)
        if backend is None:
            raise ValueError(f"Cache backend '{backend_name}' not available")
        
        return backend
    
    def get_strategy(self, strategy_name: CacheStrategy = None) -> Any:
        """
        Get caching strategy.
        
        Args:
            strategy_name: Name of strategy to get
            
        Returns:
            Cache strategy instance
        """
        if strategy_name is None:
            strategy_name = self.config.strategy
        
        strategy = self._strategies.get(strategy_name)
        if strategy is None:
            logger.warning(f"Strategy '{strategy_name}' not found, using TTL")
            strategy = self._strategies.get(CacheStrategy.TTL)
        
        return strategy
    
    async def get(self, key: str, default: Any = None, 
                  strategy: CacheStrategy = None, **kwargs) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            strategy: Caching strategy to use
            **kwargs: Additional arguments for strategy
            
        Returns:
            Cached value or default
        """
        start_time = time.time()
        operation = CacheOperation.GET
        
        try:
            cache_strategy = self.get_strategy(strategy)
            value = await cache_strategy.get(key, **kwargs)
            
            if value is not None:
                self._record_hit()
                operation = CacheOperation.HIT
                logger.debug(f"Cache hit for key: {key}")
            else:
                self._record_miss()
                operation = CacheOperation.MISS
                logger.debug(f"Cache miss for key: {key}")
                value = default
            
            # Record metrics
            self._record_operation(operation, key, time.time() - start_time)
            
            return value
            
        except Exception as e:
            self._record_error()
            logger.error(f"Cache get error for key {key}: {e}")
            self._record_operation(operation, key, time.time() - start_time, error=str(e))
            return default
    
    async def set(self, key: str, value: Any, ttl: int = None,
                  strategy: CacheStrategy = None, **kwargs) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            strategy: Caching strategy to use
            **kwargs: Additional arguments for strategy
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            if ttl is None:
                ttl = self.config.default_ttl
            
            cache_strategy = self.get_strategy(strategy)
            success = await cache_strategy.set(key, value, ttl, **kwargs)
            
            if success:
                self._record_set()
                logger.debug(f"Cache set for key: {key} (ttl: {ttl}s)")
            
            # Record metrics
            self._record_operation(CacheOperation.SET, key, time.time() - start_time)
            
            return success
            
        except Exception as e:
            self._record_error()
            logger.error(f"Cache set error for key {key}: {e}")
            self._record_operation(CacheOperation.SET, key, time.time() - start_time, error=str(e))
            return False
    
    async def delete(self, key: str, strategy: CacheStrategy = None) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            strategy: Caching strategy to use
            
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            cache_strategy = self.get_strategy(strategy)
            success = await cache_strategy.delete(key)
            
            if success:
                self._record_delete()
                logger.debug(f"Cache delete for key: {key}")
            
            # Record metrics
            self._record_operation(CacheOperation.DELETE, key, time.time() - start_time)
            
            return success
            
        except Exception as e:
            self._record_error()
            logger.error(f"Cache delete error for key {key}: {e}")
            self._record_operation(CacheOperation.DELETE, key, time.time() - start_time, error=str(e))
            return False
    
    async def exists(self, key: str, strategy: CacheStrategy = None) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            strategy: Caching strategy to use
            
        Returns:
            True if key exists
        """
        try:
            cache_strategy = self.get_strategy(strategy)
            return await cache_strategy.exists(key)
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def get_or_set(self, key: str, factory: Callable[[], Any],
                         ttl: int = None, strategy: CacheStrategy = None) -> Any:
        """
        Get value from cache or set it using factory function.
        
        Args:
            key: Cache key
            factory: Function that returns value if not cached
            ttl: Time to live in seconds
            strategy: Caching strategy to use
            
        Returns:
            Cached or newly generated value
        """
        # Try to get from cache first
        cached_value = await self.get(key, strategy=strategy)
        if cached_value is not None:
            return cached_value
        
        # Generate value using factory
        try:
            value = await asyncio.get_event_loop().run_in_executor(None, factory)
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
            Number of keys invalidated
        """
        start_time = time.time()
        
        try:
            backend = self.get_backend()
            if hasattr(backend, 'delete_pattern'):
                count = await backend.delete_pattern(pattern)
            else:
                # Fallback: get all keys and delete matching ones
                keys = await backend.keys(pattern)
                count = 0
                for key in keys:
                    if await backend.delete(key):
                        count += 1
            
            logger.info(f"Invalidated {count} keys matching pattern: {pattern}")
            
            # Record metrics
            self._record_operation(
                CacheOperation.INVALIDATE, 
                pattern, 
                time.time() - start_time,
                count=count
            )
            
            return count
            
        except Exception as e:
            logger.error(f"Cache invalidate pattern error: {e}")
            self._record_operation(
                CacheOperation.INVALIDATE, 
                pattern, 
                time.time() - start_time, 
                error=str(e)
            )
            return 0
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful
        """
        start_time = time.time()
        
        try:
            backend = self.get_backend()
            success = await backend.clear()
            
            if success:
                logger.info("Cache cleared successfully")
            
            # Record metrics
            self._record_operation(CacheOperation.CLEAR, "all", time.time() - start_time)
            
            return success
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self._record_operation(
                CacheOperation.CLEAR, 
                "all", 
                time.time() - start_time, 
                error=str(e)
            )
            return False
    
    async def get_many(self, keys: List[str], strategy: CacheStrategy = None) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            strategy: Caching strategy to use
            
        Returns:
            Dictionary of key-value pairs
        """
        result = {}
        
        for key in keys:
            value = await self.get(key, strategy=strategy)
            if value is not None:
                result[key] = value
        
        return result
    
    async def set_many(self, items: Dict[str, Any], ttl: int = None,
                       strategy: CacheStrategy = None) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            strategy: Caching strategy to use
            
        Returns:
            True if all successful
        """
        success = True
        
        for key, value in items.items():
            if not await self.set(key, value, ttl, strategy=strategy):
                success = False
        
        return success
    
    # Convenience methods for common use cases
    
    async def cache_model(self, model_name: str, model_id: Any, data: Any,
                          ttl: int = None, strategy: CacheStrategy = None) -> bool:
        """Cache a model instance."""
        key = self.key_builder.for_model(model_name, model_id)
        return await self.set(key, data, ttl, strategy=strategy)
    
    async def get_cached_model(self, model_name: str, model_id: Any,
                               strategy: CacheStrategy = None) -> Any:
        """Get cached model instance."""
        key = self.key_builder.for_model(model_name, model_id)
        return await self.get(key, strategy=strategy)
    
    async def invalidate_model(self, model_name: str, model_id: Any) -> bool:
        """Invalidate cached model instance."""
        key = self.key_builder.for_model(model_name, model_id)
        return await self.delete(key)
    
    async def cache_list(self, model_name: str, filters: Dict, data: List[Any],
                         ttl: int = None, strategy: CacheStrategy = None) -> bool:
        """Cache a list of models."""
        key = self.key_builder.for_list(model_name, filters)
        return await self.set(key, data, ttl, strategy=strategy)
    
    async def get_cached_list(self, model_name: str, filters: Dict = None,
                              strategy: CacheStrategy = None) -> Any:
        """Get cached list of models."""
        key = self.key_builder.for_list(model_name, filters)
        return await self.get(key, strategy=strategy)
    
    async def cache_user_data(self, user_id: Any, resource: str, data: Any,
                              ttl: int = None, strategy: CacheStrategy = None) -> bool:
        """Cache user-specific data."""
        key = self.key_builder.for_user(user_id, resource)
        return await self.set(key, data, ttl, strategy=strategy)
    
    async def get_cached_user_data(self, user_id: Any, resource: str,
                                   strategy: CacheStrategy = None) -> Any:
        """Get cached user data."""
        key = self.key_builder.for_user(user_id, resource)
        return await self.get(key, strategy=strategy)
    
    async def invalidate_user_data(self, user_id: Any, resource: str = None) -> int:
        """Invalidate user data (all or specific)."""
        if resource:
            key = self.key_builder.for_user(user_id, resource)
            success = await self.delete(key)
            return 1 if success else 0
        else:
            # Invalidate all user data
            pattern = self.key_builder.build("user", user_id, "*")
            return await self.invalidate_pattern(pattern)
    
    # Metrics and monitoring
    
    def _record_hit(self):
        """Record a cache hit."""
        self.stats.hits += 1
        metrics_client.increment_counter("cache_hits_total")
    
    def _record_miss(self):
        """Record a cache miss."""
        self.stats.misses += 1
        metrics_client.increment_counter("cache_misses_total")
    
    def _record_set(self):
        """Record a cache set operation."""
        self.stats.sets += 1
        metrics_client.increment_counter("cache_sets_total")
    
    def _record_delete(self):
        """Record a cache delete operation."""
        self.stats.deletes += 1
        metrics_client.increment_counter("cache_deletes_total")
    
    def _record_error(self):
        """Record a cache error."""
        self.stats.errors += 1
        metrics_client.increment_counter("cache_errors_total")
    
    def _record_operation(self, operation: CacheOperation, key: str, 
                          duration: float, **extra):
        """Record cache operation metrics."""
        # Update response time average
        total_ops = self.stats.hits + self.stats.misses + self.stats.sets + self.stats.deletes
        if total_ops > 0:
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (total_ops - 1) + duration) / total_ops
            )
        
        # Update hit rate
        total_accesses = self.stats.hits + self.stats.misses
        if total_accesses > 0:
            self.stats.hit_rate = self.stats.hits / total_accesses
        
        # Record to metrics system
        metrics_client.record_histogram(
            "cache_operation_duration_seconds",
            duration,
            operation=operation.value,
        )
        
        # Log slow operations
        if duration > 0.1:  # 100ms threshold
            logger.warning(f"Slow cache operation: {operation.value} for key {key} took {duration:.3f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self.stats.hits + self.stats.misses
        hit_rate = self.stats.hits / total_accesses if total_accesses > 0 else 0
        
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "errors": self.stats.errors,
            "hit_rate": hit_rate,
            "avg_response_time_seconds": self.stats.avg_response_time,
            "total_operations": total_accesses + self.stats.sets + self.stats.deletes,
            "config": {
                "backend": self.config.backend.value,
                "strategy": self.config.strategy.value,
                "default_ttl": self.config.default_ttl,
                "namespace": self.config.namespace,
            },
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache system.
        
        Returns:
            Health status
        """
        start_time = time.time()
        
        try:
            # Test basic operations
            test_key = f"health_check:{int(time.time())}"
            test_value = {"test": True, "timestamp": datetime.utcnow().isoformat()}
            
            # Set
            set_success = await self.set(test_key, test_value, ttl=10)
            if not set_success:
                return {"status": "unhealthy", "error": "Failed to set test value"}
            
            # Get
            retrieved = await self.get(test_key)
            if retrieved is None:
                return {"status": "unhealthy", "error": "Failed to get test value"}
            
            # Delete
            delete_success = await self.delete(test_key)
            if not delete_success:
                return {"status": "degraded", "error": "Failed to delete test value"}
            
            # Check backend-specific health
            backend = self.get_backend()
            backend_health = {}
            if hasattr(backend, 'health_check'):
                backend_health = await backend.health_check()
            
            duration = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time_seconds": duration,
                "backend": self.config.backend.value,
                "backend_health": backend_health,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }
    
    @asynccontextmanager
    async def lock(self, key: str, timeout: int = 10):
        """
        Distributed lock using cache.
        
        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            
        Yields:
            Lock context
        """
        lock_key = f"lock:{key}"
        lock_value = str(time.time())
        
        # Try to acquire lock
        acquired = await self.set(lock_key, lock_value, ttl=timeout, nx=True)
        
        if not acquired:
            raise Exception(f"Failed to acquire lock for key: {key}")
        
        try:
            yield
        finally:
            # Release lock
            current_value = await self.get(lock_key)
            if current_value == lock_value:
                await self.delete(lock_key)


# Global cache manager instance
_cache_manager_instance = None


def get_cache_manager() -> CacheManager:
    """
    Get or create global cache manager instance.
    
    Returns:
        CacheManager instance
    """
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager()
    return _cache_manager_instance


# Decorator for caching function results
def cached(ttl: int = None, key_prefix: str = "func", 
           strategy: CacheStrategy = None, ignore_args: List[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        strategy: Caching strategy to use
        ignore_args: List of argument indices to ignore in key generation
    
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Build cache key
            cache_args = list(args)
            cache_kwargs = kwargs.copy()
            
            # Remove ignored arguments
            if ignore_args:
                for idx in sorted(ignore_args, reverse=True):
                    if idx < len(cache_args):
                        del cache_args[idx]
            
            # Remove 'self' for instance methods, keep class name
            if cache_args and hasattr(cache_args[0], '__class__'):
                cache_args[0] = cache_args[0].__class__.__name__
            
            key = cache_manager.key_builder.build(
                key_prefix, func.__module__, func.__name__, *cache_args, **cache_kwargs
            )
            
            # Try to get from cache
            cached_result = await cache_manager.get(key, strategy=strategy)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if result is not None:
                await cache_manager.set(key, result, ttl, strategy=strategy)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, run in executor
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Export public API
__all__ = [
    # Main class
    "CacheManager",
    
    # Enums
    "CacheOperation",
    "CacheBackend",
    "CacheStrategy",
    
    # Data classes
    "CacheStats",
    "CacheConfig",
    
    # Helper classes
    "CacheKeyBuilder",
    
    # Functions
    "get_cache_manager",
    "cached",
    
    # Convenience functions
    "cache_model",
    "get_cached_model",
    "invalidate_model",
    "cache_list",
    "get_cached_list",
    "cache_user_data",
    "get_cached_user_data",
    "invalidate_user_data",
]