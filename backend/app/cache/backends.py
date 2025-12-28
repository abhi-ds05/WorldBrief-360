"""
Cache backend factory and registry for WorldBrief 360.

This module provides:
- Cache backend factory for creating backend instances
- Backend registry for managing multiple cache backends
- Configuration-based backend selection
- Health monitoring and metrics for all backends
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Union, Type, Callable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.metrics import metrics_client

logger = logging.getLogger(__name__)


class CacheBackendType(Enum):
    """Supported cache backend types."""
    REDIS = "redis"
    MEMORY = "memory"
    FILESYSTEM = "filesystem"
    NULL = "null"  # For testing/disabling cache


@dataclass
class CacheBackendConfig:
    """Configuration for cache backend."""
    backend_type: CacheBackendType = CacheBackendType.REDIS
    name: str = "default"
    url: Optional[str] = None
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    username: Optional[str] = None
    max_connections: int = 20
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 300
    compression: bool = True
    compression_level: int = 6
    serialization: str = "json"  # json, pickle, msgpack
    key_prefix: str = "wb360"
    enable_pubsub: bool = True
    enable_cluster: bool = False
    # Memory backend specific
    max_items: int = 10000
    max_memory_mb: int = 100
    # Filesystem backend specific
    cache_dir: str = "/tmp/wb360_cache"
    max_file_size_mb: int = 10
    # Additional options
    options: Dict[str, Any] = field(default_factory=dict)


class CacheBackend:
    """
    Base cache backend interface.
    All cache backends must implement these methods.
    """
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        raise NotImplementedError
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        raise NotImplementedError
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        raise NotImplementedError
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        raise NotImplementedError
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        raise NotImplementedError
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        raise NotImplementedError
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment integer value."""
        raise NotImplementedError
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement integer value."""
        raise NotImplementedError
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key in seconds."""
        raise NotImplementedError
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for key."""
        raise NotImplementedError
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        raise NotImplementedError
    
    async def close(self):
        """Close connections and cleanup."""
        raise NotImplementedError
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend using asyncio.
    Useful for development/testing or when Redis is not available.
    """
    
    def __init__(self, config: CacheBackendConfig):
        """
        Initialize memory cache backend.
        
        Args:
            config: Cache backend configuration
        """
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._access_times: Dict[str, float] = {}
        self._total_size = 0
        
        logger.info(f"Memory cache backend initialized: {config.name}")
    
    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix."""
        prefix = self.config.key_prefix
        if prefix:
            return f"{prefix}:{key}"
        return key
    
    def _needs_eviction(self) -> bool:
        """Check if eviction is needed."""
        # Check item count
        if len(self._cache) >= self.config.max_items:
            return True
        
        # Check memory usage
        max_memory_bytes = self.config.max_memory_mb * 1024 * 1024
        if max_memory_bytes and self._total_size >= max_memory_bytes:
            return True
        
        return False
    
    def _evict(self, count: int = 1):
        """Evict least recently used items."""
        if not self._cache:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(
            self._access_times.keys(),
            key=lambda k: self._access_times[k]
        )[:count]
        
        for key in sorted_keys:
            if key in self._cache:
                self._total_size -= self._cache[key].get('size', 0)
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        full_key = self._build_key(key)
        
        try:
            if full_key not in self._cache:
                if hasattr(metrics_client, 'increment_counter'):
                    metrics_client.increment_counter("memory_cache_misses")
                return None
            
            entry = self._cache[full_key]
            
            # Check if expired
            if entry['expires'] and time.time() > entry['expires']:
                # Auto-expire
                self._total_size -= entry.get('size', 0)
                del self._cache[full_key]
                if full_key in self._access_times:
                    del self._access_times[full_key]
                
                if hasattr(metrics_client, 'increment_counter'):
                    metrics_client.increment_counter("memory_cache_misses")
                return None
            
            # Update access time
            self._access_times[full_key] = time.time()
            
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("memory_cache_hits")
            
            # Record metrics
            duration = time.time() - start_time
            if hasattr(metrics_client, 'record_histogram'):
                metrics_client.record_histogram(
                    "memory_get_duration_seconds",
                    duration
                )
            
            return entry['value']
            
        except Exception as e:
            logger.error(f"Memory cache GET error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        start_time = time.time()
        full_key = self._build_key(key)
        
        try:
            # Check if we need to evict
            if self._needs_eviction():
                self._evict()
            
            # Calculate size (approximate)
            size = len(str(value).encode())
            
            # Determine expiration
            expires = None
            if ttl is None:
                ttl = self.config.default_ttl
            
            if ttl > 0:
                expires = time.time() + ttl
            
            # Update or create entry
            if full_key in self._cache:
                old_size = self._cache[full_key].get('size', 0)
                self._total_size -= old_size
            
            self._cache[full_key] = {
                'value': value,
                'expires': expires,
                'size': size,
                'set_time': time.time(),
            }
            self._access_times[full_key] = time.time()
            self._total_size += size
            
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("memory_cache_sets")
            
            # Record metrics
            duration = time.time() - start_time
            if hasattr(metrics_client, 'record_histogram'):
                metrics_client.record_histogram(
                    "memory_set_duration_seconds",
                    duration
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cache SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        full_key = self._build_key(key)
        
        try:
            if full_key in self._cache:
                self._total_size -= self._cache[full_key].get('size', 0)
                del self._cache[full_key]
                
                if full_key in self._access_times:
                    del self._access_times[full_key]
                
                if hasattr(metrics_client, 'increment_counter'):
                    metrics_client.increment_counter("memory_cache_deletes")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Memory cache DELETE error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        full_key = self._build_key(key)
        return full_key in self._cache
    
    async def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self._cache.clear()
            self._access_times.clear()
            self._total_size = 0
            
            logger.info("Memory cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern."""
        full_pattern = self._build_key(pattern)
        
        try:
            keys = []
            for key in self._cache.keys():
                # Simple pattern matching (supports only * at end)
                if full_pattern == "*":
                    keys.append(key.replace(f"{self.config.key_prefix}:", ""))
                elif key.startswith(full_pattern.replace("*", "")):
                    keys.append(key.replace(f"{self.config.key_prefix}:", ""))
            
            return keys
            
        except Exception as e:
            logger.error(f"Memory cache KEYS error: {e}")
            return []
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values."""
        result = {}
        
        for key in keys:
            value = await self.get(key)
            if value is not None:
                result[key] = value
        
        return result
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values."""
        success = True
        
        for key, value in items.items():
            if not await self.set(key, value, ttl):
                success = False
        
        return success
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment integer value."""
        full_key = self._build_key(key)
        
        try:
            current = await self.get(key)
            if current is None:
                new_value = amount
            else:
                new_value = int(current) + amount
            
            await self.set(key, new_value)
            return new_value
            
        except Exception as e:
            logger.error(f"Memory cache INCR error: {e}")
            return 0
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement integer value."""
        return await self.increment(key, -amount)
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key in seconds."""
        full_key = self._build_key(key)
        
        try:
            if full_key not in self._cache:
                return -2  # Key doesn't exist
            
            entry = self._cache[full_key]
            if not entry['expires']:
                return -1  # No expiry
            
            ttl = entry['expires'] - time.time()
            return int(ttl) if ttl > 0 else -2
            
        except Exception as e:
            logger.error(f"Memory cache TTL error: {e}")
            return -2
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for key."""
        full_key = self._build_key(key)
        
        try:
            if full_key not in self._cache:
                return False
            
            entry = self._cache[full_key]
            if ttl > 0:
                entry['expires'] = time.time() + ttl
            else:
                entry['expires'] = None
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cache EXPIRE error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "type": "memory",
            "stats": {
                "total_items": len(self._cache),
                "total_size_mb": self._total_size / (1024 * 1024),
                "max_items": self.config.max_items,
                "max_memory_mb": self.config.max_memory_mb,
                "hit_ratio": 0.0,  # Would need tracking
            },
            "config": {
                "key_prefix": self.config.key_prefix,
                "default_ttl": self.config.default_ttl,
            }
        }
    
    async def close(self):
        """Close connections and cleanup."""
        self._cache.clear()
        self._access_times.clear()
        self._total_size = 0
        
        logger.info("Memory cache backend closed")


class NullCacheBackend(CacheBackend):
    """
    Null cache backend that does nothing.
    Useful for testing or disabling cache.
    """
    
    def __init__(self, config: CacheBackendConfig):
        """
        Initialize null cache backend.
        
        Args:
            config: Cache backend configuration
        """
        self.config = config
        logger.info(f"Null cache backend initialized: {config.name}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (always returns None)."""
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache (always returns True)."""
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache (always returns True)."""
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache (always returns False)."""
        return False
    
    async def clear(self) -> bool:
        """Clear all cache entries (always returns True)."""
        return True
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern (always returns empty list)."""
        return []
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values (always returns empty dict)."""
        return {}
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values (always returns True)."""
        return True
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment integer value (always returns amount)."""
        return amount
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """Decrement integer value (always returns -amount)."""
        return -amount
    
    async def ttl(self, key: str) -> int:
        """Get TTL for key in seconds (always returns -2)."""
        return -2
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for key (always returns False)."""
        return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy",
            "type": "null",
            "message": "Null cache backend (no-op)",
        }
    
    async def close(self):
        """Close connections and cleanup."""
        logger.info("Null cache backend closed")


class CacheBackendFactory:
    """
    Factory for creating cache backend instances.
    """
    
    # Registry of backend types
    _backend_classes: Dict[CacheBackendType, Type[CacheBackend]] = {}
    
    @classmethod
    def register_backend(
        cls,
        backend_type: CacheBackendType,
        backend_class: Type[CacheBackend]
    ):
        """
        Register a cache backend class.
        
        Args:
            backend_type: Type of backend
            backend_class: Backend class
        """
        cls._backend_classes[backend_type] = backend_class
        logger.info(f"Registered cache backend: {backend_type.value}")
    
    @classmethod
    def create_backend(
        cls,
        config: CacheBackendConfig
    ) -> CacheBackend:
        """
        Create cache backend instance.
        
        Args:
            config: Cache backend configuration
            
        Returns:
            Cache backend instance
            
        Raises:
            ValueError: If backend type is not supported
        """
        backend_class = cls._backend_classes.get(config.backend_type)
        
        if not backend_class:
            raise ValueError(
                f"Unsupported cache backend type: {config.backend_type.value}. "
                f"Available: {list(cls._backend_classes.keys())}"
            )
        
        return backend_class(config)
    
    @classmethod
    def create_from_settings(
        cls,
        name: str = "default",
        backend_type: Optional[Union[str, CacheBackendType]] = None,
        **kwargs
    ) -> CacheBackend:
        """
        Create cache backend from settings.
        
        Args:
            name: Backend name
            backend_type: Backend type (string or enum)
            **kwargs: Additional configuration
            
        Returns:
            Cache backend instance
        """
        # Determine backend type
        if backend_type is None:
            backend_type_str = getattr(settings, 'CACHE_BACKEND', 'redis')
        elif isinstance(backend_type, CacheBackendType):
            backend_type_str = backend_type.value
        else:
            backend_type_str = backend_type
        
        # Convert string to enum
        try:
            backend_type_enum = CacheBackendType(backend_type_str.lower())
        except ValueError:
            logger.warning(
                f"Unknown cache backend type: {backend_type_str}. "
                f"Using Redis as default."
            )
            backend_type_enum = CacheBackendType.REDIS
        
        # Create configuration
        config = CacheBackendConfig(
            backend_type=backend_type_enum,
            name=name,
            url=getattr(settings, 'REDIS_URL', None),
            host=getattr(settings, 'REDIS_HOST', 'localhost'),
            port=getattr(settings, 'REDIS_PORT', 6379),
            db=getattr(settings, 'REDIS_DB', 0),
            password=getattr(settings, 'REDIS_PASSWORD', None),
            username=getattr(settings, 'REDIS_USERNAME', None),
            max_connections=getattr(settings, 'REDIS_MAX_CONNECTIONS', 20),
            socket_timeout=getattr(settings, 'REDIS_SOCKET_TIMEOUT', 5.0),
            socket_connect_timeout=getattr(settings, 'REDIS_SOCKET_CONNECT_TIMEOUT', 5.0),
            retry_on_timeout=getattr(settings, 'REDIS_RETRY_ON_TIMEOUT', True),
            default_ttl=getattr(settings, 'CACHE_DEFAULT_TTL', 300),
            compression=getattr(settings, 'CACHE_COMPRESSION', True),
            compression_level=getattr(settings, 'CACHE_COMPRESSION_LEVEL', 6),
            serialization=getattr(settings, 'CACHE_SERIALIZATION', 'json'),
            key_prefix=getattr(settings, 'CACHE_KEY_PREFIX', 'wb360'),
            enable_pubsub=getattr(settings, 'CACHE_ENABLE_PUBSUB', True),
            enable_cluster=getattr(settings, 'REDIS_ENABLE_CLUSTER', False),
            max_items=getattr(settings, 'MEMORY_CACHE_MAX_ITEMS', 10000),
            max_memory_mb=getattr(settings, 'MEMORY_CACHE_MAX_MEMORY_MB', 100),
            cache_dir=getattr(settings, 'FILESYSTEM_CACHE_DIR', '/tmp/wb360_cache'),
            max_file_size_mb=getattr(settings, 'FILESYSTEM_CACHE_MAX_FILE_SIZE_MB', 10),
        )
        
        # Update with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.options[key] = value
        
        # Create backend
        return cls.create_backend(config)


class CacheBackendRegistry:
    """
    Registry for managing multiple cache backends.
    """
    
    def __init__(self):
        """Initialize cache backend registry."""
        self._backends: Dict[str, CacheBackend] = {}
        self._default_backend: Optional[str] = None
        self._factory = CacheBackendFactory
        
        # Register default backends
        self._register_default_backends()
    
    def _register_default_backends(self):
        """Register default cache backend implementations."""
        # Register Redis backend (if available)
        try:
            from app.cache.redis_client import RedisCacheBackend
            
            class RedisBackendAdapter(CacheBackend):
                """Adapter for RedisCacheBackend."""
                
                def __init__(self, config: CacheBackendConfig):
                    self.config = config
                    self._redis_backend = RedisCacheBackend(
                        url=config.url,
                        host=config.host,
                        port=config.port,
                        db=config.db,
                        password=config.password,
                        username=config.username,
                        max_connections=config.max_connections,
                        socket_timeout=config.socket_timeout,
                        socket_connect_timeout=config.socket_connect_timeout,
                        retry_on_timeout=config.retry_on_timeout,
                        health_check_interval=config.health_check_interval,
                        default_ttl=config.default_ttl,
                        compression=config.compression,
                        compression_level=config.compression_level,
                        serialization=config.serialization,
                        key_prefix=config.key_prefix,
                        enable_pubsub=config.enable_pubsub,
                        enable_cluster=config.enable_cluster,
                    )
                
                async def get(self, key: str) -> Optional[Any]:
                    return await self._redis_backend.get(key)
                
                async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
                    return await self._redis_backend.set(key, value, ttl)
                
                async def delete(self, key: str) -> bool:
                    return await self._redis_backend.delete(key)
                
                async def exists(self, key: str) -> bool:
                    return await self._redis_backend.exists(key)
                
                async def clear(self) -> bool:
                    return await self._redis_backend.clear()
                
                async def keys(self, pattern: str = "*") -> List[str]:
                    return await self._redis_backend.keys(pattern)
                
                async def get_many(self, keys: List[str]) -> Dict[str, Any]:
                    return await self._redis_backend.get_many(keys)
                
                async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
                    return await self._redis_backend.set_many(items, ttl)
                
                async def increment(self, key: str, amount: int = 1) -> int:
                    return await self._redis_backend.increment(key, amount)
                
                async def decrement(self, key: str, amount: int = 1) -> int:
                    return await self._redis_backend.decrement(key, amount)
                
                async def ttl(self, key: str) -> int:
                    return await self._redis_backend.ttl(key)
                
                async def expire(self, key: str, ttl: int) -> bool:
                    return await self._redis_backend.expire(key, ttl)
                
                async def health_check(self) -> Dict[str, Any]:
                    return await self._redis_backend.health_check()
                
                async def close(self):
                    await self._redis_backend.close()
            
            self._factory.register_backend(CacheBackendType.REDIS, RedisBackendAdapter)
            logger.info("Registered Redis cache backend")
            
        except ImportError as e:
            logger.warning(f"Redis cache backend not available: {e}")
        
        # Register memory backend
        self._factory.register_backend(CacheBackendType.MEMORY, MemoryCacheBackend)
        logger.info("Registered memory cache backend")
        
        # Register null backend
        self._factory.register_backend(CacheBackendType.NULL, NullCacheBackend)
        logger.info("Registered null cache backend")
    
    def register_backend(
        self,
        name: str,
        backend: CacheBackend,
        set_as_default: bool = False
    ):
        """
        Register a cache backend.
        
        Args:
            name: Backend name
            backend: Cache backend instance
            set_as_default: Set as default backend
        """
        self._backends[name] = backend
        
        if set_as_default or self._default_backend is None:
            self._default_backend = name
        
        logger.info(f"Registered cache backend: {name}")
    
    def create_backend(
        self,
        name: str = "default",
        backend_type: Optional[Union[str, CacheBackendType]] = None,
        set_as_default: bool = False,
        **kwargs
    ) -> CacheBackend:
        """
        Create and register a cache backend.
        
        Args:
            name: Backend name
            backend_type: Backend type
            set_as_default: Set as default backend
            **kwargs: Additional configuration
            
        Returns:
            Created cache backend
        """
        backend = self._factory.create_from_settings(
            name=name,
            backend_type=backend_type,
            **kwargs
        )
        
        self.register_backend(name, backend, set_as_default)
        return backend
    
    def get_backend(self, name: str = None) -> CacheBackend:
        """
        Get cache backend by name.
        
        Args:
            name: Backend name (None for default)
            
        Returns:
            Cache backend instance
            
        Raises:
            KeyError: If backend not found
        """
        if name is None:
            if self._default_backend is None:
                raise KeyError("No default cache backend configured")
            name = self._default_backend
        
        if name not in self._backends:
            raise KeyError(f"Cache backend not found: {name}")
        
        return self._backends[name]
    
    def set_default_backend(self, name: str):
        """
        Set default cache backend.
        
        Args:
            name: Backend name
            
        Raises:
            KeyError: If backend not found
        """
        if name not in self._backends:
            raise KeyError(f"Cache backend not found: {name}")
        
        self._default_backend = name
        logger.info(f"Set default cache backend: {name}")
    
    def list_backends(self) -> List[str]:
        """
        List all registered backends.
        
        Returns:
            List of backend names
        """
        return list(self._backends.keys())
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform health check on all backends.
        
        Returns:
            Dictionary of health status for each backend
        """
        results = {}
        
        for name, backend in self._backends.items():
            try:
                health = await backend.health_check()
                results[name] = health
            except Exception as e:
                logger.error(f"Health check failed for backend {name}: {e}")
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        return results
    
    async def close_all(self):
        """Close all cache backends."""
        for name, backend in self._backends.items():
            try:
                await backend.close()
                logger.info(f"Closed cache backend: {name}")
            except Exception as e:
                logger.error(f"Error closing cache backend {name}: {e}")
        
        self._backends.clear()
        self._default_backend = None
    
    @asynccontextmanager
    async def backend_context(self, name: str = None):
        """
        Context manager for cache backend.
        
        Args:
            name: Backend name (None for default)
            
        Yields:
            Cache backend instance
        """
        backend = self.get_backend(name)
        
        try:
            yield backend
        finally:
            # Don't close here, let registry manage lifecycle
            pass


# Global registry instance
_cache_registry: Optional[CacheBackendRegistry] = None

def get_cache_registry() -> CacheBackendRegistry:
    """
    Get or create global cache backend registry.
    
    Returns:
        Cache backend registry
    """
    global _cache_registry
    
    if _cache_registry is None:
        _cache_registry = CacheBackendRegistry()
        logger.info("Created global cache backend registry")
    
    return _cache_registry


def get_cache_backend(name: str = None) -> CacheBackend:
    """
    Get cache backend from global registry.
    
    Args:
        name: Backend name (None for default)
        
    Returns:
        Cache backend instance
    """
    registry = get_cache_registry()
    
    # Create default backend if none exists
    if not registry.list_backends():
        logger.info("No backends registered, creating default backend")
        registry.create_backend("default", set_as_default=True)
    
    return registry.get_backend(name)


def create_cache_backend(
    name: str = "default",
    backend_type: Optional[Union[str, CacheBackendType]] = None,
    set_as_default: bool = False,
    **kwargs
) -> CacheBackend:
    """
    Create and register cache backend.
    
    Args:
        name: Backend name
        backend_type: Backend type
        set_as_default: Set as default backend
        **kwargs: Additional configuration
        
    Returns:
        Created cache backend
    """
    registry = get_cache_registry()
    return registry.create_backend(name, backend_type, set_as_default, **kwargs)


async def close_all_cache_backends():
    """Close all cache backends in global registry."""
    registry = get_cache_registry()
    await registry.close_all()


# Convenience functions for common operations

async def cache_get(
    key: str,
    fetch_func: Optional[Callable] = None,
    ttl: Optional[int] = None,
    backend_name: str = None,
) -> Any:
    """
    Get value from cache with optional fallback.
    
    Args:
        key: Cache key
        fetch_func: Function to fetch data if not in cache
        ttl: TTL for fetched data
        backend_name: Backend name
        
    Returns:
        Cached or fetched value
    """
    backend = get_cache_backend(backend_name)
    
    value = await backend.get(key)
    if value is not None:
        return value
    
    if fetch_func:
        if asyncio.iscoroutinefunction(fetch_func):
            value = await fetch_func()
        else:
            value = fetch_func()
        
        if value is not None:
            await backend.set(key, value, ttl)
        
        return value
    
    return None


async def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    backend_name: str = None,
) -> bool:
    """
    Set value in cache.
    
    Args:
        key: Cache key
        value: Value to cache
        ttl: Time to live
        backend_name: Backend name
        
    Returns:
        True if successful
    """
    backend = get_cache_backend(backend_name)
    return await backend.set(key, value, ttl)


async def cache_delete(
    key: str,
    backend_name: str = None,
) -> bool:
    """
    Delete value from cache.
    
    Args:
        key: Cache key
        backend_name: Backend name
        
    Returns:
        True if successful
    """
    backend = get_cache_backend(backend_name)
    return await backend.delete(key)


async def cache_clear(
    pattern: str = "*",
    backend_name: str = None,
) -> int:
    """
    Clear cache entries matching pattern.
    
    Args:
        pattern: Key pattern
        backend_name: Backend name
        
    Returns:
        Number of entries cleared
    """
    backend = get_cache_backend(backend_name)
    
    if pattern == "*":
        await backend.clear()
        return 1  # Can't know exact count
    
    keys = await backend.keys(pattern)
    deleted = 0
    
    for key in keys:
        if await backend.delete(key):
            deleted += 1
    
    return deleted


async def health_check(backend_name: str = None) -> Dict[str, Any]:
    """
    Perform health check on cache backend.
    
    Args:
        backend_name: Backend name (None for default)
        
    Returns:
        Health status
    """
    backend = get_cache_backend(backend_name)
    return await backend.health_check()


# Initialize default backend on import
def init_default_backend():
    """Initialize default cache backend."""
    try:
        registry = get_cache_registry()
        
        # Check if we already have a backend
        if not registry.list_backends():
            # Create default backend from settings
            registry.create_backend("default", set_as_default=True)
            
            logger.info("Initialized default cache backend")
    except Exception as e:
        logger.error(f"Failed to initialize default cache backend: {e}")


# Auto-initialize on module import
init_default_backend()