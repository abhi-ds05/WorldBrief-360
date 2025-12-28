"""
Redis cache backend for WorldBrief 360.

This module provides a Redis-based cache backend with support for:
- Connection pooling and reconnection
- Compression and serialization
- Atomic operations and transactions
- Pub/Sub for cache invalidation
- Lua scripting for complex operations
- Metrics and monitoring
"""

import json
import pickle
import zlib
import hashlib
import asyncio
import time
import random
from typing import Any, Optional, Dict, List, Union, Tuple, Set, Callable
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import logging

import redis.asyncio as aioredis
from redis.asyncio import Redis, ConnectionPool
from redis.asyncio.client import Pipeline
from redis.exceptions import (
    ConnectionError, TimeoutError, RedisError,
    AuthenticationError, ResponseError
)

# Import from your application
try:
    from app.core.config import settings
    from app.core.metrics import metrics_client
    from app.cache import CacheBackend, CacheStats
except ImportError:
    # Fallback for standalone usage
    import warnings
    warnings.warn("App imports not available, using defaults")
    
    class settings:
        REDIS_URL = "redis://localhost:6379/0"
        REDIS_PASSWORD = None
    
    class metrics_client:
        @staticmethod
        def increment_counter(*args, **kwargs):
            pass
        
        @staticmethod
        def record_histogram(*args, **kwargs):
            pass
    
    class CacheStats:
        def __init__(self):
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.deletes = 0
            self.errors = 0
        
        def to_dict(self):
            return {
                "hits": self.hits,
                "misses": self.misses,
                "sets": self.sets,
                "deletes": self.deletes,
                "errors": self.errors,
                "hit_ratio": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            }
        
        def reset(self):
            self.hits = 0
            self.misses = 0
            self.sets = 0
            self.deletes = 0
            self.errors = 0
    
    class CacheBackend:
        """Base cache backend interface"""
        pass

logger = logging.getLogger(__name__)


class RedisCacheBackend(CacheBackend):
    """
    Redis-based cache backend with advanced features.
    """
    
    def __init__(
        self,
        url: str = None,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str = None,
        username: str = None,
        max_connections: int = 20,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        retry_on_timeout: bool = True,
        health_check_interval: int = 30,
        default_ttl: int = 300,
        compression: bool = True,
        compression_level: int = 6,
        serialization: str = "json",  # json, pickle, msgpack
        key_prefix: str = "wb360",
        enable_pubsub: bool = True,
        enable_cluster: bool = False,
    ):
        """
        Initialize Redis cache backend.
        
        Args:
            url: Redis URL (redis://[:password]@localhost:6379/0)
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            username: Redis username
            max_connections: Maximum number of connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Socket connect timeout in seconds
            retry_on_timeout: Retry on timeout
            health_check_interval: Health check interval in seconds
            default_ttl: Default TTL in seconds
            compression: Enable compression
            compression_level: Compression level (1-9)
            serialization: Serialization format
            key_prefix: Prefix for all cache keys
            enable_pubsub: Enable Pub/Sub for cache invalidation
            enable_cluster: Enable Redis cluster mode
        """
        self.config = {
            "url": url or getattr(settings, 'REDIS_URL', None),
            "host": host,
            "port": port,
            "db": db,
            "password": password or getattr(settings, 'REDIS_PASSWORD', None),
            "username": username,
            "max_connections": max_connections,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
            "retry_on_timeout": retry_on_timeout,
            "health_check_interval": health_check_interval,
            "default_ttl": default_ttl,
            "compression": compression,
            "compression_level": compression_level,
            "serialization": serialization,
            "key_prefix": key_prefix,
            "enable_pubsub": enable_pubsub,
            "enable_cluster": enable_cluster,
        }
        
        self._client: Optional[Redis] = None
        self._pubsub_client: Optional[Redis] = None
        self._pubsub_task: Optional[asyncio.Task] = None
        self._subscriptions: Dict[str, List[Callable]] = {}
        self._connection_pool: Optional[ConnectionPool] = None
        self._stats = CacheStats()
        self._last_health_check = 0
        self._health_status = "unknown"
        self._lua_scripts: Dict[str, str] = {}
        
        # Initialize Lua scripts
        self._init_lua_scripts()
        
        logger.info(f"Redis cache backend initialized for {self.config['host']}:{self.config['port']}")
    
    def _init_lua_scripts(self):
        """Initialize Lua scripts for atomic operations."""
        # Atomic get and refresh TTL
        self._lua_scripts["get_and_refresh"] = """
            local key = KEYS[1]
            local ttl = ARGV[1]
            local value = redis.call('GET', key)
            if value and ttl and tonumber(ttl) > 0 then
                redis.call('EXPIRE', key, ttl)
            end
            return value
        """
        
        # Atomic compare and delete
        self._lua_scripts["compare_and_delete"] = """
            local key = KEYS[1]
            local expected = ARGV[1]
            local current = redis.call('GET', key)
            if current == expected then
                return redis.call('DEL', key)
            end
            return 0
        """
        
        # Increment with max limit
        self._lua_scripts["increment_with_limit"] = """
            local key = KEYS[1]
            local increment = tonumber(ARGV[1])
            local max_value = tonumber(ARGV[2])
            local ttl = tonumber(ARGV[3])
            
            local current = redis.call('GET', key)
            if not current then
                current = 0
            else
                current = tonumber(current)
            end
            
            local new_value = current + increment
            if max_value and new_value > max_value then
                return {current, 0}
            end
            
            redis.call('SET', key, new_value)
            if ttl and ttl > 0 then
                redis.call('EXPIRE', key, ttl)
            end
            
            return {new_value, 1}
        """
        
        # Delete keys by pattern (scan-based, safe for large datasets)
        self._lua_scripts["delete_pattern"] = """
            local pattern = ARGV[1]
            local limit = tonumber(ARGV[2]) or 1000
            local cursor = '0'
            local deleted = 0
            
            repeat
                local result = redis.call('SCAN', cursor, 'MATCH', pattern, 'COUNT', limit)
                cursor = result[1]
                local keys = result[2]
                
                if #keys > 0 then
                    redis.call('DEL', unpack(keys))
                    deleted = deleted + #keys
                end
            until cursor == '0' or (limit and deleted >= limit)
            
            return deleted
        """
        
        # Get multiple keys with TTL refresh
        self._lua_scripts["mget_and_refresh"] = """
            local keys = KEYS
            local ttl = ARGV[1]
            local values = redis.call('MGET', unpack(keys))
            
            if ttl and tonumber(ttl) > 0 then
                for i, key in ipairs(keys) do
                    if values[i] then
                        redis.call('EXPIRE', key, ttl)
                    end
                end
            end
            
            return values
        """
        
        # Get or set with fallback
        self._lua_scripts["get_or_set"] = """
            local key = KEYS[1]
            local value = ARGV[1]
            local ttl = tonumber(ARGV[2])
            
            local existing = redis.call('GET', key)
            if existing then
                return {existing, 0}  -- 0 means retrieved from cache
            end
            
            redis.call('SET', key, value)
            if ttl and ttl > 0 then
                redis.call('EXPIRE', key, ttl)
            end
            
            return {value, 1}  -- 1 means set in cache
        """
    
    async def _get_client(self) -> Redis:
        """
        Get Redis client, creating if necessary.
        
        Returns:
            Redis client
        """
        if self._client is None or await self._check_connection() is False:
            await self._connect()
        
        return self._client
    
    async def _connect(self):
        """Establish connection to Redis."""
        try:
            # Close existing connection if any
            if self._client:
                await self._client.close()
            
            # Create connection pool
            if self.config["enable_cluster"]:
                from redis.asyncio.cluster import RedisCluster
                self._client = RedisCluster.from_url(
                    self.config["url"] or f"redis://{self.config['host']}:{self.config['port']}",
                    password=self.config["password"],
                    username=self.config["username"],
                    socket_timeout=self.config["socket_timeout"],
                    socket_connect_timeout=self.config["socket_connect_timeout"],
                    retry_on_timeout=self.config["retry_on_timeout"],
                    max_connections=self.config["max_connections"],
                )
            else:
                # Parse URL or use individual parameters
                if self.config["url"]:
                    self._connection_pool = ConnectionPool.from_url(
                        self.config["url"],
                        max_connections=self.config["max_connections"],
                        socket_timeout=self.config["socket_timeout"],
                        socket_connect_timeout=self.config["socket_connect_timeout"],
                        retry_on_timeout=self.config["retry_on_timeout"],
                        health_check_interval=self.config["health_check_interval"],
                    )
                else:
                    self._connection_pool = ConnectionPool(
                        host=self.config["host"],
                        port=self.config["port"],
                        db=self.config["db"],
                        password=self.config["password"],
                        username=self.config["username"],
                        max_connections=self.config["max_connections"],
                        socket_timeout=self.config["socket_timeout"],
                        socket_connect_timeout=self.config["socket_connect_timeout"],
                        retry_on_timeout=self.config["retry_on_timeout"],
                        health_check_interval=self.config["health_check_interval"],
                    )
                
                self._client = Redis(connection_pool=self._connection_pool)
            
            # Test connection
            await self._client.ping()
            self._health_status = "healthy"
            
            logger.info("Redis connection established successfully")
            
            # Start Pub/Sub if enabled
            if self.config["enable_pubsub"]:
                await self._start_pubsub()
            
        except (ConnectionError, AuthenticationError, TimeoutError) as e:
            self._health_status = "unhealthy"
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            self._health_status = "unhealthy"
            logger.error(f"Unexpected error connecting to Redis: {e}")
            raise
    
    async def _check_connection(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            True if connection is healthy
        """
        # Only check periodically
        now = time.time()
        if now - self._last_health_check < 30:  # Check every 30 seconds
            return self._health_status == "healthy"
        
        self._last_health_check = now
        
        if self._client is None:
            return False
        
        try:
            await self._client.ping()
            self._health_status = "healthy"
            return True
        except Exception as e:
            logger.warning(f"Redis connection check failed: {e}")
            self._health_status = "unhealthy"
            return False
    
    async def _start_pubsub(self):
        """Start Pub/Sub for cache invalidation."""
        try:
            # Create separate client for Pub/Sub
            if self.config["url"]:
                self._pubsub_client = aioredis.from_url(
                    self.config["url"],
                    max_connections=5,
                    decode_responses=True,
                )
            else:
                self._pubsub_client = aioredis.Redis(
                    host=self.config["host"],
                    port=self.config["port"],
                    db=self.config["db"],
                    password=self.config["password"],
                    username=self.config["username"],
                    max_connections=5,
                    decode_responses=True,
                )
            
            # Start listening in background
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())
            logger.info("Redis Pub/Sub started")
            
        except Exception as e:
            logger.error(f"Failed to start Redis Pub/Sub: {e}")
    
    async def _pubsub_listener(self):
        """Listen for Pub/Sub messages."""
        pubsub = self._pubsub_client.pubsub()
        
        try:
            # Subscribe to all channels with callbacks
            channels = list(self._subscriptions.keys())
            if channels:
                await pubsub.subscribe(*channels)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    channel = message["channel"]
                    data = message["data"]
                    
                    try:
                        # Parse JSON data
                        parsed_data = json.loads(data)
                    except:
                        parsed_data = {"data": data}
                    
                    # Call registered callbacks
                    callbacks = self._subscriptions.get(channel, [])
                    for callback in callbacks:
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(parsed_data)
                            else:
                                callback(parsed_data)
                        except Exception as e:
                            logger.error(f"PubSub callback error: {e}")
        
        except Exception as e:
            logger.error(f"PubSub listener error: {e}")
        finally:
            await pubsub.close()
    
    def _build_key(self, key: str) -> str:
        """
        Build full cache key with prefix.
        
        Args:
            key: Original key
            
        Returns:
            Prefixed key
        """
        prefix = self.config["key_prefix"]
        if prefix:
            return f"{prefix}:{key}"
        return key
    
    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value for storage.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized bytes
        """
        try:
            if self.config["serialization"] == "pickle":
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.config["serialization"] == "msgpack":
                import msgpack
                data = msgpack.packb(value, use_bin_type=True)
            else:  # json
                data = json.dumps(value, separators=(',', ':')).encode('utf-8')
            
            # Compress if enabled
            if self.config["compression"] and len(data) > 100:  # Only compress larger data
                data = zlib.compress(data, level=self.config["compression_level"])
                # Add compression marker
                data = b'c' + data
            
            return data
            
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize value from storage.
        
        Args:
            data: Serialized bytes
            
        Returns:
            Deserialized value
        """
        if not data:
            return None
        
        try:
            # Check for compression marker
            if data.startswith(b'c'):
                data = zlib.decompress(data[1:])
            
            if self.config["serialization"] == "pickle":
                return pickle.loads(data)
            elif self.config["serialization"] == "msgpack":
                import msgpack
                return msgpack.unpackb(data, raw=False)
            else:  # json
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            return None
    
    @asynccontextmanager
    async def pipeline(self, transaction: bool = True):
        """
        Context manager for Redis pipeline/transaction.
        
        Args:
            transaction: Enable transaction (MULTI/EXEC)
            
        Yields:
            Redis pipeline
        """
        client = await self._get_client()
        async with client.pipeline(transaction=transaction) as pipe:
            try:
                yield pipe
                await pipe.execute()
            except Exception as e:
                await pipe.reset()
                raise
    
    async def execute_lua(self, script_name: str, keys: List[str] = None, 
                          args: List[Any] = None) -> Any:
        """
        Execute Lua script.
        
        Args:
            script_name: Name of Lua script
            keys: List of keys
            args: List of arguments
            
        Returns:
            Script result
        """
        if script_name not in self._lua_scripts:
            raise ValueError(f"Unknown Lua script: {script_name}")
        
        client = await self._get_client()
        script = self._lua_scripts[script_name]
        
        keys = keys or []
        args = args or []
        
        # Prepend key prefix to keys
        keys = [self._build_key(k) for k in keys]
        
        try:
            return await client.eval(script, len(keys), *keys, *args)
        except ResponseError as e:
            logger.error(f"Lua script error: {e}")
            raise
    
    # Core CacheBackend interface implementation
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        start_time = time.time()
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            data = await client.get(full_key)
            
            if data is None:
                self._stats.misses += 1
                if hasattr(metrics_client, 'increment_counter'):
                    metrics_client.increment_counter("redis_cache_misses")
                return None
            
            value = self._deserialize(data)
            self._stats.hits += 1
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_cache_hits")
            
            # Record metrics
            duration = time.time() - start_time
            if hasattr(metrics_client, 'record_histogram'):
                metrics_client.record_histogram(
                    "redis_get_duration_seconds",
                    duration,
                    hit="true"
                )
            
            if duration > 0.05:  # 50ms threshold
                logger.warning(f"Slow Redis GET: {key} took {duration:.3f}s")
            
            return value
            
        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.error(f"Redis connection error for GET {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="connection")
            return None
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis GET error for key {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="operation")
            return None
    
    async def get_and_refresh(self, key: str, ttl: int = None) -> Optional[Any]:
        """
        Get value and refresh TTL.
        
        Args:
            key: Cache key
            ttl: New TTL in seconds
            
        Returns:
            Cached value or None
        """
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        result = await self.execute_lua(
            "get_and_refresh",
            keys=[key],
            args=[ttl]
        )
        
        if result:
            return self._deserialize(result)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        start_time = time.time()
        full_key = self._build_key(key)
        
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        try:
            data = self._serialize(value)
            client = await self._get_client()
            
            if ttl > 0:
                await client.setex(full_key, ttl, data)
            else:
                await client.set(full_key, data)
            
            self._stats.sets += 1
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_cache_sets")
            
            # Record metrics
            duration = time.time() - start_time
            if hasattr(metrics_client, 'record_histogram'):
                metrics_client.record_histogram(
                    "redis_set_duration_seconds",
                    duration,
                    ttl=str(ttl)
                )
            
            # Publish invalidation if value is None (deletion via set)
            if value is None and self.config["enable_pubsub"]:
                await self.publish_invalidation(key)
            
            return True
            
        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.error(f"Redis connection error for SET {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="connection")
            return False
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis SET error for key {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="operation")
            return False
    
    async def setnx(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value if key does not exist (atomic).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if key was set
        """
        full_key = self._build_key(key)
        
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        try:
            data = self._serialize(value)
            client = await self._get_client()
            
            if ttl > 0:
                # Use SET with NX and EX options
                result = await client.set(full_key, data, ex=ttl, nx=True)
            else:
                result = await client.set(full_key, data, nx=True)
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis SETNX error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        start_time = time.time()
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            result = await client.delete(full_key)
            
            self._stats.deletes += 1
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_cache_deletes")
            
            # Record metrics
            duration = time.time() - start_time
            if hasattr(metrics_client, 'record_histogram'):
                metrics_client.record_histogram("redis_delete_duration_seconds", duration)
            
            # Publish invalidation
            if self.config["enable_pubsub"]:
                await self.publish_invalidation(key)
            
            return result > 0
            
        except (ConnectionError, TimeoutError) as e:
            self._stats.errors += 1
            logger.error(f"Redis connection error for DELETE {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="connection")
            return False
        except Exception as e:
            self._stats.errors += 1
            logger.error(f"Redis DELETE error for key {key}: {e}")
            if hasattr(metrics_client, 'increment_counter'):
                metrics_client.increment_counter("redis_errors", type="operation")
            return False
    
    async def delete_pattern(self, pattern: str, limit: int = 1000) -> int:
        """
        Delete keys matching pattern.
        
        Args:
            pattern: Pattern to match
            limit: Maximum number of keys to delete
            
        Returns:
            Number of keys deleted
        """
        full_pattern = self._build_key(pattern)
        
        try:
            deleted = await self.execute_lua(
                "delete_pattern",
                args=[full_pattern, limit]
            )
            
            # Publish invalidation for pattern
            if self.config["enable_pubsub"]:
                await self.publish_invalidation(pattern, is_pattern=True)
            
            logger.info(f"Deleted {deleted} keys matching pattern: {pattern}")
            return deleted
            
        except Exception as e:
            logger.error(f"Redis delete pattern error: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            return await client.exists(full_key) > 0
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cache entries (flush database).
        
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            
            if self.config["key_prefix"]:
                # Delete only keys with our prefix
                pattern = f"{self.config['key_prefix']}:*"
                await self.delete_pattern(pattern)
            else:
                # Flush entire database
                await client.flushdb()
            
            logger.info("Redis cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """
        Get keys matching pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            List of matching keys
        """
        full_pattern = self._build_key(pattern)
        
        try:
            client = await self._get_client()
            keys = await client.keys(full_pattern)
            
            # Remove prefix from keys
            if self.config["key_prefix"]:
                prefix = f"{self.config['key_prefix']}:"
                keys = [k.decode().replace(prefix, "") if isinstance(k, bytes) 
                       else k.replace(prefix, "") for k in keys]
            
            return keys
            
        except Exception as e:
            logger.error(f"Redis KEYS error for pattern {pattern}: {e}")
            return []
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment integer value.
        
        Args:
            key: Cache key
            amount: Amount to increment
            
        Returns:
            New value
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            return await client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis INCR error for key {key}: {e}")
            return 0
    
    async def increment_with_limit(self, key: str, amount: int = 1, 
                                   max_value: int = None, ttl: int = None) -> Tuple[int, bool]:
        """
        Increment with maximum limit.
        
        Args:
            key: Cache key
            amount: Amount to increment
            max_value: Maximum allowed value
            ttl: TTL in seconds
            
        Returns:
            Tuple of (new_value, success)
        """
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        try:
            result = await self.execute_lua(
                "increment_with_limit",
                keys=[key],
                args=[amount, max_value or 0, ttl]
            )
            
            return result[0], bool(result[1])
            
        except Exception as e:
            logger.error(f"Redis increment with limit error: {e}")
            return 0, False
    
    async def decrement(self, key: str, amount: int = 1) -> int:
        """
        Decrement integer value.
        
        Args:
            key: Cache key
            amount: Amount to decrement
            
        Returns:
            New value
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            return await client.decrby(full_key, amount)
        except Exception as e:
            logger.error(f"Redis DECR error for key {key}: {e}")
            return 0
    
    async def ttl(self, key: str) -> int:
        """
        Get TTL for key in seconds.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            return await client.ttl(full_key)
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -2
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set TTL for key.
        
        Args:
            key: Cache key
            ttl: TTL in seconds
            
        Returns:
            True if successful
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            return await client.expire(full_key, ttl)
        except Exception as e:
            logger.error(f"Redis EXPIRE error for key {key}: {e}")
            return False
    
    # Advanced operations
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        if not keys:
            return {}
        
        full_keys = [self._build_key(k) for k in keys]
        
        try:
            client = await self._get_client()
            values = await client.mget(full_keys)
            
            result = {}
            for i, key in enumerate(keys):
                if values[i] is not None:
                    result[key] = self._deserialize(values[i])
            
            return result
            
        except Exception as e:
            logger.error(f"Redis MGET error: {e}")
            return {}
    
    async def set_many(self, items: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set multiple values.
        
        Args:
            items: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if successful
        """
        if not items:
            return True
        
        if ttl is None:
            ttl = self.config["default_ttl"]
        
        try:
            client = await self._get_client()
            
            async with client.pipeline() as pipe:
                for key, value in items.items():
                    full_key = self._build_key(key)
                    data = self._serialize(value)
                    
                    if ttl > 0:
                        pipe.setex(full_key, ttl, data)
                    else:
                        pipe.set(full_key, data)
                
                await pipe.execute()
            
            # Publish invalidations
            if self.config["enable_pubsub"]:
                for key in items.keys():
                    await self.publish_invalidation(key)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis MSET error: {e}")
            return False
    
    async def publish_invalidation(self, key: str, is_pattern: bool = False):
        """
        Publish cache invalidation message.
        
        Args:
            key: Cache key or pattern
            is_pattern: Whether key is a pattern
        """
        if not self.config["enable_pubsub"] or not self._pubsub_client:
            return
        
        try:
            channel = "cache:invalidate"
            message = {
                "key": key,
                "is_pattern": is_pattern,
                "timestamp": datetime.utcnow().isoformat(),
                "source": self.config["key_prefix"]
            }
            
            await self._pubsub_client.publish(channel, json.dumps(message))
            
        except Exception as e:
            logger.error(f"Redis publish error: {e}")
    
    async def subscribe_invalidation(self, callback: Callable[[Dict], None]):
        """
        Subscribe to cache invalidation messages.
        
        Args:
            callback: Callback function receiving message dict
        """
        if not self.config["enable_pubsub"]:
            return
        
        channel = "cache:invalidate"
        
        if channel not in self._subscriptions:
            self._subscriptions[channel] = []
        
        self._subscriptions[channel].append(callback)
        
        # Resubscribe if Pub/Sub is running
        if self._pubsub_client:
            await self._pubsub_client.subscribe(channel)
    
    # Hash operations
    
    async def hset(self, key: str, field: str, value: Any) -> bool:
        """
        Set hash field.
        
        Args:
            key: Cache key
            field: Hash field
            value: Value to set
            
        Returns:
            True if successful
        """
        full_key = self._build_key(key)
        data = self._serialize(value)
        
        try:
            client = await self._get_client()
            return await client.hset(full_key, field, data) > 0
        except Exception as e:
            logger.error(f"Redis HSET error for key {key}.{field}: {e}")
            return False
    
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """
        Get hash field.
        
        Args:
            key: Cache key
            field: Hash field
            
        Returns:
            Field value or None
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            data = await client.hget(full_key, field)
            
            if data is None:
                return None
            
            return self._deserialize(data)
            
        except Exception as e:
            logger.error(f"Redis HGET error for key {key}.{field}: {e}")
            return None
    
    async def hgetall(self, key: str) -> Dict[str, Any]:
        """
        Get all hash fields.
        
        Args:
            key: Cache key
            
        Returns:
            Dictionary of field-value pairs
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            data = await client.hgetall(full_key)
            
            result = {}
            for field, value in data.items():
                if isinstance(field, bytes):
                    field = field.decode('utf-8')
                result[field] = self._deserialize(value)
            
            return result
            
        except Exception as e:
            logger.error(f"Redis HGETALL error for key {key}: {e}")
            return {}
    
    # Set operations
    
    async def sadd(self, key: str, *members: Any) -> int:
        """
        Add members to set.
        
        Args:
            key: Cache key
            *members: Members to add
            
        Returns:
            Number of members added
        """
        full_key = self._build_key(key)
        serialized_members = [self._serialize(m) for m in members]
        
        try:
            client = await self._get_client()
            return await client.sadd(full_key, *serialized_members)
        except Exception as e:
            logger.error(f"Redis SADD error for key {key}: {e}")
            return 0
    
    async def smembers(self, key: str) -> Set[Any]:
        """
        Get all set members.
        
        Args:
            key: Cache key
            
        Returns:
            Set of members
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            data = await client.smembers(full_key)
            
            result = set()
            for member in data:
                result.add(self._deserialize(member))
            
            return result
            
        except Exception as e:
            logger.error(f"Redis SMEMBERS error for key {key}: {e}")
            return set()
    
    # List operations
    
    async def lpush(self, key: str, *values: Any) -> int:
        """
        Push values to list (left).
        
        Args:
            key: Cache key
            *values: Values to push
            
        Returns:
            List length after push
        """
        full_key = self._build_key(key)
        serialized_values = [self._serialize(v) for v in values]
        
        try:
            client = await self._get_client()
            return await client.lpush(full_key, *serialized_values)
        except Exception as e:
            logger.error(f"Redis LPUSH error for key {key}: {e}")
            return 0
    
    async def rpush(self, key: str, *values: Any) -> int:
        """
        Push values to list (right).
        
        Args:
            key: Cache key
            *values: Values to push
            
        Returns:
            List length after push
        """
        full_key = self._build_key(key)
        serialized_values = [self._serialize(v) for v in values]
        
        try:
            client = await self._get_client()
            return await client.rpush(full_key, *serialized_values)
        except Exception as e:
            logger.error(f"Redis RPUSH error for key {key}: {e}")
            return 0
    
    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """
        Get range of list elements.
        
        Args:
            key: Cache key
            start: Start index
            end: End index
            
        Returns:
            List of values
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            data = await client.lrange(full_key, start, end)
            
            result = []
            for item in data:
                result.append(self._deserialize(item))
            
            return result
            
        except Exception as e:
            logger.error(f"Redis LRANGE error for key {key}: {e}")
            return []
    
    # Health and monitoring
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status
        """
        start_time = time.time()
        
        health_info = {
            "status": "unknown",
            "latency": 0,
            "version": None,
            "memory": None,
            "clients": None,
            "keys": 0,
            "timestamp": datetime.utcnow().isoformat(),
            "stats": self._stats.to_dict() if hasattr(self._stats, 'to_dict') else self._stats.__dict__,
        }
        
        try:
            client = await self._get_client()
            
            # Check basic connectivity
            pong = await client.ping()
            if not pong:
                health_info["status"] = "unhealthy"
                self._health_status = "unhealthy"
                return health_info
            
            # Get Redis info
            info = await client.info()
            
            health_info.update({
                "status": "healthy",
                "version": info.get("redis_version"),
                "memory": {
                    "used": info.get("used_memory"),
                    "peak": info.get("used_memory_peak"),
                    "fragmentation": info.get("mem_fragmentation_ratio"),
                },
                "clients": {
                    "connected": info.get("connected_clients"),
                    "blocked": info.get("blocked_clients"),
                },
                "stats": {
                    "ops_per_sec": info.get("instantaneous_ops_per_sec"),
                    "hits": info.get("keyspace_hits"),
                    "misses": info.get("keyspace_misses"),
                }
            })
            
            # Count our keys
            if self.config["key_prefix"]:
                pattern = f"{self.config['key_prefix']}:*"
                keys = await client.keys(pattern)
                health_info["keys"] = len(keys)
            else:
                # Get total keys in database
                db_key = f"db{self.config['db']}"
                if db_key in info:
                    health_info["keys"] = info[db_key].get("keys", 0)
            
            self._health_status = "healthy"
            
        except (ConnectionError, TimeoutError) as e:
            health_info.update({
                "status": "unhealthy",
                "error": f"Connection error: {str(e)}"
            })
            self._health_status = "unhealthy"
            logger.error(f"Redis health check failed: {e}")
            
        except Exception as e:
            health_info.update({
                "status": "unhealthy",
                "error": f"Unexpected error: {str(e)}"
            })
            self._health_status = "unhealthy"
            logger.error(f"Redis health check failed: {e}")
        
        finally:
            health_info["latency"] = time.time() - start_time
        
        return health_info
    
    async def get_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return self._stats
    
    async def reset_stats(self):
        """Reset cache statistics."""
        self._stats.reset() if hasattr(self._stats, 'reset') else self._stats.__init__()
    
    async def memory_usage(self, key: str) -> Optional[int]:
        """
        Get memory usage of a key.
        
        Args:
            key: Cache key
            
        Returns:
            Memory usage in bytes or None
        """
        full_key = self._build_key(key)
        
        try:
            client = await self._get_client()
            # MEMORY USAGE command might not be available in all Redis versions
            if hasattr(client, 'memory_usage'):
                return await client.memory_usage(full_key)
            else:
                # Fallback: get value and calculate size
                value = await client.get(full_key)
                if value:
                    return len(value)
        except Exception as e:
            logger.error(f"Redis memory usage error for key {key}: {e}")
        
        return None
    
    async def scan_keys(self, pattern: str = "*", count: int = 100) -> List[str]:
        """
        Scan keys using Redis SCAN command (safe for production).
        
        Args:
            pattern: Pattern to match
            count: Approximate number of keys to return per batch
            
        Returns:
            List of keys
        """
        full_pattern = self._build_key(pattern)
        cursor = 0
        keys = []
        
        try:
            client = await self._get_client()
            
            while True:
                cursor, batch = await client.scan(
                    cursor=cursor,
                    match=full_pattern,
                    count=count
                )
                
                # Remove prefix from keys
                if self.config["key_prefix"]:
                    prefix = f"{self.config['key_prefix']}:"
                    batch = [
                        k.decode().replace(prefix, "") if isinstance(k, bytes) 
                        else k.replace(prefix, "") 
                        for k in batch
                    ]
                
                keys.extend(batch)
                
                if cursor == 0:
                    break
        
        except Exception as e:
            logger.error(f"Redis SCAN error: {e}")
        
        return keys
    
    async def get_with_fallback(
        self,
        key: str,
        fallback_func: Callable[[], Any],
        ttl: Optional[int] = None,
        refresh_on_access: bool = False,
    ) -> Any:
        """
        Get value from cache or compute using fallback function.
        
        Args:
            key: Cache key
            fallback_func: Function to compute value if not in cache
            ttl: Time to live in seconds
            refresh_on_access: Refresh TTL on access
            
        Returns:
            Value from cache or fallback
        """
        # Try to get from cache
        if refresh_on_access:
            value = await self.get_and_refresh(key, ttl)
        else:
            value = await self.get(key)
        
        # If found in cache, return it
        if value is not None:
            return value
        
        # Compute using fallback
        try:
            if asyncio.iscoroutinefunction(fallback_func):
                computed_value = await fallback_func()
            else:
                computed_value = fallback_func()
        except Exception as e:
            logger.error(f"Fallback function failed for key {key}: {e}")
            raise
        
        # Store in cache
        if computed_value is not None:
            await self.set(key, computed_value, ttl)
        
        return computed_value
    
    async def lock(
        self,
        key: str,
        timeout: int = 10,
        blocking_timeout: int = 5,
        sleep: float = 0.1,
    ) -> bool:
        """
        Acquire a distributed lock.
        
        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            blocking_timeout: Maximum time to wait for lock
            sleep: Sleep interval between retries
            
        Returns:
            True if lock acquired
        """
        lock_key = f"lock:{key}"
        
        # Try to acquire lock with SET NX EX
        acquired = await self.setnx(lock_key, 1, ttl=timeout)
        
        if acquired or blocking_timeout <= 0:
            return acquired
        
        # Wait for lock with timeout
        start_time = time.time()
        while time.time() - start_time < blocking_timeout:
            await asyncio.sleep(sleep)
            
            acquired = await self.setnx(lock_key, 1, ttl=timeout)
            if acquired:
                return True
        
        return False
    
    async def unlock(self, key: str) -> bool:
        """
        Release a distributed lock.
        
        Args:
            key: Lock key
            
        Returns:
            True if lock released
        """
        lock_key = f"lock:{key}"
        return await self.delete(lock_key)
    
    @asynccontextmanager
    async def distributed_lock(
        self,
        key: str,
        timeout: int = 10,
        blocking_timeout: int = 5,
    ):
        """
        Context manager for distributed lock.
        
        Args:
            key: Lock key
            timeout: Lock timeout in seconds
            blocking_timeout: Maximum time to wait for lock
            
        Yields:
            Lock context
        """
        acquired = await self.lock(key, timeout, blocking_timeout)
        
        if not acquired:
            raise TimeoutError(f"Could not acquire lock for key: {key}")
        
        try:
            yield
        finally:
            await self.unlock(key)
    
    async def rate_limit(
        self,
        key: str,
        limit: int,
        period: int,
        increment: int = 1,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Rate limiting using sliding window algorithm.
        
        Args:
            key: Rate limit key
            limit: Maximum number of requests
            period: Time period in seconds
            increment: Number of requests to count
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        current_time = time.time()
        window_start = current_time - period
        
        full_key = self._build_key(f"ratelimit:{key}")
        
        try:
            client = await self._get_client()
            
            # Remove old entries
            await client.zremrangebyscore(full_key, 0, window_start)
            
            # Count current requests
            current_count = await client.zcard(full_key)
            
            # Check if limit is exceeded
            if current_count + increment > limit:
                # Get oldest request time
                oldest = await client.zrange(full_key, 0, 0, withscores=True)
                retry_after = 0
                if oldest:
                    retry_after = int(oldest[0][1] + period - current_time)
                
                return False, {
                    "limit": limit,
                    "remaining": max(0, limit - current_count),
                    "reset": int(window_start + period),
                    "retry_after": max(0, retry_after),
                }
            
            # Add current request
            member = f"{current_time}:{hashlib.md5(str(current_time).encode()).hexdigest()[:8]}"
            await client.zadd(full_key, {member: current_time})
            
            # Set TTL for the key
            await client.expire(full_key, period + 1)
            
            return True, {
                "limit": limit,
                "remaining": limit - current_count - increment,
                "reset": int(window_start + period),
            }
            
        except Exception as e:
            logger.error(f"Rate limit error for key {key}: {e}")
            # Fail open in case of Redis errors
            return True, {
                "limit": limit,
                "remaining": limit,
                "reset": int(current_time + period),
            }
    
    async def close(self):
        """
        Close Redis connections.
        """
        try:
            if self._pubsub_task and not self._pubsub_task.done():
                self._pubsub_task.cancel()
                try:
                    await self._pubsub_task
                except asyncio.CancelledError:
                    pass
            
            if self._pubsub_client:
                await self._pubsub_client.close()
            
            if self._client:
                await self._client.close()
            
            if self._connection_pool:
                await self._connection_pool.disconnect()
            
            logger.info("Redis connections closed")
            
        except Exception as e:
            logger.error(f"Error closing Redis connections: {e}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    # Cache warming and maintenance
    
    async def warm_cache(self, keys_pattern: str = "*", batch_size: int = 100):
        """
        Warm cache by preloading keys (touch to refresh TTL).
        
        Args:
            keys_pattern: Pattern of keys to warm
            batch_size: Batch size for scanning
        """
        logger.info(f"Starting cache warm-up for pattern: {keys_pattern}")
        
        warmed = 0
        failed = 0
        
        keys = await self.scan_keys(keys_pattern, batch_size)
        
        for i in range(0, len(keys), batch_size):
            batch = keys[i:i + batch_size]
            
            # Use pipeline to refresh TTLs
            try:
                client = await self._get_client()
                async with client.pipeline() as pipe:
                    for key in batch:
                        full_key = self._build_key(key)
                        pipe.expire(full_key, self.config["default_ttl"])
                    
                    results = await pipe.execute()
                    warmed += sum(1 for r in results if r)
                    failed += sum(1 for r in results if not r)
            
            except Exception as e:
                logger.error(f"Error warming cache batch: {e}")
                failed += len(batch)
            
            # Small delay to avoid overwhelming Redis
            if i + batch_size < len(keys):
                await asyncio.sleep(0.01)
        
        logger.info(f"Cache warm-up completed: {warmed} keys warmed, {failed} failed")
    
    async def analyze_cache_efficiency(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze cache efficiency by sampling keys.
        
        Args:
            sample_size: Number of keys to sample
            
        Returns:
            Efficiency analysis
        """
        # Get random sample of keys
        try:
            client = await self._get_client()
            
            if self.config["key_prefix"]:
                pattern = f"{self.config['key_prefix']}:*"
                keys = await client.keys(pattern)
            else:
                keys = await client.keys("*")
            
            if not keys:
                return {"message": "No keys found in cache"}
            
            # Sample random keys
            sample_keys = random.sample(keys, min(sample_size, len(keys)))
            
            analysis = {
                "total_keys": len(keys),
                "sample_size": len(sample_keys),
                "key_sizes": [],
                "ttl_distribution": {},
                "serialization_stats": {
                    "compressed": 0,
                    "uncompressed": 0,
                }
            }
            
            # Analyze each key in sample
            async with client.pipeline() as pipe:
                for key in sample_keys:
                    pipe.strlen(key)
                    pipe.ttl(key)
                    pipe.get(key)
                
                results = await pipe.execute()
            
            for i in range(0, len(results), 3):
                size = results[i]
                ttl = results[i + 1]
                value = results[i + 2]
                
                if size is not None:
                    analysis["key_sizes"].append(size)
                
                if ttl is not None:
                    ttl_bucket = "no_expiry" if ttl == -1 else \
                                f"<1min" if ttl < 60 else \
                                f"1-5min" if ttl < 300 else \
                                f"5-30min" if ttl < 1800 else \
                                f"30min-2h" if ttl < 7200 else \
                                f"2-12h" if ttl < 43200 else \
                                f"12-24h" if ttl < 86400 else \
                                f">24h"
                    
                    analysis["ttl_distribution"][ttl_bucket] = \
                        analysis["ttl_distribution"].get(ttl_bucket, 0) + 1
                
                if value and isinstance(value, bytes):
                    if value.startswith(b'c'):
                        analysis["serialization_stats"]["compressed"] += 1
                    else:
                        analysis["serialization_stats"]["uncompressed"] += 1
            
            # Calculate statistics
            if analysis["key_sizes"]:
                analysis["size_stats"] = {
                    "min": min(analysis["key_sizes"]),
                    "max": max(analysis["key_sizes"]),
                    "avg": sum(analysis["key_sizes"]) / len(analysis["key_sizes"]),
                    "total_bytes": sum(analysis["key_sizes"]),
                }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Cache analysis error: {e}")
            return {"error": str(e)}
    
    # Backup and restore utilities
    
    async def backup_keys(self, pattern: str = "*", output_format: str = "json") -> bytes:
        """
        Backup cache keys to serialized format.
        
        Args:
            pattern: Pattern of keys to backup
            output_format: Output format (json, pickle)
            
        Returns:
            Serialized backup data
        """
        keys = await self.scan_keys(pattern)
        backup_data = {}
        
        # Fetch all values
        for i in range(0, len(keys), 100):
            batch = keys[i:i + 100]
            values = await self.get_many(batch)
            backup_data.update(values)
            
            await asyncio.sleep(0.01)  # Small delay
        
        # Serialize based on format
        if output_format == "pickle":
            return pickle.dumps(backup_data, protocol=pickle.HIGHEST_PROTOCOL)
        else:  # json
            return json.dumps(backup_data, default=str).encode('utf-8')
    
    async def restore_keys(self, backup_data: bytes, input_format: str = "json", 
                          ttl: Optional[int] = None):
        """
        Restore cache keys from backup.
        
        Args:
            backup_data: Serialized backup data
            input_format: Input format (json, pickle)
            ttl: TTL for restored keys
        """
        # Deserialize backup data
        if input_format == "pickle":
            backup_dict = pickle.loads(backup_data)
        else:  # json
            backup_dict = json.loads(backup_data.decode('utf-8'))
        
        # Restore keys in batches
        items = list(backup_dict.items())
        
        for i in range(0, len(items), 100):
            batch = dict(items[i:i + 100])
            await self.set_many(batch, ttl)
            
            await asyncio.sleep(0.01)  # Small delay
        
        logger.info(f"Restored {len(items)} keys from backup")


# Factory function for easy instantiation
def create_redis_cache(
    url: str = None,
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str = None,
    key_prefix: str = "cache",
    **kwargs
) -> RedisCacheBackend:
    """
    Factory function to create Redis cache instance.
    
    Args:
        url: Redis URL
        host: Redis host
        port: Redis port
        db: Redis database
        password: Redis password
        key_prefix: Key prefix
        **kwargs: Additional arguments for RedisCacheBackend
        
    Returns:
        RedisCacheBackend instance
    """
    return RedisCacheBackend(
        url=url,
        host=host,
        port=port,
        db=db,
        password=password,
        key_prefix=key_prefix,
        **kwargs
    )


# Singleton instance for global use (optional)
_global_cache: Optional[RedisCacheBackend] = None

def get_global_cache() -> RedisCacheBackend:
    """
    Get or create global Redis cache instance.
    
    Returns:
        Global RedisCacheBackend instance
    """
    global _global_cache
    
    if _global_cache is None:
        _global_cache = create_redis_cache(
            url=getattr(settings, 'REDIS_URL', None),
            key_prefix=getattr(settings, 'REDIS_KEY_PREFIX', 'app')
        )
    
    return _global_cache


# Example usage and testing
async def example_usage():
    """Example of how to use the Redis cache backend."""
    
    # Create cache instance
    cache = RedisCacheBackend(
        host="localhost",
        port=6379,
        db=0,
        key_prefix="myapp",
        default_ttl=3600,
    )
    
    try:
        # Basic operations
        await cache.set("user:123", {"name": "John", "email": "john@example.com"}, ttl=1800)
        user = await cache.get("user:123")
        print(f"User: {user}")
        
        # Cache with fallback
        async def fetch_data():
            # Simulate database call
            await asyncio.sleep(0.1)
            return {"data": "from database"}
        
        data = await cache.get_with_fallback(
            key="data:important",
            fallback_func=fetch_data,
            ttl=300
        )
        print(f"Data: {data}")
        
        # Distributed lock
        async with cache.distributed_lock("resource:update", timeout=30):
            print("Critical section - updating resource")
            await asyncio.sleep(0.5)
        
        # Rate limiting
        allowed, info = await cache.rate_limit("api:user:123", limit=100, period=3600)
        print(f"Rate limit allowed: {allowed}, info: {info}")
        
        # Health check
        health = await cache.health_check()
        print(f"Redis health: {health['status']}")
        
        # Hash operations
        await cache.hset("user:profile:123", "settings", {"theme": "dark", "notifications": True})
        settings = await cache.hget("user:profile:123", "settings")
        print(f"User settings: {settings}")
        
    finally:
        # Cleanup
        await cache.close()


if __name__ == "__main__":
    # Run example if file is executed directly
    asyncio.run(example_usage())