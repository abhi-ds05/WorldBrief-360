"""
Redis backend for feature flags.
Provides high-performance, distributed feature flag storage with Redis.
"""

import json
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Set, Iterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import pickle
import hashlib

import redis
from redis import Redis, ConnectionPool, RedisError
from redis.cluster import RedisCluster
from redis.exceptions import ConnectionError, TimeoutError

from .base import (
    FeatureBackend,
    BackendConfig,
    BackendError,
    ConnectionError as BackendConnectionError,
    OperationError,
    BackendStatus,
    BackendStats,
    RetryMixin,
    MetricsMixin,
    ConnectionPoolMixin,
)
from ..flags import FeatureFlag, FlagType, create_flag_from_dict

logger = logging.getLogger(__name__)


@dataclass
class RedisConfig(BackendConfig):
    """Configuration for Redis backend."""
    
    # Redis connection
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: Optional[str] = None
    password: Optional[str] = field(default=None, metadata={"secret": True})
    
    # Connection pooling
    max_connections: int = 50
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    socket_keepalive: bool = True
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    
    # Redis Sentinel
    sentinel_servers: Optional[List[Tuple[str, int]]] = None
    sentinel_service_name: Optional[str] = None
    sentinel_socket_timeout: float = 0.1
    
    # Redis Cluster
    cluster_nodes: Optional[List[Dict[str, Any]]] = None
    cluster_skip_full_coverage_check: bool = True
    
    # Redis SSL
    ssl: bool = False
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    ssl_cert_reqs: str = "required"
    ssl_ca_certs: Optional[str] = None
    
    # Redis data structure
    data_structure: str = "hash"  # hash, string, json
    compression: bool = False
    compression_threshold: int = 1024  # bytes
    
    # Pub/Sub for watch support
    enable_pubsub: bool = True
    pubsub_channel_prefix: str = "features:notify:"
    
    # Lua scripting
    enable_lua_scripts: bool = True
    
    # Memory optimization
    enable_memory_optimization: bool = True
    memory_optimization_threshold: int = 1000  # flags
    
    def validate(self):
        """Validate configuration."""
        super().validate()
        
        if not self.host:
            raise ValueError("Redis host is required")
        
        if self.port <= 0 or self.port > 65535:
            raise ValueError(f"Invalid Redis port: {self.port}")
        
        if self.db < 0 or self.db > 15:
            raise ValueError(f"Invalid Redis database: {self.db}")
        
        if self.max_connections <= 0:
            raise ValueError("max_connections must be positive")
        
        if self.data_structure not in ["hash", "string", "json"]:
            raise ValueError(f"Invalid data structure: {self.data_structure}")


class RedisBackend(FeatureBackend, RetryMixin, MetricsMixin, ConnectionPoolMixin):
    """
    Redis backend for feature flags.
    
    Features:
    - High-performance distributed storage
    - Connection pooling
    - Redis Sentinel support for high availability
    - Redis Cluster support for horizontal scaling
    - Pub/Sub for real-time flag updates
    - Lua scripting for atomic operations
    - Memory optimization
    """
    
    supports_persistence = True
    supports_watches = True
    supports_transactions = True
    is_distributed = True
    supports_batch_operations = True
    
    def __init__(self, config: RedisConfig):
        """
        Initialize Redis backend.
        
        Args:
            config: RedisConfig instance
        """
        super().__init__(config)
        self.config = config
        
        # Redis client
        self._redis: Optional[Redis] = None
        self._redis_cluster: Optional[RedisCluster] = None
        self._connection_pool: Optional[ConnectionPool] = None
        
        # Pub/Sub for watches
        self._pubsub: Optional[redis.client.PubSub] = None
        self._pubsub_thread: Optional[threading.Thread] = None
        self._pubsub_running = False
        
        # Local cache for faster reads
        self._local_cache: Dict[str, Tuple[FeatureFlag, float]] = {}
        self._local_cache_lock = threading.RLock()
        self._local_cache_ttl = 5  # seconds
        
        # Lua scripts cache
        self._lua_scripts: Dict[str, str] = {}
        
        # Watch callbacks
        self._watches: Dict[str, List] = {}
        self._watch_lock = threading.RLock()
        
        # Transaction support
        self._pipeline: Optional[redis.client.Pipeline] = None
        self._transaction_stack: List[Dict[str, Any]] = []
        
        logger.debug(f"RedisBackend initialized for {config.host}:{config.port}")
    
    def connect(self) -> bool:
        """
        Connect to Redis.
        
        Returns:
            True if connected successfully
        """
        if self._connected:
            return True
        
        try:
            # Create Redis connection
            self._create_redis_connection()
            
            # Test connection
            self._test_connection()
            
            # Load Lua scripts
            if self.config.enable_lua_scripts:
                self._load_lua_scripts()
            
            # Start Pub/Sub for watches
            if self.config.enable_pubsub:
                self._start_pubsub()
            
            self._connected = True
            self._stats.status = BackendStatus.CONNECTED
            self._stats.connected_since = datetime.utcnow()
            
            logger.info(f"RedisBackend connected to {self.config.host}:{self.config.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._stats.status = BackendStatus.DISCONNECTED
            raise BackendConnectionError(
                f"Failed to connect to Redis: {str(e)}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
    
    def _create_redis_connection(self):
        """Create Redis connection based on configuration."""
        try:
            # Common connection parameters
            connection_params = {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "username": self.config.username,
                "password": self.config.password,
                "socket_timeout": self.config.socket_timeout,
                "socket_connect_timeout": self.config.socket_connect_timeout,
                "socket_keepalive": self.config.socket_keepalive,
                "retry_on_timeout": self.config.retry_on_timeout,
                "health_check_interval": self.config.health_check_interval,
                "max_connections": self.config.max_connections,
                "ssl": self.config.ssl,
                "ssl_keyfile": self.config.ssl_keyfile,
                "ssl_certfile": self.config.ssl_certfile,
                "ssl_cert_reqs": self.config.ssl_cert_reqs,
                "ssl_ca_certs": self.config.ssl_ca_certs,
                "decode_responses": False,  # We handle encoding/decoding
            }
            
            if self.config.sentinel_servers:
                # Redis Sentinel
                from redis.sentinel import Sentinel
                
                sentinel = Sentinel(
                    self.config.sentinel_servers,
                    socket_timeout=self.config.sentinel_socket_timeout,
                    password=self.config.password,
                    sentinel_kwargs={"password": self.config.password},
                )
                
                self._redis = sentinel.master_for(
                    self.config.sentinel_service_name,
                    **connection_params
                )
                
            elif self.config.cluster_nodes:
                # Redis Cluster
                self._redis_cluster = RedisCluster(
                    startup_nodes=self.config.cluster_nodes,
                    skip_full_coverage_check=self.config.cluster_skip_full_coverage_check,
                    **connection_params
                )
                
            else:
                # Single Redis instance with connection pool
                self._connection_pool = ConnectionPool(**connection_params)
                self._redis = Redis(connection_pool=self._connection_pool)
            
            logger.debug("Redis connection created")
            
        except Exception as e:
            logger.error(f"Failed to create Redis connection: {e}")
            raise
    
    def _test_connection(self):
        """Test Redis connection."""
        try:
            # Ping Redis
            if self._redis:
                response = self._redis.ping()
                if not response:
                    raise ConnectionError("Redis ping failed")
            elif self._redis_cluster:
                response = self._redis_cluster.ping()
                if not response:
                    raise ConnectionError("Redis Cluster ping failed")
            
            logger.debug("Redis connection test successful")
            
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            raise
    
    def disconnect(self) -> bool:
        """
        Disconnect from Redis.
        
        Returns:
            True if disconnected successfully
        """
        if not self._connected:
            return True
        
        try:
            # Stop Pub/Sub
            if self.config.enable_pubsub:
                self._stop_pubsub()
            
            # Close Redis connections
            if self._redis:
                self._redis.close()
                self._redis = None
            
            if self._redis_cluster:
                self._redis_cluster.close()
                self._redis_cluster = None
            
            if self._connection_pool:
                self._connection_pool.disconnect()
                self._connection_pool = None
            
            # Clear local cache
            with self._local_cache_lock:
                self._local_cache.clear()
            
            # Clear Lua scripts
            self._lua_scripts.clear()
            
            self._connected = False
            self._stats.status = BackendStatus.DISCONNECTED
            
            logger.info("RedisBackend disconnected")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from Redis: {e}")
            return False
    
    def get(self, key: str) -> Optional[FeatureFlag]:
        """
        Get a feature flag from Redis.
        
        Args:
            key: Feature flag key
        
        Returns:
            FeatureFlag if found, None otherwise
        """
        start_time = time.time()
        success = False
        
        try:
            # Check local cache first
            with self._local_cache_lock:
                if key in self._local_cache:
                    flag, timestamp = self._local_cache[key]
                    if time.time() - timestamp < self._local_cache_ttl:
                        return flag
            
            full_key = self.get_full_key(key)
            
            # Get from Redis
            redis_data = self._execute_redis_command("GET", full_key)
            
            if not redis_data:
                logger.debug(f"Flag not found: {key}")
                return None
            
            # Deserialize data
            flag = self._deserialize_redis_data(redis_data)
            
            # Update local cache
            with self._local_cache_lock:
                self._local_cache[key] = (flag, time.time())
            
            success = True
            return flag
            
        except Exception as e:
            logger.error(f"Error getting flag {key} from Redis: {e}")
            raise OperationError(
                f"Failed to get flag from Redis: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get", success, start_time)
    
    def set(self, key: str, flag: FeatureFlag) -> bool:
        """
        Set a feature flag in Redis.
        
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
            
            # Serialize flag
            redis_data = self._serialize_for_redis(flag)
            
            # Store in Redis
            if self.config.data_structure == "hash":
                # Store as Redis Hash
                result = self._store_as_hash(full_key, flag, redis_data)
            else:
                # Store as Redis String
                result = self._execute_redis_command("SET", full_key, redis_data)
            
            if result:
                # Update local cache
                with self._local_cache_lock:
                    self._local_cache[key] = (flag, time.time())
                
                # Notify watchers via Pub/Sub
                self._notify_watchers(key, flag)
                
                # Optimize memory if needed
                if self.config.enable_memory_optimization:
                    self._optimize_memory()
                
                success = True
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error setting flag {key} in Redis: {e}")
            raise OperationError(
                f"Failed to set flag in Redis: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("set", success, start_time)
    
    def _store_as_hash(self, full_key: str, flag: FeatureFlag, serialized_data: bytes) -> bool:
        """Store flag as Redis Hash for better memory usage."""
        try:
            # Create hash fields
            flag_dict = flag.to_dict()
            
            if self.config.enable_memory_optimization:
                # Store only essential fields in hash
                hash_data = {
                    "name": flag_dict["name"],
                    "enabled": str(flag_dict["enabled"]),
                    "flag_type": flag_dict["flag_type"],
                    "description": flag_dict["description"][:255],  # Limit description length
                }
                
                # Add variants as JSON if present
                if flag_dict.get("variants"):
                    hash_data["variants"] = json.dumps(flag_dict["variants"])
                
                # Add target users if present
                if flag_dict.get("target_users"):
                    hash_data["target_users"] = json.dumps(flag_dict["target_users"])
                
                # Store hash
                result = self._execute_redis_command("HMSET", full_key, hash_data)
                
                # Set TTL if configured
                if self.config.default_ttl:
                    self._execute_redis_command("EXPIRE", full_key, self.config.default_ttl)
                
                return result
            else:
                # Store full serialized data
                return self._execute_redis_command("SET", full_key, serialized_data)
                
        except Exception as e:
            logger.error(f"Error storing hash: {e}")
            # Fall back to string storage
            return self._execute_redis_command("SET", full_key, serialized_data)
    
    def delete(self, key: str) -> bool:
        """
        Delete a feature flag from Redis.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            # Delete from Redis
            result = self._execute_redis_command("DEL", full_key)
            
            if result:
                # Remove from local cache
                with self._local_cache_lock:
                    if key in self._local_cache:
                        del self._local_cache[key]
                
                # Notify watchers of deletion
                self._notify_watchers(key, None)
                
                success = True
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error deleting flag {key} from Redis: {e}")
            raise OperationError(
                f"Failed to delete flag from Redis: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("delete", success, start_time)
    
    def exists(self, key: str) -> bool:
        """
        Check if a feature flag exists in Redis.
        
        Args:
            key: Feature flag key
        
        Returns:
            True if exists
        """
        start_time = time.time()
        success = False
        
        try:
            full_key = self.get_full_key(key)
            
            # Check in Redis
            result = self._execute_redis_command("EXISTS", full_key)
            
            success = True
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error checking flag existence {key}: {e}")
            raise OperationError(
                f"Failed to check flag existence: {key}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("exists", success, start_time)
    
    def get_all(self) -> Dict[str, FeatureFlag]:
        """
        Get all feature flags from Redis.
        
        Returns:
            Dictionary of all feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            pattern = f"{self.config.key_prefix}*"
            
            # Get all keys matching pattern
            keys = self._execute_redis_command("KEYS", pattern)
            
            if not keys:
                return {}
            
            # Get all values
            flags = {}
            
            if self.config.data_structure == "hash":
                # Use pipeline for batch operations
                with self._get_pipeline() as pipe:
                    for key_bytes in keys:
                        key = key_bytes.decode('utf-8')
                        pipe.hgetall(key)
                    
                    results = pipe.execute()
                    
                    for key_bytes, result in zip(keys, results):
                        key = self.remove_key_prefix(key_bytes.decode('utf-8'))
                        
                        if result:
                            try:
                                # Convert hash to flag
                                flag = self._hash_to_flag(result)
                                flags[key] = flag
                                
                                # Update local cache
                                with self._local_cache_lock:
                                    self._local_cache[key] = (flag, time.time())
                            except Exception as e:
                                logger.warning(f"Error converting hash to flag {key}: {e}")
            else:
                # Use MGET for string values
                values = self._execute_redis_command("MGET", *keys)
                
                for key_bytes, value_bytes in zip(keys, values):
                    if value_bytes:
                        key = self.remove_key_prefix(key_bytes.decode('utf-8'))
                        
                        try:
                            flag = self._deserialize_redis_data(value_bytes)
                            flags[key] = flag
                            
                            # Update local cache
                            with self._local_cache_lock:
                                self._local_cache[key] = (flag, time.time())
                        except Exception as e:
                            logger.warning(f"Error deserializing flag {key}: {e}")
            
            success = True
            return flags
            
        except Exception as e:
            logger.error(f"Error getting all flags from Redis: {e}")
            raise OperationError(
                "Failed to get all flags from Redis",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get_all", success, start_time)
    
    def _hash_to_flag(self, hash_data: Dict[bytes, bytes]) -> FeatureFlag:
        """Convert Redis hash to FeatureFlag."""
        try:
            # Convert bytes to strings
            hash_dict = {}
            for k, v in hash_data.items():
                hash_dict[k.decode('utf-8')] = v.decode('utf-8') if v else None
            
            # Reconstruct flag data
            flag_data = {
                "name": hash_dict.get("name", ""),
                "description": hash_dict.get("description", ""),
                "flag_type": hash_dict.get("flag_type", "boolean"),
                "enabled": hash_dict.get("enabled", "false").lower() == "true",
            }
            
            # Parse variants if present
            if "variants" in hash_dict and hash_dict["variants"]:
                try:
                    flag_data["variants"] = json.loads(hash_dict["variants"])
                except:
                    flag_data["variants"] = {}
            
            # Parse target users if present
            if "target_users" in hash_dict and hash_dict["target_users"]:
                try:
                    flag_data["target_users"] = json.loads(hash_dict["target_users"])
                except:
                    flag_data["target_users"] = []
            
            return create_flag_from_dict(flag_data)
            
        except Exception as e:
            logger.error(f"Error converting hash to flag: {e}")
            raise
    
    def clear(self) -> bool:
        """
        Clear all feature flags from Redis.
        
        Returns:
            True if successful
        """
        start_time = time.time()
        success = False
        
        try:
            pattern = f"{self.config.key_prefix}*"
            
            # Get all keys
            keys = self._execute_redis_command("KEYS", pattern)
            
            if keys:
                # Delete all keys
                self._execute_redis_command("DEL", *keys)
            
            # Clear local cache
            with self._local_cache_lock:
                self._local_cache.clear()
            
            # Notify watchers
            for key in list(self._watches.keys()):
                self._notify_watchers(key, None)
            
            success = True
            return True
            
        except Exception as e:
            logger.error(f"Error clearing flags: {e}")
            raise OperationError(
                "Failed to clear flags from Redis",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("clear", success, start_time)
    
    # Batch operations with Redis pipeline
    
    def get_many(self, keys: List[str]) -> Dict[str, Optional[FeatureFlag]]:
        """
        Get multiple feature flags at once using Redis pipeline.
        
        Args:
            keys: List of feature flag keys
        
        Returns:
            Dictionary mapping keys to FeatureFlags (or None if not found)
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._get_pipeline() as pipe:
                for key in keys:
                    full_key = self.get_full_key(key)
                    pipe.get(full_key)
                
                responses = pipe.execute()
            
            for key, response in zip(keys, responses):
                if response:
                    flag = self._deserialize_redis_data(response)
                    result[key] = flag
                    
                    # Update local cache
                    with self._local_cache_lock:
                        self._local_cache[key] = (flag, time.time())
                else:
                    result[key] = None
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Error getting multiple flags: {e}")
            raise OperationError(
                "Failed to get multiple flags",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("get_many", success, start_time)
    
    def set_many(self, items: Dict[str, FeatureFlag]) -> Dict[str, bool]:
        """
        Set multiple feature flags at once using Redis pipeline.
        
        Args:
            items: Dictionary mapping keys to FeatureFlags
        
        Returns:
            Dictionary mapping keys to success status
        """
        start_time = time.time()
        success = False
        
        try:
            result = {}
            
            with self._get_pipeline() as pipe:
                for key, flag in items.items():
                    full_key = self.get_full_key(key)
                    redis_data = self._serialize_for_redis(flag)
                    pipe.set(full_key, redis_data)
                    result[key] = True
                
                pipe.execute()
            
            # Update local cache
            with self._local_cache_lock:
                for key, flag in items.items():
                    self._local_cache[key] = (flag, time.time())
            
            # Notify watchers
            for key, flag in items.items():
                self._notify_watchers(key, flag)
            
            success = True
            return result
            
        except Exception as e:
            logger.error(f"Error setting multiple flags: {e}")
            raise OperationError(
                "Failed to set multiple flags",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("set_many", success, start_time)
    
    # Search implementation
    
    def search(self, pattern: str) -> Dict[str, FeatureFlag]:
        """
        Search for feature flags by pattern.
        
        Args:
            pattern: Redis glob-style pattern
        
        Returns:
            Dictionary of matching feature flags
        """
        start_time = time.time()
        success = False
        
        try:
            full_pattern = f"{self.config.key_prefix}{pattern}"
            
            # Get matching keys
            keys = self._execute_redis_command("KEYS", full_pattern)
            
            if not keys:
                return {}
            
            # Get values
            flags = {}
            values = self._execute_redis_command("MGET", *keys)
            
            for key_bytes, value_bytes in zip(keys, values):
                if value_bytes:
                    key = self.remove_key_prefix(key_bytes.decode('utf-8'))
                    
                    try:
                        flag = self._deserialize_redis_data(value_bytes)
                        flags[key] = flag
                    except Exception as e:
                        logger.warning(f"Error deserializing flag {key}: {e}")
            
            success = True
            return flags
            
        except Exception as e:
            logger.error(f"Error searching flags: {e}")
            raise OperationError(
                f"Failed to search flags with pattern: {pattern}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        
        finally:
            self._record_operation("search", success, start_time)
    
    # Watch implementation with Redis Pub/Sub
    
    def watch(self, key: str, callback) -> bool:
        """
        Watch for changes to a feature flag.
        
        Args:
            key: Feature flag key to watch
            callback: Function to call when flag changes
        
        Returns:
            True if watch established
        """
        if not self.config.enable_pubsub:
            logger.warning("Pub/Sub is disabled, watch will not work")
            return False
        
        with self._watch_lock:
            if key not in self._watches:
                self._watches[key] = []
                
                # Subscribe to channel
                channel = self._get_pubsub_channel(key)
                self._pubsub.subscribe(channel)
                logger.debug(f"Subscribed to channel {channel}")
            
            self._watches[key].append(callback)
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
        with self._watch_lock:
            if key in self._watches:
                # Unsubscribe if no more callbacks
                if len(self._watches[key]) == 1:
                    channel = self._get_pubsub_channel(key)
                    self._pubsub.unsubscribe(channel)
                    logger.debug(f"Unsubscribed from channel {channel}")
                
                del self._watches[key]
                logger.debug(f"Removed watch for {key}")
                return True
            
            return False
    
    def _get_pubsub_channel(self, key: str) -> str:
        """Get Pub/Sub channel name for a flag."""
        return f"{self.config.pubsub_channel_prefix}{key}"
    
    def _notify_watchers(self, key: str, flag: Optional[FeatureFlag]):
        """Notify watchers of flag change via Pub/Sub."""
        if not self.config.enable_pubsub:
            return
        
        try:
            # Publish notification
            channel = self._get_pubsub_channel(key)
            message = {
                "key": key,
                "action": "deleted" if flag is None else "updated",
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            if flag:
                message["flag"] = flag.to_dict()
            
            self._execute_redis_command("PUBLISH", channel, json.dumps(message))
            logger.debug(f"Published notification for {key}")
            
        except Exception as e:
            logger.error(f"Error publishing notification for {key}: {e}")
    
    def _start_pubsub(self):
        """Start Pub/Sub listener thread."""
        if not self.config.enable_pubsub:
            return
        
        if self._pubsub_thread is not None:
            return
        
        try:
            # Create Pub/Sub client
            if self._redis:
                self._pubsub = self._redis.pubsub()
            elif self._redis_cluster:
                # Redis Cluster doesn't support Pub/Sub across cluster
                logger.warning("Pub/Sub not fully supported with Redis Cluster")
                return
            
            # Start listener thread
            self._pubsub_running = True
            
            def pubsub_listener():
                while self._pubsub_running:
                    try:
                        message = self._pubsub.get_message(
                            ignore_subscribe_messages=True,
                            timeout=1.0
                        )
                        
                        if message and message['type'] == 'message':
                            self._handle_pubsub_message(message)
                    
                    except Exception as e:
                        logger.error(f"Error in Pub/Sub listener: {e}")
                        time.sleep(1)
            
            self._pubsub_thread = threading.Thread(
                target=pubsub_listener,
                name="Redis-PubSub",
                daemon=True,
            )
            self._pubsub_thread.start()
            
            logger.debug("Started Pub/Sub listener")
            
        except Exception as e:
            logger.error(f"Failed to start Pub/Sub: {e}")
    
    def _stop_pubsub(self):
        """Stop Pub/Sub listener thread."""
        self._pubsub_running = False
        
        if self._pubsub_thread:
            self._pubsub_thread.join(timeout=5)
            self._pubsub_thread = None
        
        if self._pubsub:
            self._pubsub.close()
            self._pubsub = None
        
        logger.debug("Stopped Pub/Sub listener")
    
    def _handle_pubsub_message(self, message: Dict):
        """Handle Pub/Sub message."""
        try:
            data = json.loads(message['data'])
            key = data.get('key')
            action = data.get('action')
            
            if key and action:
                with self._watch_lock:
                    if key in self._watches:
                        for callback in self._watches[key]:
                            try:
                                if action == "deleted":
                                    callback(key, None)
                                else:
                                    flag_data = data.get('flag')
                                    if flag_data:
                                        flag = create_flag_from_dict(flag_data)
                                        callback(key, flag)
                            except Exception as e:
                                logger.error(f"Error in watch callback for {key}: {e}")
            
        except Exception as e:
            logger.error(f"Error handling Pub/Sub message: {e}")
    
    # Transaction support
    
    def begin_transaction(self):
        """
        Begin a Redis transaction.
        
        Returns:
            Transaction pipeline
        """
        if self._pipeline is not None:
            raise OperationError("Transaction already in progress")
        
        if self._redis:
            self._pipeline = self._redis.pipeline(transaction=True)
        elif self._redis_cluster:
            # Redis Cluster doesn't support transactions across keys
            self._pipeline = self._redis_cluster.pipeline()
        
        logger.debug("Started Redis transaction")
    
    def commit_transaction(self) -> bool:
        """
        Commit current transaction.
        
        Returns:
            True if successful
        """
        if self._pipeline is None:
            return False
        
        try:
            results = self._pipeline.execute()
            self._pipeline = None
            
            logger.debug(f"Committed Redis transaction with {len(results)} operations")
            return True
            
        except Exception as e:
            logger.error(f"Error committing transaction: {e}")
            self._pipeline = None
            return False
    
    def rollback_transaction(self) -> bool:
        """
        Rollback current transaction.
        
        Returns:
            True if successful
        """
        if self._pipeline is None:
            return False
        
        self._pipeline.reset()
        self._pipeline = None
        
        logger.debug("Rolled back Redis transaction")
        return True
    
    # Lua script support
    
    def _load_lua_scripts(self):
        """Load Lua scripts for atomic operations."""
        try:
            # Script to get flag with cache update
            get_flag_script = """
            local key = KEYS[1]
            local cache_key = KEYS[2]
            local ttl = ARGV[1]
            
            -- Try to get from cache
            local cached = redis.call('GET', cache_key)
            if cached then
                return cached
            end
            
            -- Get from main storage
            local data = redis.call('GET', key)
            if data then
                -- Update cache
                redis.call('SETEX', cache_key, ttl, data)
            end
            
            return data
            """
            
            self._lua_scripts['get_flag'] = self._execute_redis_command(
                'SCRIPT', 'LOAD', get_flag_script
            )
            
            # Script to update flag with cache invalidation
            update_flag_script = """
            local key = KEYS[1]
            local cache_key = KEYS[2]
            local pubsub_channel = KEYS[3]
            local data = ARGV[1]
            local cache_ttl = ARGV[2]
            
            -- Update main storage
            redis.call('SET', key, data)
            
            -- Update cache
            redis.call('SETEX', cache_key, cache_ttl, data)
            
            -- Publish notification
            redis.call('PUBLISH', pubsub_channel, 'updated')
            
            return 1
            """
            
            self._lua_scripts['update_flag'] = self._execute_redis_command(
                'SCRIPT', 'LOAD', update_flag_script
            )
            
            logger.debug("Loaded Lua scripts")
            
        except Exception as e:
            logger.warning(f"Failed to load Lua scripts: {e}")
    
    def _get_with_cache(self, key: str) -> Optional[bytes]:
        """Get flag using Lua script with cache."""
        if not self._lua_scripts.get('get_flag'):
            return None
        
        try:
            full_key = self.get_full_key(key)
            cache_key = f"{full_key}:cache"
            
            result = self._execute_redis_command(
                'EVALSHA',
                self._lua_scripts['get_flag'],
                2,  # number of keys
                full_key,
                cache_key,
                self._local_cache_ttl,
            )
            
            return result
            
        except Exception as e:
            logger.debug(f"Lua script failed, falling back to normal GET: {e}")
            return None
    
    # Utility methods
    
    def _execute_redis_command(self, command: str, *args, **kwargs):
        """Execute Redis command with retry logic."""
        try:
            if self._redis:
                return getattr(self._redis, command.lower())(*args, **kwargs)
            elif self._redis_cluster:
                return getattr(self._redis_cluster, command.lower())(*args, **kwargs)
            else:
                raise OperationError("Redis client not connected")
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis connection error: {e}")
            raise BackendConnectionError(
                f"Redis connection error: {str(e)}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
        except RedisError as e:
            logger.error(f"Redis error: {e}")
            raise OperationError(
                f"Redis error: {str(e)}",
                backend_type=self.__class__.__name__,
                cause=e,
            )
    
    def _get_pipeline(self):
        """Get Redis pipeline for batch operations."""
        if self._redis:
            return self._redis.pipeline()
        elif self._redis_cluster:
            return self._redis_cluster.pipeline()
        else:
            raise OperationError("Redis client not connected")
    
    def _serialize_for_redis(self, flag: FeatureFlag) -> bytes:
        """Serialize flag for Redis storage."""
        flag_dict = flag.to_dict()
        
        if self.config.compression and len(str(flag_dict)) > self.config.compression_threshold:
            # Use compressed JSON
            import zlib
            json_data = json.dumps(flag_dict, default=str).encode('utf-8')
            compressed = zlib.compress(json_data)
            return b'compressed:' + compressed
        
        elif self.config.serializer == "json":
            return json.dumps(flag_dict, default=str).encode('utf-8')
        
        elif self.config.serializer == "pickle":
            return pickle.dumps(flag_dict)
        
        elif self.config.serializer == "msgpack":
            import msgpack
            return msgpack.packb(flag_dict, default=str)
        
        else:
            return json.dumps(flag_dict, default=str).encode('utf-8')
    
    def _deserialize_redis_data(self, data: bytes) -> FeatureFlag:
        """Deserialize data from Redis."""
        if data.startswith(b'compressed:'):
            # Decompress
            import zlib
            compressed = data[11:]  # Remove 'compressed:' prefix
            json_data = zlib.decompress(compressed)
            flag_dict = json.loads(json_data.decode('utf-8'))
        
        elif self.config.serializer == "json":
            flag_dict = json.loads(data.decode('utf-8'))
        
        elif self.config.serializer == "pickle":
            flag_dict = pickle.loads(data)
        
        elif self.config.serializer == "msgpack":
            import msgpack
            flag_dict = msgpack.unpackb(data)
        
        else:
            flag_dict = json.loads(data.decode('utf-8'))
        
        return create_flag_from_dict(flag_dict)
    
    def _optimize_memory(self):
        """Optimize Redis memory usage."""
        try:
            pattern = f"{self.config.key_prefix}*"
            keys = self._execute_redis_command("KEYS", pattern)
            
            if len(keys) > self.config.memory_optimization_threshold:
                logger.debug(f"Optimizing memory for {len(keys)} keys")
                
                for key_bytes in keys:
                    try:
                        # Convert string to hash if beneficial
                        value = self._execute_redis_command("GET", key_bytes)
                        if value and len(value) > 1024:  # Large values
                            flag = self._deserialize_redis_data(value)
                            self._store_as_hash(key_bytes.decode('utf-8'), flag, value)
                    except Exception as e:
                        logger.debug(f"Error optimizing key {key_bytes}: {e}")
        
        except Exception as e:
            logger.debug(f"Memory optimization failed: {e}")
    
    # Health and monitoring
    
    def health_check(self) -> bool:
        """Perform health check on Redis backend."""
        try:
            # Ping Redis
            if self._redis:
                return self._redis.ping()
            elif self._redis_cluster:
                return self._redis_cluster.ping()
            return False
            
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    def get_stats(self) -> BackendStats:
        """Get backend statistics."""
        stats = super().get_stats()
        
        try:
            # Get Redis info
            if self._redis:
                info = self._redis.info()
            elif self._redis_cluster:
                # Get info from first node
                nodes = self._redis_cluster.get_nodes()
                if nodes:
                    info = nodes[0].info()
                else:
                    info = {}
            else:
                info = {}
            
            stats.extra_stats = {
                "local_cache_size": len(self._local_cache),
                "watches_count": len(self._watches),
                "pubsub_active": self._pubsub_running,
                "redis_info": {
                    "used_memory": info.get('used_memory', 0),
                    "connected_clients": info.get('connected_clients', 0),
                    "total_connections_received": info.get('total_connections_received', 0),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0),
                },
            }
            
        except Exception as e:
            logger.debug(f"Could not get Redis stats: {e}")
            stats.extra_stats = {
                "local_cache_size": len(self._local_cache),
                "watches_count": len(self._watches),
                "pubsub_active": self._pubsub_running,
            }
        
        return stats
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics."""
        diagnostics = super().get_diagnostics()
        
        diagnostics.update({
            "redis_config": {
                "host": self.config.host,
                "port": self.config.port,
                "db": self.config.db,
                "data_structure": self.config.data_structure,
                "enable_pubsub": self.config.enable_pubsub,
                "sentinel": bool(self.config.sentinel_servers),
                "cluster": bool(self.config.cluster_nodes),
            },
            "cache_info": {
                "local_cache_size": len(self._local_cache),
                "local_cache_ttl": self._local_cache_ttl,
            },
            "watch_info": {
                "watched_flags": list(self._watches.keys()),
                "total_watches": sum(len(callbacks) for callbacks in self._watches.values()),
            },
            "pubsub_info": {
                "active": self._pubsub_running,
                "thread_alive": self._pubsub_thread.is_alive() if self._pubsub_thread else False,
            },
            "lua_scripts": {
                "loaded": list(self._lua_scripts.keys()),
            },
        })
        
        return diagnostics
    
    # Special methods
    
    def __len__(self) -> int:
        """Get number of feature flags."""
        pattern = f"{self.config.key_prefix}*"
        keys = self._execute_redis_command("KEYS", pattern)
        return len(keys)
    
    def __contains__(self, key: str) -> bool:
        """Check if feature flag exists."""
        return self.exists(key)
    
    def __repr__(self) -> str:
        """String representation."""
        status = "connected" if self._connected else "disconnected"
        host_port = f"{self.config.host}:{self.config.port}"
        return f"RedisBackend({status}, {host_port}, db={self.config.db})"
    
    def __del__(self):
        """Destructor - ensure disconnection."""
        try:
            self.disconnect()
        except:
            pass


# Redis Cluster specific backend

class RedisClusterBackend(RedisBackend):
    """
    Redis Cluster backend for feature flags.
    
    Extends RedisBackend with Redis Cluster specific optimizations.
    """
    
    def __init__(self, config: RedisConfig):
        """Initialize Redis Cluster backend."""
        # Ensure cluster nodes are specified
        if not config.cluster_nodes:
            raise ValueError("cluster_nodes must be specified for RedisClusterBackend")
        
        super().__init__(config)
    
    def _create_redis_connection(self):
        """Create Redis Cluster connection."""
        try:
            self._redis_cluster = RedisCluster(
                startup_nodes=self.config.cluster_nodes,
                skip_full_coverage_check=self.config.cluster_skip_full_coverage_check,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                socket_keepalive=self.config.socket_keepalive,
                retry_on_timeout=self.config.retry_on_timeout,
                max_connections=self.config.max_connections,
                password=self.config.password,
                ssl=self.config.ssl,
                ssl_keyfile=self.config.ssl_keyfile,
                ssl_certfile=self.config.ssl_certfile,
                ssl_cert_reqs=self.config.ssl_cert_reqs,
                ssl_ca_certs=self.config.ssl_ca_certs,
                decode_responses=False,
            )
            
            logger.debug("Redis Cluster connection created")
            
        except Exception as e:
            logger.error(f"Failed to create Redis Cluster connection: {e}")
            raise
    
    def _notify_watchers(self, key: str, flag: Optional[FeatureFlag]):
        """
        Notify watchers in Redis Cluster.
        
        Note: Pub/Sub doesn't work across cluster nodes by default.
        This is a simplified implementation.
        """
        # In Redis Cluster, Pub/Sub is limited to single node
        # We'll use a simpler approach
        with self._watch_lock:
            if key in self._watches:
                for callback in self._watches[key]:
                    try:
                        callback(key, flag)
                    except Exception as e:
                        logger.error(f"Error in watch callback for {key}: {e}")