"""
Feature flag backends for WorldBrief360.
Provides different storage backends for feature flags with a unified interface.
"""

from .base import FeatureBackend, BackendError, BackendConfig
from .in_memory import InMemoryBackend, InMemoryConfig
from .redis import RedisBackend, RedisConfig, RedisClusterBackend, RedisClusterConfig
from .unleash import UnleashBackend, UnleashConfig
from .launchdarkly import LaunchDarklyBackend, LaunchDarklyConfig # type : ignore
from .postgres import PostgreSQLBackend, PostgreSQLConfig # pyright: ignore[reportMissingImports] # type : ignore
from .mongodb import MongoDBBackend, MongoDBConfig # pyright: ignore[reportMissingImports] # type : ignore
from .file import FileBackend, FileConfig # pyright: ignore[reportMissingImports] # type : ignore
from .http import HTTPBackend, HTTPConfig # pyright: ignore[reportMissingImports] # type : ignore
from .s3 import S3Backend, S3Config # pyright: ignore[reportMissingImports] # type : ignore
from .consul import ConsulBackend, ConsulConfig # type: ignore # type : ignore
from .etcd import EtcdBackend, EtcdConfig # pyright: ignore[reportMissingImports] # type : ignore

# Backend type registry
_BACKEND_REGISTRY = {
    "in_memory": InMemoryBackend,
    "redis": RedisBackend,
    "redis_cluster": RedisClusterBackend,
    "unleash": UnleashBackend,
    "launchdarkly": LaunchDarklyBackend,
    "postgres": PostgreSQLBackend,
    "mongodb": MongoDBBackend,
    "file": FileBackend,
    "http": HTTPBackend,
    "s3": S3Backend,
    "consul": ConsulBackend,
    "etcd": EtcdBackend,
}

# Backend config registry
_CONFIG_REGISTRY = {
    "in_memory": InMemoryConfig,
    "redis": RedisConfig,
    "redis_cluster": RedisClusterConfig,
    "unleash": UnleashConfig,
    "launchdarkly": LaunchDarklyConfig,
    "postgres": PostgreSQLConfig,
    "mongodb": MongoDBConfig,
    "file": FileConfig,
    "http": HTTPConfig,
    "s3": S3Config,
    "consul": ConsulConfig,
    "etcd": EtcdConfig,
}


def get_backend_class(backend_type: str):
    """
    Get backend class by type.
    
    Args:
        backend_type: Type of backend (in_memory, redis, unleash, etc.)
    
    Returns:
        Backend class
    
    Raises:
        ValueError: If backend type is not supported
    """
    if backend_type not in _BACKEND_REGISTRY:
        supported = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unsupported backend type: {backend_type}. "
            f"Supported types: {supported}"
        )
    return _BACKEND_REGISTRY[backend_type]


def get_config_class(backend_type: str):
    """
    Get config class for backend type.
    
    Args:
        backend_type: Type of backend
    
    Returns:
        Config class
    
    Raises:
        ValueError: If backend type is not supported
    """
    if backend_type not in _CONFIG_REGISTRY:
        supported = list(_CONFIG_REGISTRY.keys())
        raise ValueError(
            f"Unsupported backend type: {backend_type}. "
            f"Supported types: {supported}"
        )
    return _CONFIG_REGISTRY[backend_type]


def create_backend(
    backend_type: str = "in_memory",
    **config_kwargs,
) -> FeatureBackend:
    """
    Create a backend instance with configuration.
    
    Args:
        backend_type: Type of backend to create
        **config_kwargs: Configuration parameters for the backend
    
    Returns:
        Configured backend instance
    
    Example:
        >>> # Create Redis backend
        >>> backend = create_backend(
        ...     "redis",
        ...     host="localhost",
        ...     port=6379,
        ...     db=0,
        ...     key_prefix="features:"
        ... )
        >>>
        >>> # Create Unleash backend
        >>> backend = create_backend(
        ...     "unleash",
        ...     url="http://unleash:4242/api",
        ...     app_name="worldbrief360",
        ...     environment="production"
        ... )
    """
    backend_class = get_backend_class(backend_type)
    config_class = get_config_class(backend_type)
    
    # Create config instance
    config = config_class(**config_kwargs)
    
    # Create backend with config
    return backend_class(config)


def register_backend(
    backend_type: str,
    backend_class,
    config_class=None,
):
    """
    Register a custom backend type.
    
    Args:
        backend_type: Unique identifier for backend type
        backend_class: Backend implementation class
        config_class: Configuration class (optional)
    
    Example:
        >>> class CustomBackend(FeatureBackend):
        ...     pass
        >>>
        >>> class CustomConfig(BackendConfig):
        ...     pass
        >>>
        >>> register_backend("custom", CustomBackend, CustomConfig)
    """
    if not issubclass(backend_class, FeatureBackend):
        raise TypeError(
            f"backend_class must inherit from FeatureBackend, "
            f"got {backend_class.__name__}"
        )
    
    _BACKEND_REGISTRY[backend_type] = backend_class
    
    if config_class:
        if not issubclass(config_class, BackendConfig):
            raise TypeError(
                f"config_class must inherit from BackendConfig, "
                f"got {config_class.__name__}"
            )
        _CONFIG_REGISTRY[backend_type] = config_class
    else:
        # Use base config if none provided
        _CONFIG_REGISTRY[backend_type] = BackendConfig


def list_backends() -> list:
    """
    List all registered backend types.
    
    Returns:
        List of backend type names
    """
    return list(_BACKEND_REGISTRY.keys())


def get_backend_info(backend_type: str) -> dict:
    """
    Get information about a backend type.
    
    Args:
        backend_type: Type of backend
    
    Returns:
        Dictionary with backend information
    
    Raises:
        ValueError: If backend type is not registered
    """
    if backend_type not in _BACKEND_REGISTRY:
        raise ValueError(f"Backend type not registered: {backend_type}")
    
    backend_class = _BACKEND_REGISTRY[backend_type]
    config_class = _CONFIG_REGISTRY.get(backend_type, BackendConfig)
    
    return {
        "type": backend_type,
        "backend_class": backend_class.__name__,
        "config_class": config_class.__name__,
        "description": backend_class.__doc__.split('\n')[0] if backend_class.__doc__ else "",
        "supports_persistence": backend_class.supports_persistence,
        "supports_watches": backend_class.supports_watches,
        "is_distributed": backend_class.is_distributed,
    }


def create_backend_from_config(config: dict) -> FeatureBackend:
    """
    Create backend from configuration dictionary.
    
    Args:
        config: Configuration dictionary with 'type' and backend-specific config
    
    Returns:
        Configured backend instance
    
    Example:
        >>> config = {
        ...     "type": "redis",
        ...     "host": "localhost",
        ...     "port": 6379,
        ...     "key_prefix": "features:"
        ... }
        >>> backend = create_backend_from_config(config)
    """
    if "type" not in config:
        raise ValueError("Configuration must include 'type' field")
    
    backend_type = config["type"]
    backend_kwargs = {k: v for k, v in config.items() if k != "type"}
    
    return create_backend(backend_type, **backend_kwargs)


def create_multi_backend(configs: list) -> 'MultiBackend': # type: ignore
    """
    Create a multi-backend that tries multiple backends in order.
    
    Args:
        configs: List of backend configurations to try
    
    Returns:
        MultiBackend instance
    
    Example:
        >>> configs = [
        ...     {"type": "redis", "host": "redis-primary"},
        ...     {"type": "redis", "host": "redis-secondary"},
        ...     {"type": "in_memory"}
        ... ]
        >>> backend = create_multi_backend(configs)
    """
    from .multi import MultiBackend, MultiBackendConfig # type: ignore
    
    backends = []
    for config in configs:
        backend = create_backend_from_config(config)
        backends.append(backend)
    
    return MultiBackend(backends)


def get_recommended_backend(environment: str, scale: str = "small") -> str:
    """
    Get recommended backend for given environment and scale.
    
    Args:
        environment: Deployment environment (development, staging, production)
        scale: Application scale (small, medium, large)
    
    Returns:
        Recommended backend type
    
    Example:
        >>> # For development
        >>> backend_type = get_recommended_backend("development")
        >>> # Returns: "in_memory"
        >>>
        >>> # For production at medium scale
        >>> backend_type = get_recommended_backend("production", "medium")
        >>> # Returns: "redis"
    """
    recommendations = {
        "development": {
            "small": "in_memory",
            "medium": "in_memory",
            "large": "redis",
        },
        "staging": {
            "small": "redis",
            "medium": "redis",
            "large": "redis_cluster",
        },
        "production": {
            "small": "redis",
            "medium": "redis_cluster",
            "large": "unleash",
        },
    }
    
    if environment not in recommendations:
        environment = "production"
    
    if scale not in recommendations[environment]:
        scale = "medium"
    
    return recommendations[environment][scale]


def create_environment_backend(environment: str) -> FeatureBackend:
    """
    Create backend appropriate for the given environment.
    
    Args:
        environment: Deployment environment
    
    Returns:
        Configured backend instance
    """
    backend_type = get_recommended_backend(environment)
    
    # Default configurations for each environment
    default_configs = {
        "development": {
            "in_memory": {},
            "redis": {
                "host": "localhost",
                "port": 6379,
                "key_prefix": "features:dev:",
            },
        },
        "staging": {
            "redis": {
                "host": "redis-staging",
                "port": 6379,
                "key_prefix": "features:staging:",
                "socket_timeout": 5,
                "socket_connect_timeout": 5,
            },
        },
        "production": {
            "redis": {
                "host": "redis-primary",
                "port": 6379,
                "key_prefix": "features:prod:",
                "socket_timeout": 2,
                "socket_connect_timeout": 2,
                "retry_on_timeout": True,
                "max_connections": 50,
            },
            "redis_cluster": {
                "startup_nodes": [
                    {"host": "redis-cluster-1", "port": 6379},
                    {"host": "redis-cluster-2", "port": 6379},
                    {"host": "redis-cluster-3", "port": 6379},
                ],
                "key_prefix": "features:prod:",
            },
            "unleash": {
                "url": "https://unleash.example.com/api",
                "app_name": "worldbrief360",
                "environment": "production",
                "instance_id": "worldbrief360-api-1",
            },
        },
    }
    
    if environment not in default_configs:
        environment = "production"
    
    config = default_configs[environment].get(backend_type, {})
    return create_backend(backend_type, **config)


# Export all backend classes
__all__ = [
    # Base classes
    "FeatureBackend",
    "BackendError",
    "BackendConfig",
    
    # Backend implementations
    "InMemoryBackend",
    "InMemoryConfig",
    "RedisBackend",
    "RedisConfig",
    "RedisClusterBackend",
    "RedisClusterConfig",
    "UnleashBackend",
    "UnleashConfig",
    "LaunchDarklyBackend",
    "LaunchDarklyConfig",
    "PostgreSQLBackend",
    "PostgreSQLConfig",
    "MongoDBBackend",
    "MongoDBConfig",
    "FileBackend",
    "FileConfig",
    "HTTPBackend",
    "HTTPConfig",
    "S3Backend",
    "S3Config",
    "ConsulBackend",
    "ConsulConfig",
    "EtcdBackend",
    "EtcdConfig",
    
    # Factory functions
    "get_backend_class",
    "get_config_class",
    "create_backend",
    "register_backend",
    "list_backends",
    "get_backend_info",
    "create_backend_from_config",
    "create_multi_backend",
    "get_recommended_backend",
    "create_environment_backend",
]