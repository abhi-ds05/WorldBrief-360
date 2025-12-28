"""
Application settings and runtime configuration.
"""

import os
import sys
from typing import Any, Dict, Optional, Set
from pathlib import Path
from functools import lru_cache

from .config import Config, get_config, Environment
from .metadata import get_app_metadata


class Settings:
    """
    Application settings and runtime state.
    
    This class manages application state, feature flags, and runtime configuration.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize settings.
        
        Args:
            config: Application configuration
        """
        self.config = config or get_config()
        self.metadata = get_app_metadata()
        
        # Runtime state
        self._started = False
        self._stopped = False
        self._ready = False
        
        # Feature flags
        self._enabled_features: Set[str] = set()
        self._disabled_features: Set[str] = set()
        
        # Runtime cache
        self._cache: Dict[str, Any] = {}
        
        # Initialize
        self._initialize_features()
    
    def _initialize_features(self) -> None:
        """Initialize feature flags from configuration."""
        if self.config.enable_cache:
            self.enable_feature("cache")
        if self.config.enable_rate_limit:
            self.enable_feature("rate_limit")
        if self.config.enable_metrics:
            self.enable_feature("metrics")
        
        # Environment-specific features
        if self.config.is_development():
            self.enable_feature("debug")
            self.enable_feature("detailed_errors")
            self.enable_feature("reload")
        elif self.config.is_production():
            self.enable_feature("compression")
            self.enable_feature("caching")
            self.enable_feature("security_headers")
    
    def start(self) -> None:
        """Start application services."""
        if self._started:
            return
        
        self._started = True
        self._ready = False
        
        # Initialize services based on features
        if self.is_feature_enabled("cache"):
            self._init_cache()
        
        if self.is_feature_enabled("metrics"):
            self._init_metrics()
        
        self._ready = True
    
    def stop(self) -> None:
        """Stop application services."""
        if self._stopped:
            return
        
        # Clean up services
        if self.is_feature_enabled("cache"):
            self._cleanup_cache()
        
        if self.is_feature_enabled("metrics"):
            self._cleanup_metrics()
        
        self._ready = False
        self._stopped = True
    
    def _init_cache(self) -> None:
        """Initialize cache services."""
        # Cache initialization would go here
        # This is a placeholder for actual cache initialization
        pass
    
    def _cleanup_cache(self) -> None:
        """Cleanup cache services."""
        # Cache cleanup would go here
        pass
    
    def _init_metrics(self) -> None:
        """Initialize metrics collection."""
        # Metrics initialization would go here
        pass
    
    def _cleanup_metrics(self) -> None:
        """Cleanup metrics collection."""
        # Metrics cleanup would go here
        pass
    
    def is_ready(self) -> bool:
        """
        Check if application is ready.
        
        Returns:
            True if application is ready
        """
        return self._ready
    
    def is_started(self) -> bool:
        """
        Check if application is started.
        
        Returns:
            True if application is started
        """
        return self._started
    
    def is_stopped(self) -> bool:
        """
        Check if application is stopped.
        
        Returns:
            True if application is stopped
        """
        return self._stopped
    
    def enable_feature(self, feature: str) -> None:
        """
        Enable a feature.
        
        Args:
            feature: Feature name
        """
        self._enabled_features.add(feature)
        if feature in self._disabled_features:
            self._disabled_features.remove(feature)
    
    def disable_feature(self, feature: str) -> None:
        """
        Disable a feature.
        
        Args:
            feature: Feature name
        """
        self._disabled_features.add(feature)
        if feature in self._enabled_features:
            self._enabled_features.remove(feature)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name
            
        Returns:
            True if feature is enabled
        """
        if feature in self._disabled_features:
            return False
        if feature in self._enabled_features:
            return True
        
        # Check default based on environment
        if self.config.is_development():
            return feature in ["debug", "detailed_errors", "reload"]
        elif self.config.is_production():
            return feature in ["compression", "caching", "security_headers"]
        
        return False
    
    def get_features(self) -> Dict[str, bool]:
        """
        Get all features and their status.
        
        Returns:
            Dictionary of feature statuses
        """
        all_features = self._enabled_features.union(self._disabled_features)
        features = {}
        
        for feature in all_features:
            features[feature] = self.is_feature_enabled(feature)
        
        # Add default features
        default_features = ["debug", "detailed_errors", "reload", "compression", 
                          "caching", "security_headers", "cache", "rate_limit", "metrics"]
        
        for feature in default_features:
            if feature not in features:
                features[feature] = self.is_feature_enabled(feature)
        
        return features
    
    def set_cache_value(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in runtime cache.
        
        Args:
            key: Cache key
            value: Cache value
            ttl: Time to live in seconds (not implemented in memory cache)
        """
        self._cache[key] = {
            "value": value,
            "timestamp": self.metadata.start_time,
            "ttl": ttl,
        }
    
    def get_cache_value(self, key: str, default: Any = None) -> Any:
        """
        Get a value from runtime cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if key not in self._cache:
            return default
        
        cache_entry = self._cache[key]
        
        # Check TTL (simplified - in production use proper cache with TTL)
        if cache_entry["ttl"]:
            # This is a simplified TTL check
            pass
        
        return cache_entry["value"]
    
    def clear_cache(self) -> None:
        """Clear runtime cache."""
        self._cache.clear()
    
    def invalidate_cache(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entry or all cache.
        
        Args:
            key: Optional specific key to invalidate
        """
        if key:
            self._cache.pop(key, None)
        else:
            self.clear_cache()
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """
        Get runtime information.
        
        Returns:
            Dictionary with runtime info
        """
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        return {
            "status": {
                "started": self._started,
                "stopped": self._stopped,
                "ready": self._ready,
            },
            "process": {
                "pid": os.getpid(),
                "name": process.name(),
                "status": process.status(),
                "create_time": process.create_time(),
            },
            "memory": process.memory_info()._asdict(),
            "cpu_percent": process.cpu_percent(),
            "threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections()),
            "features": self.get_features(),
            "cache_size": len(self._cache),
        }
    
    def reload_config(self) -> None:
        """Reload configuration."""
        from .config import load_config
        self.config = load_config()
        self._initialize_features()
    
    def get_path(self, path_type: str) -> Path:
        """
        Get application path.
        
        Args:
            path_type: Type of path ("root", "config", "logs", "cache", "data")
            
        Returns:
            Path object
            
        Raises:
            ValueError: If path type is invalid
        """
        # Get current working directory
        cwd = Path.cwd()
        
        if path_type == "root":
            return cwd
        elif path_type == "config":
            return cwd / "config"
        elif path_type == "logs":
            log_path = Path(self.config.logging.file_path or "logs")
            if not log_path.is_absolute():
                log_path = cwd / log_path
            return log_path
        elif path_type == "cache":
            cache_path = Path(self.config.cache.file_cache_path)
            if not cache_path.is_absolute():
                cache_path = cwd / cache_path
            return cache_path
        elif path_type == "data":
            return cwd / "data"
        elif path_type == "static":
            return cwd / "static"
        elif path_type == "templates":
            return cwd / "templates"
        else:
            raise ValueError(f"Unknown path type: {path_type}")
    
    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.get_path("logs"),
            self.get_path("cache"),
            self.get_path("data"),
            self.get_path("static"),
            self.get_path("templates"),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


@lru_cache()
def get_settings() -> Settings:
    """
    Get or create settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Context manager for application lifecycle
class ApplicationContext:
    """Context manager for application lifecycle."""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
    
    def __enter__(self):
        self.settings.start()
        return self.settings
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.settings.stop()


# Convenience functions
def is_development() -> bool:
    """Check if running in development environment."""
    return get_settings().config.is_development()


def is_production() -> bool:
    """Check if running in production environment."""
    return get_settings().config.is_production()


def is_testing() -> bool:
    """Check if running in testing environment."""
    return get_settings().config.is_testing()


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return get_settings().is_feature_enabled(feature)


def get_runtime_info() -> Dict[str, Any]:
    """Get runtime information."""
    return get_settings().get_runtime_info()


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    get_settings().ensure_directories()