"""
Model-based cache invalidation system.
Automatically invalidates cache when models are updated.
"""

import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Set, Callable, Type, Union
from datetime import datetime
from dataclasses import dataclass, field

from app.cache.backends import get_cache_backend
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModelCacheKey:
    """Represents a cache key for model data."""
    model_name: str
    model_id: Union[int, str]
    action: str = "get"
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_key(self) -> str:
        """Convert to cache key string."""
        parts = [self.model_name, str(self.model_id), self.action]
        
        if self.extra:
            # Sort extra dict for consistent key generation
            sorted_extra = sorted(self.extra.items())
            extra_hash = hashlib.md5(
                json.dumps(sorted_extra, sort_keys=True).encode()
            ).hexdigest()[:8]
            parts.append(extra_hash)
        
        return ":".join(parts)
    
    @classmethod
    def from_key(cls, key: str) -> "ModelCacheKey":
        """Parse cache key string."""
        parts = key.split(":")
        
        if len(parts) < 3:
            raise ValueError(f"Invalid cache key format: {key}")
        
        model_name = parts[0]
        model_id = parts[1]
        action = parts[2]
        
        # Convert model_id to int if possible
        try:
            model_id = int(model_id)
        except ValueError:
            pass
        
        return cls(
            model_name=model_name,
            model_id=model_id,
            action=action,
            extra={}  # Extra info not stored in key
        )


class ModelCacheInvalidator:
    """
    Manages cache invalidation for model operations.
    
    Features:
    - Automatic invalidation on model create/update/delete
    - Relationship-aware invalidation
    - Batch invalidation support
    - Pattern-based invalidation
    - Event-driven invalidation
    """
    
    def __init__(self, cache_backend=None):
        """
        Initialize model cache invalidator.
        
        Args:
            cache_backend: Cache backend instance (uses default if None)
        """
        self.cache = cache_backend or get_cache_backend()
        self._model_registry: Dict[str, Dict] = {}
        self._relationships: Dict[str, List[str]] = {}
        self._invalidation_callbacks: Dict[str, List[Callable]] = {}
        
    def register_model(
        self,
        model_name: str,
        fields: List[str] = None,
        ttl: int = 3600,
        version: int = 1,
        dependencies: List[str] = None,
    ):
        """
        Register a model for cache invalidation.
        
        Args:
            model_name: Name of the model (e.g., "User", "Article")
            fields: List of fields that trigger invalidation when changed
            ttl: Default TTL for model cache entries
            version: Cache version for schema changes
            dependencies: Other models that depend on this model
        """
        self._model_registry[model_name] = {
            "fields": fields or [],
            "ttl": ttl,
            "version": version,
            "dependencies": dependencies or [],
            "created_at": datetime.utcnow(),
        }
        
        logger.info(f"Registered model for cache invalidation: {model_name}")
    
    def register_relationship(
        self,
        parent_model: str,
        child_model: str,
        foreign_key: str,
        cascade_invalidate: bool = True,
    ):
        """
        Register a relationship between models.
        
        Args:
            parent_model: Parent model name
            child_model: Child model name
            foreign_key: Foreign key field in child model
            cascade_invalidate: Invalidate child cache when parent changes
        """
        if parent_model not in self._relationships:
            self._relationships[parent_model] = []
        
        self._relationships[parent_model].append({
            "child_model": child_model,
            "foreign_key": foreign_key,
            "cascade_invalidate": cascade_invalidate,
        })
        
        logger.info(f"Registered relationship: {parent_model} -> {child_model}")
    
    def generate_cache_key(
        self,
        model_name: str,
        model_id: Union[int, str],
        action: str = "get",
        **kwargs,
    ) -> str:
        """
        Generate cache key for model operation.
        
        Args:
            model_name: Name of the model
            model_id: Model ID
            action: Operation action (get, list, search, etc.)
            **kwargs: Additional parameters for key generation
            
        Returns:
            Cache key string
        """
        # Get model version
        version = self._model_registry.get(model_name, {}).get("version", 1)
        
        # Create key parts
        key_parts = [
            "model",
            model_name.lower(),
            f"v{version}",
            action,
            str(model_id),
        ]
        
        # Add kwargs sorted alphabetically
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_hash = hashlib.md5(
                json.dumps(sorted_kwargs, sort_keys=True).encode()
            ).hexdigest()[:8]
            key_parts.append(kwargs_hash)
        
        return ":".join(key_parts)
    
    async def invalidate_model(
        self,
        model_name: str,
        model_id: Union[int, str, List[Union[int, str]]] = None,
        operation: str = "update",
        changed_fields: List[str] = None,
    ):
        """
        Invalidate cache for a model.
        
        Args:
            model_name: Name of the model
            model_id: Model ID(s) to invalidate (None for all)
            operation: Operation type (create, update, delete)
            changed_fields: List of fields that changed (for selective invalidation)
        """
        try:
            model_config = self._model_registry.get(model_name, {})
            
            # Determine which cache entries to invalidate
            patterns = await self._get_invalidation_patterns(
                model_name=model_name,
                model_id=model_id,
                operation=operation,
                changed_fields=changed_fields,
            )
            
            # Invalidate cache entries
            deleted_count = 0
            for pattern in patterns:
                deleted = await self.cache.delete_pattern(pattern)
                deleted_count += deleted
            
            # Cascade invalidation to dependent models
            if model_config.get("dependencies"):
                for dep_model in model_config["dependencies"]:
                    await self.invalidate_model(
                        model_name=dep_model,
                        model_id=None,  # Invalidate all
                        operation="dependency",
                    )
            
            # Cascade invalidation to related models
            if model_name in self._relationships:
                for relation in self._relationships[model_name]:
                    if relation["cascade_invalidate"]:
                        # Invalidate related child models
                        await self.invalidate_model(
                            model_name=relation["child_model"],
                            model_id=None,
                            operation=f"parent_{operation}",
                        )
            
            # Execute callbacks
            await self._execute_callbacks(
                model_name=model_name,
                model_id=model_id,
                operation=operation,
            )
            
            logger.info(
                f"Invalidated cache for {model_name}: "
                f"id={model_id}, operation={operation}, "
                f"deleted={deleted_count} entries"
            )
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error invalidating cache for {model_name}: {e}")
            return 0
    
    async def _get_invalidation_patterns(
        self,
        model_name: str,
        model_id: Union[int, str, List[Union[int, str]]] = None,
        operation: str = "update",
        changed_fields: List[str] = None,
    ) -> List[str]:
        """
        Get cache key patterns to invalidate.
        
        Args:
            model_name: Model name
            model_id: Model ID(s)
            operation: Operation type
            changed_fields: Changed fields
            
        Returns:
            List of cache key patterns
        """
        patterns = []
        version = self._model_registry.get(model_name, {}).get("version", 1)
        base_pattern = f"model:{model_name.lower()}:v{version}"
        
        if model_id is None:
            # Invalidate all cache for this model
            patterns.append(f"{base_pattern}:*")
        else:
            # Convert to list if single ID
            if not isinstance(model_id, list):
                model_ids = [model_id]
            else:
                model_ids = model_id
            
            for mid in model_ids:
                # Invalidate specific model instance
                patterns.append(f"{base_pattern}:*:{mid}:*")
                patterns.append(f"{base_pattern}:*:{mid}")
                
                # Invalidate lists that might contain this model
                patterns.append(f"{base_pattern}:list:*:{mid}:*")
                patterns.append(f"{base_pattern}:search:*:{mid}:*")
        
        # Selective field-based invalidation
        if changed_fields and operation == "update":
            model_config = self._model_registry.get(model_name, {})
            watched_fields = model_config.get("fields", [])
            
            # Only invalidate specific field caches
            for field in changed_fields:
                if field in watched_fields:
                    patterns.append(f"{base_pattern}:field:{field}:*")
        
        return list(set(patterns))  # Remove duplicates
    
    async def _execute_callbacks(
        self,
        model_name: str,
        model_id: Union[int, str, List[Union[int, str]]] = None,
        operation: str = "update",
    ):
        """Execute registered callbacks for invalidation events."""
        callbacks = self._invalidation_callbacks.get(model_name, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(model_name, model_id, operation)
                else:
                    callback(model_name, model_id, operation)
            except Exception as e:
                logger.error(f"Callback error for {model_name}: {e}")
    
    def register_callback(
        self,
        model_name: str,
        callback: Callable[[str, Union[int, str, List], str], None],
    ):
        """
        Register callback for invalidation events.
        
        Args:
            model_name: Model name
            callback: Callback function (model_name, model_id, operation)
        """
        if model_name not in self._invalidation_callbacks:
            self._invalidation_callbacks[model_name] = []
        
        self._invalidation_callbacks[model_name].append(callback)
    
    async def get_model(
        self,
        model_name: str,
        model_id: Union[int, str],
        fetch_func: Callable,
        ttl: int = None,
        **kwargs,
    ) -> Any:
        """
        Get model from cache or fetch from source.
        
        Args:
            model_name: Model name
            model_id: Model ID
            fetch_func: Function to fetch model if not in cache
            ttl: Cache TTL (uses model default if None)
            **kwargs: Additional parameters
            
        Returns:
            Model data
        """
        cache_key = self.generate_cache_key(
            model_name=model_name,
            model_id=model_id,
            action="get",
            **kwargs,
        )
        
        # Try to get from cache
        cached = await self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {model_name}:{model_id}")
            return cached
        
        # Fetch from source
        logger.debug(f"Cache miss for {model_name}:{model_id}, fetching...")
        data = await fetch_func() if asyncio.iscoroutinefunction(fetch_func) else fetch_func()
        
        if data is not None:
            # Determine TTL
            if ttl is None:
                ttl = self._model_registry.get(model_name, {}).get("ttl", 3600)
            
            # Store in cache
            await self.cache.set(cache_key, data, ttl=ttl)
        
        return data
    
    async def clear_model_cache(self, model_name: str = None):
        """
        Clear all cache for a model or all models.
        
        Args:
            model_name: Model name (None for all models)
        """
        if model_name:
            patterns = [f"model:{model_name.lower()}:*"]
        else:
            patterns = ["model:*"]
        
        deleted_total = 0
        for pattern in patterns:
            deleted = await self.cache.delete_pattern(pattern)
            deleted_total += deleted
        
        logger.info(f"Cleared model cache: {model_name or 'all'}, deleted {deleted_total} entries")
        return deleted_total


# Global instance and helper functions
_global_invalidator: Optional[ModelCacheInvalidator] = None

def get_model_invalidator() -> ModelCacheInvalidator:
    """Get or create global model cache invalidator."""
    global _global_invalidator
    
    if _global_invalidator is None:
        _global_invalidator = ModelCacheInvalidator()
    
    return _global_invalidator

def register_model_invalidator(
    model_name: str,
    **kwargs,
):
    """
    Register a model with the global invalidator.
    
    Args:
        model_name: Model name
        **kwargs: Additional parameters for register_model
    """
    invalidator = get_model_invalidator()
    invalidator.register_model(model_name, **kwargs)

async def invalidate_model_cache(
    model_name: str,
    model_id: Union[int, str, List[Union[int, str]]] = None,
    **kwargs,
) -> int:
    """
    Invalidate cache for a model using global invalidator.
    
    Args:
        model_name: Model name
        model_id: Model ID(s)
        **kwargs: Additional parameters for invalidate_model
        
    Returns:
        Number of cache entries deleted
    """
    invalidator = get_model_invalidator()
    return await invalidator.invalidate_model(model_name, model_id, **kwargs)