"""
Feature flag system for WorldBrief360.
Enables controlled feature rollouts, A/B testing, and experimentation.

The feature flag system provides:
1. Gradual feature rollouts
2. A/B testing capabilities
3. User segmentation
4. Environment-specific configurations
5. Runtime feature toggling without deployments
"""

from .manager import get_feature_manager, FeatureManager
from .flags import FeatureFlag, FeatureFlags, FlagType, VariantType
from .context import FeatureContext, UserContext, SystemContext
from .evaluator import FeatureEvaluator
from .middleware import FeatureFlagMiddleware
from .schemas import FeatureFlagResponse, VariantAssignment
from .decorators import feature_flag, require_feature
from .utils import parse_variant_value, create_feature_key

# Default feature manager instance
from .backends.base import FeatureBackend
from .backends.in_memory import InMemoryBackend
from .backends.redis import RedisBackend
from .backends.unleash import UnleashBackend
from .backends.launchdarkly import LaunchDarklyBackend

__version__ = "1.0.0"

__all__ = [
    # Core components
    "get_feature_manager",
    "FeatureManager",
    
    # Flag definitions
    "FeatureFlag",
    "FeatureFlags",
    "FlagType",
    "VariantType",
    
    # Context
    "FeatureContext",
    "UserContext",
    "SystemContext",
    
    # Evaluation
    "FeatureEvaluator",
    
    # FastAPI integration
    "FeatureFlagMiddleware",
    
    # Schemas
    "FeatureFlagResponse",
    "VariantAssignment",
    
    # Decorators
    "feature_flag",
    "require_feature",
    
    # Utilities
    "parse_variant_value",
    "create_feature_key",
    
    # Backends
    "FeatureBackend",
    "InMemoryBackend",
    "RedisBackend",
    "UnleashBackend",
    "LaunchDarklyBackend",
]

# Initialize default feature flags (development only)
try:
    from . import default_flags
    DEFAULT_FLAGS = default_flags.get_default_flags()
except ImportError:
    DEFAULT_FLAGS = {}


def initialize_features(backend: str = "in_memory", **kwargs):
    """
    Initialize the feature flag system with specified backend.
    
    Args:
        backend: Backend type ('in_memory', 'redis', 'unleash', 'launchdarkly')
        **kwargs: Backend-specific configuration
    
    Returns:
        FeatureManager instance
    """
    from .manager import FeatureManager
    
    backend_map = {
        "in_memory": InMemoryBackend,
        "redis": RedisBackend,
        "unleash": UnleashBackend,
        "launchdarkly": LaunchDarklyBackend,
    }
    
    if backend not in backend_map:
        raise ValueError(f"Unsupported backend: {backend}. Choose from: {list(backend_map.keys())}")
    
    backend_instance = backend_map[backend](**kwargs)
    manager = FeatureManager(backend_instance)
    
    # Load default flags for development
    if backend == "in_memory" and DEFAULT_FLAGS:
        for flag_name, flag_config in DEFAULT_FLAGS.items():
            manager.create_flag(flag_name, **flag_config)
    
    return manager


# Convenience function for common checks
def is_enabled(flag_name: str, user_id: str = None, context: dict = None) -> bool:
    """
    Check if a feature flag is enabled for a user.
    
    Args:
        flag_name: Name of the feature flag
        user_id: Optional user ID for targeting
        context: Optional evaluation context
    
    Returns:
        True if feature is enabled, False otherwise
    """
    manager = get_feature_manager()
    return manager.is_enabled(flag_name, user_id, context)


def get_variant(flag_name: str, user_id: str = None, context: dict = None) -> any:
    """
    Get variant value for a feature flag.
    
    Args:
        flag_name: Name of the feature flag
        user_id: Optional user ID for targeting
        context: Optional evaluation context
    
    Returns:
        Variant value or None if flag doesn't exist
    """
    manager = get_feature_manager()
    return manager.get_variant(flag_name, user_id, context)


def with_features(features: list[str], user_id: str = None, context: dict = None) -> dict:
    """
    Get status of multiple feature flags at once.
    
    Args:
        features: List of feature flag names to check
        user_id: Optional user ID for targeting
        context: Optional evaluation context
    
    Returns:
        Dictionary mapping flag names to their status/variant
    """
    manager = get_feature_manager()
    return {
        flag_name: {
            "enabled": manager.is_enabled(flag_name, user_id, context),
            "variant": manager.get_variant(flag_name, user_id, context)
        }
        for flag_name in features
    }


# Export convenience functions
__all__.extend([
    "initialize_features",
    "is_enabled",
    "get_variant",
    "with_features",
])