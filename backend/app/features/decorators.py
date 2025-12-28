"""
Decorators for feature flag integration.
Provides Python decorators for easy feature flag usage in functions and classes.
"""

import functools
import logging
import inspect
from typing import Dict, Any, Optional, List, Callable, Union, TypeVar, cast
from datetime import datetime, timedelta,time
import asyncio
import threading

from .manager import get_feature_manager, FeatureManager
from .flags import FeatureFlag, FeatureFlags, FlagType
from .context import FeatureContext, UserContext, create_feature_context
from .schemas import FeatureFlagResponse

logger = logging.getLogger(__name__)

T = TypeVar('T')
F = TypeVar('F', bound=Callable)


# ==================== FUNCTION DECORATORS ====================

def feature_flag(
    flag_name: str,
    enabled: bool = True,
    user_id: Optional[Union[str, Callable]] = None,
    context: Optional[Union[Dict[str, Any], Callable, FeatureContext]] = None,
    environment: Optional[str] = None,
    fallback: Optional[Callable] = None,
    fallback_value: Any = None,
    raise_on_disabled: bool = False,
    error_message: Optional[str] = None,
    track_usage: bool = True,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator to control function execution based on feature flag.
    
    Args:
        flag_name: Name of the feature flag to check
        enabled: Whether flag should be enabled or disabled (True=must be enabled)
        user_id: User ID for evaluation (can be callable that returns user_id)
        context: Evaluation context (can be callable that returns context)
        environment: Override environment for evaluation
        fallback: Fallback function to call if flag condition not met
        fallback_value: Value to return if flag condition not met and no fallback
        raise_on_disabled: Raise exception if flag condition not met
        error_message: Custom error message for exception
        track_usage: Track feature flag usage in metrics
        manager: FeatureManager instance (uses singleton if None)
    
    Returns:
        Decorator function
    
    Examples:
        >>> @feature_flag("new_feature", enabled=True)
        >>> def my_function():
        ...     return "New feature logic"
        >>>
        >>> @feature_flag("beta_feature", fallback_value="legacy")
        >>> def process_data():
        ...     return "Beta processing"
        >>>
        >>> @feature_flag("experimental", raise_on_disabled=True)
        >>> def experimental_op():
        ...     return "Experimental operation"
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id if specified as callable
            actual_user_id = _get_value(user_id, args, kwargs)
            
            # Get context if specified
            actual_context = _get_value(context, args, kwargs)
            
            # Evaluate flag
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                context=actual_context,
                environment=environment or "production",
            )
            
            # Check if condition is met
            condition_met = (enabled and is_enabled) or (not enabled and not is_enabled)
            
            if condition_met:
                # Flag condition met, execute function
                if track_usage:
                    _track_feature_usage(feature_manager, flag_name, actual_user_id, "enabled")
                return func(*args, **kwargs)
            else:
                # Flag condition not met
                if track_usage:
                    _track_feature_usage(feature_manager, flag_name, actual_user_id, "disabled")
                
                if raise_on_disabled:
                    msg = error_message or f"Feature flag '{flag_name}' condition not met"
                    raise FeatureFlagError(msg, flag_name, enabled, is_enabled)
                elif fallback:
                    return fallback(*args, **kwargs)
                else:
                    return fallback_value
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id if specified as callable
            actual_user_id = _get_value(user_id, args, kwargs)
            
            # Get context if specified
            actual_context = _get_value(context, args, kwargs)
            
            # Evaluate flag
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                context=actual_context,
                environment=environment or "production",
            )
            
            # Check if condition is met
            condition_met = (enabled and is_enabled) or (not enabled and not is_enabled)
            
            if condition_met:
                # Flag condition met, execute function
                if track_usage:
                    _track_feature_usage(feature_manager, flag_name, actual_user_id, "enabled")
                return await func(*args, **kwargs)
            else:
                # Flag condition not met
                if track_usage:
                    _track_feature_usage(feature_manager, flag_name, actual_user_id, "disabled")
                
                if raise_on_disabled:
                    msg = error_message or f"Feature flag '{flag_name}' condition not met"
                    raise FeatureFlagError(msg, flag_name, enabled, is_enabled)
                elif fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                else:
                    return fallback_value
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


def require_feature(
    flag_name: str,
    enabled: bool = True,
    error_message: Optional[str] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator that requires a feature flag to be in specific state.
    
    Raises FeatureFlagError if condition not met.
    
    Args:
        flag_name: Name of the feature flag
        enabled: Required flag state (True=enabled, False=disabled)
        error_message: Custom error message
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @require_feature("admin_mode", enabled=True)
        >>> def admin_operation():
        ...     return "Admin operation"
    """
    return feature_flag(
        flag_name=flag_name,
        enabled=enabled,
        raise_on_disabled=True,
        error_message=error_message,
        manager=manager,
    )


def with_feature_context(
    flag_name: str,
    context_key: str = "feature_flag",
    include_details: bool = False,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator that injects feature flag evaluation result into function.
    
    Args:
        flag_name: Name of the feature flag
        context_key: Key to use in kwargs for flag result
        include_details: Include detailed evaluation result (not just enabled state)
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @with_feature_context("dark_mode", context_key="dark_mode_enabled")
        >>> def render_ui(dark_mode_enabled: bool):
        ...     if dark_mode_enabled:
        ...         return "dark theme"
        ...     return "light theme"
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Evaluate flag
            result = feature_manager.evaluate(
                flag_name,
                environment="production",
            )
            
            # Add result to kwargs
            if context_key not in kwargs:
                if include_details:
                    kwargs[context_key] = result
                else:
                    kwargs[context_key] = result.enabled
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Evaluate flag
            result = feature_manager.evaluate(
                flag_name,
                environment="production",
            )
            
            # Add result to kwargs
            if context_key not in kwargs:
                if include_details:
                    kwargs[context_key] = result
                else:
                    kwargs[context_key] = result.enabled
            
            return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


def feature_variant(
    flag_name: str,
    variants: Dict[str, Callable],
    default_variant: Optional[str] = None,
    default_function: Optional[Callable] = None,
    user_id: Optional[Union[str, Callable]] = None,
    context: Optional[Union[Dict[str, Any], Callable]] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator for A/B testing with feature flag variants.
    
    Args:
        flag_name: Name of the multivariate feature flag
        variants: Dictionary mapping variant names to functions
        default_variant: Default variant to use if flag disabled
        default_function: Default function to use if flag disabled and no default_variant
        user_id: User ID for variant assignment
        context: Evaluation context
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @feature_variant(
        ...     "new_ui_design",
        ...     variants={
        ...         "design_a": design_a_implementation,
        ...         "design_b": design_b_implementation,
        ...     },
        ...     default_variant="design_a"
        ... )
        >>> def render_ui():
        ...     pass  # Implementation chosen by variant
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id if specified as callable
            actual_user_id = _get_value(user_id, args, kwargs)
            
            # Get context if specified
            actual_context = _get_value(context, args, kwargs)
            
            # Get variant
            variant = feature_manager.get_variant(
                flag_name,
                user_id=actual_user_id,
                context=actual_context,
                environment="production",
            )
            
            # Determine which function to call
            if variant is None or variant is False:
                # Flag disabled, use default
                if default_variant and default_variant in variants:
                    selected_function = variants[default_variant]
                elif default_function:
                    selected_function = default_function
                else:
                    selected_function = func
            elif isinstance(variant, dict) and "name" in variant:
                # Variant with name field
                variant_name = variant["name"]
                selected_function = variants.get(variant_name, func)
            elif isinstance(variant, str) and variant in variants:
                # Variant name string
                selected_function = variants[variant]
            else:
                # Unknown variant, use default
                if default_function:
                    selected_function = default_function
                else:
                    selected_function = func
            
            # Track variant usage
            _track_feature_usage(
                feature_manager,
                flag_name,
                actual_user_id,
                f"variant_{variant if isinstance(variant, str) else 'unknown'}",
            )
            
            # Call selected function
            return selected_function(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id if specified as callable
            actual_user_id = _get_value(user_id, args, kwargs)
            
            # Get context if specified
            actual_context = _get_value(context, args, kwargs)
            
            # Get variant
            variant = feature_manager.get_variant(
                flag_name,
                user_id=actual_user_id,
                context=actual_context,
                environment="production",
            )
            
            # Determine which function to call
            if variant is None or variant is False:
                # Flag disabled, use default
                if default_variant and default_variant in variants:
                    selected_function = variants[default_variant]
                elif default_function:
                    selected_function = default_function
                else:
                    selected_function = func
            elif isinstance(variant, dict) and "name" in variant:
                # Variant with name field
                variant_name = variant["name"]
                selected_function = variants.get(variant_name, func)
            elif isinstance(variant, str) and variant in variants:
                # Variant name string
                selected_function = variants[variant]
            else:
                # Unknown variant, use default
                if default_function:
                    selected_function = default_function
                else:
                    selected_function = func
            
            # Track variant usage
            _track_feature_usage(
                feature_manager,
                flag_name,
                actual_user_id,
                f"variant_{variant if isinstance(variant, str) else 'unknown'}",
            )
            
            # Call selected function
            if asyncio.iscoroutinefunction(selected_function):
                return await selected_function(*args, **kwargs)
            else:
                return selected_function(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


def gradual_rollout(
    flag_name: str,
    rollout_percentage: float = 0.0,
    user_id_getter: Optional[Callable] = None,
    fallback: Optional[Callable] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator for gradual rollout of features.
    
    Args:
        flag_name: Name of the feature flag
        rollout_percentage: Percentage of users to include (0.0 to 1.0)
        user_id_getter: Function to extract user_id from args/kwargs
        fallback: Fallback function for users not in rollout
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @gradual_rollout("new_api", rollout_percentage=0.1)
        >>> def process_request(user_id: str):
        ...     return "New API logic"
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            if not actual_user_id:
                # Try to extract from common patterns
                actual_user_id = _extract_user_id_from_args(args, kwargs)
            
            # Check if user is in rollout
            if actual_user_id:
                # Get flag to check rollout percentage
                flag = feature_manager.get_flag(flag_name)
                if flag:
                    is_in_rollout = _is_user_in_rollout(actual_user_id, flag.rollout_percentage)
                else:
                    is_in_rollout = _is_user_in_rollout(actual_user_id, rollout_percentage)
            else:
                # No user_id, use global flag state
                is_in_rollout = feature_manager.is_enabled(flag_name)
            
            if is_in_rollout:
                # User is in rollout
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "rollout_included")
                return func(*args, **kwargs)
            else:
                # User not in rollout
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "rollout_excluded")
                if fallback:
                    return fallback(*args, **kwargs)
                else:
                    # Return None or call original with safe defaults?
                    return None
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            if not actual_user_id:
                # Try to extract from common patterns
                actual_user_id = _extract_user_id_from_args(args, kwargs)
            
            # Check if user is in rollout
            if actual_user_id:
                # Get flag to check rollout percentage
                flag = feature_manager.get_flag(flag_name)
                if flag:
                    is_in_rollout = _is_user_in_rollout(actual_user_id, flag.rollout_percentage)
                else:
                    is_in_rollout = _is_user_in_rollout(actual_user_id, rollout_percentage)
            else:
                # No user_id, use global flag state
                is_in_rollout = feature_manager.is_enabled(flag_name)
            
            if is_in_rollout:
                # User is in rollout
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "rollout_included")
                return await func(*args, **kwargs)
            else:
                # User not in rollout
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "rollout_excluded")
                if fallback:
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    else:
                        return fallback(*args, **kwargs)
                else:
                    return None
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


# ==================== CLASS DECORATORS ====================

def class_feature_flag(
    flag_name: str,
    enabled: bool = True,
    methods: Optional[List[str]] = None,
    exclude_methods: Optional[List[str]] = None,
    user_id_getter: Optional[Union[str, Callable]] = None,
    context_getter: Optional[Union[str, Callable]] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator to apply feature flag to class methods.
    
    Args:
        flag_name: Name of the feature flag
        enabled: Whether flag should be enabled or disabled
        methods: List of method names to decorate (None = all public methods)
        exclude_methods: List of method names to exclude
        user_id_getter: Method name or callable to get user_id from self
        context_getter: Method name or callable to get context from self
        manager: FeatureManager instance
    
    Returns:
        Class decorator
    
    Example:
        >>> @class_feature_flag("new_algorithm", methods=["calculate", "process"])
        >>> class DataProcessor:
        ...     def calculate(self, user_id: str):
        ...         pass
        ...     def process(self, data):
        ...         pass
    """
    def decorator(cls: T) -> T:
        # Determine which methods to decorate
        if methods is None:
            # Decorate all public methods (not starting with _)
            methods_to_decorate = [
                name for name in dir(cls)
                if not name.startswith('_') and callable(getattr(cls, name))
            ]
        else:
            methods_to_decorate = methods
        
        if exclude_methods:
            methods_to_decorate = [m for m in methods_to_decorate if m not in exclude_methods]
        
        # Apply feature_flag decorator to each method
        for method_name in methods_to_decorate:
            method = getattr(cls, method_name)
            
            # Create wrapper that extracts user_id and context from self
            def make_wrapper(original_method):
                @functools.wraps(original_method)
                def wrapper(self, *args, **kwargs):
                    # Get user_id if user_id_getter specified
                    actual_user_id = None
                    if user_id_getter:
                        if isinstance(user_id_getter, str):
                            # Method name
                            if hasattr(self, user_id_getter):
                                actual_user_id = getattr(self, user_id_getter)()
                        else:
                            # Callable
                            actual_user_id = user_id_getter(self)
                    
                    # Get context if context_getter specified
                    actual_context = None
                    if context_getter:
                        if isinstance(context_getter, str):
                            # Method name
                            if hasattr(self, context_getter):
                                actual_context = getattr(self, context_getter)()
                        else:
                            # Callable
                            actual_context = context_getter(self)
                    
                    # Check feature flag
                    feature_manager = manager or get_feature_manager()
                    is_enabled = feature_manager.is_enabled(
                        flag_name,
                        user_id=actual_user_id,
                        context=actual_context,
                        environment="production",
                    )
                    
                    # Check condition
                    condition_met = (enabled and is_enabled) or (not enabled and not is_enabled)
                    
                    if condition_met:
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "enabled")
                        return original_method(self, *args, **kwargs)
                    else:
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "disabled")
                        raise FeatureFlagError(
                            f"Feature flag '{flag_name}' condition not met for method '{method_name}'",
                            flag_name,
                            enabled,
                            is_enabled,
                        )
                
                @functools.wraps(original_method)
                async def async_wrapper(self, *args, **kwargs):
                    # Get user_id if user_id_getter specified
                    actual_user_id = None
                    if user_id_getter:
                        if isinstance(user_id_getter, str):
                            # Method name
                            if hasattr(self, user_id_getter):
                                actual_user_id = getattr(self, user_id_getter)()
                        else:
                            # Callable
                            actual_user_id = user_id_getter(self)
                    
                    # Get context if context_getter specified
                    actual_context = None
                    if context_getter:
                        if isinstance(context_getter, str):
                            # Method name
                            if hasattr(self, context_getter):
                                actual_context = getattr(self, context_getter)()
                        else:
                            # Callable
                            actual_context = context_getter(self)
                    
                    # Check feature flag
                    feature_manager = manager or get_feature_manager()
                    is_enabled = feature_manager.is_enabled(
                        flag_name,
                        user_id=actual_user_id,
                        context=actual_context,
                        environment="production",
                    )
                    
                    # Check condition
                    condition_met = (enabled and is_enabled) or (not enabled and not is_enabled)
                    
                    if condition_met:
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "enabled")
                        return await original_method(self, *args, **kwargs)
                    else:
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "disabled")
                        raise FeatureFlagError(
                            f"Feature flag '{flag_name}' condition not met for method '{method_name}'",
                            flag_name,
                            enabled,
                            is_enabled,
                        )
                
                # Return appropriate wrapper
                if asyncio.iscoroutinefunction(original_method):
                    return async_wrapper
                else:
                    return wrapper
            
            # Apply the wrapper
            setattr(cls, method_name, make_wrapper(method))
        
        return cls
    
    return decorator


def feature_flag_property(
    flag_name: str,
    enabled: bool = True,
    fallback_value: Any = None,
    user_id_getter: Optional[Callable] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Create a property that depends on a feature flag.
    
    Args:
        flag_name: Name of the feature flag
        enabled: Whether flag should be enabled
        fallback_value: Value to return if flag condition not met
        user_id_getter: Callable to get user_id from self
        manager: FeatureManager instance
    
    Returns:
        Property decorator
    
    Example:
        >>> class UserSettings:
        ...     @feature_flag_property("dark_mode", enabled=True)
        ...     def theme(self):
        ...         return "dark"
        ...
        ...     @theme.setter
        ...     def theme(self, value):
        ...         self._theme = value
    """
    def decorator(func: Callable) -> property:
        @property
        @functools.wraps(func)
        def wrapper(self):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id if user_id_getter specified
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(self)
            
            # Check feature flag
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                environment="production",
            )
            
            # Check condition
            condition_met = (enabled and is_enabled) or (not enabled and not is_enabled)
            
            if condition_met:
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "enabled")
                return func(self)
            else:
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "disabled")
                return fallback_value
        
        return wrapper
    
    return decorator


# ==================== CONTEXT MANAGER ====================

class FeatureFlagContext:
    """
    Context manager for feature flags.
    
    Example:
        >>> with FeatureFlagContext("new_feature", enabled=True) as enabled:
        ...     if enabled:
        ...         do_new_thing()
        ...     else:
        ...         do_old_thing()
    """
    
    def __init__(
        self,
        flag_name: str,
        enabled: bool = True,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        environment: str = "production",
        manager: Optional[FeatureManager] = None,
    ):
        self.flag_name = flag_name
        self.enabled = enabled
        self.user_id = user_id
        self.context = context
        self.environment = environment
        self.manager = manager or get_feature_manager()
        self._is_enabled = False
    
    def __enter__(self) -> bool:
        """Enter context and evaluate flag."""
        self._is_enabled = self.manager.is_enabled(
            self.flag_name,
            user_id=self.user_id,
            context=self.context,
            environment=self.environment,
        )
        
        # Check condition
        condition_met = (self.enabled and self._is_enabled) or (not self.enabled and not self._is_enabled)
        
        if condition_met:
            _track_feature_usage(self.manager, self.flag_name, self.user_id, "enabled")
            return True
        else:
            _track_feature_usage(self.manager, self.flag_name, self.user_id, "disabled")
            return False
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        pass


# ==================== UTILITY DECORATORS ====================

def track_feature_usage(
    flag_name: str,
    user_id_getter: Optional[Callable] = None,
    success_getter: Optional[Callable] = None,
    metadata_getter: Optional[Callable] = None,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator to track feature flag usage.
    
    Args:
        flag_name: Name of the feature flag to track
        user_id_getter: Callable to extract user_id from args/kwargs
        success_getter: Callable to determine if operation was successful
        metadata_getter: Callable to extract metadata for tracking
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @track_feature_usage("search_algorithm", user_id_getter=lambda *a, **k: k.get('user_id'))
        >>> def search(query: str, user_id: str):
        ...     return f"Results for {query}"
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            if not actual_user_id:
                actual_user_id = _extract_user_id_from_args(args, kwargs)
            
            # Check if flag is enabled for user
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                environment="production",
            )
            
            # Track usage start
            start_time = datetime.utcnow()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Determine success
                success = True
                if success_getter:
                    success = success_getter(result, *args, **kwargs)
                
                # Get metadata
                metadata = None
                if metadata_getter:
                    metadata = metadata_getter(result, *args, **kwargs)
                
                # Track successful usage
                _track_detailed_usage(
                    feature_manager,
                    flag_name,
                    actual_user_id,
                    "success",
                    is_enabled,
                    start_time,
                    metadata,
                )
                
                return result
                
            except Exception as e:
                # Track failed usage
                _track_detailed_usage(
                    feature_manager,
                    flag_name,
                    actual_user_id,
                    "error",
                    is_enabled,
                    start_time,
                    {"error": str(e)},
                )
                raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            if not actual_user_id:
                actual_user_id = _extract_user_id_from_args(args, kwargs)
            
            # Check if flag is enabled for user
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                environment="production",
            )
            
            # Track usage start
            start_time = datetime.utcnow()
            
            try:
                # Execute function
                result = await func(*args, **kwargs)
                
                # Determine success
                success = True
                if success_getter:
                    success = success_getter(result, *args, **kwargs)
                
                # Get metadata
                metadata = None
                if metadata_getter:
                    metadata = metadata_getter(result, *args, **kwargs)
                
                # Track successful usage
                _track_detailed_usage(
                    feature_manager,
                    flag_name,
                    actual_user_id,
                    "success",
                    is_enabled,
                    start_time,
                    metadata,
                )
                
                return result
                
            except Exception as e:
                # Track failed usage
                _track_detailed_usage(
                    feature_manager,
                    flag_name,
                    actual_user_id,
                    "error",
                    is_enabled,
                    start_time,
                    {"error": str(e)},
                )
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


def feature_flag_cache(
    flag_name: str,
    cache_key_getter: Callable,
    enabled: bool = True,
    user_id_getter: Optional[Callable] = None,
    ttl: int = 300,
    manager: Optional[FeatureManager] = None,
):
    """
    Decorator that caches results based on feature flag state.
    
    Args:
        flag_name: Name of the feature flag
        cache_key_getter: Callable to generate cache key from args/kwargs
        enabled: Whether caching should be enabled when flag is enabled
        user_id_getter: Callable to extract user_id from args/kwargs
        ttl: Cache TTL in seconds
        manager: FeatureManager instance
    
    Returns:
        Decorator function
    
    Example:
        >>> @feature_flag_cache("cache_enabled", lambda query: f"search:{query}")
        >>> def search(query: str):
        ...     return expensive_search(query)
    """
    cache = {}
    cache_timestamps = {}
    cache_lock = threading.RLock()
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            # Check feature flag
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                environment="production",
            )
            
            # Check if caching should be enabled
            caching_enabled = (enabled and is_enabled) or (not enabled and not is_enabled)
            
            if not caching_enabled:
                # Caching disabled, execute function directly
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cache_key_getter(*args, **kwargs)
            
            with cache_lock:
                # Check cache
                current_time = time.time()
                if cache_key in cache:
                    timestamp = cache_timestamps.get(cache_key, 0)
                    if current_time - timestamp < ttl:
                        # Cache hit
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "cache_hit")
                        return cache[cache_key]
                
                # Cache miss or expired
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "cache_miss")
                result = func(*args, **kwargs)
                
                # Update cache
                cache[cache_key] = result
                cache_timestamps[cache_key] = current_time
                
                return result
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get feature manager
            feature_manager = manager or get_feature_manager()
            
            # Get user_id
            actual_user_id = None
            if user_id_getter:
                actual_user_id = user_id_getter(*args, **kwargs)
            
            # Check feature flag
            is_enabled = feature_manager.is_enabled(
                flag_name,
                user_id=actual_user_id,
                environment="production",
            )
            
            # Check if caching should be enabled
            caching_enabled = (enabled and is_enabled) or (not enabled and not is_enabled)
            
            if not caching_enabled:
                # Caching disabled, execute function directly
                return await func(*args, **kwargs)
            
            # Generate cache key
            cache_key = cache_key_getter(*args, **kwargs)
            
            with cache_lock:
                # Check cache
                current_time = time.time()
                if cache_key in cache:
                    timestamp = cache_timestamps.get(cache_key, 0)
                    if current_time - timestamp < ttl:
                        # Cache hit
                        _track_feature_usage(feature_manager, flag_name, actual_user_id, "cache_hit")
                        return cache[cache_key]
                
                # Cache miss or expired
                _track_feature_usage(feature_manager, flag_name, actual_user_id, "cache_miss")
                result = await func(*args, **kwargs)
                
                # Update cache
                cache[cache_key] = result
                cache_timestamps[cache_key] = current_time
                
                return result
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)
    
    return decorator


# ==================== HELPER FUNCTIONS ====================

def _get_value(value_or_callable, args, kwargs):
    """Get value from either a value or a callable."""
    if callable(value_or_callable):
        try:
            return value_or_callable(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Error calling callable: {e}")
            return None
    return value_or_callable


def _extract_user_id_from_args(args, kwargs) -> Optional[str]:
    """Extract user_id from function arguments."""
    # Check kwargs first
    if 'user_id' in kwargs:
        return str(kwargs['user_id'])
    if 'user' in kwargs and hasattr(kwargs['user'], 'id'):
        return str(kwargs['user'].id)
    
    # Check args by parameter name (requires inspection)
    # This is a simplified version
    for arg in args:
        if isinstance(arg, str) and len(arg) < 100:  # Simple string, could be user_id
            return arg
        elif hasattr(arg, 'id'):
            return str(arg.id)
    
    return None


def _is_user_in_rollout(user_id: str, rollout_percentage: float) -> bool:
    """Determine if user should be included in percentage rollout."""
    if rollout_percentage >= 1.0:
        return True
    if rollout_percentage <= 0.0:
        return False
    
    import hashlib
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest()[:8], 16)
    return (hash_val % 10000) / 10000.0 < rollout_percentage


def _track_feature_usage(
    manager: FeatureManager,
    flag_name: str,
    user_id: Optional[str],
    status: str,
):
    """Track feature flag usage."""
    try:
        # This would integrate with your analytics system
        # For now, we just log it
        logger.debug(f"Feature usage: {flag_name}, user: {user_id}, status: {status}")
        
        # You could also update metrics in the feature manager
        # manager.record_usage(flag_name, user_id, status)
        
    except Exception as e:
        logger.debug(f"Error tracking feature usage: {e}")


def _track_detailed_usage(
    manager: FeatureManager,
    flag_name: str,
    user_id: Optional[str],
    result: str,
    is_enabled: bool,
    start_time: datetime,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Track detailed feature flag usage."""
    try:
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.debug(
            f"Feature detailed usage: {flag_name}, "
            f"user: {user_id}, result: {result}, "
            f"enabled: {is_enabled}, duration: {duration:.2f}ms"
        )
        
        # Integrate with analytics system here
        # Example: manager.record_detailed_usage(...)
        
    except Exception as e:
        logger.debug(f"Error tracking detailed feature usage: {e}")


# ==================== EXCEPTIONS ====================

class FeatureFlagError(Exception):
    """Exception raised when feature flag condition is not met."""
    
    def __init__(
        self,
        message: str,
        flag_name: str,
        required_state: bool,
        actual_state: bool,
    ):
        self.message = message
        self.flag_name = flag_name
        self.required_state = required_state
        self.actual_state = actual_state
        
        required_str = "enabled" if required_state else "disabled"
        actual_str = "enabled" if actual_state else "disabled"
        
        super().__init__(
            f"{message} (flag: {flag_name}, required: {required_str}, actual: {actual_str})"
        )


class FeatureFlagEvaluationError(Exception):
    """Exception raised when feature flag evaluation fails."""
    
    def __init__(self, flag_name: str, error: str):
        self.flag_name = flag_name
        self.error = error
        super().__init__(f"Failed to evaluate feature flag '{flag_name}': {error}")


# ==================== EXPORTS ====================

__all__ = [
    # Function decorators
    "feature_flag",
    "require_feature",
    "with_feature_context",
    "feature_variant",
    "gradual_rollout",
    "track_feature_usage",
    "feature_flag_cache",
    
    # Class decorators
    "class_feature_flag",
    "feature_flag_property",
    
    # Context manager
    "FeatureFlagContext",
    
    # Exceptions
    "FeatureFlagError",
    "FeatureFlagEvaluationError",
]