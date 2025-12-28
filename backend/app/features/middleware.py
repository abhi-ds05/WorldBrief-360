"""
FastAPI middleware for feature flag integration.
Adds feature flag context to requests and provides request-based evaluation.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Callable, Set, Tuple
from datetime import datetime
from functools import wraps
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request, Response, Depends
from fastapi.routing import APIRoute
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Message, Receive, Send

from .manager import get_feature_manager, FeatureManager
from .context import (
    FeatureContext,
    RequestContext,
    UserContext,
    DeviceContext,
    GeoContext,
    SessionContext,
    ContextSource,
    create_request_context,
    create_feature_context,
    ContextBuilder,
)
from .flags import FeatureFlag, FeatureFlags
from .schemas import FeatureFlagResponse

logger = logging.getLogger(__name__)

# Context variable for storing feature context per request
request_context_var: ContextVar[Optional[FeatureContext]] = ContextVar("request_context", default=None)
feature_manager_var: ContextVar[Optional[FeatureManager]] = ContextVar("feature_manager", default=None)


class FeatureFlagMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that adds feature flag context to each request.
    
    Features:
    - Automatic context extraction from requests
    - User authentication integration
    - Device and geo detection
    - Request-scoped feature evaluation
    - Metrics collection per endpoint
    """
    
    def __init__(
        self,
        app: ASGIApp,
        manager: Optional[FeatureManager] = None,
        auto_extract: bool = True,
        include_headers: bool = True,
        metrics_enabled: bool = True,
        endpoint_metrics: bool = True,
        default_environment: str = "production",
    ):
        """
        Initialize middleware.
        
        Args:
            app: ASGI application
            manager: FeatureManager instance (uses singleton if None)
            auto_extract: Automatically extract context from request
            include_headers: Include request headers in context
            metrics_enabled: Enable request metrics collection
            endpoint_metrics: Collect per-endpoint feature metrics
            default_environment: Default environment for evaluation
        """
        super().__init__(app)
        self.manager = manager or get_feature_manager()
        self.auto_extract = auto_extract
        self.include_headers = include_headers
        self.metrics_enabled = metrics_enabled
        self.endpoint_metrics = endpoint_metrics
        self.default_environment = default_environment
        
        # Endpoint-specific metrics
        self.endpoint_metrics_data: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"FeatureFlagMiddleware initialized (auto_extract={auto_extract})")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request through middleware.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response
        """
        # Skip middleware for certain paths (health checks, static files, etc.)
        if self._should_skip(request):
            return await call_next(request)
        
        # Start timing for metrics
        request_start = datetime.utcnow()
        
        # Create feature context for this request
        context = await self._create_request_context(request)
        
        # Store context in contextvar
        context_token = request_context_var.set(context)
        manager_token = feature_manager_var.set(self.manager)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add feature context to response headers (optional)
            if self.include_headers:
                response = await self._add_context_headers(response, context)
            
            # Record metrics if enabled
            if self.metrics_enabled:
                await self._record_request_metrics(request, context, request_start, response)
            
            return response
            
        except Exception as e:
            # Log feature-related errors
            logger.error(f"Feature flag middleware error: {e}")
            raise
            
        finally:
            # Clean up contextvars
            request_context_var.reset(context_token)
            feature_manager_var.reset(manager_token)
    
    async def _create_request_context(self, request: Request) -> FeatureContext:
        """
        Create feature context from request.
        
        Args:
            request: HTTP request
        
        Returns:
            FeatureContext
        """
        builder = ContextBuilder()
        
        # Extract request context
        request_context = self._extract_request_context(request)
        builder.with_request(request_context)
        
        # Extract user context if available
        user_context = await self._extract_user_context(request)
        if user_context:
            builder.with_user(user_context)
        
        # Extract device context
        device_context = self._extract_device_context(request)
        if device_context:
            builder.with_device(device_context)
        
        # Extract geo context (if available)
        geo_context = self._extract_geo_context(request)
        if geo_context:
            builder.with_geo(geo_context)
        
        # Extract session context
        session_context = self._extract_session_context(request)
        if session_context:
            builder.with_session(session_context)
        
        # Add custom context from request state
        if hasattr(request.state, "feature_context"):
            custom_context = request.state.feature_context
            if isinstance(custom_context, dict):
                for key, value in custom_context.items():
                    builder.with_custom(key, value)
        
        # Build context
        context = builder.build()
        
        # Store in request state for later access
        request.state.feature_context = context
        
        return context
    
    def _extract_request_context(self, request: Request) -> RequestContext:
        """Extract request context from HTTP request."""
        # Get client IP (handling proxies)
        if "x-forwarded-for" in request.headers:
            ip_address = request.headers["x-forwarded-for"].split(",")[0].strip()
        else:
            ip_address = request.client.host if request.client else None
        
        # Create request context
        request_context = RequestContext(
            request_id=request.headers.get("x-request-id"),
            method=request.method,
            path=request.path,
            endpoint=request.url.path,
            ip_address=ip_address,
            user_agent=request.headers.get("user-agent"),
            referer=request.headers.get("referer"),
            headers=dict(request.headers) if self.include_headers else {},
            query_params=dict(request.query_params),
            request_start=datetime.utcnow(),
        )
        
        # Generate request ID if not present
        if not request_context.request_id:
            request_context.generate_request_id()
        
        return request_context
    
    async def _extract_user_context(self, request: Request) -> Optional[UserContext]:
        """Extract user context from request (supports common auth patterns)."""
        # Try to get user from request state (FastAPI dependency injection)
        if hasattr(request.state, "user"):
            user = request.state.user
            return self._create_user_context_from_obj(user)
        
        # Check for JWT in headers
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            # In a real implementation, you would decode the JWT
            # and extract user information
            pass
        
        # Check for session-based authentication
        session = request.cookies.get("session_id")
        if session:
            # In a real implementation, you would look up session in database
            pass
        
        # Check for API key
        api_key = request.headers.get("x-api-key")
        if api_key:
            # In a real implementation, you would validate API key
            pass
        
        return None
    
    def _create_user_context_from_obj(self, user_obj: Any) -> UserContext:
        """Create UserContext from user object (adapt based on your User model)."""
        # This is a generic implementation - adapt to your User model
        user_context = UserContext(
            user_id=str(getattr(user_obj, "id", None)),
            username=getattr(user_obj, "username", None),
            email=getattr(user_obj, "email", None),
            role=getattr(user_obj, "role", "user"),
            tier=getattr(user_obj, "tier", "free"),
            is_verified=getattr(user_obj, "is_verified", False),
            is_active=getattr(user_obj, "is_active", True),
            is_staff=getattr(user_obj, "is_staff", False),
            is_superuser=getattr(user_obj, "is_superuser", False),
            created_at=getattr(user_obj, "created_at", None),
            last_login=getattr(user_obj, "last_login", None),
            language=getattr(user_obj, "language", "en"),
            theme=getattr(user_obj, "theme", "light"),
        )
        
        # Add engagement metrics if available
        if hasattr(user_obj, "engagement_score"):
            user_context.engagement_score = user_obj.engagement_score
        if hasattr(user_obj, "days_active"):
            user_context.days_active = user_obj.days_active
        
        return user_context
    
    def _extract_device_context(self, request: Request) -> Optional[DeviceContext]:
        """Extract device context from request headers."""
        user_agent = request.headers.get("user-agent", "")
        
        # Simple device detection from user agent
        # In production, use a library like ua-parser
        is_mobile = any(term in user_agent.lower() for term in ["mobile", "android", "iphone"])
        is_tablet = any(term in user_agent.lower() for term in ["tablet", "ipad"])
        is_desktop = not (is_mobile or is_tablet)
        
        device_type = None
        if is_mobile:
            device_type = "mobile"
        elif is_tablet:
            device_type = "tablet"
        elif is_desktop:
            device_type = "desktop"
        
        return DeviceContext(
            device_type=device_type,
            user_agent=user_agent,
            is_mobile=is_mobile,
            is_tablet=is_tablet,
            is_desktop=is_desktop,
        )
    
    def _extract_geo_context(self, request: Request) -> Optional[GeoContext]:
        """Extract geographic context from request."""
        # Check for Cloudflare headers
        country = request.headers.get("cf-ipcountry")
        if country:
            return GeoContext(country=country)
        
        # Check for other common geo headers
        # In production, you might use a geolocation service based on IP
        
        return None
    
    def _extract_session_context(self, request: Request) -> Optional[SessionContext]:
        """Extract session context from request."""
        session_id = request.cookies.get("session_id") or request.headers.get("x-session-id")
        if not session_id:
            return None
        
        return SessionContext(
            session_id=session_id,
            session_start=datetime.utcnow(),
            is_new_session=False,  # Would need session store to determine
        )
    
    async def _add_context_headers(self, response: Response, context: FeatureContext) -> Response:
        """Add feature context information to response headers."""
        # Add context ID for debugging
        if context.evaluation_id:
            response.headers["X-Feature-Context-ID"] = context.evaluation_id
        
        # Add feature flags that were evaluated during request
        # (This would require tracking which flags were evaluated)
        
        return response
    
    async def _record_request_metrics(
        self,
        request: Request,
        context: FeatureContext,
        start_time: datetime,
        response: Response,
    ):
        """Record metrics for the request."""
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Get endpoint path
        endpoint = request.url.path
        
        # Initialize endpoint metrics if needed
        if endpoint not in self.endpoint_metrics_data:
            self.endpoint_metrics_data[endpoint] = {
                "total_requests": 0,
                "total_duration": 0,
                "feature_evaluations": {},
                "last_request": None,
            }
        
        # Update endpoint metrics
        endpoint_data = self.endpoint_metrics_data[endpoint]
        endpoint_data["total_requests"] += 1
        endpoint_data["total_duration"] += duration
        endpoint_data["last_request"] = datetime.utcnow().isoformat()
        
        # Record feature evaluations if we tracked them
        # (This would require modifying flag evaluation to track per-request)
        
        logger.debug(f"Request metrics: {endpoint} - {duration:.2f}ms")
    
    def _should_skip(self, request: Request) -> bool:
        """Determine if middleware should skip this request."""
        skip_paths = {
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/favicon.ico",
            "/static/",
            "/assets/",
        }
        
        path = request.url.path
        return any(path.startswith(skip_path) for skip_path in skip_paths)
    
    def get_endpoint_metrics(self) -> Dict[str, Any]:
        """Get metrics for all endpoints."""
        return self.endpoint_metrics_data


# Dependency injection functions
def get_request_context() -> Optional[FeatureContext]:
    """
    FastAPI dependency to get request feature context.
    
    Returns:
        FeatureContext for current request
    """
    return request_context_var.get()


def get_feature_manager_dep() -> FeatureManager:
    """
    FastAPI dependency to get feature manager.
    
    Returns:
        FeatureManager instance
    """
    manager = feature_manager_var.get()
    if manager is None:
        manager = get_feature_manager()
    return manager


# Decorators for endpoints
def feature_flag_required(
    flag_name: str,
    enabled: bool = True,
    environment: Optional[str] = None,
    fallback_endpoint: Optional[str] = None,
    error_response: Optional[Dict[str, Any]] = None,
):
    """
    Decorator to require a feature flag for endpoint access.
    
    Args:
        flag_name: Name of required feature flag
        enabled: Required flag state (True for enabled, False for disabled)
        environment: Override environment for evaluation
        fallback_endpoint: Redirect to this endpoint if flag not met
        error_response: Custom error response if flag not met
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs (FastAPI pattern)
            request = kwargs.get("request")
            if not request:
                # Try to find request in args (depends on function signature)
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise RuntimeError("Request not found in function arguments")
            
            # Get context from request state
            context = getattr(request.state, "feature_context", None)
            if not context:
                # Create minimal context
                context = create_feature_context()
            
            # Get manager
            manager = get_feature_manager_dep()
            
            # Evaluate flag
            is_enabled = manager.is_enabled(
                flag_name,
                user_id=context.get_user_id(),
                context=context,
                environment=environment or "production",
            )
            
            # Check if flag meets requirement
            if (enabled and not is_enabled) or (not enabled and is_enabled):
                if fallback_endpoint:
                    # Redirect to fallback endpoint
                    from fastapi.responses import RedirectResponse
                    return RedirectResponse(url=fallback_endpoint)
                elif error_response:
                    # Return custom error response
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=403,
                        content=error_response,
                    )
                else:
                    # Return default 403
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=403,
                        content={
                            "error": "feature_flag_required",
                            "flag_name": flag_name,
                            "required_state": "enabled" if enabled else "disabled",
                            "actual_state": "enabled" if is_enabled else "disabled",
                            "message": f"Feature flag '{flag_name}' must be {('enabled' if enabled else 'disabled')}",
                        },
                    )
            
            # Flag requirement met, proceed with endpoint
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


def with_feature_flag(
    flag_name: str,
    environment: Optional[str] = None,
    pass_result: bool = False,
    result_key: str = "feature_flag",
):
    """
    Decorator to inject feature flag evaluation result into endpoint.
    
    Args:
        flag_name: Name of feature flag to evaluate
        environment: Override environment for evaluation
        pass_result: Whether to pass result to endpoint function
        result_key: Key to use when passing result
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs (FastAPI pattern)
            request = kwargs.get("request")
            if not request:
                # Try to find request in args
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise RuntimeError("Request not found in function arguments")
            
            # Get context from request state
            context = getattr(request.state, "feature_context", None)
            if not context:
                context = create_feature_context()
            
            # Get manager
            manager = get_feature_manager_dep()
            
            # Evaluate flag
            result = manager.evaluate(
                flag_name,
                user_id=context.get_user_id(),
                context=context,
                environment=environment or "production",
            )
            
            # Store result in request state
            if not hasattr(request.state, "feature_flags"):
                request.state.feature_flags = {}
            request.state.feature_flags[flag_name] = result
            
            # Pass result to function if requested
            if pass_result:
                kwargs[result_key] = result
            
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# FastAPI dependency for checking feature flags
def require_feature(
    flag_name: str,
    enabled: bool = True,
    environment: Optional[str] = None,
):
    """
    FastAPI dependency to require a feature flag.
    
    Args:
        flag_name: Name of required feature flag
        enabled: Required flag state
        environment: Override environment
    
    Returns:
        Dependency function
    """
    async def dependency(
        request: Request,
        manager: FeatureManager = Depends(get_feature_manager_dep),
        context: Optional[FeatureContext] = Depends(get_request_context),
    ):
        # Use provided context or create from request
        if not context:
            context = getattr(request.state, "feature_context", None)
            if not context:
                context = create_feature_context()
        
        # Evaluate flag
        is_enabled = manager.is_enabled(
            flag_name,
            user_id=context.get_user_id(),
            context=context,
            environment=environment or "production",
        )
        
        # Check requirement
        if (enabled and not is_enabled) or (not enabled and is_enabled):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "feature_flag_required",
                    "flag_name": flag_name,
                    "required_state": "enabled" if enabled else "disabled",
                    "actual_state": "enabled" if is_enabled else "disabled",
                }
            )
        
        return True
    
    return dependency


# Helper functions for manual context manipulation
def set_request_context(context: FeatureContext):
    """
    Set feature context for current request.
    Useful in custom middleware or before endpoints.
    
    Args:
        context: FeatureContext to set
    """
    request_context_var.set(context)


def add_to_context(key: str, value: Any):
    """
    Add custom data to current request's feature context.
    
    Args:
        key: Context key
        value: Context value
    """
    context = request_context_var.get()
    if context:
        context.custom[key] = value
    else:
        # Create new context if none exists
        context = FeatureContext(custom={key: value})
        request_context_var.set(context)


# API endpoint utilities
def create_feature_flag_router(app: FastAPI, prefix: str = "/features"):
    """
    Create and register feature flag API endpoints.
    
    Args:
        app: FastAPI application
        prefix: URL prefix for feature endpoints
    """
    from fastapi import APIRouter, Depends, Query, Body, HTTPException
    
    router = APIRouter(prefix=prefix, tags=["features"])
    
    @router.get("/", response_model=List[Dict[str, Any]])
    async def list_flags(
        enabled_only: bool = Query(True, description="Only return enabled flags"),
        manager: FeatureManager = Depends(get_feature_manager_dep),
    ):
        """List all feature flags."""
        flags = manager.get_all_flags(include_disabled=not enabled_only)
        return [flag.to_dict() for flag in flags.values()]
    
    @router.get("/{flag_name}", response_model=Dict[str, Any])
    async def get_flag(
        flag_name: str,
        manager: FeatureManager = Depends(get_feature_manager_dep),
    ):
        """Get details for a specific flag."""
        flag = manager.get_flag(flag_name)
        if not flag:
            raise HTTPException(status_code=404, detail=f"Flag '{flag_name}' not found")
        
        status = manager.get_flag_status(flag_name)
        return status
    
    @router.post("/{flag_name}/evaluate", response_model=FeatureFlagResponse)
    async def evaluate_flag(
        flag_name: str,
        user_id: Optional[str] = Body(None),
        context: Optional[Dict[str, Any]] = Body(None),
        environment: str = Body("production"),
        manager: FeatureManager = Depends(get_feature_manager_dep),
    ):
        """Evaluate a feature flag for a user."""
        result = manager.evaluate(flag_name, user_id, context, environment)
        return result
    
    @router.get("/{flag_name}/metrics", response_model=Dict[str, Any])
    async def get_flag_metrics(
        flag_name: str,
        manager: FeatureManager = Depends(get_feature_manager_dep),
    ):
        """Get metrics for a feature flag."""
        metrics = manager.get_metrics(flag_name)
        if not metrics:
            raise HTTPException(status_code=404, detail=f"No metrics for flag '{flag_name}'")
        return metrics
    
    @router.get("/stats", response_model=Dict[str, Any])
    async def get_stats(
        manager: FeatureManager = Depends(get_feature_manager_dep),
    ):
        """Get feature flag system statistics."""
        return manager.get_stats()
    
    # Register router with app
    app.include_router(router)


# Middleware factory for easy integration
def create_feature_flag_middleware(
    manager: Optional[FeatureManager] = None,
    auto_extract: bool = True,
    include_headers: bool = True,
    metrics_enabled: bool = True,
    endpoint_metrics: bool = True,
    default_environment: str = "production",
) -> Middleware:
    """
    Create feature flag middleware configuration.
    
    Args:
        manager: FeatureManager instance
        auto_extract: Auto-extract context from requests
        include_headers: Include request headers in context
        metrics_enabled: Enable request metrics
        endpoint_metrics: Collect per-endpoint metrics
        default_environment: Default environment
    
    Returns:
        Starlette Middleware configuration
    """
    return Middleware(
        FeatureFlagMiddleware,
        manager=manager,
        auto_extract=auto_extract,
        include_headers=include_headers,
        metrics_enabled=metrics_enabled,
        endpoint_metrics=endpoint_metrics,
        default_environment=default_environment,
    )


# Context processor for templates (if using templating)
class FeatureContextProcessor:
    """Context processor for template rendering."""
    
    def __init__(self, manager: Optional[FeatureManager] = None):
        self.manager = manager or get_feature_manager()
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Add feature flags to template context."""
        context = {}
        
        # Get request feature context
        feature_context = getattr(request.state, "feature_context", None)
        if not feature_context:
            return context
        
        # Add commonly used flags
        user_id = feature_context.get_user_id()
        
        # You can add specific flags you want in templates
        context["features"] = {
            "dark_mode": self.manager.is_enabled("dark_mode", user_id, feature_context),
            "audio_briefings": self.manager.is_enabled("audio_briefings", user_id, feature_context),
            # Add more as needed
        }
        
        return context