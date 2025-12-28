"""
FastAPI middleware for event-driven architecture integration.

This module provides middleware for:
- Event context propagation
- Request/response event emission
- Error event handling
- Performance monitoring
- Correlation ID management
"""
import asyncio
import time
import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, List, Tuple
from datetime import datetime

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Send

from app.core.logging_config import logger
from app.events.event_bus import get_event_bus, publish_event_nowait
from app.events.event_types import EventType, EventPriority, EventContext, create_event
from app.events.event_schemas import (
    UserActivityEvent,
    FeatureUsageEvent,
    SystemAlertEvent,
    ExternalAPICalledEvent
)


# Context variables for request-scoped data
request_start_time = ContextVar('request_start_time', default=0.0)
current_request_id = ContextVar('current_request_id', default=None)
current_correlation_id = ContextVar('current_correlation_id', default=None)
current_user_id = ContextVar('current_user_id', default=None)
event_context_store = ContextVar('event_context_store', default=None)


class EventContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for managing event context across requests.
    
    This middleware:
    1. Sets up request context (IDs, user info, timing)
    2. Propagates correlation IDs
    3. Creates event context for downstream handlers
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._event_bus = get_event_bus()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Get correlation ID from headers or generate new
        correlation_id = request.headers.get('X-Correlation-ID') or str(uuid.uuid4())
        
        # Get user ID from authentication (if available)
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = getattr(request.state.user, 'id', None)
        
        # Set context variables
        token_request_id = current_request_id.set(request_id)
        token_correlation_id = current_correlation_id.set(correlation_id)
        token_user_id = current_user_id.set(user_id)
        token_start_time = request_start_time.set(time.time())
        
        try:
            # Create event context
            context = EventContext(
                user_id=user_id,
                session_id=request.headers.get('X-Session-ID'),
                request_id=request_id,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get('user-agent'),
                trace_id=request.headers.get('X-Trace-ID'),
                tenant_id=request.headers.get('X-Tenant-ID'),
                environment=request.app.state.settings.ENVIRONMENT if hasattr(request.app.state, 'settings') else 'production'
            )
            
            # Store context
            token_context = event_context_store.set(context)
            
            # Add correlation ID to response headers
            response = await call_next(request)
            response.headers['X-Request-ID'] = request_id
            response.headers['X-Correlation-ID'] = correlation_id
            
            # Emit request completion event
            await self._emit_request_event(request, response, context)
            
            return response
            
        except Exception as exc:
            # Emit error event
            await self._emit_error_event(request, exc, context)
            raise
            
        finally:
            # Clean up context variables
            current_request_id.reset(token_request_id)
            current_correlation_id.reset(token_correlation_id)
            current_user_id.reset(token_user_id)
            request_start_time.reset(token_start_time)
            event_context_store.reset(token_context)
    
    async def _emit_request_event(self, request: Request, response: Response, context: EventContext) -> None:
        """Emit an event for the completed request."""
        try:
            duration = time.time() - request_start_time.get()
            
            event_data = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "status_code": response.status_code,
                "duration_ms": duration * 1000,
                "request_id": context.request_id,
                "user_id": context.user_id,
                "ip_address": context.ip_address,
                "user_agent": context.user_agent
            }
            
            # Emit based on status code
            if response.status_code >= 500:
                event_type = EventType.ANALYTICS_ERROR_OCCURRED
            elif response.status_code >= 400:
                event_type = EventType.AUTH_FAILED_ATTEMPT if response.status_code == 401 else EventType.ANALYTICS_EVENT_TRACKED
            else:
                event_type = EventType.ANALYTICS_EVENT_TRACKED
            
            await self._event_bus.publish(
                event_type,
                event_data,
                correlation_id=context.request_id
            )
            
        except Exception as e:
            logger.error(f"Failed to emit request event: {e}")
    
    async def _emit_error_event(self, request: Request, exc: Exception, context: EventContext) -> None:
        """Emit an event for request errors."""
        try:
            event_data = {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_id": context.request_id,
                "user_id": context.user_id,
                "ip_address": context.ip_address
            }
            
            await self._event_bus.publish(
                EventType.ANALYTICS_ERROR_OCCURRED,
                event_data,
                correlation_id=context.request_id
            )
            
        except Exception as e:
            logger.error(f"Failed to emit error event: {e}")


class EventEmissionMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic event emission based on request patterns.
    
    This middleware:
    1. Emits events for specific API endpoints
    2. Tracks feature usage
    3. Records user activity
    """
    
    # Map of API paths to event types
    PATH_EVENT_MAPPINGS = {
        # User endpoints
        "/api/v1/auth/register": EventType.USER_REGISTERED,
        "/api/v1/auth/login": EventType.USER_LOGIN,
        "/api/v1/auth/logout": EventType.USER_LOGOUT,
        "/api/v1/users/{id}/profile": EventType.USER_PROFILE_UPDATED,
        
        # Incident endpoints
        "/api/v1/incidents": EventType.INCIDENT_REPORTED,
        "/api/v1/incidents/{id}/verify": EventType.INCIDENT_VERIFIED,
        "/api/v1/incidents/{id}/resolve": EventType.INCIDENT_RESOLVED,
        
        # Briefing endpoints
        "/api/v1/briefings": EventType.BRIEFING_REQUESTED,
        "/api/v1/briefings/{id}/deliver": EventType.BRIEFING_DELIVERED,
        "/api/v1/briefings/{id}/feedback": EventType.BRIEFING_FEEDBACK_GIVEN,
        
        # Chat endpoints
        "/api/v1/chat/sessions": EventType.CHAT_SESSION_STARTED,
        "/api/v1/chat/messages": EventType.CHAT_MESSAGE_SENT,
        
        # Wallet endpoints
        "/api/v1/wallet/transactions": EventType.WALLET_TRANSACTION,
    }
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self._event_bus = get_event_bus()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Only emit events for successful requests
        if 200 <= response.status_code < 300:
            await self._emit_mapped_event(request, response)
        
        return response
    
    async def _emit_mapped_event(self, request: Request, response: Response) -> None:
        """Emit event based on API path mapping."""
        try:
            # Check if path matches any mapped pattern
            for path_pattern, event_type in self.PATH_EVENT_MAPPINGS.items():
                if self._path_matches(request.url.path, path_pattern):
                    await self._emit_event_for_path(request, response, event_type, path_pattern)
                    break
        except Exception as e:
            logger.error(f"Failed to emit mapped event: {e}")
    
    def _path_matches(self, request_path: str, pattern: str) -> bool:
        """Check if request path matches pattern with placeholders."""
        # Simple pattern matching - replace {id} with wildcard
        pattern_regex = pattern.replace('{id}', '[^/]+').replace('{uuid}', '[^/]+')
        import re
        return bool(re.match(f"^{pattern_regex}$", request_path))
    
    async def _emit_event_for_path(self, request: Request, response: Response, 
                                  event_type: EventType, path_pattern: str) -> None:
        """Emit a specific event for a matched path."""
        try:
            # Get request body if available and appropriate
            request_body = None
            if request.method in ['POST', 'PUT', 'PATCH']:
                try:
                    request_body = await request.json()
                except:
                    request_body = await request.body()
            
            # Get response body if available
            response_body = None
            if hasattr(response, 'body'):
                response_body = response.body
            
            # Extract path parameters
            path_params = self._extract_path_params(request.url.path, path_pattern)
            
            # Prepare event data
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "path_pattern": path_pattern,
                "path_params": path_params,
                "query_params": dict(request.query_params),
                "request_body": request_body,
                "response_status": response.status_code,
                "user_id": current_user_id.get(),
                "request_id": current_request_id.get(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add specific data based on event type
            if event_type == EventType.USER_REGISTERED and request_body:
                event_data.update({
                    "email": request_body.get('email'),
                    "username": request_body.get('username'),
                    "registration_source": request.headers.get('X-Registration-Source', 'web')
                })
            elif event_type == EventType.INCIDENT_REPORTED and request_body:
                event_data.update({
                    "incident_type": request_body.get('type'),
                    "severity": request_body.get('severity'),
                    "location": request_body.get('location')
                })
            
            # Emit event
            await self._event_bus.publish(
                event_type,
                event_data,
                correlation_id=current_correlation_id.get()
            )
            
        except Exception as e:
            logger.error(f"Failed to emit event for path {path_pattern}: {e}")
    
    def _extract_path_params(self, request_path: str, pattern: str) -> Dict[str, str]:
        """Extract path parameters from request path using pattern."""
        params = {}
        request_parts = request_path.strip('/').split('/')
        pattern_parts = pattern.strip('/').split('/')
        
        if len(request_parts) != len(pattern_parts):
            return params
        
        for req_part, pat_part in zip(request_parts, pattern_parts):
            if pat_part.startswith('{') and pat_part.endswith('}'):
                param_name = pat_part[1:-1]
                params[param_name] = req_part
        
        return params


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for performance monitoring and alerting.
    
    This middleware:
    1. Measures request/response times
    2. Emits performance events
    3. Triggers alerts for slow requests
    """
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self._event_bus = get_event_bus()
        self.slow_request_threshold = slow_request_threshold  # seconds
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Emit performance event
            await self._emit_performance_event(request, response, duration)
            
            # Check for slow requests
            if duration > self.slow_request_threshold:
                await self._emit_slow_request_alert(request, duration)
            
            return response
            
        except Exception as exc:
            duration = time.time() - start_time
            await self._emit_error_performance_event(request, exc, duration)
            raise
    
    async def _emit_performance_event(self, request: Request, response: Response, duration: float) -> None:
        """Emit performance monitoring event."""
        try:
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration * 1000,
                "status_code": response.status_code,
                "request_id": current_request_id.get(),
                "user_id": current_user_id.get(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._event_bus.publish(
                EventType.ANALYTICS_PERFORMANCE_MEASURED,
                event_data,
                correlation_id=current_correlation_id.get()
            )
            
        except Exception as e:
            logger.error(f"Failed to emit performance event: {e}")
    
    async def _emit_slow_request_alert(self, request: Request, duration: float) -> None:
        """Emit alert for slow request."""
        try:
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration * 1000,
                "threshold_ms": self.slow_request_threshold * 1000,
                "request_id": current_request_id.get(),
                "user_id": current_user_id.get(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._event_bus.publish(
                EventType.MONITORING_ALERT_TRIGGERED,
                event_data,
                correlation_id=current_correlation_id.get(),
                priority=EventPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Failed to emit slow request alert: {e}")
    
    async def _emit_error_performance_event(self, request: Request, exc: Exception, duration: float) -> None:
        """Emit performance event for failed request."""
        try:
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "duration_ms": duration * 1000,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "request_id": current_request_id.get(),
                "user_id": current_user_id.get(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._event_bus.publish(
                EventType.ANALYTICS_ERROR_OCCURRED,
                event_data,
                correlation_id=current_correlation_id.get()
            )
            
        except Exception as e:
            logger.error(f"Failed to emit error performance event: {e}")


class EventContextPropagationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for propagating event context to external API calls.
    
    This middleware adds correlation IDs and context information
    to outgoing HTTP requests.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Store original send function
        original_send = request.scope.get('send')
        
        if original_send:
            # Wrap send to intercept outgoing requests
            async def wrapped_send(message: Message) -> None:
                if message['type'] == 'http.request':
                    # Add context headers to request
                    headers = message.get('headers', [])
                    
                    # Add correlation ID
                    correlation_id = current_correlation_id.get()
                    if correlation_id:
                        headers.append((b'x-correlation-id', correlation_id.encode()))
                    
                    # Add request ID
                    request_id = current_request_id.get()
                    if request_id:
                        headers.append((b'x-request-id', request_id.encode()))
                    
                    # Add user ID if available
                    user_id = current_user_id.get()
                    if user_id:
                        headers.append((b'x-user-id', str(user_id).encode()))
                    
                    # Update message with new headers
                    message['headers'] = headers
                
                await original_send(message)
            
            # Replace send in scope
            request.scope['send'] = wrapped_send
        
        return await call_next(request)


class EventRateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting based on event emissions.
    
    This middleware tracks request rates and emits events
    when thresholds are exceeded.
    """
    
    def __init__(self, app: ASGIApp, limits: Optional[Dict[str, Tuple[int, int]]] = None):
        super().__init__(app)
        self._event_bus = get_event_bus()
        self._limits = limits or {
            "auth": (10, 60),      # 10 requests per minute for auth
            "incidents": (50, 60),  # 50 requests per minute for incidents
            "api": (100, 60),      # 100 requests per minute overall
        }
        self._request_counts: Dict[str, List[float]] = {}
        self._cleanup_interval = 300  # Cleanup every 5 minutes
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Determine rate limit key
        limit_key = self._get_limit_key(request)
        
        # Check rate limit
        if limit_key and self._is_rate_limited(limit_key):
            await self._emit_rate_limit_event(request, limit_key)
            from fastapi import HTTPException
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Record request
        if limit_key:
            self._record_request(limit_key)
        
        # Periodically cleanup old records
        if int(time.time()) % self._cleanup_interval == 0:
            self._cleanup_old_records()
        
        return await call_next(request)
    
    def _get_limit_key(self, request: Request) -> Optional[str]:
        """Get rate limit key based on request."""
        path = request.url.path
        
        if path.startswith('/api/v1/auth'):
            return "auth"
        elif path.startswith('/api/v1/incidents'):
            return "incidents"
        elif path.startswith('/api/v1/'):
            return "api"
        
        return None
    
    def _is_rate_limited(self, key: str) -> bool:
        """Check if rate limit is exceeded for a key."""
        if key not in self._limits:
            return False
        
        max_requests, time_window = self._limits[key]
        current_time = time.time()
        
        # Get recent requests for this key
        recent_requests = [
            t for t in self._request_counts.get(key, [])
            if current_time - t <= time_window
        ]
        
        return len(recent_requests) >= max_requests
    
    def _record_request(self, key: str) -> None:
        """Record a request for rate limiting."""
        if key not in self._request_counts:
            self._request_counts[key] = []
        
        self._request_counts[key].append(time.time())
    
    def _cleanup_old_records(self) -> None:
        """Clean up old request records."""
        current_time = time.time()
        max_age = max(window for _, window in self._limits.values())
        
        for key in list(self._request_counts.keys()):
            self._request_counts[key] = [
                t for t in self._request_counts[key]
                if current_time - t <= max_age
            ]
            
            # Remove empty lists
            if not self._request_counts[key]:
                del self._request_counts[key]
    
    async def _emit_rate_limit_event(self, request: Request, limit_key: str) -> None:
        """Emit event when rate limit is exceeded."""
        try:
            max_requests, time_window = self._limits[limit_key]
            recent_count = len(self._request_counts.get(limit_key, []))
            
            event_data = {
                "method": request.method,
                "path": request.url.path,
                "limit_key": limit_key,
                "max_requests": max_requests,
                "time_window": time_window,
                "current_count": recent_count,
                "user_id": current_user_id.get(),
                "ip_address": request.client.host if request.client else None,
                "user_agent": request.headers.get('user-agent'),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            await self._event_bus.publish(
                EventType.SECURITY_THREAT_DETECTED,
                event_data,
                correlation_id=current_correlation_id.get(),
                priority=EventPriority.HIGH
            )
            
        except Exception as e:
            logger.error(f"Failed to emit rate limit event: {e}")


class EventContextGetter:
    """
    Utility class for accessing event context in request handlers.
    """
    
    @staticmethod
    def get_context() -> Optional[EventContext]:
        """Get the current event context."""
        return event_context_store.get()
    
    @staticmethod
    def get_request_id() -> Optional[str]:
        """Get the current request ID."""
        return current_request_id.get()
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get the current correlation ID."""
        return current_correlation_id.get()
    
    @staticmethod
    def get_user_id() -> Optional[str]:
        """Get the current user ID."""
        return current_user_id.get()
    
    @staticmethod
    def create_child_context(**kwargs) -> EventContext:
        """Create a child context with additional properties."""
        parent = event_context_store.get()
        if parent:
            return parent.merge(EventContext(**kwargs))
        return EventContext(**kwargs)


def setup_event_middleware(app: FastAPI) -> None:
    """
    Set up all event-related middleware for the FastAPI application.
    
    This should be called during application initialization.
    """
    logger.info("Setting up event middleware...")
    
    # Add middleware in order of execution
    # (First added = outermost = executed last in request, first in response)
    
    # 1. Rate limiting (first to reject requests early)
    app.add_middleware(EventRateLimitingMiddleware)
    
    # 2. Context propagation (needs to be before context middleware)
    app.add_middleware(EventContextPropagationMiddleware)
    
    # 3. Event context (sets up context for other middleware)
    app.add_middleware(EventContextMiddleware)
    
    # 4. Performance monitoring (measures full request time)
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    # 5. Event emission (emits events based on request patterns)
    app.add_middleware(EventEmissionMiddleware)
    
    # 6. Standard FastAPI middleware (after our custom middleware)
    # Add CORS if not already added
    if not any(isinstance(m, CORSMiddleware) for m in app.user_middleware):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Add TrustedHostMiddleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    
    logger.info("Event middleware setup complete")


# Context manager for event context
class EventContextManager:
    """Context manager for temporary event context."""
    
    def __init__(self, **context_kwargs):
        self.context_kwargs = context_kwargs
        self.original_context = None
        self.new_context = None
    
    async def __aenter__(self):
        self.original_context = event_context_store.get()
        self.new_context = EventContextGetter.create_child_context(**self.context_kwargs)
        event_context_store.set(self.new_context)
        return self.new_context
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        event_context_store.set(self.original_context)


# Dependency for FastAPI endpoints
async def get_event_context() -> Optional[EventContext]:
    """
    FastAPI dependency to get the current event context.
    
    Usage:
        @app.post("/endpoint")
        async def endpoint(context: EventContext = Depends(get_event_context)):
            # Use context in endpoint
    """
    return EventContextGetter.get_context()


# Decorator for automatic event emission in endpoints
def emit_event_on_success(event_type: EventType, extract_data: Optional[Callable] = None):
    """
    Decorator to emit an event when an endpoint succeeds.
    
    Args:
        event_type: Type of event to emit
        extract_data: Function to extract event data from endpoint result
    
    Usage:
        @app.post("/users")
        @emit_event_on_success(EventType.USER_REGISTERED, lambda result: {"user_id": result.id})
        async def create_user(user_data: UserCreate):
            return await user_service.create(user_data)
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                
                # Extract event data
                event_data = {}
                if extract_data:
                    event_data = extract_data(result)
                elif hasattr(result, 'dict'):
                    event_data = result.dict()
                
                # Add context information
                context = EventContextGetter.get_context()
                if context:
                    event_data.update({
                        "request_id": context.request_id,
                        "user_id": context.user_id,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                # Emit event
                event_bus = get_event_bus()
                await event_bus.publish(
                    event_type,
                    event_data,
                    correlation_id=EventContextGetter.get_correlation_id()
                )
                
                return result
                
            except Exception as e:
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator