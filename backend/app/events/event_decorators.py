"""
Event decorators for easy event handler registration and management.
"""
import functools
import inspect
import asyncio
from typing import Any, Callable, Dict, List, Optional, Type, Union 
from datetime import datetime

from app.core.logging_config import logger
from app.events.event_bus import get_event_bus, publish_event, publish_event_nowait
from app.events.event_types import EventType, Event, BaseEvent
from app.events.event_schemas import EventSchemaBase


def event_handler(event_type: Union[EventType, List[EventType]]):
    """
    Decorator to register a function as an event handler.
    
    Args:
        event_type: Single event type or list of event types to handle
        
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        async def send_welcome_email(event_data: dict, event: Event):
            await email_service.send_welcome(event_data["user_id"])
        
        @event_handler([EventType.USER_LOGIN, EventType.USER_REGISTERED])
        async def log_user_activity(event_data: dict, event: Event):
            await analytics.track(event.type, event_data["user_id"])
        ```
    """
    def decorator(func: Callable):
        # Convert single event type to list
        if isinstance(event_type, EventType):
            event_types = [event_type]
        else:
            event_types = event_type
        
        # Get the event bus instance
        bus = get_event_bus()
        
        # Register handler for each event type
        for et in event_types:
            handler_id = bus.subscribe(et, func)
            
            # Store handler ID on the function for potential unsubscription
            if not hasattr(func, '_event_handler_ids'):
                func._event_handler_ids = {}
            func._event_handler_ids[et] = handler_id
        
        # Add metadata to the function
        if not hasattr(func, '_is_event_handler'):
            func._is_event_handler = True
        if not hasattr(func, '_event_types'):
            func._event_types = event_types
        
        # Update docstring to show event registration
        original_doc = func.__doc__ or ""
        event_list = ", ".join([et.value for et in event_types])
        func.__doc__ = f"""
{original_doc}
        
Event Handler Information:
- Registered for: {event_list}
- Handler ID(s): {func._event_handler_ids}
"""
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Wrapper for async handlers."""
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Wrapper for sync handlers."""
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_event_schema(schema_class: Type[EventSchemaBase]):
    """
    Decorator to validate event data against a Pydantic schema.
    
    Args:
        schema_class: Pydantic schema class to validate against
        
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        @validate_event_schema(UserRegisteredSchema)
        async def send_welcome_email(event_data: UserRegisteredSchema, event: Event):
            await email_service.send_welcome(event_data.user_id)
        ```
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(event_data: Any, event: Event):
            try:
                # Validate and parse event data
                validated_data = schema_class(**event_data)
                # Call original function with validated data
                return await func(validated_data, event)
            except Exception as e:
                logger.error(f"Event validation failed for {event.id}: {e}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(event_data: Any, event: Event):
            try:
                # Validate and parse event data
                validated_data = schema_class(**event_data)
                # Call original function with validated data
                return func(validated_data, event)
            except Exception as e:
                logger.error(f"Event validation failed for {event.id}: {e}")
                raise
        
        # Mark function as schema-validated
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._validated_schema = schema_class
        
        return wrapper
    
    return decorator


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry event handlers on failure with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for exponential backoff
        
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        @retry_on_failure(max_retries=3, delay=2.0, backoff=2.0)
        async def send_welcome_email(event_data: dict, event: Event):
            await email_service.send_welcome(event_data["user_id"])
        ```
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):  # +1 for the initial attempt
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Handler {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {wait_time:.1f}s. Error: {e}"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"Handler {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {last_exception}"
                        )
                        raise last_exception
            
            # This should never be reached, but just in case
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Handler {func.__name__} failed on attempt {attempt + 1}/{max_retries + 1}. "
                            f"Retrying in {wait_time:.1f}s. Error: {e}"
                        )
                        import time
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"Handler {func.__name__} failed after {max_retries + 1} attempts. "
                            f"Last error: {last_exception}"
                        )
                        raise last_exception
            
            raise last_exception
        
        # Mark function as retry-enabled
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._max_retries = max_retries
        wrapper._retry_delay = delay
        wrapper._retry_backoff = backoff
        
        return wrapper
    
    return decorator


def timeout_handler(timeout_seconds: float = 30.0):
    """
    Decorator to add timeout to event handlers.
    
    Args:
        timeout_seconds: Maximum execution time in seconds
        
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        @timeout_handler(timeout_seconds=10.0)
        async def send_welcome_email(event_data: dict, event: Event):
            await email_service.send_welcome(event_data["user_id"])
        ```
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Handler {func.__name__} timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"Handler execution exceeded {timeout_seconds} seconds")
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, we can't easily add timeout without threads
            # So we'll just log a warning and execute normally
            logger.warning(
                f"Timeout decorator used on sync function {func.__name__}. "
                f"Timeout of {timeout_seconds}s will not be enforced."
            )
            return func(*args, **kwargs)
        
        # Mark function as timeout-protected
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._timeout_seconds = timeout_seconds
        
        return wrapper
    
    return decorator


def transactional_event_handler(session_provider: Callable):
    """
    Decorator to wrap event handlers in a database transaction.
    
    Args:
        session_provider: Function that returns a database session
        
    Example:
        ```python
        from app.db.session import get_db
        
        @event_handler(EventType.USER_REGISTERED)
        @transactional_event_handler(get_db)
        async def create_user_profile(event_data: dict, event: Event, db: Session):
            # db session is automatically provided and committed/rolled back
            profile = UserProfile(user_id=event_data["user_id"])
            db.add(profile)
            # No need to call db.commit() - it's done automatically
        ```
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(event_data: Any, event: Event):
            import asyncio
            from contextlib import asynccontextmanager
            
            # Get database session
            if inspect.iscoroutinefunction(session_provider):
                db = await session_provider()
            else:
                db = session_provider()
            
            try:
                # Call handler with database session
                result = await func(event_data, event, db)
                
                # Commit transaction
                if hasattr(db, 'commit'):
                    if inspect.iscoroutinefunction(db.commit):
                        await db.commit()
                    else:
                        db.commit()
                
                return result
                
            except Exception as e:
                # Rollback on error
                if hasattr(db, 'rollback'):
                    if inspect.iscoroutinefunction(db.rollback):
                        await db.rollback()
                    else:
                        db.rollback()
                logger.error(f"Transaction failed in handler {func.__name__}: {e}")
                raise
            finally:
                # Close session
                if hasattr(db, 'close'):
                    if inspect.iscoroutinefunction(db.close):
                        await db.close()
                    else:
                        db.close()
        
        @functools.wraps(func)
        def sync_wrapper(event_data: Any, event: Event):
            # Get database session
            db = session_provider()
            
            try:
                # Call handler with database session
                result = func(event_data, event, db)
                
                # Commit transaction
                if hasattr(db, 'commit'):
                    db.commit()
                
                return result
                
            except Exception as e:
                # Rollback on error
                if hasattr(db, 'rollback'):
                    db.rollback()
                logger.error(f"Transaction failed in handler {func.__name__}: {e}")
                raise
            finally:
                # Close session
                if hasattr(db, 'close'):
                    db.close()
        
        # Mark function as transactional
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._transactional = True
        wrapper._session_provider = session_provider
        
        return wrapper
    
    return decorator


def event_emitter(result_event_type: Optional[EventType] = None):
    """
    Decorator that emits an event with the handler's result.
    
    Args:
        result_event_type: Event type to emit with the result.
                          If None, emits an event based on handler name.
    
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        @event_emitter(result_event_type=EventType.WELCOME_EMAIL_SENT)
        async def send_welcome_email(event_data: dict, event: Event):
            result = await email_service.send_welcome(event_data["user_id"])
            return {"user_id": event_data["user_id"], "email_sent": result}
        ```
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(event_data: Any, event: Event):
            # Execute handler
            result = await func(event_data, event)
            
            # Emit result event
            if result_event_type:
                emit_event_type = result_event_type
            else:
                # Generate event type from handler name
                handler_name = func.__name__
                emit_event_type = EventType(f"handler.{handler_name}.completed")
            
            # Emit event with result
            await publish_event(
                emit_event_type,
                {
                    "original_event_id": event.id,
                    "original_event_type": event.type,
                    "handler": func.__name__,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=event.correlation_id
            )
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(event_data: Any, event: Event):
            # Execute handler
            result = func(event_data, event)
            
            # Emit result event
            if result_event_type:
                emit_event_type = result_event_type
            else:
                handler_name = func.__name__
                emit_event_type = EventType(f"handler.{handler_name}.completed")
            
            # Emit event with result (non-blocking)
            publish_event_nowait(
                emit_event_type,
                {
                    "original_event_id": event.id,
                    "original_event_type": event.type,
                    "handler": func.__name__,
                    "result": result,
                    "timestamp": datetime.utcnow().isoformat()
                },
                correlation_id=event.correlation_id
            )
            
            return result
        
        # Mark function as event-emitting
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._emits_event = True
        wrapper._result_event_type = result_event_type
        
        return wrapper
    
    return decorator


def rate_limited(max_calls: int = 10, period: float = 60.0):
    """
    Decorator to rate limit event handlers.
    
    Args:
        max_calls: Maximum number of calls allowed in the period
        period: Time period in seconds
        
    Example:
        ```python
        @event_handler(EventType.USER_REGISTERED)
        @rate_limited(max_calls=5, period=60.0)
        async def send_welcome_email(event_data: dict, event: Event):
            await email_service.send_welcome(event_data["user_id"])
        ```
    """
    def decorator(func: Callable):
        import time
        from collections import deque
        
        # Store call timestamps
        calls = deque(maxlen=max_calls)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old calls
            while calls and calls[0] <= current_time - period:
                calls.popleft()
            
            # Check rate limit
            if len(calls) >= max_calls:
                wait_time = period - (current_time - calls[0])
                logger.warning(
                    f"Handler {func.__name__} rate limited. "
                    f"Waiting {wait_time:.1f}s before next execution."
                )
                await asyncio.sleep(wait_time)
                # After waiting, remove old calls again
                current_time = time.time()
                while calls and calls[0] <= current_time - period:
                    calls.popleft()
            
            # Record call and execute
            calls.append(current_time)
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            current_time = time.time()
            
            # Remove old calls
            while calls and calls[0] <= current_time - period:
                calls.popleft()
            
            # Check rate limit
            if len(calls) >= max_calls:
                wait_time = period - (current_time - calls[0])
                logger.warning(
                    f"Handler {func.__name__} rate limited. "
                    f"Waiting {wait_time:.1f}s before next execution."
                )
                time.sleep(wait_time)
                # After waiting, remove old calls again
                current_time = time.time()
                while calls and calls[0] <= current_time - period:
                    calls.popleft()
            
            # Record call and execute
            calls.append(current_time)
            return func(*args, **kwargs)
        
        # Mark function as rate-limited
        wrapper = async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
        wrapper._rate_limited = True
        wrapper._max_calls = max_calls
        wrapper._period = period
        
        return wrapper
    
    return decorator


# Helper function to check if a function is decorated as an event handler
def is_event_handler(func: Callable) -> bool:
    """Check if a function is decorated as an event handler."""
    return getattr(func, '_is_event_handler', False)


# Helper function to get event types a handler is registered for
def get_handler_event_types(func: Callable) -> List[EventType]:
    """Get the event types a handler is registered for."""
    return getattr(func, '_event_types', [])


# Helper function to unregister an event handler
def unregister_event_handler(func: Callable) -> bool:
    """
    Unregister an event handler from all event types.
    
    Returns:
        True if successfully unregistered, False otherwise
    """
    if not is_event_handler(func):
        return False
    
    bus = get_event_bus()
    handler_ids = getattr(func, '_event_handler_ids', {})
    
    success = True
    for event_type, handler_id in handler_ids.items():
        if not bus.unsubscribe(event_type, handler_id):
            success = False
    
    return success