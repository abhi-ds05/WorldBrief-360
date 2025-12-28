"""
Event-driven architecture module for WorldBrief 360.

This module provides an event bus system for decoupled communication
between different parts of the application. Events are published and
subscribed to asynchronously, allowing for extensible and maintainable
code.

Example:
    ```python
    from app.events import EventBus, EventType
    
    # Publish an event
    await EventBus.publish(
        EventType.USER_REGISTERED,
        {"user_id": 123, "email": "user@example.com"}
    )
    
    # Subscribe to events
    @EventBus.subscribe(EventType.USER_REGISTERED)
    async def send_welcome_email(event_data: dict):
        await email_service.send_welcome(event_data["user_id"])
    ```
"""

from app.events.event_bus import EventBus, get_event_bus
from app.events.event_types import EventType, Event, BaseEvent
from app.events.event_schemas import (
    UserRegisteredEvent,
    UserLoginEvent,
    IncidentReportedEvent,
    BriefingGeneratedEvent,
    ChatMessageSentEvent,
    WalletTransactionEvent,
    ContentModeratedEvent,
    SystemAlertEvent
)
from app.events.event_handlers import EventHandlerRegistry
from app.events.event_decorators import event_handler
from app.events.middleware import EventMiddleware

# Import subscribers to ensure they're registered
from app.events.subscribers import (
    notification_subscriber,
    analytics_subscriber,
    moderation_subscriber,
    audit_subscriber,
    background_task_subscriber
)

__all__ = [
    # Core components
    "EventBus",
    "get_event_bus",
    
    # Event types and schemas
    "EventType",
    "Event",
    "BaseEvent",
    
    # Event data schemas
    "UserRegisteredEvent",
    "UserLoginEvent",
    "IncidentReportedEvent",
    "BriefingGeneratedEvent",
    "ChatMessageSentEvent",
    "WalletTransactionEvent",
    "ContentModeratedEvent",
    "SystemAlertEvent",
    
    # Registry and utilities
    "EventHandlerRegistry",
    "event_handler",
    
    # Middleware
    "EventMiddleware",
    
    # Subscriber modules
    "notification_subscriber",
    "analytics_subscriber",
    "moderation_subscriber",
    "audit_subscriber",
    "background_task_subscriber",
]

# Initialize event system when module is imported
async def init_event_system() -> None:
    """
    Initialize the event system on application startup.
    This should be called from the main application startup event.
    """
    from app.core.logging_config import logger
    try:
        # Initialize event bus
        await EventBus.initialize()
        
        # Register all subscribers
        from app.events import subscribers
        
        logger.info("Event system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize event system: {e}")
        raise

async def shutdown_event_system() -> None:
    """
    Cleanup event system on application shutdown.
    This should be called from the main application shutdown event.
    """
    from app.core.logging_config import logger
    try:
        await EventBus.shutdown()
        logger.info("Event system shut down successfully")
    except Exception as e:
        logger.error(f"Error shutting down event system: {e}")

# Version info
__version__ = "1.0.0"
__author__ = "WorldBrief 360 Team"
__description__ = "Event-driven architecture module for decoupled communication"