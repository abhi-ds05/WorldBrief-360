"""
Event handler registry for managing event subscriptions and handlers.
"""
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from app.core.logging_config import logger
from app.events.event_types import EventType, Event


@dataclass
class HandlerInfo:
    """Information about an event handler."""
    id: str
    function: Callable
    name: str
    module: str
    is_async: bool
    registered_at: datetime
    call_count: int = 0
    error_count: int = 0
    last_called_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventHandlerRegistry:
    """
    Registry for managing event handlers and their subscriptions.
    
    This class maintains a mapping between event types and their handlers,
    providing methods to register, unregister, and retrieve handlers.
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, Dict[str, HandlerInfo]] = defaultdict(dict)
        self._handler_to_events: Dict[str, Set[EventType]] = defaultdict(set)
        self._lock = None  # Will be initialized as asyncio.Lock when needed
    
    def _get_async_lock(self):
        """Lazy initialization of async lock."""
        import asyncio
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    def register(self, event_type: EventType, handler: Callable, **metadata) -> str:
        """
        Register a handler for an event type.
        
        Args:
            event_type: Event type to register handler for
            handler: Callable function (async or sync) to handle the event
            **metadata: Additional metadata to attach to the handler
            
        Returns:
            Handler ID for future reference
            
        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError(f"Handler must be callable, got {type(handler)}")
        
        # Generate unique handler ID
        handler_id = str(uuid.uuid4())
        
        # Get handler information
        handler_name = handler.__name__
        handler_module = handler.__module__
        is_async = inspect.iscoroutinefunction(handler)
        
        # Create handler info
        handler_info = HandlerInfo(
            id=handler_id,
            function=handler,
            name=handler_name,
            module=handler_module,
            is_async=is_async,
            registered_at=datetime.utcnow(),
            metadata=metadata
        )
        
        # Register handler
        self._handlers[event_type][handler_id] = handler_info
        self._handler_to_events[handler_id].add(event_type)
        
        logger.debug(
            f"Registered handler {handler_name} (ID: {handler_id}) "
            f"for event type {event_type}"
        )
        
        return handler_id
    
    def unregister(self, event_type: EventType, handler_id: str) -> bool:
        """
        Unregister a handler from an event type.
        
        Args:
            event_type: Event type to unregister from
            handler_id: ID of the handler to unregister
            
        Returns:
            True if handler was unregistered, False if not found
        """
        if event_type not in self._handlers or handler_id not in self._handlers[event_type]:
            return False
        
        # Remove handler from event type
        del self._handlers[event_type][handler_id]
        
        # Clean up empty event types
        if not self._handlers[event_type]:
            del self._handlers[event_type]
        
        # Remove event type from handler mapping
        if handler_id in self._handler_to_events:
            self._handler_to_events[handler_id].discard(event_type)
            # Clean up empty handler mappings
            if not self._handler_to_events[handler_id]:
                del self._handler_to_events[handler_id]
        
        logger.debug(f"Unregistered handler {handler_id} from event type {event_type}")
        return True
    
    def unregister_all(self, handler_id: str) -> bool:
        """
        Unregister a handler from all event types.
        
        Args:
            handler_id: ID of the handler to unregister
            
        Returns:
            True if handler was unregistered from any events, False if not found
        """
        if handler_id not in self._handler_to_events:
            return False
        
        # Get all event types this handler is registered for
        event_types = self._handler_to_events[handler_id].copy()
        
        # Unregister from each event type
        success = False
        for event_type in event_types:
            if self.unregister(event_type, handler_id):
                success = True
        
        return success
    
    def get_handlers(self, event_type: EventType) -> List[Tuple[str, Callable]]:
        """
        Get all handlers for an event type.
        
        Args:
            event_type: Event type to get handlers for
            
        Returns:
            List of (handler_id, handler_function) tuples
        """
        handlers = self._handlers.get(event_type, {})
        return [(handler_id, handler_info.function) 
                for handler_id, handler_info in handlers.items()]
    
    def get_handler_info(self, event_type: EventType, handler_id: str) -> Optional[HandlerInfo]:
        """
        Get information about a specific handler.
        
        Args:
            event_type: Event type the handler is registered for
            handler_id: ID of the handler
            
        Returns:
            HandlerInfo object if found, None otherwise
        """
        return self._handlers.get(event_type, {}).get(handler_id)
    
    def get_all_handlers_info(self) -> Dict[EventType, List[HandlerInfo]]:
        """
        Get information about all registered handlers.
        
        Returns:
            Dictionary mapping event types to lists of HandlerInfo objects
        """
        result = {}
        for event_type, handlers in self._handlers.items():
            result[event_type] = list(handlers.values())
        return result
    
    def get_handler_ids_for_event(self, event_type: EventType) -> List[str]:
        """
        Get all handler IDs for an event type.
        
        Args:
            event_type: Event type to get handler IDs for
            
        Returns:
            List of handler IDs
        """
        return list(self._handlers.get(event_type, {}).keys())
    
    def get_event_types_for_handler(self, handler_id: str) -> List[EventType]:
        """
        Get all event types a handler is registered for.
        
        Args:
            handler_id: ID of the handler
            
        Returns:
            List of event types
        """
        return list(self._handler_to_events.get(handler_id, set()))
    
    def has_handlers(self, event_type: EventType) -> bool:
        """
        Check if any handlers are registered for an event type.
        
        Args:
            event_type: Event type to check
            
        Returns:
            True if handlers exist, False otherwise
        """
        return bool(self._handlers.get(event_type))
    
    def count_handlers(self, event_type: Optional[EventType] = None) -> int:
        """
        Count handlers for an event type or total handlers.
        
        Args:
            event_type: Optional event type to count handlers for.
                       If None, counts all handlers.
            
        Returns:
            Number of handlers
        """
        if event_type is None:
            # Count all unique handlers across all event types
            return len(self._handler_to_events)
        else:
            return len(self._handlers.get(event_type, {}))
    
    async def record_handler_call(self, event_type: EventType, handler_id: str, 
                                 success: bool = True, error: Optional[Exception] = None):
        """
        Record a handler call for metrics and monitoring.
        
        Args:
            event_type: Event type that was handled
            handler_id: ID of the handler that was called
            success: Whether the handler call was successful
            error: Exception if the handler failed
        """
        import asyncio
        
        lock = self._get_async_lock()
        async with lock:
            handler_info = self.get_handler_info(event_type, handler_id)
            if not handler_info:
                logger.warning(f"Cannot record call for unknown handler {handler_id}")
                return
            
            handler_info.call_count += 1
            handler_info.last_called_at = datetime.utcnow()
            
            if not success:
                handler_info.error_count += 1
                handler_info.last_error_at = datetime.utcnow()
                
                # Store last error in metadata
                if error:
                    handler_info.metadata['last_error'] = str(error)
                    handler_info.metadata['last_error_type'] = type(error).__name__
    
    def get_handler_metrics(self, event_type: Optional[EventType] = None) -> Dict[str, Any]:
        """
        Get metrics for handlers.
        
        Args:
            event_type: Optional event type to get metrics for.
                       If None, returns metrics for all handlers.
            
        Returns:
            Dictionary with handler metrics
        """
        metrics = {
            'total_handlers': 0,
            'total_calls': 0,
            'total_errors': 0,
            'error_rate': 0.0,
            'by_event_type': {},
            'top_handlers': []
        }
        
        if event_type:
            # Get metrics for specific event type
            handlers = self._handlers.get(event_type, {})
            event_metrics = {
                'handler_count': len(handlers),
                'total_calls': sum(h.call_count for h in handlers.values()),
                'total_errors': sum(h.error_count for h in handlers.values()),
                'handlers': []
            }
            
            if event_metrics['total_calls'] > 0:
                event_metrics['error_rate'] = (
                    event_metrics['total_errors'] / event_metrics['total_calls']
                )
            
            for handler_info in handlers.values():
                handler_metrics = {
                    'id': handler_info.id,
                    'name': handler_info.name,
                    'call_count': handler_info.call_count,
                    'error_count': handler_info.error_count,
                    'last_called': handler_info.last_called_at,
                    'registered_at': handler_info.registered_at
                }
                event_metrics['handlers'].append(handler_metrics)
            
            return event_metrics
        
        else:
            # Get metrics for all handlers
            all_handlers = []
            for event_type, handlers in self._handlers.items():
                event_metrics = {
                    'event_type': event_type.value,
                    'handler_count': len(handlers),
                    'total_calls': sum(h.call_count for h in handlers.values()),
                    'total_errors': sum(h.error_count for h in handlers.values()),
                }
                
                if event_metrics['total_calls'] > 0:
                    event_metrics['error_rate'] = (
                        event_metrics['total_errors'] / event_metrics['total_calls']
                    )
                
                metrics['by_event_type'][event_type.value] = event_metrics
                metrics['total_handlers'] += len(handlers)
                metrics['total_calls'] += event_metrics['total_calls']
                metrics['total_errors'] += event_metrics['total_errors']
                
                # Collect all handlers for top handlers list
                all_handlers.extend(handlers.values())
            
            # Calculate overall error rate
            if metrics['total_calls'] > 0:
                metrics['error_rate'] = metrics['total_errors'] / metrics['total_calls']
            
            # Get top 10 most called handlers
            top_handlers = sorted(
                all_handlers,
                key=lambda h: h.call_count,
                reverse=True
            )[:10]
            
            metrics['top_handlers'] = [
                {
                    'id': h.id,
                    'name': h.name,
                    'event_types': [et.value for et in self.get_event_types_for_handler(h.id)],
                    'call_count': h.call_count,
                    'error_count': h.error_count,
                    'error_rate': h.error_count / h.call_count if h.call_count > 0 else 0
                }
                for h in top_handlers
            ]
            
            return metrics
    
    def clear(self) -> None:
        """
        Clear all registered handlers.
        
        Warning: This will remove all event handlers from the registry.
        """
        self._handlers.clear()
        self._handler_to_events.clear()
        logger.info("Cleared all event handlers from registry")
    
    def get_handler_by_name(self, handler_name: str, 
                           event_type: Optional[EventType] = None) -> List[HandlerInfo]:
        """
        Find handlers by name.
        
        Args:
            handler_name: Name of the handler function to find
            event_type: Optional event type to filter by
            
        Returns:
            List of HandlerInfo objects matching the name
        """
        results = []
        
        if event_type:
            handlers = self._handlers.get(event_type, {})
            for handler_info in handlers.values():
                if handler_info.name == handler_name:
                    results.append(handler_info)
        else:
            for handlers in self._handlers.values():
                for handler_info in handlers.values():
                    if handler_info.name == handler_name:
                        results.append(handler_info)
        
        return results
    
    def get_handlers_by_module(self, module_name: str) -> List[HandlerInfo]:
        """
        Find all handlers from a specific module.
        
        Args:
            module_name: Name of the module to find handlers from
            
        Returns:
            List of HandlerInfo objects from the module
        """
        results = []
        
        for handlers in self._handlers.values():
            for handler_info in handlers.values():
                if handler_info.module == module_name:
                    results.append(handler_info)
        
        return results
    
    def export_registry(self) -> Dict[str, Any]:
        """
        Export the current registry state for debugging or persistence.
        
        Returns:
            Dictionary representation of the registry
        """
        export_data = {
            'exported_at': datetime.utcnow().isoformat(),
            'total_event_types': len(self._handlers),
            'total_handlers': self.count_handlers(),
            'event_types': {}
        }
        
        for event_type, handlers in self._handlers.items():
            event_data = {
                'handler_count': len(handlers),
                'handlers': []
            }
            
            for handler_info in handlers.values():
                handler_data = {
                    'id': handler_info.id,
                    'name': handler_info.name,
                    'module': handler_info.module,
                    'is_async': handler_info.is_async,
                    'registered_at': handler_info.registered_at.isoformat(),
                    'call_count': handler_info.call_count,
                    'error_count': handler_info.error_count,
                    'last_called_at': (
                        handler_info.last_called_at.isoformat() 
                        if handler_info.last_called_at else None
                    ),
                    'metadata': handler_info.metadata
                }
                event_data['handlers'].append(handler_data)
            
            export_data['event_types'][event_type.value] = event_data
        
        return export_data
    
    def import_registry(self, data: Dict[str, Any]) -> None:
        """
        Import registry state from previously exported data.
        
        Warning: This will replace the current registry state.
        
        Args:
            data: Dictionary containing registry export data
        """
        import warnings
        warnings.warn(
            "Registry import does not restore handler functions, "
            "only metadata and statistics. Handlers must be re-registered.",
            UserWarning
        )
        
        # Clear current registry
        self.clear()
        
        # Import metadata (note: handlers must be re-registered)
        for event_type_str, event_data in data.get('event_types', {}).items():
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                logger.warning(f"Skipping unknown event type in import: {event_type_str}")
                continue
            
            # Note: We can't restore the actual handler functions from export
            # This import is mainly for statistics and metadata
            for handler_data in event_data.get('handlers', []):
                # We can store the metadata but handlers need to be re-registered
                logger.debug(
                    f"Imported metadata for handler {handler_data['name']} "
                    f"(event: {event_type})"
                )


# Global registry instance
_registry_instance: Optional[EventHandlerRegistry] = None


def get_registry() -> EventHandlerRegistry:
    """
    Get the global event handler registry instance.
    
    Returns:
        EventHandlerRegistry singleton instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = EventHandlerRegistry()
    return _registry_instance


def clear_registry() -> None:
    """
    Clear the global event handler registry.
    
    Warning: This will remove all registered event handlers.
    """
    global _registry_instance
    if _registry_instance:
        _registry_instance.clear()


def get_handler_metrics(event_type: Optional[EventType] = None) -> Dict[str, Any]:
    """
    Convenience function to get handler metrics.
    
    Args:
        event_type: Optional event type to filter metrics
        
    Returns:
        Dictionary with handler metrics
    """
    return get_registry().get_handler_metrics(event_type)