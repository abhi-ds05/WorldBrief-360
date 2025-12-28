"""
Event registry for managing event type definitions, schemas, and metadata.
"""
from typing import Any, Dict, List, Optional, Set, Type, Union
from enum import Enum
import inspect
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from app.core.logging_config import logger
from app.events.event_types import EventType
from app.events.event_schemas import EventSchemaBase


@dataclass
class EventDefinition:
    """Definition and metadata for an event type."""
    event_type: EventType
    name: str
    description: str
    schema_class: Optional[Type[EventSchemaBase]] = None
    category: str = "general"
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    is_deprecated: bool = False
    deprecated_since: Optional[str] = None
    replacement_event_type: Optional[EventType] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_updated_at: datetime = field(default_factory=datetime.utcnow)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event definition to dictionary."""
        return {
            "event_type": self.event_type.value,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "tags": self.tags,
            "is_deprecated": self.is_deprecated,
            "deprecated_since": self.deprecated_since,
            "replacement_event_type": (
                self.replacement_event_type.value 
                if self.replacement_event_type else None
            ),
            "created_at": self.created_at.isoformat(),
            "last_updated_at": self.last_updated_at.isoformat(),
            "has_schema": self.schema_class is not None,
            "example_count": len(self.examples)
        }


@dataclass
class EventCategory:
    """Category grouping for events."""
    name: str
    description: str
    event_types: Set[EventType] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_event_type(self, event_type: EventType) -> None:
        """Add an event type to this category."""
        self.event_types.add(event_type)
    
    def remove_event_type(self, event_type: EventType) -> bool:
        """Remove an event type from this category."""
        if event_type in self.event_types:
            self.event_types.remove(event_type)
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert category to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "event_type_count": len(self.event_types),
            "event_types": [et.value for et in self.event_types],
            "created_at": self.created_at.isoformat()
        }


class EventRegistry:
    """
    Central registry for event type definitions and metadata.
    
    This registry provides:
    - Event type definitions with schemas
    - Event categorization
    - Version tracking
    - Deprecation management
    - Schema validation
    - Documentation generation
    """
    
    def __init__(self):
        self._definitions: Dict[EventType, EventDefinition] = {}
        self._categories: Dict[str, EventCategory] = {}
        self._schema_to_event_type: Dict[Type[EventSchemaBase], EventType] = {}
        
        # Auto-register categories
        self._init_default_categories()
    
    def _init_default_categories(self) -> None:
        """Initialize default event categories."""
        default_categories = {
            "user": EventCategory(
                name="user",
                description="User-related events (registration, login, profile updates)"
            ),
            "incident": EventCategory(
                name="incident",
                description="Incident reporting and verification events"
            ),
            "briefing": EventCategory(
                name="briefing",
                description="Briefing generation and delivery events"
            ),
            "chat": EventCategory(
                name="chat",
                description="AI chat and conversation events"
            ),
            "wallet": EventCategory(
                name="wallet",
                description="Wallet and transaction events"
            ),
            "notification": EventCategory(
                name="notification",
                description="Notification and messaging events"
            ),
            "analytics": EventCategory(
                name="analytics",
                description="Analytics and tracking events"
            ),
            "system": EventCategory(
                name="system",
                description="System and administrative events"
            ),
            "audit": EventCategory(
                name="audit",
                description="Audit and security events"
            ),
            "integration": EventCategory(
                name="integration",
                description="Third-party integration events"
            )
        }
        
        self._categories.update(default_categories)
        logger.debug(f"Initialized {len(self._categories)} default event categories")
    
    def register_event(
        self,
        event_type: EventType,
        name: str,
        description: str,
        schema_class: Optional[Type[EventSchemaBase]] = None,
        category: str = "general",
        version: str = "1.0.0",
        tags: Optional[List[str]] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        is_deprecated: bool = False,
        deprecated_since: Optional[str] = None,
        replacement_event_type: Optional[EventType] = None
    ) -> EventDefinition:
        """
        Register a new event type with the registry.
        
        Args:
            event_type: The event type enum value
            name: Human-readable name of the event
            description: Detailed description of the event
            schema_class: Optional Pydantic schema for event data validation
            category: Category for grouping events
            version: Version of the event definition
            tags: List of tags for categorization
            examples: Example event data payloads
            is_deprecated: Whether this event type is deprecated
            deprecated_since: Version since which the event is deprecated
            replacement_event_type: Replacement event type if deprecated
            
        Returns:
            EventDefinition object
            
        Raises:
            ValueError: If event type is already registered or invalid
        """
        if event_type in self._definitions:
            raise ValueError(f"Event type {event_type} is already registered")
        
        if schema_class and not issubclass(schema_class, EventSchemaBase):
            raise ValueError(
                f"Schema class must inherit from EventSchemaBase, got {schema_class}"
            )
        
        # Create event definition
        definition = EventDefinition(
            event_type=event_type,
            name=name,
            description=description,
            schema_class=schema_class,
            category=category,
            version=version,
            tags=tags or [],
            is_deprecated=is_deprecated,
            deprecated_since=deprecated_since,
            replacement_event_type=replacement_event_type,
            examples=examples or [],
            created_at=datetime.utcnow(),
            last_updated_at=datetime.utcnow()
        )
        
        # Store definition
        self._definitions[event_type] = definition
        
        # Update schema mapping
        if schema_class:
            self._schema_to_event_type[schema_class] = event_type
        
        # Add to category
        if category not in self._categories:
            # Create category if it doesn't exist
            self._categories[category] = EventCategory(
                name=category,
                description=f"Events related to {category}"
            )
        
        self._categories[category].add_event_type(event_type)
        
        logger.info(f"Registered event type: {event_type} ({name})")
        return definition
    
    def update_event(
        self,
        event_type: EventType,
        **updates
    ) -> EventDefinition:
        """
        Update an existing event definition.
        
        Args:
            event_type: Event type to update
            **updates: Fields to update
            
        Returns:
            Updated EventDefinition
            
        Raises:
            KeyError: If event type is not registered
        """
        if event_type not in self._definitions:
            raise KeyError(f"Event type {event_type} is not registered")
        
        definition = self._definitions[event_type]
        
        # Update allowed fields
        allowed_fields = {
            'name', 'description', 'version', 'tags', 'examples',
            'is_deprecated', 'deprecated_since', 'replacement_event_type'
        }
        
        for field_name, value in updates.items():
            if field_name in allowed_fields:
                setattr(definition, field_name, value)
            else:
                logger.warning(f"Cannot update field {field_name} on event definition")
        
        # Update timestamp
        definition.last_updated_at = datetime.utcnow()
        
        logger.debug(f"Updated event type: {event_type}")
        return definition
    
    def deprecate_event(
        self,
        event_type: EventType,
        deprecated_since: str,
        replacement_event_type: Optional[EventType] = None
    ) -> None:
        """
        Mark an event type as deprecated.
        
        Args:
            event_type: Event type to deprecate
            deprecated_since: Version since which the event is deprecated
            replacement_event_type: Optional replacement event type
        """
        if event_type not in self._definitions:
            raise KeyError(f"Event type {event_type} is not registered")
        
        self.update_event(
            event_type,
            is_deprecated=True,
            deprecated_since=deprecated_since,
            replacement_event_type=replacement_event_type
        )
        
        logger.warning(f"Deprecated event type: {event_type}")
    
    def unregister_event(self, event_type: EventType) -> bool:
        """
        Unregister an event type.
        
        Args:
            event_type: Event type to unregister
            
        Returns:
            True if event was unregistered, False if not found
        """
        if event_type not in self._definitions:
            return False
        
        definition = self._definitions[event_type]
        
        # Remove from category
        if definition.category in self._categories:
            self._categories[definition.category].remove_event_type(event_type)
        
        # Remove from schema mapping
        if definition.schema_class:
            self._schema_to_event_type.pop(definition.schema_class, None)
        
        # Remove definition
        del self._definitions[event_type]
        
        logger.info(f"Unregistered event type: {event_type}")
        return True
    
    def get_event_definition(self, event_type: EventType) -> Optional[EventDefinition]:
        """
        Get event definition for a given event type.
        
        Args:
            event_type: Event type to get definition for
            
        Returns:
            EventDefinition if found, None otherwise
        """
        return self._definitions.get(event_type)
    
    def get_event_by_schema(self, schema_class: Type[EventSchemaBase]) -> Optional[EventType]:
        """
        Get event type associated with a schema class.
        
        Args:
            schema_class: Schema class to look up
            
        Returns:
            EventType if found, None otherwise
        """
        return self._schema_to_event_type.get(schema_class)
    
    def get_all_definitions(self) -> List[EventDefinition]:
        """Get all registered event definitions."""
        return list(self._definitions.values())
    
    def get_definitions_by_category(self, category: str) -> List[EventDefinition]:
        """
        Get event definitions for a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of EventDefinition objects
        """
        if category not in self._categories:
            return []
        
        category_obj = self._categories[category]
        return [
            self._definitions[et] 
            for et in category_obj.event_types 
            if et in self._definitions
        ]
    
    def get_categories(self) -> List[EventCategory]:
        """Get all event categories."""
        return list(self._categories.values())
    
    def get_category(self, name: str) -> Optional[EventCategory]:
        """
        Get a specific category.
        
        Args:
            name: Category name
            
        Returns:
            EventCategory if found, None otherwise
        """
        return self._categories.get(name)
    
    def validate_event_data(self, event_type: EventType, data: Any) -> Optional[EventSchemaBase]:
        """
        Validate event data against its schema.
        
        Args:
            event_type: Event type to validate for
            data: Event data to validate
            
        Returns:
            Validated schema instance if validation succeeds, None otherwise
            
        Raises:
            ValueError: If schema validation fails
        """
        definition = self.get_event_definition(event_type)
        if not definition or not definition.schema_class:
            # No schema defined for this event, validation passes
            return None
        
        try:
            return definition.schema_class(**data)
        except Exception as e:
            raise ValueError(f"Event data validation failed for {event_type}: {e}")
    
    def generate_example_payload(self, event_type: EventType) -> Optional[Dict[str, Any]]:
        """
        Generate an example payload for an event type.
        
        Args:
            event_type: Event type to generate example for
            
        Returns:
            Example payload dictionary, or None if no examples available
        """
        definition = self.get_event_definition(event_type)
        if not definition or not definition.examples:
            return None
        
        # Return the first example
        return definition.examples[0]
    
    def search_events(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        deprecated: Optional[bool] = None
    ) -> List[EventDefinition]:
        """
        Search for events based on various criteria.
        
        Args:
            query: Search query string
            category: Filter by category
            tags: Filter by tags
            deprecated: Filter by deprecation status
            
        Returns:
            List of matching EventDefinition objects
        """
        results = []
        query_lower = query.lower()
        
        for definition in self._definitions.values():
            # Apply filters
            if category and definition.category != category:
                continue
            
            if tags and not any(tag in definition.tags for tag in tags):
                continue
            
            if deprecated is not None and definition.is_deprecated != deprecated:
                continue
            
            # Search in name, description, and tags
            if (query_lower in definition.name.lower() or
                query_lower in definition.description.lower() or
                any(query_lower in tag.lower() for tag in definition.tags)):
                results.append(definition)
        
        return results
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about registered events.
        
        Returns:
            Dictionary with event statistics
        """
        total_events = len(self._definitions)
        deprecated_events = sum(1 for d in self._definitions.values() if d.is_deprecated)
        events_with_schema = sum(1 for d in self._definitions.values() if d.schema_class)
        
        # Count by category
        category_counts = {}
        for definition in self._definitions.values():
            category_counts[definition.category] = category_counts.get(definition.category, 0) + 1
        
        # Tag statistics
        all_tags = []
        for definition in self._definitions.values():
            all_tags.extend(definition.tags)
        
        from collections import Counter
        tag_counts = Counter(all_tags)
        
        return {
            "total_events": total_events,
            "deprecated_events": deprecated_events,
            "events_with_schema": events_with_schema,
            "category_counts": category_counts,
            "total_categories": len(self._categories),
            "unique_tags": len(tag_counts),
            "top_tags": dict(tag_counts.most_common(10))
        }
    
    def export_documentation(self, format: str = "markdown") -> str:
        """
        Export event documentation in various formats.
        
        Args:
            format: Output format ("markdown", "json", "yaml")
            
        Returns:
            Documentation string in requested format
        """
        if format == "markdown":
            return self._export_markdown()
        elif format == "json":
            return self._export_json()
        elif format == "yaml":
            return self._export_yaml()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self) -> str:
        """Export documentation as Markdown."""
        lines = []
        
        # Header
        lines.append("# Event Documentation")
        lines.append("")
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append("")
        
        # Statistics
        stats = self.get_event_statistics()
        lines.append("## Statistics")
        lines.append("")
        lines.append(f"- Total Events: {stats['total_events']}")
        lines.append(f"- Deprecated Events: {stats['deprecated_events']}")
        lines.append(f"- Events with Schema: {stats['events_with_schema']}")
        lines.append(f"- Categories: {stats['total_categories']}")
        lines.append("")
        
        # By Category
        lines.append("## Events by Category")
        lines.append("")
        
        for category_name in sorted(self._categories.keys()):
            category = self._categories[category_name]
            definitions = self.get_definitions_by_category(category_name)
            
            if not definitions:
                continue
            
            lines.append(f"### {category.name.title()}")
            lines.append("")
            lines.append(f"{category.description}")
            lines.append("")
            
            for definition in sorted(definitions, key=lambda d: d.name):
                deprecation_note = " **DEPRECATED**" if definition.is_deprecated else ""
                lines.append(f"#### {definition.name}{deprecation_note}")
                lines.append("")
                lines.append(f"- **Event Type:** `{definition.event_type.value}`")
                lines.append(f"- **Description:** {definition.description}")
                lines.append(f"- **Version:** {definition.version}")
                
                if definition.tags:
                    tags_str = ", ".join(f"`{tag}`" for tag in definition.tags)
                    lines.append(f"- **Tags:** {tags_str}")
                
                if definition.is_deprecated:
                    lines.append(f"- **Deprecated Since:** {definition.deprecated_since}")
                    if definition.replacement_event_type:
                        lines.append(f"- **Replaced By:** `{definition.replacement_event_type.value}`")
                
                if definition.schema_class:
                    lines.append(f"- **Schema:** `{definition.schema_class.__name__}`")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def _export_json(self) -> str:
        """Export documentation as JSON."""
        import json
        
        export_data = {
            "generated_at": datetime.utcnow().isoformat(),
            "statistics": self.get_event_statistics(),
            "categories": {},
            "events": []
        }
        
        # Export categories
        for category_name, category in self._categories.items():
            export_data["categories"][category_name] = category.to_dict()
        
        # Export events
        for definition in self._definitions.values():
            event_data = definition.to_dict()
            export_data["events"].append(event_data)
        
        return json.dumps(export_data, indent=2)
    
    def _export_yaml(self) -> str:
        """Export documentation as YAML."""
        import yaml
        
        # Use safe dumper
        class SafeDumper(yaml.SafeDumper):
            pass
        
        # Add custom representers for datetime
        def datetime_representer(dumper, data):
            return dumper.represent_scalar('tag:yaml.org,2002:timestamp', data.isoformat())
        
        SafeDumper.add_representer(datetime, datetime_representer)
        
        export_data = {
            "generated_at": datetime.utcnow(),
            "statistics": self.get_event_statistics(),
            "categories": {name: cat.to_dict() for name, cat in self._categories.items()},
            "events": [defn.to_dict() for defn in self._definitions.values()]
        }
        
        return yaml.dump(export_data, Dumper=SafeDumper, default_flow_style=False)


# Global registry instance
_global_registry: Optional[EventRegistry] = None


def get_registry() -> EventRegistry:
    """
    Get the global event registry instance.
    
    Returns:
        EventRegistry singleton instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = EventRegistry()
    return _global_registry


def register_event(*args, **kwargs) -> EventDefinition:
    """
    Convenience function to register an event with the global registry.
    
    Returns:
        EventDefinition object
    """
    return get_registry().register_event(*args, **kwargs)


def validate_event_data(event_type: EventType, data: Any) -> Optional[EventSchemaBase]:
    """
    Convenience function to validate event data.
    
    Returns:
        Validated schema instance if validation succeeds, None otherwise
    """
    return get_registry().validate_event_data(event_type, data)


def generate_documentation(format: str = "markdown") -> str:
    """
    Generate event documentation.
    
    Args:
        format: Output format ("markdown", "json", "yaml")
        
    Returns:
        Documentation string
    """
    return get_registry().export_documentation(format)


# Auto-register core events
def _register_core_events():
    """Register core event types on module import."""
    try:
        from app.events.event_schemas import (
            UserRegisteredEvent, UserLoginEvent, IncidentReportedEvent,
            BriefingGeneratedEvent, ChatMessageSentEvent, WalletTransactionEvent
        )
        
        registry = get_registry()
        
        # User events
        registry.register_event(
            event_type=EventType.USER_REGISTERED,
            name="User Registered",
            description="Triggered when a new user registers for an account",
            schema_class=UserRegisteredEvent,
            category="user",
            tags=["authentication", "onboarding"],
            examples=[{
                "user_id": "12345",
                "email": "user@example.com",
                "username": "newuser",
                "registration_source": "web"
            }]
        )
        
        registry.register_event(
            event_type=EventType.USER_LOGIN,
            name="User Login",
            description="Triggered when a user logs into the system",
            schema_class=UserLoginEvent,
            category="user",
            tags=["authentication", "security"],
            examples=[{
                "user_id": "12345",
                "ip_address": "192.168.1.1",
                "user_agent": "Mozilla/5.0",
                "login_method": "password"
            }]
        )
        
        # Incident events
        registry.register_event(
            event_type=EventType.INCIDENT_REPORTED,
            name="Incident Reported",
            description="Triggered when a user reports a new incident",
            schema_class=IncidentReportedEvent,
            category="incident",
            tags=["reporting", "community"],
            examples=[{
                "incident_id": "inc_67890",
                "reporter_id": "12345",
                "location": {"lat": 40.7128, "lng": -74.0060},
                "category": "natural_disaster",
                "description": "Flood reported in downtown area",
                "severity": "high"
            }]
        )
        
        # Briefing events
        registry.register_event(
            event_type=EventType.BRIEFING_GENERATED,
            name="Briefing Generated",
            description="Triggered when a new briefing is generated",
            schema_class=BriefingGeneratedEvent,
            category="briefing",
            tags=["generation", "content"],
            examples=[{
                "briefing_id": "brief_78901",
                "user_id": "12345",
                "topic": "Climate Change",
                "level": "intermediate",
                "language": "en",
                "duration_seconds": 180
            }]
        )
        
        logger.info(f"Registered {len(registry.get_all_definitions())} core events")
        
    except ImportError as e:
        logger.warning(f"Could not auto-register core events: {e}")


# Auto-register on module import
_register_core_events()