"""
Event type definitions and base event models for the event-driven architecture.

This module defines all event types used in the WorldBrief 360 application
as well as base classes for events and event handling.
"""
from enum import Enum
from typing import Any, Dict, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class EventType(str, Enum):
    """
    Enumeration of all event types in the WorldBrief 360 application.
    
    Naming Convention:
    - Use lowercase with dots as separators
    - Format: <domain>.<action>.<subaction> (if needed)
    - Be descriptive but concise
    """
    
    # ============== User Events ==============
    USER_REGISTERED = "user.registered"
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_PROFILE_UPDATED = "user.profile_updated"
    USER_PREFERENCE_CHANGED = "user.preference_changed"
    USER_SUBSCRIPTION_CREATED = "user.subscription_created"
    USER_SUBSCRIPTION_UPDATED = "user.subscription_updated"
    USER_SUBSCRIPTION_CANCELLED = "user.subscription_cancelled"
    USER_EMAIL_VERIFIED = "user.email_verified"
    USER_PASSWORD_CHANGED = "user.password_changed"
    USER_PASSWORD_RESET_REQUESTED = "user.password_reset_requested"
    USER_ACCOUNT_DEACTIVATED = "user.account_deactivated"
    USER_ACCOUNT_DELETED = "user.account_deleted"
    
    # ============== Authentication & Authorization ==============
    AUTH_TOKEN_CREATED = "auth.token_created"
    AUTH_TOKEN_REVOKED = "auth.token_revoked"
    AUTH_ROLE_ASSIGNED = "auth.role_assigned"
    AUTH_ROLE_REVOKED = "auth.role_revoked"
    AUTH_PERMISSION_GRANTED = "auth.permission_granted"
    AUTH_PERMISSION_REVOKED = "auth.permission_revoked"
    AUTH_SESSION_STARTED = "auth.session_started"
    AUTH_SESSION_ENDED = "auth.session_ended"
    AUTH_FAILED_ATTEMPT = "auth.failed_attempt"
    
    # ============== Incident Events ==============
    INCIDENT_REPORTED = "incident.reported"
    INCIDENT_UPDATED = "incident.updated"
    INCIDENT_VERIFIED = "incident.verified"
    INCIDENT_DISPUTED = "incident.disputed"
    INCIDENT_RESOLVED = "incident.resolved"
    INCIDENT_ESCALATED = "incident.escalated"
    INCIDENT_ARCHIVED = "incident.archived"
    INCIDENT_COMMENT_ADDED = "incident.comment_added"
    INCIDENT_COMMENT_UPDATED = "incident.comment_updated"
    INCIDENT_COMMENT_DELETED = "incident.comment_deleted"
    INCIDENT_VOTE_ADDED = "incident.vote_added"
    INCIDENT_VOTE_REMOVED = "incident.vote_removed"
    INCIDENT_MEDIA_ADDED = "incident.media_added"
    INCIDENT_MEDIA_REMOVED = "incident.media_removed"
    INCIDENT_LOCATION_UPDATED = "incident.location_updated"
    INCIDENT_SEVERITY_CHANGED = "incident.severity_changed"
    INCIDENT_CATEGORY_CHANGED = "incident.category_changed"
    INCIDENT_ASSIGNED = "incident.assigned"
    INCIDENT_UNASSIGNED = "incident.unassigned"
    
    # ============== Briefing Events ==============
    BRIEFING_REQUESTED = "briefing.requested"
    BRIEFING_GENERATED = "briefing.generated"
    BRIEFING_DELIVERED = "briefing.delivered"
    BRIEFING_READ = "briefing.read"
    BRIEFING_SHARED = "briefing.shared"
    BRIEFING_SAVED = "briefing.saved"
    BRIEFING_DELETED = "briefing.deleted"
    BRIEFING_FEEDBACK_GIVEN = "briefing.feedback_given"
    BRIEFING_TRANSLATED = "briefing.translated"
    BRIEFING_AUDIO_GENERATED = "briefing.audio_generated"
    BRIEFING_IMAGE_GENERATED = "briefing.image_generated"
    BRIEFING_SUMMARY_GENERATED = "briefing.summary_generated"
    BRIEFING_CUSTOMIZED = "briefing.customized"
    BRIEFING_SCHEDULED = "briefing.scheduled"
    BRIEFING_SCHEDULE_CANCELLED = "briefing.schedule_cancelled"
    
    # ============== Topic Events ==============
    TOPIC_SUBSCRIBED = "topic.subscribed"
    TOPIC_UNSUBSCRIBED = "topic.unsubscribed"
    TOPIC_FOLLOWED = "topic.followed"
    TOPIC_UNFOLLOWED = "topic.unfollowed"
    TOPIC_TREND_DETECTED = "topic.trend_detected"
    TOPIC_ALERT_CREATED = "topic.alert_created"
    TOPIC_ALERT_TRIGGERED = "topic.alert_triggered"
    TOPIC_ALERT_RESOLVED = "topic.alert_resolved"
    TOPIC_ANALYSIS_COMPLETED = "topic.analysis_completed"
    TOPIC_SUMMARY_UPDATED = "topic.summary_updated"
    TOPIC_STATISTICS_UPDATED = "topic.statistics_updated"
    
    # ============== Chat & AI Events ==============
    CHAT_SESSION_STARTED = "chat.session_started"
    CHAT_SESSION_ENDED = "chat.session_ended"
    CHAT_MESSAGE_SENT = "chat.message_sent"
    CHAT_MESSAGE_RECEIVED = "chat.message_received"
    CHAT_MESSAGE_READ = "chat.message_read"
    CHAT_CONTEXT_UPDATED = "chat.context_updated"
    CHAT_HISTORY_CLEARED = "chat.history_cleared"
    CHAT_TOOL_CALLED = "chat.tool_called"
    CHAT_TOOL_RESPONSE = "chat.tool_response"
    CHAT_CITATION_ADDED = "chat.citation_added"
    CHAT_SUGGESTION_GENERATED = "chat.suggestion_generated"
    CHAT_SUGGESTION_ACCEPTED = "chat.suggestion_accepted"
    CHAT_SUGGESTION_REJECTED = "chat.suggestion_rejected"
    CHAT_TRANSLATION_REQUESTED = "chat.translation_requested"
    CHAT_TRANSLATION_COMPLETED = "chat.translation_completed"
    
    # ============== Wallet & Rewards Events ==============
    WALLET_CREATED = "wallet.created"
    WALLET_BALANCE_UPDATED = "wallet.balance_updated"
    WALLET_TRANSACTION = "wallet.transaction"
    WALLET_TRANSACTION_FAILED = "wallet.transaction_failed"
    WALLET_TRANSACTION_REVERSED = "wallet.transaction_reversed"
    WALLET_BONUS_AWARDED = "wallet.bonus_awarded"
    WALLET_PENALTY_APPLIED = "wallet.penalty_applied"
    WALLET_BALANCE_LOW = "wallet.balance_low"
    WALLET_BALANCE_HIGH = "wallet.balance_high"
    REWARD_EARNED = "reward.earned"
    REWARD_REDEEMED = "reward.redeemed"
    REWARD_EXPIRED = "reward.expired"
    REWARD_REVOKED = "reward.revoked"
    REWARD_TIER_UPGRADED = "reward.tier_upgraded"
    REWARD_TIER_DOWNGRADED = "reward.tier_downgraded"
    
    # ============== Notification Events ==============
    NOTIFICATION_SENT = "notification.sent"
    NOTIFICATION_DELIVERED = "notification.delivered"
    NOTIFICATION_READ = "notification.read"
    NOTIFICATION_CLICKED = "notification.clicked"
    NOTIFICATION_DISMISSED = "notification.dismissed"
    NOTIFICATION_PREFERENCE_UPDATED = "notification.preference_updated"
    NOTIFICATION_TEMPLATE_CREATED = "notification.template_created"
    NOTIFICATION_TEMPLATE_UPDATED = "notification.template_updated"
    NOTIFICATION_TEMPLATE_DELETED = "notification.template_deleted"
    NOTIFICATION_CHANNEL_ENABLED = "notification.channel_enabled"
    NOTIFICATION_CHANNEL_DISABLED = "notification.channel_disabled"
    
    # ============== Content & Media Events ==============
    CONTENT_CREATED = "content.created"
    CONTENT_UPDATED = "content.updated"
    CONTENT_DELETED = "content.deleted"
    CONTENT_PUBLISHED = "content.published"
    CONTENT_UNPUBLISHED = "content.unpublished"
    CONTENT_VIEWED = "content.viewed"
    CONTENT_LIKED = "content.liked"
    CONTENT_DISLIKED = "content.disliked"
    CONTENT_SHARED = "content.shared"
    CONTENT_BOOKMARKED = "content.bookmarked"
    CONTENT_REPORTED = "content.reported"
    CONTENT_MODERATED = "content.moderated"
    CONTENT_FLAGGED = "content.flagged"
    CONTENT_UNFLAGGED = "content.unflagged"
    MEDIA_UPLOADED = "media.uploaded"
    MEDIA_PROCESSED = "media.processed"
    MEDIA_DELETED = "media.deleted"
    MEDIA_TRANSCODED = "media.transcoded"
    MEDIA_THUMBNAIL_GENERATED = "media.thumbnail_generated"
    
    # ============== Analytics & Monitoring Events ==============
    ANALYTICS_EVENT_TRACKED = "analytics.event_tracked"
    ANALYTICS_SESSION_STARTED = "analytics.session_started"
    ANALYTICS_SESSION_ENDED = "analytics.session_ended"
    ANALYTICS_PAGE_VIEWED = "analytics.page_viewed"
    ANALYTICS_FEATURE_USED = "analytics.feature_used"
    ANALYTICS_CONVERSION_COMPLETED = "analytics.conversion_completed"
    ANALYTICS_ERROR_OCCURRED = "analytics.error_occurred"
    ANALYTICS_PERFORMANCE_MEASURED = "analytics.performance_measured"
    MONITORING_METRIC_COLLECTED = "monitoring.metric_collected"
    MONITORING_ALERT_TRIGGERED = "monitoring.alert_triggered"
    MONITORING_ALERT_RESOLVED = "monitoring.alert_resolved"
    MONITORING_HEALTH_CHECK_FAILED = "monitoring.health_check_failed"
    MONITORING_HEALTH_CHECK_PASSED = "monitoring.health_check_passed"
    
    # ============== System & Infrastructure Events ==============
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    SYSTEM_BACKUP_STARTED = "system.backup_started"
    SYSTEM_BACKUP_COMPLETED = "system.backup_completed"
    SYSTEM_BACKUP_FAILED = "system.backup_failed"
    SYSTEM_RESTORE_STARTED = "system.restore_started"
    SYSTEM_RESTORE_COMPLETED = "system.restore_completed"
    SYSTEM_RESTORE_FAILED = "system.restore_failed"
    SYSTEM_MAINTENANCE_STARTED = "system.maintenance_started"
    SYSTEM_MAINTENANCE_COMPLETED = "system.maintenance_completed"
    SYSTEM_CONFIG_UPDATED = "system.config_updated"
    SYSTEM_FEATURE_FLAG_CHANGED = "system.feature_flag_changed"
    SYSTEM_CACHE_CLEARED = "system.cache_cleared"
    SYSTEM_CACHE_INVALIDATED = "system.cache_invalidated"
    SYSTEM_QUEUE_OVERFLOW = "system.queue_overflow"
    SYSTEM_RESOURCE_LIMIT_EXCEEDED = "system.resource_limit_exceeded"
    
    # ============== Data Pipeline Events ==============
    PIPELINE_STARTED = "pipeline.started"
    PIPELINE_COMPLETED = "pipeline.completed"
    PIPELINE_FAILED = "pipeline.failed"
    PIPELINE_STEP_STARTED = "pipeline.step_started"
    PIPELINE_STEP_COMPLETED = "pipeline.step_completed"
    PIPELINE_STEP_FAILED = "pipeline.step_failed"
    DATA_INGESTION_STARTED = "data.ingestion_started"
    DATA_INGESTION_COMPLETED = "data.ingestion_completed"
    DATA_INGESTION_FAILED = "data.ingestion_failed"
    DATA_PROCESSING_STARTED = "data.processing_started"
    DATA_PROCESSING_COMPLETED = "data.processing_completed"
    DATA_PROCESSING_FAILED = "data.processing_failed"
    DATA_VALIDATION_STARTED = "data.validation_started"
    DATA_VALIDATION_COMPLETED = "data.validation_completed"
    DATA_VALIDATION_FAILED = "data.validation_failed"
    DATA_TRANSFORMATION_STARTED = "data.transformation_started"
    DATA_TRANSFORMATION_COMPLETED = "data.transformation_completed"
    DATA_TRANSFORMATION_FAILED = "data.transformation_failed"
    
    # ============== Machine Learning Events ==============
    ML_MODEL_TRAINED = "ml.model_trained"
    ML_MODEL_DEPLOYED = "ml.model_deployed"
    ML_MODEL_UNDEPLOYED = "ml.model_undeployed"
    ML_MODEL_INFERENCE = "ml.model_inference"
    ML_MODEL_EVALUATED = "ml.model_evaluated"
    ML_MODEL_MONITORED = "ml.model_monitored"
    ML_MODEL_DRIFT_DETECTED = "ml.model_drift_detected"
    ML_MODEL_RETRAINED = "ml.model_retrained"
    ML_DATASET_CREATED = "ml.dataset_created"
    ML_DATASET_UPDATED = "ml.dataset_updated"
    ML_DATASET_DELETED = "ml.dataset_deleted"
    ML_EXPERIMENT_STARTED = "ml.experiment_started"
    ML_EXPERIMENT_COMPLETED = "ml.experiment_completed"
    ML_EXPERIMENT_FAILED = "ml.experiment_failed"
    
    # ============== Integration & External Events ==============
    EXTERNAL_API_CALLED = "external_api.called"
    EXTERNAL_API_RESPONSE = "external_api.response"
    EXTERNAL_API_ERROR = "external_api.error"
    WEBHOOK_RECEIVED = "webhook.received"
    WEBHOOK_PROCESSED = "webhook.processed"
    WEBHOOK_FAILED = "webhook.failed"
    THIRD_PARTY_SYNC_STARTED = "third_party.sync_started"
    THIRD_PARTY_SYNC_COMPLETED = "third_party.sync_completed"
    THIRD_PARTY_SYNC_FAILED = "third_party.sync_failed"
    EMAIL_SENT = "email.sent"
    EMAIL_DELIVERED = "email.delivered"
    EMAIL_BOUNCED = "email.bounced"
    EMAIL_OPENED = "email.opened"
    EMAIL_CLICKED = "email.clicked"
    SMS_SENT = "sms.sent"
    SMS_DELIVERED = "sms.delivered"
    SMS_FAILED = "sms.failed"
    PUSH_NOTIFICATION_SENT = "push_notification.sent"
    PUSH_NOTIFICATION_DELIVERED = "push_notification.delivered"
    PUSH_NOTIFICATION_FAILED = "push_notification.failed"
    PUSH_NOTIFICATION_OPENED = "push_notification.opened"
    
    # ============== Audit & Security Events ==============
    AUDIT_LOG_CREATED = "audit.log_created"
    AUDIT_TRAIL_GENERATED = "audit.trail_generated"
    SECURITY_BREACH_DETECTED = "security.breach_detected"
    SECURITY_THREAT_DETECTED = "security.threat_detected"
    SECURITY_VULNERABILITY_FOUND = "security.vulnerability_found"
    SECURITY_PATCH_APPLIED = "security.patch_applied"
    SECURITY_SCAN_STARTED = "security.scan_started"
    SECURITY_SCAN_COMPLETED = "security.scan_completed"
    SECURITY_SCAN_FAILED = "security.scan_failed"
    COMPLIANCE_CHECK_PASSED = "compliance.check_passed"
    COMPLIANCE_CHECK_FAILED = "compliance.check_failed"
    PRIVACY_DATA_ACCESSED = "privacy.data_accessed"
    PRIVACY_DATA_MODIFIED = "privacy.data_modified"
    PRIVACY_DATA_DELETED = "privacy.data_deleted"
    PRIVACY_CONSENT_GIVEN = "privacy.consent_given"
    PRIVACY_CONSENT_REVOKED = "privacy.consent_revoked"
    
    # ============== Business & Workflow Events ==============
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_STEP_COMPLETED = "workflow.step_completed"
    WORKFLOW_STEP_FAILED = "workflow.step_failed"
    BUSINESS_RULE_TRIGGERED = "business.rule_triggered"
    BUSINESS_PROCESS_EXECUTED = "business.process_executed"
    BUSINESS_METRIC_UPDATED = "business.metric_updated"
    BUSINESS_KPI_ACHIEVED = "business.kpi_achieved"
    BUSINESS_KPI_MISSED = "business.kpi_missed"
    BUSINESS_ALERT_CREATED = "business.alert_created"
    BUSINESS_ALERT_RESOLVED = "business.alert_resolved"
    
    # ============== Community & Social Events ==============
    COMMUNITY_POST_CREATED = "community.post_created"
    COMMUNITY_POST_UPDATED = "community.post_updated"
    COMMUNITY_POST_DELETED = "community.post_deleted"
    COMMUNITY_COMMENT_ADDED = "community.comment_added"
    COMMUNITY_COMMENT_UPDATED = "community.comment_updated"
    COMMUNITY_COMMENT_DELETED = "community.comment_deleted"
    COMMUNITY_REACTION_ADDED = "community.reaction_added"
    COMMUNITY_REACTION_REMOVED = "community.reaction_removed"
    COMMUNITY_USER_FOLLOWED = "community.user_followed"
    COMMUNITY_USER_UNFOLLOWED = "community.user_unfollowed"
    COMMUNITY_REPUTATION_CHANGED = "community.reputation_changed"
    COMMUNITY_BADGE_EARNED = "community.badge_earned"
    COMMUNITY_BADGE_REVOKED = "community.badge_revoked"
    COMMUNITY_LEADERBOARD_UPDATED = "community.leaderboard_updated"
    
    # ============== Geography & Location Events ==============
    LOCATION_DETECTED = "location.detected"
    LOCATION_UPDATED = "location.updated"
    GEO_FENCE_ENTERED = "geo_fence.entered"
    GEO_FENCE_EXITED = "geo_fence.exited"
    REGION_SUBSCRIBED = "region.subscribed"
    REGION_UNSUBSCRIBED = "region.unsubscribed"
    REGION_ALERT_CREATED = "region.alert_created"
    REGION_ALERT_TRIGGERED = "region.alert_triggered"
    REGION_ALERT_RESOLVED = "region.alert_resolved"
    MAP_VIEW_CHANGED = "map.view_changed"
    MAP_MARKER_ADDED = "map.marker_added"
    MAP_MARKER_REMOVED = "map.marker_removed"
    MAP_MARKER_CLICKED = "map.marker_clicked"
    
    def get_category(self) -> str:
        """Get the category of this event type."""
        return self.value.split('.')[0]
    
    def get_action(self) -> str:
        """Get the action of this event type."""
        parts = self.value.split('.')
        return parts[1] if len(parts) > 1 else ""
    
    def get_subaction(self) -> Optional[str]:
        """Get the subaction of this event type if present."""
        parts = self.value.split('.')
        return parts[2] if len(parts) > 2 else None
    
    def is_user_event(self) -> bool:
        """Check if this is a user-related event."""
        return self.get_category() == "user"
    
    def is_system_event(self) -> bool:
        """Check if this is a system-related event."""
        return self.get_category() == "system"
    
    def is_security_event(self) -> bool:
        """Check if this is a security-related event."""
        return self.get_category() == "security"
    
    def __str__(self) -> str:
        return self.value


class EventPriority(str, Enum):
    """Priority levels for event processing."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class BaseEvent(BaseModel):
    """
    Base class for all events in the system.
    
    This provides the fundamental structure that all events must follow.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: EventPriority = Field(default=EventPriority.NORMAL)
    source: str = Field(default="system")
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            EventType: lambda v: v.value,
            EventPriority: lambda v: v.value,
        }
        extra = "forbid"
    
    def get_category(self) -> str:
        """Get the category of this event."""
        return self.type.get_category()
    
    def is_high_priority(self) -> bool:
        """Check if this event is high priority."""
        return self.priority in [EventPriority.HIGH, EventPriority.CRITICAL]
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the event."""
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata from the event."""
        return self.metadata.get(key, default)


class Event(BaseEvent):
    """
    Generic event class that can be used for any event type.
    
    This is the main event class used throughout the system.
    """
    data: Dict[str, Any] = Field(default_factory=dict)
    
    def get_data_field(self, field: str, default: Any = None) -> Any:
        """Get a field from the event data."""
        return self.data.get(field, default)
    
    def set_data_field(self, field: str, value: Any) -> None:
        """Set a field in the event data."""
        self.data[field] = value
    
    def validate_data_schema(self) -> bool:
        """
        Validate event data against its expected schema.
        
        This method should be implemented by subclasses or
        use the event registry for validation.
        """
        # Default implementation - always valid
        # In practice, this would use the event registry to validate
        return True


class DomainEvent(BaseEvent):
    """
    Base class for domain-specific events.
    
    Domain events represent business-level events that have
    meaning within the domain model.
    """
    aggregate_id: str
    aggregate_type: str
    version: int = Field(default=1)
    occurred_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_aggregate_identifier(self) -> str:
        """Get the full identifier for the aggregate."""
        return f"{self.aggregate_type}:{self.aggregate_id}"


class IntegrationEvent(BaseEvent):
    """
    Base class for integration events.
    
    Integration events are used for communication between
    different bounded contexts or external systems.
    """
    integration_type: str
    external_system_id: Optional[str] = None
    payload_format: str = Field(default="json")
    retry_count: int = Field(default=0)
    max_retries: int = Field(default=3)
    
    def should_retry(self) -> bool:
        """Check if this event should be retried."""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """Increment the retry count."""
        self.retry_count += 1


class EventContext(BaseModel):
    """
    Context information for event processing.
    
    This contains metadata about the current execution context
    that can be passed along with events.
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    tenant_id: Optional[str] = None
    environment: str = Field(default="production")
    additional_context: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return self.dict(exclude_none=True)
    
    def merge(self, other: 'EventContext') -> 'EventContext':
        """Merge with another context."""
        merged = self.dict(exclude_none=True)
        merged.update(other.dict(exclude_none=True))
        return EventContext(**merged)


def create_event(
    event_type: EventType,
    data: Dict[str, Any],
    correlation_id: Optional[str] = None,
    causation_id: Optional[str] = None,
    priority: EventPriority = EventPriority.NORMAL,
    source: str = "system",
    metadata: Optional[Dict[str, Any]] = None,
    context: Optional[EventContext] = None
) -> Event:
    """
    Factory function to create a new event.
    
    Args:
        event_type: Type of event to create
        data: Event data payload
        correlation_id: Correlation ID for tracing
        causation_id: ID of event that caused this event
        priority: Priority level
        source: Source of the event
        metadata: Additional metadata
        context: Event context information
        
    Returns:
        New Event instance
    """
    # Create base metadata
    event_metadata = metadata or {}
    
    # Add context to metadata if provided
    if context:
        event_metadata['context'] = context.to_dict()
    
    # Create the event
    event = Event(
        type=event_type,
        data=data,
        correlation_id=correlation_id,
        causation_id=causation_id,
        priority=priority,
        source=source,
        metadata=event_metadata
    )
    
    return event


def create_domain_event(
    event_type: EventType,
    aggregate_id: str,
    aggregate_type: str,
    data: Dict[str, Any],
    version: int = 1,
    **kwargs
) -> DomainEvent:
    """
    Factory function to create a new domain event.
    
    Args:
        event_type: Type of event to create
        aggregate_id: ID of the aggregate
        aggregate_type: Type of the aggregate
        data: Event data payload
        version: Event version
        **kwargs: Additional arguments passed to DomainEvent
        
    Returns:
        New DomainEvent instance
    """
    return DomainEvent(
        type=event_type,
        aggregate_id=aggregate_id,
        aggregate_type=aggregate_type,
        data=data,
        version=version,
        **kwargs
    )


def parse_event_type(event_type_str: str) -> Optional[EventType]:
    """
    Parse an event type string into an EventType enum.
    
    Args:
        event_type_str: String representation of event type
        
    Returns:
        EventType enum if valid, None otherwise
    """
    try:
        return EventType(event_type_str)
    except ValueError:
        return None


def get_event_type_by_category(category: str) -> List[EventType]:
    """
    Get all event types for a given category.
    
    Args:
        category: Event category to filter by
        
    Returns:
        List of EventType enums in the category
    """
    return [et for et in EventType if et.get_category() == category]


def get_event_type_by_action(action: str) -> List[EventType]:
    """
    Get all event types for a given action.
    
    Args:
        action: Event action to filter by
        
    Returns:
        List of EventType enums with the action
    """
    return [et for et in EventType if et.get_action() == action]


def is_event_type_deprecated(event_type: EventType) -> bool:
    """
    Check if an event type is deprecated.
    
    Note: This would typically check against a registry
    or configuration. This is a stub implementation.
    
    Args:
        event_type: Event type to check
        
    Returns:
        True if deprecated, False otherwise
    """
    # In a real implementation, this would check a registry
    # or configuration for deprecated event types
    deprecated_events = set()
    return event_type in deprecated_events


# Convenience lists for common event categories
USER_EVENTS = get_event_type_by_category("user")
SYSTEM_EVENTS = get_event_type_by_category("system")
SECURITY_EVENTS = get_event_type_by_category("security")
INCIDENT_EVENTS = get_event_type_by_category("incident")
BRIEFING_EVENTS = get_event_type_by_category("briefing")
CHAT_EVENTS = get_event_type_by_category("chat")
WALLET_EVENTS = get_event_type_by_category("wallet")
ANALYTICS_EVENTS = get_event_type_by_category("analytics")
INTEGRATION_EVENTS = get_event_type_by_category("external_api")
AUDIT_EVENTS = get_event_type_by_category("audit")