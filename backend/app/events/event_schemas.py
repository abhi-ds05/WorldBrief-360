"""
Pydantic schemas for event data validation and serialization.

These schemas define the structure of event data payloads for different event types.
All event schemas should inherit from EventSchemaBase.
"""
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import uuid

from app.core.logging_config import logger


class EventSchemaBase(BaseModel):
    """
    Base schema for all event data.
    
    All event schemas should inherit from this class to ensure consistency.
    """
    class Config:
        extra = "forbid"  # Reject extra fields not defined in schema
        validate_assignment = True
        anystr_strip_whitespace = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            uuid.UUID: lambda v: str(v),
        }
    
    @validator('*', pre=True)
    def empty_str_to_none(cls, v):
        """Convert empty strings to None for all fields."""
        if v == "":
            return None
        return v


# ============== User Events ==============

class UserRegisteredEvent(EventSchemaBase):
    """Schema for user registration events."""
    user_id: str = Field(..., description="Unique identifier for the user")
    email: str = Field(..., description="User's email address")
    username: Optional[str] = Field(None, description="User's chosen username")
    registration_source: str = Field(
        default="web",
        description="Source of registration (web, mobile, api, etc.)"
    )
    referrer_id: Optional[str] = Field(None, description="ID of referring user if any")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional registration metadata"
    )
    
    @validator('email')
    def validate_email(cls, v):
        """Basic email validation."""
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class UserLoginEvent(EventSchemaBase):
    """Schema for user login events."""
    user_id: str = Field(..., description="Unique identifier for the user")
    ip_address: str = Field(..., description="IP address of the login")
    user_agent: Optional[str] = Field(None, description="User agent string")
    login_method: str = Field(
        default="password",
        description="Login method (password, oauth, magic_link, etc.)"
    )
    device_id: Optional[str] = Field(None, description="Unique device identifier")
    location: Optional[Dict[str, float]] = Field(
        None,
        description="Geolocation data if available"
    )
    success: bool = Field(default=True, description="Whether login was successful")
    failure_reason: Optional[str] = Field(
        None,
        description="Reason for login failure if unsuccessful"
    )


class UserProfileUpdatedEvent(EventSchemaBase):
    """Schema for user profile update events."""
    user_id: str = Field(..., description="Unique identifier for the user")
    updated_fields: List[str] = Field(
        ...,
        description="List of fields that were updated"
    )
    old_values: Dict[str, Any] = Field(
        default_factory=dict,
        description="Old values of updated fields"
    )
    new_values: Dict[str, Any] = Field(
        ...,
        description="New values of updated fields"
    )
    update_source: str = Field(
        default="user",
        description="Source of update (user, admin, system, etc.)"
    )


class UserPreferenceChangedEvent(EventSchemaBase):
    """Schema for user preference change events."""
    user_id: str = Field(..., description="Unique identifier for the user")
    preference_group: str = Field(
        ...,
        description="Group of preferences (notifications, privacy, display, etc.)"
    )
    preference_key: str = Field(..., description="Specific preference key")
    old_value: Optional[Any] = Field(None, description="Old preference value")
    new_value: Any = Field(..., description="New preference value")


# ============== Incident Events ==============

class IncidentCategory(str, Enum):
    """Categories for incidents."""
    NATURAL_DISASTER = "natural_disaster"
    CONFLICT = "conflict"
    ACCIDENT = "accident"
    HEALTH_CRISIS = "health_crisis"
    ENVIRONMENTAL = "environmental"
    INFRASTRUCTURE = "infrastructure"
    SOCIAL = "social"
    OTHER = "other"


class IncidentSeverity(str, Enum):
    """Severity levels for incidents."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentReportedEvent(EventSchemaBase):
    """Schema for incident reporting events."""
    incident_id: str = Field(..., description="Unique identifier for the incident")
    reporter_id: str = Field(..., description="ID of the user reporting the incident")
    title: str = Field(..., description="Title of the incident", max_length=200)
    description: str = Field(..., description="Detailed description of the incident")
    category: IncidentCategory = Field(..., description="Category of the incident")
    severity: IncidentSeverity = Field(default=IncidentSeverity.MEDIUM, description="Severity level")
    location: Dict[str, Any] = Field(
        ...,
        description="Location data including coordinates and address"
    )
    coordinates: Optional[Dict[str, float]] = Field(
        None,
        description="Geographic coordinates (lat, lng)"
    )
    media_urls: List[str] = Field(
        default_factory=list,
        description="URLs to media files (images, videos)"
    )
    source_urls: List[str] = Field(
        default_factory=list,
        description="URLs to source information"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional incident metadata"
    )
    
    @validator('coordinates')
    def validate_coordinates(cls, v):
        """Validate geographic coordinates."""
        if v:
            if 'lat' not in v or 'lng' not in v:
                raise ValueError('Coordinates must include lat and lng')
            if not (-90 <= v['lat'] <= 90):
                raise ValueError('Latitude must be between -90 and 90')
            if not (-180 <= v['lng'] <= 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v


class IncidentVerifiedEvent(EventSchemaBase):
    """Schema for incident verification events."""
    incident_id: str = Field(..., description="Unique identifier for the incident")
    verifier_id: str = Field(..., description="ID of the user verifying the incident")
    verification_type: str = Field(
        ...,
        description="Type of verification (witness, official, media, etc.)"
    )
    confidence_score: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Confidence score (0-100)"
    )
    evidence_urls: List[str] = Field(
        default_factory=list,
        description="URLs to supporting evidence"
    )
    comments: Optional[str] = Field(None, description="Verification comments")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional verification metadata"
    )


class IncidentResolvedEvent(EventSchemaBase):
    """Schema for incident resolution events."""
    incident_id: str = Field(..., description="Unique identifier for the incident")
    resolver_id: str = Field(..., description="ID of the user resolving the incident")
    resolution_status: str = Field(
        ...,
        description="Resolution status (resolved, false_report, duplicate, etc.)"
    )
    resolution_details: Optional[str] = Field(None, description="Details of resolution")
    official_response_urls: List[str] = Field(
        default_factory=list,
        description="URLs to official responses"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional resolution metadata"
    )


# ============== Briefing Events ==============

class BriefingLevel(str, Enum):
    """Levels of briefing detail."""
    QUICK = "quick"        # 30-second summary
    STANDARD = "standard"  # 2-minute overview
    DEEP_DIVE = "deep_dive"  # 5-minute detailed analysis


class BriefingGeneratedEvent(EventSchemaBase):
    """Schema for briefing generation events."""
    briefing_id: str = Field(..., description="Unique identifier for the briefing")
    user_id: Optional[str] = Field(None, description="ID of the requesting user")
    topic: str = Field(..., description="Topic of the briefing")
    level: BriefingLevel = Field(default=BriefingLevel.STANDARD, description="Briefing level")
    language: str = Field(default="en", description="Language of the briefing")
    duration_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Estimated duration in seconds"
    )
    content_summary: Optional[str] = Field(None, description="Summary of briefing content")
    source_count: int = Field(default=0, description="Number of sources used")
    has_audio: bool = Field(default=False, description="Whether audio version is available")
    has_images: bool = Field(default=False, description="Whether images are included")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional briefing metadata"
    )


class BriefingDeliveredEvent(EventSchemaBase):
    """Schema for briefing delivery events."""
    briefing_id: str = Field(..., description="Unique identifier for the briefing")
    user_id: str = Field(..., description="ID of the receiving user")
    delivery_method: str = Field(
        ...,
        description="Delivery method (push, email, in_app, etc.)"
    )
    delivery_status: str = Field(
        ...,
        description="Delivery status (sent, delivered, read, failed)"
    )
    delivery_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of delivery"
    )
    failure_reason: Optional[str] = Field(None, description="Reason for delivery failure")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional delivery metadata"
    )


class BriefingFeedbackEvent(EventSchemaBase):
    """Schema for briefing feedback events."""
    briefing_id: str = Field(..., description="Unique identifier for the briefing")
    user_id: str = Field(..., description="ID of the user providing feedback")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating (1-5 stars)")
    helpful: Optional[bool] = Field(None, description="Whether briefing was helpful")
    accuracy_score: Optional[int] = Field(None, ge=0, le=100, description="Accuracy score")
    completeness_score: Optional[int] = Field(None, ge=0, le=100, description="Completeness score")
    comments: Optional[str] = Field(None, description="Additional feedback comments")
    suggested_improvements: List[str] = Field(
        default_factory=list,
        description="Suggested improvements"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional feedback metadata"
    )


# ============== Chat Events ==============

class ChatMessageSentEvent(EventSchemaBase):
    """Schema for chat message events."""
    message_id: str = Field(..., description="Unique identifier for the message")
    conversation_id: str = Field(..., description="ID of the conversation")
    user_id: str = Field(..., description="ID of the user sending the message")
    message_type: str = Field(
        default="text",
        description="Type of message (text, image, audio, command)"
    )
    content: str = Field(..., description="Message content")
    language: Optional[str] = Field(None, description="Language of the message")
    is_ai_response: bool = Field(default=False, description="Whether message is from AI")
    parent_message_id: Optional[str] = Field(None, description="ID of parent message if replying")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Citations or references in the message"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional message metadata"
    )
    
    @validator('content')
    def validate_content_length(cls, v):
        """Validate message content length."""
        if len(v) > 10000:
            raise ValueError('Message content too long (max 10000 characters)')
        return v


class ChatConversationStartedEvent(EventSchemaBase):
    """Schema for chat conversation start events."""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    user_id: str = Field(..., description="ID of the user starting the conversation")
    topic: Optional[str] = Field(None, description="Topic of conversation")
    initial_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Initial context for the conversation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation metadata"
    )


class ChatConversationEndedEvent(EventSchemaBase):
    """Schema for chat conversation end events."""
    conversation_id: str = Field(..., description="Unique identifier for the conversation")
    user_id: str = Field(..., description="ID of the user ending the conversation")
    message_count: int = Field(default=0, description="Total number of messages")
    duration_seconds: int = Field(default=0, description="Duration of conversation in seconds")
    end_reason: str = Field(
        ...,
        description="Reason for ending (user_ended, timeout, error, etc.)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional conversation metadata"
    )


# ============== Wallet Events ==============

class TransactionType(str, Enum):
    """Types of wallet transactions."""
    EARN = "earn"          # Earned coins
    SPEND = "spend"        # Spent coins
    TRANSFER = "transfer"  # Transferred coins
    BONUS = "bonus"        # Bonus coins
    PENALTY = "penalty"    # Penalty deduction


class WalletTransactionEvent(EventSchemaBase):
    """Schema for wallet transaction events."""
    transaction_id: str = Field(..., description="Unique identifier for the transaction")
    user_id: str = Field(..., description="ID of the user involved")
    wallet_id: str = Field(..., description="ID of the wallet")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    amount: int = Field(..., description="Amount of coins (positive for earn, negative for spend)")
    description: str = Field(..., description="Description of the transaction")
    source_event_id: Optional[str] = Field(
        None,
        description="ID of the event that triggered this transaction"
    )
    source_event_type: Optional[str] = Field(
        None,
        description="Type of source event"
    )
    balance_before: int = Field(..., description="Wallet balance before transaction")
    balance_after: int = Field(..., description="Wallet balance after transaction")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional transaction metadata"
    )
    
    @validator('amount')
    def validate_amount(cls, v, values):
        """Validate transaction amount based on type."""
        transaction_type = values.get('transaction_type')
        
        if transaction_type == TransactionType.EARN and v <= 0:
            raise ValueError('Earn transactions must have positive amount')
        elif transaction_type == TransactionType.SPEND and v >= 0:
            raise ValueError('Spend transactions must have negative amount')
        elif transaction_type == TransactionType.PENALTY and v >= 0:
            raise ValueError('Penalty transactions must have negative amount')
        
        return v


class WalletBalanceLowEvent(EventSchemaBase):
    """Schema for wallet low balance warning events."""
    user_id: str = Field(..., description="ID of the user")
    wallet_id: str = Field(..., description="ID of the wallet")
    current_balance: int = Field(..., description="Current wallet balance")
    threshold: int = Field(..., description="Low balance threshold")
    days_since_last_earn: Optional[int] = Field(
        None,
        description="Days since last coin earn"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ============== Notification Events ==============

class NotificationType(str, Enum):
    """Types of notifications."""
    SYSTEM = "system"
    INCIDENT = "incident"
    BRIEFING = "briefing"
    CHAT = "chat"
    WALLET = "wallet"
    COMMUNITY = "community"
    SECURITY = "security"


class NotificationSentEvent(EventSchemaBase):
    """Schema for notification sending events."""
    notification_id: str = Field(..., description="Unique identifier for the notification")
    user_id: str = Field(..., description="ID of the receiving user")
    notification_type: NotificationType = Field(..., description="Type of notification")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    delivery_channels: List[str] = Field(
        default_factory=lambda: ["in_app"],
        description="Delivery channels (in_app, email, push, sms)"
    )
    priority: str = Field(default="normal", description="Priority level")
    related_event_id: Optional[str] = Field(
        None,
        description="ID of related event if any"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional notification metadata"
    )


class NotificationReadEvent(EventSchemaBase):
    """Schema for notification read events."""
    notification_id: str = Field(..., description="Unique identifier for the notification")
    user_id: str = Field(..., description="ID of the user")
    read_timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when notification was read"
    )
    read_via: str = Field(
        default="in_app",
        description="How notification was read (in_app, email, etc.)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


# ============== Analytics Events ==============

class UserActivityEvent(EventSchemaBase):
    """Schema for user activity tracking events."""
    user_id: str = Field(..., description="ID of the user")
    activity_type: str = Field(..., description="Type of activity")
    activity_details: Dict[str, Any] = Field(
        ...,
        description="Details of the activity"
    )
    session_id: Optional[str] = Field(None, description="Session identifier")
    device_info: Optional[Dict[str, Any]] = Field(
        None,
        description="Device information"
    )
    location: Optional[Dict[str, Any]] = Field(None, description="Location data")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional activity metadata"
    )


class FeatureUsageEvent(EventSchemaBase):
    """Schema for feature usage tracking events."""
    user_id: str = Field(..., description="ID of the user")
    feature_name: str = Field(..., description="Name of the feature used")
    feature_action: str = Field(..., description="Action performed on the feature")
    duration_seconds: Optional[int] = Field(None, description="Duration of usage in seconds")
    success: bool = Field(default=True, description="Whether the action was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional usage metadata"
    )


# ============== System Events ==============

class SystemAlertEvent(EventSchemaBase):
    """Schema for system alert events."""
    alert_id: str = Field(..., description="Unique identifier for the alert")
    alert_type: str = Field(..., description="Type of alert")
    severity: str = Field(..., description="Severity level")
    message: str = Field(..., description="Alert message")
    component: Optional[str] = Field(None, description="Affected system component")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Related metrics")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata"
    )


class ContentModeratedEvent(EventSchemaBase):
    """Schema for content moderation events."""
    content_id: str = Field(..., description="Unique identifier for the content")
    content_type: str = Field(..., description="Type of content (incident, comment, etc.)")
    moderation_action: str = Field(
        ...,
        description="Action taken (approved, rejected, flagged, etc.)"
    )
    moderator_id: Optional[str] = Field(None, description="ID of the moderator")
    reason: Optional[str] = Field(None, description="Reason for moderation action")
    automated: bool = Field(default=False, description="Whether moderation was automated")
    confidence_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Confidence score for automated moderation"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional moderation metadata"
    )


# ============== Integration Events ==============

class ExternalAPICalledEvent(EventSchemaBase):
    """Schema for external API call events."""
    api_name: str = Field(..., description="Name of the external API")
    endpoint: str = Field(..., description="API endpoint called")
    method: str = Field(..., description="HTTP method used")
    status_code: int = Field(..., description="HTTP status code received")
    duration_ms: int = Field(..., description="Duration of the call in milliseconds")
    success: bool = Field(..., description="Whether the call was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    request_id: Optional[str] = Field(None, description="Request identifier")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional API call metadata"
    )


class WebhookReceivedEvent(EventSchemaBase):
    """Schema for webhook reception events."""
    webhook_id: str = Field(..., description="Unique identifier for the webhook")
    source: str = Field(..., description="Source of the webhook")
    event_type: str = Field(..., description="Type of webhook event")
    payload: Dict[str, Any] = Field(..., description="Webhook payload data")
    signature: Optional[str] = Field(None, description="Signature for verification")
    processing_status: str = Field(
        default="pending",
        description="Processing status (pending, processing, processed, failed)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional webhook metadata"
    )


# ============== Helper Functions ==============

def get_schema_for_event_type(
    event_type_name: str,
) -> Optional[Type[EventSchemaBase]]:
    """
    Get the schema class for a given event type name.
    
    Args:
        event_type_name: Name of the event type
        
    Returns:
        Schema class if found, None otherwise
    """
    # Map event type names to schema classes
    schema_map = {
        "user.registered": UserRegisteredEvent,
        "user.login": UserLoginEvent,
        "user.profile_updated": UserProfileUpdatedEvent,
        "user.preference_changed": UserPreferenceChangedEvent,
        "incident.reported": IncidentReportedEvent,
        "incident.verified": IncidentVerifiedEvent,
        "incident.resolved": IncidentResolvedEvent,
        "briefing.generated": BriefingGeneratedEvent,
        "briefing.delivered": BriefingDeliveredEvent,
        "briefing.feedback": BriefingFeedbackEvent,
        "chat.message_sent": ChatMessageSentEvent,
        "chat.conversation_started": ChatConversationStartedEvent,
        "chat.conversation_ended": ChatConversationEndedEvent,
        "wallet.transaction": WalletTransactionEvent,
        "wallet.balance_low": WalletBalanceLowEvent,
        "notification.sent": NotificationSentEvent,
        "notification.read": NotificationReadEvent,
        "user.activity": UserActivityEvent,
        "feature.usage": FeatureUsageEvent,
        "system.alert": SystemAlertEvent,
        "content.moderated": ContentModeratedEvent,
        "external_api.called": ExternalAPICalledEvent,
        "webhook.received": WebhookReceivedEvent,
    }
    
    return schema_map.get(event_type_name)


def validate_event_data_with_schema(event_type_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate event data against its schema.
    
    Args:
        event_type_name: Name of the event type
        data: Event data to validate
        
    Returns:
        Validated and cleaned event data
        
    Raises:
        ValueError: If validation fails
    """
    schema_class = get_schema_for_event_type(event_type_name)
    
    if not schema_class:
        logger.warning(f"No schema found for event type: {event_type_name}")
        return data
    
    try:
        validated = schema_class(**data)
        return validated.dict(exclude_unset=True)
    except Exception as e:
        raise ValueError(f"Event data validation failed for {event_type_name}: {e}")


def generate_example_payload(event_type_name: str) -> Dict[str, Any]:
    """
    Generate example payload for an event type.
    
    Args:
        event_type_name: Name of the event type
        
    Returns:
        Example payload dictionary
    """
    schema_class = get_schema_for_event_type(event_type_name)
    
    if not schema_class:
        return {"error": f"No schema found for event type: {event_type_name}"}
    
    # Generate example based on schema
    example = {}
    
    for field_name, field in schema_class.__fields__.items():
        # Skip internal fields
        if field_name.startswith('_'):
            continue
        
        # Generate example value based on field type
        field_type = field.type_
        
        if field_type == str:
            example[field_name] = f"example_{field_name}"
        elif field_type == int:
            example[field_name] = 123
        elif field_type == bool:
            example[field_name] = True
        elif field_type == list:
            example[field_name] = []
        elif field_type == dict:
            example[field_name] = {}
        elif field.default is not None:
            example[field_name] = field.default
        elif field.default_factory is not None:
            example[field_name] = field.default_factory()
        else:
            example[field_name] = None
    
    return example