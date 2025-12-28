"""
notification.py - Notification Model

This module defines the Notification model for managing user notifications,
alerts, and messages across the platform. This includes:
- System notifications (updates, maintenance, announcements)
- User notifications (mentions, replies, follows)
- Content notifications (new articles, incidents, comments)
- Verification notifications (status changes, assignments)
- Alert notifications (thresholds, warnings, emergencies)

Key Features:
- Multi-channel notifications (in-app, email, SMS, push)
- Notification grouping and batching
- Priority and urgency levels
- Delivery status tracking
- User preference management
- Notification templates
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.incident import Incident
    from models.article import Article
    from models.comment import Comment
    from models.incident_verification import IncidentVerification
    from models.feedback import Feedback


class NotificationType(Enum):
    """Types of notifications."""
    # System notifications
    SYSTEM_ALERT = "system_alert"              # System-wide alerts
    MAINTENANCE = "maintenance"                # Maintenance announcements
    UPDATE = "update"                          # Platform updates
    ANNOUNCEMENT = "announcement"              # General announcements
    
    # User notifications
    MENTION = "mention"                        # User mentioned in content
    REPLY = "reply"                            # Reply to user's content
    FOLLOW = "follow"                          # New follower
    MESSAGE = "message"                        # Direct message
    INVITATION = "invitation"                  # Invitation to join/participate
    WELCOME = "welcome"                        # Welcome notification
    
    # Content notifications
    NEW_ARTICLE = "new_article"                # New article published
    ARTICLE_UPDATE = "article_update"          # Article updated
    NEW_INCIDENT = "new_incident"              # New incident reported
    INCIDENT_UPDATE = "incident_update"        # Incident updated
    NEW_COMMENT = "new_comment"                # New comment on content
    COMMENT_REPLY = "comment_reply"            # Reply to user's comment
    
    # Verification notifications
    VERIFICATION_ASSIGNED = "verification_assigned"    # Assigned to verification
    VERIFICATION_STATUS_CHANGE = "verification_status_change"  # Verification status changed
    VERIFICATION_NEEDS_REVIEW = "verification_needs_review"    # Verification needs review
    EVIDENCE_ADDED = "evidence_added"          # New evidence added
    SOURCE_ADDED = "source_added"              # New source added
    
    # Alert notifications
    THRESHOLD_ALERT = "threshold_alert"        # Threshold exceeded
    WARNING = "warning"                        # Warning notification
    EMERGENCY_ALERT = "emergency_alert"        # Emergency alert
    RISK_ALERT = "risk_alert"                  # Risk level increased
    
    # Feedback notifications
    FEEDBACK_RECEIVED = "feedback_received"    # New feedback received
    FEEDBACK_RESPONSE = "feedback_response"    # Response to feedback
    FEEDBACK_RESOLVED = "feedback_resolved"    # Feedback resolved
    
    # Dataset notifications
    DATASET_READY = "dataset_ready"            # Dataset processing complete
    DATASET_EXPORTED = "dataset_exported"      # Dataset exported
    DATASET_SHARED = "dataset_shared"          # Dataset shared with user
    
    # Other notifications
    REMINDER = "reminder"                      # Reminder notification
    DEADLINE = "deadline"                      # Deadline approaching
    ACHIEVEMENT = "achievement"                # Achievement unlocked
    OTHER = "other"                            # Other notification types


class NotificationChannel(Enum):
    """Delivery channels for notifications."""
    IN_APP = "in_app"          # In-app notifications
    EMAIL = "email"            # Email notifications
    SMS = "sms"                # SMS/text messages
    PUSH = "push"              # Push notifications
    WEBHOOK = "webhook"        # Webhook notifications
    SLACK = "slack"            # Slack notifications
    TEAMS = "teams"            # Microsoft Teams
    DISCORD = "discord"        # Discord notifications


class NotificationStatus(Enum):
    """Delivery status of notifications."""
    PENDING = "pending"            # Created but not sent
    SENDING = "sending"            # Currently being sent
    SENT = "sent"                  # Successfully sent
    DELIVERED = "delivered"        # Delivered to recipient
    READ = "read"                  # Recipient has read
    FAILED = "failed"              # Failed to send
    BOUNCED = "bounced"            # Email/SMS bounced
    BLOCKED = "blocked"            # Blocked by recipient
    ARCHIVED = "archived"          # Archived


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    LOW = "low"             # Low priority (informational)
    NORMAL = "normal"       # Normal priority
    HIGH = "high"           # High priority (important)
    URGENT = "urgent"       # Urgent (requires attention)
    EMERGENCY = "emergency" # Emergency (immediate action)


class NotificationCategory(Enum):
    """Categories for organizing notifications."""
    SYSTEM = "system"           # System notifications
    USER = "user"               # User interactions
    CONTENT = "content"         # Content updates
    VERIFICATION = "verification"  # Verification updates
    ALERT = "alert"             # Alerts and warnings
    FEEDBACK = "feedback"       # Feedback updates
    DATASET = "dataset"         # Dataset updates
    REMINDER = "reminder"       # Reminders
    ACHIEVEMENT = "achievement" # Achievements
    OTHER = "other"             # Other categories


class Notification(Base, UUIDMixin, TimestampMixin):
    """
    Notification model for managing user notifications.
    
    This model handles the creation, delivery, and tracking of notifications
    across multiple channels with comprehensive status tracking and preferences.
    
    Attributes:
        id: Primary key UUID
        user_id: Recipient user ID
        notification_type: Type of notification
        channel: Delivery channel
        status: Delivery status
        priority: Notification priority
        category: Notification category
        title: Notification title
        message: Notification message/content
        summary: Short summary
        action_url: URL for action/redirect
        action_label: Label for action button
        metadata: Additional JSON data
        template_id: Template used for notification
        template_variables: Variables for template
        sender_id: User/system that sent notification
        related_incident_id: Related incident
        related_article_id: Related article
        related_comment_id: Related comment
        related_verification_id: Related verification
        related_feedback_id: Related feedback
        scheduled_for: When to send (for scheduled notifications)
        sent_at: When notification was sent
        delivered_at: When notification was delivered
        read_at: When notification was read
        expires_at: When notification expires
        retry_count: Number of send retries
        error_message: Error message if failed
        is_batchable: Whether can be batched with others
        batch_key: Key for grouping batchable notifications
        is_silent: Whether notification is silent
        badge_count: Badge count to display
        sound: Sound to play
        vibration_pattern: Vibration pattern (for mobile)
        icon: Icon to display
        image_url: Image URL for notification
        deep_link: Deep link for app navigation
        tags: Categorization tags
        is_archived: Whether notification is archived
    """
    
    __tablename__ = "notifications"
    
    # Recipient
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Notification classification
    notification_type = Column(
        SQLEnum(NotificationType),
        nullable=False,
        index=True
    )
    channel = Column(
        SQLEnum(NotificationChannel),
        default=NotificationChannel.IN_APP,
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(NotificationStatus),
        default=NotificationStatus.PENDING,
        nullable=False,
        index=True
    )
    priority = Column(
        SQLEnum(NotificationPriority),
        default=NotificationPriority.NORMAL,
        nullable=False,
        index=True
    )
    category = Column(
        SQLEnum(NotificationCategory),
        nullable=False,
        index=True
    )
    
    # Content
    title = Column(String(255), nullable=False, index=True)
    message = Column(Text, nullable=False)
    summary = Column(String(500), nullable=True)
    action_url = Column(String(2000), nullable=True)
    action_label = Column(String(100), nullable=True)
    
    # Metadata and templates
    metadata = Column(JSONB, default=dict, nullable=False)
    template_id = Column(String(100), nullable=True, index=True)
    template_variables = Column(JSONB, nullable=True)
    
    # Sender
    sender_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Related entities
    related_incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_comment_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("comments.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    related_feedback_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("feedback.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Delivery timestamps
    scheduled_for = Column(DateTime(timezone=True), nullable=True, index=True)
    sent_at = Column(DateTime(timezone=True), nullable=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    read_at = Column(DateTime(timezone=True), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Delivery tracking
    retry_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True, index=True)
    
    # Batching and grouping
    is_batchable = Column(Boolean, default=False, nullable=False, index=True)
    batch_key = Column(String(255), nullable=True, index=True)
    
    # Presentation
    is_silent = Column(Boolean, default=False, nullable=False)
    badge_count = Column(Integer, nullable=True)
    sound = Column(String(100), nullable=True)
    vibration_pattern = Column(String(100), nullable=True)
    icon = Column(String(255), nullable=True)
    image_url = Column(String(2000), nullable=True)
    deep_link = Column(String(2000), nullable=True)
    
    # Categorization
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="notifications")
    sender = relationship("User", foreign_keys=[sender_id])
    related_incident = relationship("Incident")
    related_article = relationship("Article")
    related_comment = relationship("Comment")
    related_verification = relationship("IncidentVerification")
    related_feedback = relationship("Feedback")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'retry_count >= 0',
            name='check_retry_count_non_negative'
        ),
        CheckConstraint(
            'badge_count IS NULL OR badge_count >= 0',
            name='check_badge_count_non_negative'
        ),
        Index('ix_notifications_user_status', 'user_id', 'status'),
        Index('ix_notifications_user_read', 'user_id', 'read_at'),
        Index('ix_notifications_scheduled', 'scheduled_for', 'status'),
        Index('ix_notifications_batch', 'batch_key', 'user_id'),
        UniqueConstraint('user_id', 'notification_type', 'related_incident_id', 
                        'created_at', name='uq_notification_incident', deferrable=True),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Notification(id={self.id}, type={self.notification_type.value}, user={self.user_id})>"
    
    def __init__(self, **kwargs):
        """Initialize notification with default values."""
        # Set category based on type if not provided
        if 'category' not in kwargs and 'notification_type' in kwargs:
            kwargs['category'] = self._get_category_from_type(kwargs['notification_type'])
        
        super().__init__(**kwargs)
        
        # Set default expiration if not provided
        if self.expires_at is None and self.channel == NotificationChannel.IN_APP:
            self.expires_at = datetime.utcnow() + timedelta(days=30)
    
    @staticmethod
    def _get_category_from_type(notification_type: NotificationType) -> NotificationCategory:
        """Get category from notification type."""
        category_map = {
            # System notifications
            NotificationType.SYSTEM_ALERT: NotificationCategory.SYSTEM,
            NotificationType.MAINTENANCE: NotificationCategory.SYSTEM,
            NotificationType.UPDATE: NotificationCategory.SYSTEM,
            NotificationType.ANNOUNCEMENT: NotificationCategory.SYSTEM,
            
            # User notifications
            NotificationType.MENTION: NotificationCategory.USER,
            NotificationType.REPLY: NotificationCategory.USER,
            NotificationType.FOLLOW: NotificationCategory.USER,
            NotificationType.MESSAGE: NotificationCategory.USER,
            NotificationType.INVITATION: NotificationCategory.USER,
            NotificationType.WELCOME: NotificationCategory.USER,
            
            # Content notifications
            NotificationType.NEW_ARTICLE: NotificationCategory.CONTENT,
            NotificationType.ARTICLE_UPDATE: NotificationCategory.CONTENT,
            NotificationType.NEW_INCIDENT: NotificationCategory.CONTENT,
            NotificationType.INCIDENT_UPDATE: NotificationCategory.CONTENT,
            NotificationType.NEW_COMMENT: NotificationCategory.CONTENT,
            NotificationType.COMMENT_REPLY: NotificationCategory.CONTENT,
            
            # Verification notifications
            NotificationType.VERIFICATION_ASSIGNED: NotificationCategory.VERIFICATION,
            NotificationType.VERIFICATION_STATUS_CHANGE: NotificationCategory.VERIFICATION,
            NotificationType.VERIFICATION_NEEDS_REVIEW: NotificationCategory.VERIFICATION,
            NotificationType.EVIDENCE_ADDED: NotificationCategory.VERIFICATION,
            NotificationType.SOURCE_ADDED: NotificationCategory.VERIFICATION,
            
            # Alert notifications
            NotificationType.THRESHOLD_ALERT: NotificationCategory.ALERT,
            NotificationType.WARNING: NotificationCategory.ALERT,
            NotificationType.EMERGENCY_ALERT: NotificationCategory.ALERT,
            NotificationType.RISK_ALERT: NotificationCategory.ALERT,
            
            # Feedback notifications
            NotificationType.FEEDBACK_RECEIVED: NotificationCategory.FEEDBACK,
            NotificationType.FEEDBACK_RESPONSE: NotificationCategory.FEEDBACK,
            NotificationType.FEEDBACK_RESOLVED: NotificationCategory.FEEDBACK,
            
            # Dataset notifications
            NotificationType.DATASET_READY: NotificationCategory.DATASET,
            NotificationType.DATASET_EXPORTED: NotificationCategory.DATASET,
            NotificationType.DATASET_SHARED: NotificationCategory.DATASET,
            
            # Other notifications
            NotificationType.REMINDER: NotificationCategory.REMINDER,
            NotificationType.DEADLINE: NotificationCategory.REMINDER,
            NotificationType.ACHIEVEMENT: NotificationCategory.ACHIEVEMENT,
            NotificationType.OTHER: NotificationCategory.OTHER,
        }
        
        return category_map.get(notification_type, NotificationCategory.OTHER)
    
    @validates('title')
    def validate_title(self, key: str, title: str) -> str:
        """Validate notification title."""
        title = title.strip()
        if not title:
            raise ValueError("Notification title cannot be empty")
        if len(title) > 255:
            raise ValueError("Notification title cannot exceed 255 characters")
        return title
    
    @validates('message')
    def validate_message(self, key: str, message: str) -> str:
        """Validate notification message."""
        message = message.strip()
        if not message:
            raise ValueError("Notification message cannot be empty")
        return message
    
    @validates('tags')
    def validate_tags(self, key: str, tags: List[str]) -> List[str]:
        """Validate tags."""
        if len(tags) > 20:
            raise ValueError("Cannot have more than 20 tags")
        return [tag.strip().lower() for tag in tags if tag.strip()]
    
    @property
    def is_read(self) -> bool:
        """Check if notification has been read."""
        return self.read_at is not None
    
    @property
    def is_delivered(self) -> bool:
        """Check if notification has been delivered."""
        return self.delivered_at is not None
    
    @property
    def is_sent(self) -> bool:
        """Check if notification has been sent."""
        return self.sent_at is not None
    
    @property
    def is_failed(self) -> bool:
        """Check if notification delivery failed."""
        return self.status == NotificationStatus.FAILED
    
    @property
    def is_scheduled(self) -> bool:
        """Check if notification is scheduled for future."""
        return self.scheduled_for is not None and self.scheduled_for > datetime.utcnow()
    
    @property
    def is_expired(self) -> bool:
        """Check if notification has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """Get age of notification in seconds."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds()
    
    @property
    def age_minutes(self) -> float:
        """Get age of notification in minutes."""
        return self.age_seconds / 60
    
    @property
    def delivery_time_seconds(self) -> Optional[float]:
        """Get delivery time in seconds."""
        if self.sent_at and self.created_at:
            delta = self.sent_at - self.created_at
            return delta.total_seconds()
        return None
    
    @property
    def read_time_seconds(self) -> Optional[float]:
        """Get time to read in seconds."""
        if self.read_at and self.delivered_at:
            delta = self.read_at - self.delivered_at
            return delta.total_seconds()
        return None
    
    @property
    def urgency_score(self) -> float:
        """Calculate urgency score (0-100)."""
        score = 0.0
        
        # Priority contributes 50%
        priority_scores = {
            NotificationPriority.LOW: 10,
            NotificationPriority.NORMAL: 30,
            NotificationPriority.HIGH: 60,
            NotificationPriority.URGENT: 80,
            NotificationPriority.EMERGENCY: 100,
        }
        score += priority_scores.get(self.priority, 0) * 0.5
        
        # Age contributes 30% (newer notifications are more urgent)
        age_hours = self.age_seconds / 3600
        if age_hours <= 0.5:  # 30 minutes
            score += 30
        elif age_hours <= 2:  # 2 hours
            score += 25
        elif age_hours <= 6:  # 6 hours
            score += 20
        elif age_hours <= 24:  # 1 day
            score += 15
        elif age_hours <= 72:  # 3 days
            score += 10
        
        # Channel contributes 20% (some channels are more urgent)
        channel_scores = {
            NotificationChannel.SMS: 20,
            NotificationChannel.PUSH: 15,
            NotificationChannel.IN_APP: 10,
            NotificationChannel.EMAIL: 5,
            NotificationChannel.WEBHOOK: 5,
            NotificationChannel.SLACK: 5,
            NotificationChannel.TEAMS: 5,
            NotificationChannel.DISCORD: 5,
        }
        score += channel_scores.get(self.channel, 0)
        
        return min(100.0, score)
    
    @property
    def related_entity(self) -> Optional[Dict[str, Any]]:
        """Get related entity information."""
        if self.related_incident:
            return {
                "type": "incident",
                "id": str(self.related_incident.id),
                "title": self.related_incident.title,
                "url": f"/incidents/{self.related_incident.id}"
            }
        elif self.related_article:
            return {
                "type": "article",
                "id": str(self.related_article.id),
                "title": self.related_article.title,
                "url": f"/articles/{self.related_article.id}"
            }
        elif self.related_comment:
            return {
                "type": "comment",
                "id": str(self.related_comment.id),
                "preview": self.related_comment.content[:100] + "..." if len(self.related_comment.content) > 100 else self.related_comment.content
            }
        elif self.related_verification:
            return {
                "type": "verification",
                "id": str(self.related_verification.id),
                "status": self.related_verification.status.value
            }
        elif self.related_feedback:
            return {
                "type": "feedback",
                "id": str(self.related_feedback.id),
                "title": self.related_feedback.title
            }
        return None
    
    def mark_as_sent(self, sent_at: Optional[datetime] = None) -> None:
        """Mark notification as sent."""
        self.status = NotificationStatus.SENT
        self.sent_at = sent_at or datetime.utcnow()
        self.retry_count = 0
    
    def mark_as_delivered(self, delivered_at: Optional[datetime] = None) -> None:
        """Mark notification as delivered."""
        self.status = NotificationStatus.DELIVERED
        self.delivered_at = delivered_at or datetime.utcnow()
    
    def mark_as_read(self, read_at: Optional[datetime] = None) -> None:
        """Mark notification as read."""
        self.status = NotificationStatus.READ
        self.read_at = read_at or datetime.utcnow()
        
        # Update badge count if needed
        if self.badge_count is not None:
            self.badge_count = None
    
    def mark_as_failed(self, error_message: str, error_code: Optional[str] = None) -> None:
        """Mark notification as failed."""
        self.status = NotificationStatus.FAILED
        self.error_message = error_message
        self.error_code = error_code
        self.retry_count += 1
    
    def schedule(self, scheduled_for: datetime) -> None:
        """Schedule notification for future delivery."""
        if scheduled_for <= datetime.utcnow():
            raise ValueError("Scheduled time must be in the future")
        
        self.scheduled_for = scheduled_for
        self.status = NotificationStatus.PENDING
    
    def retry(self, max_retries: int = 3) -> bool:
        """Retry sending notification."""
        if self.retry_count >= max_retries:
            self.mark_as_failed("Max retries exceeded", "MAX_RETRIES")
            return False
        
        if self.status == NotificationStatus.FAILED:
            self.status = NotificationStatus.PENDING
            self.error_message = None
            self.error_code = None
            return True
        
        return False
    
    def archive(self) -> None:
        """Archive the notification."""
        self.is_archived = True
    
    def unarchive(self) -> None:
        """Unarchive the notification."""
        self.is_archived = False
    
    def to_dict(self, include_related: bool = True, include_metadata: bool = False) -> Dict[str, Any]:
        """
        Convert notification to dictionary.
        
        Args:
            include_related: Whether to include related entity info
            include_metadata: Whether to include metadata
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "notification_type": self.notification_type.value,
            "channel": self.channel.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "category": self.category.value,
            "title": self.title,
            "message": self.message,
            "summary": self.summary,
            "action_url": self.action_url,
            "action_label": self.action_label,
            "is_read": self.is_read,
            "is_delivered": self.is_delivered,
            "is_sent": self.is_sent,
            "is_failed": self.is_failed,
            "is_scheduled": self.is_scheduled,
            "is_expired": self.is_expired,
            "is_silent": self.is_silent,
            "is_archived": self.is_archived,
            "is_batchable": self.is_batchable,
            "age_seconds": round(self.age_seconds, 2),
            "age_minutes": round(self.age_minutes, 2),
            "urgency_score": round(self.urgency_score, 2),
            "delivery_time_seconds": round(self.delivery_time_seconds, 2) if self.delivery_time_seconds else None,
            "read_time_seconds": round(self.read_time_seconds, 2) if self.read_time_seconds else None,
            "scheduled_for": self.scheduled_for.isoformat() if self.scheduled_for else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "read_at": self.read_at.isoformat() if self.read_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "retry_count": self.retry_count,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "badge_count": self.badge_count,
            "sound": self.sound,
            "icon": self.icon,
            "image_url": self.image_url,
            "deep_link": self.deep_link,
            "tags": self.tags,
            "template_id": self.template_id,
            "sender_id": str(self.sender_id) if self.sender_id else None,
            "related_incident_id": str(self.related_incident_id) if self.related_incident_id else None,
            "related_article_id": str(self.related_article_id) if self.related_article_id else None,
            "related_comment_id": str(self.related_comment_id) if self.related_comment_id else None,
            "related_verification_id": str(self.related_verification_id) if self.related_verification_id else None,
            "related_feedback_id": str(self.related_feedback_id) if self.related_feedback_id else None,
            "batch_key": self.batch_key
        }
        
        if include_metadata:
            result["metadata"] = self.metadata
            result["template_variables"] = self.template_variables
        
        if include_related and self.related_entity:
            result["related_entity"] = self.related_entity
        
        if self.sender:
            result["sender"] = {
                "id": str(self.sender.id),
                "username": self.sender.username,
                "email": getattr(self.sender, 'email', None)
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        user_id: uuid.UUID,
        notification_type: NotificationType,
        title: str,
        message: str,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        sender_id: Optional[uuid.UUID] = None,
        related_incident_id: Optional[uuid.UUID] = None,
        related_article_id: Optional[uuid.UUID] = None,
        related_comment_id: Optional[uuid.UUID] = None,
        related_verification_id: Optional[uuid.UUID] = None,
        related_feedback_id: Optional[uuid.UUID] = None,
        summary: Optional[str] = None,
        action_url: Optional[str] = None,
        action_label: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        template_variables: Optional[Dict[str, Any]] = None,
        scheduled_for: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        is_batchable: bool = False,
        batch_key: Optional[str] = None,
        is_silent: bool = False,
        badge_count: Optional[int] = None,
        sound: Optional[str] = None,
        icon: Optional[str] = None,
        image_url: Optional[str] = None,
        deep_link: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> 'Notification':
        """
        Factory method to create a new notification.
        
        Args:
            user_id: Recipient user ID
            notification_type: Type of notification
            title: Notification title
            message: Notification message
            channel: Delivery channel
            priority: Notification priority
            sender_id: Sender user ID
            related_incident_id: Related incident ID
            related_article_id: Related article ID
            related_comment_id: Related comment ID
            related_verification_id: Related verification ID
            related_feedback_id: Related feedback ID
            summary: Short summary
            action_url: Action URL
            action_label: Action button label
            metadata: Additional metadata
            template_id: Template ID
            template_variables: Template variables
            scheduled_for: Scheduled delivery time
            expires_at: Expiration time
            is_batchable: Whether notification is batchable
            batch_key: Batch grouping key
            is_silent: Whether notification is silent
            badge_count: Badge count
            sound: Sound to play
            icon: Icon to display
            image_url: Image URL
            deep_link: Deep link
            tags: Categorization tags
            
        Returns:
            A new Notification instance
        """
        # Auto-generate batch key if batchable and no key provided
        if is_batchable and not batch_key:
            batch_key = f"{notification_type.value}_{user_id}_{int(datetime.utcnow().timestamp() / 3600)}"
        
        notification = cls(
            user_id=user_id,
            notification_type=notification_type,
            title=title.strip(),
            message=message.strip(),
            channel=channel,
            priority=priority,
            sender_id=sender_id,
            related_incident_id=related_incident_id,
            related_article_id=related_article_id,
            related_comment_id=related_comment_id,
            related_verification_id=related_verification_id,
            related_feedback_id=related_feedback_id,
            summary=summary,
            action_url=action_url,
            action_label=action_label,
            metadata=metadata or {},
            template_id=template_id,
            template_variables=template_variables,
            scheduled_for=scheduled_for,
            expires_at=expires_at,
            is_batchable=is_batchable,
            batch_key=batch_key,
            is_silent=is_silent,
            badge_count=badge_count,
            sound=sound,
            icon=icon,
            image_url=image_url,
            deep_link=deep_link,
            tags=tags or [],
            status=NotificationStatus.PENDING
        )
        
        return notification
    
    @classmethod
    def create_verification_assigned(
        cls,
        user_id: uuid.UUID,
        verification_id: uuid.UUID,
        incident_title: str,
        assigned_by_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Notification':
        """
        Create verification assignment notification.
        
        Args:
            user_id: Assignee user ID
            verification_id: Verification ID
            incident_title: Incident title
            assigned_by_id: User who made assignment
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.VERIFICATION_ASSIGNED,
            title="New Verification Assigned",
            message=f"You have been assigned to verify incident: {incident_title}",
            priority=NotificationPriority.HIGH,
            sender_id=assigned_by_id,
            related_verification_id=verification_id,
            action_url=f"/verifications/{verification_id}",
            action_label="Review Verification",
            tags=["verification", "assignment", "task"],
            **kwargs
        )
    
    @classmethod
    def create_mention(
        cls,
        user_id: uuid.UUID,
        mentioned_by_id: uuid.UUID,
        content_type: str,
        content_id: uuid.UUID,
        content_preview: str,
        **kwargs
    ) -> 'Notification':
        """
        Create mention notification.
        
        Args:
            user_id: Mentioned user ID
            mentioned_by_id: User who mentioned
            content_type: Type of content (incident, article, comment)
            content_id: Content ID
            content_preview: Content preview
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.MENTION,
            title="You were mentioned",
            message=f"You were mentioned in a {content_type}",
            priority=NotificationPriority.NORMAL,
            sender_id=mentioned_by_id,
            metadata={
                "content_type": content_type,
                "content_preview": content_preview[:200]
            },
            action_url=f"/{content_type}s/{content_id}",
            action_label="View Mention",
            tags=["mention", "social", content_type],
            **kwargs
        )
    
    @classmethod
    def create_new_incident(
        cls,
        user_ids: List[uuid.UUID],
        incident_id: uuid.UUID,
        incident_title: str,
        incident_type: str,
        location: str,
        **kwargs
    ) -> List['Notification']:
        """
        Create new incident notifications for multiple users.
        
        Args:
            user_ids: List of user IDs to notify
            incident_id: Incident ID
            incident_title: Incident title
            incident_type: Type of incident
            location: Incident location
            **kwargs: Additional arguments
            
        Returns:
            List of Notification instances
        """
        notifications = []
        batch_key = f"new_incident_{incident_id}"
        
        for user_id in user_ids:
            notification = cls.create(
                user_id=user_id,
                notification_type=NotificationType.NEW_INCIDENT,
                title=f"New {incident_type} Reported",
                message=f"New incident reported: {incident_title} in {location}",
                priority=NotificationPriority.HIGH,
                related_incident_id=incident_id,
                action_url=f"/incidents/{incident_id}",
                action_label="View Incident",
                is_batchable=True,
                batch_key=batch_key,
                tags=["incident", "new", incident_type.lower()],
                **kwargs
            )
            notifications.append(notification)
        
        return notifications
    
    @classmethod
    def create_emergency_alert(
        cls,
        user_ids: List[uuid.UUID],
        alert_title: str,
        alert_message: str,
        location: str,
        severity: str = "high",
        **kwargs
    ) -> List['Notification']:
        """
        Create emergency alert notifications.
        
        Args:
            user_ids: List of user IDs to notify
            alert_title: Alert title
            alert_message: Alert message
            location: Location of emergency
            severity: Alert severity
            **kwargs: Additional arguments
            
        Returns:
            List of Notification instances
        """
        notifications = []
        batch_key = f"emergency_alert_{int(datetime.utcnow().timestamp())}"
        
        for user_id in user_ids:
            notification = cls.create(
                user_id=user_id,
                notification_type=NotificationType.EMERGENCY_ALERT,
                title=f"🚨 {alert_title}",
                message=f"Emergency alert in {location}: {alert_message}",
                priority=NotificationPriority.EMERGENCY,
                channel=NotificationChannel.PUSH,
                is_silent=False,
                sound="emergency",
                metadata={"severity": severity, "location": location},
                action_url="/alerts",
                action_label="View Details",
                is_batchable=True,
                batch_key=batch_key,
                expires_at=datetime.utcnow() + timedelta(hours=24),
                tags=["emergency", "alert", severity],
                **kwargs
            )
            notifications.append(notification)
        
        return notifications
    
    @classmethod
    def create_system_maintenance(
        cls,
        user_ids: List[uuid.UUID],
        maintenance_title: str,
        maintenance_message: str,
        start_time: datetime,
        end_time: datetime,
        **kwargs
    ) -> List['Notification']:
        """
        Create system maintenance notifications.
        
        Args:
            user_ids: List of user IDs to notify
            maintenance_title: Maintenance title
            maintenance_message: Maintenance details
            start_time: Maintenance start time
            end_time: Maintenance end time
            **kwargs: Additional arguments
            
        Returns:
            List of Notification instances
        """
        notifications = []
        batch_key = f"maintenance_{start_time.date()}"
        
        for user_id in user_ids:
            notification = cls.create(
                user_id=user_id,
                notification_type=NotificationType.MAINTENANCE,
                title=f"System Maintenance: {maintenance_title}",
                message=maintenance_message,
                priority=NotificationPriority.NORMAL,
                channel=NotificationChannel.EMAIL,
                metadata={
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_minutes": (end_time - start_time).total_seconds() / 60
                },
                scheduled_for=start_time - timedelta(hours=24),
                action_url="/status",
                action_label="Check Status",
                is_batchable=True,
                batch_key=batch_key,
                tags=["system", "maintenance"],
                **kwargs
            )
            notifications.append(notification)
        
        return notifications
    
    @classmethod
    def create_threshold_alert(
        cls,
        user_id: uuid.UUID,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        unit: str = "",
        **kwargs
    ) -> 'Notification':
        """
        Create threshold alert notification.
        
        Args:
            user_id: User ID to notify
            metric_name: Name of the metric
            current_value: Current metric value
            threshold_value: Threshold value
            unit: Unit of measurement
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.THRESHOLD_ALERT,
            title=f"Threshold Alert: {metric_name}",
            message=f"{metric_name} has reached {current_value}{unit}, exceeding threshold of {threshold_value}{unit}",
            priority=NotificationPriority.HIGH,
            metadata={
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "unit": unit,
                "exceeded_by": current_value - threshold_value
            },
            action_url="/dashboard/alerts",
            action_label="View Metrics",
            tags=["alert", "threshold", metric_name.lower().replace(" ", "_")],
            **kwargs
        )
    
    @classmethod
    def create_new_follower(
        cls,
        user_id: uuid.UUID,
        follower_id: uuid.UUID,
        follower_username: str,
        **kwargs
    ) -> 'Notification':
        """
        Create new follower notification.
        
        Args:
            user_id: User who gained a follower
            follower_id: User who followed
            follower_username: Username of follower
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.FOLLOW,
            title="New Follower",
            message=f"{follower_username} started following you",
            priority=NotificationPriority.LOW,
            sender_id=follower_id,
            action_url=f"/users/{follower_username}",
            action_label="View Profile",
            tags=["social", "follow"],
            **kwargs
        )
    
    @classmethod
    def create_comment_reply(
        cls,
        user_id: uuid.UUID,
        comment_id: uuid.UUID,
        replied_by_id: uuid.UUID,
        replied_by_username: str,
        comment_preview: str,
        parent_comment_preview: str,
        **kwargs
    ) -> 'Notification':
        """
        Create comment reply notification.
        
        Args:
            user_id: User who received reply
            comment_id: New comment ID
            replied_by_id: User who replied
            replied_by_username: Username of replier
            comment_preview: Reply content preview
            parent_comment_preview: Original comment preview
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.COMMENT_REPLY,
            title="New Reply to Your Comment",
            message=f"{replied_by_username} replied to your comment",
            priority=NotificationPriority.NORMAL,
            sender_id=replied_by_id,
            related_comment_id=comment_id,
            metadata={
                "comment_preview": comment_preview[:200],
                "parent_comment_preview": parent_comment_preview[:200]
            },
            action_url=f"/comments/{comment_id}",
            action_label="View Reply",
            tags=["comment", "reply", "social"],
            **kwargs
        )
    
    @classmethod
    def create_dataset_ready(
        cls,
        user_id: uuid.UUID,
        dataset_name: str,
        dataset_id: uuid.UUID,
        rows_count: int,
        processing_time_seconds: float,
        **kwargs
    ) -> 'Notification':
        """
        Create dataset ready notification.
        
        Args:
            user_id: User ID to notify
            dataset_name: Name of dataset
            dataset_id: Dataset ID
            rows_count: Number of rows in dataset
            processing_time_seconds: Time taken to process
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.DATASET_READY,
            title=f"Dataset Ready: {dataset_name}",
            message=f"Your dataset '{dataset_name}' with {rows_count:,} rows is ready for download",
            priority=NotificationPriority.NORMAL,
            metadata={
                "dataset_name": dataset_name,
                "rows_count": rows_count,
                "processing_time_seconds": processing_time_seconds
            },
            action_url=f"/datasets/{dataset_id}",
            action_label="Download Dataset",
            tags=["dataset", "export"],
            **kwargs
        )
    
    @classmethod
    def create_welcome(
        cls,
        user_id: uuid.UUID,
        username: str,
        **kwargs
    ) -> 'Notification':
        """
        Create welcome notification for new users.
        
        Args:
            user_id: New user ID
            username: New username
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.WELCOME,
            title=f"Welcome to the Platform, {username}!",
            message="We're excited to have you on board. Here are some things you can do to get started...",
            priority=NotificationPriority.NORMAL,
            action_url="/getting-started",
            action_label="Get Started",
            metadata={
                "welcome_steps": [
                    "Complete your profile",
                    "Explore the dashboard",
                    "Join your first incident verification",
                    "Set up notification preferences"
                ]
            },
            tags=["welcome", "onboarding"],
            **kwargs
        )
    
    @classmethod
    def create_feedback_response(
        cls,
        user_id: uuid.UUID,
        feedback_id: uuid.UUID,
        feedback_title: str,
        responder_id: uuid.UUID,
        responder_username: str,
        response_preview: str,
        **kwargs
    ) -> 'Notification':
        """
        Create feedback response notification.
        
        Args:
            user_id: User who submitted feedback
            feedback_id: Feedback ID
            feedback_title: Feedback title
            responder_id: User who responded
            responder_username: Username of responder
            response_preview: Response content preview
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.FEEDBACK_RESPONSE,
            title=f"Response to Your Feedback: {feedback_title}",
            message=f"{responder_username} responded to your feedback",
            priority=NotificationPriority.NORMAL,
            sender_id=responder_id,
            related_feedback_id=feedback_id,
            metadata={
                "feedback_title": feedback_title,
                "response_preview": response_preview[:200]
            },
            action_url=f"/feedback/{feedback_id}",
            action_label="View Response",
            tags=["feedback", "response"],
            **kwargs
        )
    
    @classmethod
    def create_verification_status_change(
        cls,
        user_ids: List[uuid.UUID],
        verification_id: uuid.UUID,
        incident_title: str,
        old_status: str,
        new_status: str,
        changed_by_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> List['Notification']:
        """
        Create verification status change notifications.
        
        Args:
            user_ids: List of user IDs to notify
            verification_id: Verification ID
            incident_title: Incident title
            old_status: Previous verification status
            new_status: New verification status
            changed_by_id: User who changed status
            **kwargs: Additional arguments
            
        Returns:
            List of Notification instances
        """
        notifications = []
        batch_key = f"verification_status_{verification_id}_{new_status}"
        
        for user_id in user_ids:
            notification = cls.create(
                user_id=user_id,
                notification_type=NotificationType.VERIFICATION_STATUS_CHANGE,
                title=f"Verification Status Updated: {incident_title}",
                message=f"Verification status changed from {old_status} to {new_status}",
                priority=NotificationPriority.NORMAL,
                sender_id=changed_by_id,
                related_verification_id=verification_id,
                metadata={
                    "old_status": old_status,
                    "new_status": new_status,
                    "incident_title": incident_title
                },
                action_url=f"/verifications/{verification_id}",
                action_label="View Verification",
                is_batchable=True,
                batch_key=batch_key,
                tags=["verification", "status", new_status.lower()],
                **kwargs
            )
            notifications.append(notification)
        
        return notifications
    
    @classmethod
    def create_article_published(
        cls,
        user_ids: List[uuid.UUID],
        article_id: uuid.UUID,
        article_title: str,
        author_name: str,
        **kwargs
    ) -> List['Notification']:
        """
        Create article published notifications.
        
        Args:
            user_ids: List of user IDs to notify
            article_id: Article ID
            article_title: Article title
            author_name: Author name
            **kwargs: Additional arguments
            
        Returns:
            List of Notification instances
        """
        notifications = []
        batch_key = f"article_published_{article_id}"
        
        for user_id in user_ids:
            notification = cls.create(
                user_id=user_id,
                notification_type=NotificationType.NEW_ARTICLE,
                title=f"New Article: {article_title}",
                message=f"A new article '{article_title}' by {author_name} has been published",
                priority=NotificationPriority.NORMAL,
                related_article_id=article_id,
                action_url=f"/articles/{article_id}",
                action_label="Read Article",
                is_batchable=True,
                batch_key=batch_key,
                tags=["article", "publication"],
                **kwargs
            )
            notifications.append(notification)
        
        return notifications
    
    @classmethod
    def create_direct_message(
        cls,
        user_id: uuid.UUID,
        sender_id: uuid.UUID,
        sender_username: str,
        message_preview: str,
        conversation_id: uuid.UUID,
        **kwargs
    ) -> 'Notification':
        """
        Create direct message notification.
        
        Args:
            user_id: Recipient user ID
            sender_id: Sender user ID
            sender_username: Sender username
            message_preview: Message preview
            conversation_id: Conversation ID
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.MESSAGE,
            title=f"New Message from {sender_username}",
            message=message_preview,
            priority=NotificationPriority.HIGH,
            sender_id=sender_id,
            metadata={
                "conversation_id": str(conversation_id),
                "message_preview": message_preview[:200]
            },
            action_url=f"/messages/{conversation_id}",
            action_label="Reply",
            sound="message",
            tags=["message", "direct"],
            **kwargs
        )
    
    @classmethod
    def create_reminder(
        cls,
        user_id: uuid.UUID,
        reminder_title: str,
        reminder_message: str,
        due_date: datetime,
        entity_type: Optional[str] = None,
        entity_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'Notification':
        """
        Create reminder notification.
        
        Args:
            user_id: User ID to notify
            reminder_title: Reminder title
            reminder_message: Reminder message
            due_date: Due date for reminder
            entity_type: Type of entity (incident, verification, etc.)
            entity_id: Entity ID
            **kwargs: Additional arguments
            
        Returns:
            Notification instance
        """
        action_url = None
        if entity_type and entity_id:
            action_url = f"/{entity_type}s/{entity_id}"
        
        return cls.create(
            user_id=user_id,
            notification_type=NotificationType.REMINDER,
            title=f"Reminder: {reminder_title}",
            message=reminder_message,
            priority=NotificationPriority.NORMAL,
            metadata={
                "due_date": due_date.isoformat(),
                "entity_type": entity_type,
                "entity_id": str(entity_id) if entity_id else None
            },
            action_url=action_url,
            action_label="View Details" if action_url else None,
            scheduled_for=due_date - timedelta(hours=1),  # Send 1 hour before due
            tags=["reminder", "task"],
            **kwargs
        )


# Additional helper model for notification preferences
class NotificationPreference(Base, UUIDMixin, TimestampMixin):
    """
    User notification preferences model.
    
    Stores user preferences for different types of notifications
    across different channels.
    """
    
    __tablename__ = "notification_preferences"
    
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Notification type preferences
    notification_type = Column(
        SQLEnum(NotificationType),
        nullable=False,
        index=True
    )
    
    # Channel preferences (JSONB storing channel -> enabled mapping)
    channel_preferences = Column(
        JSONB,
        default={
            "in_app": True,
            "email": True,
            "push": True,
            "sms": False,
            "webhook": False,
            "slack": False,
            "teams": False,
            "discord": False
        },
        nullable=False
    )
    
    # Priority thresholds (only send notifications above this priority)
    minimum_priority = Column(
        SQLEnum(NotificationPriority),
        default=NotificationPriority.LOW,
        nullable=False
    )
    
    # Do not disturb settings
    do_not_disturb_start = Column(DateTime(timezone=True), nullable=True)
    do_not_disturb_end = Column(DateTime(timezone=True), nullable=True)
    
    # Global mute
    is_muted = Column(Boolean, default=False, nullable=False)
    
    # Frequency limiting
    max_notifications_per_day = Column(Integer, default=100, nullable=False)
    max_notifications_per_hour = Column(Integer, default=20, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="notification_preferences")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'notification_type', name='uq_user_notification_type'),
        CheckConstraint('max_notifications_per_day >= 0', name='check_max_per_day_non_negative'),
        CheckConstraint('max_notifications_per_hour >= 0', name='check_max_per_hour_non_negative'),
    )
    
    @property
    def is_in_do_not_disturb(self) -> bool:
        """Check if current time is within do not disturb period."""
        if not self.do_not_disturb_start or not self.do_not_disturb_end:
            return False
        
        now = datetime.utcnow()
        return self.do_not_disturb_start <= now <= self.do_not_disturb_end
    
    def is_channel_enabled(self, channel: NotificationChannel) -> bool:
        """Check if a specific channel is enabled for this notification type."""
        if self.is_muted or self.is_in_do_not_disturb:
            return False
        
        channel_key = channel.value
        return self.channel_preferences.get(channel_key, False)
    
    def get_enabled_channels(self) -> List[NotificationChannel]:
        """Get list of enabled channels for this notification type."""
        if self.is_muted or self.is_in_do_not_disturb:
            return []
        
        enabled_channels = []
        for channel in NotificationChannel:
            if self.channel_preferences.get(channel.value, False):
                enabled_channels.append(channel)
        
        return enabled_channels
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preference to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "notification_type": self.notification_type.value,
            "channel_preferences": self.channel_preferences,
            "minimum_priority": self.minimum_priority.value,
            "do_not_disturb_start": self.do_not_disturb_start.isoformat() if self.do_not_disturb_start else None,
            "do_not_disturb_end": self.do_not_disturb_end.isoformat() if self.do_not_disturb_end else None,
            "is_muted": self.is_muted,
            "max_notifications_per_day": self.max_notifications_per_day,
            "max_notifications_per_hour": self.max_notifications_per_hour,
            "is_in_do_not_disturb": self.is_in_do_not_disturb,
            "enabled_channels": [channel.value for channel in self.get_enabled_channels()],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Notification template model for reusable notification formats
class NotificationTemplate(Base, UUIDMixin, TimestampMixin):
    """
    Notification template model for reusable notification formats.
    
    Templates allow for consistent notification formatting across the platform.
    """
    
    __tablename__ = "notification_templates"
    
    template_id = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Template content
    title_template = Column(Text, nullable=False)
    message_template = Column(Text, nullable=False)
    summary_template = Column(Text, nullable=True)
    action_url_template = Column(Text, nullable=True)
    action_label_template = Column(Text, nullable=True)
    
    # Template configuration
    notification_type = Column(SQLEnum(NotificationType), nullable=False)
    default_priority = Column(
        SQLEnum(NotificationPriority),
        default=NotificationPriority.NORMAL,
        nullable=False
    )
    default_channel = Column(
        SQLEnum(NotificationChannel),
        default=NotificationChannel.IN_APP,
        nullable=False
    )
    
    # Template variables schema
    variables_schema = Column(JSONB, nullable=True)
    
    # Metadata
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    version = Column(String(20), nullable=False, default="1.0.0")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('version ~* \'^\\d+\\.\\d+\\.\\d+$\'', name='check_version_format'),
    )
    
    def render(
        self, 
        variables: Dict[str, Any], 
        user_id: Optional[uuid.UUID] = None
    ) -> Dict[str, Any]:
        """
        Render template with variables.
        
        Args:
            variables: Template variables
            user_id: Optional user ID for user-specific rendering
            
        Returns:
            Rendered template dictionary
        """
        # Simple template rendering (can be enhanced with Jinja2 or similar)
        rendered = {}
        
        def render_template(template: str, vars_dict: Dict[str, Any]) -> str:
            """Simple template rendering."""
            if not template:
                return ""
            
            result = template
            for key, value in vars_dict.items():
                placeholder = f"{{{key}}}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            return result
        
        rendered["title"] = render_template(self.title_template, variables)
        rendered["message"] = render_template(self.message_template, variables)
        
        if self.summary_template:
            rendered["summary"] = render_template(self.summary_template, variables)
        
        if self.action_url_template:
            rendered["action_url"] = render_template(self.action_url_template, variables)
        
        if self.action_label_template:
            rendered["action_label"] = render_template(self.action_label_template, variables)
        
        return rendered
    
    def to_dict(self, include_templates: bool = False) -> Dict[str, Any]:
        """Convert template to dictionary."""
        result = {
            "id": str(self.id),
            "template_id": self.template_id,
            "name": self.name,
            "description": self.description,
            "notification_type": self.notification_type.value,
            "default_priority": self.default_priority.value,
            "default_channel": self.default_channel.value,
            "variables_schema": self.variables_schema,
            "is_active": self.is_active,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_templates:
            result["title_template"] = self.title_template
            result["message_template"] = self.message_template
            result["summary_template"] = self.summary_template
            result["action_url_template"] = self.action_url_template
            result["action_label_template"] = self.action_label_template
        
        return result