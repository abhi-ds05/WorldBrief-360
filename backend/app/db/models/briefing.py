"""
Briefing/digest model for curated content delivery.
"""

import enum
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column,
    DateTime,
    Boolean,
    String,
    Text,
    Integer,
    ForeignKey,
    UniqueConstraint,
    Index,
    Enum,
    CheckConstraint,
    func,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, validates

from db.base import Base, TimestampMixin, SoftDeleteMixin, generate_uuid


class BriefingType(str, enum.Enum):
    """Briefing content types."""
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"
    MONTHLY_DIGEST = "monthly_digest"
    TOPIC_DIGEST = "topic_digest"
    PERSONALIZED = "personalized"
    BREAKING_NEWS = "breaking_news"
    SECURITY_ALERT = "security_alert"
    MARKET_UPDATE = "market_update"
    TECHNICAL_ANALYSIS = "technical_analysis"


class BriefingStatus(str, enum.Enum):
    """Briefing delivery status."""
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BriefingFormat(str, enum.Enum):
    """Briefing output formats."""
    EMAIL = "email"
    WEB = "web"
    PDF = "pdf"
    MOBILE = "mobile"
    RSS = "rss"
    API = "api"


class BriefingFrequency(str, enum.Enum):
    """Briefing frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    HOURLY = "hourly"
    REAL_TIME = "real_time"
    ON_DEMAND = "on_demand"


class Briefing(Base, TimestampMixin, SoftDeleteMixin):
    """
    Briefing model for curated content digests and newsletters.
    """
    
    __tablename__ = "briefing"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    author_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    topic_id = Column(UUID(as_uuid=True), ForeignKey('topic.id'), nullable=True, index=True)
    
    # Basic information
    title = Column(String(500), nullable=False, index=True)
    slug = Column(String(500), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Content type
    briefing_type = Column(Enum(BriefingType), nullable=False, index=True)
    format = Column(Enum(BriefingFormat), default=BriefingFormat.EMAIL, nullable=False, index=True)
    frequency = Column(Enum(BriefingFrequency), nullable=True, index=True)
    
    # Content
    introduction = Column(Text, nullable=True)
    conclusion = Column(Text, nullable=True)
    content = Column(JSONB, default=list)  # Structured content sections
    
    # Content references
    article_ids = Column(JSONB, default=list)  # List of article IDs included
    incident_ids = Column(JSONB, default=list)  # List of incident IDs included
    topic_ids = Column(JSONB, default=list)  # List of topic IDs covered
    
    # Scheduling and delivery
    status = Column(Enum(BriefingStatus), default=BriefingStatus.DRAFT, nullable=False, index=True)
    scheduled_for = Column(DateTime(timezone=True), nullable=True, index=True)
    sent_at = Column(DateTime(timezone=True), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Delivery settings
    timezone = Column(String(50), default='UTC', nullable=False)
    delivery_method = Column(JSONB, default=list)  # [email, push, web, etc.]
    target_audience = Column(JSONB, default=dict)  # Audience filters
    
    # Statistics
    total_recipients = Column(Integer, default=0, nullable=False)
    sent_count = Column(Integer, default=0, nullable=False)
    opened_count = Column(Integer, default=0, nullable=False)
    clicked_count = Column(Integer, default=0, nullable=False)
    bounce_count = Column(Integer, default=0, nullable=False)
    unsubscribe_count = Column(Integer, default=0, nullable=False)
    
    # Engagement metrics
    open_rate = Column(Float, default=0.0, nullable=False)
    click_rate = Column(Float, default=0.0, nullable=False)
    engagement_score = Column(Float, default=0.0, nullable=False)
    
    # Personalization
    is_personalized = Column(Boolean, default=False, nullable=False, index=True)
    personalization_rules = Column(JSONB, default=dict)
    
    # Templates and styling
    template_id = Column(String(100), nullable=True)
    theme = Column(String(50), nullable=True)
    css_overrides = Column(Text, nullable=True)
    
    # Metadata
    tags = Column(JSONB, default=list, index=True)
    metadata = Column(JSONB, default=dict)
    custom_fields = Column(JSONB, default=dict)
    
    # Relationships
    author = relationship("User", backref="briefings")
    topic = relationship("Topic", back_populates="briefings")
    deliveries = relationship("BriefingDelivery", back_populates="briefing", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_briefing_status_scheduled', status, scheduled_for),
        Index('idx_briefing_type_status', briefing_type, status, scheduled_for.desc()),
        Index('idx_briefing_author_status', author_id, status, scheduled_for.desc()),
        Index('idx_briefing_sent_at', sent_at.desc()),
        Index('idx_briefing_tags', tags, postgresql_using='gin'),
        UniqueConstraint('slug', name='uq_briefing_slug'),
        CheckConstraint('total_recipients >= 0', name='check_recipients_non_negative'),
        CheckConstraint('sent_count >= 0', name='check_sent_count_non_negative'),
        CheckConstraint('opened_count >= 0', name='check_opened_count_non_negative'),
        CheckConstraint('clicked_count >= 0', name='check_clicked_count_non_negative'),
        CheckConstraint('bounce_count >= 0', name='check_bounce_count_non_negative'),
        CheckConstraint('unsubscribe_count >= 0', name='check_unsubscribe_count_non_negative'),
        CheckConstraint('open_rate >= 0 AND open_rate <= 1', name='check_open_rate_range'),
        CheckConstraint('click_rate >= 0 AND click_rate <= 1', name='check_click_rate_range'),
        CheckConstraint('engagement_score >= 0 AND engagement_score <= 1', name='check_engagement_range'),
    )
    
    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate briefing slug."""
        if not slug:
            raise ValueError("Slug cannot be empty")
        if ' ' in slug:
            raise ValueError("Slug cannot contain spaces")
        return slug.lower()
    
    @validates('title')
    def validate_title(self, key, title):
        """Validate briefing title."""
        if not title or len(title.strip()) < 5:
            raise ValueError("Title must be at least 5 characters")
        if len(title) > 500:
            raise ValueError("Title cannot exceed 500 characters")
        return title.strip()
    
    def is_scheduled(self) -> bool:
        """Check if briefing is scheduled."""
        return self.status == BriefingStatus.SCHEDULED and self.scheduled_for is not None
    
    def is_sent(self) -> bool:
        """Check if briefing has been sent."""
        return self.status == BriefingStatus.SENT and self.sent_at is not None
    
    def is_draft(self) -> bool:
        """Check if briefing is a draft."""
        return self.status == BriefingStatus.DRAFT
    
    def should_send_now(self) -> bool:
        """Check if briefing should be sent now."""
        if self.status != BriefingStatus.SCHEDULED:
            return False
        
        if not self.scheduled_for:
            return False
        
        return datetime.utcnow() >= self.scheduled_for
    
    def mark_sending(self) -> None:
        """Mark briefing as being sent."""
        self.status = BriefingStatus.SENDING
    
    def mark_sent(self, sent_count: int = 0) -> None:
        """Mark briefing as sent."""
        self.status = BriefingStatus.SENT
        self.sent_at = datetime.utcnow()
        self.sent_count = sent_count
    
    def mark_failed(self, error: str = None) -> None:
        """Mark briefing as failed."""
        self.status = BriefingStatus.FAILED
        
        if error and self.metadata:
            if 'delivery_errors' not in self.metadata:
                self.metadata['delivery_errors'] = []
            self.metadata['delivery_errors'].append({
                'error': error,
                'failed_at': datetime.utcnow().isoformat()
            })
    
    def update_engagement_metrics(self) -> None:
        """Update engagement metrics."""
        if self.sent_count > 0:
            self.open_rate = self.opened_count / self.sent_count
            self.click_rate = self.clicked_count / self.sent_count
            
            # Simple engagement score
            self.engagement_score = (self.open_rate * 0.4) + (self.click_rate * 0.6)
        else:
            self.open_rate = 0.0
            self.click_rate = 0.0
            self.engagement_score = 0.0
    
    def increment_opened_count(self) -> None:
        """Increment opened count."""
        self.opened_count += 1
        self.update_engagement_metrics()
    
    def increment_clicked_count(self) -> None:
        """Increment clicked count."""
        self.clicked_count += 1
        self.update_engagement_metrics()
    
    def increment_bounce_count(self) -> None:
        """Increment bounce count."""
        self.bounce_count += 1
    
    def increment_unsubscribe_count(self) -> None:
        """Increment unsubscribe count."""
        self.unsubscribe_count += 1
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of briefing content."""
        if not self.content:
            return {}
        
        summary = {
            'total_sections': len(self.content),
            'total_items': 0,
            'section_types': {},
        }
        
        for section in self.content:
            section_type = section.get('type', 'unknown')
            items = section.get('items', [])
            
            summary['total_items'] += len(items)
            summary['section_types'][section_type] = summary['section_types'].get(section_type, 0) + 1
        
        return summary
    
    def get_delivery_stats(self) -> Dict[str, Any]:
        """Get delivery statistics."""
        return {
            'total_recipients': self.total_recipients,
            'sent_count': self.sent_count,
            'opened_count': self.opened_count,
            'clicked_count': self.clicked_count,
            'bounce_count': self.bounce_count,
            'unsubscribe_count': self.unsubscribe_count,
            'open_rate': self.open_rate,
            'click_rate': self.click_rate,
            'engagement_score': self.engagement_score,
            'delivery_rate': self.sent_count / self.total_recipients if self.total_recipients > 0 else 0,
            'bounce_rate': self.bounce_count / self.sent_count if self.sent_count > 0 else 0,
        }
    
    def generate_preview(self) -> Dict[str, Any]:
        """Generate briefing preview for display."""
        return {
            'id': str(self.id),
            'title': self.title,
            'description': self.description,
            'type': self.briefing_type.value,
            'format': self.format.value,
            'status': self.status.value,
            'scheduled_for': self.scheduled_for.isoformat() if self.scheduled_for else None,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'content_summary': self.get_content_summary(),
            'delivery_stats': self.get_delivery_stats(),
            'is_personalized': self.is_personalized,
            'author': {
                'id': str(self.author.id) if self.author else None,
                'username': self.author.username if self.author else None,
                'full_name': self.author.full_name if self.author else None,
            } if self.author else None,
        }
    
    def to_dict(self, include_content: bool = True) -> dict:
        """Convert to dictionary representation."""
        data = super().to_dict()
        
        if not include_content:
            # Don't include full content in summary
            if 'content' in data:
                del data['content']
        
        # Add computed fields
        data['is_scheduled'] = self.is_scheduled()
        data['is_sent'] = self.is_sent()
        data['is_draft'] = self.is_draft()
        data['should_send_now'] = self.should_send_now()
        data['content_summary'] = self.get_content_summary()
        data['delivery_stats'] = self.get_delivery_stats()
        
        return data


class BriefingDelivery(Base, TimestampMixin):
    """
    Individual briefing delivery tracking.
    """
    
    __tablename__ = "briefing_delivery"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    briefing_id = Column(UUID(as_uuid=True), ForeignKey('briefing.id'), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    
    # Delivery information
    delivery_method = Column(String(50), nullable=False, index=True)  # email, push, web, etc.
    delivery_status = Column(String(50), default='pending', nullable=False, index=True)  # pending, sent, delivered, failed, bounced
    
    # Tracking
    sent_at = Column(DateTime(timezone=True), nullable=True, index=True)
    delivered_at = Column(DateTime(timezone=True), nullable=True, index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True, index=True)
    first_click_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Engagement tracking
    open_count = Column(Integer, default=0, nullable=False)
    click_count = Column(Integer, default=0, nullable=False)
    last_opened_at = Column(DateTime(timezone=True), nullable=True)
    last_clicked_at = Column(DateTime(timezone=True), nullable=True)
    
    # Device and location
    device_type = Column(String(50), nullable=True)
    browser = Column(String(100), nullable=True)
    operating_system = Column(String(100), nullable=True)
    ip_address = Column(String(45), nullable=True)
    country = Column(String(2), nullable=True)
    
    # Error information
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    
    # Personalization data
    personalized_content = Column(JSONB, nullable=True)
    
    # Tracking tokens
    tracking_token = Column(String(100), unique=True, nullable=False, index=True)
    unsubscribe_token = Column(String(100), unique=True, nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    briefing = relationship("Briefing", back_populates="deliveries")
    user = relationship("User", backref="briefing_deliveries")
    
    # Indexes
    __table_args__ = (
        Index('idx_briefing_delivery_briefing_user', briefing_id, user_id),
        Index('idx_briefing_delivery_status_method', delivery_status, delivery_method),
        Index('idx_briefing_delivery_sent_at', sent_at.desc()),
        Index('idx_briefing_delivery_tracking', tracking_token),
        UniqueConstraint('briefing_id', 'user_id', 'delivery_method', name='uq_briefing_delivery'),
        CheckConstraint('open_count >= 0', name='check_open_count_non_negative'),
        CheckConstraint('click_count >= 0', name='check_click_count_non_negative'),
        CheckConstraint('retry_count >= 0', name='check_retry_count_non_negative'),
    )
    
    @classmethod
    def generate_tracking_token(cls) -> str:
        """Generate a unique tracking token."""
        import secrets
        return secrets.token_urlsafe(32)
    
    @classmethod
    def generate_unsubscribe_token(cls) -> str:
        """Generate a unique unsubscribe token."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def mark_sent(self) -> None:
        """Mark delivery as sent."""
        self.delivery_status = 'sent'
        self.sent_at = datetime.utcnow()
    
    def mark_delivered(self) -> None:
        """Mark delivery as delivered."""
        self.delivery_status = 'delivered'
        self.delivered_at = datetime.utcnow()
    
    def mark_failed(self, error_code: str = None, error_message: str = None) -> None:
        """Mark delivery as failed."""
        self.delivery_status = 'failed'
        self.error_code = error_code
        self.error_message = error_message
    
    def mark_bounced(self) -> None:
        """Mark delivery as bounced."""
        self.delivery_status = 'bounced'
    
    def record_open(self, device_info: Dict[str, Any] = None) -> None:
        """Record that the briefing was opened."""
        if not self.opened_at:
            self.opened_at = datetime.utcnow()
        
        self.open_count += 1
        self.last_opened_at = datetime.utcnow()
        
        if device_info:
            self.device_type = device_info.get('device_type')
            self.browser = device_info.get('browser')
            self.operating_system = device_info.get('operating_system')
            self.ip_address = device_info.get('ip_address')
            self.country = device_info.get('country')
    
    def record_click(self) -> None:
        """Record that a link was clicked."""
        if not self.first_click_at:
            self.first_click_at = datetime.utcnow()
        
        self.click_count += 1
        self.last_clicked_at = datetime.utcnow()
    
    def is_delivered(self) -> bool:
        """Check if delivery was successful."""
        return self.delivery_status in ['sent', 'delivered']
    
    def is_failed(self) -> bool:
        """Check if delivery failed."""
        return self.delivery_status in ['failed', 'bounced']
    
    def is_opened(self) -> bool:
        """Check if briefing was opened."""
        return self.open_count > 0
    
    def is_clicked(self) -> bool:
        """Check if any links were clicked."""
        return self.click_count > 0
    
    def get_engagement_score(self) -> float:
        """Calculate engagement score for this delivery."""
        if not self.is_delivered():
            return 0.0
        
        score = 0.0
        
        if self.is_opened():
            score += 0.4
        
        if self.is_clicked():
            score += 0.6
        
        # Additional factors
        if self.click_count > 1:
            score += min(0.2, (self.click_count - 1) * 0.05)
        
        return min(1.0, score)
    
    def should_retry(self) -> bool:
        """Check if delivery should be retried."""
        if self.delivery_status != 'failed':
            return False
        
        if self.retry_count >= 3:  # Max 3 retries
            return False
        
        # Don't retry certain error types
        if self.error_code in ['hard_bounce', 'spam', 'unsubscribed']:
            return False
        
        return True
    
    def retry(self) -> None:
        """Reset for retry."""
        self.delivery_status = 'pending'
        self.error_code = None
        self.error_message = None
        self.retry_count += 1


class BriefingTemplate(Base, TimestampMixin, SoftDeleteMixin):
    """
    Template for briefing formatting and styling.
    """
    
    __tablename__ = "briefing_template"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Template information
    name = Column(String(255), nullable=False, index=True)
    slug = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Template type
    template_type = Column(String(50), nullable=False, index=True)  # email, web, pdf, etc.
    briefing_type = Column(Enum(BriefingType), nullable=True, index=True)  # Optional: specific to briefing type
    
    # Content structure
    structure = Column(JSONB, default=list)  # Template sections and fields
    default_content = Column(JSONB, default=dict)  # Default values
    
    # Styling
    css = Column(Text, nullable=True)
    header_html = Column(Text, nullable=True)
    footer_html = Column(Text, nullable=True)
    theme = Column(String(50), nullable=True)
    
    # Responsive design
    is_responsive = Column(Boolean, default=True, nullable=False)
    mobile_optimized = Column(Boolean, default=True, nullable=False)
    
    # Compatibility
    supported_formats = Column(JSONB, default=list)
    browser_compatibility = Column(JSONB, default=list)
    
    # Versioning
    version = Column(String(50), default='1.0.0', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    deprecated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Usage tracking
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    briefings = relationship("Briefing", backref="template", primaryjoin="Briefing.template_id==cast(BriefingTemplate.id, String)")
    
    # Indexes
    __table_args__ = (
        Index('idx_briefing_template_type_active', template_type, is_active),
        Index('idx_briefing_template_slug_version', slug, version),
        UniqueConstraint('slug', 'version', name='uq_briefing_template_version'),
        CheckConstraint('usage_count >= 0', name='check_usage_count_non_negative'),
    )
    
    def is_deprecated(self) -> bool:
        """Check if template is deprecated."""
        return self.deprecated_at is not None
    
    def can_use_for_briefing_type(self, briefing_type: BriefingType) -> bool:
        """Check if template can be used for a briefing type."""
        if not self.briefing_type:
            return True  # Generic template
        return self.briefing_type == briefing_type
    
    def render_preview(self, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Render template preview with sample data."""
        return {
            'template_id': str(self.id),
            'name': self.name,
            'type': self.template_type,
            'structure': self.structure,
            'default_content': self.default_content,
            'is_responsive': self.is_responsive,
            'mobile_optimized': self.mobile_optimized,
            'sample_output': self._generate_sample_output(data),
        }
    
    def _generate_sample_output(self, data: Dict[str, Any] = None) -> str:
        """Generate sample HTML output."""
        # This would be a full template rendering implementation
        # For now, return a simple placeholder
        return f"<!-- Template: {self.name} -->\n<div class='briefing-template'>Sample content would appear here</div>"