"""
feedback.py - Feedback Model for User Feedback and Ratings

This module defines the Feedback model for collecting and managing user feedback,
ratings, reviews, and survey responses across the platform.

Key Features:
- Multi-type feedback (rating, review, bug report, feature request, survey)
- Sentiment analysis and scoring
- Categorization and tagging
- Response tracking and follow-up
- Analytics and reporting
"""

import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint, ARRAY
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.sql import func, case

from db.base import Base
from models.mixins import TimestampMixin, SoftDeleteMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.incident import Incident
    from models.comment import Comment
    from models.dataset import Dataset


class FeedbackType(Enum):
    """Types of feedback that can be submitted."""
    RATING = "rating"  # Simple star/numeric rating
    REVIEW = "review"  # Detailed text review
    BUG_REPORT = "bug_report"  # Bug/issue reporting
    FEATURE_REQUEST = "feature_request"  # Request for new features
    SURVEY_RESPONSE = "survey_response"  # Survey/questionnaire response
    GENERAL_FEEDBACK = "general_feedback"  # General comments
    SUPPORT_REQUEST = "support_request"  # Customer support request
    CONTENT_FEEDBACK = "content_feedback"  # Feedback on specific content
    SYSTEM_FEEDBACK = "system_feedback"  # System/performance feedback
    OTHER = "other"


class FeedbackStatus(Enum):
    """Status of feedback items."""
    SUBMITTED = "submitted"  # Newly submitted
    ACKNOWLEDGED = "acknowledged"  # Acknowledged by team
    IN_REVIEW = "in_review"  # Being reviewed
    PLANNED = "planned"  # Planned for implementation
    IN_PROGRESS = "in_progress"  # Being worked on
    COMPLETED = "completed"  # Implemented/resolved
    REJECTED = "rejected"  # Won't be implemented
    DUPLICATE = "duplicate"  # Duplicate of existing feedback
    ON_HOLD = "on_hold"  # Put on hold
    CLOSED = "closed"  # Closed (resolved or rejected)


class SentimentScore(Enum):
    """Sentiment classifications."""
    VERY_NEGATIVE = "very_negative"  # -2 to -1
    NEGATIVE = "negative"  # -1 to -0.25
    NEUTRAL = "neutral"  # -0.25 to 0.25
    POSITIVE = "positive"  # 0.25 to 1
    VERY_POSITIVE = "very_positive"  # 1 to 2


class PriorityLevel(Enum):
    """Priority levels for feedback processing."""
    LOW = "low"  # Nice to have
    MEDIUM = "medium"  # Should be addressed
    HIGH = "high"  # Important to address
    CRITICAL = "critical"  # Urgent/blocking issue
    EMERGENCY = "emergency"  # System-breaking issue


class Feedback(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    Feedback model for collecting user feedback and ratings.
    
    Supports multiple feedback types with rich metadata, sentiment analysis,
    and workflow tracking for follow-up and resolution.
    
    Attributes:
        id: Primary key UUID
        type: Type of feedback
        status: Current workflow status
        title: Brief title/summary
        content: Detailed feedback content
        rating: Numeric rating (1-5 stars or similar)
        sentiment_score: Calculated sentiment (-1 to 1)
        sentiment_label: Categorized sentiment
        priority: Priority level for processing
        metadata: JSON field for additional data
        tags: Array of tags for categorization
        is_anonymous: Whether feedback is submitted anonymously
        is_public: Whether feedback is visible to others
        user_id: Foreign key to submitting User
        assigned_to_id: Foreign key to User assigned to handle
        parent_id: Foreign key to parent Feedback (for follow-ups)
        source_url: URL where feedback was submitted from
        user_agent: Browser/user agent info
        ip_address: IP address of submitter
        response_count: Number of responses to this feedback
        upvote_count: Number of upvotes/agreements
        view_count: Number of views
        resolved_at: When feedback was resolved
        resolved_by_id: Who resolved it
    """
    
    __tablename__ = "feedback"
    
    # Type and classification
    type = Column(
        SQLEnum(FeedbackType),
        default=FeedbackType.GENERAL_FEEDBACK,
        nullable=False,
        index=True
    )
    status = Column(
        SQLEnum(FeedbackStatus),
        default=FeedbackStatus.SUBMITTED,
        nullable=False,
        index=True
    )
    priority = Column(
        SQLEnum(PriorityLevel),
        default=PriorityLevel.MEDIUM,
        nullable=False,
        index=True
    )
    
    # Content
    title = Column(String(255), nullable=True, index=True)
    content = Column(Text, nullable=False)
    
    # Rating and sentiment
    rating = Column(
        Float,
        nullable=True,
        doc="Numeric rating (e.g., 1-5 stars, 0-10 score)"
    )
    sentiment_score = Column(
        Float,
        nullable=True,
        index=True,
        doc="Sentiment score from -1 (negative) to 1 (positive)"
    )
    sentiment_label = Column(
        SQLEnum(SentimentScore),
        nullable=True,
        index=True
    )
    
    # Metadata
    metadata = Column(MutableDict.as_mutable(JSONB), default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    categories = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Privacy and attribution
    is_anonymous = Column(Boolean, default=False, nullable=False)
    is_public = Column(Boolean, default=False, nullable=False, index=True)
    allow_contact = Column(Boolean, default=True, nullable=False)
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    assigned_to_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    parent_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("feedback.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    resolved_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    
    # Related content references (polymorphic)
    article_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    incident_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    comment_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    dataset_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Engagement metrics
    response_count = Column(Integer, default=0, nullable=False)
    upvote_count = Column(Integer, default=0, nullable=False)
    view_count = Column(Integer, default=0, nullable=False)
    share_count = Column(Integer, default=0, nullable=False)
    
    # Technical metadata
    source_url = Column(String(2000), nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    browser_info = Column(JSONB, nullable=True)
    device_info = Column(JSONB, nullable=True)
    
    # Resolution tracking
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    estimated_resolution_date = Column(DateTime(timezone=True), nullable=True)
    actual_resolution_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id], back_populates="submitted_feedback")
    assigned_to = relationship("User", foreign_keys=[assigned_to_id])
    resolved_by = relationship("User", foreign_keys=[resolved_by_id])
    parent = relationship(
        "Feedback", 
        remote_side=[id], 
        back_populates="follow_ups",
        post_update=True
    )
    follow_ups = relationship(
        "Feedback", 
        back_populates="parent",
        cascade="all, delete-orphan"
    )
    responses = relationship(
        "FeedbackResponse",
        back_populates="feedback",
        cascade="all, delete-orphan",
        order_by="FeedbackResponse.created_at"
    )
    
    # Many-to-many relationships
    upvoted_by = relationship(
        "User",
        secondary="feedback_upvotes",
        back_populates="upvoted_feedback"
    )
    viewed_by = relationship(
        "User",
        secondary="feedback_views",
        back_populates="viewed_feedback"
    )
    
    # Direct relationships (optional, if you want explicit joins)
    article = relationship("Article", back_populates="feedback")
    incident = relationship("Incident", back_populates="feedback")
    comment = relationship("Comment", back_populates="feedback")
    dataset = relationship("Dataset", back_populates="feedback")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint(
            'rating IS NULL OR (rating >= 0 AND rating <= 5)',
            name='check_rating_range'
        ),
        CheckConstraint(
            'sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)',
            name='check_sentiment_range'
        ),
    )
    
    def __repr__(self) -> str:
        """String representation of the Feedback."""
        return f"<Feedback(id={self.id}, type={self.type.value}, status={self.status.value})>"
    
    @validates('title')
    def validate_title(self, key: str, title: Optional[str]) -> Optional[str]:
        """Validate feedback title."""
        if title is not None:
            title = title.strip()
            if len(title) > 255:
                raise ValueError("Title is too long (max 255 characters)")
        return title
    
    @validates('content')
    def validate_content(self, key: str, content: str) -> str:
        """Validate feedback content."""
        content = content.strip()
        if not content:
            raise ValueError("Feedback content cannot be empty")
        if len(content) > 10000:  # Adjust as needed
            raise ValueError("Feedback content is too long")
        return content
    
    @validates('rating')
    def validate_rating(self, key: str, rating: Optional[float]) -> Optional[float]:
        """Validate rating value."""
        if rating is not None:
            if rating < 0 or rating > 5:
                raise ValueError("Rating must be between 0 and 5")
            # Round to 1 decimal place
            rating = round(rating, 1)
        return rating
    
    @property
    def is_resolved(self) -> bool:
        """Check if feedback is resolved."""
        return self.status in [
            FeedbackStatus.COMPLETED, 
            FeedbackStatus.CLOSED,
            FeedbackStatus.REJECTED,
            FeedbackStatus.DUPLICATE
        ] and self.resolved_at is not None
    
    @property
    def requires_response(self) -> bool:
        """Check if feedback requires a response."""
        return self.status not in [
            FeedbackStatus.COMPLETED,
            FeedbackStatus.CLOSED,
            FeedbackStatus.REJECTED,
            FeedbackStatus.DUPLICATE
        ] and self.response_count == 0
    
    @property
    def is_high_priority(self) -> bool:
        """Check if feedback is high priority."""
        return self.priority in [PriorityLevel.HIGH, PriorityLevel.CRITICAL, PriorityLevel.EMERGENCY]
    
    @property
    def age_days(self) -> float:
        """Get age of feedback in days."""
        if self.created_at:
            delta = datetime.utcnow() - self.created_at
            return delta.total_seconds() / (24 * 3600)
        return 0.0
    
    @property
    def engagement_score(self) -> float:
        """Calculate engagement score based on interactions."""
        score = 0.0
        score += self.upvote_count * 2
        score += self.response_count * 3
        score += self.view_count * 0.1
        score += self.share_count * 5
        
        # Weight by priority
        priority_weights = {
            PriorityLevel.LOW: 0.5,
            PriorityLevel.MEDIUM: 1.0,
            PriorityLevel.HIGH: 1.5,
            PriorityLevel.CRITICAL: 2.0,
            PriorityLevel.EMERGENCY: 3.0
        }
        score *= priority_weights.get(self.priority, 1.0)
        
        return score
    
    def calculate_sentiment(self, score: Optional[float] = None) -> None:
        """
        Calculate and set sentiment label based on score.
        
        Args:
            score: Optional sentiment score (will use existing if None)
        """
        if score is None:
            score = self.sentiment_score
        
        if score is None:
            self.sentiment_label = None
            return
        
        if score <= -1:
            self.sentiment_label = SentimentScore.VERY_NEGATIVE
        elif score <= -0.25:
            self.sentiment_label = SentimentScore.NEGATIVE
        elif score <= 0.25:
            self.sentiment_label = SentimentScore.NEUTRAL
        elif score <= 1:
            self.sentiment_label = SentimentScore.POSITIVE
        else:
            self.sentiment_label = SentimentScore.VERY_POSITIVE
    
    def update_status(self, 
                     new_status: FeedbackStatus, 
                     updated_by: Optional['User'] = None,
                     notes: Optional[str] = None) -> FeedbackStatus:
        """
        Update feedback status with audit trail.
        
        Args:
            new_status: New status to set
            updated_by: User making the change
            notes: Optional notes about the status change
            
        Returns:
            Previous status
        """
        old_status = self.status
        self.status = new_status
        
        # Update metadata
        if 'status_history' not in self.metadata:
            self.metadata['status_history'] = []
        
        self.metadata['status_history'].append({
            'from_status': old_status.value,
            'to_status': new_status.value,
            'changed_at': datetime.utcnow().isoformat(),
            'changed_by': str(updated_by.id) if updated_by else None,
            'notes': notes
        })
        
        # Set resolved timestamp if moving to resolved status
        if new_status in [FeedbackStatus.COMPLETED, FeedbackStatus.CLOSED] and not self.resolved_at:
            self.resolved_at = datetime.utcnow()
            if updated_by:
                self.resolved_by_id = updated_by.id
        
        return old_status
    
    def assign_to(self, user: 'User', assigned_by: Optional['User'] = None) -> None:
        """Assign feedback to a user for handling."""
        self.assigned_to_id = user.id
        
        # Update metadata
        if 'assignment_history' not in self.metadata:
            self.metadata['assignment_history'] = []
        
        self.metadata['assignment_history'].append({
            'assigned_to': str(user.id),
            'assigned_by': str(assigned_by.id) if assigned_by else None,
            'assigned_at': datetime.utcnow().isoformat()
        })
    
    def add_response(self, response: 'FeedbackResponse') -> None:
        """Add a response to this feedback."""
        self.response_count += 1
        
        # Auto-acknowledge if this is the first staff response
        if (response.is_staff_response and 
            self.status == FeedbackStatus.SUBMITTED and 
            self.response_count == 1):
            self.update_status(FeedbackStatus.ACKNOWLEDGED)
    
    def add_upvote(self, user: 'User') -> bool:
        """
        Add an upvote from a user.
        
        Args:
            user: User upvoting
            
        Returns:
            True if upvote was added, False if already upvoted
        """
        if user not in self.upvoted_by:
            self.upvoted_by.append(user)
            self.upvote_count += 1
            return True
        return False
    
    def remove_upvote(self, user: 'User') -> bool:
        """
        Remove an upvote from a user.
        
        Args:
            user: User removing upvote
            
        Returns:
            True if upvote was removed, False if not found
        """
        if user in self.upvoted_by:
            self.upvoted_by.remove(user)
            self.upvote_count = max(0, self.upvote_count - 1)
            return True
        return False
    
    def record_view(self, user: Optional['User'] = None) -> None:
        """Record a view of this feedback."""
        self.view_count += 1
        if user and user not in self.viewed_by:
            self.viewed_by.append(user)
    
    def to_dict(self, include_responses: bool = False, include_user: bool = True) -> Dict[str, Any]:
        """
        Convert feedback to dictionary for API responses.
        
        Args:
            include_responses: Whether to include responses
            include_user: Whether to include user information
            
        Returns:
            Dictionary representation of the feedback
        """
        result = {
            "id": str(self.id),
            "type": self.type.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "title": self.title,
            "content": self.content,
            "rating": self.rating,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label.value if self.sentiment_label else None,
            "tags": self.tags,
            "categories": self.categories,
            "is_anonymous": self.is_anonymous,
            "is_public": self.is_public,
            "is_resolved": self.is_resolved,
            "requires_response": self.requires_response,
            "is_high_priority": self.is_high_priority,
            "age_days": round(self.age_days, 2),
            "engagement_score": round(self.engagement_score, 2),
            "response_count": self.response_count,
            "upvote_count": self.upvote_count,
            "view_count": self.view_count,
            "share_count": self.share_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "source_url": self.source_url,
            "user_id": str(self.user_id) if self.user_id else None,
            "assigned_to_id": str(self.assigned_to_id) if self.assigned_to_id else None,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "resolved_by_id": str(self.resolved_by_id) if self.resolved_by_id else None,
            "article_id": str(self.article_id) if self.article_id else None,
            "incident_id": str(self.incident_id) if self.incident_id else None,
            "comment_id": str(self.comment_id) if self.comment_id else None,
            "dataset_id": str(self.dataset_id) if self.dataset_id else None,
            "metadata": self.metadata,
            "is_deleted": self.is_deleted,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None
        }
        
        if include_user and self.user and not self.is_anonymous:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username,
                "display_name": getattr(self.user, 'display_name', None)
            }
        
        if include_user and self.assigned_to:
            result["assigned_to"] = {
                "id": str(self.assigned_to.id),
                "username": self.assigned_to.username,
                "display_name": getattr(self.assigned_to, 'display_name', None)
            }
        
        if include_responses and self.responses:
            result["responses"] = [
                response.to_dict(include_user=include_user)
                for response in self.responses
                if not response.is_deleted
            ]
        
        return result
    
    @classmethod
    def create(
        cls,
        content: str,
        feedback_type: FeedbackType = FeedbackType.GENERAL_FEEDBACK,
        user_id: Optional[uuid.UUID] = None,
        title: Optional[str] = None,
        rating: Optional[float] = None,
        sentiment_score: Optional[float] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        is_anonymous: bool = False,
        is_public: bool = False,
        priority: PriorityLevel = PriorityLevel.MEDIUM,
        article_id: Optional[uuid.UUID] = None,
        incident_id: Optional[uuid.UUID] = None,
        comment_id: Optional[uuid.UUID] = None,
        dataset_id: Optional[uuid.UUID] = None,
        parent_id: Optional[uuid.UUID] = None,
        source_url: Optional[str] = None,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'Feedback':
        """
        Factory method to create a new feedback.
        
        Args:
            content: Feedback content
            feedback_type: Type of feedback
            user_id: Submitting user ID
            title: Optional title
            rating: Optional numeric rating
            sentiment_score: Optional sentiment score
            tags: Optional tags
            categories: Optional categories
            is_anonymous: Whether feedback is anonymous
            is_public: Whether feedback is public
            priority: Priority level
            article_id: Related article ID
            incident_id: Related incident ID
            comment_id: Related comment ID
            dataset_id: Related dataset ID
            parent_id: Parent feedback ID (for follow-ups)
            source_url: Source URL
            user_agent: User agent string
            ip_address: IP address
            metadata: Additional metadata
            
        Returns:
            A new Feedback instance
        """
        feedback = cls(
            type=feedback_type,
            title=title,
            content=content,
            rating=rating,
            sentiment_score=sentiment_score,
            tags=tags or [],
            categories=categories or [],
            is_anonymous=is_anonymous,
            is_public=is_public,
            priority=priority,
            user_id=user_id,
            parent_id=parent_id,
            article_id=article_id,
            incident_id=incident_id,
            comment_id=comment_id,
            dataset_id=dataset_id,
            source_url=source_url,
            user_agent=user_agent,
            ip_address=ip_address,
            metadata=metadata or {},
            status=FeedbackStatus.SUBMITTED
        )
        
        # Calculate sentiment label if score provided
        if sentiment_score is not None:
            feedback.calculate_sentiment(sentiment_score)
        
        return feedback


class FeedbackResponse(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    Response model for feedback items.
    
    Allows staff/users to respond to feedback, creating a conversation thread.
    """
    
    __tablename__ = "feedback_responses"
    
    feedback_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("feedback.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    parent_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("feedback_responses.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    content = Column(Text, nullable=False)
    is_internal = Column(Boolean, default=False, nullable=False, 
                        doc="Whether response is internal-only")
    is_staff_response = Column(Boolean, default=False, nullable=False, 
                              doc="Whether response is from staff/team")
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    feedback = relationship("Feedback", back_populates="responses")
    user = relationship("User")
    parent = relationship(
        "FeedbackResponse", 
        remote_side=[id], 
        back_populates="replies"
    )
    replies = relationship(
        "FeedbackResponse", 
        back_populates="parent",
        cascade="all, delete-orphan"
    )
    
    # Engagement
    helpful_count = Column(Integer, default=0, nullable=False)
    not_helpful_count = Column(Integer, default=0, nullable=False)
    
    def __repr__(self) -> str:
        return f"<FeedbackResponse(id={self.id}, feedback={self.feedback_id})>"
    
    @property
    def is_reply(self) -> bool:
        """Check if this is a reply to another response."""
        return self.parent_id is not None
    
    @property
    def helpful_score(self) -> float:
        """Calculate helpfulness score."""
        total = self.helpful_count + self.not_helpful_count
        if total == 0:
            return 0.0
        return self.helpful_count / total
    
    def to_dict(self, include_user: bool = True) -> Dict[str, Any]:
        """Convert response to dictionary."""
        result = {
            "id": str(self.id),
            "feedback_id": str(self.feedback_id),
            "content": self.content,
            "is_internal": self.is_internal,
            "is_staff_response": self.is_staff_response,
            "is_reply": self.is_reply,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "helpful_count": self.helpful_count,
            "not_helpful_count": self.not_helpful_count,
            "helpful_score": round(self.helpful_score, 3),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_deleted": self.is_deleted,
            "metadata": self.metadata
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username,
                "display_name": getattr(self.user, 'display_name', None),
                "is_staff": self.is_staff_response
            }
        
        if self.replies:
            result["replies"] = [
                reply.to_dict(include_user=include_user)
                for reply in self.replies
                if not reply.is_deleted
            ]
        
        return result


class FeedbackSurvey(Base, UUIDMixin, TimestampMixin):
    """
    Survey/questionnaire definition for collecting structured feedback.
    """
    
    __tablename__ = "feedback_surveys"
    
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_public = Column(Boolean, default=False, nullable=False)
    
    # Survey configuration
    questions = Column(JSONB, nullable=False, doc="Survey questions in JSON format")
    settings = Column(JSONB, default=dict, nullable=False)
    
    # Access control
    required_role = Column(String(50), nullable=True)
    access_code = Column(String(100), nullable=True)
    
    # Scheduling
    starts_at = Column(DateTime(timezone=True), nullable=True)
    ends_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metrics
    response_count = Column(Integer, default=0, nullable=False)
    average_completion_time = Column(Float, nullable=True)
    
    # Relationships
    responses = relationship("FeedbackSurveyResponse", back_populates="survey", cascade="all, delete-orphan")
    
    def __repr__(self) -> str:
        return f"<FeedbackSurvey(id={self.id}, title='{self.title}')>"
    
    @property
    def is_available(self) -> bool:
        """Check if survey is currently available."""
        if not self.is_active:
            return False
        
        now = datetime.utcnow()
        if self.starts_at and now < self.starts_at:
            return False
        if self.ends_at and now > self.ends_at:
            return False
        
        return True


class FeedbackSurveyResponse(Base, UUIDMixin, TimestampMixin):
    """
    Individual survey responses.
    """
    
    __tablename__ = "feedback_survey_responses"
    
    survey_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("feedback_surveys.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    
    # Response data
    answers = Column(JSONB, nullable=False, doc="Survey answers in JSON format")
    completion_time_seconds = Column(Float, nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    survey = relationship("FeedbackSurvey", back_populates="responses")
    user = relationship("User")
    
    def __repr__(self) -> str:
        return f"<FeedbackSurveyResponse(id={self.id}, survey={self.survey_id})>"


# Association tables for many-to-many relationships
from sqlalchemy import Table, Column

feedback_upvotes = Table(
    'feedback_upvotes',
    Base.metadata,
    Column('feedback_id', UUID(as_uuid=True), ForeignKey('feedback.id', ondelete='CASCADE'), primary_key=True),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('created_at', DateTime(timezone=True), server_default=func.now())
)

feedback_views = Table(
    'feedback_views',
    Base.metadata,
    Column('feedback_id', UUID(as_uuid=True), ForeignKey('feedback.id', ondelete='CASCADE'), primary_key=True),
    Column('user_id', UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), primary_key=True),
    Column('viewed_at', DateTime(timezone=True), server_default=func.now()),
    Column('view_count', Integer, default=1)
)


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the Feedback model.
"""

from pydantic import BaseModel, Field, validator, conlist
from typing import Optional, List, Dict, Any
from datetime import datetime


class FeedbackBase(BaseModel):
    """Base schema for feedback operations."""
    type: FeedbackType = Field(default=FeedbackType.GENERAL_FEEDBACK)
    title: Optional[str] = Field(None, max_length=255)
    content: str = Field(..., min_length=1, max_length=10000)
    rating: Optional[float] = Field(None, ge=0, le=5)
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    is_anonymous: bool = Field(default=False)
    is_public: bool = Field(default=False)
    priority: PriorityLevel = Field(default=PriorityLevel.MEDIUM)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Feedback content cannot be empty')
        return v.strip()


class FeedbackCreate(FeedbackBase):
    """Schema for creating new feedback."""
    article_id: Optional[str] = None
    incident_id: Optional[str] = None
    comment_id: Optional[str] = None
    dataset_id: Optional[str] = None
    parent_id: Optional[str] = None
    source_url: Optional[str] = None


class FeedbackUpdate(BaseModel):
    """Schema for updating existing feedback."""
    status: Optional[FeedbackStatus] = None
    priority: Optional[PriorityLevel] = None
    assigned_to_id: Optional[str] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    is_public: Optional[bool] = None
    resolution_notes: Optional[str] = None


class FeedbackResponseCreate(BaseModel):
    """Schema for creating feedback responses."""
    content: str = Field(..., min_length=1, max_length=5000)
    is_internal: bool = Field(default=False)
    parent_id: Optional[str] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Response content cannot be empty')
        return v.strip()


class FeedbackInDBBase(FeedbackBase):
    """Base schema for feedback in database."""
    id: str
    status: FeedbackStatus
    sentiment_score: Optional[float]
    sentiment_label: Optional[SentimentScore]
    response_count: int
    upvote_count: int
    view_count: int
    engagement_score: float
    user_id: Optional[str]
    assigned_to_id: Optional[str]
    resolved_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    is_resolved: bool
    requires_response: bool
    
    class Config:
        from_attributes = True


class Feedback(FeedbackInDBBase):
    """Schema for feedback API responses."""
    user: Optional[Dict[str, Any]] = None
    assigned_to: Optional[Dict[str, Any]] = None
    responses: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        from_attributes = True


class FeedbackResponseSchema(BaseModel):
    """Schema for feedback response API responses."""
    id: str
    content: str
    is_internal: bool
    is_staff_response: bool
    helpful_count: int
    not_helpful_count: int
    helpful_score: float
    user: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class FeedbackStats(BaseModel):
    """Schema for feedback statistics."""
    total: int
    by_type: Dict[str, int]
    by_status: Dict[str, int]
    by_priority: Dict[str, int]
    by_sentiment: Dict[str, int]
    average_rating: Optional[float]
    response_rate: float
    resolution_rate: float
    average_response_time_hours: Optional[float]
    
    class Config:
        from_attributes = True


class FeedbackSearchRequest(BaseModel):
    """Schema for feedback search requests."""
    query: Optional[str] = None
    type: Optional[FeedbackType] = None
    status: Optional[FeedbackStatus] = None
    priority: Optional[PriorityLevel] = None
    sentiment: Optional[SentimentScore] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    assigned_to_id: Optional[str] = None
    user_id: Optional[str] = None
    is_resolved: Optional[bool] = None
    is_public: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = Field(default="created_at", pattern="^(created_at|updated_at|rating|engagement_score)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=50, ge=1, le=200)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True