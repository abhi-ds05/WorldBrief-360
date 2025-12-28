"""
vote.py - Voting, Rating, and Feedback Model

This module defines models for voting, rating, and feedback systems.
This includes:
- Upvote/downvote systems
- Star ratings and reviews
- Content quality assessment
- User reputation systems
- Moderation votes
- Polls and surveys
- Sentiment analysis
- Feedback aggregation

Key Features:
- Multiple vote types (upvote, downvote, star, emoji, etc.)
- Weighted voting based on user reputation
- Vote fraud detection and prevention
- Real-time vote aggregation
- Anonymous and authenticated voting
- Vote expiry and recalibration
- Multi-dimensional ratings
- Feedback moderation
"""

import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, TYPE_CHECKING, Union
from enum import Enum
from decimal import Decimal
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, CheckConstraint,
    Index, Table, UniqueConstraint, Numeric, BigInteger
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.sql import func, expression
from sqlalchemy.ext.hybrid import hybrid_property

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.incident import Incident
    from models.comment import Comment
    from models.organization import Organization
    from models.topic import Topic


class VoteType(Enum):
    """Types of votes."""
    UPVOTE = "upvote"                # Positive vote
    DOWNVOTE = "downvote"            # Negative vote
    STAR = "star"                    # Star rating (1-5)
    EMOJI = "emoji"                  # Emoji reaction
    LIKERT = "likert"                # Likert scale (1-5, 1-7, etc.)
    THUMBS = "thumbs"                # Thumbs up/down
    HEART = "heart"                  # Heart/love
    FLAG = "flag"                    # Flag/report
    VERIFY = "verify"                # Verification vote
    TRUST = "trust"                  # Trust rating
    HELPFUL = "helpful"              # Helpful rating
    FUNNY = "funny"                  # Funny rating
    SAD = "sad"                      # Sad reaction
    ANGRY = "angry"                  # Angry reaction
    CUSTOM = "custom"                # Custom vote type


class ContentType(Enum):
    """Types of content that can be voted on."""
    ARTICLE = "article"              # Articles
    INCIDENT = "incident"            # Incidents
    COMMENT = "comment"              # Comments
    USER = "user"                    # Users
    ORGANIZATION = "organization"    # Organizations
    TOPIC = "topic"                  # Topics
    REVIEW = "review"                # Reviews
    ANSWER = "answer"                # Answers (Q&A)
    MEDIA = "media"                  # Media files
    POLL = "poll"                    # Polls
    OTHER = "other"                  # Other content types


class VoteStatus(Enum):
    """Vote status."""
    ACTIVE = "active"                # Active vote
    DELETED = "deleted"              # Vote deleted
    REVOKED = "revoked"              # Vote revoked (e.g., user banned)
    EXPIRED = "expired"              # Vote expired
    INVALID = "invalid"              # Invalid vote (fraud detection)
    PENDING = "pending"              # Pending moderation


class VoteWeightMethod(Enum):
    """Methods for calculating vote weight."""
    FIXED = "fixed"                  # Fixed weight (1.0)
    USER_REPUTATION = "user_reputation"  # Based on user reputation
    SENIORITY = "seniority"          # Based on user account age
    ACTIVITY = "activity"            # Based on user activity level
    CUSTOM = "custom"                # Custom weight calculation


class Vote(Base, UUIDMixin, TimestampMixin):
    """
    Vote model for user feedback and ratings.
    
    This model represents individual votes, ratings, and reactions
    to various types of content across the platform.
    
    Attributes:
        id: Primary key UUID
        user_id: User who cast the vote
        content_type: Type of content being voted on
        content_id: ID of content being voted on
        vote_type: Type of vote
        value: Vote value (e.g., 1 for upvote, -1 for downvote, 1-5 for stars)
        weight: Weight of this vote (for weighted averaging)
        weight_method: Method used to calculate weight
        status: Vote status
        is_anonymous: Whether vote is anonymous
        ip_address: IP address of voter
        user_agent: User agent string
        metadata: Additional metadata
        expires_at: When vote expires (for temporary votes)
        moderated_by: User who moderated this vote
        moderated_at: When vote was moderated
        moderation_reason: Reason for moderation
        context: Additional context about the vote
        tags: Categorization tags
    """
    
    __tablename__ = "votes"
    
    # Voter
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,  # Nullable for anonymous votes
        index=True
    )
    
    # Content being voted on
    content_type = Column(SQLEnum(ContentType), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    
    # Vote details
    vote_type = Column(SQLEnum(VoteType), nullable=False, index=True)
    value = Column(Numeric(5, 2), nullable=False)  # Can be decimal for weighted votes
    weight = Column(Float, default=1.0, nullable=False)
    weight_method = Column(SQLEnum(VoteWeightMethod), default=VoteWeightMethod.FIXED, nullable=False)
    
    # Status
    status = Column(SQLEnum(VoteStatus), default=VoteStatus.ACTIVE, nullable=False, index=True)
    is_anonymous = Column(Boolean, default=False, nullable=False, index=True)
    
    # Technical details
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Moderation
    moderated_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    moderated_at = Column(DateTime(timezone=True), nullable=True)
    moderation_reason = Column(Text, nullable=True)
    
    # Context
    context = Column(JSONB, nullable=True)  # Additional context like section, page, etc.
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    moderator = relationship("User", foreign_keys=[moderated_by])
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'content_type', 'content_id', 'vote_type', name='uq_user_content_vote'),
        CheckConstraint('weight >= 0', name='check_weight_non_negative'),
        CheckConstraint(
            'value >= -5 AND value <= 5',  # Reasonable range for most vote types
            name='check_value_range'
        ),
        Index('ix_votes_content', 'content_type', 'content_id', 'vote_type'),
        Index('ix_votes_status_expires', 'status', 'expires_at'),
        Index('ix_votes_anonymous', 'is_anonymous', 'content_type', 'content_id'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        user_str = f"user={self.user_id}" if self.user_id else "anonymous"
        return f"<Vote(id={self.id}, {user_str}, content={self.content_type.value}:{self.content_id}, type={self.vote_type.value}, value={self.value})>"
    
    @property
    def composite_content_id(self) -> str:
        """Get composite content identifier."""
        return f"{self.content_type.value}:{self.content_id}"
    
    @property
    def is_active(self) -> bool:
        """Check if vote is active."""
        return self.status == VoteStatus.ACTIVE and not self.is_expired
    
    @property
    def is_expired(self) -> bool:
        """Check if vote has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_moderated(self) -> bool:
        """Check if vote has been moderated."""
        return self.moderated_at is not None
    
    @property
    def weighted_value(self) -> Decimal:
        """Get weighted vote value."""
        return Decimal(str(self.value)) * Decimal(str(self.weight))
    
    @property
    def is_positive(self) -> bool:
        """Check if vote is positive."""
        return self.value > 0
    
    @property
    def is_negative(self) -> bool:
        """Check if vote is negative."""
        return self.value < 0
    
    @property
    def is_neutral(self) -> bool:
        """Check if vote is neutral."""
        return self.value == 0
    
    @property
    def age_days(self) -> float:
        """Get age of vote in days."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() / (24 * 3600)
    
    @validates('value')
    def validate_value(self, key: str, value: Union[int, float, Decimal]) -> Decimal:
        """Validate vote value based on vote type."""
        value = Decimal(str(value))
        
        if self.vote_type == VoteType.UPVOTE:
            if value != 1:
                raise ValueError("Upvote must have value 1")
        elif self.vote_type == VoteType.DOWNVOTE:
            if value != -1:
                raise ValueError("Downvote must have value -1")
        elif self.vote_type == VoteType.STAR:
            if value < 1 or value > 5:
                raise ValueError("Star rating must be between 1 and 5")
        elif self.vote_type == VoteType.LIKERT:
            if value < 1 or value > 7:
                raise ValueError("Likert scale must be between 1 and 7")
        elif self.vote_type == VoteType.THUMBS:
            if value not in [1, -1]:
                raise ValueError("Thumbs must be 1 (up) or -1 (down)")
        elif self.vote_type == VoteType.HEART:
            if value != 1:
                raise ValueError("Heart must have value 1")
        elif self.vote_type == VoteType.FLAG:
            if value != 1:
                raise ValueError("Flag must have value 1")
        elif self.vote_type == VoteType.VERIFY:
            if value not in [1, -1]:
                raise ValueError("Verify must be 1 (verify) or -1 (dispute)")
        elif self.vote_type == VoteType.TRUST:
            if value < 0 or value > 10:
                raise ValueError("Trust rating must be between 0 and 10")
        
        return value
    
    def calculate_weight(self, user_reputation: Optional[float] = None, user_seniority_days: Optional[int] = None) -> float:
        """Calculate vote weight based on weight method."""
        if self.weight_method == VoteWeightMethod.FIXED:
            return 1.0
        elif self.weight_method == VoteWeightMethod.USER_REPUTATION and user_reputation is not None:
            # Normalize reputation to 0.5-2.0 range
            return max(0.5, min(2.0, 1.0 + (user_reputation / 100)))
        elif self.weight_method == VoteWeightMethod.SENIORITY and user_seniority_days is not None:
            # Seniority weight: 1.0 for new users, up to 1.5 for users > 1 year
            seniority_years = user_seniority_days / 365
            return min(1.5, 1.0 + (seniority_years * 0.5))
        elif self.weight_method == VoteWeightMethod.ACTIVITY:
            # Default for activity-based weight
            return 1.0
        else:
            return 1.0
    
    def revoke(self, reason: Optional[str] = None, moderator_id: Optional[uuid.UUID] = None) -> None:
        """Revoke the vote."""
        self.status = VoteStatus.REVOKED
        if moderator_id:
            self.moderated_by = moderator_id
            self.moderated_at = datetime.utcnow()
        if reason:
            self.moderation_reason = reason
    
    def delete(self, soft_delete: bool = True) -> None:
        """Delete the vote."""
        if soft_delete:
            self.status = VoteStatus.DELETED
        else:
            # In actual implementation, would delete from database
            pass
    
    def expire(self) -> None:
        """Expire the vote."""
        self.status = VoteStatus.EXPIRED
        self.expires_at = datetime.utcnow()
    
    def set_expiration(self, days: int = 30) -> None:
        """Set expiration date for the vote."""
        self.expires_at = datetime.utcnow() + timedelta(days=days)
    
    def to_dict(self, include_user: bool = False, include_moderator: bool = False) -> Dict[str, Any]:
        """Convert vote to dictionary."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "content_type": self.content_type.value,
            "content_id": str(self.content_id),
            "composite_content_id": self.composite_content_id,
            "vote_type": self.vote_type.value,
            "value": float(self.value),
            "weight": self.weight,
            "weighted_value": float(self.weighted_value),
            "weight_method": self.weight_method.value,
            "status": self.status.value,
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "is_positive": self.is_positive,
            "is_negative": self.is_negative,
            "is_neutral": self.is_neutral,
            "is_anonymous": self.is_anonymous,
            "is_moderated": self.is_moderated,
            "age_days": round(self.age_days, 2),
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "moderated_by": str(self.moderated_by) if self.moderated_by else None,
            "moderated_at": self.moderated_at.isoformat() if self.moderated_at else None,
            "moderation_reason": self.moderation_reason,
            "context": self.context,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username,
                "reputation": getattr(self.user, 'reputation_score', None)
            }
        
        if include_moderator and self.moderator:
            result["moderator"] = {
                "id": str(self.moderator.id),
                "username": self.moderator.username
            }
        
        return result
    
    @classmethod
    def create_upvote(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        weight: float = 1.0,
        is_anonymous: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Vote':
        """Create an upvote."""
        return cls.create(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=VoteType.UPVOTE,
            value=1,
            weight=weight,
            is_anonymous=is_anonymous,
            context=context,
            **kwargs
        )
    
    @classmethod
    def create_downvote(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        weight: float = 1.0,
        is_anonymous: bool = False,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Vote':
        """Create a downvote."""
        return cls.create(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=VoteType.DOWNVOTE,
            value=-1,
            weight=weight,
            is_anonymous=is_anonymous,
            context=context,
            **kwargs
        )
    
    @classmethod
    def create_star_rating(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        rating: int,
        user_id: Optional[uuid.UUID] = None,
        weight: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Vote':
        """Create a star rating."""
        if rating < 1 or rating > 5:
            raise ValueError("Star rating must be between 1 and 5")
        
        return cls.create(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=VoteType.STAR,
            value=rating,
            weight=weight,
            context=context,
            **kwargs
        )
    
    @classmethod
    def create_heart(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        weight: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Vote':
        """Create a heart/love reaction."""
        return cls.create(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=VoteType.HEART,
            value=1,
            weight=weight,
            context=context,
            **kwargs
        )
    
    @classmethod
    def create_flag(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        user_id: Optional[uuid.UUID] = None,
        reason: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> 'Vote':
        """Create a flag/report."""
        metadata = kwargs.get('metadata', {})
        if reason:
            metadata['flag_reason'] = reason
        
        return cls.create(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=VoteType.FLAG,
            value=1,
            context=context,
            metadata=metadata,
            **kwargs
        )
    
    @classmethod
    def create(
        cls,
        content_type: ContentType,
        content_id: uuid.UUID,
        vote_type: VoteType,
        value: Union[int, float, Decimal],
        user_id: Optional[uuid.UUID] = None,
        weight: float = 1.0,
        weight_method: VoteWeightMethod = VoteWeightMethod.FIXED,
        is_anonymous: bool = False,
        status: VoteStatus = VoteStatus.ACTIVE,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> 'Vote':
        """
        Factory method to create a new vote.
        
        Args:
            content_type: Type of content being voted on
            content_id: ID of content being voted on
            vote_type: Type of vote
            value: Vote value
            user_id: User who cast the vote
            weight: Vote weight
            weight_method: Method used to calculate weight
            is_anonymous: Whether vote is anonymous
            status: Vote status
            ip_address: IP address of voter
            user_agent: User agent string
            session_id: Session ID
            metadata: Additional metadata
            expires_at: When vote expires
            context: Additional context
            tags: Categorization tags
            **kwargs: Additional arguments
            
        Returns:
            A new Vote instance
        """
        vote = cls(
            user_id=user_id,
            content_type=content_type,
            content_id=content_id,
            vote_type=vote_type,
            value=value,
            weight=weight,
            weight_method=weight_method,
            is_anonymous=is_anonymous,
            status=status,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            metadata=metadata or {},
            expires_at=expires_at,
            context=context,
            tags=tags or [],
            **kwargs
        )
        
        return vote


class VoteAggregate(Base, UUIDMixin, TimestampMixin):
    """
    Vote aggregate model for caching vote statistics.
    
    This model caches aggregated vote statistics for content
    to avoid expensive real-time calculations.
    
    Attributes:
        id: Primary key UUID
        content_type: Type of content
        content_id: ID of content
        vote_type: Type of votes being aggregated
        total_votes: Total number of votes
        total_value: Sum of all vote values
        total_weight: Sum of all vote weights
        weighted_average: Weighted average value
        simple_average: Simple average value
        upvotes: Number of upvotes (if applicable)
        downvotes: Number of downvotes (if applicable)
        star_1: Number of 1-star ratings
        star_2: Number of 2-star ratings
        star_3: Number of 3-star ratings
        star_4: Number of 4-star ratings
        star_5: Number of 5-star ratings
        last_calculated_at: When aggregate was last calculated
        recalc_needed: Whether recalculation is needed
        metadata: Additional metadata
    """
    
    __tablename__ = "vote_aggregates"
    
    # Content
    content_type = Column(SQLEnum(ContentType), nullable=False, index=True)
    content_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    vote_type = Column(SQLEnum(VoteType), nullable=False, index=True)
    
    # Aggregate statistics
    total_votes = Column(Integer, default=0, nullable=False)
    total_value = Column(Numeric(15, 4), default=0, nullable=False)
    total_weight = Column(Float, default=0.0, nullable=False)
    weighted_average = Column(Numeric(5, 3), nullable=True)
    simple_average = Column(Numeric(5, 3), nullable=True)
    
    # Vote type specific counts
    upvotes = Column(Integer, default=0, nullable=False)
    downvotes = Column(Integer, default=0, nullable=False)
    
    # Star rating distribution
    star_1 = Column(Integer, default=0, nullable=False)
    star_2 = Column(Integer, default=0, nullable=False)
    star_3 = Column(Integer, default=0, nullable=False)
    star_4 = Column(Integer, default=0, nullable=False)
    star_5 = Column(Integer, default=0, nullable=False)
    
    # Emoji counts
    heart_count = Column(Integer, default=0, nullable=False)
    helpful_count = Column(Integer, default=0, nullable=False)
    funny_count = Column(Integer, default=0, nullable=False)
    sad_count = Column(Integer, default=0, nullable=False)
    angry_count = Column(Integer, default=0, nullable=False)
    
    # Status
    last_calculated_at = Column(DateTime(timezone=True), nullable=False)
    recalc_needed = Column(Boolean, default=False, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('content_type', 'content_id', 'vote_type', name='uq_content_vote_aggregate'),
        CheckConstraint('total_votes >= 0', name='check_total_votes_non_negative'),
        CheckConstraint('upvotes >= 0', name='check_upvotes_non_negative'),
        CheckConstraint('downvotes >= 0', name='check_downvotes_non_negative'),
        CheckConstraint('star_1 >= 0', name='check_star_1_non_negative'),
        CheckConstraint('star_2 >= 0', name='check_star_2_non_negative'),
        CheckConstraint('star_3 >= 0', name='check_star_3_non_negative'),
        CheckConstraint('star_4 >= 0', name='check_star_4_non_negative'),
        CheckConstraint('star_5 >= 0', name='check_star_5_non_negative'),
        Index('ix_vote_aggregates_weighted_avg', 'vote_type', 'weighted_average'),
        Index('ix_vote_aggregates_total_votes', 'vote_type', 'total_votes'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VoteAggregate(id={self.id}, content={self.content_type.value}:{self.content_id}, type={self.vote_type.value}, total={self.total_votes})>"
    
    @property
    def composite_content_id(self) -> str:
        """Get composite content identifier."""
        return f"{self.content_type.value}:{self.content_id}"
    
    @property
    def net_score(self) -> int:
        """Get net score (upvotes - downvotes)."""
        return self.upvotes - self.downvotes
    
    @property
    def approval_ratio(self) -> Optional[float]:
        """Get approval ratio (upvotes / total votes)."""
        total = self.upvotes + self.downvotes
        if total == 0:
            return None
        return self.upvotes / total
    
    @property
    def star_distribution(self) -> Dict[int, int]:
        """Get star rating distribution."""
        return {
            1: self.star_1,
            2: self.star_2,
            3: self.star_3,
            4: self.star_4,
            5: self.star_5
        }
    
    @property
    def total_stars(self) -> int:
        """Get total star ratings."""
        return self.star_1 + self.star_2 + self.star_3 + self.star_4 + self.star_5
    
    @property
    def star_average(self) -> Optional[float]:
        """Get average star rating."""
        if self.total_stars == 0:
            return None
        
        weighted_sum = (
            self.star_1 * 1 +
            self.star_2 * 2 +
            self.star_3 * 3 +
            self.star_4 * 4 +
            self.star_5 * 5
        )
        return weighted_sum / self.total_stars
    
    @property
    def emoji_counts(self) -> Dict[str, int]:
        """Get emoji reaction counts."""
        return {
            "heart": self.heart_count,
            "helpful": self.helpful_count,
            "funny": self.funny_count,
            "sad": self.sad_count,
            "angry": self.angry_count
        }
    
    def update_from_votes(self, votes: List['Vote']) -> None:
        """Update aggregate statistics from list of votes."""
        # Reset counters
        self.total_votes = 0
        self.total_value = Decimal('0')
        self.total_weight = 0.0
        self.upvotes = 0
        self.downvotes = 0
        self.star_1 = self.star_2 = self.star_3 = self.star_4 = self.star_5 = 0
        self.heart_count = self.helpful_count = self.funny_count = self.sad_count = self.angry_count = 0
        
        # Process votes
        weighted_sum = Decimal('0')
        for vote in votes:
            if not vote.is_active:
                continue
            
            self.total_votes += 1
            self.total_value += Decimal(str(vote.value))
            self.total_weight += vote.weight
            weighted_sum += vote.weighted_value
            
            # Count by vote type
            if vote.vote_type == VoteType.UPVOTE and vote.value > 0:
                self.upvotes += 1
            elif vote.vote_type == VoteType.DOWNVOTE and vote.value < 0:
                self.downvotes += 1
            elif vote.vote_type == VoteType.STAR:
                star_value = int(vote.value)
                if star_value == 1:
                    self.star_1 += 1
                elif star_value == 2:
                    self.star_2 += 1
                elif star_value == 3:
                    self.star_3 += 1
                elif star_value == 4:
                    self.star_4 += 1
                elif star_value == 5:
                    self.star_5 += 1
            elif vote.vote_type == VoteType.HEART:
                self.heart_count += 1
            elif vote.vote_type == VoteType.HELPFUL:
                self.helpful_count += 1
            elif vote.vote_type == VoteType.FUNNY:
                self.funny_count += 1
            elif vote.vote_type == VoteType.SAD:
                self.sad_count += 1
            elif vote.vote_type == VoteType.ANGRY:
                self.angry_count += 1
        
        # Calculate averages
        if self.total_votes > 0:
            self.simple_average = self.total_value / Decimal(str(self.total_votes))
            
            if self.total_weight > 0:
                self.weighted_average = weighted_sum / Decimal(str(self.total_weight))
            else:
                self.weighted_average = self.simple_average
        
        self.last_calculated_at = datetime.utcnow()
        self.recalc_needed = False
    
    def mark_for_recalculation(self) -> None:
        """Mark aggregate as needing recalculation."""
        self.recalc_needed = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vote aggregate to dictionary."""
        return {
            "id": str(self.id),
            "content_type": self.content_type.value,
            "content_id": str(self.content_id),
            "composite_content_id": self.composite_content_id,
            "vote_type": self.vote_type.value,
            "total_votes": self.total_votes,
            "total_value": float(self.total_value),
            "total_weight": self.total_weight,
            "weighted_average": float(self.weighted_average) if self.weighted_average else None,
            "simple_average": float(self.simple_average) if self.simple_average else None,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "net_score": self.net_score,
            "approval_ratio": self.approval_ratio,
            "star_1": self.star_1,
            "star_2": self.star_2,
            "star_3": self.star_3,
            "star_4": self.star_4,
            "star_5": self.star_5,
            "star_distribution": self.star_distribution,
            "total_stars": self.total_stars,
            "star_average": self.star_average,
            "heart_count": self.heart_count,
            "helpful_count": self.helpful_count,
            "funny_count": self.funny_count,
            "sad_count": self.sad_count,
            "angry_count": self.angry_count,
            "emoji_counts": self.emoji_counts,
            "last_calculated_at": self.last_calculated_at.isoformat(),
            "recalc_needed": self.recalc_needed,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class UserReputation(Base, UUIDMixin, TimestampMixin):
    """
    User reputation model.
    
    This model tracks user reputation scores based on voting
    and other user activities.
    
    Attributes:
        id: Primary key UUID
        user_id: User ID
        reputation_score: Overall reputation score (0-1000)
        vote_weight: Vote weight derived from reputation
        upvotes_received: Number of upvotes received
        downvotes_received: Number of downvotes received
        helpful_votes_received: Number of helpful votes received
        quality_score: Content quality score
        contribution_score: Contribution activity score
        trust_score: Trustworthiness score
        last_calculated_at: When reputation was last calculated
        rank: User rank/level based on reputation
        badges: List of reputation badges earned
        metadata: Additional metadata
    """
    
    __tablename__ = "user_reputations"
    
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        unique=True,
        index=True
    )
    
    # Reputation scores
    reputation_score = Column(Float, default=0.0, nullable=False, index=True)
    vote_weight = Column(Float, default=1.0, nullable=False)
    
    # Vote statistics
    upvotes_received = Column(Integer, default=0, nullable=False)
    downvotes_received = Column(Integer, default=0, nullable=False)
    helpful_votes_received = Column(Integer, default=0, nullable=False)
    
    # Component scores
    quality_score = Column(Float, default=0.0, nullable=False)  # Based on content quality votes
    contribution_score = Column(Float, default=0.0, nullable=False)  # Based on activity level
    trust_score = Column(Float, default=0.0, nullable=False)  # Based on trust votes
    
    # Calculation
    last_calculated_at = Column(DateTime(timezone=True), nullable=False)
    
    # Rank and badges
    rank = Column(String(50), default="beginner", nullable=False, index=True)
    badges = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('reputation_score >= 0', name='check_reputation_non_negative'),
        CheckConstraint('vote_weight >= 0', name='check_vote_weight_non_negative'),
        CheckConstraint('upvotes_received >= 0', name='check_upvotes_received_non_negative'),
        CheckConstraint('downvotes_received >= 0', name='check_downvotes_received_non_negative'),
        CheckConstraint('helpful_votes_received >= 0', name='check_helpful_votes_non_negative'),
        Index('ix_user_reputations_score_rank', 'reputation_score', 'rank'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<UserReputation(id={self.id}, user={self.user_id}, score={self.reputation_score}, rank={self.rank})>"
    
    @property
    def net_votes_received(self) -> int:
        """Get net votes received (upvotes - downvotes)."""
        return self.upvotes_received - self.downvotes_received
    
    @property
    def total_votes_received(self) -> int:
        """Get total votes received."""
        return self.upvotes_received + self.downvotes_received
    
    @property
    def approval_rate(self) -> Optional[float]:
        """Get approval rate for received votes."""
        total = self.total_votes_received
        if total == 0:
            return None
        return self.upvotes_received / total
    
    @property
    def rank_progress(self) -> float:
        """Get progress to next rank (0-1)."""
        rank_thresholds = {
            "beginner": 0,
            "contributor": 100,
            "expert": 500,
            "master": 1000,
            "legend": 2000
        }
        
        current_rank = self.rank
        current_score = self.reputation_score
        
        ranks = list(rank_thresholds.keys())
        current_index = ranks.index(current_rank) if current_rank in ranks else 0
        
        if current_index >= len(ranks) - 1:
            return 1.0  # At max rank
        
        next_rank = ranks[current_index + 1]
        current_threshold = rank_thresholds.get(current_rank, 0)
        next_threshold = rank_thresholds.get(next_rank, 0)
        
        if next_threshold <= current_threshold:
            return 1.0
        
        progress = (current_score - current_threshold) / (next_threshold - current_threshold)
        return min(1.0, max(0.0, progress))
    
    def calculate_reputation(self) -> None:
        """Calculate reputation score based on various factors."""
        # Base score from votes received
        vote_score = self.net_votes_received * 10
        
        # Quality score component
        quality_component = self.quality_score * 100
        
        # Contribution component
        contribution_component = self.contribution_score * 50
        
        # Trust component
        trust_component = self.trust_score * 200
        
        # Helpful votes bonus
        helpful_bonus = self.helpful_votes_received * 5
        
        # Calculate total
        self.reputation_score = max(0, 
            vote_score + 
            quality_component + 
            contribution_component + 
            trust_component + 
            helpful_bonus
        )
        
        # Update vote weight based on reputation
        self.vote_weight = 1.0 + (self.reputation_score / 1000)  # 1.0 to 2.0
        
        # Update rank
        self._update_rank()
        
        self.last_calculated_at = datetime.utcnow()
    
    def _update_rank(self) -> None:
        """Update user rank based on reputation score."""
        score = self.reputation_score
        
        if score >= 2000:
            self.rank = "legend"
        elif score >= 1000:
            self.rank = "master"
        elif score >= 500:
            self.rank = "expert"
        elif score >= 100:
            self.rank = "contributor"
        else:
            self.rank = "beginner"
    
    def add_badge(self, badge_name: str) -> None:
        """Add a badge to the user."""
        if badge_name not in self.badges:
            self.badges = self.badges + [badge_name]
    
    def remove_badge(self, badge_name: str) -> None:
        """Remove a badge from the user."""
        if badge_name in self.badges:
            self.badges = [b for b in self.badges if b != badge_name]
    
    def record_upvote_received(self) -> None:
        """Record that user received an upvote."""
        self.upvotes_received += 1
    
    def record_downvote_received(self) -> None:
        """Record that user received a downvote."""
        self.downvotes_received += 1
    
    def record_helpful_vote_received(self) -> None:
        """Record that user received a helpful vote."""
        self.helpful_votes_received += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user reputation to dictionary."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "reputation_score": self.reputation_score,
            "vote_weight": self.vote_weight,
            "upvotes_received": self.upvotes_received,
            "downvotes_received": self.downvotes_received,
            "helpful_votes_received": self.helpful_votes_received,
            "net_votes_received": self.net_votes_received,
            "total_votes_received": self.total_votes_received,
            "approval_rate": self.approval_rate,
            "quality_score": self.quality_score,
            "contribution_score": self.contribution_score,
            "trust_score": self.trust_score,
            "rank": self.rank,
            "rank_progress": self.rank_progress,
            "badges": self.badges,
            "last_calculated_at": self.last_calculated_at.isoformat(),
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Poll(Base, UUIDMixin, TimestampMixin):
    """
    Poll model for surveys and voting.
    
    This model represents polls/surveys that users can vote on.
    
    Attributes:
        id: Primary key UUID
        title: Poll title
        description: Poll description
        content_type: Type of content poll is associated with
        content_id: ID of associated content
        poll_type: Type of poll
        options: Poll options
        is_multiple_choice: Whether multiple options can be selected
        is_anonymous: Whether votes are anonymous
        is_public: Whether poll is public
        status: Poll status
        starts_at: When poll starts
        ends_at: When poll ends
        created_by: User who created poll
        total_votes: Total number of votes
        metadata: Additional metadata
        tags: Categorization tags
    """
    
    __tablename__ = "polls"
    
    # Poll information
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Associated content
    content_type = Column(SQLEnum(ContentType), nullable=True, index=True)
    content_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Poll configuration
    poll_type = Column(String(50), default="single_choice", nullable=False)
    options = Column(JSONB, default=list, nullable=False)  # List of option objects
    is_multiple_choice = Column(Boolean, default=False, nullable=False)
    is_anonymous = Column(Boolean, default=False, nullable=False, index=True)
    is_public = Column(Boolean, default=True, nullable=False, index=True)
    status = Column(String(50), default="active", nullable=False, index=True)
    
    # Timing
    starts_at = Column(DateTime(timezone=True), nullable=True, index=True)
    ends_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # Creator
    created_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Statistics
    total_votes = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    tags = Column(ARRAY(String), default=[], nullable=False, index=True)
    
    # Relationships
    creator = relationship("User", foreign_keys=[created_by])
    poll_votes = relationship("PollVote", back_populates="poll", cascade="all, delete-orphan")
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('total_votes >= 0', name='check_total_votes_non_negative'),
        CheckConstraint(
            'ends_at IS NULL OR starts_at IS NULL OR ends_at > starts_at',
            name='check_poll_dates_valid'
        ),
        Index('ix_polls_status_dates', 'status', 'starts_at', 'ends_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<Poll(id={self.id}, title={self.title}, votes={self.total_votes})>"
    
    @property
    def composite_content_id(self) -> Optional[str]:
        """Get composite content identifier if exists."""
        if self.content_type and self.content_id:
            return f"{self.content_type.value}:{self.content_id}"
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if poll is active."""
        if self.status != "active":
            return False
        
        now = datetime.utcnow()
        if self.starts_at and now < self.starts_at:
            return False
        if self.ends_at and now > self.ends_at:
            return False
        
        return True
    
    @property
    def has_started(self) -> bool:
        """Check if poll has started."""
        if not self.starts_at:
            return True
        return datetime.utcnow() >= self.starts_at
    
    @property
    def has_ended(self) -> bool:
        """Check if poll has ended."""
        if not self.ends_at:
            return False
        return datetime.utcnow() > self.ends_at
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Get days remaining until poll ends."""
        if not self.ends_at:
            return None
        
        remaining = self.ends_at - datetime.utcnow()
        return max(0, remaining.days)
    
    @property
    def option_count(self) -> int:
        """Get number of poll options."""
        return len(self.options) if self.options else 0
    
    def get_option(self, option_id: str) -> Optional[Dict[str, Any]]:
        """Get poll option by ID."""
        for option in self.options:
            if option.get('id') == option_id:
                return option
        return None
    
    def update_option_votes(self) -> None:
        """Update vote counts for each option from poll votes."""
        # Reset option votes
        for option in self.options:
            option['votes'] = 0
        
        # Count votes
        for poll_vote in self.poll_votes:
            if poll_vote.is_active:
                for option_id in poll_vote.selected_options:
                    option = self.get_option(option_id)
                    if option:
                        option['votes'] = option.get('votes', 0) + 1
        
        # Update total votes
        self.total_votes = sum(option.get('votes', 0) for option in self.options)
    
    def get_results(self, include_percentages: bool = True) -> List[Dict[str, Any]]:
        """Get poll results."""
        results = []
        total_votes = self.total_votes
        
        for option in self.options:
            votes = option.get('votes', 0)
            result = {
                "id": option.get('id'),
                "text": option.get('text'),
                "description": option.get('description'),
                "votes": votes
            }
            
            if include_percentages and total_votes > 0:
                result["percentage"] = (votes / total_votes) * 100
            
            results.append(result)
        
        # Sort by votes (descending)
        results.sort(key=lambda x: x["votes"], reverse=True)
        
        return results
    
    def add_option(self, text: str, description: Optional[str] = None) -> str:
        """Add an option to the poll."""
        option_id = str(uuid.uuid4())
        option = {
            "id": option_id,
            "text": text,
            "description": description,
            "votes": 0,
            "created_at": datetime.utcnow().isoformat()
        }
        
        self.options.append(option)
        return option_id
    
    def to_dict(self, include_results: bool = True) -> Dict[str, Any]:
        """Convert poll to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "content_type": self.content_type.value if self.content_type else None,
            "content_id": str(self.content_id) if self.content_id else None,
            "composite_content_id": self.composite_content_id,
            "poll_type": self.poll_type,
            "option_count": self.option_count,
            "is_multiple_choice": self.is_multiple_choice,
            "is_anonymous": self.is_anonymous,
            "is_public": self.is_public,
            "status": self.status,
            "is_active": self.is_active,
            "has_started": self.has_started,
            "has_ended": self.has_ended,
            "starts_at": self.starts_at.isoformat() if self.starts_at else None,
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "days_remaining": self.days_remaining,
            "created_by": str(self.created_by),
            "total_votes": self.total_votes,
            "metadata": self.metadata,
            "tags": self.tags,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class PollVote(Base, UUIDMixin, TimestampMixin):
    """
    Poll vote model.
    
    This model represents individual votes in polls.
    
    Attributes:
        id: Primary key UUID
        poll_id: Poll ID
        user_id: User who voted
        selected_options: List of selected option IDs
        is_anonymous: Whether vote is anonymous
        ip_address: IP address of voter
        user_agent: User agent string
        status: Vote status
        metadata: Additional metadata
    """
    
    __tablename__ = "poll_votes"
    
    # Poll and voter
    poll_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("polls.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=True,  # Nullable for anonymous votes
        index=True
    )
    
    # Vote details
    selected_options = Column(ARRAY(String), nullable=False)
    is_anonymous = Column(Boolean, default=False, nullable=False, index=True)
    
    # Technical details
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    
    # Status
    status = Column(String(50), default="active", nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    poll = relationship("Poll", back_populates="poll_votes")
    user = relationship("User")
    
    # Check constraints
    __table_args__ = (
        UniqueConstraint('poll_id', 'user_id', name='uq_poll_user_vote'),
        CheckConstraint(
            'NOT (user_id IS NULL AND is_anonymous = FALSE)',
            name='check_anonymous_or_user'
        ),
        Index('ix_poll_votes_status', 'poll_id', 'status'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        user_str = f"user={self.user_id}" if self.user_id else "anonymous"
        return f"<PollVote(id={self.id}, poll={self.poll_id}, {user_str}, options={len(self.selected_options)})>"
    
    @property
    def is_active(self) -> bool:
        """Check if poll vote is active."""
        return self.status == "active"
    
    @property
    def option_count(self) -> int:
        """Get number of selected options."""
        return len(self.selected_options)
    
    def to_dict(self, include_poll: bool = False, include_user: bool = False) -> Dict[str, Any]:
        """Convert poll vote to dictionary."""
        result = {
            "id": str(self.id),
            "poll_id": str(self.poll_id),
            "user_id": str(self.user_id) if self.user_id else None,
            "selected_options": self.selected_options,
            "option_count": self.option_count,
            "is_anonymous": self.is_anonymous,
            "is_active": self.is_active,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
        
        if include_poll and self.poll:
            result["poll"] = {
                "id": str(self.poll.id),
                "title": self.poll.title
            }
        
        if include_user and self.user:
            result["user"] = {
                "id": str(self.user.id),
                "username": self.user.username
            }
        
        return result


class VoteFraudDetection(Base, UUIDMixin, TimestampMixin):
    """
    Vote fraud detection model.
    
    This model tracks suspicious voting patterns and fraud detection.
    
    Attributes:
        id: Primary key UUID
        detection_type: Type of fraud detection
        severity: Severity level (low, medium, high, critical)
        confidence: Detection confidence (0-1)
        user_id: Suspected user ID
        ip_address: Suspicious IP address
        content_type: Type of content involved
        content_id: ID of content involved
        vote_ids: List of suspicious vote IDs
        evidence: Evidence of fraud
        status: Detection status
        action_taken: Action taken
        resolved_by: User who resolved detection
        resolved_at: When detection was resolved
        metadata: Additional metadata
    """
    
    __tablename__ = "vote_fraud_detections"
    
    # Detection details
    detection_type = Column(String(100), nullable=False, index=True)
    severity = Column(String(20), nullable=False, index=True)
    confidence = Column(Float, default=0.0, nullable=False)
    
    # Suspicious entities
    user_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    ip_address = Column(String(50), nullable=True, index=True)
    
    # Content involved
    content_type = Column(SQLEnum(ContentType), nullable=True, index=True)
    content_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    
    # Suspicious votes
    vote_ids = Column(ARRAY(UUID(as_uuid=True)), default=[], nullable=False)
    
    # Evidence and status
    evidence = Column(JSONB, default=dict, nullable=False)
    status = Column(String(50), default="pending", nullable=False, index=True)
    action_taken = Column(JSONB, nullable=True)
    
    # Resolution
    resolved_by = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", foreign_keys=[user_id])
    resolver = relationship("User", foreign_keys=[resolved_by])
    
    # Check constraints
    __table_args__ = (
        CheckConstraint('confidence >= 0 AND confidence <= 1', name='check_confidence_range'),
        Index('ix_fraud_detections_status_severity', 'status', 'severity'),
        Index('ix_fraud_detections_user_ip', 'user_id', 'ip_address'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VoteFraudDetection(id={self.id}, type={self.detection_type}, severity={self.severity}, confidence={self.confidence})>"
    
    @property
    def is_resolved(self) -> bool:
        """Check if detection is resolved."""
        return self.status == "resolved" and self.resolved_at is not None
    
    @property
    def is_pending(self) -> bool:
        """Check if detection is pending."""
        return self.status == "pending"
    
    @property
    def suspicious_vote_count(self) -> int:
        """Get number of suspicious votes."""
        return len(self.vote_ids)
    
    def resolve(self, resolver_id: uuid.UUID, action_taken: Optional[Dict[str, Any]] = None) -> None:
        """Resolve the fraud detection."""
        self.status = "resolved"
        self.resolved_by = resolver_id
        self.resolved_at = datetime.utcnow()
        if action_taken:
            self.action_taken = action_taken
    
    def add_vote(self, vote_id: uuid.UUID) -> None:
        """Add a suspicious vote to the detection."""
        if vote_id not in self.vote_ids:
            self.vote_ids = self.vote_ids + [vote_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert vote fraud detection to dictionary."""
        return {
            "id": str(self.id),
            "detection_type": self.detection_type,
            "severity": self.severity,
            "confidence": self.confidence,
            "user_id": str(self.user_id) if self.user_id else None,
            "ip_address": self.ip_address,
            "content_type": self.content_type.value if self.content_type else None,
            "content_id": str(self.content_id) if self.content_id else None,
            "vote_ids": [str(vid) for vid in self.vote_ids],
            "suspicious_vote_count": self.suspicious_vote_count,
            "evidence": self.evidence,
            "status": self.status,
            "is_resolved": self.is_resolved,
            "is_pending": self.is_pending,
            "action_taken": self.action_taken,
            "resolved_by": str(self.resolved_by) if self.resolved_by else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


# Helper functions
def detect_vote_fraud(votes: List['Vote']) -> List['VoteFraudDetection']:
    """Detect potential vote fraud from list of votes."""
    # This is a simplified example. In production, you would use
    # more sophisticated algorithms and machine learning.
    
    detections = []
    
    # Group votes by user and content
    user_votes = {}
    ip_votes = {}
    
    for vote in votes:
        if vote.user_id:
            key = f"user:{vote.user_id}:{vote.content_type}:{vote.content_id}"
            user_votes.setdefault(key, []).append(vote)
        
        if vote.ip_address:
            key = f"ip:{vote.ip_address}:{vote.content_type}:{vote.content_id}"
            ip_votes.setdefault(key, []).append(vote)
    
    # Check for duplicate votes from same user
    for key, vote_list in user_votes.items():
        if len(vote_list) > 1:
            detection = VoteFraudDetection(
                detection_type="duplicate_votes",
                severity="medium",
                confidence=0.8,
                user_id=vote_list[0].user_id,
                content_type=vote_list[0].content_type,
                content_id=vote_list[0].content_id,
                vote_ids=[v.id for v in vote_list],
                evidence={
                    "vote_count": len(vote_list),
                    "time_range": {
                        "first": vote_list[0].created_at.isoformat(),
                        "last": vote_list[-1].created_at.isoformat()
                    }
                },
                status="pending"
            )
            detections.append(detection)
    
    # Check for vote stuffing from same IP
    for key, vote_list in ip_votes.items():
        if len(vote_list) > 5:  # Threshold
            detection = VoteFraudDetection(
                detection_type="vote_stuffing",
                severity="high",
                confidence=0.9,
                ip_address=vote_list[0].ip_address,
                content_type=vote_list[0].content_type,
                content_id=vote_list[0].content_id,
                vote_ids=[v.id for v in vote_list],
                evidence={
                    "vote_count": len(vote_list),
                    "unique_users": len(set(v.user_id for v in vote_list if v.user_id))
                },
                status="pending"
            )
            detections.append(detection)
    
    return detections


def calculate_wilson_score(upvotes: int, downvotes: int, confidence: float = 0.95) -> float:
    """
    Calculate Wilson score interval lower bound.
    
    Useful for ranking content by votes while accounting for
    small sample sizes.
    
    Args:
        upvotes: Number of upvotes
        downvotes: Number of downvotes
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        Wilson score lower bound
    """
    import math
    
    n = upvotes + downvotes
    if n == 0:
        return 0.0
    
    # z-score for confidence level
    z = {
        0.80: 1.28,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576
    }.get(confidence, 1.96)
    
    p_hat = upvotes / n
    
    # Wilson score interval
    denominator = 1 + (z**2 / n)
    centre_adjusted_probability = p_hat + (z**2 / (2 * n))
    adjusted_standard_deviation = math.sqrt(
        (p_hat * (1 - p_hat) + (z**2 / (4 * n))) / n
    )
    
    lower_bound = (
        centre_adjusted_probability - 
        z * adjusted_standard_deviation
    ) / denominator
    
    return max(0.0, lower_bound)


def calculate_bayesian_average(ratings: List[float], prior_mean: float = 3.0, prior_weight: int = 10) -> float:
    """
    Calculate Bayesian average for ratings.
    
    Useful for ranking items with few ratings.
    
    Args:
        ratings: List of ratings
        prior_mean: Prior mean rating
        prior_weight: Weight given to prior
        
    Returns:
        Bayesian average
    """
    if not ratings:
        return prior_mean
    
    n = len(ratings)
    avg_rating = sum(ratings) / n
    
    return (prior_weight * prior_mean + n * avg_rating) / (prior_weight + n)


def normalize_vote_value(value: float, vote_type: VoteType) -> float:
    """
    Normalize vote value to 0-1 range.
    
    Args:
        value: Original vote value
        vote_type: Type of vote
        
    Returns:
        Normalized value (0-1)
    """
    if vote_type == VoteType.UPVOTE:
        return 1.0
    elif vote_type == VoteType.DOWNVOTE:
        return 0.0
    elif vote_type == VoteType.STAR:
        return (value - 1) / 4  # Convert 1-5 to 0-1
    elif vote_type == VoteType.LIKERT:
        return (value - 1) / 6  # Convert 1-7 to 0-1
    elif vote_type == VoteType.THUMBS:
        return (value + 1) / 2  # Convert -1/1 to 0/1
    elif vote_type == VoteType.TRUST:
        return value / 10  # Convert 0-10 to 0-1
    else:
        return max(0.0, min(1.0, value))  # Clamp to 0-1 range