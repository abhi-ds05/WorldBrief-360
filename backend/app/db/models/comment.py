"""
comment.py - Database Model for Comments

This module defines the Comment model for storing user comments 
related to articles, incidents, or other entities in the system.

Relationships:
- Comments belong to a User (author)
- Comments can belong to an Article or Incident
- Comments can have parent comments (for threaded replies)
"""

import uuid
from datetime import datetime
from typing import Optional, TYPE_CHECKING
from sqlalchemy import Column, String, Text, ForeignKey, Integer, DateTime, Boolean, Enum
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, SoftDeleteMixin

if TYPE_CHECKING:
    from models.user import User
    from models.article import Article
    from models.incident import Incident


class CommentStatus:
    """Constants for comment status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SPAM = "spam"
    DELETED = "deleted"


class Comment(Base, TimestampMixin, SoftDeleteMixin):
    """
    Comment model for user-generated comments on various content types.
    
    Attributes:
        id: Primary key UUID
        content: The comment text/content
        html_content: Sanitized HTML content (if applicable)
        status: Current status of the comment
        author_id: Foreign key to the User who created the comment
        article_id: Foreign key to Article (if comment is on an article)
        incident_id: Foreign key to Incident (if comment is on an incident)
        parent_id: Foreign key to parent Comment (for threaded replies)
        like_count: Number of likes the comment has received
        report_count: Number of times the comment has been reported
        is_edited: Flag indicating if the comment has been edited
        edited_at: Timestamp of last edit
        ip_address: IP address of the commenter (for moderation)
        user_agent: Browser/user agent info
        metadata: JSON field for additional data (flags, tags, etc.)
    """
    
    __tablename__ = "comments"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    
    # Content fields
    content = Column(Text, nullable=False, comment="Original comment text")
    html_content = Column(Text, nullable=True, comment="Sanitized HTML version")
    
    # Status
    status = Column(
        String(20),
        default=CommentStatus.PENDING,
        nullable=False,
        index=True,
        comment="Comment status: pending, approved, rejected, spam, deleted"
    )
    
    # Foreign keys
    author_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    article_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("articles.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    incident_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incidents.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    parent_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("comments.id", ondelete="CASCADE"), 
        nullable=True,
        index=True
    )
    
    # Counters
    like_count = Column(Integer, default=0, nullable=False)
    report_count = Column(Integer, default=0, nullable=False)
    
    # Edit tracking
    is_edited = Column(Boolean, default=False, nullable=False)
    edited_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata for moderation
    ip_address = Column(String(45), nullable=True, comment="IPv4 or IPv6 address")
    user_agent = Column(Text, nullable=True, comment="HTTP User-Agent header")
    
    # Additional data (could be used for flags, moderation notes, etc.)
    metadata = Column(Text, nullable=True, comment="JSON metadata field")
    
    # Relationships
    author = relationship("User", back_populates="comments")
    article = relationship("Article", back_populates="comments")
    incident = relationship("Incident", back_populates="comments")
    
    # Self-referential relationship for threaded comments
    parent = relationship(
        "Comment", 
        remote_side=[id], 
        back_populates="replies",
        post_update=True
    )
    replies = relationship(
        "Comment", 
        back_populates="parent",
        cascade="all, delete-orphan"
    )
    
    # Many-to-many relationships (if you have them)
    # liked_by = relationship("User", secondary="comment_likes", back_populates="liked_comments")
    # reported_by = relationship("User", secondary="comment_reports", back_populates="reported_comments")
    
    def __repr__(self) -> str:
        """String representation of the Comment."""
        return f"<Comment(id={self.id}, author={self.author_id}, status={self.status})>"
    
    @validates('content')
    def validate_content(self, key: str, content: str) -> str:
        """Validate comment content before saving."""
        content = content.strip()
        if not content:
            raise ValueError("Comment content cannot be empty")
        if len(content) > 10000:  # Adjust limit as needed
            raise ValueError("Comment content is too long")
        return content
    
    @property
    def is_reply(self) -> bool:
        """Check if this comment is a reply to another comment."""
        return self.parent_id is not None
    
    @property
    def is_top_level(self) -> bool:
        """Check if this is a top-level comment (not a reply)."""
        return self.parent_id is None
    
    @property
    def can_be_replied_to(self) -> bool:
        """Check if this comment can receive replies."""
        return self.status == CommentStatus.APPROVED and not self.is_deleted
    
    @property
    def reply_count(self) -> int:
        """Get the number of replies to this comment."""
        return len(self.replies) if self.replies else 0
    
    def mark_as_edited(self) -> None:
        """Mark the comment as edited and update timestamp."""
        self.is_edited = True
        self.edited_at = datetime.utcnow()
    
    def increment_like_count(self) -> None:
        """Increment the like counter."""
        self.like_count += 1
    
    def decrement_like_count(self) -> None:
        """Decrement the like counter, ensuring it doesn't go below zero."""
        self.like_count = max(0, self.like_count - 1)
    
    def increment_report_count(self) -> None:
        """Increment the report counter."""
        self.report_count += 1
    
    def approve(self) -> None:
        """Approve the comment."""
        self.status = CommentStatus.APPROVED
    
    def reject(self) -> None:
        """Reject the comment."""
        self.status = CommentStatus.REJECTED
    
    def mark_as_spam(self) -> None:
        """Mark the comment as spam."""
        self.status = CommentStatus.SPAM
    
    def to_dict(self, include_replies: bool = False, depth: int = 0, max_depth: int = 3) -> dict:
        """
        Convert comment to dictionary for API responses.
        
        Args:
            include_replies: Whether to include replies in the output
            depth: Current depth in the reply tree (for recursion)
            max_depth: Maximum depth to include replies
        
        Returns:
            Dictionary representation of the comment
        """
        result = {
            "id": str(self.id),
            "content": self.content,
            "html_content": self.html_content,
            "status": self.status,
            "author_id": str(self.author_id),
            "article_id": str(self.article_id) if self.article_id else None,
            "incident_id": str(self.incident_id) if self.incident_id else None,
            "parent_id": str(self.parent_id) if self.parent_id else None,
            "like_count": self.like_count,
            "report_count": self.report_count,
            "is_edited": self.is_edited,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "edited_at": self.edited_at.isoformat() if self.edited_at else None,
            "reply_count": self.reply_count,
            "is_top_level": self.is_top_level,
            "can_be_replied_to": self.can_be_replied_to
        }
        
        # Include author info if available (prevents circular import)
        if hasattr(self, 'author') and self.author:
            result["author"] = {
                "id": str(self.author.id),
                "username": self.author.username,
                "display_name": getattr(self.author, 'display_name', None)
            }
        
        # Include replies if requested and not at max depth
        if include_replies and depth < max_depth and self.replies:
            result["replies"] = [
                reply.to_dict(include_replies=True, depth=depth + 1, max_depth=max_depth)
                for reply in sorted(self.replies, key=lambda x: x.created_at)
                if reply.status == CommentStatus.APPROVED and not reply.is_deleted
            ]
        
        return result
    
    @classmethod
    def create(
        cls,
        content: str,
        author_id: uuid.UUID,
        article_id: Optional[uuid.UUID] = None,
        incident_id: Optional[uuid.UUID] = None,
        parent_id: Optional[uuid.UUID] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> 'Comment':
        """
        Factory method to create a new comment with proper initialization.
        
        Args:
            content: The comment text
            author_id: ID of the user creating the comment
            article_id: Optional article ID
            incident_id: Optional incident ID
            parent_id: Optional parent comment ID
            ip_address: Optional IP address
            user_agent: Optional user agent string
            html_content: Optional sanitized HTML content
        
        Returns:
            A new Comment instance
        """
        if not article_id and not incident_id:
            raise ValueError("Comment must be associated with either an article or incident")
        
        if article_id and incident_id:
            raise ValueError("Comment cannot be associated with both an article and incident")
        
        comment = cls(
            content=content,
            html_content=html_content or content,  # Default to plain content if no HTML provided
            author_id=author_id,
            article_id=article_id,
            incident_id=incident_id,
            parent_id=parent_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return comment


# Pydantic schemas for API validation (usually in schemas/comment.py, but included here for completeness)
"""
If you're using Pydantic, you might also have these schemas in a separate schemas file:
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime


class CommentBase(BaseModel):
    """Base schema for comment operations."""
    content: str = Field(..., min_length=1, max_length=10000)
    parent_id: Optional[str] = None
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Comment content cannot be empty')
        return v.strip()


class CommentCreate(CommentBase):
    """Schema for creating a new comment."""
    article_id: Optional[str] = None
    incident_id: Optional[str] = None


class CommentUpdate(BaseModel):
    """Schema for updating an existing comment."""
    content: str = Field(..., min_length=1, max_length=10000)


class CommentInDBBase(CommentBase):
    """Base schema for comment in database."""
    id: str
    author_id: str
    status: str
    like_count: int
    report_count: int
    is_edited: bool
    created_at: datetime
    updated_at: datetime
    reply_count: int = 0
    
    class Config:
        from_attributes = True


class Comment(CommentInDBBase):
    """Schema for comment API responses."""
    author: Optional[dict] = None
    replies: Optional[List['Comment']] = None
    
    class Config:
        from_attributes = True


# Update forward reference for nested replies
Comment.update_forward_refs()