"""
Article and content management models.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional, List

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
    Text,
    func,
    Float,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import expression

from db.base import Base, TimestampMixin, SoftDeleteMixin, AuditMixin, generate_uuid


class ArticleStatus(str, enum.Enum):
    """Article publication status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    SCHEDULED = "scheduled"
    ARCHIVED = "archived"
    REJECTED = "rejected"
    DELETED = "deleted"


class ArticleType(str, enum.Enum):
    """Article content types."""
    NEWS = "news"
    ANALYSIS = "analysis"
    OPINION = "opinion"
    FEATURE = "feature"
    INTERVIEW = "interview"
    REPORT = "report"
    BLOG = "blog"
    TUTORIAL = "tutorial"
    RESEARCH = "research"
    PRESS_RELEASE = "press_release"


class ArticleVisibility(str, enum.Enum):
    """Article visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    UNLISTED = "unlisted"
    SUBSCRIBERS = "subscribers"
    MEMBERS = "members"


class Article(Base, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """
    Article model for news, blog posts, and other content.
    """
    
    __tablename__ = "article"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    author_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    category_id = Column(UUID(as_uuid=True), ForeignKey('category.id'), nullable=True, index=True)
    topic_id = Column(UUID(as_uuid=True), ForeignKey('topic.id'), nullable=True, index=True)
    
    # Basic information
    title = Column(String(500), nullable=False, index=True)
    slug = Column(String(500), unique=True, nullable=False, index=True)
    excerpt = Column(Text, nullable=True)
    
    # Content
    content = Column(Text, nullable=False)
    content_html = Column(Text, nullable=True)  # Rendered HTML version
    
    # Metadata
    status = Column(Enum(ArticleStatus), default=ArticleStatus.DRAFT, nullable=False, index=True)
    type = Column(Enum(ArticleType), default=ArticleType.NEWS, nullable=False, index=True)
    visibility = Column(Enum(ArticleVisibility), default=ArticleVisibility.PUBLIC, nullable=False, index=True)
    
    # Publishing
    published_at = Column(DateTime(timezone=True), nullable=True, index=True)
    scheduled_at = Column(DateTime(timezone=True), nullable=True, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    # SEO and discovery
    meta_title = Column(String(500), nullable=True)
    meta_description = Column(Text, nullable=True)
    meta_keywords = Column(JSONB, default=list)
    canonical_url = Column(String(500), nullable=True)
    
    # Media
    featured_image = Column(Text, nullable=True)
    image_caption = Column(Text, nullable=True)
    image_credit = Column(String(255), nullable=True)
    
    # Statistics
    view_count = Column(Integer, default=0, nullable=False)
    share_count = Column(Integer, default=0, nullable=False)
    comment_count = Column(Integer, default=0, nullable=False)
    like_count = Column(Integer, default=0, nullable=False)
    bookmark_count = Column(Integer, default=0, nullable=False)
    
    # Ratings and engagement
    average_rating = Column(Float, default=0.0, nullable=False)
    rating_count = Column(Integer, default=0, nullable=False)
    
    # Content properties
    reading_time_minutes = Column(Integer, default=0, nullable=False)
    word_count = Column(Integer, default=0, nullable=False)
    
    # Localization
    language = Column(String(10), default='en', nullable=False, index=True)
    locale = Column(String(10), default='en-US', nullable=False)
    
    # Content flags
    is_featured = Column(Boolean, default=False, nullable=False, index=True)
    is_pinned = Column(Boolean, default=False, nullable=False, index=True)
    is_sponsored = Column(Boolean, default=False, nullable=False)
    has_comments_enabled = Column(Boolean, default=True, nullable=False)
    
    # Full-text search
    search_vector = Column(TSVECTOR, nullable=True)
    
    # Additional metadata
    tags = Column(JSONB, default=list)
    attributes = Column(JSONB, default=dict)
    custom_fields = Column(JSONB, default=dict)
    
    # Relationships
    author = relationship("User", foreign_keys=[author_id], back_populates="articles")
    category = relationship("Category", back_populates="articles")
    topic = relationship("Topic", back_populates="articles")
    comments = relationship("Comment", back_populates="article", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="article", cascade="all, delete-orphan")
    bookmarks = relationship("Bookmark", back_populates="article", cascade="all, delete-orphan")
    versions = relationship("ArticleVersion", back_populates="article", cascade="all, delete-orphan")
    related_articles = relationship(
        "Article",
        secondary="article_relation",
        primaryjoin="Article.id==ArticleRelation.article_id",
        secondaryjoin="Article.id==ArticleRelation.related_article_id",
        backref="related_to",
        lazy="dynamic"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_article_status_published', status, published_at.desc()),
        Index('idx_article_author_status', author_id, status),
        Index('idx_article_featured_published', is_featured, published_at.desc()),
        Index('idx_article_language_status', language, status, published_at.desc()),
        Index('idx_article_search', search_vector, postgresql_using='gin'),
        Index('idx_article_slug_language', slug, language, unique=True),
        UniqueConstraint('slug', 'language', name='uq_article_slug_language'),
        CheckConstraint('view_count >= 0', name='check_view_count_non_negative'),
        CheckConstraint('share_count >= 0', name='check_share_count_non_negative'),
        CheckConstraint('comment_count >= 0', name='check_comment_count_non_negative'),
        CheckConstraint('average_rating >= 0 AND average_rating <= 5', name='check_rating_range'),
        CheckConstraint('reading_time_minutes >= 0', name='check_reading_time_non_negative'),
        CheckConstraint('word_count >= 0', name='check_word_count_non_negative'),
    )
    
    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate article slug."""
        if not slug:
            raise ValueError("Slug cannot be empty")
        if ' ' in slug:
            raise ValueError("Slug cannot contain spaces")
        return slug.lower()
    
    @validates('title')
    def validate_title(self, key, title):
        """Validate article title."""
        if not title or len(title.strip()) < 5:
            raise ValueError("Title must be at least 5 characters")
        if len(title) > 500:
            raise ValueError("Title cannot exceed 500 characters")
        return title.strip()
    
    def is_published(self) -> bool:
        """Check if article is published."""
        return self.status == ArticleStatus.PUBLISHED and not self.is_deleted
    
    def is_scheduled(self) -> bool:
        """Check if article is scheduled for publication."""
        return self.status == ArticleStatus.SCHEDULED and self.scheduled_at is not None
    
    def is_draft(self) -> bool:
        """Check if article is a draft."""
        return self.status == ArticleStatus.DRAFT
    
    def is_visible_to_public(self) -> bool:
        """Check if article is visible to the public."""
        if self.is_deleted:
            return False
        
        if self.visibility != ArticleVisibility.PUBLIC:
            return False
        
        if not self.is_published():
            return False
        
        # Check if article has expired
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        
        return True
    
    def can_be_viewed_by(self, user_id: Optional[str] = None, is_subscriber: bool = False) -> bool:
        """Check if article can be viewed by a specific user."""
        if self.is_visible_to_public():
            return True
        
        if self.visibility == ArticleVisibility.SUBSCRIBERS and is_subscriber:
            return True
        
        if self.visibility == ArticleVisibility.MEMBERS and user_id:
            # Check if user is a member
            return True
        
        if self.visibility == ArticleVisibility.PRIVATE and user_id == str(self.author_id):
            return True
        
        return False
    
    def increment_view_count(self) -> None:
        """Increment view count."""
        self.view_count += 1
    
    def increment_share_count(self) -> None:
        """Increment share count."""
        self.share_count += 1
    
    def update_comment_count(self) -> None:
        """Update comment count from related comments."""
        from sqlalchemy import select, func
        from db.session import SessionLocal
        
        with SessionLocal() as db:
            from .comment import Comment
            count = db.execute(
                select(func.count(Comment.id)).where(
                    Comment.article_id == self.id,
                    Comment.is_deleted == False
                )
            ).scalar()
            self.comment_count = count
    
    def update_like_count(self) -> None:
        """Update like count from related votes."""
        from sqlalchemy import select, func
        from db.session import SessionLocal
        
        with SessionLocal() as db:
            from .vote import Vote, VoteType
            count = db.execute(
                select(func.count(Vote.id)).where(
                    Vote.article_id == self.id,
                    Vote.vote_type == VoteType.LIKE,
                    Vote.is_deleted == False
                )
            ).scalar()
            self.like_count = count
    
    def calculate_reading_time(self) -> int:
        """Calculate reading time in minutes based on word count."""
        # Average reading speed: 200-250 words per minute
        words_per_minute = 225
        if self.word_count <= 0:
            return 1
        reading_time = max(1, round(self.word_count / words_per_minute))
        return reading_time
    
    def update_word_count(self) -> None:
        """Update word count from content."""
        import re
        if self.content:
            # Count words (simple approach)
            words = re.findall(r'\b\w+\b', self.content)
            self.word_count = len(words)
            self.reading_time_minutes = self.calculate_reading_time()
    
    def publish(self) -> None:
        """Publish the article."""
        self.status = ArticleStatus.PUBLISHED
        self.published_at = datetime.utcnow()
    
    def schedule(self, schedule_time: datetime) -> None:
        """Schedule article for publication."""
        self.status = ArticleStatus.SCHEDULED
        self.scheduled_at = schedule_time
    
    def archive(self) -> None:
        """Archive the article."""
        self.status = ArticleStatus.ARCHIVED
    
    def reject(self, reason: str = None) -> None:
        """Reject the article."""
        self.status = ArticleStatus.REJECTED
        if reason and self.attributes:
            self.attributes['rejection_reason'] = reason
    
    def get_reading_time_display(self) -> str:
        """Get human-readable reading time."""
        if self.reading_time_minutes <= 1:
            return "1 min read"
        return f"{self.reading_time_minutes} min read"
    
    def get_absolute_url(self) -> str:
        """Get absolute URL for the article."""
        # This would use your site's URL configuration
        base_url = "https://example.com"
        return f"{base_url}/articles/{self.slug}"
    
    def to_dict(self, include_content: bool = True) -> dict:
        """
        Convert to dictionary representation.
        
        Args:
            include_content: Whether to include the full content
            
        Returns:
            Dictionary representation
        """
        data = super().to_dict()
        
        if not include_content:
            # Don't include full content in summary
            if 'content' in data:
                del data['content']
            if 'content_html' in data:
                del data['content_html']
        
        # Add computed fields
        data['is_published'] = self.is_published()
        data['is_visible'] = self.is_visible_to_public()
        data['reading_time_display'] = self.get_reading_time_display()
        data['url'] = self.get_absolute_url()
        
        # Add author information if relationship is loaded
        if self.author:
            data['author'] = {
                'id': str(self.author.id),
                'username': self.author.username,
                'full_name': self.author.full_name,
                'avatar_url': self.author.avatar_url,
            }
        
        return data


class ArticleVersion(Base, TimestampMixin):
    """
    Article version history for tracking changes.
    """
    
    __tablename__ = "article_version"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Foreign keys
    article_id = Column(UUID(as_uuid=True), ForeignKey('article.id'), nullable=False, index=True)
    author_id = Column(UUID(as_uuid=True), ForeignKey('user.id'), nullable=False, index=True)
    
    # Version information
    version_number = Column(Integer, nullable=False)
    change_summary = Column(Text, nullable=True)
    
    # Content snapshot
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    excerpt = Column(Text, nullable=True)
    
    # Metadata snapshot
    status = Column(String(50), nullable=False)
    type = Column(String(50), nullable=False)
    meta_title = Column(String(500), nullable=True)
    meta_description = Column(Text, nullable=True)
    
    # Media
    featured_image = Column(Text, nullable=True)
    
    # Relationships
    article = relationship("Article", back_populates="versions")
    author = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_article_version_article', article_id, version_number.desc()),
        UniqueConstraint('article_id', 'version_number', name='uq_article_version'),
    )
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        data = super().to_dict()
        
        # Add author information
        if self.author:
            data['author'] = {
                'id': str(self.author.id),
                'username': self.author.username,
                'full_name': self.author.full_name,
            }
        
        return data


class Category(Base, TimestampMixin, SoftDeleteMixin):
    """
    Article category for organization.
    """
    
    __tablename__ = "category"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=generate_uuid)
    
    # Category information
    name = Column(String(100), nullable=False, index=True)
    slug = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Hierarchy
    parent_id = Column(UUID(as_uuid=True), ForeignKey('category.id'), nullable=True, index=True)
    level = Column(Integer, default=0, nullable=False)
    
    # Display
    color = Column(String(20), nullable=True)
    icon = Column(String(50), nullable=True)
    is_featured = Column(Boolean, default=False, nullable=False, index=True)
    
    # Statistics
    article_count = Column(Integer, default=0, nullable=False)
    
    # Ordering
    sort_order = Column(Integer, default=0, nullable=False, index=True)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Relationships
    parent = relationship("Category", remote_side=[id], backref="children")
    articles = relationship("Article", back_populates="category")
    
    # Indexes
    __table_args__ = (
        Index('idx_category_parent_level', parent_id, level),
        Index('idx_category_slug', slug, unique=True),
        CheckConstraint('article_count >= 0', name='check_category_article_count'),
        CheckConstraint('level >= 0', name='check_category_level'),
    )
    
    @validates('slug')
    def validate_slug(self, key, slug):
        """Validate category slug."""
        if not slug:
            raise ValueError("Slug cannot be empty")
        if ' ' in slug:
            raise ValueError("Slug cannot contain spaces")
        return slug.lower()
    
    def get_full_path(self) -> str:
        """Get full hierarchical path for the category."""
        if self.parent:
            return f"{self.parent.get_full_path()}/{self.slug}"
        return self.slug
    
    def get_breadcrumbs(self) -> List[dict]:
        """Get breadcrumb trail for the category."""
        breadcrumbs = []
        current = self
        
        while current:
            breadcrumbs.insert(0, {
                'id': str(current.id),
                'name': current.name,
                'slug': current.slug,
            })
            current = current.parent
        
        return breadcrumbs
    
    def increment_article_count(self) -> None:
        """Increment article count."""
        self.article_count += 1
    
    def decrement_article_count(self) -> None:
        """Decrement article count."""
        if self.article_count > 0:
            self.article_count -= 1


# Article relations (many-to-many)
class ArticleRelation(Base, TimestampMixin):
    """
    Relationship between articles.
    """
    
    __tablename__ = "article_relation"
    
    # Foreign keys (composite primary key)
    article_id = Column(UUID(as_uuid=True), ForeignKey('article.id'), primary_key=True)
    related_article_id = Column(UUID(as_uuid=True), ForeignKey('article.id'), primary_key=True)
    
    # Relation type
    relation_type = Column(String(50), default='related', nullable=False)  # related, series, translation, etc.
    
    # Ordering
    sort_order = Column(Integer, default=0, nullable=False)
    
    # Metadata
    metadata = Column(JSONB, default=dict)
    
    # Indexes
    __table_args__ = (
        Index('idx_article_relation_article', article_id, relation_type, sort_order),
        Index('idx_article_relation_related', related_article_id, relation_type),
        UniqueConstraint('article_id', 'related_article_id', name='uq_article_relation'),
    )