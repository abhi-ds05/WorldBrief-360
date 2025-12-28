"""
mixins.py - Reusable SQLAlchemy Model Mixins

This module provides common mixin classes that can be inherited by models
to add standardized functionality like timestamps, soft deletion, audit trails,
and other cross-cutting concerns.

Key Mixins:
1. TimestampMixin - created_at, updated_at fields
2. SoftDeleteMixin - is_deleted, deleted_at fields with soft deletion logic
3. AuditMixin - created_by, updated_by fields for user tracking
4. UUIDMixin - UUID primary key instead of integer
5. SearchableMixin - Full-text search capabilities
6. VersionMixin - Model versioning support
7. SlugMixin - URL-friendly slug fields
8. StatusMixin - Standardized status fields
9. OwnableMixin - Ownership and permission tracking
10. CachableMixin - Cache-related fields
"""

import uuid
import re
from datetime import datetime,timedelta
from typing import Optional, Any, Dict, TYPE_CHECKING
from sqlalchemy import (
    Column, String, DateTime, Boolean, Integer, Text, 
    ForeignKey, event, Index, func, Text,JSON
)
from sqlalchemy.orm import declared_attr, Session, relationship,validates
from sqlalchemy.dialects.postgresql import UUID, TSVECTOR,JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import expression

if TYPE_CHECKING:
    from models.user import User


class TimestampMixin:
    """
    Adds created_at and updated_at timestamp fields to models.
    
    Usage:
        class MyModel(Base, TimestampMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    
    Automatically sets created_at on insert and updated_at on update.
    """
    
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
        doc="Timestamp when the record was created"
    )
    
    updated_at = Column(
        DateTime(timezone=True),
        onupdate=func.now(),
        nullable=True,
        doc="Timestamp when the record was last updated"
    )
    
    @classmethod
    def __declare_last__(cls):
        """Register event listeners after class is fully declared."""
        @event.listens_for(cls, 'before_insert', propagate=True)
        def receive_before_insert(mapper, connection, target):
            """Set created_at on insert if not already set."""
            if target.created_at is None:
                target.created_at = datetime.utcnow()
        
        @event.listens_for(cls, 'before_update', propagate=True)
        def receive_before_update(mapper, connection, target):
            """Set updated_at on update."""
            target.updated_at = datetime.utcnow()


class SoftDeleteMixin:
    """
    Adds soft deletion functionality to models.
    
    Instead of physically deleting records, marks them as deleted
    with is_deleted flag and deleted_at timestamp.
    
    Usage:
        class MyModel(Base, SoftDeleteMixin):
            __tablename__ = 'mymodel'
            # ... other fields
        
        # Soft delete
        instance.delete()
        
        # Restore
        instance.restore()
        
        # Query non-deleted records
        active = session.query(MyModel).filter(MyModel.is_deleted == False)
    """
    
    is_deleted = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether the record is marked as deleted"
    )
    
    deleted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when the record was soft deleted"
    )
    
    deleted_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="User who deleted the record"
    )
    
    @declared_attr
    def deleted_by(cls):
        """Relationship to the user who deleted the record."""
        return relationship(
            "User",
            foreign_keys=[cls.deleted_by_id],
            primaryjoin="User.id == %s.deleted_by_id" % cls.__name__
        )
    
    def delete(self, deleted_by: Optional['User'] = None) -> None:
        """
        Soft delete the record.
        
        Args:
            deleted_by: Optional user who is performing the deletion
        """
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        if deleted_by:
            self.deleted_by_id = deleted_by.id
    
    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None
        self.deleted_by_id = None
    
    @property
    def is_active(self) -> bool:
        """Check if the record is active (not deleted)."""
        return not self.is_deleted
    
    @classmethod
    def active_query(cls, session: Session):
        """
        Get a query that filters out deleted records.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Query with is_deleted=False filter applied
        """
        return session.query(cls).filter(cls.is_deleted == False)
    
    @classmethod
    def deleted_query(cls, session: Session):
        """
        Get a query that returns only deleted records.
        
        Args:
            session: SQLAlchemy session
            
        Returns:
            Query with is_deleted=True filter applied
        """
        return session.query(cls).filter(cls.is_deleted == True)


class AuditMixin:
    """
    Adds created_by and updated_by fields for tracking user actions.
    
    Usage:
        class MyModel(Base, AuditMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    
    created_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="User who created the record"
    )
    
    updated_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="User who last updated the record"
    )
    
    @declared_attr
    def created_by(cls):
        """Relationship to the user who created the record."""
        return relationship(
            "User",
            foreign_keys=[cls.created_by_id],
            primaryjoin="User.id == %s.created_by_id" % cls.__name__,
            lazy='joined'
        )
    
    @declared_attr
    def updated_by(cls):
        """Relationship to the user who last updated the record."""
        return relationship(
            "User",
            foreign_keys=[cls.updated_by_id],
            primaryjoin="User.id == %s.updated_by_id" % cls.__name__,
            lazy='joined'
        )
    
    def set_created_by(self, user: 'User') -> None:
        """Set the user who created this record."""
        self.created_by_id = user.id
    
    def set_updated_by(self, user: 'User') -> None:
        """Set the user who last updated this record."""
        self.updated_by_id = user.id


class UUIDMixin:
    """
    Adds a UUID primary key to models.
    
    Usage:
        class MyModel(Base, UUIDMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        doc="Unique identifier (UUID)"
    )
    
    @property
    def short_id(self) -> str:
        """Get a shortened version of the UUID (first 8 chars)."""
        return str(self.id)[:8]


class SearchableMixin:
    """
    Adds full-text search capabilities to models.
    
    Requires PostgreSQL and the pg_trgm extension for best results.
    
    Usage:
        class Article(Base, SearchableMixin):
            __tablename__ = 'articles'
            title = Column(String(255))
            content = Column(Text)
            
            # Define which columns to include in search
            __searchable__ = ['title', 'content']
    """
    
    search_vector = Column(
        TSVECTOR,
        nullable=True,
        doc="Full-text search vector for PostgreSQL"
    )
    
    @declared_attr
    def __searchable__(cls):
        """Override this in your model to specify searchable columns."""
        return []
    
    @classmethod
    def __declare_last__(cls):
        """Create search vector index after class declaration."""
        if cls.__searchable__:
            # Create GIN index for faster text search
            Index(
                f'ix_{cls.__tablename__}_search_vector',
                cls.search_vector,
                postgresql_using='gin'
            )
            
            @event.listens_for(cls, 'before_insert', propagate=True)
            @event.listens_for(cls, 'before_update', propagate=True)
            def update_search_vector(mapper, connection, target):
                """Update search vector before insert/update."""
                searchable_values = []
                for column_name in cls.__searchable__:
                    value = getattr(target, column_name, '')
                    if value:
                        searchable_values.append(str(value))
                
                if searchable_values:
                    # Create search vector from concatenated values
                    search_text = ' '.join(searchable_values)
                    target.search_vector = func.to_tsvector('english', search_text)
    
    @classmethod
    def search(cls, session: Session, query: str, limit: int = 50) -> list:
        """
        Perform full-text search on this model.
        
        Args:
            session: SQLAlchemy session
            query: Search query string
            limit: Maximum number of results
            
        Returns:
            List of model instances matching the search
        """
        if not cls.__searchable__:
            raise ValueError(f"{cls.__name__} does not have searchable columns defined")
        
        search_query = func.plainto_tsquery('english', query)
        return (
            session.query(cls)
            .filter(cls.search_vector.op('@@')(search_query))
            .order_by(func.ts_rank(cls.search_vector, search_query).desc())
            .limit(limit)
            .all()
        )


class VersionMixin:
    """
    Adds version tracking to models for optimistic concurrency control.
    
    Usage:
        class MyModel(Base, VersionMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    
    version = Column(
        Integer,
        default=1,
        nullable=False,
        doc="Version number for optimistic locking"
    )
    
    version_created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        doc="Timestamp when this version was created"
    )
    
    version_comment = Column(
        Text,
        nullable=True,
        doc="Comment describing the changes in this version"
    )
    
    @classmethod
    def __declare_last__(cls):
        """Increment version on update."""
        @event.listens_for(cls, 'before_update', propagate=True)
        def increment_version(mapper, connection, target):
            """Increment version number before update."""
            target.version += 1
            target.version_created_at = datetime.utcnow()


class SlugMixin:
    """
    Adds slug field for URL-friendly identifiers.
    
    Usage:
        class Article(Base, SlugMixin):
            __tablename__ = 'articles'
            title = Column(String(255))
            
            # Define which column to use for slug generation
            __sluggable__ = 'title'
    """
    
    slug = Column(
        String(255),
        nullable=False,
        unique=True,
        index=True,
        doc="URL-friendly identifier"
    )
    
    @declared_attr
    def __sluggable__(cls):
        """Override this in your model to specify the source column for slugs."""
        return None
    
    @classmethod
    def __declare_last__(cls):
        """Generate slug before insert if not provided."""
        if cls.__sluggable__:
            @event.listens_for(cls, 'before_insert', propagate=True)
            def generate_slug(mapper, connection, target):
                """Generate slug from sluggable column if not provided."""
                if not target.slug and cls.__sluggable__:
                    source = getattr(target, cls.__sluggable__, '')
                    if source:
                        target.slug = cls.create_slug(source)
    
    @staticmethod
    def create_slug(text: str, max_length: int = 255) -> str:
        """
        Create a URL-friendly slug from text.
        
        Args:
            text: Text to convert to slug
            max_length: Maximum slug length
            
        Returns:
            URL-friendly slug
        """
        # Convert to lowercase
        slug = text.lower()
        
        # Replace spaces and underscores with hyphens
        slug = re.sub(r'[\s_]+', '-', slug)
        
        # Remove all non-word characters except hyphens
        slug = re.sub(r'[^\w\-]+', '', slug)
        
        # Remove leading/trailing hyphens
        slug = slug.strip('-')
        
        # Limit length
        if len(slug) > max_length:
            slug = slug[:max_length].rstrip('-')
        
        # If slug is empty after processing, generate a random one
        if not slug:
            slug = f"item-{uuid.uuid4().hex[:8]}"
        
        return slug


class StatusMixin:
    """
    Adds standardized status fields to models.
    
    Usage:
        class MyModel(Base, StatusMixin):
            __tablename__ = 'mymodel'
            # ... other fields
            
            # Define allowed status values
            __statuses__ = ['draft', 'published', 'archived']
    """
    
    status = Column(
        String(50),
        default='draft',
        nullable=False,
        index=True,
        doc="Current status of the record"
    )
    
    status_changed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp when status was last changed"
    )
    
    status_changed_by_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        doc="User who changed the status"
    )
    
    status_reason = Column(
        Text,
        nullable=True,
        doc="Reason for status change"
    )
    
    @declared_attr
    def __statuses__(cls):
        """Override this in your model to define allowed status values."""
        return ['active', 'inactive']
    
    @declared_attr
    def status_changed_by(cls):
        """Relationship to the user who changed the status."""
        return relationship(
            "User",
            foreign_keys=[cls.status_changed_by_id],
            primaryjoin="User.id == %s.status_changed_by_id" % cls.__name__
        )
    
    @validates('status')
    def validate_status(self, key: str, status: str) -> str:
        """Validate that status is in allowed values."""
        allowed_statuses = self.__statuses__
        if status not in allowed_statuses:
            raise ValueError(
                f"Status must be one of {allowed_statuses}, got '{status}'"
            )
        return status
    
    def set_status(
        self, 
        status: str, 
        changed_by: Optional['User'] = None,
        reason: Optional[str] = None
    ) -> None:
        """
        Change the status of the record.
        
        Args:
            status: New status
            changed_by: User changing the status
            reason: Reason for status change
        """
        if status not in self.__statuses__:
            raise ValueError(f"Invalid status: {status}. Must be one of {self.__statuses__}")
        
        old_status = self.status
        self.status = status
        self.status_changed_at = datetime.utcnow()
        self.status_changed_by_id = changed_by.id if changed_by else None
        self.status_reason = reason
        
        return old_status
    
    def is_status(self, status: str) -> bool:
        """Check if record has specific status."""
        return self.status == status


class OwnableMixin:
    """
    Adds ownership and permission tracking to models.
    
    Usage:
        class MyModel(Base, OwnableMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    
    owner_id = Column(
        UUID(as_uuid=True),
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True,
        doc="Owner of the record"
    )
    
    is_public = Column(
        Boolean,
        default=False,
        nullable=False,
        index=True,
        doc="Whether the record is publicly accessible"
    )
    
    permissions = Column(
        JSON,  # Store as JSON: {"user_id": ["read", "write"], "group_id": ["read"]}
        default=dict,
        nullable=False,
        doc="Custom permissions for users/groups"
    )
    
    @declared_attr
    def owner(cls):
        """Relationship to the owner user."""
        return relationship(
            "User",
            foreign_keys=[cls.owner_id],
            primaryjoin="User.id == %s.owner_id" % cls.__name__
        )
    
    def can_read(self, user: Optional['User'] = None) -> bool:
        """
        Check if a user can read this record.
        
        Args:
            user: User to check permissions for
            
        Returns:
            True if user can read the record
        """
        # Public records are readable by everyone
        if self.is_public:
            return True
        
        # No user specified
        if not user:
            return False
        
        # Owner can always read
        if self.owner_id and self.owner_id == user.id:
            return True
        
        # Check custom permissions
        user_perms = self.permissions.get(str(user.id), [])
        if 'read' in user_perms:
            return True
        
        # Check group permissions if user has groups
        if hasattr(user, 'groups'):
            for group in user.groups:
                group_perms = self.permissions.get(str(group.id), [])
                if 'read' in group_perms:
                    return True
        
        return False
    
    def can_write(self, user: Optional['User'] = None) -> bool:
        """
        Check if a user can write/modify this record.
        
        Args:
            user: User to check permissions for
            
        Returns:
            True if user can write the record
        """
        # No user specified
        if not user:
            return False
        
        # Owner can always write
        if self.owner_id and self.owner_id == user.id:
            return True
        
        # Check custom permissions
        user_perms = self.permissions.get(str(user.id), [])
        if 'write' in user_perms or 'admin' in user_perms:
            return True
        
        # Check group permissions if user has groups
        if hasattr(user, 'groups'):
            for group in user.groups:
                group_perms = self.permissions.get(str(group.id), [])
                if 'write' in group_perms or 'admin' in group_perms:
                    return True
        
        return False
    
    def grant_permission(self, user_id: uuid.UUID, permission: str) -> None:
        """
        Grant a permission to a user.
        
        Args:
            user_id: User ID to grant permission to
            permission: Permission to grant (read, write, admin)
        """
        user_key = str(user_id)
        if user_key not in self.permissions:
            self.permissions[user_key] = []
        
        if permission not in self.permissions[user_key]:
            self.permissions[user_key].append(permission)
    
    def revoke_permission(self, user_id: uuid.UUID, permission: str) -> None:
        """
        Revoke a permission from a user.
        
        Args:
            user_id: User ID to revoke permission from
            permission: Permission to revoke
        """
        user_key = str(user_id)
        if user_key in self.permissions and permission in self.permissions[user_key]:
            self.permissions[user_key].remove(permission)
            # Clean up empty permission lists
            if not self.permissions[user_key]:
                del self.permissions[user_key]


class CachableMixin:
    """
    Adds cache-related fields for managing cached data.
    
    Usage:
        class MyModel(Base, CachableMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    
    cache_key = Column(
        String(255),
        nullable=True,
        unique=True,
        index=True,
        doc="Unique key for caching this record"
    )
    
    cache_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when cached data expires"
    )
    
    cache_version = Column(
        Integer,
        default=1,
        nullable=False,
        doc="Cache version for invalidation"
    )
    
    cache_hits = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of cache hits"
    )
    
    cache_last_accessed = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Timestamp of last cache access"
    )
    
    @property
    def is_cached(self) -> bool:
        """Check if record has cache information."""
        return self.cache_key is not None
    
    @property
    def is_cache_expired(self) -> bool:
        """Check if cache has expired."""
        if self.cache_expires_at is None:
            return False
        return datetime.utcnow() > self.cache_expires_at
    
    @property
    def is_cache_valid(self) -> bool:
        """Check if cache is valid (exists and not expired)."""
        return self.is_cached and not self.is_cache_expired
    
    def generate_cache_key(self, prefix: str = "cache") -> str:
        """
        Generate a cache key for this record.
        
        Args:
            prefix: Cache key prefix
            
        Returns:
            Generated cache key
        """
        return f"{prefix}:{self.__tablename__}:{self.id}"
    
    def set_cache_info(
        self, 
        ttl_seconds: int = 3600,
        cache_key: Optional[str] = None
    ) -> None:
        """
        Set cache information for this record.
        
        Args:
            ttl_seconds: Time to live in seconds
            cache_key: Custom cache key (generated if None)
        """
        self.cache_key = cache_key or self.generate_cache_key()
        self.cache_expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self.cache_version += 1
    
    def clear_cache_info(self) -> None:
        """Clear cache information."""
        self.cache_key = None
        self.cache_expires_at = None
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        self.cache_hits += 1
        self.cache_last_accessed = datetime.utcnow()


class PolymorphicMixin:
    """
    Adds support for polymorphic associations (single table inheritance).
    
    Usage:
        class Media(Base, PolymorphicMixin):
            __tablename__ = 'media'
            type = Column(String(50))
            
            __mapper_args__ = {
                'polymorphic_identity': 'media',
                'polymorphic_on': type
            }
        
        class Image(Media):
            __mapper_args__ = {'polymorphic_identity': 'image'}
            width = Column(Integer)
            height = Column(Integer)
    """
    
    type = Column(
        String(50),
        nullable=False,
        index=True,
        doc="Discriminator column for polymorphic inheritance"
    )
    
    @declared_attr
    def __mapper_args__(cls):
        """Configure polymorphic inheritance."""
        if cls.__name__ == 'PolymorphicMixin':
            return {}
        
        # Determine polymorphic identity from class name
        polymorphic_identity = cls.__name__.lower()
        
        return {
            'polymorphic_identity': polymorphic_identity,
            'polymorphic_on': cls.type
        }


# Convenience class combining common mixins
class BaseMixin(UUIDMixin, TimestampMixin, SoftDeleteMixin, AuditMixin):
    """
    Convenience class combining the most commonly used mixins.
    
    Usage:
        class MyModel(Base, BaseMixin):
            __tablename__ = 'mymodel'
            # ... other fields
    """
    pass


class PublishableMixin(TimestampMixin, StatusMixin):
    """
    Mixin for publishable content with draft/published/archived states.
    
    Usage:
        class Article(Base, PublishableMixin):
            __tablename__ = 'articles'
            __statuses__ = ['draft', 'published', 'archived']
            # ... other fields
    """
    
    published_at = Column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        doc="Timestamp when the record was published"
    )
    
    @property
    def is_published(self) -> bool:
        """Check if the record is published."""
        return self.status == 'published' and self.published_at is not None
    
    def publish(self, published_by: Optional['User'] = None) -> None:
        """Publish the record."""
        old_status = self.set_status('published', published_by, 'Published')
        if old_status != 'published':
            self.published_at = datetime.utcnow()
    
    def unpublish(self, unpublished_by: Optional['User'] = None) -> None:
        """Unpublish the record (return to draft)."""
        self.set_status('draft', unpublished_by, 'Unpublished')
        self.published_at = None
    
    def archive(self, archived_by: Optional['User'] = None) -> None:
        """Archive the record."""
        self.set_status('archived', archived_by, 'Archived')


# Event listeners for common functionality
def setup_model_listeners():
    """
    Set up global event listeners for models.
    Can be called during application startup.
    """
    from sqlalchemy import event
    from sqlalchemy.orm import mapper
    
    @event.listens_for(mapper, 'init')
    def receive_init(target, args, kwargs):
        """Called when a model instance is initialized."""
        # Add any global initialization logic here
        pass


# Helper functions for mixin usage
def get_mixin_fields(mixin_class) -> Dict[str, Any]:
    """
    Get all column fields defined in a mixin class.
    
    Args:
        mixin_class: Mixin class to inspect
        
    Returns:
        Dictionary of field names and their column definitions
    """
    import inspect
    
    fields = {}
    for name, value in inspect.getmembers(mixin_class):
        if isinstance(value, Column):
            fields[name] = value
    
    return fields


def apply_mixin_defaults(instance):
    """
    Apply default values from mixins to a model instance.
    
    Args:
        instance: Model instance to initialize
    """
    # This would be called in model __init__ if needed
    pass