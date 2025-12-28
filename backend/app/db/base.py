"""
Base database models and mixins for PostgreSQL.
Located in db/ folder as per your structure.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Type, TypeVar
from enum import Enum as PyEnum

from sqlalchemy import (
    Column,
    DateTime,
    Boolean,
    String,
    Text,
    Integer,
    BigInteger,
    Float,
    Numeric,
    ForeignKey,
    UniqueConstraint,
    Index,
    Enum,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.sql import sqltypes

T = TypeVar('T', bound='Base')

class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models.
    Uses PostgreSQL-specific types for better performance.
    """
    
    @declared_attr
    def __tablename__(cls) -> str:
        """
        Convert class name to snake_case for table name.
        Example: 'UserProfile' -> 'user_profile'
        """
        import re
        # Convert CamelCase to snake_case
        name = cls.__name__
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()
        return name
    
    def to_dict(self, exclude: Optional[list] = None) -> Dict[str, Any]:
        """
        Convert model instance to dictionary.
        
        Args:
            exclude: List of field names to exclude
            
        Returns:
            Dictionary representation
        """
        exclude = exclude or []
        result = {}
        
        for column in self.__table__.columns:
            if column.name not in exclude:
                value = getattr(self, column.name)
                
                # Handle special types
                if isinstance(value, datetime):
                    result[column.name] = value.isoformat()
                elif isinstance(value, uuid.UUID):
                    result[column.name] = str(value)
                elif isinstance(value, PyEnum):
                    result[column.name] = value.value
                elif isinstance(value, Base):
                    # Handle relationships
                    result[column.name] = value.to_dict()
                elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    # Handle lists of related objects
                    result[column.name] = [item.to_dict() for item in value]
                else:
                    result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Update model instance from dictionary.
        
        Args:
            data: Dictionary with field values
        """
        for key, value in data.items():
            if hasattr(self, key) and not key.startswith('_'):
                # Check if the field is a relationship
                if key in self.__mapper__.relationships:
                    # Skip relationships for simple update
                    continue
                setattr(self, key, value)
    
    @classmethod
    def create_from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a new instance from dictionary.
        
        Args:
            data: Dictionary with field values
            
        Returns:
            New model instance
        """
        # Filter out relationship fields
        mapper = cls.__mapper__
        valid_fields = {c.key for c in mapper.columns}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)


class TimestampMixin:
    """
    Mixin for created_at and updated_at timestamps.
    Automatically sets timestamps on creation and update.
    """
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
        doc="Timestamp when the record was created"
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        index=True,
        doc="Timestamp when the record was last updated"
    )


class SoftDeleteMixin:
    """
    Mixin for soft delete functionality.
    Records are marked as deleted instead of being physically removed.
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
        doc="Timestamp when the record was marked as deleted"
    )
    
    def soft_delete(self) -> None:
        """Mark the record as deleted."""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
    
    def restore(self) -> None:
        """Restore a soft-deleted record."""
        self.is_deleted = False
        self.deleted_at = None


class AuditMixin:
    """
    Mixin for audit tracking (who created/updated the record).
    """
    created_by = Column(
        UUID(as_uuid=True),
        ForeignKey('user.id', ondelete='SET NULL'),
        nullable=True,
        doc="ID of the user who created the record"
    )
    updated_by = Column(
        UUID(as_uuid=True),
        ForeignKey('user.id', ondelete='SET NULL'),
        nullable=True,
        doc="ID of the user who last updated the record"
    )
    
    # Relationships will be defined in the actual models
    @declared_attr
    def creator(cls):
        """Relationship to the user who created the record."""
        return relationship(
            'User',
            foreign_keys=[cls.created_by],
            primaryjoin=f"User.id == {cls.__name__}.created_by",
            remote_side='User.id',
            lazy='select',
            doc="User who created this record"
        )
    
    @declared_attr
    def updater(cls):
        """Relationship to the user who last updated the record."""
        return relationship(
            'User',
            foreign_keys=[cls.updated_by],
            primaryjoin=f"User.id == {cls.__name__}.updated_by",
            remote_side='User.id',
            lazy='select',
            doc="User who last updated this record"
        )


class StatusMixin:
    """
    Mixin for status tracking with state transitions.
    """
    status = Column(
        String(50),
        nullable=False,
        default='active',
        index=True,
        doc="Current status of the record"
    )
    
    # Store status history in JSON
    status_history = Column(
        JSONB,
        default=list,
        doc="History of status changes with timestamps and reasons"
    )
    
    def change_status(self, new_status: str, reason: str = None, user_id: str = None) -> None:
        """
        Change status with audit trail.
        
        Args:
            new_status: New status value
            reason: Reason for status change
            user_id: ID of user making the change
        """
        old_status = self.status
        self.status = new_status
        
        # Record status change in history
        history_entry = {
            'from_status': old_status,
            'to_status': new_status,
            'changed_at': datetime.utcnow().isoformat(),
            'reason': reason,
            'user_id': user_id
        }
        
        if self.status_history is None:
            self.status_history = []
        
        self.status_history.append(history_entry)
        
        # Keep only last 100 status changes
        if len(self.status_history) > 100:
            self.status_history = self.status_history[-100:]


# PostgreSQL-specific UUID generation
def generate_uuid() -> uuid.UUID:
    """
    Generate a UUID for primary keys.
    
    Returns:
        UUID4 value
    """
    return uuid.uuid4()


def generate_short_uuid() -> str:
    """
    Generate a short UUID string (8 characters).
    
    Returns:
        Short UUID string
    """
    return uuid.uuid4().hex[:8]


# Custom PostgreSQL types
class CIText(sqltypes.TypeDecorator):
    """
    Case-insensitive text type for PostgreSQL.
    Requires CIText extension in PostgreSQL.
    """
    impl = sqltypes.TEXT
    cache_ok = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(sqltypes.TEXT)
        else:
            return dialect.type_descriptor(sqltypes.VARCHAR)


# Event listeners for automatic updates
@event.listens_for(Base, 'before_insert', propagate=True)
def set_created_at(mapper, connection, target):
    """Automatically set created_at timestamp on insert."""
    if hasattr(target, 'created_at') and target.created_at is None:
        target.created_at = datetime.utcnow()
    
    if hasattr(target, 'updated_at') and target.updated_at is None:
        target.updated_at = datetime.utcnow()


@event.listens_for(Base, 'before_update', propagate=True)
def set_updated_at(mapper, connection, target):
    """Automatically update updated_at timestamp."""
    if hasattr(target, 'updated_at'):
        target.updated_at = datetime.utcnow()


@event.listens_for(Base, 'before_insert')
def generate_ids(mapper, connection, target):
    """Generate UUID for primary key if not set."""
    for column in mapper.columns:
        if column.primary_key and isinstance(column.type, UUID):
            if getattr(target, column.key) is None:
                setattr(target, column.key, generate_uuid())


# Utility functions for common operations
def paginate_query(query, page: int = 1, per_page: int = 20):
    """
    Paginate a SQLAlchemy query.
    
    Args:
        query: SQLAlchemy query object
        page: Page number (1-indexed)
        per_page: Items per page
        
    Returns:
        Paginated query
    """
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 20
    
    offset = (page - 1) * per_page
    return query.offset(offset).limit(per_page)


def filter_by_params(query, model_class, **filters):
    """
    Apply filters to a query dynamically.
    
    Args:
        query: SQLAlchemy query object
        model_class: SQLAlchemy model class
        **filters: Filter parameters
        
    Returns:
        Filtered query
    """
    for key, value in filters.items():
        if value is None:
            continue
            
        if hasattr(model_class, key):
            column = getattr(model_class, key)
            
            # Handle different filter types
            if isinstance(value, (list, tuple)):
                query = query.filter(column.in_(value))
            elif isinstance(value, dict):
                # Handle comparison operators
                for op, op_value in value.items():
                    if op == 'eq':
                        query = query.filter(column == op_value)
                    elif op == 'ne':
                        query = query.filter(column != op_value)
                    elif op == 'gt':
                        query = query.filter(column > op_value)
                    elif op == 'gte':
                        query = query.filter(column >= op_value)
                    elif op == 'lt':
                        query = query.filter(column < op_value)
                    elif op == 'lte':
                        query = query.filter(column <= op_value)
                    elif op == 'like':
                        query = query.filter(column.like(f'%{op_value}%'))
                    elif op == 'ilike':
                        query = query.filter(column.ilike(f'%{op_value}%'))
            else:
                # Exact match
                query = query.filter(column == value)
    
    return query