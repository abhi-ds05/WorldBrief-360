"""
verification_log.py - Verification Log Model

This module defines the VerificationLog model for tracking all activities
related to incident verification processes, providing a complete audit trail.
"""

import uuid
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlalchemy import (
    Column, String, Text, ForeignKey, Integer, DateTime, 
    Boolean, Enum as SQLEnum, JSON, Float, Index
)
from sqlalchemy.orm import relationship, validates
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func

from db.base import Base
from models.mixins import TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from models.incident_verification import IncidentVerification
    from models.user import User


class LogType(Enum):
    """Types of verification log entries."""
    STATUS_CHANGE = "status_change"          # Verification status changed
    ASSIGNMENT = "assignment"                # Verification assigned to user
    UNASSIGNMENT = "unassignment"            # Verification unassigned
    SOURCE_ADDED = "source_added"           # Verification source added
    SOURCE_UPDATED = "source_updated"       # Verification source updated
    SOURCE_REMOVED = "source_removed"       # Verification source removed
    SOURCE_VERIFIED = "source_verified"     # Source verification status changed
    EVIDENCE_ADDED = "evidence_added"       # Verification evidence added
    EVIDENCE_UPDATED = "evidence_updated"   # Verification evidence updated
    EVIDENCE_REMOVED = "evidence_removed"   # Verification evidence removed
    EVIDENCE_VERIFIED = "evidence_verified" # Evidence verification changed
    SCORE_UPDATE = "score_update"           # Verification scores updated
    REVIEW_SCHEDULED = "review_scheduled"   # Review scheduled
    REVIEW_COMPLETED = "review_completed"   # Review completed
    NOTE_ADDED = "note_added"               # Note added to verification
    NOTE_UPDATED = "note_updated"           # Note updated
    NOTE_REMOVED = "note_removed"           # Note removed
    METADATA_UPDATE = "metadata_update"     # Metadata updated
    PRIORITY_CHANGE = "priority_change"     # Priority changed
    METHOD_ADDED = "method_added"           # Verification method added
    METHOD_REMOVED = "method_removed"       # Verification method removed
    CONTESTED = "contested"                 # Incident marked as contested
    RESOLVED = "resolved"                   # Contestation resolved
    AUTO_VERIFICATION = "auto_verification" # Automated verification attempt
    SYSTEM = "system"                       # System-generated log
    ERROR = "error"                         # Error occurred during verification
    OTHER = "other"                         # Other log type


class LogSeverity(Enum):
    """Severity levels for log entries."""
    INFO = "info"           # Informational message
    WARNING = "warning"     # Warning message
    ERROR = "error"         # Error message
    CRITICAL = "critical"   # Critical error
    DEBUG = "debug"         # Debug information
    AUDIT = "audit"         # Audit trail entry


class VerificationLog(Base, UUIDMixin, TimestampMixin):
    """
    Verification Log model for tracking verification activities.
    
    Provides a complete audit trail of all actions taken during
    the verification process for incidents, including user actions,
    system events, and automated processes.
    
    Attributes:
        id: Primary key UUID
        verification_id: Foreign key to IncidentVerification
        log_type: Type of log entry
        log_severity: Severity of the log entry
        action: Action performed (human-readable)
        description: Detailed description of the action
        old_value: Previous value (for changes)
        new_value: New value (for changes)
        changed_by_id: User who performed the action
        ip_address: IP address for audit trail
        user_agent: Browser/user agent info
        session_id: Session identifier
        request_id: Request identifier for tracing
        duration_ms: Action duration in milliseconds
        is_successful: Whether action was successful
        error_message: Error message if action failed
        affected_records: JSON array of affected record IDs
        metadata: Additional JSON metadata
        is_archived: Whether log is archived
        archived_at: When log was archived
    """
    
    __tablename__ = "verification_logs"
    
    # Foreign key to verification
    verification_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("incident_verifications.id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    # Log classification
    log_type = Column(
        SQLEnum(LogType),
        nullable=False,
        index=True
    )
    log_severity = Column(
        SQLEnum(LogSeverity),
        default=LogSeverity.INFO,
        nullable=False,
        index=True
    )
    
    # Action details
    action = Column(String(255), nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Change tracking
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    old_values = Column(JSONB, nullable=True)  # For complex changes
    new_values = Column(JSONB, nullable=True)  # For complex changes
    
    # User and session info
    changed_by_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("users.id", ondelete="SET NULL"), 
        nullable=True,
        index=True
    )
    ip_address = Column(String(45), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(100), nullable=True, index=True)
    request_id = Column(String(100), nullable=True, index=True)
    
    # Performance and status
    duration_ms = Column(Integer, nullable=True)
    is_successful = Column(Boolean, default=True, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    error_code = Column(String(50), nullable=True, index=True)
    stack_trace = Column(Text, nullable=True)
    
    # Affected records
    affected_records = Column(JSONB, nullable=True)
    
    # Metadata and archiving
    metadata = Column(JSONB, default=dict, nullable=False)
    is_archived = Column(Boolean, default=False, nullable=False, index=True)
    archived_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    verification = relationship("IncidentVerification", back_populates="logs")
    changed_by = relationship("User")
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_verification_logs_verification_created', 'verification_id', 'created_at'),
        Index('ix_verification_logs_type_severity', 'log_type', 'log_severity'),
        Index('ix_verification_logs_changed_by_created', 'changed_by_id', 'created_at'),
        Index('ix_verification_logs_successful', 'is_successful', 'created_at'),
    )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"<VerificationLog(id={self.id}, type={self.log_type.value}, verification={self.verification_id})>"
    
    @validates('action')
    def validate_action(self, key: str, action: str) -> str:
        """Validate action field."""
        action = action.strip()
        if not action:
            raise ValueError("Action cannot be empty")
        if len(action) > 255:
            raise ValueError("Action cannot exceed 255 characters")
        return action
    
    @validates('duration_ms')
    def validate_duration(self, key: str, duration: Optional[int]) -> Optional[int]:
        """Validate duration."""
        if duration is not None:
            if duration < 0:
                raise ValueError("Duration cannot be negative")
            if duration > 3600000:  # 1 hour
                raise ValueError("Duration is too large")
        return duration
    
    @property
    def is_system_log(self) -> bool:
        """Check if this is a system-generated log."""
        return self.log_type in [LogType.SYSTEM, LogType.AUTO_VERIFICATION, LogType.ERROR]
    
    @property
    def is_user_action(self) -> bool:
        """Check if this log represents a user action."""
        return self.changed_by_id is not None and not self.is_system_log
    
    @property
    def is_error(self) -> bool:
        """Check if this is an error log."""
        return self.log_severity in [LogSeverity.ERROR, LogSeverity.CRITICAL] or not self.is_successful
    
    @property
    def is_high_severity(self) -> bool:
        """Check if this is a high severity log."""
        return self.log_severity in [LogSeverity.ERROR, LogSeverity.CRITICAL, LogSeverity.WARNING]
    
    @property
    def age_seconds(self) -> float:
        """Get age of log in seconds."""
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds()
    
    @property
    def age_hours(self) -> float:
        """Get age of log in hours."""
        return self.age_seconds / 3600
    
    @property
    def change_summary(self) -> Dict[str, Any]:
        """Get a summary of changes."""
        summary = {
            "type": self.log_type.value,
            "action": self.action,
            "has_old_value": self.old_value is not None,
            "has_new_value": self.new_value is not None,
            "has_complex_changes": self.old_values is not None or self.new_values is not None,
        }
        
        # Add simple change if available
        if self.old_value and self.new_value:
            summary["simple_change"] = {
                "from": self.old_value,
                "to": self.new_value
            }
        
        # Add complex changes if available
        if self.old_values or self.new_values:
            summary["complex_changes"] = {
                "old": self.old_values,
                "new": self.new_values
            }
        
        return summary
    
    def mark_as_archived(self) -> None:
        """Mark log as archived."""
        self.is_archived = True
        self.archived_at = datetime.utcnow()
    
    def mark_as_unarchived(self) -> None:
        """Mark log as unarchived."""
        self.is_archived = False
        self.archived_at = None
    
    def record_error(self, error_message: str, error_code: Optional[str] = None, stack_trace: Optional[str] = None) -> None:
        """Record an error in the log."""
        self.is_successful = False
        self.error_message = error_message
        self.error_code = error_code
        self.stack_trace = stack_trace
        self.log_severity = LogSeverity.ERROR
    
    def add_affected_record(self, record_type: str, record_id: str, action: str = "modified") -> None:
        """Add an affected record to the log."""
        if self.affected_records is None:
            self.affected_records = []
        
        record_entry = {
            "type": record_type,
            "id": record_id,
            "action": action,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.affected_records.append(record_entry)
    
    def to_dict(self, include_metadata: bool = True, include_stack_trace: bool = False) -> Dict[str, Any]:
        """
        Convert log to dictionary for API responses.
        
        Args:
            include_metadata: Whether to include metadata
            include_stack_trace: Whether to include stack trace (for errors)
            
        Returns:
            Dictionary representation
        """
        result = {
            "id": str(self.id),
            "verification_id": str(self.verification_id),
            "log_type": self.log_type.value,
            "log_severity": self.log_severity.value,
            "action": self.action,
            "description": self.description,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "is_system_log": self.is_system_log,
            "is_user_action": self.is_user_action,
            "is_error": self.is_error,
            "is_high_severity": self.is_high_severity,
            "is_successful": self.is_successful,
            "is_archived": self.is_archived,
            "duration_ms": self.duration_ms,
            "age_seconds": round(self.age_seconds, 2),
            "age_hours": round(self.age_hours, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "changed_by_id": str(self.changed_by_id) if self.changed_by_id else None,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "affected_records_count": len(self.affected_records) if self.affected_records else 0,
            "change_summary": self.change_summary
        }
        
        if include_stack_trace and self.stack_trace:
            result["stack_trace"] = self.stack_trace
        
        if include_metadata:
            result["metadata"] = self.metadata
        
        if self.affected_records:
            result["affected_records"] = self.affected_records
        
        if self.changed_by:
            result["changed_by"] = {
                "id": str(self.changed_by.id),
                "username": self.changed_by.username,
                "email": getattr(self.changed_by, 'email', None)
            }
        
        return result
    
    @classmethod
    def create(
        cls,
        verification_id: uuid.UUID,
        log_type: LogType,
        action: str,
        description: Optional[str] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        changed_by_id: Optional[uuid.UUID] = None,
        log_severity: LogSeverity = LogSeverity.INFO,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        is_successful: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'VerificationLog':
        """
        Factory method to create a new verification log.
        
        Args:
            verification_id: Verification ID
            log_type: Type of log
            action: Action performed
            description: Detailed description
            old_value: Previous value
            new_value: New value
            old_values: Complex old values
            new_values: Complex new values
            changed_by_id: User who performed action
            log_severity: Log severity
            ip_address: IP address
            user_agent: User agent
            session_id: Session ID
            request_id: Request ID
            duration_ms: Action duration
            is_successful: Whether action succeeded
            metadata: Additional metadata
            
        Returns:
            A new VerificationLog instance
        """
        log = cls(
            verification_id=verification_id,
            log_type=log_type,
            action=action,
            description=description,
            old_value=old_value,
            new_value=new_value,
            old_values=old_values,
            new_values=new_values,
            changed_by_id=changed_by_id,
            log_severity=log_severity,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            duration_ms=duration_ms,
            is_successful=is_successful,
            metadata=metadata or {}
        )
        
        return log
    
    @classmethod
    def create_status_change(
        cls,
        verification_id: uuid.UUID,
        old_status: str,
        new_status: str,
        changed_by_id: Optional[uuid.UUID] = None,
        notes: Optional[str] = None,
        **kwargs
    ) -> 'VerificationLog':
        """
        Create a status change log.
        
        Args:
            verification_id: Verification ID
            old_status: Previous status
            new_status: New status
            changed_by_id: User who changed status
            notes: Change notes
            **kwargs: Additional arguments
            
        Returns:
            VerificationLog instance
        """
        return cls.create(
            verification_id=verification_id,
            log_type=LogType.STATUS_CHANGE,
            action=f"Status changed from {old_status} to {new_status}",
            description=notes,
            old_value=old_status,
            new_value=new_status,
            changed_by_id=changed_by_id,
            **kwargs
        )
    
    @classmethod
    def create_assignment(
        cls,
        verification_id: uuid.UUID,
        assigned_to_username: str,
        assigned_by_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> 'VerificationLog':
        """
        Create an assignment log.
        
        Args:
            verification_id: Verification ID
            assigned_to_username: Username of assignee
            assigned_by_id: User who made assignment
            **kwargs: Additional arguments
            
        Returns:
            VerificationLog instance
        """
        return cls.create(
            verification_id=verification_id,
            log_type=LogType.ASSIGNMENT,
            action=f"Assigned to {assigned_to_username}",
            description=f"Verification assigned to user: {assigned_to_username}",
            new_value=assigned_to_username,
            changed_by_id=assigned_by_id,
            **kwargs
        )
    
    @classmethod
    def create_error(
        cls,
        verification_id: uuid.UUID,
        error_message: str,
        error_code: Optional[str] = None,
        stack_trace: Optional[str] = None,
        **kwargs
    ) -> 'VerificationLog':
        """
        Create an error log.
        
        Args:
            verification_id: Verification ID
            error_message: Error message
            error_code: Error code
            stack_trace: Stack trace
            **kwargs: Additional arguments
            
        Returns:
            VerificationLog instance
        """
        log = cls.create(
            verification_id=verification_id,
            log_type=LogType.ERROR,
            action="Error occurred",
            description=error_message,
            log_severity=LogSeverity.ERROR,
            is_successful=False,
            **kwargs
        )
        
        log.error_message = error_message
        log.error_code = error_code
        log.stack_trace = stack_trace
        
        return log
    
    @classmethod
    def create_system_log(
        cls,
        verification_id: uuid.UUID,
        action: str,
        description: Optional[str] = None,
        **kwargs
    ) -> 'VerificationLog':
        """
        Create a system-generated log.
        
        Args:
            verification_id: Verification ID
            action: Action performed
            description: Action description
            **kwargs: Additional arguments
            
        Returns:
            VerificationLog instance
        """
        return cls.create(
            verification_id=verification_id,
            log_type=LogType.SYSTEM,
            action=action,
            description=description,
            log_severity=LogSeverity.INFO,
            **kwargs
        )


# Pydantic schemas for API validation
"""
If you're using Pydantic, here are the schemas for the VerificationLog model.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class VerificationLogBase(BaseModel):
    """Base schema for verification log operations."""
    log_type: LogType
    action: str = Field(..., max_length=255)
    description: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    
    @validator('action')
    def validate_action(cls, v):
        if not v or not v.strip():
            raise ValueError('Action cannot be empty')
        return v.strip()


class VerificationLogCreate(VerificationLogBase):
    """Schema for creating verification logs."""
    verification_id: str
    changed_by_id: Optional[str] = None
    log_severity: LogSeverity = Field(default=LogSeverity.INFO)
    
    @validator('verification_id', 'changed_by_id')
    def validate_uuids(cls, v):
        if v is not None:
            try:
                uuid.UUID(v)
            except ValueError:
                raise ValueError('Invalid UUID format')
        return v


class VerificationLogUpdate(BaseModel):
    """Schema for updating verification logs."""
    is_archived: Optional[bool] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class VerificationLogInDBBase(VerificationLogBase):
    """Base schema for verification log in database."""
    id: str
    verification_id: str
    log_severity: LogSeverity
    is_system_log: bool
    is_user_action: bool
    is_error: bool
    is_successful: bool
    is_archived: bool
    duration_ms: Optional[int]
    changed_by_id: Optional[str]
    ip_address: Optional[str]
    created_at: datetime
    archived_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class VerificationLog(VerificationLogInDBBase):
    """Schema for verification log API responses."""
    changed_by: Optional[Dict[str, Any]] = None
    age_seconds: float
    age_hours: float
    change_summary: Dict[str, Any]
    affected_records_count: int
    
    class Config:
        from_attributes = True


class VerificationLogSearchRequest(BaseModel):
    """Schema for verification log search requests."""
    verification_id: Optional[str] = None
    log_type: Optional[LogType] = None
    log_severity: Optional[LogSeverity] = None
    changed_by_id: Optional[str] = None
    is_system_log: Optional[bool] = None
    is_error: Optional[bool] = None
    is_successful: Optional[bool] = None
    is_archived: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    action_contains: Optional[str] = None
    min_duration_ms: Optional[int] = None
    max_duration_ms: Optional[int] = None
    sort_by: str = Field(default="created_at", pattern="^(created_at|log_severity|duration_ms)$")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    
    class Config:
        from_attributes = True


class VerificationLogStats(BaseModel):
    """Schema for verification log statistics."""
    total_logs: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    by_hour: Dict[str, int]
    error_rate: float
    average_duration_ms: Optional[float]
    top_actions: List[Dict[str, Any]]
    recent_errors: List[Dict[str, Any]]
    
    class Config:
        from_attributes = True


class LogExportRequest(BaseModel):
    """Schema for log export requests."""
    date_from: datetime
    date_to: datetime
    format: str = Field(default="json", pattern="^(json|csv|excel)$")
    include_metadata: bool = Field(default=False)
    include_stack_trace: bool = Field(default=False)
    
    class Config:
        from_attributes = True